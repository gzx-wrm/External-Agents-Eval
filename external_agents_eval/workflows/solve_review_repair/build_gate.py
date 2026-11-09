"""V6.4 build/collection gate.

The gate compiles or collects a project's tests without executing them. Commands
come only from ecosystem conventions detected in the frozen base repository; the
gate never reads hidden tests, verifier files or reference solutions. A candidate
failure counts only when the identical command succeeds on the base repository,
so pre-existing breakage, missing tools and offline limitations yield
``unavailable`` rather than a signal.
"""
from __future__ import annotations

import posixpath
import re
import shlex
import time
from dataclasses import asdict, dataclass, replace
from typing import Any, Callable, Iterable, Protocol

from .artifacts import sha256_bytes

GATE_SCHEMA_VERSION = "v6.4-build-gate-1"
DEFAULT_RUN_TIMEOUT_SEC = 600
MAX_COMMANDS = 4
LOG_TAIL_LINES = 200
LOG_TAIL_BYTES = 16 * 1024
TIMEOUT_EXIT_CODES = frozenset({124, 137})
PASS, FAIL, TIMEOUT, ERROR, UNAVAILABLE = "pass", "fail", "timeout", "error", "unavailable"

# Offline, non-interactive defaults shared by every command.
_COMMON_ENV = (
    "CI=1 PYTHONDONTWRITEBYTECODE=1 CARGO_NET_OFFLINE=true GOPROXY=off "
    "GOTOOLCHAIN=local DENO_NO_UPDATE_CHECK=1"
)
_TS_SUFFIXES = (".ts", ".tsx", ".mts", ".cts")


@dataclass(frozen=True)
class GateCommand:
    ecosystem: str
    cwd: str
    command: str

    @property
    def key(self) -> str:
        return f"{self.ecosystem}\0{self.cwd}\0{self.command}"

    def shell(self) -> str:
        return f"cd {shlex.quote(self.cwd)} && env {_COMMON_ENV} {self.command}"


@dataclass(frozen=True)
class CommandOutcome:
    status: str
    exit_code: int | None
    duration_sec: float
    log_tail: str = ""
    error_codes: tuple[str, ...] = ()
    note: str = ""


_TS_ERROR = re.compile(r"\berror (TS\d{4,5})\b")


def error_codes(text: str) -> tuple[str, ...]:
    """TypeScript diagnostic codes in a compiler log (collected from the full output)."""
    return tuple(sorted(set(_TS_ERROR.findall(text or ""))))


def _semantic_ts(code: str) -> bool:
    number = int(code[2:])
    return 2000 <= number < 3000 or 7000 <= number < 8000


def refine(command: "GateCommand", outcome: CommandOutcome) -> CommandOutcome:
    """A TypeScript type check fails the gate only for errors that stop any build.

    Transpiling test runners execute code that has semantic type errors (TS2xxx,
    TS7xxx); syntax, configuration and unknown failures still count. Deno keeps
    its default type-checked test runs and is not refined.
    """
    if command.ecosystem != "typescript" or outcome.status != FAIL:
        return outcome
    codes = outcome.error_codes or error_codes(outcome.log_tail)
    if codes and all(_semantic_ts(code) for code in codes):
        return replace(outcome, status=PASS,
                       note="semantic type errors only; not a build failure")
    return outcome


class GateExecutor(Protocol):
    """Runs gate commands in a fresh, offline copy of the task environment.

    ``patch`` is relative to the image's environment baseline (empty for base).
    """

    async def list_files(self) -> list[str]: ...

    async def read_file(self, path: str) -> str | None: ...

    async def run(
        self, label: str, patch: bytes, commands: list[GateCommand], timeout_sec: float,
    ) -> list[CommandOutcome]: ...


def tail(text: str, *, lines: int = LOG_TAIL_LINES, limit: int = LOG_TAIL_BYTES) -> str:
    kept = "\n".join((text or "").splitlines()[-lines:])
    encoded = kept.encode("utf-8", errors="replace")
    if len(encoded) > limit:
        kept = encoded[-limit:].decode("utf-8", errors="replace")
    return kept


def classify(exit_code: int | None, *, timed_out: bool = False) -> str:
    if timed_out or exit_code in TIMEOUT_EXIT_CODES:
        return TIMEOUT
    if exit_code is None:
        return ERROR
    return PASS if exit_code == 0 else FAIL


def changed_paths(patch: bytes) -> list[str]:
    """Paths touched by a git patch (both sides of renames), in first-seen order."""
    seen: dict[str, None] = {}
    for line in patch.decode("utf-8", errors="replace").splitlines():
        if line.startswith("diff --git "):
            match = re.match(r'^diff --git "?a/(.+?)"? "?b/(.+?)"?$', line)
            if match:
                seen.setdefault(match.group(1), None)
                seen.setdefault(match.group(2), None)
    return list(seen)


def _dirs_up(path: str) -> list[str]:
    """Ancestor directories of a file path, nearest first, ending with '.'."""
    current = posixpath.dirname(path)
    result = []
    while current not in ("", "."):
        result.append(current)
        current = posixpath.dirname(current)
    result.append(".")
    return result


def _join(directory: str, name: str) -> str:
    return name if directory == "." else f"{directory}/{name}"


def _nearest(path: str, predicate: Callable[[str], bool]) -> str | None:
    for directory in _dirs_up(path):
        if predicate(directory):
            return directory
    return None


def _cargo_package(text: str | None) -> str | None:
    if not text:
        return None
    section = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line.startswith("[") and line.endswith("]"):
            section = line.strip("[]").strip()
            continue
        if section == "package":
            match = re.match(r'^name\s*=\s*"([^"]+)"', line)
            if match:
                return match.group(1)
    return None


def _pytest_config(directory: str, files: set[str], read: Callable[[str], str | None]) -> bool:
    if _join(directory, "pytest.ini") in files or _join(directory, "conftest.py") in files:
        return True
    for name, marker in (("pyproject.toml", "[tool.pytest"), ("setup.cfg", "[tool:pytest]"),
                         ("tox.ini", "[pytest]")):
        candidate = _join(directory, name)
        if candidate in files and marker in (read(candidate) or ""):
            return True
    return False


def detect_commands(
    files: Iterable[str], changed: Iterable[str], read: Callable[[str], str | None],
) -> list[GateCommand]:
    """Ecosystem conventions only; scoped to projects that contain changed files."""
    files = set(files)
    changed = [path for path in changed if path]
    commands: list[GateCommand] = []

    def add(command: GateCommand) -> None:
        if command not in commands:
            commands.append(command)

    # Rust: compile the test targets of every package that owns a changed file.
    if any(posixpath.basename(path) == "Cargo.toml" for path in files):
        packages: list[str] = []
        for path in changed:
            owner = _nearest(path, lambda d: _cargo_package(read(_join(d, "Cargo.toml")))
                             is not None if _join(d, "Cargo.toml") in files else False)
            if owner is not None:
                name = _cargo_package(read(_join(owner, "Cargo.toml")))
                if name and name not in packages:
                    packages.append(name)
        if packages:
            selection = " ".join("-p " + shlex.quote(name) for name in packages)
            add(GateCommand("rust", ".", f"cargo test --no-run --offline --quiet {selection}"))
        elif "Cargo.toml" in files:
            add(GateCommand("rust", ".", "cargo test --no-run --offline --quiet"))

    # Go: build every test binary of each module that owns a changed file; run none.
    modules = [d for d in {posixpath.dirname(p) or "." for p in files
                           if posixpath.basename(p) == "go.mod"}]
    if modules:
        owners = []
        for path in changed:
            owner = _nearest(path, lambda d: _join(d, "go.mod") in files)
            if owner is not None and owner not in owners:
                owners.append(owner)
        for owner in owners or (["."] if "go.mod" in files else []):
            add(GateCommand("go", owner, "go test -count=1 -run '^$' ./..."))

    deno_roots = {posixpath.dirname(p) or "." for p in files
                  if posixpath.basename(p) in {"deno.json", "deno.jsonc"}}

    # Deno: type-check test modules from the cache without running them.
    for path in changed:
        owner = _nearest(path, lambda d: d in deno_roots)
        if owner is not None:
            add(GateCommand("deno", owner, "deno test --no-run --cached-only"))

    # TypeScript (non-Deno): the project's own compiler, type-check only.
    for path in changed:
        if not path.endswith(_TS_SUFFIXES) or _nearest(path, lambda d: d in deno_roots):
            continue
        owner = _nearest(path, lambda d: _join(d, "tsconfig.json") in files)
        if owner is not None:
            add(GateCommand(
                "typescript", owner,
                'sh -c \'d="$PWD"; while [ "$d" != / ]; do '
                'if [ -x "$d/node_modules/.bin/tsc" ]; then '
                'exec "$d/node_modules/.bin/tsc" --noEmit -p tsconfig.json; fi; '
                'd=$(dirname "$d"); done; exit 127\'',
            ))

    # Python: collect tests in the pytest project that owns each changed file.
    for path in changed:
        if not path.endswith(".py"):
            continue
        owner = _nearest(path, lambda d: _pytest_config(d, files, read))
        if owner is not None:
            add(GateCommand("python", owner,
                            "python -m pytest --collect-only -q -p no:cacheprovider"))

    # JVM: compile test sources offline with the project's own wrapper or Maven.
    for path in changed:
        owner = _nearest(path, lambda d: _join(d, "gradlew") in files)
        if owner is not None:
            add(GateCommand("gradle", owner, "./gradlew --offline -q testClasses"))
            continue
        owner = _nearest(path, lambda d: _join(d, "pom.xml") in files)
        if owner is not None:
            add(GateCommand("maven", owner, "mvn -o -q test-compile"))

    return commands[:MAX_COMMANDS]


def compare(base: CommandOutcome, candidate: CommandOutcome) -> str:
    """Differential verdict: only a base success can make a candidate failure count."""
    if base.status != PASS:
        return UNAVAILABLE
    if candidate.status == PASS:
        return PASS
    if candidate.status == FAIL:
        return FAIL
    return UNAVAILABLE


def overall(verdicts: Iterable[str]) -> str:
    verdicts = list(verdicts)
    if FAIL in verdicts:
        return FAIL
    if PASS in verdicts:
        return PASS
    return UNAVAILABLE


class BuildGate:
    """Differential gate; base outcomes are computed once per command and cached."""

    def __init__(self, executor: GateExecutor, *, run_timeout_sec: float = DEFAULT_RUN_TIMEOUT_SEC):
        if run_timeout_sec <= 0:
            raise ValueError("Build gate timeout must be positive")
        self.executor = executor
        self.run_timeout_sec = run_timeout_sec
        self._files: list[str] | None = None
        self._reads: dict[str, str | None] = {}
        self._base: dict[str, CommandOutcome] = {}

    def config(self) -> dict[str, Any]:
        return {"schema_version": GATE_SCHEMA_VERSION, "run_timeout_sec": self.run_timeout_sec,
                "max_commands": MAX_COMMANDS, "log_tail_lines": LOG_TAIL_LINES,
                "log_tail_bytes": LOG_TAIL_BYTES}

    async def _detect(self, patch: bytes) -> list[GateCommand]:
        if self._files is None:
            self._files = list(await self.executor.list_files())
        files = set(self._files)
        changed = changed_paths(patch)
        wanted = {path for path in files if posixpath.basename(path) in {
            "Cargo.toml", "pyproject.toml", "setup.cfg", "tox.ini"}}
        for path in sorted(wanted - set(self._reads)):
            self._reads[path] = await self.executor.read_file(path)
        return detect_commands(files, changed, lambda path: self._reads.get(path))

    async def check(self, label: str, patch: bytes) -> dict[str, Any]:
        started = time.monotonic()
        commands = await self._detect(patch)
        report: dict[str, Any] = {
            **self.config(), "label": label, "patch_sha256": sha256_bytes(patch),
            "changed_paths": changed_paths(patch), "commands": [], "failure_logs": [],
        }
        if not commands:
            report.update(verdict=UNAVAILABLE, reason="no supported ecosystem for changed files")
            return report
        pending = [command for command in commands if command.key not in self._base]
        if pending:
            outcomes = await self.executor.run("base", b"", pending, self.run_timeout_sec)
            self._base.update({command.key: refine(command, outcome)
                               for command, outcome in zip(pending, outcomes)})
        if not patch:
            candidates = [self._base[command.key] for command in commands]
        elif any(self._base[command.key].status == PASS for command in commands):
            candidates = [refine(command, outcome) for command, outcome in zip(
                commands, await self.executor.run(label, patch, commands, self.run_timeout_sec))]
        else:
            # No command works on the base repository: nothing can count as a signal.
            candidates = [CommandOutcome(UNAVAILABLE, None, 0.0) for _ in commands]
        verdicts = []
        for command, candidate in zip(commands, candidates):
            base = self._base[command.key]
            verdict = compare(base, candidate)
            verdicts.append(verdict)
            report["commands"].append({
                **asdict(command), "verdict": verdict,
                "base": {k: v for k, v in asdict(base).items() if k != "log_tail"},
                "candidate": {k: v for k, v in asdict(candidate).items() if k != "log_tail"},
            })
            if verdict == FAIL:
                report["failure_logs"].append({"command": command.command, "cwd": command.cwd,
                                               "exit_code": candidate.exit_code,
                                               "log_tail": tail(candidate.log_tail)})
        report["verdict"] = overall(verdicts)
        report["elapsed_sec"] = round(time.monotonic() - started, 3)
        return report


def build_check_for_repair(report: dict[str, Any]) -> dict[str, Any] | None:
    """The only gate content shown to Repair: failing command, cwd and log tail."""
    if report.get("verdict") != FAIL:
        return None
    return {"description": "A build/collection command that succeeds on the base repository "
                           "fails after P0.",
            "failures": report["failure_logs"]}


__all__ = [
    "BuildGate", "CommandOutcome", "GateCommand", "GateExecutor", "build_check_for_repair",
    "changed_paths", "classify", "compare", "detect_commands", "error_codes", "overall",
    "refine", "tail",
]
