"""V7 acceptance-test generation: prompt, test-patch filtering, manifest validation, verdicts.

A test author works on the frozen baseline (feature absent) from the task text alone and
writes new test files plus a manifest. The host then runs every manifest command on the
baseline and on candidate implementations. A test is *hard* when it is bound to an exact
task quote and behaves as its kind predicts on the baseline (new behavior fails, preserved
behavior passes). A candidate is rejected when a hard test fails twice on it.

Everything here is pure logic; container work lives in scripts/deepswe_v7_testgen/run.py.
"""
from __future__ import annotations

import copy
import json
import re
from typing import Any

from external_agents_eval.benchmarks.deepswe.submission_filter import is_test_path

VERSION = "v7.2.0"
MANIFEST_DIR = "/tmp/v7"
MANIFEST_PATH = f"{MANIFEST_DIR}/manifest.json"
MAX_TESTS = 40
KINDS = ("new", "preserve", "existing")
MIN_QUOTE_CHARS = 8
TEMPLATE_MARKER = "## Command Execution Rules"

TG_HEADER_V70 = """You are the V7 acceptance-test author. /app contains the repository BEFORE the requested
feature is implemented: the feature described below does NOT exist yet, and you must NOT
implement it. Write executable acceptance tests that a correct implementation of the task
passes and an incomplete or incorrect implementation fails.

TASK (the only source of truth for the new behavior)
<task>
{{task}}
</task>

RULES
1. Requirements come only from the task text. Split the task into its normative statements:
   every sentence or bullet that requires a behavior, an API, a format, an error, a default,
   or that existing behavior stays unchanged. Give each an id (R1, R2, ...) and an exact quote
   copied from the task text (a contiguous substring of at least a few words).
2. Assert only what a quote states. If a detail (a name, a signature, an exact message, an
   ordering, a format, a default) is neither stated in the task nor already present in the
   repository, do not assert on it.
3. Use only public API that the task names or that already exists in /app. Never invent names
   of new functions, options, types or files that the task does not give.
4. Test through the public API, the way a user of the project would, as scenarios. For every
   statement consider which of these scenario types apply, and write a test for each that does:
   - basic: the stated behavior in its plainest form
   - repeated calls / idempotency: calling twice or more, re-running after success
   - lifecycle and ordering: objects created before configuration or reset, before/after
     start, mount, close or dispose
   - change after the initial state: state, input or geometry changes after the first result
   - several items: wherever the task uses a plural (several definitions, entries, files)
   - error paths: stated errors, failure propagation, rejected inputs
   - defaults and omitted or empty values that the task defines
   - structural boundaries: multi-line constructs, nesting, indentation, empty containers
   - usage forms consistent with existing analogous APIs in the repository
   - preservation: existing behavior that the task says (or clearly implies) must not change
5. Files: create NEW test files only, in the repository's usual test locations and naming
   (for example *_test.go, test_*.py, *.test.ts, files under tests/ or __tests__/). Do not
   modify, move or delete any existing file. Do not create non-test source files.
6. Every test must run on its own with one shell command from /app that exits 0 if and only if
   the test passes, for example `go test ./pkg -run '^TestX$' -count=1`,
   `python -m pytest tests/test_x.py::test_y -q`, `npx vitest run path/x.test.ts -t "name"`,
   or `cargo test --test x name`. Keep each command fast. No network is available. If the
   tests need a one-time build step, put it in the manifest's "setup" command.
7. Before finishing, run every command on this baseline:
   - kind "new" (new behavior) must FAIL now, and the failure must come from the missing
     feature (missing symbol, option or behavior), not from a mistake in the test (syntax
     error, wrong import of existing code, wrong runner usage). Fix such mistakes.
   - kind "preserve" (a new test of existing behavior that must not change) must PASS now.
   - kind "existing" (a command running existing repository tests of behavior the task must
     not break; no new file needed) must PASS now.
8. Write the manifest to /tmp/v7/manifest.json (create /tmp/v7 first; it is outside the
   repository on purpose) in this JSON form:
   {"setup": "<optional one-time command, or null>",
    "requirements": [{"id": "R1", "quote": "<exact task text>"}],
    "tests": [{"id": "T1", "requirements": ["R1"], "kind": "new",
               "category": "<scenario type from rule 4>",
               "files": ["<new test file, relative to /app>"],
               "command": "<one shell command run from /app>",
               "asserts": "<what the test checks, one sentence>"}]}
   At most 40 tests. Write the manifest early and update it as you go, so finished work
   survives if you run out of steps.
9. When the manifest is complete and every command has been run on this baseline, finish by
   issuing `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` as a separate command.

"""


# v7.2: clause-level coverage, adversarial (plausible-wrong-implementation) test design,
# assertion discipline, strict interface use and compile isolation between test files.
TG_HEADER_V72 = """You are the V7 acceptance-test author. /app contains the repository BEFORE the requested
feature is implemented: the feature described below does NOT exist yet, and you must NOT
implement it. Write executable acceptance tests that a correct implementation of the task
passes and an incomplete or incorrect implementation fails.

TASK (the only source of truth for the new behavior)
<task>
{{task}}
</task>

RULES
1. Requirements come only from the task text. Split the task into its normative statements:
   every sentence or bullet that requires a behavior, an API, a format, an error, a default,
   or that existing behavior stays unchanged. Give each an id (R1, R2, ...) and an exact quote
   copied from the task text (a contiguous substring of at least a few words).
2. Split every statement into its clauses. Each condition, qualifier, quantifier and exception
   (for example "even when", "only", "always", "each", "every", "all", "at most", "unless",
   "non-empty", "multi-line", "exactly once", "in order", "for the whole statement") is a check
   point of its own and needs its own assertion. A test of the common case alone does not cover
   a qualifier.
3. Think adversarially. For every requirement, name the most plausible incomplete or wrong
   implementation: one that handles the common case but misses a clause, a corner of the input
   space, a later change of state, a repeated call, or a combination with another feature. Write
   at least one test that a correct implementation passes and that plausible wrong one fails,
   and describe that wrong implementation in the test's "catches" field.
4. Assert only what the task states. If a detail (an exact message, an ordering, a
   classification, a count, a timing, a default) is not stated, do not assert on it; assert the
   stated property only (for example that an error occurs, not its text).
5. Interface: use exactly the names, signatures, import paths, options and build tags that the
   task gives, or that already exist in /app. Never invent a name for something the task does not
   name; test such behavior only through entry points the task names.
6. Test through the public API, the way a user of the project would, as scenarios. Consider for
   every statement which scenario types apply: basic; repeated calls / idempotency; lifecycle and
   ordering (objects created before configuration or reset, before/after start, mount, close);
   change after the initial state; several items wherever the task uses a plural; error paths;
   defaults and omitted or empty values the task defines; structural boundaries (multi-line
   constructs, nesting, indentation, empty containers); usage forms consistent with existing
   analogous APIs; preservation of existing behavior the task says must not change.
7. Files: create NEW test files only, in the repository's usual test locations and naming (for
   example *_test.go, test_*.py, *.test.ts, files under tests/ or __tests__/). Do not modify, move
   or delete any existing file, and do not create non-test source files. Put tests for different
   requirements in separate files, so one file that fails to compile cannot break the others. In
   Go, put each new test file in its own new directory as an external test package (package
   <dir>_test importing the project packages) unless the test needs unexported identifiers.
8. Every test must run on its own with one shell command from /app that exits 0 if and only if
   the test passes, for example `go test ./pkg -run '^TestX$' -count=1`,
   `python -m pytest tests/test_x.py::test_y -q`, `npx vitest run path/x.test.ts -t "name"`,
   or `cargo test --test x name`. Keep each command fast. No network is available. If the tests
   need a one-time build step, put it in the manifest's "setup" command.
9. Before finishing, run every command on this baseline:
   - kind "new" (new behavior) must FAIL now, and the failure must come from the missing feature
     (missing symbol, option or behavior), not from a mistake in the test (syntax error, wrong
     import of existing code, wrong runner usage). Fix such mistakes.
   - kind "preserve" (a new test of existing behavior that must not change) must PASS now.
   - kind "existing" (a command running existing repository tests of behavior the task must not
     break; no new file needed) must PASS now.
10. Write the manifest to /tmp/v7/manifest.json (create /tmp/v7 first; it is outside the
   repository on purpose) in this JSON form:
   {"setup": "<optional one-time command, or null>",
    "requirements": [{"id": "R1", "quote": "<exact task text>"}],
    "tests": [{"id": "T1", "requirements": ["R1"], "kind": "new",
               "category": "<scenario type from rule 6>",
               "files": ["<new test file, relative to /app>"],
               "command": "<one shell command run from /app>",
               "asserts": "<what the test checks, one sentence>",
               "catches": "<the plausible wrong implementation this test fails>"}]}
   At most 40 tests. Write the manifest early and update it as you go, so finished work survives
   if you run out of steps.
11. When the manifest is complete and every command has been run on this baseline, finish by
   issuing `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` as a separate command.

"""
TG_HEADER = TG_HEADER_V72


def tg_config(config: dict[str, Any], *, step_limit: int, wall_time_sec: int,
              command_timeout_sec: int, labels: list[str] | None = None) -> dict[str, Any]:
    """The M0 Mini model/environment with the test-author task section and limits."""
    result = copy.deepcopy(config)
    template = config["agent"]["instance_template"]
    if template.count(TEMPLATE_MARKER) != 1:
        raise ValueError("Unexpected upstream Mini instance template")
    result["agent"]["instance_template"] = TG_HEADER + template[template.index(TEMPLATE_MARKER):]
    result["agent"].update(step_limit=step_limit, wall_time_limit_seconds=wall_time_sec, cost_limit=0)
    result["environment"]["timeout"] = command_timeout_sec
    result["environment"]["run_args"] = list(result["environment"]["run_args"]) + list(labels or [])
    return result


# ---------------------------------------------------------------------------
# Patches
# ---------------------------------------------------------------------------

_HEADER = re.compile(r'^diff --git "?a/(?P<a>.+?)"? "?b/(?P<b>.+?)"?$')


def patch_sections(patch: str) -> list[str]:
    sections: list[str] = []
    current: list[str] = []
    for line in patch.splitlines(keepends=True):
        if line.startswith("diff --git ") and current:
            sections.append("".join(current))
            current = []
        current.append(line)
    if current and current[0].startswith("diff --git "):
        sections.append("".join(current))
    return sections


def section_info(section: str) -> dict[str, Any]:
    lines = section.splitlines()
    match = _HEADER.match(lines[0]) if lines else None
    path = match.group("b") if match else None
    header = []
    for line in lines[1:]:
        if line.startswith(("@@", "GIT binary patch", "Binary files ")):
            break
        header.append(line)
    new = any(line.startswith("new file mode") for line in header)
    deleted = any(line.startswith("deleted file mode") for line in header)
    renamed = any(line.startswith(("rename from", "rename to", "copy from")) for line in header)
    for line in header:
        if line.startswith("+++ b/"):
            path = line[len("+++ b/"):].rstrip("\t")
    return {"path": path, "new": new, "deleted": deleted, "renamed": renamed}


def tg_test_patch(snapshot: str, environment_patch: str) -> tuple[str, list[str], list[dict[str, str]]]:
    """Keep only new test files the author created; drop the image baseline; record the rest."""
    baseline = {section_info(s)["path"]: s for s in patch_sections(environment_patch)}
    kept: list[str] = []
    files: list[str] = []
    violations: list[dict[str, str]] = []
    for section in patch_sections(snapshot):
        info = section_info(section)
        path = info["path"] or "?"
        if baseline.get(path) == section:
            continue
        if info["new"] and not info["renamed"] and is_test_path(path):
            kept.append(section)
            files.append(path)
        else:
            reason = ("deleted existing file" if info["deleted"] else "modified existing file"
                      if not info["new"] else "new non-test file")
            violations.append({"path": path, "reason": reason})
    return "".join(kept), files, violations


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def squash(text: str) -> str:
    return " ".join(str(text).split())


def quote_in_task(quote: Any, instruction: str) -> bool:
    if not isinstance(quote, str):
        return False
    value = squash(quote).strip(" .\"'`")
    return len(value) >= MIN_QUOTE_CHARS and value in squash(instruction)


def parse_manifest(text: str | None) -> Any:
    if not text:
        return None
    try:
        return json.loads(text)
    except ValueError:
        return None


def validate_manifest(payload: Any, instruction: str, kept_files: list[str]) -> dict[str, Any]:
    """Normalize the manifest; every problem keeps the affected test out of the gate."""
    problems: list[str] = []
    if not isinstance(payload, dict) or not isinstance(payload.get("tests"), list):
        return {"setup": None, "requirements": [], "tests": [],
                "problems": ["manifest missing or has no tests array"]}
    requirements: dict[str, dict[str, Any]] = {}
    for item in payload.get("requirements") or []:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            problems.append("requirement without id")
            continue
        requirements[item["id"]] = {"id": item["id"], "quote": item.get("quote"),
                                    "quote_ok": quote_in_task(item.get("quote"), instruction)}
    kept = set(kept_files)
    tests: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(payload["tests"]):
        if index >= MAX_TESTS:
            problems.append(f"tests beyond {MAX_TESTS} ignored")
            break
        if not isinstance(item, dict):
            problems.append(f"tests[{index}] is not an object")
            continue
        tid = item.get("id") if isinstance(item.get("id"), str) else f"#{index}"
        errors = []
        if tid in seen:
            errors.append("duplicate id")
        seen.add(tid)
        kind = item.get("kind")
        if kind not in KINDS:
            errors.append(f"unknown kind {kind!r}")
        command = item.get("command")
        if not isinstance(command, str) or not command.strip():
            errors.append("missing command")
        files = [f for f in item.get("files") or [] if isinstance(f, str)]
        files = [f[len("/app/"):] if f.startswith("/app/") else f for f in files]
        if kind in ("new", "preserve"):
            if not files:
                errors.append("a new/preserve test needs its new test file")
            missing = [f for f in files if f not in kept]
            if missing:
                errors.append("test files not in the kept test patch: " + ", ".join(missing))
        linked = [requirements[r] for r in item.get("requirements") or [] if isinstance(r, str)
                  and r in requirements]
        quote_ok = any(r["quote_ok"] for r in linked)
        tests.append({"id": tid, "kind": kind, "category": item.get("category"), "files": files,
                      "command": command if isinstance(command, str) else "",
                      "requirements": [r["id"] for r in linked], "quote_ok": quote_ok,
                      "asserts": item.get("asserts"), "catches": item.get("catches"),
                      "valid": not errors, "errors": errors})
    setup = payload.get("setup")
    return {"setup": setup if isinstance(setup, str) and setup.strip() else None,
            "requirements": list(requirements.values()), "tests": tests, "problems": problems}


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------

def status(result: dict[str, Any] | None) -> str:
    if not result:
        return "missing"
    if result.get("timeout"):
        return "timeout"
    return "pass" if result.get("returncode") == 0 else "fail"


def classify(test: dict[str, Any], base: dict[str, Any] | None) -> str:
    """hard / invalid / unquoted / base_pass (vacuous new test) / base_fail (broken preserve)."""
    if not test["valid"]:
        return "invalid"
    observed = status(base)
    if test["kind"] == "existing":
        return "hard" if observed == "pass" else "base_fail"
    if not test["quote_ok"]:
        return "unquoted"
    if test["kind"] == "new":
        return "hard" if observed in ("fail", "timeout") else "base_pass"
    return "hard" if observed == "pass" else "base_fail"


# ---------------------------------------------------------------------------
# Interface-guess check (v7.1): names the baseline reports missing must come from the task
# ---------------------------------------------------------------------------

_MISSING_NAME_PATTERNS = (
    r"undefined: (?:[A-Za-z_]\w*\.)?([A-Za-z_]\w*)",                      # Go
    r"has no field or method ([A-Za-z_]\w*)",                               # Go
    r"cannot find (?:function|value|type|struct|trait|macro|module|crate)(?: `[^`]*`)? `?([A-Za-z_][\w:]*)`?",  # Rust
    r"no (?:method|function|associated item|field|variant) named `([^`]+)`",  # Rust
    r"unresolved imports? `([^`]+)`",                                       # Rust
    r"([A-Za-z_$][\w$]*) is not a function",                                # JS/TS
    r"([A-Za-z_$][\w$]*) is not a constructor",                             # JS/TS
    r"does not provide an export named '([^']+)'",                          # ESM
    r"cannot import name '([^']+)'",                                        # Python
    r"has no attribute '([^']+)'",                                          # Python
    r"No module named '([^']+)'",                                           # Python
    r"Cannot find module '([^']+)'",                                        # Node
    r"package ([\w./-]+) is not in (?:GOROOT|std)",                         # Go
    r"cannot find package \"([^\"]+)\"",                                    # Go
    r"imports ([\w./-]+): build constraints exclude",                       # Go
)


def missing_names(output: str) -> list[str]:
    names: list[str] = []
    for pattern in _MISSING_NAME_PATTERNS:
        for match in re.finditer(pattern, output or ""):
            value = match.group(1).strip()
            if value and value not in names:
                names.append(value)
    return names


def guessed_names(output: str, instruction: str) -> list[str]:
    """Missing names (reported by the baseline run) that the task text never mentions."""
    guessed = []
    for name in missing_names(output):
        parts = [p for p in re.split(r"[./:]+", name) if p]
        tail = parts[-1] if parts else name
        if tail not in instruction and name not in instruction:
            guessed.append(name)
    return guessed


# ---------------------------------------------------------------------------
# Blind test audit (v7.1): task text + test code only, never an implementation
# ---------------------------------------------------------------------------

AUDIT_MAX_FILE_CHARS = 60_000
AUDIT_MAX_TOTAL_CHARS = 200_000
AUDIT_CRITERIA = ("D1", "D2", "D3", "D4", "D5")
AUDIT_SYSTEM = """You audit acceptance tests that were written for a feature request before the
feature was implemented. You see only the task text and the test code, never an
implementation. Decide for every listed test whether a correct implementation of the task
is guaranteed to pass it. Drop a test when any of these applies:
D1 over-specified: it asserts a value, message text, ordering, count, timing, category,
   default or side effect that the task text does not state. Consequences that follow
   necessarily from the task text are fine.
D2 implementation-coupled: it mocks, spies on or patches internal functions, methods or
   attributes that the task does not name, asserts how often internals are called, or
   reads private state.
D3 guessed interface: it imports or calls a new name, signature, option, file path or
   export location that the task text does not give. Names that clearly belong to the
   existing project are fine.
D4 scenario mismatch: its setup or input misreads the task, or depends on conditions the
   task does not define, so a correct implementation could reasonably fail it. For an
   "existing" entry (a command running existing project tests), drop it when the task
   changes the behavior those tests are likely to cover.
D5 broken test: the test would fail for reasons unrelated to the feature.
Keep a test only when none applies. When unsure, drop it.
Return only a JSON object, listing every test id exactly once:
{"tests": [{"id": "T1", "decision": "keep" or "drop", "criteria": ["D1"], "reason": "<one sentence>"}]}"""


def patch_new_files(patch: str) -> dict[str, str]:
    """Contents of the files a new-files-only patch creates."""
    files: dict[str, str] = {}
    for section in patch_sections(patch):
        info = section_info(section)
        if not info["new"] or not info["path"]:
            continue
        body, started = [], False
        for line in section.splitlines():
            if line.startswith("@@"):
                started = True
                continue
            if started and line.startswith("+"):
                body.append(line[1:])
        files[info["path"]] = "\n".join(body) + ("\n" if body else "")
    return files


def audit_messages(instruction: str, manifest: dict[str, Any], test_patch: str) -> list[dict[str, str]]:
    quotes = {r["id"]: r.get("quote") for r in manifest["requirements"]}
    entries = [{"id": t["id"], "kind": t["kind"], "category": t["category"], "files": t["files"],
                "command": t["command"], "asserts": t["asserts"],
                "quotes": [quotes.get(r) for r in t["requirements"]]}
               for t in manifest["tests"] if t["valid"]]
    parts, total = [], 0
    for path, content in patch_new_files(test_patch).items():
        if len(content) > AUDIT_MAX_FILE_CHARS:
            content = content[:AUDIT_MAX_FILE_CHARS] + "\n[... file truncated ...]\n"
        if total + len(content) > AUDIT_MAX_TOTAL_CHARS:
            parts.append(f"### {path}\n[omitted: size limit]\n")
            continue
        total += len(content)
        parts.append(f"### {path}\n```\n{content}```\n")
    user = ("TASK\n<task>\n" + instruction.strip() + "\n</task>\n\nTESTS (manifest entries)\n"
            + json.dumps(entries, ensure_ascii=False, indent=1) + "\n\nTEST FILES\n" + "\n".join(parts))
    return [{"role": "system", "content": AUDIT_SYSTEM}, {"role": "user", "content": user}]


def parse_audit(text: str, test_ids: list[str]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Decisions by test id; a test the audit omits or garbles counts as dropped."""
    problems: list[str] = []
    cleaned = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, re.S)
    if fence:
        cleaned = fence.group(1)
    try:
        payload = json.loads(cleaned)
    except ValueError:
        return {tid: {"decision": "drop", "criteria": [], "reason": "audit output unparsable"}
                for tid in test_ids}, ["audit output is not JSON"]
    decisions: dict[str, dict[str, Any]] = {}
    for item in payload.get("tests", []) if isinstance(payload, dict) else []:
        if not isinstance(item, dict) or item.get("id") not in test_ids or item["id"] in decisions:
            problems.append(f"unexpected audit entry {item!r:.80}")
            continue
        decision = item.get("decision") if item.get("decision") in ("keep", "drop") else "drop"
        decisions[item["id"]] = {"decision": decision,
                                 "criteria": [c for c in item.get("criteria") or [] if c in AUDIT_CRITERIA],
                                 "reason": str(item.get("reason") or "")[:500]}
    for tid in test_ids:
        if tid not in decisions:
            problems.append(f"{tid} missing from audit")
            decisions[tid] = {"decision": "drop", "criteria": [], "reason": "missing from audit"}
    return decisions, problems


def candidate_verdict(tests: list[dict[str, Any]], base: dict[str, Any], first: dict[str, Any],
                      rerun: dict[str, Any], *, exclude: set[str] | None = None) -> dict[str, Any]:
    """Reject when a hard test fails on both runs; a pass on either run marks it flaky/passing."""
    exclude = exclude or set()
    hard = [t["id"] for t in tests if classify(t, base.get(t["id"])) == "hard" and t["id"] not in exclude]
    failing, flaky = [], []
    for tid in hard:
        if status(first.get(tid)) == "pass":
            continue
        if status(rerun.get(tid)) == "pass":
            flaky.append(tid)
        else:
            failing.append(tid)
    return {"reject": bool(failing), "hard": len(hard), "failing_hard": failing, "flaky": flaky}
