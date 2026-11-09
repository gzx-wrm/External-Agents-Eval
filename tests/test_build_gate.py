from __future__ import annotations

import asyncio

from external_agents_eval.workflows.solve_review_repair.build_gate import (
    FAIL,
    PASS,
    UNAVAILABLE,
    BuildGate,
    CommandOutcome,
    GateCommand,
    build_check_for_repair,
    changed_paths,
    classify,
    compare,
    detect_commands,
    refine,
    tail,
)

CARGO_WS = '[workspace]\nmembers = ["core/engine"]\n'
CARGO_PKG = '[package]\nname = "boa_engine" # comment\nversion = "0.1.0"\n\n[dependencies]\nname = "x"\n'


def patch_for(*paths: str) -> bytes:
    return "".join(f"diff --git a/{p} b/{p}\n--- a/{p}\n+++ b/{p}\n" for p in paths).encode()


def test_changed_paths_handles_renames_and_order():
    patch = b"diff --git a/src/a.py b/src/b.py\nsimilarity index 90%\ndiff --git a/x.go b/x.go\n"
    assert changed_paths(patch) == ["src/a.py", "src/b.py", "x.go"]


def test_classify_and_compare_are_differential():
    assert classify(0) == PASS and classify(2) == FAIL and classify(124) == "timeout"
    assert classify(None) == "error" and classify(0, timed_out=True) == "timeout"
    ok, bad = CommandOutcome(PASS, 0, 1.0), CommandOutcome(FAIL, 1, 1.0)
    slow = CommandOutcome("timeout", 124, 600.0)
    assert compare(ok, bad) == FAIL
    assert compare(ok, ok) == PASS
    assert compare(bad, bad) == UNAVAILABLE  # pre-existing breakage is no signal
    assert compare(bad, ok) == UNAVAILABLE
    assert compare(ok, slow) == UNAVAILABLE  # a timeout is never a failure


def test_rust_scopes_to_owning_package_and_ignores_dependency_name_keys():
    files = {"Cargo.toml", "core/engine/Cargo.toml", "core/engine/src/lib.rs"}
    reads = {"Cargo.toml": CARGO_WS, "core/engine/Cargo.toml": CARGO_PKG}
    commands = detect_commands(files, ["core/engine/src/lib.rs", "core/engine/tests/new.rs"],
                               reads.get)
    assert commands == [GateCommand("rust", ".", "cargo test --no-run --offline --quiet -p boa_engine")]


def test_go_python_ts_deno_detection_is_scoped():
    files = {"go.mod", "vm/vm.go", "pyproject.toml", "pkg/mod.py", "libs/core/pyproject.toml",
             "libs/core/x/y.py", "tsconfig.json", "src/a.ts", "tools/deno.json", "tools/m.ts"}
    reads = {"pyproject.toml": "[project]\nname='a'\n",
             "libs/core/pyproject.toml": "[tool.pytest.ini_options]\naddopts=''\n"}
    commands = detect_commands(
        files, ["vm/vm.go", "libs/core/x/y.py", "pkg/mod.py", "src/a.ts", "tools/m.ts", "README.md"],
        reads.get)
    assert GateCommand("go", ".", "go test -count=1 -run '^$' ./...") in commands
    assert GateCommand("python", "libs/core",
                       "python -m pytest --collect-only -q -p no:cacheprovider") in commands
    # The root pyproject has no pytest configuration, so pkg/mod.py adds no command.
    assert [c for c in commands if c.ecosystem == "python"] == [
        GateCommand("python", "libs/core", "python -m pytest --collect-only -q -p no:cacheprovider")]
    assert any(c.ecosystem == "typescript" and c.cwd == "." for c in commands)
    assert GateCommand("deno", "tools", "deno test --no-run --cached-only") in commands
    # tools/m.ts belongs to Deno, so it does not also produce a tsc command.
    assert not any(c.ecosystem == "typescript" and c.cwd == "tools" for c in commands)


def test_unsupported_changes_produce_no_command():
    assert detect_commands({"README.md", "docs/a.md"}, ["docs/a.md"], {}.get) == []


def test_log_tail_is_bounded():
    text = "\n".join(f"line {i}" for i in range(1000))
    out = tail(text)
    assert out.splitlines()[-1] == "line 999" and len(out.splitlines()) == 200
    assert len(tail("x" * 100_000).encode()) <= 16 * 1024


class FakeExecutor:
    def __init__(self, files, reads, results):
        self.files, self.reads, self.results = files, reads, results
        self.calls = []

    async def list_files(self):
        return sorted(self.files)

    async def read_file(self, path):
        return self.reads.get(path)

    async def run(self, label, patch, commands, timeout_sec):
        self.calls.append((label, [c.command for c in commands], timeout_sec))
        return [self.results[(label, c.ecosystem)] for c in commands]


def test_gate_caches_base_and_reports_failures_for_repair_only():
    files = {"go.mod", "vm/vm.go"}
    results = {("base", "go"): CommandOutcome(PASS, 0, 3.0, "ok"),
               ("P0", "go"): CommandOutcome(FAIL, 1, 2.0, "vm/x_test.go:3: undefined: Foo"),
               ("P1", "go"): CommandOutcome(PASS, 0, 2.0, "ok")}
    executor = FakeExecutor(files, {}, results)
    gate = BuildGate(executor, run_timeout_sec=600)
    p0 = asyncio.run(gate.check("P0", patch_for("vm/vm.go", "vm/x_test.go")))
    p1 = asyncio.run(gate.check("P1", patch_for("vm/vm.go")))
    assert p0["verdict"] == FAIL and p1["verdict"] == PASS
    assert [call[0] for call in executor.calls] == ["base", "P0", "P1"]  # base ran once
    check = build_check_for_repair(p0)
    assert check["failures"][0]["log_tail"].endswith("undefined: Foo")
    assert build_check_for_repair(p1) is None
    assert "log_tail" not in p0["commands"][0]["candidate"]


def test_gate_skips_candidate_when_base_cannot_build():
    files = {"go.mod", "vm/vm.go"}
    results = {("base", "go"): CommandOutcome(FAIL, 1, 1.0, "network")}
    executor = FakeExecutor(files, {}, results)
    report = asyncio.run(BuildGate(executor).check("P0", patch_for("vm/vm.go")))
    assert report["verdict"] == UNAVAILABLE
    assert [call[0] for call in executor.calls] == ["base"]


def test_gate_without_supported_changes_is_unavailable_without_running():
    executor = FakeExecutor({"README.md"}, {}, {})
    report = asyncio.run(BuildGate(executor).check("P0", patch_for("README.md")))
    assert report["verdict"] == UNAVAILABLE and executor.calls == []


def test_typescript_semantic_type_errors_are_not_build_failures():
    ts = GateCommand("typescript", ".", "tsc --noEmit -p tsconfig.json")
    semantic = CommandOutcome(FAIL, 2, 1.0, "a.test.ts(2,4): error TS2345: Argument ...",
                              ("TS2345", "TS7006"))
    assert refine(ts, semantic).status == PASS and refine(ts, semantic).note
    syntax = CommandOutcome(FAIL, 2, 1.0, "a.ts(1,1): error TS1005: ';' expected.")
    assert refine(ts, syntax).status == FAIL  # codes recovered from the log when absent
    config = CommandOutcome(FAIL, 1, 1.0, "error TS5058: The specified path does not exist")
    assert refine(ts, config).status == FAIL
    assert refine(ts, CommandOutcome(FAIL, 1, 1.0, "Segmentation fault")).status == FAIL
    go = GateCommand("go", ".", "go test")
    assert refine(go, semantic).status == FAIL  # only TypeScript is refined


def test_gate_applies_refinement_to_base_and_candidate():
    files = {"tsconfig.json", "src/a.ts"}
    results = {("base", "typescript"): CommandOutcome(FAIL, 2, 3.0, "error TS2322: old", ("TS2322",)),
               ("P0", "typescript"): CommandOutcome(FAIL, 2, 3.0, "error TS1005: x", ("TS1005",)),
               ("P1", "typescript"): CommandOutcome(FAIL, 2, 3.0, "error TS2345: y", ("TS2345",))}
    gate = BuildGate(FakeExecutor(files, {}, results))
    assert asyncio.run(gate.check("P0", patch_for("src/a.ts")))["verdict"] == FAIL
    p1 = asyncio.run(gate.check("P1", patch_for("src/a.ts")))
    assert p1["verdict"] == PASS and p1["commands"][0]["candidate"]["note"]
