"""V6.5 DeepSWE runner pieces that do not need Docker or model calls."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT / "scripts/deepswe_v65/run.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("v65_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run = load_runner()


def test_frozen_sources_cover_every_v65_module_and_exist():
    for name in run.FROZEN_SOURCES:
        assert (PROJECT / name).is_file(), name
    package = PROJECT / "external_agents_eval/workflows/solve_review_repair"
    for module in ("review_view.py", "v65_config.py", "v65_prompts.py", "v65_contracts.py",
                   "v65_state.py", "v65_controller.py", "v5_controller.py", "v5_contracts.py"):
        assert f"external_agents_eval/workflows/solve_review_repair/{module}" in run.FROZEN_SOURCES
        assert (package / module).is_file()
    assert (PROJECT / run.SKILL_DIR / "references/disambiguation.md").is_file()
    hashes = run.code_hashes()
    assert set(hashes) == set(run.FROZEN_SOURCES) | {run.SKILL_DIR + "/"}


def test_raw_p0_copies_are_removed_after_normalization(tmp_path):
    checked = []
    native = SimpleNamespace(
        upload=lambda container, data, path: None,
        checked=lambda env, command: checked.append(command) or json.dumps({"ok": True}),
    )
    output = tmp_path / "review"
    output.mkdir()
    solver = tmp_path / "solver"
    solver.mkdir()
    for directory in (output, solver):
        (directory / "environment-baseline.patch").write_bytes(b"baseline")
    (solver / "model.patch").write_bytes(b"p0")

    class Bridge:
        def __init__(self, *args, **kwargs):
            pass

        async def download_file(self, source, target):
            Path(target).write_bytes(b"normalized" if source.endswith("normalized.patch") else b"base")

    original = run.IsolatedEnvironment
    run.IsolatedEnvironment = Bridge
    try:
        run.normalize_p0(native, "env", "container", {"task_id": "t"},
                         {"synthetic_commit": "abc1234"}, solver, output)
    finally:
        run.IsolatedEnvironment = original
    assert checked[-1] == "rm -f -- " + " ".join(run.RAW_P0_TMP)
    assert all(path.startswith("/tmp/v65-") for path in run.RAW_P0_TMP)


def test_pilot_requires_task_ids_and_a_solver_source(tmp_path):
    with pytest.raises(ValueError, match="needs both"):
        run.prepare(tmp_path / "new", [], task_ids=["a"], solver_source=None)


def test_analysis_plan_reuses_the_final_score_only_for_unchanged_p0():
    assert run.analysis_plan(b"p0", b"p0") == "reuse_final_score"
    assert run.analysis_plan(b"p0", b"p1") == "score_p0_separately"
