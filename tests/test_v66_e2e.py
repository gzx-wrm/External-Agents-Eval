"""V6.6 downstream pieces that need neither Docker nor model calls."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from external_agents_eval.agents.mini_v66_review import V66MiniReviewAgent, V66ReviewController
from external_agents_eval.workflows.solve_review_repair import v5_controller
from external_agents_eval.workflows.solve_review_repair.v66_review_prompts import (
    SOLVER_HEADING,
    augment_solver_task,
    lane_addendum,
    summary_addendum,
)

PROJECT = Path(__file__).resolve().parents[1]
ledger_input = importlib.util.module_from_spec(
    spec := importlib.util.spec_from_file_location("li", PROJECT / "scripts/deepswe_ft_m1/ledger_input.py"))
spec.loader.exec_module(ledger_input)


def cp(targets, resolution="analog", **extra):
    value = {"targets": targets, "facet": "D2", "given": "g", "when": "w", "then": "t",
             "resolution": resolution, "rules_out": None if resolution == "open" else "r",
             "alternatives": ["a", "b"] if resolution == "open" else [], "favored": None,
             "basis_quote": "q" if resolution in {"analog", "contract"} else None,
             "basis_behavior": "b", "convention": None}
    value.update(extra)
    return value


POINTS = [cp(["O001"], "open"), cp(["O002"]), cp(["P001"]), cp(["O001", "P002"], "task")]


def test_solver_input_keeps_the_ledger_supplement_and_adds_one_checkpoint_section():
    task = {"task_id": "t", "instruction": "Do it."}
    ledger = {"summary": "s", "obligations": [], "preservation_obligations": [], "open_questions": []}
    with_ledger = ledger_input.augment_task(task, ledger)
    result = augment_solver_task(with_ledger, "- [open] O001/D2: given g\n")
    assert result["instruction"].startswith(with_ledger["instruction"].rstrip("\n"))
    assert result["instruction"].count(SOLVER_HEADING) == 1
    assert result["instruction"].count(ledger_input.HEADING) == 1
    assert result["instruction"].rstrip().endswith("- [open] O001/D2: given g")
    assert task["instruction"] == "Do it."


def test_lane_addenda_split_by_role_with_global_numbers():
    i_text, r_text = lane_addendum("I", POINTS), lane_addendum("R", POINTS)
    assert "CP1 [open] O001/D2" in i_text and "CP2 [analog] O002/D2" in i_text
    assert "CP4 [task] O001,P002/D2" in i_text and "CP3" not in i_text
    assert "CP3 [analog] P001/D2" in r_text and "CP4" in r_text and "CP1" not in r_text
    assert "Candidate readings: a | b" in i_text and 'Basis: "q" (b)' in i_text
    assert "makes its target violated" in i_text and "never for a missing probe" in i_text
    assert lane_addendum("R", [cp(["O001"])]) == ""


def test_summary_addendum_lists_every_point_and_the_decision_order():
    text = summary_addendum(POINTS)
    for index in range(1, 5):
        assert f"CP{index} " in text
    assert "task text > existing repository analog > downstream contract" in text
    assert summary_addendum([]) == ""


def test_controller_appends_checkpoints_to_every_summary_only(monkeypatch):
    monkeypatch.setattr(v5_controller.V5Controller, "_summary_prompt", lambda self, s, **kw: "SUMMARY\n")
    controller = object.__new__(V66ReviewController)
    controller.checkpoints = POINTS
    assert controller._summary_prompt({}, freeze_id="f").startswith("SUMMARY\n\nCHECK POINTS AND DECISION ORDER")
    assert "_lane_prompt" not in V66ReviewController.__dict__
    controller.checkpoints = []
    assert controller._summary_prompt({}) == "SUMMARY\n"


def test_agent_builds_the_v66_controller_with_the_frozen_points(tmp_path, monkeypatch):
    path = tmp_path / "review-targets.json"
    path.write_text(json.dumps(POINTS))
    monkeypatch.setattr(V66MiniReviewAgent.__mro__[1], "__init__", lambda self, *a, **k: None)
    agent = V66MiniReviewAgent(checkpoints=str(path))
    monkeypatch.setattr(V66ReviewController.__mro__[1], "__init__", lambda self, **kw: None)
    controller = agent._make_controller()
    assert isinstance(controller, V66ReviewController) and controller.checkpoints == POINTS
    from external_agents_eval.agents.mini_v66_review import V66FrozenReviewRunner
    monkeypatch.setattr(V66FrozenReviewRunner.__mro__[1], "__init__", lambda self, **kw: None)
    runner = agent._make_role_runner()
    assert isinstance(runner, V66FrozenReviewRunner) and runner.v66_checkpoints == POINTS
    assert V66MiniReviewAgent.name() == "v6.6-mini-review"


def test_e2e_runner_sources_exist():
    spec = importlib.util.spec_from_file_location("v66e2e", PROJECT / "scripts/deepswe_v66_e2e/run.py")
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    for name in run.FROZEN_SOURCES:
        assert (PROJECT / name).is_file(), name
    for name in ("external_agents_eval/agents/mini_v66_review.py",
                 "external_agents_eval/workflows/solve_review_repair/v66_review_prompts.py",
                 "external_agents_eval/workflows/solve_review_repair/v5_controller.py",
                 "external_agents_eval/workflows/solve_review_repair/v5_contracts.py"):
        assert name in run.FROZEN_SOURCES
    with pytest.raises((ValueError, FileNotFoundError)):
        run.prepare(Path("/nonexistent-root-for-test"), Path("/nonexistent-m0"), Path("/x"), Path("/y"), ["t"], None)
