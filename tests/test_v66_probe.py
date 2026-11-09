"""V6.6 probe stage logic that needs neither Docker nor model calls."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from external_agents_eval.workflows.solve_review_repair.v66_probe import (
    RESULT_PATH,
    convert,
    open_points,
    probe_task,
    validate_result,
    verify,
)

PROJECT = Path(__file__).resolve().parents[1]
LEDGER = {"summary": "s", "obligations": [{"id": "O001", "content": {"statement": "Added(p) reports new matches"}}],
          "preservation_obligations": [], "open_questions": []}


def cp(resolution="open", **extra):
    value = {"targets": ["O001"], "facet": "D6", "given": "an entity matches before the first query",
             "when": "the first Added(p) query runs", "then": "the first-run behavior is open",
             "resolution": resolution, "strength": "open" if resolution == "open" else "required",
             "rules_out": None if resolution == "open" else "r", "sources": [], "basis_quote": None,
             "basis_behavior": None, "convention": None,
             "alternatives": ["it is reported", "it is not reported"] if resolution == "open" else [],
             "favored": None}
    value.update(extra)
    return value


POINTS = [cp("task"), cp(), cp("analog"), cp(facet="D2", alternatives=["x", "y"])]


def resolved_entry(cp_id="CP2", reading="it is reported", quote="FIRST RUN: 1"):
    return {"cp": cp_id, "decision": "resolved", "reading": reading,
            "analog": {"path": "src/added.ts", "line_start": 3, "line_end": 9,
                       "same_kind_reason": "trait Added is the same tracking kind"},
            "probe": {"command": "node /tmp/v66-probe/a.js", "observation_quote": quote}, "reason": "r"}


def test_open_points_use_the_global_review_numbering():
    assert [cp_id for cp_id, _ in open_points(POINTS)] == ["CP2", "CP4"]


def test_probe_task_lists_open_points_readings_and_the_result_contract():
    text = probe_task({"instruction": "Add predicates."}, LEDGER, POINTS)
    assert "CP2 [D6] targets O001: Added(p) reports new matches" in text and "CP4 [D2]" in text
    assert "(1) it is reported" in text and "(2) it is not reported" in text
    assert "- CP1 " not in text and "- CP3 " not in text
    assert RESULT_PATH in text and "must not implement it" in text and "Never use the network" in text


def test_validation_keeps_only_well_formed_resolutions():
    payload = {"checkpoints": [resolved_entry(), {"cp": "CP4", "decision": "open"}]}
    resolved, problems = validate_result(payload, POINTS)
    assert list(resolved) == ["CP2"] and problems == []
    bad = {"checkpoints": [resolved_entry(reading="it might be reported"), {"cp": "CP9", "decision": "open"}]}
    resolved, problems = validate_result(bad, POINTS)
    assert resolved == {}
    assert any("not one of the candidate readings" in p for p in problems)
    assert any("unknown check point 'CP9'" in p for p in problems)
    assert any("CP4 missing" in p for p in problems)
    no_probe = {"checkpoints": [{**resolved_entry(), "probe": None}, {"cp": "CP4", "decision": "open"}]}
    assert validate_result(no_probe, POINTS)[0] == {}
    assert validate_result(None, POINTS) == ({}, ["result has no checkpoints array"])


def test_verification_requires_the_quote_in_the_fresh_output():
    resolved = {"CP2": resolved_entry()}
    assert verify(resolved, lambda c: "noise\nFIRST   RUN: 1\n")["CP2"]["verified"] is True
    assert verify(resolved, lambda c: "FIRST RUN: 0")["CP2"]["verified"] is False

    def boom(command):
        raise TimeoutError("slow")

    failed = verify(resolved, boom)["CP2"]
    assert failed["verified"] is False and "TimeoutError" in failed["rerun_error"]


def test_conversion_turns_only_verified_points_into_analog_points():
    checked = {"CP2": {**resolved_entry(), "verified": True}, "CP4": {**resolved_entry("CP4", "x"), "verified": False}}
    converted = convert(POINTS, checked)
    assert [p["resolution"] for p in converted] == ["task", "analog", "analog", "open"]
    point = converted[1]
    assert point["then"].endswith("Resolved by probe: it is reported")
    assert point["rules_out"] == "it is not reported" and point["alternatives"] == []
    assert point["basis_quote"] == "FIRST RUN: 1" and point["probe"]["verified"] is True
    assert point["sources"] == [{"kind": "repository", "path": "src/added.ts", "line_start": 3, "line_end": 9}]
    assert converted[3] == POINTS[3]


def test_probe_config_replaces_only_the_task_section_and_limits():
    spec = importlib.util.spec_from_file_location("v66probe", PROJECT / "scripts/deepswe_v66_probe/run.py")
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    config = {"agent": {"instance_template": "Please solve this issue: {{task}}\nSTEPS\n## Command Execution Rules\nRULES",
                        "system_template": "S", "step_limit": 0, "cost_limit": 0, "wall_time_limit_seconds": 10800},
              "model": {"m": 1}, "environment": {"e": 1}}
    probe = run.probe_config(config)
    assert probe["agent"]["instance_template"].startswith("{{task}}")
    assert probe["agent"]["instance_template"].endswith("## Command Execution Rules\nRULES")
    assert "Please solve this issue" not in probe["agent"]["instance_template"]
    assert probe["agent"]["step_limit"] == run.PROBE_STEP_LIMIT and probe["model"] == config["model"]
    assert config["agent"]["step_limit"] == 0
    for name in run.FROZEN_SOURCES:
        assert (PROJECT / name).is_file(), name
    with pytest.raises(ValueError):
        run.probe_config({"agent": {"instance_template": "no marker"}})


@pytest.mark.parametrize("reading, expected", [
    ("it is reported", "it is reported"),
    ("(1) it is reported", "it is reported"),
    ("1) it is reported", "it is reported"),
    ("(2)", "it is not reported"),
    ("(2) it is reported", None),          # index and text disagree
    ("(3) it is reported", None),          # no such candidate
    ("it might be reported", None),
])
def test_reading_match_accepts_only_the_candidate_or_its_own_index(reading, expected):
    from external_agents_eval.workflows.solve_review_repair.v66_probe import match_reading

    assert match_reading(reading, ["it is reported", "it is not reported"]) == expected
