"""V6.4 build gate inside the V5 controller: routing, Repair context and delivery."""
from __future__ import annotations

import asyncio
import json

import pytest
from test_v5_controller import RepairWorkspace, ScriptedRunner, _assessment, _spec

from external_agents_eval.workflows.solve_review_repair.artifacts import ArtifactStore, sha256_bytes
from external_agents_eval.workflows.solve_review_repair.models import RoleName, TaskInput
from external_agents_eval.workflows.solve_review_repair.v5_contracts import V5ContractError
from external_agents_eval.workflows.solve_review_repair.v5_controller import V5Controller
from external_agents_eval.workflows.solve_review_repair.v5_prompts import I_PROBE_RULES_V56
from external_agents_eval.workflows.solve_review_repair.v64_config import V64Config

ACCEPT = {"summary": "No repair is required.", "decisions": [], "repair_requests": [],
          "new_questions": []}
REPAIR = {
    "summary": "One repair is required.",
    "decisions": [{"finding_ref": "I/O001", "disposition": "repair",
                   "reason": "The required behavior is violated.", "supporting_refs": ["I/O001"],
                   "sources": [{"kind": "task", "field": "description", "quote": "Add behavior"}]}],
    "repair_requests": [{"finding_refs": ["I/O001"], "change": "Implement the behavior.",
                         "validation_record_refs": ["I/O001"], "preserve_refs": ["R/P001"]}],
    "new_questions": [],
}


class FakeGate:
    def __init__(self, verdicts, *, error=None):
        self.verdicts, self.error, self.calls = dict(verdicts), error, []

    def config(self):
        return {"schema_version": "v6.4-build-gate-1", "run_timeout_sec": 600}

    async def check(self, label, patch):
        self.calls.append((label, patch))
        if self.error:
            raise self.error
        verdict = self.verdicts[label]
        logs = [{"command": "go test -count=1 -run '^$' ./...", "cwd": ".", "exit_code": 1,
                 "log_tail": "x_test.go:3: undefined: Foo"}] if verdict == "fail" else []
        return {"label": label, "patch_sha256": sha256_bytes(patch), "verdict": verdict,
                "commands": [], "failure_logs": logs}


def run(tmp_path, *, s_output, i_status="satisfied", gate, run_id="v64"):
    scripts = {
        RoleName.V5_SPECIFICATION_REVIEWER: [_spec()],
        RoleName.V5_IMPLEMENTATION_REVIEWER: [_assessment("O001", status=i_status)],
        RoleName.V5_REGRESSION_REVIEWER: [_assessment("P001")],
        RoleName.V5_SUMMARY_REVIEWER: [s_output],
        RoleName.REPAIR: ["repair complete"],
    }
    runner = ScriptedRunner(tmp_path, scripts)
    controller = V5Controller(role_runner=runner, workspace=RepairWorkspace(),
                              artifacts=ArtifactStore(tmp_path / "artifacts"),
                              config=V64Config(), build_gate=gate)
    try:
        result = asyncio.run(controller.run(
            TaskInput("task", "Add behavior", "repo", "base", "title"),
            candidate_patch=b"p0", base_worktree="base", candidate_worktree="candidate",
            repair_worktree="repair", run_id=run_id,
        ))
    finally:
        controller.close()
    artifacts = tmp_path / "artifacts/v5"
    return result, runner, artifacts


def test_v64_requires_a_gate():
    with pytest.raises(V5ContractError, match="build gate"):
        V5Controller(role_runner=object(), workspace=object(), artifacts=None, config=V64Config())


def test_failing_p0_build_forces_repair_with_build_check_and_keeps_buildable_p1(tmp_path):
    gate = FakeGate({"P0": "fail", "P1": "pass"})
    result, runner, artifacts = run(tmp_path, s_output=ACCEPT, gate=gate)
    assert result.route == "repair" and result.p1_patch == b"p1"
    repair = next(c for c in runner.call_details if c["role"] == RoleName.REPAIR)["prompt"]
    assert repair.startswith("ROLE: Repair\nGoal:") and "BUILD CHECK\n" in repair
    assert "undefined: Foo" in repair and "about 30 minutes and 250 commands" in repair
    decision = json.loads((artifacts / "decision.json").read_text())
    assert decision["route"] == "repair"
    assert decision["build_gate"] == {"P0": "fail", "route_override": True}
    summary = json.loads((artifacts / "build-gate.json").read_text())
    assert summary["P0"] == "fail" and summary["P1"] == "pass" and summary["delivered"] == "P1"
    assert [label for label, _ in gate.calls] == ["P0", "P1"]
    assert result.build_gate == summary
    # I/R and S never see the gate result.
    for call in runner.call_details:
        if call["role"] != RoleName.REPAIR:
            assert "undefined: Foo" not in call["prompt"]


def test_repair_that_breaks_a_buildable_p0_delivers_p0(tmp_path):
    gate = FakeGate({"P0": "pass", "P1": "fail"})
    result, runner, artifacts = run(tmp_path, s_output=REPAIR, i_status="violated", gate=gate)
    assert result.route == "repair" and result.p1_patch == b"p0"
    repair = next(c for c in runner.call_details if c["role"] == RoleName.REPAIR)["prompt"]
    assert "BUILD CHECK\n" not in repair
    summary = json.loads((artifacts / "build-gate.json").read_text())
    assert summary["delivered"] == "P0" and summary["route_override"] is False
    assert (artifacts.parent / "outputs/v5-p1.patch").read_bytes() == b"p0"
    events = (artifacts.parent / "events.jsonl").read_text()
    assert "V64_BUILD_GATE_REVERT_TO_P0" in events


def test_unavailable_gate_changes_nothing(tmp_path):
    gate = FakeGate({"P0": "unavailable"})
    result, runner, artifacts = run(tmp_path, s_output=ACCEPT, gate=gate)
    assert result.route == "accept" and result.p1_patch == b"p0"
    assert RoleName.REPAIR not in runner.calls
    summary = json.loads((artifacts / "build-gate.json").read_text())
    assert summary == {**summary, "P0": "unavailable", "P1": None, "delivered": "P0",
                       "route_override": False}


def test_gate_errors_are_recorded_as_unavailable(tmp_path):
    gate = FakeGate({}, error=RuntimeError("docker down"))
    result, _, artifacts = run(tmp_path, s_output=ACCEPT, gate=gate)
    assert result.route == "accept"
    report = json.loads((artifacts / "build-gate/P0.json").read_text())
    assert report["verdict"] == "unavailable" and "docker down" in report["error"]


def test_v64_reviewers_receive_v56_prompts(tmp_path):
    _, runner, _ = run(tmp_path, s_output=ACCEPT, gate=FakeGate({"P0": "pass"}))
    i_prompt = next(c for c in runner.call_details
                    if c["role"] == RoleName.V5_IMPLEMENTATION_REVIEWER)["prompt"]
    r_prompt = next(c for c in runner.call_details
                    if c["role"] == RoleName.V5_REGRESSION_REVIEWER)["prompt"]
    assert I_PROBE_RULES_V56 in i_prompt and I_PROBE_RULES_V56 not in r_prompt
    assert "v6.4-review-note-files/SKILL.md" in i_prompt
    assert "\nOUTPUT CONTRACT\n" in i_prompt and "\nOUTPUT CONTRACT\n" in r_prompt
