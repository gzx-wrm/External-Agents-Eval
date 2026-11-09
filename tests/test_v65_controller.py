"""V6.5 controller: review view, assumption extraction and the D lane (fake roles)."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from test_v5_controller import NOTE_TEMPLATE, ScriptedRunner, _assessment
from test_v64_controller import FakeGate

from external_agents_eval.workflows.solve_review_repair.artifacts import ArtifactStore, sha256_bytes
from external_agents_eval.workflows.solve_review_repair.models import RoleName, TaskInput
from external_agents_eval.workflows.solve_review_repair.v5_contracts import (
    V5ContractError,
    validate_summary_output,
)
from external_agents_eval.workflows.solve_review_repair.v65_config import V65Config
from external_agents_eval.workflows.solve_review_repair.v65_controller import (
    V65Controller,
    V65Role,
)
from external_agents_eval.workflows.solve_review_repair.v65_state import V65StateStore

TASK_SOURCE = {"kind": "task", "field": "description", "quote": "Add behavior"}
LEDGER = {
    "summary": "One behavior, one preservation, one question.",
    "obligations": [{"id": "O001", "content": {
        "statement": "The new behavior works.", "importance": "required",
        "sources": [TASK_SOURCE]}}],
    "preservation_obligations": [{"id": "P001", "content": {
        "statement": "The old behavior remains available.", "importance": "required",
        "sources": [TASK_SOURCE]}}],
    "open_questions": [{"id": "Q001", "content": {
        "question": "Does the behavior also apply to the second entry point?",
        "blocking": True, "sources": [TASK_SOURCE]}}],
}
P0 = b"diff --git a/src/a.ts b/src/a.ts\n+// rationale: keep the old reading\n+x\n"
VIEW_MANIFEST = {"schema_version": "v6.5-review-view-1", "stripped_item_count": 1,
                 "stripped": [{"path": "src/a.ts", "items": [
                     {"line": 1, "kind": "comment", "text": "// rationale: keep the old reading"}]}],
                 "skipped": []}
CLAIM = {"claim": "With the feature enabled, parse('k [a] = b') returns an expression statement.",
         "tension": "conflict", "related_target_ids": ["O001"],
         "candidate_refs": [{"path": "src/a.ts", "line_start": 1, "line_end": 2}]}
DERIVED = {"statement": "parse('k [a] = b') fails with the documented error.",
           "importance": "required", "sources": [TASK_SOURCE]}


class ViewWorkspace:
    def __init__(self):
        self.patch = b""
        self.resets, self.views, self.events = [], [], []

    async def apply_patch(self, *, worktree, patch):
        self.patch = patch
        self.events.append(("apply", worktree, patch))

    async def capture_patch(self, *, worktree, base_commit, **_):
        return b"p1" if worktree == "repair" else self.patch

    async def build_review_view(self, *, base_commit, baseline_patch, candidate_patch, level):
        self.views.append(level)
        return b"view:" + level.encode(), VIEW_MANIFEST

    async def discard_review_view(self, *, candidate_patch):
        self.events.append(("discard-view", candidate_patch))

    async def reset_review_worktree(self, *, worktree, base_commit, baseline_patch, patch):
        self.resets.append((worktree, patch))

    async def prepare_review_worktree(self, *, name, base_commit, baseline_patch, patch):
        self.events.append(("prepare", name, patch))
        return "d-worktree"

    async def remove_candidate_inputs(self, *, candidate_patch):
        self.events.append(("remove-inputs", candidate_patch))
        return ["/tmp/external_eval/contracts/apply-x.patch"]

    async def prune_unreachable_objects(self):
        self.events.append(("prune",))
        return {"before": {}, "after": {}}


class V65Runner(ScriptedRunner):
    async def run_role(self, *, role, prompt, **kwargs):
        if role == V65Role.DISAMBIGUATION_REVIEWER and self.scripts[role]:
            value = self.scripts[role][0]
            root = Path(next(line.split(": ", 1)[1] for line in prompt.splitlines()
                             if line.startswith("OUTPUT ROOT:")))
            for assessment in value.get("assessments", []):
                path = root / assessment["review_note_path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(NOTE_TEMPLATE.format(target=assessment["target_id"]))
        return await super().run_role(role=role, prompt=prompt, **kwargs)


def d_batch(*items, new_targets=(), done=True):
    return {"new_targets": list(new_targets), "revisit_target_ids": [], "done": done,
            "assessments": [
                {"target_id": target, "status": status, "expected": "The supported reading.",
                 "observed": "The candidate reading.", "reason": "Class (a) evidence.",
                 "code_refs": [], "review_note_path": f"notes/{target}.md"}
                for target, status in items]}


def decision(ref, disposition, *support):
    return {"finding_ref": ref, "disposition": disposition, "reason": "Adjudicated.",
            "supporting_refs": list(support or [ref]), "sources": [TASK_SOURCE]}


REPAIR_D001 = {
    "summary": "The derived obligation is violated.",
    "decisions": [decision("A/Q001", "dismiss", "D/Q001"), decision("D/X001", "dismiss", "D/D001"),
                  decision("D/D001", "repair")],
    "repair_requests": [{"finding_refs": ["D/D001"], "change": "Reject the bracketed form.",
                         "validation_record_refs": ["D/D001"], "preserve_refs": ["I/O001"]}],
    "new_questions": [],
}


def run(tmp_path, scripts, *, config=None, gate=None, run_id="v65"):
    runner = V65Runner(tmp_path, scripts)
    workspace = ViewWorkspace()
    controller = V65Controller(role_runner=runner, workspace=workspace,
                               artifacts=ArtifactStore(tmp_path / "artifacts"),
                               config=config or V65Config(),
                               build_gate=gate or FakeGate({"P0": "pass"}))
    task = TaskInput("task", "Add behavior", "repo", "base", "title")
    try:
        asyncio.run(controller.prepare_ledger(task, base_worktree="base", imported_ledger=LEDGER))
        result = asyncio.run(controller.run(
            task, candidate_patch=P0, base_worktree="base", candidate_worktree="candidate",
            regression_worktree="regression", summary_worktree="summary",
            repair_worktree="repair", run_id=run_id,
        ))
    finally:
        controller.close()
    return result, runner, workspace, tmp_path / "artifacts"


def prompts(runner, role):
    return [call for call in runner.call_details if call["role"] == role]


def test_full_v65_flow_resolves_assumption_and_repairs_derived_obligation(tmp_path):
    scripts = {
        RoleName.V5_IMPLEMENTATION_REVIEWER: [_assessment("O001")],
        RoleName.V5_REGRESSION_REVIEWER: [_assessment("P001")],
        V65Role.ASSUMPTION_EXTRACTOR: [{"assumptions": [CLAIM]}],
        V65Role.DISAMBIGUATION_REVIEWER: [
            d_batch(("Q001", "satisfied"), ("X001", "violated"), new_targets=[DERIVED], done=False),
            d_batch(("D001", "violated")),
        ],
        RoleName.V5_SUMMARY_REVIEWER: [REPAIR_D001],
        RoleName.REPAIR: ["repair complete"],
    }
    result, runner, workspace, artifacts = run(tmp_path, scripts)
    assert result.route == "repair" and result.p1_patch == b"p1"

    view = b"view:strip-all-tests"
    assert workspace.views == ["strip-all-tests"]
    assert sorted(workspace.resets) == sorted(
        [("candidate", view), ("regression", view), ("summary", view)])
    names = [event[0] for event in workspace.events]
    assert names.index("prepare") < names.index("remove-inputs") < names.index("prune")
    # Repair edits the real P0, never the view.
    assert ("apply", "repair", P0) in workspace.events

    i_prompt = prompts(runner, RoleName.V5_IMPLEMENTATION_REVIEWER)[0]["prompt"]
    assert "REVIEW VIEW\nThe candidate is shown as a behavioral view of P0." in i_prompt
    assert f"sha256:{sha256_bytes(view)}" in i_prompt and "keep the old reading" not in i_prompt

    x_call = prompts(runner, V65Role.ASSUMPTION_EXTRACTOR)[0]
    assert x_call["tools"] == () and x_call.get("session_id") is None
    assert P0.decode() in x_call["prompt"] and "keep the old reading" in x_call["prompt"]

    d_calls = prompts(runner, V65Role.DISAMBIGUATION_REVIEWER)
    assert len(d_calls) == 2 and d_calls[0]["worktree"] == "d-worktree"
    assert d_calls[0]["prompt"].startswith("ROLE: V5 D Reviewer (disambiguation)")
    assert "keep the old reading" not in d_calls[0]["prompt"]
    assert CLAIM["claim"] in d_calls[0]["prompt"]
    assert "D001" in d_calls[1]["prompt"]

    s_prompt = prompts(runner, RoleName.V5_SUMMARY_REVIEWER)[0]["prompt"]
    assert "The D report resolves the ledger questions" in s_prompt
    assert '"disambiguation_targets"' in s_prompt and "keep the old reading" not in s_prompt
    repair = prompts(runner, RoleName.REPAIR)[0]["prompt"]
    assert "D/D001" in repair and "Reject the bracketed form." in repair

    targets = json.loads((artifacts / "v65/disambiguation-targets.json").read_text())
    assert [item["id"] for item in targets["targets"]] == ["X001", "Q001"]
    assert json.loads((artifacts / "v65/assumptions.json").read_text())["status"] == "completed"
    record = json.loads((artifacts / "v65/review-view.json").read_text())
    assert record["level"] == "strip-all-tests" and record["view_sha256"] == sha256_bytes(view)
    assert json.loads((artifacts / "v65/barrier-audit.json").read_text())["clean"] is True
    report = json.loads((artifacts / "v5/exports/review-D.json").read_text())["content"]
    assert report["role"] == "disambiguation"
    assert [item["id"] for item in report["additional_targets"]] == ["D001"]


def test_question_and_assumption_findings_are_not_directly_repairable(tmp_path):
    scripts = {
        RoleName.V5_IMPLEMENTATION_REVIEWER: [_assessment("O001")],
        RoleName.V5_REGRESSION_REVIEWER: [_assessment("P001")],
        V65Role.ASSUMPTION_EXTRACTOR: [{"assumptions": [CLAIM]}],
        V65Role.DISAMBIGUATION_REVIEWER: [
            d_batch(("Q001", "satisfied"), ("X001", "violated"), new_targets=[DERIVED], done=False),
            d_batch(("D001", "violated")),
        ],
        RoleName.V5_SUMMARY_REVIEWER: [REPAIR_D001],
        RoleName.REPAIR: ["repair complete"],
    }
    run(tmp_path, scripts)
    store = V65StateStore(tmp_path / "artifacts/v5/workflow.sqlite")
    try:
        assert store.lane("v65", "D")["phase"] == "complete"
    finally:
        store.close()
    bad = {**REPAIR_D001, "decisions": [
        decision("A/Q001", "dismiss"), decision("D/X001", "repair"), decision("D/D001", "repair")],
        "repair_requests": [{"finding_refs": ["D/X001", "D/D001"], "change": "x",
                             "validation_record_refs": [], "preserve_refs": []}]}
    context = {"A/Q001": {"kind": "question", "blocking": True, "sources": [TASK_SOURCE]},
               "D/X001": {"kind": "assessment", "status": "violated", "importance": "advisory",
                          "sources": [TASK_SOURCE]},
               "D/D001": {"kind": "assessment", "status": "violated", "importance": "required",
                          "sources": [TASK_SOURCE]}}
    with pytest.raises(V5ContractError, match="may repair only a required violated assessment"):
        validate_summary_output(bad, {"A/Q001", "D/X001", "D/D001"}, decision_context=context)


def test_view_falls_back_when_it_breaks_a_buildable_p0(tmp_path):
    gate = FakeGate({"P0": "pass", "VIEW-strip-all-tests": "fail", "VIEW-strip-new-tests": "pass"})
    scripts = {
        RoleName.V5_IMPLEMENTATION_REVIEWER: [_assessment("O001")],
        RoleName.V5_REGRESSION_REVIEWER: [_assessment("P001")],
        V65Role.ASSUMPTION_EXTRACTOR: [{"assumptions": []}],
        V65Role.DISAMBIGUATION_REVIEWER: [d_batch(("Q001", "disputed"))],
        RoleName.V5_SUMMARY_REVIEWER: [{"summary": "ok", "decisions": [
            decision("A/Q001", "dismiss", "D/Q001"), decision("D/Q001", "dismiss")],
            "repair_requests": [], "new_questions": []}],
    }
    result, runner, workspace, artifacts = run(tmp_path, scripts, gate=gate)
    assert result.route == "accept"
    assert workspace.views == ["strip-all-tests", "strip-new-tests"]
    record = json.loads((artifacts / "v65/review-view.json").read_text())
    assert record["level"] == "strip-new-tests"
    assert [a.get("gate") for a in record["attempts"]] == ["fail", "pass"]
    i_prompt = prompts(runner, RoleName.V5_IMPLEMENTATION_REVIEWER)[0]["prompt"]
    assert "P0's changes to existing test files are shown." in i_prompt


def test_failed_extraction_leaves_d_with_ledger_questions(tmp_path):
    scripts = {
        RoleName.V5_IMPLEMENTATION_REVIEWER: [_assessment("O001")],
        RoleName.V5_REGRESSION_REVIEWER: [_assessment("P001")],
        V65Role.ASSUMPTION_EXTRACTOR: ["not json", {"assumptions": [
            {**CLAIM, "claim": CLAIM["claim"] + " because the proposal says so"}]}],
        V65Role.DISAMBIGUATION_REVIEWER: [d_batch(("Q001", "unresolved"))],
        RoleName.V5_SUMMARY_REVIEWER: [{"summary": "held", "decisions": [
            decision("A/Q001", "unresolved"), decision("D/Q001", "unresolved")],
            "repair_requests": [], "new_questions": []}],
    }
    result, runner, _, artifacts = run(tmp_path, scripts)
    record = json.loads((artifacts / "v65/assumptions.json").read_text())
    assert record["status"] == "failed" and "gives a reason" in record["error"]
    assert len(prompts(runner, V65Role.ASSUMPTION_EXTRACTOR)) == 2
    assert "VALIDATION ERROR" in prompts(runner, V65Role.ASSUMPTION_EXTRACTOR)[1]["prompt"]
    targets = json.loads((artifacts / "v65/disambiguation-targets.json").read_text())["targets"]
    assert [item["id"] for item in targets] == ["Q001"]
    # The blocking ledger question stays unresolved, so P0 is held as in V6.4.
    assert result.route == "hold"


def test_no_targets_means_no_d_lane(tmp_path):
    config = V65Config()
    ledger_without_questions = {**LEDGER, "open_questions": []}
    scripts = {
        RoleName.V5_IMPLEMENTATION_REVIEWER: [_assessment("O001")],
        RoleName.V5_REGRESSION_REVIEWER: [_assessment("P001")],
        V65Role.ASSUMPTION_EXTRACTOR: [{"assumptions": []}],
        RoleName.V5_SUMMARY_REVIEWER: [{"summary": "ok", "decisions": [],
                                        "repair_requests": [], "new_questions": []}],
    }
    runner = V65Runner(tmp_path, scripts)
    controller = V65Controller(role_runner=runner, workspace=ViewWorkspace(),
                               artifacts=ArtifactStore(tmp_path / "artifacts"), config=config,
                               build_gate=FakeGate({"P0": "pass"}))
    task = TaskInput("task", "Add behavior", "repo", "base", "title")
    try:
        asyncio.run(controller.prepare_ledger(task, base_worktree="base",
                                              imported_ledger=ledger_without_questions))
        result = asyncio.run(controller.run(
            task, candidate_patch=P0, base_worktree="base", candidate_worktree="candidate",
            repair_worktree="repair", run_id="v65-empty"))
        assert not controller.state_store.has_lane("v65-empty", "D")
    finally:
        controller.close()
    assert result.route == "accept" and set(result.reports) == {"I", "R"}
    assert V65Role.DISAMBIGUATION_REVIEWER not in runner.calls


def test_all_mechanisms_off_keeps_reviewers_on_raw_p0(tmp_path):
    config = V65Config(review_view_enabled=False, assumptions_enabled=False,
                       disambiguation_enabled=False)
    scripts = {
        RoleName.V5_IMPLEMENTATION_REVIEWER: [_assessment("O001")],
        RoleName.V5_REGRESSION_REVIEWER: [_assessment("P001")],
        RoleName.V5_SUMMARY_REVIEWER: [{"summary": "held", "decisions": [
            decision("A/Q001", "unresolved")], "repair_requests": [], "new_questions": []}],
    }
    result, runner, workspace, _ = run(tmp_path, scripts, config=config)
    assert workspace.views == [] and workspace.resets == []
    i_prompt = prompts(runner, RoleName.V5_IMPLEMENTATION_REVIEWER)[0]["prompt"]
    assert "REVIEW VIEW" not in i_prompt and f"sha256:{sha256_bytes(P0)}" in i_prompt
    assert result.route == "hold"


def test_d_lane_numbers_new_targets_in_its_own_namespace():
    state = {"targets": {"Q001": {}}, "enrolled": ["Q001"], "pending": ["Q001"],
             "effective": {}, "history": {}, "revisable": [], "optional": [],
             "reopened_target_ids": [], "observation_fingerprints": [],
             "next_target_number": 1, "no_progress": 0, "round": 1, "phase": "running"}
    payload = d_batch(("Q001", "violated"), new_targets=[DERIVED], done=False)
    new_state, assigned = V65StateStore._transition(state, payload, "D")
    assert assigned == ["D001"] and "D001" in new_state["targets"]
    assert new_state["pending"] == ["D001"] and "R001" not in json.dumps(new_state)


def test_config_rejects_assumptions_without_a_view():
    with pytest.raises(ValueError, match="require the review view"):
        V65Config(review_view_enabled=False).validate()
