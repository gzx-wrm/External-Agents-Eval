from __future__ import annotations

import asyncio
import json
import runpy
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from external_agents_eval.agents.mini_ft_review import FrozenMiniReviewRunner
from external_agents_eval.agents.mini_review_files import (
    PROTOCOL,
    SCOPE_PATH,
    adapt_prompt,
    collect_batch,
    controller_state,
)
from external_agents_eval.agents.mini_v63 import (
    MiniFileReviewEnvironment,
    MiniV63Agent,
    MiniV63RoleRunner,
)
from external_agents_eval.workflows.solve_review_repair.artifacts import ArtifactStore
from external_agents_eval.workflows.solve_review_repair.models import (
    RoleName,
    RoleRunResult,
    TaskInput,
)
from external_agents_eval.workflows.solve_review_repair.v5_contracts import V5ContractError
from external_agents_eval.workflows.solve_review_repair.v5_prompts import reviewer_prompt
from external_agents_eval.workflows.solve_review_repair.v6_controller import V6Controller
from external_agents_eval.workflows.solve_review_repair.v62_controller import V62Controller
from external_agents_eval.workflows.solve_review_repair.v63_config import V63Config


def narrative():
    return """
## Summary
The stated target was inspected against public sources.
## Review steps
Read the relevant entry point and compared its documented behavior.
## Limits and conflicting observations
Static inspection only; no test was executed in this synthetic fixture.
## Reproduction materials
No executable material was produced by this static check.
"""


def record(target, status="satisfied"):
    fields = {"target_id": target, "status": status, "expected": "Public behavior is preserved.",
              "observed": "The public entry point preserves the specified branch.",
              "reason": "The inspected branch agrees with the stated contract.", "code_refs": []}
    return (f"# Review note: {target}\n\n## Assessment\n\n```json\n"
            + json.dumps(fields) + "\n```\n" + narrative())


def scope(role="R", complete=True, new=None, revisit=None):
    fields = {"scope_complete": complete, "new_targets": new or [], "revisit_target_ids": revisit or []}
    return (f"# Review note: {role}_SCOPE\n\n## Scope\n\n```json\n"
            + json.dumps(fields) + "\n```\n" + narrative())


def state(root, pending=("P001",), accepted=(), optional=(), revisable=(), batch_size=3):
    eligible = set(pending) | set(optional) | set(revisable)
    return {
        "round": 1, "mode": "continue", "batch_size": batch_size,
        "registered_targets": [], "pending_target_ids": list(pending),
        "accepted": [{"target_id": t, "status": "satisfied"} for t in accepted],
        "revisable_target_ids": list(revisable),
        "review_notes": {"output_root": str(root), "skill_path": "skill.md",
                         "target_paths": [{"target_id": t, "path": f"notes/{t}.md"} for t in sorted(eligible)]},
        "catalog_path": "catalog.json", "errors": [], "time_remaining_sec": 60,
        "rounds_remaining": 64,
    }


def original_prompt(s, role="R"):
    return reviewer_prompt(
        task={"title": "Task", "description": "Add behavior"}, repository={},
        ledger={"obligations": [], "preservation_obligations": [], "open_questions": []},
        state=s, role=role, skill_path="skill.md", output_root=s["review_notes"]["output_root"],
        initial=True,
    )


def new_target():
    return {"statement": "Preserve the public boundary behavior.", "importance": "required",
            "sources": [{"kind": "task", "field": "description", "quote": "Add behavior"}]}


class LocalBridge:
    def exec_sync(self, command, *, cwd=None, **kwargs):
        result = subprocess.run(["bash", "-lc", command], cwd=cwd, text=True,
                                capture_output=True, timeout=20, check=False)
        return SimpleNamespace(return_code=result.returncode, stdout=result.stdout, stderr=result.stderr)

    async def upload_file(self, source, target):
        # The native runner's remote conversation copy is not needed by this
        # in-process bridge; the authoritative local conversation remains real.
        assert Path(source).is_file()


def write_files(files):
    commands = []
    for relative, text in files.items():
        path = '"$REVIEW_OUTPUT_ROOT/' + relative + '"'
        parent = '"$REVIEW_OUTPUT_ROOT/' + str(Path(relative).parent) + '"'
        commands.extend([
            "mkdir -p " + parent,
            "printf '%s' " + shlex.quote(text) + " > " + path + ".tmp",
            "mv " + path + ".tmp " + path,
        ])
    return "\n".join(commands)


@pytest.mark.parametrize("status", ["satisfied", "violated", "unresolved", "disputed"])
def test_file_status_does_not_determine_the_summary_route(tmp_path, status):
    files = {"notes/P001.md": record("P001", status), SCOPE_PATH: scope()}
    batch, _ = collect_batch(files, role="R", state=state(tmp_path), archive=tmp_path / "archive")
    assert batch["done"] is True
    assert batch["assessments"][0]["status"] == status
    assert batch["assessments"][0]["review_note_path"] == "notes/P001.md"
    assert "route" not in batch


def test_r_scope_is_required_even_if_original_ledger_has_no_p_targets(tmp_path):
    s = state(tmp_path, pending=())
    assert collect_batch({}, role="R", state=s, archive=tmp_path / "archive")[0] is None
    assert collect_batch({SCOPE_PATH: scope(complete=False)}, role="R", state=s,
                         archive=tmp_path / "archive")[0] is None
    assert collect_batch({SCOPE_PATH: scope()}, role="R", state=s,
                         archive=tmp_path / "archive")[0]["done"] is True


def test_new_targets_are_registered_before_completion_and_not_assigned_by_adapter(tmp_path):
    files = {"notes/P001.md": record("P001"), SCOPE_PATH: scope(new=[new_target()])}
    batch, _ = collect_batch(files, role="R", state=state(tmp_path), archive=tmp_path / "archive")
    assert batch["done"] is False
    assert batch["new_targets"] == [new_target()]
    assert all("id" not in t for t in batch["new_targets"])
    files["notes/R001.md"] = record("R001")
    with pytest.raises(V5ContractError, match="Unexpected record"):
        collect_batch(files, role="R", state=state(tmp_path), archive=tmp_path / "archive")


def test_reopening_and_batch_size_keep_the_original_v6_rules(tmp_path):
    s = state(tmp_path, pending=(), accepted=("P001",))
    batch, _ = collect_batch({SCOPE_PATH: scope(revisit=["P001"])}, role="R", state=s,
                             archive=tmp_path / "archive")
    assert batch["revisit_target_ids"] == ["P001"] and not batch["done"]
    files = {"notes/P001.md": record("P001"), SCOPE_PATH: scope(new=[new_target()])}
    with pytest.raises(V5ContractError, match="limit is 1"):
        collect_batch(files, role="R", state=state(tmp_path, batch_size=1), archive=tmp_path / "archive")


def test_file_existence_cannot_replace_a_valid_record(tmp_path):
    files = {"notes/P001.md": "P001.md\n", SCOPE_PATH: scope()}
    with pytest.raises(V5ContractError, match="title does not match"):
        collect_batch(files, role="R", state=state(tmp_path), archive=tmp_path / "archive")
    files["notes/P001.md"] = record("P002")
    with pytest.raises(V5ContractError, match="title does not match"):
        collect_batch(files, role="R", state=state(tmp_path), archive=tmp_path / "archive")


def test_native_bash_ignores_markers_and_temporary_files_then_repairs_in_place(tmp_path):
    root = tmp_path / "current"
    root.mkdir()
    env = MiniFileReviewEnvironment(LocalBridge(), str(tmp_path), role="R", state=state(root),
                                    archive=tmp_path / "host/records")
    first = env.execute({"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", "tool_call_id": "one"})
    assert env.batch is None and "Found 0 record" in first["output"]
    env.execute({"command": 'mkdir -p "$REVIEW_OUTPUT_ROOT/notes"\nprintf broken > "$REVIEW_OUTPUT_ROOT/notes/P001.md.tmp"'})
    assert env.batch is None
    bad = env.execute({"command": write_files({"notes/P001.md": "invalid", SCOPE_PATH: scope()})})
    assert env.batch is None and "Fix current files" in bad["output"]
    env.execute({"command": write_files({"notes/P001.md": record("P001")})})
    assert env.batch["done"] is True
    assert (root / "notes/P001.md").read_text() == record("P001")


def test_old_or_symlinked_notes_cannot_satisfy_current_delivery(tmp_path):
    root = tmp_path / "current"
    (root / "notes").mkdir(parents=True)
    old = tmp_path / "old.md"
    old.write_text(record("P001"))
    (root / "notes/P001.md").symlink_to(old)
    env = MiniFileReviewEnvironment(LocalBridge(), str(tmp_path), role="R", state=state(root),
                                    archive=tmp_path / "host/records")
    output = env.execute({"command": "true"})
    assert env.batch is None and "Not a regular current-attempt record" in output["output"]


def install_stub_model(monkeypatch, commands_for):
    pytest.importorskip("minisweagent")
    import minisweagent.models.litellm_model as models

    class FakeModel:
        def __init__(self, **kwargs):
            self.commands = None
            self.calls = 0

        def query(self, messages, **kwargs):
            if self.commands is None:
                prompt = messages[-1]["content"]
                self.commands = iter(commands_for(prompt))
            self.calls += 1
            command = next(self.commands)
            call_id = f"native-{self.calls}"
            return {"role": "assistant", "content": "",
                    "tool_calls": [{"id": call_id, "type": "function",
                                    "function": {"name": "bash", "arguments": json.dumps({"command": command})}}],
                    "extra": {"actions": [{"command": command, "tool_call_id": call_id}], "cost": 0}}

        def format_message(self, **kwargs):
            return kwargs

        def format_observation_messages(self, message, outputs, *args):
            return [{"role": "tool", "tool_call_id": a["tool_call_id"], "content": o["output"]}
                    for a, o in zip(message["extra"]["actions"], outputs)]

        def get_template_vars(self):
            return {}

        def serialize(self):
            return {}

    monkeypatch.setattr(models, "LitellmModel", FakeModel)


def make_runner(root):
    return MiniV63RoleRunner(environment=LocalBridge(), logs_root=root, model="deepseek-flash",
                            provider="deepseek", agent_version="2.4.6", env={}, redact=lambda x: x)


def test_real_mini_loop_recovers_files_without_final_json_and_preserves_last_observation(tmp_path, monkeypatch):
    root = tmp_path / "current"
    root.mkdir()
    commands = [write_files({"notes/P001.md": "broken", SCOPE_PATH: scope()}),
                write_files({"notes/P001.md": record("P001")}) + "\nprintf 'LAST_REAL_OUTPUT'\n"]
    install_stub_model(monkeypatch, lambda prompt: commands)
    runner = make_runner(tmp_path / "host")
    kwargs = {
        "role": RoleName.V5_REGRESSION_REVIEWER, "prompt": original_prompt(state(root)),
        "worktree": str(tmp_path), "tools": ("bash",), "timeout_sec": 30,
        "artifact_prefix": "R/round-1", "session_id": "R-session", "invocation_id": "test.R.1",
    }
    result = asyncio.run(runner.run_role(**kwargs))
    assert result.exit_code == 0
    assert json.loads(result.final_assistant_text)["done"] is True
    native = json.loads(result.transcript_path.read_text())
    assert native["messages"][-1]["extra"]["review_delivery"] == PROTOCOL
    assert any(m["role"] == "tool" and "LAST_REAL_OUTPUT" in m.get("content", "")
               for m in native["messages"])
    trace = json.loads(result.trajectory_path.read_text())
    assert not trace["executed_but_not_returned"]
    assert len([s for s in trace["steps"] if s.get("tool_calls")]) == 2
    assert (tmp_path / "host/R/round-1/controller-prompt.md").read_text() == kwargs["prompt"]
    assert "V6.3 FILE DELIVERY" in (tmp_path / "host/R/round-1/prompt.md").read_text()
    # Re-reading a completed invocation must reuse it, not run the stub twice.
    assert asyncio.run(runner.run_role(**kwargs)).final_assistant_text == result.final_assistant_text


def test_frozen_controller_preserves_each_reviewer_conversation_across_rounds(tmp_path, monkeypatch):
    requests = []

    def commands_for(prompt):
        current = controller_state(prompt)
        target = current["pending_target_ids"][0]
        lane = "I" if target.startswith("O") else "R"
        files = {
            f"notes/{target}.md": record(target),
            SCOPE_PATH: scope(role=lane, complete=current["round"] == 2),
        }
        return [write_files(files) + f"\nprintf 'OBSERVATION_{lane}_{current['round']}'\n"]

    install_stub_model(monkeypatch, commands_for)
    from minisweagent.models.litellm_model import LitellmModel

    original_query = LitellmModel.query

    def capture_query(self, messages, **kwargs):
        requests.append(json.loads(json.dumps(messages)))
        return original_query(self, messages, **kwargs)

    monkeypatch.setattr(LitellmModel, "query", capture_query)
    runner = FrozenMiniReviewRunner(
        environment=LocalBridge(), logs_root=tmp_path / "host", model="deepseek-flash",
        provider="deepseek", agent_version="2.4.6", env={}, redact=lambda text: text,
    )
    controller = V6Controller(
        role_runner=runner, workspace=object(), artifacts=ArtifactStore(tmp_path / "workflow"),
        config=V63Config(),
    )
    sessions = {}
    try:
        for lane, role, targets in [
            ("I", RoleName.V5_IMPLEMENTATION_REVIEWER, ("O001", "O002")),
            ("R", RoleName.V5_REGRESSION_REVIEWER, ("P001", "P002")),
        ]:
            results, invocations = [], []
            first_request = len(requests)
            for round_number in (1, 2):
                notes = tmp_path / f"{lane}-round-{round_number}"
                notes.mkdir()
                current = state(
                    notes, pending=targets if round_number == 1 else targets[1:],
                    accepted=() if round_number == 1 else targets[:1],
                )
                current["round"] = round_number
                prompt = reviewer_prompt(
                    task={"title": "Task", "description": f"PUBLIC_ISSUE_{lane}"},
                    repository={}, ledger={}, state=current, role=lane, skill_path="skill.md",
                    output_root=str(notes), initial=round_number == 1,
                )
                kwargs = {
                    "role": role, "prompt": prompt, "worktree": str(tmp_path),
                    "prefix": f"{lane}/round-{round_number}", "session_role": lane,
                    "invocation_id": f"session-regression.{lane}.{round_number}", "timeout_sec": 30,
                }
                invocations.append(kwargs)
                results.append(asyncio.run(controller._run_role(**kwargs)))
            first, second = results
            assert first.exit_code == second.exit_code == 0
            assert first.pre_session_sha256 is None
            assert second.pre_session_sha256 == first.post_session_sha256
            assert second.session_file == first.session_file
            assert second.remote_session_file == first.remote_session_file
            second_request = requests[first_request + 1]
            assert any(f"PUBLIC_ISSUE_{lane}" in m.get("content", "") for m in second_request)
            assert any(m["role"] == "tool" and f"OBSERVATION_{lane}_1" in m.get("content", "")
                       for m in second_request)
            sessions[lane] = second.session_file
            count = len(requests)
            replay = asyncio.run(controller._run_role(**invocations[-1]))
            assert replay.post_session_sha256 == second.post_session_sha256
            assert len(requests) == count  # Completed round is replayed without another model query.
        assert sessions["I"] != sessions["R"]
        assert not any("OBSERVATION_I_1" in m.get("content", "") for m in requests[2])
    finally:
        controller.close()


def test_tool_activity_without_records_exhausts_mini_budget_without_committing_a_batch(tmp_path, monkeypatch):
    root = tmp_path / "current"
    root.mkdir()
    install_stub_model(monkeypatch, lambda prompt: ["echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"] * 5)
    runner = make_runner(tmp_path / "host")
    factory = runner._make_mini_agent

    def bounded(**kwargs):
        agent = factory(**kwargs)
        agent.config.step_limit = 3
        return agent

    monkeypatch.setattr(runner, "_make_mini_agent", bounded)
    result = asyncio.run(runner.run_role(
        role=RoleName.V5_REGRESSION_REVIEWER, prompt=original_prompt(state(root)),
        worktree=str(tmp_path), tools=("bash",), timeout_sec=30, artifact_prefix="R/stalled",
    ))
    assert result.exit_code != 0
    assert not result.final_assistant_text
    receipt = json.loads((tmp_path / "host/R/stalled/file-review.json").read_text())
    assert receipt["batch"] is None


def test_dynamic_r_files_work_with_the_unchanged_controller_and_s_decision(tmp_path, monkeypatch):
    def commands_for(prompt):
        s = controller_state(prompt)
        role = "I" if prompt.startswith("ROLE: V5 I Reviewer") else "R"
        target = s["pending_target_ids"][0]
        added = [new_target()] if role == "R" and s["round"] == 1 else []
        return [write_files({f"notes/{target}.md": record(target), SCOPE_PATH: scope(role, new=added)})]

    install_stub_model(monkeypatch, commands_for)
    native = make_runner(tmp_path / "native")

    class Runner:
        def __init__(self):
            self.calls = []

        async def run_role(self, *, role, prompt, worktree, tools, timeout_sec,
                           artifact_prefix, session_id=None, invocation_id=None, **kwargs):
            self.calls.append(role)
            if role in {RoleName.V5_IMPLEMENTATION_REVIEWER, RoleName.V5_REGRESSION_REVIEWER}:
                return await native.run_role(role=role, prompt=prompt, worktree=worktree, tools=tools,
                                             timeout_sec=timeout_sec, artifact_prefix=artifact_prefix,
                                             session_id=session_id, invocation_id=invocation_id)
            assert role == RoleName.V5_SUMMARY_REVIEWER
            path = tmp_path / "summary.json"
            path.write_text("{}")
            return RoleRunResult(exit_code=0, final_assistant_text=json.dumps({
                "summary": "All submitted judgments are satisfied.", "decisions": [],
                "repair_requests": [], "new_questions": [],
            }), transcript_path=path, trajectory_path=path, started_at="now", finished_at="now")

    class Workspace:
        async def capture_patch(self, **kwargs):
            return b"p0"

    source = {"kind": "task", "field": "description", "quote": "Add behavior"}
    ledger = {"summary": "Add behavior", "obligations": [
        {"id": "O001", "content": {"statement": "Add behavior", "importance": "required", "sources": [source]}},
    ], "preservation_obligations": [
        {"id": "P001", "content": {"statement": "Preserve behavior", "importance": "required", "sources": [source]}},
    ], "open_questions": []}
    runner = Runner()
    controller = V62Controller(ledger_client=None, role_runner=runner, workspace=Workspace(),
                               artifacts=ArtifactStore(tmp_path / "workflow"), config=V63Config())
    task = TaskInput("task", "Add behavior", "repo", "base", "Task")
    asyncio.run(controller.prepare_ledger(task, base_worktree=str(tmp_path), imported_ledger=ledger))
    result = asyncio.run(controller.run(task, candidate_patch=b"p0", base_worktree=str(tmp_path),
                                        candidate_worktree=str(tmp_path), regression_worktree=str(tmp_path)))
    assert result.route == "accept" and result.p1_patch == b"p0"
    assert result.reports["R"]["completion"] == "complete"
    assert not result.reports["R"]["pending_target_ids"]
    assert {a["content"]["target_id"] for a in result.reports["R"]["assessments"]} == {"P001", "R001"}
    assert runner.calls.count(RoleName.V5_REGRESSION_REVIEWER) == 2
    assert RoleName.JSON_CORRECTOR not in runner.calls
    archived_scope = list((tmp_path / "workflow/v5/bundles").glob("*/materials/logs/review-scope.md"))
    assert len(archived_scope) == 3
    controller.close()


def test_v63_has_separate_version_and_only_adapts_the_review_delivery(tmp_path):
    from external_agents_eval.agents.mini_v62 import MiniV62Agent

    s = state(tmp_path)
    prompt = original_prompt(s)
    adapted = adapt_prompt(prompt)
    assert controller_state(adapted) == s
    assert "UNCHANGED V6 FIELD DEFINITIONS" in adapted
    assert "TargetDraft" in adapted
    assert MiniV63Agent.workflow_version == "6.3"
    assert MiniV62Agent.workflow_version == "6.2"
    original = V63Config()
    assert original.batch_size == 3 and original.max_rounds == 64
    assert original.prompt_version == "v5-5" and original.controller_version == "v5-4"


@pytest.mark.parametrize("workflow", ["v6.2", "v6.3"])
def test_verified_entry_point_selects_workflow_without_changing_the_default(tmp_path, monkeypatch, workflow):
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / "verified_random100_seed20261026.json").write_text(json.dumps({"task_ids": ["task"]}))
    (manifests / "public_tasks.jsonl").write_text(json.dumps({"instance_id": "task"}) + "\n")
    script = Path(__file__).resolve().parents[1] / "scripts/swe_verified_v62/run.py"
    args = [str(script), "plan", "--root", str(tmp_path)]
    if workflow == "v6.3":
        args += ["--workflow", workflow]
    monkeypatch.setattr(sys, "argv", args)
    runpy.run_path(str(script), run_name="__main__")
    plan = json.loads((tmp_path / "preflight/plan.json").read_text())
    assert plan["workflow"] == workflow
    assert plan["model_calls"] == 0
