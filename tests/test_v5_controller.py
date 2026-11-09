from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

import pytest

from external_agents_eval.workflows.solve_review_repair.artifacts import (
    ArtifactStore,
    sha256_bytes,
    sha256_json,
)
from external_agents_eval.workflows.solve_review_repair.models import (
    IndeterminateInvocationError,
    RoleName,
    RoleRunResult,
    TaskInput,
)
from external_agents_eval.workflows.solve_review_repair.v5_config import V5Config
from external_agents_eval.workflows.solve_review_repair.v5_contracts import (
    V5ContractError,
)
from external_agents_eval.workflows.solve_review_repair.v5_controller import V5Controller
from external_agents_eval.workflows.solve_review_repair.v6_fork import V6StageSource

NOTE_TEMPLATE = """# Review note: {target}

## Summary

The target was checked against the supplied source.

## Review steps

1. **Check**

The fake reviewer inspected the target.

## Limits and conflicting observations

The synthetic check has no additional limitation.

## Reproduction materials

No synthetic materials were needed.
"""


class FakeRunner:
    def __init__(self, root: Path):
        self.root = root
        self.calls: list[RoleName] = []
        self.call_details: list[dict] = []

    async def run_role(self, *, role, prompt, worktree, tools, timeout_sec, artifact_prefix, **kwargs):
        self.calls.append(role)
        self.call_details.append(
            {
                "role": role,
                "prompt": prompt,
                "worktree": worktree,
                "tools": tools,
                "timeout_sec": timeout_sec,
                "artifact_prefix": artifact_prefix,
                **kwargs,
            }
        )
        trajectory = self.root / (artifact_prefix.replace("/", "_") + ".json")
        trajectory.write_text("{}", encoding="utf-8")
        if role == RoleName.V5_SPECIFICATION_REVIEWER:
            payload = {
                "summary": "The task adds one behavior.",
                "obligations": [{"statement": "The new behavior works.", "importance": "required", "sources": [{"kind": "task", "field": "description", "quote": "Add behavior"}]}],
                "preservation_obligations": [{"statement": "The old behavior remains available.", "importance": "required", "sources": [{"kind": "repository", "path": "src/a.py", "line_start": 1, "line_end": 1}]}],
                "open_questions": [],
            }
        elif role in {RoleName.V5_IMPLEMENTATION_REVIEWER, RoleName.V5_REGRESSION_REVIEWER}:
            target = "O001" if role == RoleName.V5_IMPLEMENTATION_REVIEWER else "P001"
            output_root = next(line.split(": ", 1)[1] for line in prompt.splitlines() if line.startswith("OUTPUT ROOT:"))
            root = Path(output_root)
            (root / "notes").mkdir(parents=True, exist_ok=True)
            (root / f"notes/{target}.md").write_text(NOTE_TEMPLATE.format(target=target), encoding="utf-8")
            payload = {
                "new_targets": [],
                "assessments": [{"target_id": target, "status": "satisfied", "expected": "The behavior works.", "observed": "The behavior works.", "reason": "The synthetic observation agrees.", "code_refs": [], "review_note_path": f"notes/{target}.md"}],
                "revisit_target_ids": [], "done": True,
            }
        elif role == RoleName.V5_SUMMARY_REVIEWER:
            payload = {"summary": "No repair is required.", "decisions": [], "repair_requests": [], "new_questions": []}
        else:
            payload = {}
        return RoleRunResult(
            exit_code=0,
            final_assistant_text=json.dumps(payload),
            transcript_path=trajectory,
            trajectory_path=trajectory,
            started_at="now",
            finished_at="now",
        )


class FakeWorkspace:
    def __init__(self):
        self.patch = b""

    async def apply_patch(self, *, worktree, patch):
        self.patch = patch

    async def capture_patch(self, *, worktree, base_commit):
        return self.patch


def test_v5_controller_runs_a_ir_s_and_persists_freeze(tmp_path):
    runner = FakeRunner(tmp_path)
    controller = V5Controller(
        role_runner=runner,
        workspace=FakeWorkspace(),
        artifacts=ArtifactStore(tmp_path / "artifacts"),
    )
    result = asyncio.run(controller.run(
        TaskInput("task", "Add behavior", "repo", "base", "title"),
        candidate_patch=b"p0",
        base_worktree="base",
        candidate_worktree="candidate",
    ))
    assert result.route == "accept"
    assert result.p1_patch == b"p0"
    assert result.freeze_id
    assert runner.calls == [
        RoleName.V5_SPECIFICATION_REVIEWER,
        RoleName.V5_IMPLEMENTATION_REVIEWER,
        RoleName.V5_REGRESSION_REVIEWER,
        RoleName.V5_SUMMARY_REVIEWER,
    ]
    assert (tmp_path / "artifacts/v5/workflow.sqlite").is_file()
    assert (tmp_path / "artifacts/v5/ledger.json").is_file()
    assert (tmp_path / "artifacts/v5/summary.json").is_file()
    assert (tmp_path / "artifacts/v5/decision.json").is_file()
    exports = tmp_path / "artifacts/v5/exports"
    assert {path.name for path in exports.iterdir()} == {
        "ledger.json",
        "review-I.json",
        "review-R.json",
        "summary.json",
    }
    summary_export = json.loads((exports / "summary.json").read_text())
    assert summary_export["schema_version"] == "5.0"
    assert summary_export["bindings"]["candidate_sha256"]
    assert summary_export["bindings"]["summary_freeze_id"] == result.freeze_id
    assert summary_export["content_sha256"]
    spec_call = runner.call_details[0]
    summary_call = runner.call_details[-1]
    assert '"diff_path"' not in spec_call["prompt"]
    assert "Do not use the network" in spec_call["prompt"]
    assert "bash" not in spec_call["tools"]
    assert "bash" not in summary_call["tools"]


def test_v6_fork_reuses_one_reviewer_then_retries_summary(tmp_path):
    task = TaskInput("task", "Add behavior", "repo", "base", "title")
    original_root = tmp_path / "original"
    original_runner = FakeRunner(tmp_path)
    original = V5Controller(
        role_runner=original_runner,
        workspace=FakeWorkspace(),
        artifacts=ArtifactStore(original_root),
    )
    try:
        original_result = asyncio.run(original.run(
            task,
            candidate_patch=b"p0",
            base_worktree="base",
            candidate_worktree="candidate",
            export_v6_stages=True,
        ))
    finally:
        original.close()
    source = V6StageSource(
        original_root,
        task_id="task",
        base_commit="base",
        baseline_sha256=sha256_bytes(b""),
    )
    ledger_sha = sha256_json(original_result.ledger)
    p0_sha = sha256_bytes(b"p0")
    reused_r = source.review("R", ledger_sha256=ledger_sha, p0_sha256=p0_sha)
    reused_i = source.review("I", ledger_sha256=ledger_sha, p0_sha256=p0_sha)
    reused_s = source.summary(
        ledger_sha256=ledger_sha,
        p0_sha256=p0_sha,
        review_sha256={
            role: sha256_json(value["report"])
            for role, value in (("I", reused_i), ("R", reused_r))
        },
    )

    retry_root = tmp_path / "retry-r"
    (retry_root / "v5").mkdir(parents=True)
    for name in ("ledger.json", "source-bindings.json"):
        shutil.copy2(original_root / "v5" / name, retry_root / "v5" / name)
    retry_runner = FakeRunner(tmp_path)
    retry = V5Controller(
        role_runner=retry_runner,
        workspace=FakeWorkspace(),
        artifacts=ArtifactStore(retry_root),
    )
    try:
        result = asyncio.run(retry.run(
            task,
            candidate_patch=b"p0",
            base_worktree="base",
            candidate_worktree="candidate",
            reused_reviews={"R": reused_r},
            export_v6_stages=True,
        ))
    finally:
        retry.close()
    assert result.route == "accept"
    assert RoleName.V5_IMPLEMENTATION_REVIEWER in retry_runner.calls
    assert RoleName.V5_REGRESSION_REVIEWER not in retry_runner.calls
    assert RoleName.V5_SUMMARY_REVIEWER in retry_runner.calls
    assert (retry_root / "v6/stages/R.json").is_file()
    assert (retry_root / "v5/bundles").is_dir()

    retry_s_root = tmp_path / "retry-s"
    (retry_s_root / "v5").mkdir(parents=True)
    for name in ("ledger.json", "source-bindings.json"):
        shutil.copy2(original_root / "v5" / name, retry_s_root / "v5" / name)
    retry_s_runner = FakeRunner(tmp_path)
    retry_s = V5Controller(
        role_runner=retry_s_runner,
        workspace=FakeWorkspace(),
        artifacts=ArtifactStore(retry_s_root),
    )
    try:
        result = asyncio.run(retry_s.run(
            task,
            candidate_patch=b"p0",
            base_worktree="base",
            candidate_worktree="candidate",
            reused_reviews={"I": reused_i, "R": reused_r},
            reused_summary=reused_s,
            export_v6_stages=True,
        ))
    finally:
        retry_s.close()
    assert result.route == "accept"
    assert retry_s_runner.calls == []


def test_v6_fork_rejects_review_for_other_p0(tmp_path):
    task = TaskInput("task", "Add behavior", "repo", "base", "title")
    root = tmp_path / "source"
    controller = V5Controller(
        role_runner=FakeRunner(tmp_path),
        workspace=FakeWorkspace(),
        artifacts=ArtifactStore(root),
    )
    try:
        result = asyncio.run(controller.run(
            task,
            candidate_patch=b"p0",
            base_worktree="base",
            candidate_worktree="candidate",
            export_v6_stages=True,
        ))
    finally:
        controller.close()
    source = V6StageSource(
        root, task_id="task", base_commit="base", baseline_sha256=sha256_bytes(b""),
    )
    with pytest.raises(ValueError, match="selected A/P0"):
        source.review(
            "I", ledger_sha256=sha256_json(result.ledger), p0_sha256=sha256_bytes(b"other"),
        )


def test_v5_managed_delivery_is_exactly_bound_and_forwarded(tmp_path):
    runner = FakeRunner(tmp_path)
    controller = V5Controller(
        role_runner=runner,
        workspace=FakeWorkspace(),
        artifacts=ArtifactStore(tmp_path / "artifacts"),
    )
    delivery = {
        "kind": "task",
        "field": "description",
        "quote": "Do not create a commit.",
    }
    task = TaskInput(
        "task",
        "Add behavior. Do not create a commit.",
        "repo",
        "base",
        "title",
        (delivery,),
    )

    result = asyncio.run(
        controller.run(
            task,
            candidate_patch=b"p0",
            base_worktree="base",
            candidate_worktree="candidate",
            run_id="managed-delivery",
        )
    )

    assert result.route == "accept"
    for role in (
        RoleName.V5_SPECIFICATION_REVIEWER,
        RoleName.V5_IMPLEMENTATION_REVIEWER,
        RoleName.V5_REGRESSION_REVIEWER,
        RoleName.V5_SUMMARY_REVIEWER,
    ):
        call = next(item for item in runner.call_details if item["role"] == role)
        assert "Do not create a commit." in call["prompt"]
    summary_call = next(
        item
        for item in runner.call_details
        if item["role"] == RoleName.V5_SUMMARY_REVIEWER
    )
    for field in (
        "allowed_supporting_refs",
        "allowed_validation_record_refs",
        "allowed_preserve_refs",
        "allowed_sources",
    ):
        assert f'"{field}"' in summary_call["prompt"]

    invalid_controller = V5Controller(
        role_runner=FakeRunner(tmp_path),
        workspace=FakeWorkspace(),
        artifacts=ArtifactStore(tmp_path / "invalid-artifacts"),
    )
    invalid = TaskInput(
        "task",
        "Add behavior.",
        "repo",
        "base",
        "title",
        (delivery,),
    )
    with pytest.raises(V5ContractError, match="not contiguous text"):
        asyncio.run(
            invalid_controller.run(
                invalid,
                candidate_patch=b"p0",
                base_worktree="base",
                candidate_worktree="candidate",
                run_id="invalid-managed-delivery",
            )
        )


class ScriptedRunner(FakeRunner):
    def __init__(self, root: Path, scripts: dict[RoleName, list[dict | str]]):
        super().__init__(root)
        self.scripts = {role: list(items) for role, items in scripts.items()}

    async def run_role(
        self,
        *,
        role,
        prompt,
        worktree,
        tools,
        timeout_sec,
        artifact_prefix,
        session_id=None,
        session_dir=None,
        session_file=None,
        no_compaction=False,
        invocation_id=None,
        **kwargs,
    ):
        self.calls.append(role)
        self.call_details.append(
            {
                "role": role,
                "prompt": prompt,
                "worktree": worktree,
                "tools": tools,
                "timeout_sec": timeout_sec,
                "artifact_prefix": artifact_prefix,
                "session_id": session_id,
                "session_dir": session_dir,
                "session_file": session_file,
                "no_compaction": no_compaction,
                "invocation_id": invocation_id,
                **kwargs,
            }
        )
        trajectory = self.root / (artifact_prefix.replace("/", "_") + ".json")
        trajectory.parent.mkdir(parents=True, exist_ok=True)
        trajectory.write_text("{}", encoding="utf-8")
        value = self.scripts[role].pop(0)
        if isinstance(value, BaseException):
            raise value
        if role in {
            RoleName.V5_IMPLEMENTATION_REVIEWER,
            RoleName.V5_REGRESSION_REVIEWER,
        }:
            output_root = next(
                line.split(": ", 1)[1]
                for line in prompt.splitlines()
                if line.startswith("OUTPUT ROOT:")
            )
            root = Path(output_root)
            if isinstance(value, dict):
                for assessment in value.get("assessments", []):
                    target = assessment["target_id"]
                    path = root / assessment["review_note_path"]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(NOTE_TEMPLATE.format(target=target), encoding="utf-8")
            elif value == "MALFORMED_WITH_NOTE":
                path = root / "notes/O001.md"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(NOTE_TEMPLATE.format(target="O001"), encoding="utf-8")
                value = "{not json"
        raw = value if isinstance(value, str) else json.dumps(value)
        return RoleRunResult(
            exit_code=0,
            final_assistant_text=raw,
            transcript_path=trajectory,
            trajectory_path=trajectory,
            started_at="now",
            finished_at="now",
            invocation_id=invocation_id,
        )


def _spec():
    return {
        "summary": "One required behavior and one preservation behavior.",
        "obligations": [
            {
                "statement": "The new behavior works.",
                "importance": "required",
                "sources": [
                    {
                        "kind": "task",
                        "field": "description",
                        "quote": "Add behavior",
                    }
                ],
            }
        ],
        "preservation_obligations": [
            {
                "statement": "The old behavior remains available.",
                "importance": "required",
                "sources": [
                    {
                        "kind": "repository",
                        "path": "src/a.py",
                        "line_start": 1,
                        "line_end": 1,
                    }
                ],
            }
        ],
        "open_questions": [],
    }


def _assessment(target: str, *, status: str = "satisfied", done: bool = True):
    return {
        "new_targets": [],
        "assessments": [
            {
                "target_id": target,
                "status": status,
                "expected": "The target behavior holds.",
                "observed": "The synthetic observation was recorded.",
                "reason": "The synthetic evidence supports this judgment.",
                "code_refs": [],
                "review_note_path": f"notes/{target}.md",
            }
        ],
        "revisit_target_ids": [],
        "done": done,
    }


def test_v5_multi_round_reviewer_keeps_one_session_and_registers_target(tmp_path):
    new_target = {
        "statement": "An adjacent behavior stays stable.",
        "importance": "advisory",
        "sources": [
            {"kind": "task", "field": "description", "quote": "Add behavior"}
        ],
    }
    second_i = _assessment("O001", done=False)
    second_i["assessments"].append(_assessment("I001")["assessments"][0])
    second_i["done"] = True
    runner = ScriptedRunner(
        tmp_path,
        {
            RoleName.V5_SPECIFICATION_REVIEWER: [_spec()],
            RoleName.V5_IMPLEMENTATION_REVIEWER: [
                {
                    "new_targets": [new_target],
                    "assessments": [],
                    "revisit_target_ids": [],
                    "done": False,
                },
                second_i,
            ],
            RoleName.V5_REGRESSION_REVIEWER: [_assessment("P001")],
            RoleName.V5_SUMMARY_REVIEWER: [
                {
                    "summary": "All required targets are satisfied.",
                    "decisions": [],
                    "repair_requests": [],
                    "new_questions": [],
                }
            ],
        },
    )
    controller = V5Controller(
        role_runner=runner,
        workspace=FakeWorkspace(),
        artifacts=ArtifactStore(tmp_path / "artifacts"),
    )
    result = asyncio.run(
        controller.run(
            TaskInput("task", "Add behavior", "repo", "base", "title"),
            candidate_patch=b"p0",
            base_worktree="base",
            candidate_worktree="candidate-i",
            regression_worktree="candidate-r",
            run_id="multi-round",
        )
    )
    i_calls = [
        item
        for item in runner.call_details
        if item["role"] == RoleName.V5_IMPLEMENTATION_REVIEWER
    ]
    assert len(i_calls) == 2
    assert i_calls[0]["session_id"] == i_calls[1]["session_id"]
    assert i_calls[0]["session_dir"] == i_calls[1]["session_dir"]
    assert i_calls[0]["invocation_id"] != i_calls[1]["invocation_id"]
    assert all(item["no_compaction"] for item in i_calls)
    assert "INITIAL REVIEW INPUT" in i_calls[0]["prompt"]
    assert "INITIAL REVIEW INPUT" not in i_calls[1]["prompt"]
    assert "Continue the same review" in i_calls[1]["prompt"]
    assert result.route == "accept"
    assert result.reports["I"]["additional_targets"][0]["id"] == "I001"


def test_v5_a_source_binding_failure_retries_original_a_session(tmp_path):
    invalid = _spec()
    invalid["obligations"][0]["sources"][0]["quote"] = "absent task quote"
    runner = ScriptedRunner(
        tmp_path,
        {
            RoleName.V5_SPECIFICATION_REVIEWER: [invalid, _spec()],
            RoleName.V5_IMPLEMENTATION_REVIEWER: [_assessment("O001")],
            RoleName.V5_REGRESSION_REVIEWER: [_assessment("P001")],
            RoleName.V5_SUMMARY_REVIEWER: [
                {
                    "summary": "All required targets are satisfied.",
                    "decisions": [],
                    "repair_requests": [],
                    "new_questions": [],
                }
            ],
        },
    )
    controller = V5Controller(
        role_runner=runner,
        workspace=FakeWorkspace(),
        artifacts=ArtifactStore(tmp_path / "artifacts"),
    )

    result = asyncio.run(
        controller.run(
            TaskInput("task", "Add behavior", "repo", "base", "title"),
            candidate_patch=b"p0",
            base_worktree="base",
            candidate_worktree="candidate",
            run_id="a-source-retry",
        )
    )

    a_calls = [
        item
        for item in runner.call_details
        if item["role"] == RoleName.V5_SPECIFICATION_REVIEWER
    ]
    assert result.route == "accept"
    assert len(a_calls) == 2
    assert a_calls[0]["session_id"] == a_calls[1]["session_id"]
    assert "quote is not contiguous text" in a_calls[1]["prompt"]


def test_v5_format_correction_does_not_start_a_second_reviewer_attempt(tmp_path):
    corrected = _assessment("O001")
    runner = ScriptedRunner(
        tmp_path,
        {
            RoleName.V5_SPECIFICATION_REVIEWER: [_spec()],
            RoleName.V5_IMPLEMENTATION_REVIEWER: ["MALFORMED_WITH_NOTE"],
            RoleName.JSON_CORRECTOR: [corrected],
            RoleName.V5_REGRESSION_REVIEWER: [_assessment("P001")],
            RoleName.V5_SUMMARY_REVIEWER: [
                {
                    "summary": "All required targets are satisfied.",
                    "decisions": [],
                    "repair_requests": [],
                    "new_questions": [],
                }
            ],
        },
    )
    controller = V5Controller(
        role_runner=runner,
        workspace=FakeWorkspace(),
        artifacts=ArtifactStore(tmp_path / "artifacts"),
    )
    result = asyncio.run(
        controller.run(
            TaskInput("task", "Add behavior", "repo", "base", "title"),
            candidate_patch=b"p0",
            base_worktree="base",
            candidate_worktree="candidate",
            run_id="format-correction",
        )
    )
    assert result.route == "accept"
    assert runner.calls.count(RoleName.V5_IMPLEMENTATION_REVIEWER) == 1
    assert runner.calls.count(RoleName.JSON_CORRECTOR) == 1


class RepairWorkspace(FakeWorkspace):
    async def capture_patch(self, *, worktree, base_commit):
        return b"p1" if worktree == "repair" else self.patch


def test_v5_repair_route_captures_repaired_patch_and_preservation_context(tmp_path):
    decision = {
        "finding_ref": "I/O001",
        "disposition": "repair",
        "reason": "The required behavior is violated.",
        "supporting_refs": ["I/O001"],
        "sources": [
            {"kind": "task", "field": "description", "quote": "Add behavior"}
        ],
    }
    summary = {
        "summary": "One repair is required.",
        "decisions": [decision],
        "repair_requests": [
            {
                "finding_refs": ["I/O001"],
                "change": "Implement the required behavior.",
                "validation_record_refs": ["I/O001"],
                "preserve_refs": ["R/P001"],
            }
        ],
        "new_questions": [],
    }
    runner = ScriptedRunner(
        tmp_path,
        {
            RoleName.V5_SPECIFICATION_REVIEWER: [_spec()],
            RoleName.V5_IMPLEMENTATION_REVIEWER: [
                _assessment("O001", status="violated")
            ],
            RoleName.V5_REGRESSION_REVIEWER: [_assessment("P001")],
            RoleName.V5_SUMMARY_REVIEWER: [summary],
            RoleName.REPAIR: ["repair complete"],
        },
    )
    controller = V5Controller(
        role_runner=runner,
        workspace=RepairWorkspace(),
        artifacts=ArtifactStore(tmp_path / "artifacts"),
    )
    result = asyncio.run(
        controller.run(
            TaskInput("task", "Add behavior", "repo", "base", "title"),
            candidate_patch=b"p0",
            base_worktree="base",
            candidate_worktree="candidate",
            repair_worktree="repair",
            run_id="repair-route",
        )
    )
    repair_call = next(
        item for item in runner.call_details if item["role"] == RoleName.REPAIR
    )
    summary_call = next(
        item
        for item in runner.call_details
        if item["role"] == RoleName.V5_SUMMARY_REVIEWER
    )
    summary_input = json.loads(
        summary_call["prompt"]
        .split("SUMMARY INPUT\n", 1)[1]
        .split("\n\nCONTROLLER FREEZE ID", 1)[0]
    )
    assert result.route == "repair"
    assert result.p1_patch == b"p1"
    assert '"preserve_refs": [' in repair_call["prompt"]
    assert "R/P001" in repair_call["prompt"]
    assert summary_input["allowed_supporting_refs"] == ["I/O001", "R/P001"]
    assert summary_input["allowed_validation_record_refs"] == [
        "I/O001",
        "R/P001",
    ]
    assert summary_input["allowed_preserve_refs"] == ["R/P001"]
    assert {
        json.dumps(source, sort_keys=True)
        for source in summary_input["allowed_sources"]
    } == {
        json.dumps(
            {"kind": "task", "field": "description", "quote": "Add behavior"},
            sort_keys=True,
        ),
        json.dumps(
            {
                "kind": "repository",
                "path": "src/a.py",
                "line_start": 1,
                "line_end": 1,
            },
            sort_keys=True,
        ),
    }


def test_v5_rejects_unbound_new_target_source_then_retries_same_round(tmp_path):
    invalid_target = {
        "statement": "An adjacent behavior stays stable.",
        "importance": "advisory",
        "sources": [
            {
                "kind": "task",
                "field": "description",
                "quote": "text absent from the task",
            }
        ],
    }
    runner = ScriptedRunner(
        tmp_path,
        {
            RoleName.V5_SPECIFICATION_REVIEWER: [_spec()],
            RoleName.V5_IMPLEMENTATION_REVIEWER: [
                {
                    "new_targets": [invalid_target],
                    "assessments": [],
                    "revisit_target_ids": [],
                    "done": False,
                },
                _assessment("O001"),
            ],
            RoleName.V5_REGRESSION_REVIEWER: [_assessment("P001")],
            RoleName.V5_SUMMARY_REVIEWER: [
                {
                    "summary": "All required targets are satisfied.",
                    "decisions": [],
                    "repair_requests": [],
                    "new_questions": [],
                }
            ],
        },
    )
    controller = V5Controller(
        role_runner=runner,
        workspace=FakeWorkspace(),
        artifacts=ArtifactStore(tmp_path / "artifacts"),
    )

    result = asyncio.run(
        controller.run(
            TaskInput("task", "Add behavior", "repo", "base", "title"),
            candidate_patch=b"p0",
            base_worktree="base",
            candidate_worktree="candidate-i",
            regression_worktree="candidate-r",
            run_id="invalid-new-source",
        )
    )

    calls = [
        item
        for item in runner.call_details
        if item["role"] == RoleName.V5_IMPLEMENTATION_REVIEWER
    ]
    assert result.route == "accept"
    assert len(calls) == 2
    assert calls[0]["invocation_id"].endswith(".1.1")
    assert calls[1]["invocation_id"].endswith(".1.2")
    assert "quote is not contiguous text" in calls[1]["prompt"]


class EvidenceWorkspace(FakeWorkspace):
    def __init__(self):
        super().__init__()
        self.validation_calls = []

    async def validate_v5_code_references(self, **kwargs):
        self.validation_calls.append(kwargs)
        if any(
            ref["path"] == "missing.py"
            for assessment in kwargs["assessments"]
            for ref in assessment["code_refs"]
        ):
            return ["candidate code reference range is invalid: missing.py"]
        return []


def test_v5_validates_code_references_before_committing_batch(tmp_path):
    invalid = _assessment("O001")
    invalid["assessments"][0]["code_refs"] = [
        {
            "snapshot": "candidate",
            "path": "missing.py",
            "line_start": 1,
            "line_end": 1,
        }
    ]
    runner = ScriptedRunner(
        tmp_path,
        {
            RoleName.V5_SPECIFICATION_REVIEWER: [_spec()],
            RoleName.V5_IMPLEMENTATION_REVIEWER: [invalid, _assessment("O001")],
            RoleName.V5_REGRESSION_REVIEWER: [_assessment("P001")],
            RoleName.V5_SUMMARY_REVIEWER: [
                {
                    "summary": "All required targets are satisfied.",
                    "decisions": [],
                    "repair_requests": [],
                    "new_questions": [],
                }
            ],
        },
    )
    workspace = EvidenceWorkspace()
    controller = V5Controller(
        role_runner=runner,
        workspace=workspace,
        artifacts=ArtifactStore(tmp_path / "artifacts"),
    )

    result = asyncio.run(
        controller.run(
            TaskInput("task", "Add behavior", "repo", "base", "title"),
            candidate_patch=b"p0",
            base_worktree="base",
            candidate_worktree="candidate-i",
            regression_worktree="candidate-r",
            run_id="invalid-code-ref",
        )
    )

    assert result.route == "accept"
    assert any(
        call["candidate_worktree"] == "candidate-i"
        and call["assessments"][0]["code_refs"]
        for call in workspace.validation_calls
    )
    i_calls = [
        item
        for item in runner.call_details
        if item["role"] == RoleName.V5_IMPLEMENTATION_REVIEWER
    ]
    assert len(i_calls) == 2
    assert "candidate code reference range is invalid" in i_calls[1]["prompt"]


def test_v5_snapshot_mutation_blocks_only_when_summary_has_no_clean_tree(tmp_path):
    class MutationWorkspace(FakeWorkspace):
        def __init__(self):
            super().__init__()
            self.candidate_reads = 0

        async def tracked_diff_digest(self, *, worktree, base_commit):
            del base_commit
            if worktree != "candidate-i":
                return "stable"
            self.candidate_reads += 1
            return "before" if self.candidate_reads == 1 else "after"

    runner = ScriptedRunner(
        tmp_path,
        {
            RoleName.V5_SPECIFICATION_REVIEWER: [_spec()],
            RoleName.V5_IMPLEMENTATION_REVIEWER: [_assessment("O001")],
            RoleName.V5_REGRESSION_REVIEWER: [_assessment("P001")],
        },
    )
    controller = V5Controller(
        role_runner=runner,
        workspace=MutationWorkspace(),
        artifacts=ArtifactStore(tmp_path / "artifacts"),
    )

    with pytest.raises(V5ContractError, match="modified its frozen worktree"):
        asyncio.run(
            controller.run(
                TaskInput("task", "Add behavior", "repo", "base", "title"),
                candidate_patch=b"p0",
                base_worktree="base",
                candidate_worktree="candidate-i",
                regression_worktree="candidate-r",
                run_id="snapshot-mutation",
            )
        )

    assert runner.calls.count(RoleName.V5_IMPLEMENTATION_REVIEWER) == 1
    assert RoleName.V5_SUMMARY_REVIEWER not in runner.calls


def test_v5_reviewer_mutation_is_recorded_and_s_runs_on_separate_tree(tmp_path):
    class RecoveringWorkspace(FakeWorkspace):
        def __init__(self):
            super().__init__()
            self.mutated = False

        async def capture_patch(self, *, worktree, base_commit):
            del base_commit
            return b"p0" if worktree == "candidate-i" else b""

        async def tracked_diff_digest(self, *, worktree, base_commit):
            del base_commit
            if worktree == "candidate-i":
                return "dirty" if self.mutated else "clean"
            return "clean"

        async def restore_reviewer_worktree(self, *, worktree, base_commit):
            del worktree, base_commit
            self.mutated = False
            return b"incidental"

    workspace = RecoveringWorkspace()

    class MutatingRunner(ScriptedRunner):
        async def run_role(self, **kwargs):
            result = await super().run_role(**kwargs)
            if kwargs["role"] == RoleName.V5_IMPLEMENTATION_REVIEWER:
                workspace.mutated = True
            return result

    runner = MutatingRunner(
        tmp_path,
        {
            RoleName.V5_SPECIFICATION_REVIEWER: [_spec()],
            RoleName.V5_IMPLEMENTATION_REVIEWER: [_assessment("O001")],
            RoleName.V5_REGRESSION_REVIEWER: [_assessment("P001")],
            RoleName.V5_SUMMARY_REVIEWER: [
                {
                    "summary": "No repair is required.",
                    "decisions": [],
                    "repair_requests": [],
                    "new_questions": [],
                }
            ],
        },
    )
    controller = V5Controller(
        role_runner=runner,
        workspace=workspace,
        artifacts=ArtifactStore(tmp_path / "artifacts"),
    )

    result = asyncio.run(
        controller.run(
            TaskInput("task", "Add behavior", "repo", "base", "title"),
            candidate_patch=b"p0",
            base_worktree="base",
            candidate_worktree="candidate-i",
            summary_worktree="summary",
            regression_worktree="candidate-r",
            run_id="review-mutation-warning",
        )
    )

    assert result.route == "accept"
    integrity = result.reports["I"]["review_integrity"]
    assert integrity["status"] == "degraded"
    assert integrity["events"][0]["status"] == "restored"
    summary_call = next(
        item for item in runner.call_details if item["role"] == RoleName.V5_SUMMARY_REVIEWER
    )
    assert summary_call["worktree"] == "summary"


def test_v5_completed_run_is_replayed_without_new_model_invocations(tmp_path):
    runner = FakeRunner(tmp_path)
    controller = V5Controller(
        role_runner=runner,
        workspace=FakeWorkspace(),
        artifacts=ArtifactStore(tmp_path / "artifacts"),
    )
    task = TaskInput("task", "Add behavior", "repo", "base", "title")
    first = asyncio.run(
        controller.run(
            task,
            candidate_patch=b"p0",
            base_worktree="base",
            candidate_worktree="candidate",
            run_id="completed-replay",
        )
    )
    call_count = len(runner.calls)
    second = asyncio.run(
        controller.run(
            task,
            candidate_patch=b"p0",
            base_worktree="base",
            candidate_worktree="candidate",
            run_id="completed-replay",
        )
    )

    assert second == first
    assert len(runner.calls) == call_count

    with pytest.raises(V5ContractError, match="different inputs"):
        asyncio.run(
            controller.run(
                TaskInput(
                    "task",
                    "A different task body must not reuse this run.",
                    "repo",
                    "base",
                    "title",
                ),
                candidate_patch=b"p0",
                base_worktree="base",
                candidate_worktree="candidate",
                run_id="completed-replay",
            )
        )


def test_v5_transport_exception_leaves_invocation_recoverable(tmp_path):
    class RaisingRunner:
        async def run_role(self, **_kwargs):
            raise ConnectionError("transport disconnected")

    controller = V5Controller(
        role_runner=RaisingRunner(),
        workspace=FakeWorkspace(),
        artifacts=ArtifactStore(tmp_path / "artifacts"),
    )
    controller._active_run_id = "run"
    controller.state_store.create_run("run", {"task": "x"})

    with pytest.raises(ConnectionError, match="transport disconnected"):
        asyncio.run(
            controller._run_role(
                role=RoleName.V5_SPECIFICATION_REVIEWER,
                prompt="prompt",
                worktree="base",
                prefix="v5/test",
                invocation_id="recoverable-invocation",
            )
        )

    assert controller.state_store.invocation("recoverable-invocation")["status"] == (
        "PREPARED"
    )


def test_v5_resume_reuses_exact_prepared_prompt_and_attempt(tmp_path):
    runner = ScriptedRunner(
        tmp_path,
        {
            RoleName.V5_SPECIFICATION_REVIEWER: [_spec()],
            RoleName.V5_IMPLEMENTATION_REVIEWER: [
                IndeterminateInvocationError("transport state is uncertain"),
                _assessment("O001"),
            ],
            RoleName.V5_REGRESSION_REVIEWER: [
                _assessment("P001"),
                _assessment("P001"),
            ],
            RoleName.V5_SUMMARY_REVIEWER: [
                {
                    "summary": "All required targets are satisfied.",
                    "decisions": [],
                    "repair_requests": [],
                    "new_questions": [],
                }
            ],
        },
    )
    controller = V5Controller(
        role_runner=runner,
        workspace=FakeWorkspace(),
        artifacts=ArtifactStore(tmp_path / "artifacts"),
    )
    task = TaskInput("task", "Add behavior", "repo", "base", "title")

    with pytest.raises(IndeterminateInvocationError):
        asyncio.run(
            controller.run(
                task,
                candidate_patch=b"p0",
                base_worktree="base",
                candidate_worktree="candidate-i",
                regression_worktree="candidate-r",
                run_id="prepared-resume",
            )
        )

    prepared = controller.state_store.attempt_record(
        "prepared-resume", "I", 1, 1
    )
    assert prepared["status"] == "PREPARED"
    assert prepared["input_payload"]["prompt"]
    result = asyncio.run(
        controller.run(
            task,
            candidate_patch=b"p0",
            base_worktree="base",
            candidate_worktree="candidate-i",
            regression_worktree="candidate-r",
            run_id="prepared-resume",
        )
    )

    i_calls = [
        item
        for item in runner.call_details
        if item["role"] == RoleName.V5_IMPLEMENTATION_REVIEWER
    ]
    assert result.route == "accept"
    assert len(i_calls) == 2
    assert i_calls[0]["prompt"] == i_calls[1]["prompt"]
    assert i_calls[0]["invocation_id"] == i_calls[1]["invocation_id"]
    committed = controller.state_store.attempt_record(
        "prepared-resume", "I", 1, 1
    )
    assert committed["status"] == "COMMITTED"


def test_v5_no_progress_limit_stops_lane_and_holds_p0(tmp_path):
    empty_batch = {
        "new_targets": [],
        "assessments": [],
        "revisit_target_ids": [],
        "done": False,
    }
    runner = ScriptedRunner(
        tmp_path,
        {
            RoleName.V5_SPECIFICATION_REVIEWER: [_spec()],
            RoleName.V5_IMPLEMENTATION_REVIEWER: [empty_batch, empty_batch],
            RoleName.V5_REGRESSION_REVIEWER: [_assessment("P001")],
            RoleName.V5_SUMMARY_REVIEWER: [
                {
                    "summary": "The implementation lane is incomplete.",
                    "decisions": [],
                    "repair_requests": [],
                    "new_questions": [],
                }
            ],
        },
    )
    controller = V5Controller(
        role_runner=runner,
        workspace=FakeWorkspace(),
        artifacts=ArtifactStore(tmp_path / "artifacts"),
        config=V5Config(max_no_progress=2),
    )

    result = asyncio.run(
        controller.run(
            TaskInput("task", "Add behavior", "repo", "base", "title"),
            candidate_patch=b"p0",
            base_worktree="base",
            candidate_worktree="candidate-i",
            regression_worktree="candidate-r",
            run_id="no-progress",
        )
    )

    assert result.route == "hold"
    assert result.p1_patch == b"p0"
    assert result.reports["I"]["completion"] == "incomplete"
    assert result.reports["I"]["stop_reason"] == "max_no_progress"
    assert runner.calls.count(RoleName.V5_IMPLEMENTATION_REVIEWER) == 2


def test_v5_empty_ledger_can_complete_without_inventing_targets(tmp_path):
    empty_spec = {
        "summary": "The task creates no reviewable product obligation.",
        "obligations": [],
        "preservation_obligations": [],
        "open_questions": [],
    }
    done = {
        "new_targets": [],
        "assessments": [],
        "revisit_target_ids": [],
        "done": True,
    }
    runner = ScriptedRunner(
        tmp_path,
        {
            RoleName.V5_SPECIFICATION_REVIEWER: [empty_spec],
            RoleName.V5_IMPLEMENTATION_REVIEWER: [done],
            RoleName.V5_REGRESSION_REVIEWER: [done],
            RoleName.V5_SUMMARY_REVIEWER: [
                {
                    "summary": "There are no findings to adjudicate.",
                    "decisions": [],
                    "repair_requests": [],
                    "new_questions": [],
                }
            ],
        },
    )
    controller = V5Controller(
        role_runner=runner,
        workspace=FakeWorkspace(),
        artifacts=ArtifactStore(tmp_path / "artifacts"),
    )

    result = asyncio.run(
        controller.run(
            TaskInput("task", "No product change is required.", "repo", "base"),
            candidate_patch=b"",
            base_worktree="base",
            candidate_worktree="candidate-i",
            regression_worktree="candidate-r",
            run_id="empty-ledger",
        )
    )

    assert result.route == "accept"
    assert result.reports["I"]["assessments"] == []
    assert result.reports["R"]["assessments"] == []


def test_v5_last_legal_round_is_kept_but_incomplete_without_done(tmp_path):
    runner = ScriptedRunner(
        tmp_path,
        {
            RoleName.V5_SPECIFICATION_REVIEWER: [_spec()],
            RoleName.V5_IMPLEMENTATION_REVIEWER: [
                _assessment("O001", done=False)
            ],
            RoleName.V5_REGRESSION_REVIEWER: [_assessment("P001")],
            RoleName.V5_SUMMARY_REVIEWER: [
                {
                    "summary": "The implementation lane did not declare completion.",
                    "decisions": [],
                    "repair_requests": [],
                    "new_questions": [],
                }
            ],
        },
    )
    controller = V5Controller(
        role_runner=runner,
        workspace=FakeWorkspace(),
        artifacts=ArtifactStore(tmp_path / "artifacts"),
        config=V5Config(max_rounds=1),
    )

    result = asyncio.run(
        controller.run(
            TaskInput("task", "Add behavior", "repo", "base", "title"),
            candidate_patch=b"p0",
            base_worktree="base",
            candidate_worktree="candidate-i",
            regression_worktree="candidate-r",
            run_id="last-round-incomplete",
        )
    )

    assert result.route == "hold"
    assert result.reports["I"]["completion"] == "incomplete"
    assert result.reports["I"]["assessments"][0]["id"] == "I/O001"
    assert result.reports["I"]["stop_reason"] == "deadline_or_round_limit"


def test_v5_archives_reviewer_probe_material_with_bundle(tmp_path):
    class MaterialRunner(FakeRunner):
        async def run_role(self, **kwargs):
            result = await super().run_role(**kwargs)
            if kwargs["role"] == RoleName.V5_IMPLEMENTATION_REVIEWER:
                output_root = next(
                    line.split(": ", 1)[1]
                    for line in kwargs["prompt"].splitlines()
                    if line.startswith("OUTPUT ROOT:")
                )
                log = Path(output_root) / "logs/check.txt"
                log.parent.mkdir(parents=True, exist_ok=True)
                log.write_text("exit=0\nobserved=ok\n", encoding="utf-8")
            return result

    runner = MaterialRunner(tmp_path)
    controller = V5Controller(
        role_runner=runner,
        workspace=FakeWorkspace(),
        artifacts=ArtifactStore(tmp_path / "artifacts"),
    )

    asyncio.run(
        controller.run(
            TaskInput("task", "Add behavior", "repo", "base", "title"),
            candidate_patch=b"p0",
            base_worktree="base",
            candidate_worktree="candidate",
            run_id="material-archive",
        )
    )

    manifests = list((tmp_path / "artifacts/v5/bundles").glob("*/materials-manifest.json"))
    implementation_manifest = next(
        json.loads(path.read_text(encoding="utf-8"))
        for path in manifests
        if json.loads(path.read_text(encoding="utf-8"))["role"] == "I"
    )
    entry = implementation_manifest["files"][0]
    bundle_root = next(
        path.parent for path in manifests if path.parent.joinpath(entry["archived_path"]).is_file()
    )
    assert entry["material_type"] == "log"
    assert (bundle_root / entry["archived_path"]).read_text(encoding="utf-8") == (
        "exit=0\nobserved=ok\n"
    )


def test_v5_s_challenge_to_required_satisfied_finding_routes_hold(tmp_path):
    summary = {
        "summary": "The two archived observations leave the result uncertain.",
        "decisions": [
            {
                "finding_ref": "I/O001",
                "disposition": "unresolved",
                "reason": "A conflicting observation was not resolved.",
                "supporting_refs": ["I/O001", "R/P001"],
                "sources": [
                    {
                        "kind": "task",
                        "field": "description",
                        "quote": "Add behavior",
                    }
                ],
            }
        ],
        "repair_requests": [],
        "new_questions": [],
    }
    runner = ScriptedRunner(
        tmp_path,
        {
            RoleName.V5_SPECIFICATION_REVIEWER: [_spec()],
            RoleName.V5_IMPLEMENTATION_REVIEWER: [_assessment("O001")],
            RoleName.V5_REGRESSION_REVIEWER: [_assessment("P001")],
            RoleName.V5_SUMMARY_REVIEWER: [summary],
        },
    )
    controller = V5Controller(
        role_runner=runner,
        workspace=FakeWorkspace(),
        artifacts=ArtifactStore(tmp_path / "artifacts"),
    )

    result = asyncio.run(
        controller.run(
            TaskInput("task", "Add behavior", "repo", "base", "title"),
            candidate_patch=b"p0",
            base_worktree="base",
            candidate_worktree="candidate",
            run_id="challenged-satisfied",
        )
    )

    assert result.route == "hold"


def test_v5_summary_artifact_paths_are_relative_to_artifact_root(tmp_path):
    root = tmp_path / "bundles"
    bundle = root / ("a" * 64)
    artifact = {
        "assessment_ref": "I/O001",
        "version": 1,
        "note_path": str(bundle / "notes/O001.md"),
        "note_sha256": "b" * 64,
        "trace_manifest_path": str(bundle / "trace-manifest.json"),
        "materials_manifest_path": str(bundle / "materials-manifest.json"),
    }

    public = V5Controller._public_review_artifacts(
        [artifact], artifact_root=str(root)
    )[0]

    assert public["note_path"] == f"{'a' * 64}/notes/O001.md"
    assert public["trace_manifest_path"] == f"{'a' * 64}/trace-manifest.json"
    assert public["materials_manifest_path"] == (
        f"{'a' * 64}/materials-manifest.json"
    )


def test_v5_repaired_target_is_not_also_automatic_preservation_context():
    ledger = {
        "obligations": [
            {
                "id": "O001",
                "content": {
                    "statement": "Behavior works.",
                    "importance": "required",
                    "sources": [],
                },
            }
        ],
        "preservation_obligations": [],
    }
    reports = {
        "I": {
            "assessments": [
                {
                    "id": "I/O001",
                    "content": {
                        "target_id": "O001",
                        "status": "violated",
                    },
                }
            ]
        },
        "R": {
            "assessments": [
                {
                    "id": "R/O001",
                    "content": {
                        "target_id": "O001",
                        "status": "satisfied",
                    },
                }
            ]
        },
    }
    summary = {
        "decisions": [
            {"finding_ref": "I/O001", "disposition": "repair"}
        ]
    }

    assert V5Controller._preserve_refs(ledger, reports, summary) == []
