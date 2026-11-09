from __future__ import annotations

import asyncio
import json

from a_agent_finetune.run_e_inference import FrozenWorkspace, prepare_e_ledger
from external_agents_eval.workflows.solve_review_repair.artifacts import ArtifactStore, sha256_bytes
from external_agents_eval.workflows.solve_review_repair.models import (
    RoleName,
    RoleRunResult,
    TaskInput,
)
from external_agents_eval.workflows.solve_review_repair.v6_config import V6Config
from external_agents_eval.workflows.solve_review_repair.v6_fork import V6StageSource


def test_e_stops_after_a_and_exports_reusable_ledger(tmp_path):
    repository = tmp_path / "base"
    repository.mkdir()
    source = b"def retry():\n    return False\n"
    (repository / "retry.py").write_bytes(source)
    store = ArtifactStore(tmp_path / "output")

    class Runner:
        def __init__(self):
            self.invocations = []

        async def run_role(self, *, role, tools, **_kwargs):
            assert role == RoleName.V5_SPECIFICATION_REVIEWER
            assert tools == ("read", "grep", "find", "ls")
            self.invocations.append({"role": role.value})
            payload = {
                "summary": "Add bounded retries while preserving the public function.",
                "obligations": [
                    {
                        "statement": "Retry transient failures.",
                        "importance": "required",
                        "sources": [
                            {
                                "kind": "task",
                                "field": "description",
                                "quote": "Retry transient failures.",
                            }
                        ],
                    }
                ],
                "preservation_obligations": [
                    {
                        "statement": "Keep the retry function callable without arguments.",
                        "importance": "required",
                        "sources": [
                            {
                                "kind": "repository",
                                "path": "retry.py",
                                "line_start": 1,
                                "line_end": 2,
                            }
                        ],
                    }
                ],
                "open_questions": [],
            }
            return RoleRunResult(
                exit_code=0,
                final_assistant_text=json.dumps(payload),
                transcript_path=store.root / "pi.jsonl",
                trajectory_path=store.root / "trajectory.json",
                started_at="now",
                finished_at="now",
            )

    task = TaskInput(
        task_id="retry-example",
        issue_body="Retry transient failures.",
        repository_name="retry",
        base_commit="a" * 40,
        issue_title="Add retries",
    )
    result = asyncio.run(
        prepare_e_ledger(
            task=task,
            runner=Runner(),
            workspace=FrozenWorkspace(repository),
            store=store,
            config=V6Config(max_format_retries=1, max_semantic_retries=0),
        )
    )
    assert result["final_state"] == "A_COMPLETED"
    assert result["ledger_quality"] == "valid"
    assert result["correction_calls"] == 0
    assert not result["downstream_started"]
    assert result["baseline_unchanged"]
    assert len(result["invocations"]) == 1
    ledger = V6StageSource(
        store.root,
        task_id=task.task_id,
        base_commit=task.base_commit,
        baseline_sha256=sha256_bytes(b""),
    ).ledger()
    assert ledger["obligations"][0]["id"] == "O001"
    bindings = json.loads((store.root / "v5/source-bindings.json").read_text())
    assert bindings[1]["file_sha256"] == sha256_bytes(source)
    assert bindings[1]["quote"] == source.decode().rstrip("\n")


def test_e_rejects_absolute_source_without_rewriting(tmp_path):
    (tmp_path / "file.py").write_text("value = 1\n")
    source = {
        "id": "O001/S001",
        "kind": "repository",
        "path": "/work/base/file.py",
        "line_start": 1,
        "line_end": 1,
    }
    bound, errors = asyncio.run(
        FrozenWorkspace(tmp_path).bind_repository_sources(base_commit="a" * 40, sources=[source])
    )
    assert not bound
    assert "must be relative" in errors[0]
    assert source["path"] == "/work/base/file.py"
