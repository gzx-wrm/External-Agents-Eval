from __future__ import annotations

import ast
import asyncio
import inspect
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from external_agents_eval.agents.pier.pi_role_runner import ContainerGitWorkspaceManager
from external_agents_eval.benchmarks.deepswe import runner
from external_agents_eval.benchmarks.deepswe.manifest import DeepSWEManifest
from external_agents_eval.cli import build_parser
from external_agents_eval.workflows.solve_review_repair import ft_inference, ft_protocol
from external_agents_eval.workflows.solve_review_repair.artifacts import ArtifactStore, sha256_json
from external_agents_eval.workflows.solve_review_repair.ft_evidence import observed_evidence
from external_agents_eval.workflows.solve_review_repair.ft_inference import (
    FT_RESOURCES,
    FT_SYSTEM_PROMPT,
    FTLedgerClient,
)
from external_agents_eval.workflows.solve_review_repair.models import (
    IndeterminateInvocationError,
    RoleRunResult,
    TaskInput,
)
from external_agents_eval.workflows.solve_review_repair.v5_prompts import spec_prompt
from external_agents_eval.workflows.solve_review_repair.v62_controller import V62Controller

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("source,names", [
    ("a_agent_finetune/build_f_observed.py", ["returned_span"]),
    ("a_agent_finetune/build_f_compact.py", ["merge", "render_user"]),
    ("a_agent_finetune/runs/ft-ds41-fixed-evidence30-thinking-20261020-w3-r1/f_validation_snapshot.py",
     ["strict_json", "validate_output"]),
])
def test_v62_protocol_functions_are_unmodified_ft_snapshots(source, names):
    text = (ROOT / source).read_text()
    functions = {node.name: ast.get_source_segment(text, node) for node in ast.parse(text).body
                 if isinstance(node, ast.FunctionDef)}
    for name in names:
        assert inspect.getsource(getattr(ft_protocol, name)).strip() == functions[name]


def test_v62_prompt_schema_and_all_current_inputs_match_ft():
    assert (FT_RESOURCES / "prompt.txt").read_bytes() == (
        ROOT / "a_agent_finetune/f_prompt_v4.txt"
    ).read_bytes()
    assert (FT_RESOURCES / "spec_output_schema.json").read_bytes() == (
        ROOT / "a_agent_finetune/spec_output_schema.json"
    ).read_bytes()
    data = ROOT / "a_agent_finetune/data/f_teacher_observed_30_current"
    rows = [json.loads(line) for line in (data / "training_candidates.jsonl").read_text().splitlines()]
    manifest = json.loads((data / "manifest.json").read_text())
    assert len(rows) == len(manifest["tasks"]) == 30
    for row, item in zip(rows, manifest["tasks"], strict=True):
        record = json.loads((data / "records" / f"{item['task_id']}.json").read_text())
        assert row["prompt"] == [
            {"role": "system", "content": FT_SYSTEM_PROMPT},
            {"role": "user", "content": ft_protocol.render_user(record, record["evidence"])},
        ]


def _read_step(number, path, content, **arguments):
    return {
        "step_id": number,
        "tool_calls": [{"tool_call_id": str(number), "function_name": "read",
                        "arguments": {"path": path, **arguments}}],
        "observation": {"results": [{"source_call_id": str(number), "content": content}]},
    }


class _Workspace:
    def __init__(self):
        self.reads = []

    async def tracked_diff_digest(self, **kwargs):
        return "baseline"

    async def read_frozen_file(self, *, path, **kwargs):
        self.reads.append(path)
        if path == "a.py":
            return path, b"one\ntwo\nthree\nfour\n"
        return None


def test_v62_packet_keeps_only_actual_returns_and_merges_overlaps():
    workspace = _Workspace()
    trajectory = {"steps": [
        _read_step(1, "/base/a.py", "one\ntwo", limit=2),
        _read_step(2, "/base/a.py", "two\nthree", offset=2, limit=2),
        _read_step(3, "/base/a.py", "", offset=99),
        _read_step(4, "/scratch/a.py", "outside"),
        _read_step(5, "/base/not-tracked.py", "not tracked"),
    ]}
    packet = asyncio.run(observed_evidence(
        trajectory, task_id="task", base_root="/base", base_commit="base", workspace=workspace,
    ))
    assert workspace.reads == ["a.py", "not-tracked.py"]
    assert len(packet["evidence"]) == 1
    assert packet["evidence"][0]["text"] == "one\ntwo\nthree\n"
    assert packet["evidence"][0]["line_end"] == 3
    assert len(packet["skipped_reads"]) == 3


def _payload():
    return {"summary": "Deliver the requested behavior.", "obligations": [{
        "statement": "Deliver the requested behavior.", "importance": "required",
        "sources": [{"kind": "task", "field": "description", "quote": "Implement the feature."}],
    }], "preservation_obligations": [], "open_questions": []}


class _SearchRunner:
    def __init__(self, root):
        self.root = root
        self.calls = []

    async def run_role(self, **kwargs):
        self.calls.append(kwargs)
        path = self.root / "search.json"
        path.write_text(json.dumps({"steps": [_read_step(1, "/base/a.py", "one", limit=1)]}))
        return RoleRunResult(
            exit_code=0, final_assistant_text="Discard this draft, even if it is not valid JSON.",
            transcript_path=path, trajectory_path=path, started_at="now", finished_at="now",
        )


class _LedgerClient:
    def __init__(self):
        self.samples = []

    async def generate(self, sample, **kwargs):
        self.samples.append(sample)
        return _payload(), {"errors": []}


def test_v62_original_search_prompt_and_ft_ledger_are_connected_without_draft_leakage(tmp_path):
    search = _SearchRunner(tmp_path)
    client = _LedgerClient()
    artifacts = ArtifactStore(tmp_path / "artifacts")
    task = TaskInput(task_id="repo-long-task-id", repository_name="repo", base_commit="base",
                     issue_title="Feature", issue_body="Implement the feature.")
    controller = V62Controller(
        role_runner=search, ledger_client=client, workspace=_Workspace(), artifacts=artifacts,
    )
    try:
        ledger = asyncio.run(controller.prepare_ledger(task, base_worktree="/base"))
        assert ledger is not None
        assert ledger["obligations"][0]["content"]["statement"] == "Deliver the requested behavior."
        assert len(search.calls) == len(client.samples) == 1
        assert search.calls[0]["prompt"] == spec_prompt(
            {"title": "Feature", "description": "Implement the feature."},
            {"name": "repo", "base_commit": "base", "base_root": "/base"}, managed_delivery=[],
        )
        assert search.calls[0]["tools"] == ("read", "grep", "find", "ls")
        sample = client.samples[0]
        assert sample["task_id"] == task.task_id
        assert sample["messages"][0]["content"] == FT_SYSTEM_PROMPT
        historical_record = {"task": {"description": "Implement the feature.", "title": "Feature"},
                             "managed_delivery": []}
        assert sample["messages"][1]["content"] == ft_protocol.render_user(
            historical_record, sample["evidence"],
        )
        assert "Discard this draft" not in json.dumps(sample["messages"])
        assert "one" in sample["messages"][1]["content"]
        # An interrupted compilation can reuse the frozen packet without another search.
        (artifacts.root / "v5/ledger.json").unlink()
        repeated = asyncio.run(controller.prepare_ledger(task, base_worktree="/base"))
        assert repeated == ledger
        assert len(search.calls) == 1
        assert client.samples[0]["messages"] == client.samples[1]["messages"]
        envelope = json.loads((artifacts.root / "v6/pre-solver-ledger.json").read_text())
        assert envelope["schema_version"] == "v6.2-pre-solver-ledger-1"
    finally:
        controller.close()


def test_v62_keeps_fixed_evidence_source_warnings_after_resume(tmp_path):
    source_error = {"check": "source", "path": "obligations/0/sources/0",
                    "message": "Repository range is absent from or empty in the supplied evidence"}

    class Client(_LedgerClient):
        async def generate(self, sample, *, artifacts):
            payload = _payload()
            payload["obligations"][0]["sources"] = [{
                "kind": "repository", "path": "a.py", "line_start": 4, "line_end": 4,
            }]
            checks = {"errors": [source_error]}
            artifacts.write_json("v6.2/ledger/validation.json", checks)
            return payload, checks

    controller = V62Controller(
        role_runner=_SearchRunner(tmp_path), ledger_client=Client(), workspace=_Workspace(),
        artifacts=ArtifactStore(tmp_path / "artifacts"),
    )
    task = TaskInput(task_id="task", repository_name="repo", base_commit="base",
                     issue_body="Implement the feature.")
    try:
        for _ in range(2):
            assert asyncio.run(controller.prepare_ledger(task, base_worktree="/base")) is not None
            assert controller.ledger_quality == "degraded"
            assert json.dumps(source_error, ensure_ascii=False, sort_keys=True) in controller.ledger_warnings
    finally:
        controller.close()


class _Response:
    status = 200

    def __init__(self, finish_reason="stop"):
        self.body = {"id": "response", "model": "deepseek-flash", "choices": [{
            "finish_reason": finish_reason,
            "message": {"content": json.dumps(_payload()), "reasoning_content": "Think."},
        }], "usage": {"prompt_tokens": 100, "completion_tokens": 50, "prompt_cache_hit_tokens": 80}}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return json.dumps(self.body).encode()


def _sample():
    messages = [{"role": "system", "content": FT_SYSTEM_PROMPT}, {"role": "user", "content": "Task"}]
    return {"task_id": "task", "base_commit": "base", "task": {"description": "Implement the feature."},
            "evidence": [], "messages": messages, "messages_sha256": sha256_json(messages)}


@pytest.mark.parametrize("finish_reason", ["stop", "length"])
def test_v62_ft_request_settings_single_attempt_and_usage(tmp_path, monkeypatch, finish_reason):
    requests = []

    def respond(request, *, timeout):
        requests.append(json.loads(request.data))
        assert timeout == 1800
        return _Response(finish_reason)

    monkeypatch.setattr(ft_inference, "urlopen", respond)
    artifacts = ArtifactStore(tmp_path, secrets=("test-secret",))
    client = FTLedgerClient(model="deepseek-flash", base_url="https://example.test/v1", api_key="test-secret")
    if finish_reason == "stop":
        assert asyncio.run(client.generate(_sample(), artifacts=artifacts))[0] == _payload()
        asyncio.run(client.generate(_sample(), artifacts=artifacts))
    else:
        for _ in range(2):
            with pytest.raises(ValueError, match="truncated"):
                asyncio.run(client.generate(_sample(), artifacts=artifacts))
    assert len(requests) == 1
    request = requests[0]
    assert request["thinking"] == {"type": "enabled"}
    assert request["reasoning_effort"] == "high"
    assert request["max_tokens"] == 65536
    assert request["seed"] == 42
    assert not {"tools", "temperature"} & request.keys()
    trajectory = json.loads((tmp_path / "v6.2/ledger/trajectory.json").read_text())
    assert trajectory["final_metrics"]["total_prompt_tokens"] == 100
    assert trajectory["final_metrics"]["total_cached_tokens"] == 80
    assert all("test-secret" not in path.read_text() for path in tmp_path.rglob("*") if path.is_file())


def test_v62_does_not_resend_an_unfinished_ft_request(tmp_path):
    client = FTLedgerClient(model="deepseek-flash", base_url="https://example.test/v1", api_key="unused")
    artifacts = ArtifactStore(tmp_path)
    sample = _sample()
    artifacts.write_json("v6.2/ledger/request.json", {
        "model": "deepseek-flash", "messages": sample["messages"], "thinking": {"type": "enabled"},
        "reasoning_effort": "high", "max_tokens": 65536, "seed": 42, "stream": False,
    })
    with pytest.raises(IndeterminateInvocationError):
        asyncio.run(client.generate(sample, artifacts=artifacts))


def test_v62_frozen_file_reads_git_instead_of_modified_worktree(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True).stdout

    git("init")
    (repo / "source.txt").write_bytes(b"original\n\n")
    (repo / "alias.txt").symlink_to("source.txt")
    git("add", ".")
    git("-c", "user.name=Test", "-c", "user.email=test@example.test", "commit", "-m", "baseline")
    commit = git("rev-parse", "HEAD").decode().strip()
    (repo / "source.txt").write_text("modified")

    class LocalEnvironment:
        async def exec(self, command, **kwargs):
            result = await asyncio.to_thread(
                subprocess.run, command, shell=True, cwd=kwargs.get("cwd"),
                capture_output=True, text=True, check=False,
            )
            return SimpleNamespace(return_code=result.returncode, stdout=result.stdout, stderr=result.stderr)

        async def download_file(self, remote, local):
            shutil.copyfile(remote, local)

    workspace = ContainerGitWorkspaceManager(environment=LocalEnvironment(), transfer_dir=tmp_path / "transfer")
    workspace.INPUT_ROOT = str(tmp_path / "contracts")
    Path(workspace.INPUT_ROOT).mkdir()
    assert asyncio.run(workspace.read_frozen_file(worktree=str(repo), base_commit=commit, path="alias.txt")) == (
        "source.txt", b"original\n\n",
    )


def test_v62_cli_and_dry_run_use_independent_entrypoint(tmp_path, monkeypatch, capsys):
    args = build_parser().parse_args(["deepswe", "run", "--agent", "pi-v62-review-repair"])
    assert args.agent == "pi-v62-review-repair"
    manifest = DeepSWEManifest(
        path=tmp_path / "manifest.json", task_ids=("task-a",), sample_seed=1, sample_size=1,
        population_count=113, population_fingerprint="a" * 64, raw={},
    )
    monkeypatch.setattr(runner, "validate_manifest_against_checkout", lambda *_: tmp_path)
    runner.run_pier(checkout=tmp_path, manifest=manifest, agent_id=args.agent, jobs_dir=tmp_path,
                    dry_run=True, environment="modal", workflow_timeout_sec=3600)
    metadata = json.loads(capsys.readouterr().out)
    assert metadata["workflow"] == "solve-review-repair-v6.2"
    assert metadata["review_prompt_version"] == "v5-5"
    assert "workflow_timeout_sec=3600" in metadata["command"]
    assert any("pi_v62_solve_review_repair:PiV62" in part for part in metadata["command"])
