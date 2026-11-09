"""V6.5 Mini backend: D file contract, tool-less X and the view workspace."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_mini_v63 import LocalBridge, install_stub_model, narrative, record, state, write_files

from external_agents_eval.agents.mini_review_files import SCOPE_PATH, collect_batch, controller_state
from external_agents_eval.agents.mini_v64 import REVIEW_SYSTEM_V64
from external_agents_eval.agents.mini_v65 import (
    ASSUMPTION_SYSTEM_V65,
    FILE_CONTRACT_V65_D,
    MiniV65ReviewAgent,
    MiniV65ReviewRunner,
    MiniV65WorkspaceManager,
    adapt_prompt_v65_d,
)
from external_agents_eval.workflows.solve_review_repair.v65_config import V65Config
from external_agents_eval.workflows.solve_review_repair.v65_controller import V65Role
from external_agents_eval.workflows.solve_review_repair.v65_prompts import disambiguation_prompt


def d_prompt(s):
    return disambiguation_prompt(
        task={"title": "Task", "description": "Add behavior"}, repository={},
        ledger={"obligations": [], "preservation_obligations": [], "open_questions": []},
        state=s, role="D", skill_path="skill.md", output_root=s["review_notes"]["output_root"],
        initial=True, managed_delivery=[], limits={"batch_size": 3, "time_budget_sec": 2700},
    )


def d_scope(complete=True, new=None):
    fields = {"scope_complete": complete, "new_targets": new or [], "revisit_target_ids": []}
    return "# Review note: D_SCOPE\n\n## Scope\n\n```json\n" + json.dumps(fields) + "\n```\n" + narrative()


def test_d_prompt_adapts_to_the_d_file_contract(tmp_path):
    prompt = d_prompt(state(tmp_path, pending=("Q001", "X001")))
    adapted = adapt_prompt_v65_d(prompt)
    assert prompt.startswith("ROLE: V5 D Reviewer (disambiguation)")
    assert "OUTPUT CONTRACT (files)" in adapted and "'# Review note: D_SCOPE'" in adapted
    assert "(ROLE is I or R)" not in adapted and "For D, the Assessment block" in adapted
    assert "OUTPUT SCHEMA" not in adapted.split("OUTPUT CONTRACT (files)", 1)[1]
    assert controller_state(adapted)["pending_target_ids"] == ["Q001", "X001"]
    assert FILE_CONTRACT_V65_D.count("D_SCOPE") == 1


def test_d_records_use_the_unchanged_batch_rules(tmp_path):
    derived = {"statement": "The bracketed form fails.", "importance": "required",
               "sources": [{"kind": "task", "field": "description", "quote": "Add behavior"}]}
    s = state(tmp_path, pending=("Q001", "X001"))
    files = {"notes/Q001.md": record("Q001"), "notes/X001.md": record("X001", "violated"),
             SCOPE_PATH: d_scope(new=[derived])}
    batch, _ = collect_batch(files, role="D", state=s, archive=tmp_path / "archive")
    assert batch["done"] is False and batch["new_targets"] == [derived]
    assert {item["target_id"] for item in batch["assessments"]} == {"Q001", "X001"}
    with pytest.raises(Exception, match="title does not match"):
        collect_batch({"notes/Q001.md": record("Q001"), "notes/X001.md": record("X001"),
                       SCOPE_PATH: d_scope().replace("D_SCOPE", "R_SCOPE")},
                      role="D", state=s, archive=tmp_path / "archive2")


def test_role_system_prompts():
    runner = object.__new__(MiniV65ReviewRunner)
    assert runner._mini_system(role=V65Role.DISAMBIGUATION_REVIEWER, tools=("bash",),
                               schemas=None) == REVIEW_SYSTEM_V64
    x = runner._mini_system(role=V65Role.ASSUMPTION_EXTRACTOR, tools=(), schemas=None)
    assert x == ASSUMPTION_SYSTEM_V65 and "tools are disabled" in " ".join(x.split())


def test_agent_wiring():
    assert MiniV65ReviewAgent.workflow_config_type is V65Config
    assert MiniV65ReviewAgent.workflow_version == "6.5"
    assert MiniV65ReviewAgent.name() == "v65-mini-review"
    with pytest.raises(ValueError, match="build gate"):
        MiniV65ReviewAgent(logs_dir=None)


class RecordingBridge(LocalBridge):
    def __init__(self):
        self.uploads = []

    async def upload_file(self, source, target):
        self.uploads.append(str(target))


def make_runner(tmp_path, bridge):
    return MiniV65ReviewRunner(environment=bridge, logs_root=tmp_path / "host", model="deepseek-flash",
                               provider="deepseek", agent_version="2.4.6", env={},
                               redact=lambda value: value)


class CatalogEnvironment:
    def __init__(self):
        self.commands, self.uploads = [], []

    async def exec(self, command, cwd=None, **kwargs):
        self.commands.append(command)
        return SimpleNamespace(return_code=1 if command.startswith("test -f") else 0,
                               stdout="", stderr="")

    async def upload_file(self, source, target):
        self.uploads.append((Path(source).read_bytes(), target))


@pytest.mark.parametrize("role", ["I", "R", "D"])
def test_review_catalog_is_published_for_every_lane(tmp_path, role):
    environment = CatalogEnvironment()
    runner = make_runner(tmp_path, environment)
    catalog = {"role": role, "targets": [{"id": "Q001", "content": {"statement": "s"}}]}
    remote = asyncio.run(runner.prepare_review_catalog(role=role, catalog=catalog))
    assert remote.startswith(f"/external_eval/v5_resources/catalog-{role}-")
    assert environment.uploads[0][1] == remote
    assert json.loads(environment.uploads[0][0]) == catalog
    assert any(command.startswith("chmod 0444") for command in environment.commands)


def test_assumption_extractor_refuses_tools(tmp_path):
    runner = make_runner(tmp_path, RecordingBridge())
    with pytest.raises(ValueError, match="tool-less"):
        asyncio.run(runner.run_role(role=V65Role.ASSUMPTION_EXTRACTOR, prompt="x", worktree="/w",
                                    tools=("bash",), timeout_sec=30, artifact_prefix="X/1"))


def test_assumption_extractor_session_never_reaches_the_container(tmp_path, monkeypatch):
    submission = json.dumps({"assumptions": []})
    install_stub_model(monkeypatch, lambda prompt: ["ls /", "submit_json '" + submission + "'"])
    bridge = RecordingBridge()
    result = asyncio.run(make_runner(tmp_path, bridge).run_role(
        role=V65Role.ASSUMPTION_EXTRACTOR, prompt="read the patch", worktree=str(tmp_path),
        tools=(), timeout_sec=30, artifact_prefix="v65/assumptions/attempt-1"))
    assert result.exit_code == 0 and json.loads(result.final_assistant_text) == {"assumptions": []}
    assert bridge.uploads == []
    native = json.loads(result.transcript_path.read_text())
    blocked = [m for m in native["messages"] if m["role"] == "tool"]
    assert "Filesystem tools are disabled" in blocked[0]["content"]


def test_real_mini_loop_for_d(tmp_path, monkeypatch):
    root = tmp_path / "current"
    root.mkdir()
    commands = [write_files({"notes/Q001.md": record("Q001", "disputed"), SCOPE_PATH: d_scope()})]
    install_stub_model(monkeypatch, lambda prompt: commands)
    prompt = d_prompt(state(root, pending=("Q001",)))
    result = asyncio.run(make_runner(tmp_path, RecordingBridge()).run_role(
        role=V65Role.DISAMBIGUATION_REVIEWER, prompt=prompt, worktree=str(tmp_path),
        tools=("bash",), timeout_sec=30, artifact_prefix="D/round-1", session_id="D-session"))
    assert json.loads(result.final_assistant_text)["done"] is True
    assert (tmp_path / "host/D/round-1/controller-prompt.md").read_text() == prompt
    assert "'# Review note: D_SCOPE'" in (tmp_path / "host/D/round-1/prompt.md").read_text()


class FakeEnvironment:
    def __init__(self, existing=()):
        self.commands, self.existing = [], set(existing)

    async def exec(self, command, cwd=None, **kwargs):
        self.commands.append((command, cwd))
        if command.startswith("test -e "):
            path = command.split(" ", 2)[2].strip("'")
            return SimpleNamespace(return_code=0 if path in self.existing else 1, stdout="", stderr="")
        if command == "git count-objects -v":
            return SimpleNamespace(return_code=0, stdout="count: 3\nsize: 12\n", stderr="")
        return SimpleNamespace(return_code=0, stdout="", stderr="")


def make_workspace(environment, tmp_path):
    workspace = object.__new__(MiniV65WorkspaceManager)
    workspace.environment = environment
    workspace.repository = "/app"
    workspace.transfer_dir = tmp_path / ".transfer"
    workspace.redact = lambda text: text
    return workspace


def test_barrier_removes_raw_p0_inputs_and_prunes_objects(tmp_path):
    patch = b"raw p0"
    from external_agents_eval.workflows.solve_review_repair.artifacts import sha256_bytes

    digest = sha256_bytes(patch)[:16]
    raw = f"/tmp/external_eval/contracts/apply-{digest}.patch"
    environment = FakeEnvironment(existing={raw})
    workspace = make_workspace(environment, tmp_path)
    removed = asyncio.run(workspace.remove_candidate_inputs(candidate_patch=patch))
    assert removed == [raw]
    assert any(command.startswith("rm -f -- ") and raw in command for command, _ in environment.commands)
    pruned = asyncio.run(workspace.prune_unreachable_objects())
    run = [command for command, _ in environment.commands]
    assert "git prune --expire=now" in run and pruned["before"]["count"] == "3"
    assert all(cwd == "/app" for command, cwd in environment.commands if command.startswith("git "))
