"""V6.4 Mini backend: role-specific system prompts and the single file contract."""
from __future__ import annotations

import asyncio
import json

import pytest
from test_mini_v63 import LocalBridge, install_stub_model, record, scope, state, write_files

from external_agents_eval.agents.mini_review_files import PROTOCOL, SCOPE_PATH
from external_agents_eval.agents.mini_v62 import MINI_SYSTEM
from external_agents_eval.agents.mini_v63 import REVIEW_SYSTEM
from external_agents_eval.agents.mini_v64 import (
    REPAIR_SYSTEM_V64,
    REVIEW_SYSTEM_V64,
    SUMMARY_SYSTEM_V64,
    MiniV64ReviewAgent,
    MiniV64ReviewRunner,
)
from external_agents_eval.workflows.solve_review_repair.models import RoleName
from external_agents_eval.workflows.solve_review_repair.v5_prompts import reviewer_prompt
from external_agents_eval.workflows.solve_review_repair.v64_config import V64Config


def v56_prompt(s, role="R"):
    return reviewer_prompt(
        task={"title": "Task", "description": "Add behavior"}, repository={},
        ledger={"obligations": [], "preservation_obligations": [], "open_questions": []},
        state=s, role=role, skill_path="skill.md", output_root=s["review_notes"]["output_root"],
        initial=True, limits={"batch_size": 3, "time_budget_sec": 2700}, prompt_version="v5-6",
    )


def test_role_specific_system_prompts_without_a_stage_text():
    runner = object.__new__(MiniV64ReviewRunner)
    review = runner._mini_system(role=RoleName.V5_IMPLEMENTATION_REVIEWER, tools=("bash",), schemas=None)
    repair = runner._mini_system(role=RoleName.REPAIR, tools=("bash",), schemas=None)
    summary = runner._mini_system(role=RoleName.V5_SUMMARY_REVIEWER, tools=("read",), schemas=None)
    assert review == REVIEW_SYSTEM_V64 and review != REVIEW_SYSTEM
    assert repair == REPAIR_SYSTEM_V64.replace("{baseline}", "/app") and "/app is the original" in repair
    assert summary == SUMMARY_SYSTEM_V64
    for text in (review, repair, summary):
        assert "During A exploration" not in text and "{baseline}" not in text
    assert "Edit and run tests" not in summary and "do not run new tests" in summary
    # S keeps both submission paths of the original Mini protocol.
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in summary and "submit_json '<complete JSON>'" in summary
    assert "printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\n'; cat /tmp/summary.json" in summary
    for text in (repair, summary):
        assert "exactly one Bash tool call; never answer in plain\ntext" in text or \
            "exactly one Bash tool call; never answer in plain text" in text.replace("\n", " ")
    assert "submit_json" not in repair and "JSON" not in repair
    # Other roles keep the inherited text (with the DeepSWE baseline path).
    corrector = runner._mini_system(role=RoleName.JSON_CORRECTOR, tools=(), schemas=None)
    assert corrector.startswith(MINI_SYSTEM.replace("/testbed", "/app"))


def test_agent_requires_gate_and_uses_v64_config():
    with pytest.raises(ValueError, match="build gate"):
        MiniV64ReviewAgent(logs_dir=None)
    assert MiniV64ReviewAgent.workflow_config_type is V64Config
    assert MiniV64ReviewAgent.workflow_version == "6.4"


def test_real_mini_loop_uses_the_v64_file_contract(tmp_path, monkeypatch):
    root = tmp_path / "current"
    root.mkdir()
    commands = [write_files({"notes/P001.md": record("P001"), SCOPE_PATH: scope()})]
    seen = {}

    def commands_for(prompt):
        seen["prompt"] = prompt
        return commands

    install_stub_model(monkeypatch, commands_for)
    runner = MiniV64ReviewRunner(environment=LocalBridge(), logs_root=tmp_path / "host",
                                 model="deepseek-flash", provider="deepseek", agent_version="2.4.6",
                                 env={}, redact=lambda value: value)
    prompt = v56_prompt(state(root))
    result = asyncio.run(runner.run_role(
        role=RoleName.V5_REGRESSION_REVIEWER, prompt=prompt, worktree=str(tmp_path),
        tools=("bash",), timeout_sec=30, artifact_prefix="R/round-1", session_id="R-session",
        invocation_id="v64.R.1",
    ))
    assert result.exit_code == 0 and json.loads(result.final_assistant_text)["done"] is True
    sent = (tmp_path / "host/R/round-1/prompt.md").read_text()
    assert "OUTPUT CONTRACT (files)" in sent and "V6.3 FILE DELIVERY" not in sent
    assert (tmp_path / "host/R/round-1/controller-prompt.md").read_text() == prompt
    native = json.loads(result.transcript_path.read_text())
    assert native["messages"][0]["content"].startswith("You are a coding reviewer")
    assert "OUTPUT CONTRACT (files)" in native["messages"][0]["content"]
    assert native["messages"][-1]["extra"]["review_delivery"] == PROTOCOL
