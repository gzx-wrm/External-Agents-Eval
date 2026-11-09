"""V6.5 cross-model review: GPT-6.1 Sol reviewers, DeepSeek Repair (no network)."""
from __future__ import annotations

import asyncio
import copy
import importlib.util
import json
import shlex
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from test_mini_v63 import LocalBridge, narrative, state

from external_agents_eval.agents.mini_v65_gpt import (
    DEEPSEEK_ROLES,
    GPT_PARAMETERS,
    MiniV65GptReviewAgent,
    MiniV65GptReviewRunner,
    ResponsesChatBridge,
    V65GptConfig,
    initial_endpoint_for,
)
from external_agents_eval.workflows.solve_review_repair.models import RoleName
from external_agents_eval.workflows.solve_review_repair.v65_controller import V65Role
from external_agents_eval.workflows.solve_review_repair.v65_prompts import disambiguation_prompt

PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture
def endpoints(tmp_path):
    path = tmp_path / "private.json"
    path.write_text(json.dumps([
        {"id": "endpoint1", "base_url": "https://one.invalid/v1", "api_key": "test-one-secret"},
        {"id": "endpoint2", "base_url": "https://two.invalid/v1", "api_key": "test-two-secret"},
    ]))
    return path


def make_runner(tmp_path, endpoints):
    return MiniV65GptReviewRunner(
        environment=LocalBridge(), logs_root=tmp_path / "logs", model="deepseek-flash",
        provider="deepseek", agent_version="2.4.6", env={}, redact=lambda text: text,
        endpoints_file=endpoints)


def response(number, command):
    raw = {"id": "resp_" + str(number), "model": "gpt-6.1-sol", "status": "completed",
           "object": "response", "output": [
               {"type": "reasoning", "id": "rs_" + str(number), "encrypted_content": "opaque-state",
                "summary": [], "status": "completed", "content": None},
               {"type": "function_call", "name": "bash", "call_id": "call_" + str(number),
                "arguments": json.dumps({"command": command})}],
           "usage": {"input_tokens": 120, "output_tokens": 20, "total_tokens": 140,
                     "input_tokens_details": {"cached_tokens": 80}}}
    return SimpleNamespace(**{**raw, "usage": SimpleNamespace(model_dump=lambda **_: raw["usage"])},
                           model_dump=lambda **_: copy.deepcopy(raw))


def test_only_solver_and_repair_stay_on_deepseek():
    assert DEEPSEEK_ROLES == {RoleName.REPAIR, RoleName.SOLVER}
    for role in (RoleName.V5_IMPLEMENTATION_REVIEWER, RoleName.V5_REGRESSION_REVIEWER,
                 RoleName.V5_SUMMARY_REVIEWER, RoleName.JSON_CORRECTOR,
                 V65Role.DISAMBIGUATION_REVIEWER, V65Role.ASSUMPTION_EXTRACTOR):
        assert MiniV65GptReviewRunner.uses_gpt(role)
    assert not MiniV65GptReviewRunner.uses_gpt(RoleName.REPAIR)
    assert GPT_PARAMETERS["reasoning"] == {"effort": "xhigh"}
    assert GPT_PARAMETERS["max_output_tokens"] == 65536


def test_model_routing(tmp_path, endpoints):
    pytest.importorskip("minisweagent")
    from minisweagent.models.litellm_model import LitellmModel

    runner = make_runner(tmp_path, endpoints)
    kwargs = {"model_name": "deepseek/deepseek-flash", "cost_tracking": "ignore_errors",
              "model_kwargs": {"reasoning_effort": "high"}}
    repair = runner._make_model(LitellmModel, role=RoleName.REPAIR, host=tmp_path, **dict(kwargs))
    assert not isinstance(repair, ResponsesChatBridge)
    assert repair.config.model_name == "deepseek/deepseek-flash"
    review = runner._make_model(LitellmModel, role=V65Role.DISAMBIGUATION_REVIEWER, host=tmp_path,
                                **dict(kwargs))
    assert isinstance(review, ResponsesChatBridge)
    assert review.config.model_name == "openai/gpt-6.1-sol"
    assert review.config.model_kwargs == GPT_PARAMETERS
    assert review.responses_transport.config.endpoints_file == endpoints


def d_scope():
    fields = {"scope_complete": True, "new_targets": [], "revisit_target_ids": []}
    return "# Review note: D_SCOPE\n\n## Scope\n```json\n" + json.dumps(fields) + "\n```\n" + narrative()


def test_d_lane_runs_on_gpt_and_records_the_serving_model(tmp_path, endpoints):
    pytest.importorskip("minisweagent")
    from external_agents_eval.agents.mini_v65_gpt import _routed_model_class

    RoutedResponseModel = _routed_model_class()
    root = tmp_path / "attempt"
    root.mkdir()
    prompt = disambiguation_prompt(
        task={"title": "T", "description": "Add behavior"}, repository={},
        ledger={"obligations": [], "preservation_obligations": [], "open_questions": []},
        state=state(root, pending=("Q001",)), role="D", skill_path="skill.md",
        output_root=str(root), initial=True, managed_delivery=[],
        limits={"batch_size": 3, "time_budget_sec": 2700})
    assessment = {"target_id": "Q001", "status": "disputed", "expected": "Readings differ.",
                  "observed": "No observable difference.", "reason": "Class (a) silent.", "code_refs": []}
    note = "# Review note: Q001\n\n## Assessment\n```json\n" + json.dumps(assessment) + "\n```\n" + narrative()
    command = ('mkdir -p "$REVIEW_OUTPUT_ROOT/notes" "$REVIEW_OUTPUT_ROOT/logs"\n'
               "printf '%s' " + shlex.quote(note) + ' > "$REVIEW_OUTPUT_ROOT/notes/Q001.md"\n'
               "printf '%s' " + shlex.quote(d_scope()) + ' > "$REVIEW_OUTPUT_ROOT/logs/review-scope.md"')
    calls = []

    def request(transport, endpoint, **kwargs):
        calls.append(endpoint["id"])
        return response(len(calls), command)

    runner = make_runner(tmp_path, endpoints)
    with patch.object(RoutedResponseModel, "_request", request):
        result = asyncio.run(runner.run_role(
            role=V65Role.DISAMBIGUATION_REVIEWER, prompt=prompt, worktree=str(tmp_path),
            tools=("bash",), timeout_sec=60, artifact_prefix="v5/reviewer/D/round-0001/attempt-1",
            session_id="D-session"))
    assert result.exit_code == 0 and json.loads(result.final_assistant_text)["done"] is True
    assert calls == ["endpoint1"]
    host = tmp_path / "logs/v5/reviewer/D/round-0001/attempt-1"
    identity = json.loads((host / "mini-invocation.json").read_text())
    assert identity["model"]["model"] == "gpt-6.1-sol"
    assert identity["model"]["parameters"] == GPT_PARAMETERS
    assert json.loads((host / "trajectory.json").read_text())["agent"]["model_name"] == "gpt-6.1-sol"
    routes = [json.loads(line) for line in (host / "endpoint-routes.jsonl").read_text().splitlines()]
    assert routes[-1]["status"] == "success" and routes[-1]["endpoint"] == "endpoint1"
    assert "test-one-secret" not in (host / "mini.json").read_text()


def test_repair_runs_on_deepseek_through_the_same_runner(tmp_path, endpoints, monkeypatch):
    from test_mini_v63 import install_stub_model

    install_stub_model(monkeypatch, lambda prompt: ["echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\necho done"])
    runner = make_runner(tmp_path, endpoints)
    result = asyncio.run(runner.run_role(
        role=RoleName.REPAIR, prompt="Repair request", worktree=str(tmp_path), tools=("bash",),
        timeout_sec=60, artifact_prefix="v5/repair"))
    assert result.exit_code == 0
    identity = json.loads((tmp_path / "logs/v5/repair/mini-invocation.json").read_text())
    assert identity["model"]["provider"] == "deepseek" and "transport" not in identity["model"]
    assert not (tmp_path / "logs/v5/repair/endpoint-routes.jsonl").exists()


def test_agent_wiring_and_redaction(endpoints):
    assert MiniV65GptReviewAgent.workflow_config_type is V65GptConfig
    assert V65GptConfig().workflow_controller_version == "v6.5-gpt-review"
    assert V65GptConfig().prompt_version == "v5-7"
    agent = object.__new__(MiniV65GptReviewAgent)
    agent.endpoints_file = endpoints
    agent.extra_env = {}
    agent.stage_model_config = SimpleNamespace(required_env_keys=lambda: ())
    assert agent.redact("key test-two-secret here") == "key [REDACTED] here"
    assert {initial_endpoint_for(f"task-{i}") for i in range(20)} == {0, 1}


def load_launcher():
    spec = importlib.util.spec_from_file_location("v65_gpt_launcher",
                                                  PROJECT / "scripts/deepswe_v65_gpt/run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_launcher_overrides_only_the_review_agent(tmp_path, endpoints, monkeypatch):
    launcher = load_launcher()
    base = launcher.base
    assert base.EXPERIMENT_ID == "v6.5-gpt-review"
    assert base.PILOT_EXPERIMENT_ID == "v6.5-gpt-review-pilot"
    for name in ("scripts/deepswe_v65_gpt/run.py", "external_agents_eval/agents/mini_v65_gpt.py",
                 "scripts/deepswe_ft_m1_gpt/routed_model.py", "scripts/deepswe_v65/run.py"):
        assert name in base.FROZEN_SOURCES and (PROJECT / name).is_file()
    assert base.V65Config is V65GptConfig
    monkeypatch.setenv("GPT_REVIEW_ENDPOINTS_FILE", str(endpoints))
    seen = {}

    class Recorder:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(launcher, "MiniV65GptReviewAgent", Recorder)
    launcher.review_agent(logs_dir=tmp_path / "outputs/meriyah/review/agent/logs", build_gate=object())
    assert seen["endpoints_file"] == endpoints
    assert seen["initial_endpoint"] == initial_endpoint_for("meriyah")
    path = tmp_path / "result.json"
    launcher.save(path, {"error": "upstream echoed test-one-secret"})
    assert "test-one-secret" not in path.read_text() and "[REDACTED]" in path.read_text()
