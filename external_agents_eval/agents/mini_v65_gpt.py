"""V6.5 with a cross-model review: GPT-6.1 Sol reviews, DeepSeek solves and repairs.

Every review role (X, D, I, R, S and the JSON format corrector) runs on GPT-6.1 Sol
through the Responses transport of the earlier GPT review experiment
(``gpt_ft_review.py`` and ``scripts/deepswe_ft_m1_gpt/routed_model.py`` in snapshot
/root/ft_m1_gpt61sol_review-code-20261031-r1), ported unchanged. Repair (and the
reused Solver) keeps the DeepSeek V4.1 Flash call of V6.5. Prompts, the file
contract, the V6.5 controller, budgets and the build gate are those of V6.5.

``_run_mini`` repeats ``MiniRoleRunner._run_mini`` with one model hook and records
the model that actually serves each role, so V6.4/V6.5 code stays untouched.
"""
from __future__ import annotations

import hashlib
import json
import sys
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from external_agents_eval.agents.mini_v62 import (
    PI_EVIDENCE_PROTOCOL,
    PROTOCOL,
    close_tool_calls,
    load_tool_schemas,
    model_visible_events,
    parse_pi_actions,
)
from external_agents_eval.agents.mini_v65 import MiniV65ReviewAgent, MiniV65ReviewRunner
from external_agents_eval.workflows.solve_review_repair.artifacts import sha256_bytes, sha256_json
from external_agents_eval.workflows.solve_review_repair.models import (
    IndeterminateInvocationError,
    RoleName,
    RoleRunResult,
)
from external_agents_eval.workflows.solve_review_repair.v65_config import V65Config

PROJECT = Path(__file__).resolve().parents[2]
ROUTED_MODEL_DIR = PROJECT / "scripts/deepswe_ft_m1_gpt"

REVIEW_MODEL = "openai/gpt-6.1-sol"
# Identical to the approved GPT-6.1 Sol runs (Solver and the earlier review).
GPT_PARAMETERS = {"reasoning": {"effort": "xhigh"}, "max_output_tokens": 65536,
                  "store": False, "include": ["reasoning.encrypted_content"]}
GPT_METADATA = {"provider": "openai", "model": "gpt-6.1-sol", "base_url": None,
                "effective_base_url": "host-private routed endpoints (2)",
                "transport": "responses-routed", "parameters": GPT_PARAMETERS}
DEEPSEEK_ROLES = frozenset({RoleName.REPAIR, RoleName.SOLVER})


@dataclass(frozen=True)
class V65GptConfig(V65Config):
    workflow_controller_version: str = "v6.5-gpt-review"


def _routed_model_class():
    if str(ROUTED_MODEL_DIR) not in sys.path:
        sys.path.insert(0, str(ROUTED_MODEL_DIR))
    from routed_model import RoutedResponseModel

    return RoutedResponseModel


class ResponsesChatBridge:
    """Keep Mini's Chat-shaped actions/session records and all raw Responses items."""

    def _prepare_messages_for_api(self, messages):
        prepared = []
        for message in messages:
            role = message.get("role")
            if role == "assistant":
                raw = message.get("extra", {}).get("response", {}).get("native_responses")
                if raw is None:
                    raise ValueError("Assistant history is missing its raw Responses output")
                prepared.extend(raw["output"])
            elif role == "tool":
                prepared.append({"type": "function_call_output",
                                 "call_id": message["tool_call_id"],
                                 "output": message["content"]})
            elif role in {"system", "user"}:
                prepared.append({"role": role, "content": message["content"]})
            else:
                raise ValueError("Unexpected message in Reviewer session: " + str(role))
        return prepared

    def _query(self, messages, **kwargs):
        from litellm import ModelResponse

        response = self.responses_transport._query(messages)
        raw = response.model_dump(mode="json")
        calls, texts = [], []
        for item in raw["output"]:
            if item["type"] == "function_call":
                calls.append({"id": item["call_id"], "type": "function",
                              "function": {"name": item["name"], "arguments": item["arguments"]}})
            elif item["type"] == "message":
                texts.extend(part["text"] for part in item.get("content", [])
                             if part.get("type") == "output_text")
            elif item["type"] != "reasoning":
                raise ValueError("Unsupported model output item: " + item["type"])
        usage = raw.get("usage") or {}
        details = usage.get("input_tokens_details") or {}
        converted = ModelResponse(
            model=raw["model"], id=raw["id"],
            choices=[{"index": 0, "finish_reason": "tool_calls" if calls else "stop",
                      "message": {"role": "assistant", "content": "\n".join(texts),
                                  "tool_calls": calls}}],
            usage={"prompt_tokens": usage.get("input_tokens", 0),
                   "completion_tokens": usage.get("output_tokens", 0),
                   "total_tokens": usage.get("total_tokens", 0),
                   "prompt_tokens_details": {"cached_tokens": details.get("cached_tokens", 0)}},
            native_responses=raw,
        )
        self.endpoint_state.write_text(json.dumps({"endpoint": self.responses_transport._current}))
        return converted


class MiniV65GptReviewRunner(MiniV65ReviewRunner):
    def __init__(self, *args, endpoints_file, initial_endpoint=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.endpoints_file = Path(endpoints_file)
        self.initial_endpoint = initial_endpoint
        self._invocation = threading.local()

    @staticmethod
    def uses_gpt(role) -> bool:
        return role not in DEEPSEEK_ROLES

    def _make_model(self, model_type, *, role, host, **kwargs):
        if not self.uses_gpt(role):
            return model_type(**kwargs)

        class Model(ResponsesChatBridge, model_type):
            pass

        kwargs.update(model_name=REVIEW_MODEL, model_kwargs=dict(GPT_PARAMETERS))
        model = Model(**kwargs)
        state = self.logs_root / "gpt-endpoint-state" / (getattr(role, "value", str(role)) + ".json")
        state.parent.mkdir(parents=True, exist_ok=True)
        endpoint = json.loads(state.read_text())["endpoint"] if state.exists() else self.initial_endpoint
        model.responses_transport = _routed_model_class()(
            model_name=REVIEW_MODEL, cost_tracking="ignore_errors",
            endpoints_file=self.endpoints_file, route_log=host / "endpoint-routes.jsonl",
            initial_endpoint=endpoint, model_kwargs=dict(GPT_PARAMETERS),
        )
        model.endpoint_state = state
        return model

    def _run_mini(self, role, prompt, worktree, timeout_sec, artifact_prefix, session_id, tools, pi_runtime=None):
        from minisweagent.models.litellm_model import LitellmModel

        spec = self.stage_model_config.for_role(role)
        gpt = self.uses_gpt(role)
        served = GPT_METADATA if gpt else spec.to_metadata()
        host = self.logs_root / artifact_prefix
        host.mkdir(parents=True, exist_ok=True)
        identity = {"prompt_sha256": sha256_bytes(prompt.encode()), "model": served,
                    "worktree": worktree, "session_id": session_id}
        schemas = load_tool_schemas(pi_runtime) if pi_runtime else None
        if schemas:
            runtime_metadata = json.loads((pi_runtime / "runtime.json").read_text())
            identity.update(tool_protocol=PI_EVIDENCE_PROTOCOL, tool_schemas_sha256=sha256_json(schemas),
                            tool_runtime_sha256=runtime_metadata["archive_sha256"],
                            tool_helper_sha256=sha256_bytes(
                                (PROJECT / "external_agents_eval/agents/pi_file_tools.mjs").read_bytes()))
        request = host / "mini-invocation.json"
        completed = host / "role-result.json"
        if request.exists():
            if json.loads(request.read_text()) != identity:
                raise ValueError("Mini invocation inputs changed; use a new run")
            if not completed.exists():
                raise IndeterminateInvocationError("Mini invocation has no completed result; do not resample")
            saved = json.loads(completed.read_text())
            for key in ("transcript_path", "trajectory_path", "session_file"):
                if saved.get(key):
                    saved[key] = Path(saved[key])
            return RoleRunResult(**saved)
        request.write_text(json.dumps(identity))
        if schemas:
            (host / "tool-schemas.json").write_text(json.dumps(schemas, indent=2))
            (host / "tool-runtime.json").write_text(json.dumps(runtime_metadata, indent=2))
        (host / "prompt.md").write_text(prompt)
        key = sha256_json([session_id or artifact_prefix, served])
        session = self.logs_root / "mini-sessions" / f"{key}.json"
        session.parent.mkdir(exist_ok=True)
        prefix = json.loads(session.read_text()) if session.exists() else []
        pre_hash = sha256_bytes(session.read_bytes()) if session.exists() else None
        progress = {"model_calls": 0, "completed_calls": 0}

        def save_progress(**changes):
            progress.update(changes)
            progress["updated_at"] = datetime.now(timezone.utc).isoformat()
            temporary = host / "progress.json.tmp"
            temporary.write_text(json.dumps(progress))
            temporary.replace(host / "progress.json")

        class Model(LitellmModel):
            def _query(self, messages, **kwargs):
                if schemas is None:
                    return super()._query(messages, **kwargs)
                import litellm

                options = self.config.model_kwargs | kwargs
                options["tools"] = schemas
                return litellm.completion(model=self.config.model_name, messages=messages, **options)

            def _parse_actions(self, response):
                if schemas is None:
                    return super()._parse_actions(response)
                return parse_pi_actions(response, schemas)

            def format_observation_messages(self, message, outputs, template_vars=None):
                if schemas is None:
                    return super().format_observation_messages(message, outputs, template_vars)
                from minisweagent.models.utils.actions_toolcall import (
                    format_toolcall_observation_messages,
                )

                return format_toolcall_observation_messages(
                    actions=message["extra"]["actions"], outputs=outputs,
                    observation_template="{{output.output}}", template_vars=template_vars,
                )

            def _prepare_messages_for_api(self, messages):
                prepared = super()._prepare_messages_for_api(messages)
                if spec.provider == "deepseek" and not gpt:
                    for message in prepared:
                        if message.get("role") == "assistant" and message.get("reasoning_content") == "":
                            message["reasoning_content"] = " "
                return prepared

            def query(self, messages, **kwargs):
                save_progress(model_calls=progress["model_calls"] + 1, state="awaiting_response")
                try:
                    response = super().query(prefix + messages[1:] if prefix else messages, **kwargs)
                except Exception as exception:
                    save_progress(state="query_error", error_type=type(exception).__name__)
                    raise
                usage = (response.get("extra", {}).get("response") or {}).get("usage") or {}
                save_progress(completed_calls=progress["completed_calls"] + 1,
                              state="response_received", last_usage=usage)
                return response

        model = self._make_model(
            Model, role=role, host=host,
            model_name=f"{spec.provider}/{spec.model}", cost_tracking="ignore_errors",
            model_kwargs={"api_base": spec.effective_base_url, "reasoning_effort": "high",
                          "extra_body": {"thinking": {"type": "enabled"}}, "max_tokens": 65536,
                          "timeout": 180, "num_retries": 0},
        )
        env = self._make_mini_environment(
            role=role, prompt=prompt, worktree=worktree, tools=tools, schemas=schemas, host=host,
        )
        system = self._mini_system(role=role, tools=tools, schemas=schemas)
        agent = self._make_mini_agent(model=model, env=env, system=system, timeout_sec=timeout_sec)
        started = datetime.now(timezone.utc)
        status, error = {}, None
        try:
            status = agent.run(prompt)
        except Exception as exception:  # noqa: BLE001 - archive the failed agent invocation
            error = self.redact(str(exception))
        raw = agent.serialize()
        (host / "mini.json").write_text(self.redact(json.dumps(raw, ensure_ascii=False, indent=2)))
        submission = status.get("submission", "")
        current = close_tool_calls(agent.messages, submission)
        full = prefix + current[1:] if prefix else current
        session.write_text(self.redact(json.dumps(full, ensure_ascii=False)))
        totals = {"total_prompt_tokens": 0, "total_completion_tokens": 0, "total_cached_tokens": 0}
        for message in agent.messages:
            usage = (message.get("extra", {}).get("response") or {}).get("usage") or {}
            totals["total_prompt_tokens"] += usage.get("prompt_tokens") or 0
            totals["total_completion_tokens"] += usage.get("completion_tokens") or 0
            totals["total_cached_tokens"] += usage.get("prompt_cache_hit_tokens") or (
                usage.get("prompt_tokens_details") or {}
            ).get("cached_tokens", 0)
        steps = [{"step_id": 1, "source": "user", "message": prompt}]
        visible, unreturned = model_visible_events(env.events, agent.messages)
        for name, arguments, output in visible:
            call_id = f"mini-{len(steps)}"
            steps.append({"step_id": len(steps) + 1, "source": "agent", "message": "",
                          "tool_calls": [{"tool_call_id": call_id, "function_name": name, "arguments": arguments}],
                          "observation": {"results": [{"source_call_id": call_id, "content": output["output"],
                                                       "exit_code": output["returncode"],
                                                       "source_text_returned": output.get("source_text_returned", True)}]}})
        steps.append({"step_id": len(steps) + 1, "source": "agent", "message": submission})
        trajectory = host / "trajectory.json"
        trajectory.write_text(self.redact(json.dumps({
            "schema_version": "ATIF-v1.7", "agent": {"name": "mini-swe-agent", "version": "2.4.6",
                                                       "model_name": served["model"]},
            "steps": steps, "final_metrics": {"total_steps": len(steps), **totals},
            "evidence_protocol": PI_EVIDENCE_PROTOCOL if schemas else PROTOCOL,
            "executed_but_not_returned": unreturned,
            "adapter": "Pi 0.84.2 file tools; raw mini.json is authoritative" if schemas else
                       "native Bash observations; raw mini.json is authoritative",
        }, ensure_ascii=False)))
        result = RoleRunResult(
            exit_code=0 if status.get("exit_status") == "Submitted" else 1,
            final_assistant_text=submission, transcript_path=host / "mini.json", trajectory_path=trajectory,
            started_at=started.isoformat(), finished_at=datetime.now(timezone.utc).isoformat(),
            resource_usage=totals, timed_out=status.get("exit_status") == "TimeExceeded",
            exception=error or (None if status.get("exit_status") == "Submitted" else str(status)),
            session_file=session, pre_session_sha256=pre_hash,
            post_session_sha256=sha256_bytes(session.read_bytes()),
            remote_session_file=f"/external_eval/mini_sessions/{key}.json",
        )
        completed.write_text(self.redact(json.dumps(asdict(result), default=str)))
        return result


def initial_endpoint_for(task_id: str) -> int:
    """Spread concurrent tasks over the two endpoints deterministically."""
    return int(hashlib.sha256(task_id.encode()).hexdigest(), 16) % 2


class MiniV65GptReviewAgent(MiniV65ReviewAgent):
    workflow_version = "6.5"
    workflow_config_type = V65GptConfig

    def __init__(self, *args, endpoints_file, initial_endpoint=0, **kwargs):
        self.endpoints_file = Path(endpoints_file)
        self.initial_endpoint = initial_endpoint
        super().__init__(*args, **kwargs)

    @staticmethod
    def name():
        return "v65-gpt-review"

    def _make_role_runner(self, **kwargs):
        return MiniV65GptReviewRunner(endpoints_file=self.endpoints_file,
                                      initial_endpoint=self.initial_endpoint, **kwargs)

    def redact(self, text):
        text = super().redact(text)
        for endpoint in json.loads(self.endpoints_file.read_text()):
            text = text.replace(endpoint["api_key"], "[REDACTED]")
        return text
