"""V6.6 A: the V6.2 A path with an evidence plan, followed by check-point stage C."""
from __future__ import annotations

import dataclasses
import json
import re
import time
from typing import Any

from .models import RoleName, RoleRunResult
from .v62_controller import V62Controller
from .v66_checkpoints import CheckpointClient
from .v66_config import V66Config
from .v66_prompts import evidence_plan_continuation, evidence_plan_prompt

EVIDENCE_SEARCH_PREFIX = "v6.2/evidence-search"
CONTINUATION_PREFIX = "v6.6/evidence-continuation"
INFRASTRUCTURE_MARKER = "A_INFRASTRUCTURE"
TEST_PATH = re.compile(
    r"(^|/)(tests?|__tests__|specs?|testing)/|(_test|\.test|\.spec|_spec)\.[^/]+$"
    r"|(^|/)test_[^/]+$|(^|/)[^/]*Tests?\.[^/]+$")


class PiApiFailure(RuntimeError):
    """A Pi session that never finished normally: an infrastructure, not a model, outcome."""


def pi_completion(transcript: str) -> dict[str, Any]:
    """The last assistant stop reason and whether any assistant turn ended normally."""
    last, normal = None, False
    for line in transcript.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        message = event.get("message")
        if event.get("type") in {"message_end", "turn_end"} and isinstance(message, dict) \
                and message.get("role") == "assistant":
            last = message.get("stopReason")
            normal = normal or last == "stop"
    return {"last_stop_reason": last, "normal_completion": normal}


def require_pi_completed(result: RoleRunResult, label: str) -> dict[str, Any]:
    status = pi_completion(result.transcript_path.read_text(encoding="utf-8")
                           if result.transcript_path.exists() else "")
    if status["last_stop_reason"] == "error" or not status["normal_completion"]:
        raise PiApiFailure(
            f"{INFRASTRUCTURE_MARKER}: {label} ended without a normal completion "
            f"(last stopReason={status['last_stop_reason']})")
    return status


def evidence_plan_gaps(trajectory: dict[str, Any], *, base_root: str) -> list[str]:
    """Mechanical proxies for the evidence plan: a search call and a test-file read."""
    prefix = base_root.rstrip("/") + "/"
    calls = [call for step in trajectory.get("steps", []) for call in step.get("tool_calls") or []]
    reads = [call["arguments"].get("path", "") for call in calls if call["function_name"] == "read"]
    missing = []
    if not any(call["function_name"] in {"grep", "find"} for call in calls):
        missing.append("search")
    if not any(path.startswith(prefix) and TEST_PATH.search(path[len(prefix):]) for path in reads):
        missing.append("test_read")
    return missing


def merge_trajectories(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """Append the continuation's new steps; tool calls already seen are not repeated."""
    seen = {call["tool_call_id"] for step in first.get("steps", []) for call in step.get("tool_calls") or []}
    steps = [dict(step) for step in first.get("steps", [])]
    for step in second.get("steps", []):
        calls = step.get("tool_calls") or []
        if calls and all(call["tool_call_id"] in seen for call in calls):
            continue
        steps.append(dict(step))
    for number, step in enumerate(steps, 1):
        step["step_id"] = number
    return {**first, "steps": steps, "v66_merged_from": ["evidence-search", "evidence-continuation"]}


class V66Controller(V62Controller):
    def __init__(self, *, ledger_client: Any, checkpoint_client: CheckpointClient, **kwargs: Any) -> None:
        kwargs.setdefault("config", V66Config())
        super().__init__(ledger_client=ledger_client, **kwargs)
        self.checkpoint_client = checkpoint_client

    async def _run_role(self, *, role: RoleName, prompt: str, worktree: str, prefix: str,
                        **kwargs: Any) -> RoleRunResult:
        # Only the A evidence search changes: the historical prompt plus the evidence plan.
        if not (role == RoleName.V5_SPECIFICATION_REVIEWER and prefix == EVIDENCE_SEARCH_PREFIX):
            return await super()._run_role(role=role, prompt=prompt, worktree=worktree, prefix=prefix,
                                           **kwargs)
        started = time.monotonic()
        result = await super()._run_role(role=role, prompt=evidence_plan_prompt(prompt),
                                         worktree=worktree, prefix=prefix, **kwargs)
        check: dict[str, Any] = {"search": require_pi_completed(result, "A evidence search")}
        trajectory = json.loads(result.trajectory_path.read_text(encoding="utf-8"))
        check["initial_missing"] = evidence_plan_gaps(trajectory, base_root=worktree)
        check["continuations"] = 0
        if check["initial_missing"] and self.config.evidence_plan_continuations > 0:
            self._require_success(result, "V6.2 evidence search")
            continuation_kwargs = dict(kwargs)
            continuation_kwargs["invocation_id"] = f"{kwargs.get('invocation_id') or 'A'}.plan"
            if kwargs.get("timeout_sec") is not None:
                continuation_kwargs["timeout_sec"] = max(
                    300.0, float(kwargs["timeout_sec"]) - (time.monotonic() - started))
            continuation = await super()._run_role(
                role=role, prompt=evidence_plan_continuation(check["initial_missing"]),
                worktree=worktree, prefix=CONTINUATION_PREFIX, **continuation_kwargs)
            check["continuation"] = require_pi_completed(continuation, "A evidence continuation")
            self._require_success(continuation, "A evidence continuation")
            merged = merge_trajectories(
                trajectory, json.loads(continuation.trajectory_path.read_text(encoding="utf-8")))
            path = self.artifacts.write_json("v6.6/evidence-search-merged/trajectory.json", merged)
            check.update(continuations=1,
                         remaining_missing=evidence_plan_gaps(merged, base_root=worktree))
            result = dataclasses.replace(result, trajectory_path=path)
        self.artifacts.write_json("v6.6/evidence-plan-check.json", check)
        return result

    async def prepare_checkpoints(self, ledger: dict[str, Any]) -> dict[str, Any]:
        """Stage C on the frozen evidence packet; FT's ledger is never rewritten."""
        packet = json.loads((self.artifacts.root / "v6.2/evidence.json").read_text(encoding="utf-8"))
        task = packet["task"]
        # Same task-field order that the FT request uses.
        packet.update(task={"description": task["description"], "title": task["title"]})
        return await self.checkpoint_client.generate(packet=packet, ledger=ledger,
                                                     artifacts=self.artifacts)
