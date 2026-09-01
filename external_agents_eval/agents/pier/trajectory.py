from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any


def _text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        values: list[str] = []
        for part in content:
            if isinstance(part, str):
                values.append(part)
            elif isinstance(part, dict):
                value = part.get("text", part.get("content", part.get("thinking")))
                if value is not None:
                    values.append(str(value))
        return "\n".join(value for value in values if value)
    return str(content)


def _arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"value": value}
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    return {"value": value}


def _number(value: Any) -> int | float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    return None


def _metrics_from_usage(usage: Any) -> dict[str, Any] | None:
    if not isinstance(usage, dict):
        return None
    prompt = _number(usage.get("input", usage.get("prompt_tokens")))
    completion = _number(usage.get("output", usage.get("completion_tokens")))
    cached = _number(usage.get("cacheRead", usage.get("cached_tokens")))
    cost_value = usage.get("cost")
    if isinstance(cost_value, dict):
        cost_value = cost_value.get("total")
    cost = _number(cost_value)
    metrics = {
        "prompt_tokens": int(prompt) if prompt is not None else None,
        "completion_tokens": int(completion) if completion is not None else None,
        "cached_tokens": int(cached) if cached is not None else None,
        "cost_usd": float(cost) if cost is not None else None,
    }
    return {key: value for key, value in metrics.items() if value is not None} or None


def _base_trajectory(
    *,
    instruction: str,
    agent_name: str,
    agent_version: str,
    model: str,
    session_id: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": "ATIF-v1.7",
        "session_id": session_id or str(uuid.uuid4()),
        "trajectory_id": str(uuid.uuid4()),
        "agent": {
            "name": agent_name,
            "version": agent_version,
            "model_name": model,
        },
        "steps": [
            {
                "step_id": 1,
                "source": "user",
                "message": instruction,
            }
        ],
    }


def _append_agent_step(
    trajectory: dict[str, Any],
    *,
    model: str,
    message: str,
    reasoning: str | None,
    tool_calls: list[dict[str, Any]] | None,
    metrics: dict[str, Any] | None,
    timestamp: str | None,
) -> dict[str, Any]:
    step: dict[str, Any] = {
        "step_id": len(trajectory["steps"]) + 1,
        "source": "agent",
        "model_name": model,
        "message": message,
        "llm_call_count": 1,
    }
    if timestamp:
        step["timestamp"] = timestamp
    if reasoning:
        step["reasoning_content"] = reasoning
    if tool_calls:
        step["tool_calls"] = tool_calls
    if metrics:
        step["metrics"] = metrics
    trajectory["steps"].append(step)
    return step


def _attach_observation(step: dict[str, Any], *, source_call_id: str | None, content: str) -> None:
    result: dict[str, Any] = {"content": content}
    if source_call_id:
        result["source_call_id"] = source_call_id
    observation = step.setdefault("observation", {"results": []})
    observation["results"].append(result)


def _finish_metrics(trajectory: dict[str, Any]) -> None:
    prompt = completion = cached = 0
    cost = 0.0
    saw_prompt = saw_completion = saw_cached = saw_cost = False
    agent_steps = 0
    for step in trajectory["steps"]:
        if step["source"] != "agent":
            continue
        agent_steps += 1
        metrics = step.get("metrics") or {}
        if "prompt_tokens" in metrics:
            prompt += metrics["prompt_tokens"]
            saw_prompt = True
        if "completion_tokens" in metrics:
            completion += metrics["completion_tokens"]
            saw_completion = True
        if "cached_tokens" in metrics:
            cached += metrics["cached_tokens"]
            saw_cached = True
        if "cost_usd" in metrics:
            cost += metrics["cost_usd"]
            saw_cost = True
    if not agent_steps:
        _append_agent_step(
            trajectory,
            model=trajectory["agent"]["model_name"],
            message="Agent produced no parseable transcript.",
            reasoning=None,
            tool_calls=None,
            metrics=None,
            timestamp=None,
        )
        agent_steps = 1
    final_metrics: dict[str, Any] = {
        "total_steps": len(trajectory["steps"]),
        "extra": {"agent_steps": agent_steps},
    }
    if saw_prompt:
        final_metrics["total_prompt_tokens"] = prompt
    if saw_completion:
        final_metrics["total_completion_tokens"] = completion
    if saw_cached:
        final_metrics["total_cached_tokens"] = cached
    if saw_cost:
        final_metrics["total_cost_usd"] = cost
    trajectory["final_metrics"] = final_metrics


def pi_jsonl_to_atif(
    transcript: str,
    *,
    instruction: str,
    model: str,
    agent_version: str,
) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    for line in transcript.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    session_id = next(
        (
            str(event["id"])
            for event in events
            if event.get("type") == "session" and event.get("id")
        ),
        None,
    )
    trajectory = _base_trajectory(
        instruction=instruction,
        agent_name="pi",
        agent_version=agent_version,
        model=model,
        session_id=session_id,
    )
    calls: dict[str, dict[str, Any]] = {}
    for event in events:
        if event.get("type") != "message_end":
            continue
        message = event.get("message")
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "assistant":
            text_parts: list[str] = []
            reasoning_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for part in message.get("content") or []:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type == "text" and part.get("text"):
                    text_parts.append(str(part["text"]))
                elif part_type == "thinking" and part.get("thinking"):
                    reasoning_parts.append(str(part["thinking"]))
                elif part_type == "toolCall":
                    call_id = str(part.get("id") or f"call-{uuid.uuid4()}")
                    tool_calls.append(
                        {
                            "tool_call_id": call_id,
                            "function_name": str(part.get("name") or "tool"),
                            "arguments": _arguments(part.get("arguments") or {}),
                        }
                    )
            timestamp = message.get("timestamp")
            if isinstance(timestamp, (int, float)):
                timestamp = datetime.fromtimestamp(
                    timestamp / 1000,
                    timezone.utc,
                ).isoformat()
            elif not isinstance(timestamp, str):
                timestamp = None
            step = _append_agent_step(
                trajectory,
                model=str(message.get("model") or model),
                message="\n".join(text_parts),
                reasoning="\n".join(reasoning_parts) or None,
                tool_calls=tool_calls or None,
                metrics=_metrics_from_usage(message.get("usage")),
                timestamp=timestamp,
            )
            for call in tool_calls:
                calls[call["tool_call_id"]] = step
        elif role == "toolResult":
            call_id = message.get("toolCallId")
            step = calls.get(str(call_id)) if call_id is not None else None
            if step is not None:
                _attach_observation(
                    step,
                    source_call_id=str(call_id),
                    content=_text(message.get("content")),
                )
    _finish_metrics(trajectory)
    return trajectory


def hermes_json_to_atif(
    payload: dict[str, Any],
    *,
    instruction: str,
    model: str,
    agent_version: str,
) -> dict[str, Any]:
    trajectory = _base_trajectory(
        instruction=instruction,
        agent_name="hermes",
        agent_version=agent_version,
        model=model,
        session_id=None,
    )
    calls: dict[str, dict[str, Any]] = {}
    for message in payload.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "assistant":
            tool_calls: list[dict[str, Any]] = []
            for raw_call in message.get("tool_calls") or []:
                if not isinstance(raw_call, dict):
                    continue
                function = raw_call.get("function") or {}
                call_id = str(
                    raw_call.get("id") or raw_call.get("call_id") or f"call-{uuid.uuid4()}"
                )
                tool_calls.append(
                    {
                        "tool_call_id": call_id,
                        "function_name": str(
                            function.get("name") or raw_call.get("name") or "tool"
                        ),
                        "arguments": _arguments(
                            function.get("arguments", raw_call.get("arguments", {}))
                        ),
                    }
                )
            timestamp = message.get("timestamp")
            if isinstance(timestamp, (int, float)):
                timestamp = datetime.fromtimestamp(
                    timestamp,
                    timezone.utc,
                ).isoformat()
            elif not isinstance(timestamp, str):
                timestamp = None
            step = _append_agent_step(
                trajectory,
                model=model,
                message=_text(message.get("content")),
                reasoning=_text(message.get("reasoning_content", message.get("reasoning"))) or None,
                tool_calls=tool_calls or None,
                metrics=_metrics_from_usage(message.get("usage")),
                timestamp=timestamp,
            )
            for call in tool_calls:
                calls[call["tool_call_id"]] = step
        elif role in {"tool", "toolResult"}:
            call_id = message.get("tool_call_id", message.get("toolCallId"))
            step = calls.get(str(call_id)) if call_id is not None else None
            if step is not None:
                _attach_observation(
                    step,
                    source_call_id=str(call_id),
                    content=_text(message.get("content")),
                )
    if len(trajectory["steps"]) == 1 and payload.get("final_response"):
        _append_agent_step(
            trajectory,
            model=model,
            message=str(payload["final_response"]),
            reasoning=None,
            tool_calls=None,
            metrics=None,
            timestamp=None,
        )
    trajectory["extra"] = {
        key: payload.get(key)
        for key in ("completed", "partial", "failed", "error", "api_calls")
        if key in payload
    }
    _finish_metrics(trajectory)
    return trajectory
