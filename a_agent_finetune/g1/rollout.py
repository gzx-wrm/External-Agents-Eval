"""Sequential online trajectories: exact sampled tokens retained; tool observations masked."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .reward import DeliveryReward
from .tools import RepositoryTools, tool_schemas


@dataclass(frozen=True)
class Budgets:
    max_context_tokens: int = 147456
    max_trajectory_tokens: int = 131072
    max_generated_tokens: int = 16384
    max_turn_tokens: int = 8192
    timeout_sec: float = 1800
    tool_output_chars: int = 12000


@dataclass
class Trajectory:
    prompt_ids: list[int]
    completion_ids: list[int]
    env_mask: list[int]
    messages: list[dict[str, Any]]
    turns: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    final_text: str
    termination: str
    score: dict[str, Any]
    elapsed_sec: float

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


def tool_suffix_ids(tokenizer, tool_messages: list[dict[str, Any]]) -> list[int]:
    """Append only deterministic tool/header tokens; never reserialize sampled tool calls."""
    dummy = [
        {"role": "user", "content": "dummy"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "type": "function", "function": {"name": tool_messages[0]["name"], "arguments": {}},
        }]},
    ]
    kwargs = {"tokenize": True, "return_dict": False, "enable_thinking": False}
    prefix = tokenizer.apply_chat_template(dummy, add_generation_prompt=False, **kwargs)
    full = tokenizer.apply_chat_template(
        dummy + tool_messages, add_generation_prompt=True, **kwargs,
    )
    eos = [i for i, token in enumerate(prefix) if token == tokenizer.eos_token_id]
    if eos:
        prefix = prefix[:eos[-1] + 1]
    if full[:len(prefix)] != prefix:
        raise ValueError("G1 requires a prefix-preserving training chat template")
    return full[len(prefix):]


def run_trajectory(
    *, prompt: list[dict[str, Any]], repository: Path, tokenizer,
    generate: Callable[[list[int], int, float], list[int]],
    parse_response: Callable[[list[int], list[int]], dict[str, Any]],
    reward: DeliveryReward, budgets: Budgets, clock: Callable[[], float] = time.monotonic,
) -> Trajectory:
    started = clock()
    deadline = started + budgets.timeout_sec
    tools = RepositoryTools(repository, budgets.tool_output_chars)
    prompt_ids = tokenizer.apply_chat_template(
        prompt, tools=tool_schemas(), enable_thinking=False,
        add_generation_prompt=True, tokenize=True, return_dict=False,
    )
    context, mask, completion = list(prompt_ids), [], []
    messages, turns = [], []
    final_text, termination, generated_count = "", "context_budget", 0
    if len(context) >= budgets.max_context_tokens:
        raise ValueError("Initial task prompt exceeds G1 context budget")
    while True:
        remaining = min(
            budgets.max_context_tokens - len(context),
            budgets.max_trajectory_tokens - len(completion),
            budgets.max_generated_tokens - generated_count,
        )
        if clock() >= deadline:
            termination = "timeout"
            break
        if remaining <= 0:
            termination = "token_budget"
            break
        maximum = min(remaining, budgets.max_turn_tokens)
        sampled = generate(context, maximum, deadline)
        if not sampled or len(sampled) > maximum:
            raise RuntimeError("Generation backend returned an invalid token count")
        # Keep sampled IDs exactly. Parsing is for tool routing, never JSON repair.
        before = list(context)
        completion.extend(sampled)
        context.extend(sampled)
        mask.extend([1] * len(sampled))
        generated_count += len(sampled)
        raw = tokenizer.decode(sampled, skip_special_tokens=True)
        try:
            parsed = parse_response(sampled, before)
        except (ValueError, TypeError) as error:
            turns.append({"sampled_ids": sampled, "raw_text": raw, "parse_error": str(error)})
            messages.append({"role": "assistant", "content": raw})
            termination = "malformed_response"
            break
        message = {"role": "assistant", **parsed}
        messages.append(message)
        turns.append({"sampled_ids": sampled, "raw_text": raw, "parsed": message})
        if clock() >= deadline:
            termination = "timeout"
            break
        if sampled[-1] != tokenizer.eos_token_id:
            termination = "length"
            break
        calls = message.get("tool_calls") or []
        if not calls:
            # Score the raw final turn; deterministic JSON extraction belongs to the reward.
            final_text, termination = raw, "stop"
            break
        observations = []
        for call in calls:
            function = call.get("function", {})
            name, arguments = function.get("name", "unknown"), function.get("arguments", {})
            output = tools.execute(name, arguments)
            observation = {"role": "tool", "name": name, "content": output}
            observations.append(observation)
            messages.append(observation)
            if clock() >= deadline:
                termination = "timeout"
                break
        if termination == "timeout":
            break
        suffix = tool_suffix_ids(tokenizer, observations)
        if (
            len(context) + len(suffix) >= budgets.max_context_tokens
            or len(completion) + len(suffix) >= budgets.max_trajectory_tokens
        ):
            termination = "context_budget"
            break
        completion.extend(suffix)
        context.extend(suffix)
        mask.extend([0] * len(suffix))
    if not completion:
        raise RuntimeError("G1 rollout ended before sampling any model tokens")
    elapsed = clock() - started
    return Trajectory(
        prompt_ids, completion, mask, messages, turns, tools.calls, final_text, termination,
        reward.score(final_text, termination).to_dict(), elapsed,
    )


def save_trajectory(directory: Path, trajectory: Trajectory, *, task_id: str, **metadata) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    value = {"experiment": "G1", "task_id": task_id, **metadata, **trajectory.to_dict()}
    (directory / "trajectory.json").write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    (directory / "final.txt").write_text(trajectory.final_text, encoding="utf-8")
    (directory / "reward.json").write_text(json.dumps(trajectory.score, indent=2) + "\n")
