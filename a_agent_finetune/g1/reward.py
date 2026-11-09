"""One terminal reward: extract complete JSON, without repair or a semantic judge."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

PROTOCOL = Path(__file__).resolve().parent / "protocol"


@dataclass(frozen=True)
class DeliveryScore:
    reward: float
    reason: str
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-JSON constant: {value}")


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _has_blank_text(value: Any) -> bool:
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, dict):
        return any(_has_blank_text(v) for v in value.values())
    if isinstance(value, list):
        return any(_has_blank_text(v) for v in value)
    return False


def _json_candidates(text: str):
    """Find complete objects in prose/fences, preserving their original JSON text."""
    decoder = json.JSONDecoder()
    position = 0
    while True:
        start = text.find("{", position)
        if start < 0:
            return
        try:
            _, end = decoder.raw_decode(text, start)
        except ValueError:
            position = start + 1
            continue
        # Strict parsing below rejects duplicate keys and non-JSON constants. Skip
        # the whole object even if invalid, rather than accepting its nested fields.
        yield text[start:end]
        position = end


class DeliveryReward:
    def __init__(self, schema_path: Path = PROTOCOL / "spec_output_schema.json") -> None:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        self.validator = Draft202012Validator(schema)

    def score(self, final_text: str, termination: str) -> DeliveryScore:
        if termination != "stop":
            return DeliveryScore(0.0, termination)
        failure = None
        for candidate in _json_candidates(final_text):
            result = self._score_json(candidate)
            if result.reward:
                return result
            if failure is None:
                failure = result
        return failure or DeliveryScore(0.0, "invalid_json", "No complete JSON object in final answer")

    def _score_json(self, text: str) -> DeliveryScore:
        try:
            value = json.loads(
                text,
                parse_constant=_reject_constant,
                object_pairs_hook=_object,
            )
        except (ValueError, TypeError) as error:
            return DeliveryScore(0.0, "invalid_json", str(error))
        error = next(self.validator.iter_errors(value), None)
        if error is not None:
            return DeliveryScore(0.0, "invalid_schema", error.message)
        if not value["obligations"]:
            return DeliveryScore(0.0, "empty_obligations")
        if _has_blank_text(value):
            return DeliveryScore(0.0, "blank_text")
        return DeliveryScore(1.0, "delivered")


def binary_delivery_reward(completions, g1_reward, **kwargs) -> list[float]:
    """TRL callback: scores were computed from the untouched terminal assistant turn."""
    if len(completions) != len(g1_reward) or any(r not in (0.0, 1.0) for r in g1_reward):
        raise ValueError("G1 requires one binary terminal score per candidate")
    return list(g1_reward)
