"""V6.5 deterministic contracts: assumption claims and D-lane target selection."""
from __future__ import annotations

import json
import re
from typing import Any

from .v5_contracts import V5ContractError, validate_output_schema
from .v65_prompts import ASSUMPTION_OUTPUT_SCHEMA

REASON_WORDS = re.compile(r"\b(because|since|so that|in order to|due to|the reason)\b", re.I)
SHINGLE = 6
MAX_COPIED_FRACTION = 0.5
TENSION_ORDER = {"conflict": 0, "interpretation": 2, "unspecified": 4}


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.casefold())


def _shingles(words: list[str]) -> set[tuple[str, ...]]:
    return {tuple(words[index:index + SHINGLE]) for index in range(len(words) - SHINGLE + 1)}


def copied_fraction(claim: str, withheld_texts: list[str]) -> float:
    """Share of the claim's word 6-grams that also occur in withheld comment text."""
    claim_shingles = _shingles(_words(claim))
    if not claim_shingles:
        return 0.0
    withheld: set[tuple[str, ...]] = set()
    for text in withheld_texts:
        withheld |= _shingles(_words(text))
    return len(claim_shingles & withheld) / len(claim_shingles)


def validate_assumption_output(
    payload: Any,
    *,
    ledger: dict[str, Any],
    withheld_texts: list[str],
    max_items: int,
) -> list[dict[str, Any]]:
    payload = validate_output_schema(payload, ASSUMPTION_OUTPUT_SCHEMA)
    items = payload["assumptions"]
    if len(items) > max_items:
        raise V5ContractError(f"assumptions contains {len(items)} claims, limit is {max_items}")
    known = {
        item["id"]
        for field in ("obligations", "preservation_obligations", "open_questions")
        for item in ledger.get(field, [])
    }
    errors: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(items):
        claim = " ".join(item["claim"].split())
        key = claim.casefold()
        if key in seen:
            errors.append(f"assumptions[{index}] duplicates an earlier claim")
        seen.add(key)
        unknown = sorted(set(item["related_target_ids"]) - known)
        if unknown:
            errors.append(f"assumptions[{index}] cites unknown ledger IDs: {', '.join(unknown)}")
        if REASON_WORDS.search(claim):
            errors.append(
                f"assumptions[{index}].claim gives a reason; state only the observable behavior"
            )
        if copied_fraction(claim, withheld_texts) > MAX_COPIED_FRACTION:
            errors.append(
                f"assumptions[{index}].claim copies withheld comment text; restate it as behavior"
            )
        for ref_index, ref in enumerate(item["candidate_refs"]):
            if ref["line_end"] < ref["line_start"]:
                errors.append(f"assumptions[{index}].candidate_refs[{ref_index}] has an invalid range")
    if errors:
        raise V5ContractError("; ".join(errors[:8]))
    return [
        {**item, "id": f"X{index:03d}", "claim": " ".join(item["claim"].split())}
        for index, item in enumerate(items, 1)
    ]


def _sources_for(ids: list[str], ledger: dict[str, Any]) -> list[dict[str, Any]]:
    catalog = {
        item["id"]: item["content"]
        for field in ("obligations", "preservation_obligations", "open_questions")
        for item in ledger.get(field, [])
    }
    unique: dict[str, dict[str, Any]] = {}
    for target_id in ids:
        for source in catalog.get(target_id, {}).get("sources", []):
            unique[json.dumps(source, sort_keys=True)] = source
    return list(unique.values())


def select_disambiguation_targets(
    ledger: dict[str, Any],
    assumptions: list[dict[str, Any]],
    *,
    max_targets: int,
) -> list[dict[str, Any]]:
    """Ledger questions and X claims for the D lane, in a fixed priority order.

    Priority: conflicting claims, blocking questions, interpretation claims, other
    questions, unspecified claims; ties keep their original order.
    """
    candidates: list[tuple[int, int, dict[str, Any]]] = []
    for position, item in enumerate(ledger.get("open_questions", [])):
        content = item["content"]
        rank = 1 if content.get("blocking") else 3
        candidates.append((rank, position, {
            "id": item["id"],
            "content": {
                "kind": "question",
                "statement": content["question"],
                "importance": "advisory",
                "blocking": bool(content.get("blocking")),
                "sources": content.get("sources", []),
            },
        }))
    for position, item in enumerate(assumptions):
        candidates.append((TENSION_ORDER[item["tension"]], position, {
            "id": item["id"],
            "content": {
                "kind": "assumption",
                "statement": item["claim"],
                "importance": "advisory",
                "tension": item["tension"],
                "related_target_ids": list(item["related_target_ids"]),
                "sources": _sources_for(item["related_target_ids"], ledger),
            },
        }))
    candidates.sort(key=lambda entry: (entry[0], entry[1]))
    return [entry[2] for entry in candidates[:max_targets]]


__all__ = [
    "copied_fraction",
    "select_disambiguation_targets",
    "validate_assumption_output",
]
