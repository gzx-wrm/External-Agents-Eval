"""V6.4 Solver input: the frozen FT ledger with English guidance on how to use it.

Only the instruction text gains a supplement; the official Mini templates, the
ledger content, tools and budgets are unchanged.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HEADING = "## Requirement ledger (frozen)"
GUIDANCE = """This ledger was prepared from the task description and the base repository to help you
implement and verify the change. The task description remains authoritative: where the
ledger conflicts with the task or with the repository's actual contracts, follow those.
- obligations: behavior the change must add. "required" items must hold; "advisory" items
  are recommended.
- preservation_obligations: existing behavior that must keep working.
- open_questions: points the task leaves unclear. Choose the reading most consistent with
  the task text and the repository's existing conventions; it must not override what the
  task states.
Before submitting, check each obligation and preservation item against your change. Make
sure the project's test targets, including tests you added, still build and the relevant
tests pass. Do not modify, delete or weaken existing tests unless the task requires a
change to the behavior they check."""


def canonical_sha256(value):
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def load_frozen_ledger(path: Path, *, file_sha256: str, ledger_sha256: str):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != file_sha256:
        raise ValueError("Frozen ledger file changed: " + str(path))
    ledger = json.loads(data)
    if canonical_sha256(ledger) != ledger_sha256:
        raise ValueError("Frozen ledger content changed: " + str(path))
    return ledger


def augment_task(task: dict, ledger: dict) -> dict:
    """No evidence packet, review output or additional agent stage is inserted."""
    text = json.dumps(ledger, ensure_ascii=False, indent=2)
    result = dict(task)
    result["instruction"] = (
        task["instruction"] + "\n\n" + HEADING + "\n\n" + GUIDANCE
        + "\n\n```json\n" + text + "\n```\n"
    )
    return result
