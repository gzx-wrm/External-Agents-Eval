from __future__ import annotations

import json
from typing import Any


def solver_prompt(
    instruction: str,
    ledger: dict[str, Any],
    *,
    ledger_quality: str = "valid",
    ledger_warnings: list[str] | tuple[str, ...] = (),
) -> str:
    """Append the frozen A ledger without changing the original task text."""

    rendered_ledger = json.dumps(
        ledger,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    quality_note = ""
    if ledger_quality == "degraded":
        rendered_warnings = "\n".join(f"- {warning}" for warning in ledger_warnings)
        quality_note = f"""

LEDGER QUALITY
This ledger is structurally usable but one or more mechanical source-anchor
checks were not conclusive. Treat the ledger as a fallible checklist, inspect the
base repository directly, and do not treat an unresolved anchor as proof of a
requirement or of its absence.
{rendered_warnings or '- Source-anchor validation was incomplete.'}
"""
    return f"""{instruction}

--- V6.1 FROZEN REQUIREMENTS LEDGER ---
Before this coding run, a specification reviewer derived the complete ledger below
from the original task and the frozen base repository. Use its O obligations as an
implementation checklist, preserve applicable P obligations, and account for Q open
questions while making the code changes requested by the original task.

The original task and applicable public contracts remain authoritative. The ledger
is a fallible aid: if an item conflicts with its cited source, follow the source; if
you discover an omission, still implement the source-grounded behavior. Do not edit
or regenerate the ledger. Complete the task using your normal coding workflow.

FROZEN LEDGER JSON
{rendered_ledger}
{quality_note}
--- END V6.1 FROZEN REQUIREMENTS LEDGER ---"""
