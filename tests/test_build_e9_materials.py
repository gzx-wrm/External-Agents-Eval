from __future__ import annotations

import json

import pytest

from a_agent_finetune import build_e9_materials as e9

LEDGER = json.dumps({"summary": "uses {braces} inside strings", "obligations": []})


def test_plain_final_turn_is_accepted_without_closing_note():
    assert e9.check_final_turn("t", LEDGER, LEDGER) is False


def test_closing_note_is_accepted_when_the_controller_recovers_the_ledger():
    assert e9.check_final_turn("t", "Evidence is sufficient; I return the Ledger now.\n\n" + LEDGER, LEDGER)


@pytest.mark.parametrize("content", [
    "Note with a {brace}.\n\n" + LEDGER,
    "Note.\n\n" + LEDGER.replace("uses", "changed"),
    "Note. " + LEDGER,
    "\n\n" + LEDGER,
])
def test_closing_note_rejects_anything_else(content):
    with pytest.raises(ValueError):
        e9.check_final_turn("t", content, LEDGER)
