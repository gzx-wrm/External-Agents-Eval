"""V6.5 prompts and contracts: I/R text unchanged apart from the view notice."""
from __future__ import annotations

import pytest
from test_mini_v63 import state

from external_agents_eval.workflows.solve_review_repair.v5_contracts import V5ContractError
from external_agents_eval.workflows.solve_review_repair.v5_prompts import reviewer_prompt
from external_agents_eval.workflows.solve_review_repair.v65_contracts import (
    copied_fraction,
    select_disambiguation_targets,
    validate_assumption_output,
)
from external_agents_eval.workflows.solve_review_repair.v65_prompts import (
    ASSUMPTION_OUTPUT_SCHEMA,
    assumption_prompt,
    insert_view_notice,
    view_notice,
)

SOURCE = {"kind": "task", "field": "description", "quote": "Add behavior"}
LEDGER = {
    "summary": "s",
    "obligations": [{"id": "O001", "content": {"statement": "x", "importance": "required",
                                               "sources": [SOURCE]}}],
    "preservation_obligations": [],
    "open_questions": [
        {"id": "Q001", "content": {"question": "Non-blocking?", "blocking": False, "sources": [SOURCE]}},
        {"id": "Q002", "content": {"question": "Blocking?", "blocking": True, "sources": [SOURCE]}},
    ],
}


def claim(text="With option A enabled, render('x') returns 'y' unchanged.", tension="conflict",
          related=("O001",)):
    return {"claim": text, "tension": tension, "related_target_ids": list(related),
            "candidate_refs": [{"path": "a.go", "line_start": 1, "line_end": 2}]}


@pytest.mark.parametrize("role", ["I", "R"])
@pytest.mark.parametrize("level", ["strip-all-tests", "strip-new-tests", "strip-comments"])
def test_view_notice_is_the_only_change_to_v56_reviewer_prompts(tmp_path, role, level):
    s = state(tmp_path)
    original = reviewer_prompt(
        task={"title": "T", "description": "Add behavior"}, repository={}, ledger=LEDGER, state=s,
        role=role, skill_path="skill.md", output_root=str(tmp_path), initial=True,
        limits={"batch_size": 3, "time_budget_sec": 2700}, prompt_version="v5-6",
    )
    notice = view_notice(level)
    adapted = insert_view_notice(original, notice)
    assert adapted.replace("\n" + notice + "\n", "", 1) == original
    assert adapted.index("REVIEW VIEW") < adapted.index("REVIEW RECORDS AND BATCHES")


def test_raw_view_adds_nothing():
    assert view_notice(None) == "" and insert_view_notice("any prompt", "") == "any prompt"


def test_assumption_prompt_carries_raw_patch_withheld_text_and_schema():
    prompt = assumption_prompt(task={"title": "T", "description": "D"}, ledger=LEDGER,
                               candidate_patch="diff --git a/x b/x\n+// why\n",
                               withheld=[{"path": "x", "line": 1, "kind": "comment", "text": "// why"}],
                               max_items=12)
    assert "+// why" in prompt and '"text": "// why"' in prompt
    assert "at most 12" in " ".join(prompt.split()) and "submit_json" in prompt
    assert '"tension"' in prompt and ASSUMPTION_OUTPUT_SCHEMA["$defs"]["Assumption"]


def test_assumption_contract_accepts_neutral_claims_and_numbers_them():
    items = validate_assumption_output({"assumptions": [claim(), claim("Parse('[a]') yields an expression node.", "interpretation", ("Q002",))]},
                                       ledger=LEDGER, withheld_texts=[], max_items=12)
    assert [item["id"] for item in items] == ["X001", "X002"]


@pytest.mark.parametrize("payload, message", [
    ({"assumptions": [claim("Render returns y because the spec says resets end styles.")]}, "gives a reason"),
    ({"assumptions": [claim(related=("O009",))]}, "unknown ledger IDs"),
    ({"assumptions": [claim(), claim()]}, "duplicates"),
    ({"assumptions": [claim(tension="maybe")]}, "JSON Schema validation failed"),
    ({"assumptions": [claim()] * 13}, "limit is 12"),
])
def test_assumption_contract_rejects(payload, message):
    with pytest.raises(V5ContractError, match=message):
        validate_assumption_output(payload, ledger=LEDGER, withheld_texts=[], max_items=12)


def test_copied_comment_text_is_rejected():
    comment = "// using [x] = y is a valid member-expression assignment so only an identifier starts"
    copied = claim("Using [x] = y is a valid member-expression assignment so only an identifier starts here.")
    assert copied_fraction(copied["claim"], [comment]) > 0.5
    with pytest.raises(V5ContractError, match="copies withheld comment text"):
        validate_assumption_output({"assumptions": [copied]}, ledger=LEDGER,
                                   withheld_texts=[comment], max_items=12)


def test_target_selection_priority_and_cap():
    assumptions = validate_assumption_output({"assumptions": [
        claim("Render('a') keeps resets without reopening styles.", "unspecified"),
        claim("Truncate(0) returns an empty string for any input.", "interpretation"),
        claim("Parse('[a]') yields an expression statement node.", "conflict"),
    ]}, ledger=LEDGER, withheld_texts=[], max_items=12)
    targets = select_disambiguation_targets(LEDGER, assumptions, max_targets=4)
    assert [item["id"] for item in targets] == ["X003", "Q002", "X002", "Q001"]
    assert all(item["content"]["importance"] == "advisory" for item in targets)
    assert targets[0]["content"]["sources"] == [SOURCE]
    assert select_disambiguation_targets(LEDGER, [], max_targets=6)[0]["id"] == "Q002"
