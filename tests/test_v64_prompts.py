"""V6.4 prompt revisions: V5-5 renderings stay byte-identical; V5-6 is coherent."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from external_agents_eval.agents.mini_review_files import adapt_prompt, adapt_prompt_v64
from external_agents_eval.workflows.solve_review_repair.v5_config import (
    REVIEW_MATERIAL_MAX_FILE_BYTES,
    REVIEW_MATERIAL_MAX_FILES,
    REVIEW_MATERIAL_MAX_TOTAL_BYTES,
    V5_ROLE_STEP_LIMIT,
)
from external_agents_eval.workflows.solve_review_repair.v5_prompts import (
    I_PROBE_RULES_V56,
    OUTPUT_CONTRACT_MARKER,
    repair_prompt,
    reviewer_prompt,
    summary_prompt,
)

SKILLS = Path(__file__).resolve().parents[1] / "external_agents_eval/workflows/solve_review_repair/skills"

# Renderings of the unchanged V5-5 builders, recorded from the pre-V6.4 source.
GOLDEN_V55 = {
    "reviewer_I_True": "b1f5b3d49d92e92751e75d55e7aa20710a006fa2580d9f9245b0a6da1729a336",
    "reviewer_I_False": "0b3dddaf262820d232358ffd5942957d232dc1eebb71147e26ae51af03e1917c",
    "reviewer_R_True": "a3a4f820d06aaf6e34999b2ce86c5becdee9f48f5cad9c0d1ef00ca2df4eacf4",
    "reviewer_R_False": "a793d807b79522cc2bc165875f4b4b8695ec1a047cb19b70084ac5a38ec50134",
    "repair": "427a55af31143e9b00b41a0edc0488ffc6bfb8ec88464a1cd4af73514298bee1",
    "summary": "f4b0d13d40bfcf9d4b924a60022e5e4945f1fcee1187dd363cd29f1eab46f18f",
}
LIMITS = {"batch_size": 3, "max_rounds": 64, "max_format_retries": 2, "max_semantic_retries": 1,
          "max_no_progress": 2, "time_budget_sec": 2700}
SUMMARY = {"summary": "s", "decisions": [], "new_questions": [], "repair_requests": [
    {"finding_refs": ["I/O001"], "change": "fix", "validation_record_refs": [], "preserve_refs": []}]}


def state():
    return {"round": 1, "mode": "continue", "batch_size": 3, "registered_targets": [],
            "pending_target_ids": ["O001"], "accepted": [], "revisable_target_ids": [],
            "review_notes": {"output_root": "/out", "skill_path": "skill.md",
                             "target_paths": [{"target_id": "O001", "path": "notes/O001.md"}]},
            "catalog_path": "c.json", "errors": [], "time_remaining_sec": 60, "rounds_remaining": 64}


def render(role, initial, **extra):
    return reviewer_prompt(
        task={"title": "T", "description": "Add behavior"},
        repository={"base_root": "/b", "candidate_root": "/c"},
        ledger={"obligations": [], "preservation_obligations": [], "open_questions": []},
        state=state(), role=role, skill_path="skill.md", output_root="/out", managed_delivery=[],
        limits=LIMITS, initial=initial, **extra)


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.mark.parametrize("version", [None, "v5-5"])
def test_v55_renderings_are_byte_identical(version):
    extra = {} if version is None else {"prompt_version": version}
    for role in "IR":
        for initial in (True, False):
            assert sha(render(role, initial, **extra)) == GOLDEN_V55[f"reviewer_{role}_{initial}"]
    repair = repair_prompt({"title": "T"}, "abc", SUMMARY, review_context={"x": 1}, **extra)
    assert sha(repair) == GOLDEN_V55["repair"]
    assert sha(summary_prompt({"task": {"title": "T"}, "x": 1}, freeze_id="f")) == GOLDEN_V55["summary"]


def test_v55_repair_rejects_build_check_context():
    with pytest.raises(ValueError, match="V5-6"):
        repair_prompt({"title": "T"}, "abc", SUMMARY, build_check={"failures": []})


def test_v56_rules_only_for_i_and_limits_follow_constants():
    i_prompt, r_prompt = render("I", True, prompt_version="v5-6"), render("R", True, prompt_version="v5-6")
    assert I_PROBE_RULES_V56 in i_prompt and I_PROBE_RULES_V56 not in r_prompt
    for prompt in (i_prompt, r_prompt):
        assert prompt.count(OUTPUT_CONTRACT_MARKER) == 1
        assert f"at most {REVIEW_MATERIAL_MAX_FILES} files" in prompt
        assert f"{REVIEW_MATERIAL_MAX_FILE_BYTES // (1024 * 1024)} MiB per" in prompt
        assert f"{REVIEW_MATERIAL_MAX_TOTAL_BYTES // (1024 * 1024)} MiB in total" in prompt
        assert "about 45 minutes" in prompt and "must not exceed 3" in prompt
        assert "\nCONTROLLER STATE\n" in prompt and "\nOUTPUT ROOT: /out\n" in prompt
        body = prompt.split(OUTPUT_CONTRACT_MARKER, 1)[0]
        # The body is delivery-neutral: no JSON-batch or done-flag instructions.
        assert "review_note_path" not in body and "done" not in re.sub(r"\w+done\w*", "", body)
    with pytest.raises(ValueError):
        reviewer_prompt(task={}, repository={}, ledger={}, state=state(), role="I",
                        skill_path="s", output_root="/out", limits={}, prompt_version="v5-6")


def test_v64_file_contract_replaces_only_the_output_contract():
    prompt = render("I", True, prompt_version="v5-6")
    adapted = adapt_prompt_v64(prompt)
    body = prompt.split(OUTPUT_CONTRACT_MARKER, 1)[0]
    assert adapted.startswith(body)
    tail = adapted[len(body):]
    assert "OUTPUT CONTRACT (files)" in tail and "FIELD DEFINITIONS" in tail
    assert "V6.3 FILE DELIVERY" not in adapted and "OUTPUT SCHEMA" not in adapted
    assert '"done"' not in adapted and '"review_note_path"' not in adapted
    assert "Do not write review_note_path" in tail and "Do not call submit_json" in tail
    with pytest.raises(ValueError):
        adapt_prompt_v64(render("I", True))  # a V5-5 prompt has no OUTPUT CONTRACT
    # The V6.3 adapter is untouched for V5-5 prompts.
    assert "V6.3 FILE DELIVERY" in adapt_prompt(render("I", True))


def test_v56_repair_prompt_states_budget_and_build_check_only_when_present():
    context = {"preserve_refs": ["R/P001"], "reports": {"I": {}}, "artifact_root": "/a"}
    budget = {"minutes": 30, "steps": V5_ROLE_STEP_LIMIT}
    plain = repair_prompt({"title": "T"}, "abc", SUMMARY, review_context=dict(context),
                          prompt_version="v5-6", budget=budget)
    assert "about 30 minutes and 250 commands" in plain and "BUILD CHECK\n" not in plain
    assert plain.index("REPAIR REQUESTS") < plain.index("TASK\n") < plain.index("REVIEW EVIDENCE")
    assert json.loads(plain.split("PRESERVE REFS\n", 1)[1].split("\n\n", 1)[0]) == ["R/P001"]
    assert plain.rstrip().endswith("the project's test targets build.")
    check = {"description": "d", "failures": [{"command": "go test", "cwd": ".", "exit_code": 1,
                                               "log_tail": "undefined: Foo"}]}
    gated = repair_prompt({"title": "T"}, "abc", SUMMARY, review_context=dict(context),
                          prompt_version="v5-6", budget=budget, build_check=check)
    assert gated.index("BUILD CHECK\n") < gated.index("REPAIR REQUESTS")
    assert "undefined: Foo" in gated
    assert "V5's first experiment" not in gated
    with pytest.raises(ValueError):
        repair_prompt({"title": "T"}, "abc", SUMMARY, prompt_version="v5-6")


def test_v64_skill_is_file_delivery_only_and_v5_skill_is_unchanged():
    new = (SKILLS / "v6.4-review-note-files/SKILL.md").read_text()
    assert "Return only ReviewBatch JSON" not in new and "final JSON" not in new
    assert "derives `review_note_path`" in new
    template = (SKILLS / "v6.4-review-note-files/assets/review-note-template.md").read_text()
    assert template.index("## Assessment") < template.index("## Summary")
    old = (SKILLS / "v5-review-note/SKILL.md").read_text()
    assert "Return\nonly ReviewBatch JSON" in old or "Return only ReviewBatch JSON" in old.replace("\n", " ")
    for name in ["static-satisfied.md", "dynamic-violated.md", "regression-violated.md",
                 "blocked-unresolved.md", "source-disputed.md"]:
        assert ((SKILLS / "v6.4-review-note-files/references" / name).read_bytes()
                == (SKILLS / "v5-review-note/references" / name).read_bytes())
