"""V6.6 per-batch check points in the Mini review adapter and their result validation."""
from __future__ import annotations

import json

import pytest

from external_agents_eval.agents import mini_v66_review
from external_agents_eval.agents.mini_review_files import metadata
from external_agents_eval.workflows.solve_review_repair.v66_review_checks import (
    RESULTS_HEADING,
    batch_points,
    batch_section,
    checkpoint_result_errors,
)


def cp(targets, resolution="analog", **extra):
    value = {"targets": targets, "facet": "D6", "given": "g", "when": "w", "then": "t",
             "resolution": resolution, "rules_out": None if resolution == "open" else "r",
             "alternatives": ["reported", "not reported"] if resolution == "open" else [], "favored": None,
             "basis_quote": "q" if resolution in {"analog", "contract"} else None,
             "basis_behavior": "b", "convention": None}
    value.update(extra)
    return value


POINTS = [cp(["O001"], "task"), cp(["O002", "O003"]), cp(["O004"], "open"), cp(["P001"])]


def note(target, results=None):
    body = f"# Review note: {target}\n\n## Assessment\n```json\n{{}}\n```\n"
    if results is not None:
        body += f"\n{RESULTS_HEADING}\n```json\n{json.dumps({'checkpoint_results': results})}\n```\n"
    return body + "\n## Summary\nok\n"


def result(partition, observed="o"):
    return {"input": f"run {partition}", "partition": partition, "observed": observed}


def test_batch_points_follow_the_batch_targets_with_global_numbers():
    assert [cp_id for cp_id, _ in batch_points(POINTS, {"O003"})] == ["CP2"]
    assert [cp_id for cp_id, _ in batch_points(POINTS, {"O001", "O004"})] == ["CP1", "CP3"]
    section = batch_section(POINTS, {"O002", "O004"})
    assert "- CP2 [analog] O002,O003/D6" in section and "- CP3 [open] O004/D6" in section
    assert "CP1" not in section and RESULTS_HEADING in section and "siblings_checked" not in section
    assert batch_section(POINTS, {"O009"}) == ""


def test_complete_results_close_the_batch():
    notes = {"O001": note("O001", [{"cp": "CP1", "outcome": "met", "results": [result("normal")]}]),
             "O002": note("O002", [{"cp": "CP2", "outcome": "not_met", "siblings_checked": ["O003"],
                                    "results": [result("normal"), result("empty")]}])}
    assert checkpoint_result_errors(notes, POINTS, {"O001", "O002"}, metadata) == []


@pytest.mark.parametrize("notes, assessed, message", [
    ({"O001": note("O001")}, {"O001"}, "CP1: no checkpoint_results entry"),
    ({"O002": note("O002", [{"cp": "CP2", "outcome": "met", "siblings_checked": [],
                             "results": [result("normal"), result(" Normal ")]}])}, {"O002"},
     "CP2: needs at least 2 results from different partitions"),
    ({"O004": note("O004", [{"cp": "CP3", "outcome": "met", "siblings_checked": [], "implemented_reading": "maybe",
                             "results": [result("first run"), result("second run")]}])}, {"O004"},
     "CP3: implemented_reading must be one of"),
    ({"O001": note("O001", [{"cp": "CP1", "outcome": "fine", "results": [result("normal")]}])}, {"O001"},
     "CP1: outcome must be"),
    ({"O001": note("O001", [{"cp": "CP1", "outcome": "met", "results": [{"input": "x", "partition": "", "observed": "y"}]}])},
     {"O001"}, "CP1: every result needs"),
])
def test_incomplete_results_keep_the_batch_open(notes, assessed, message):
    errors = checkpoint_result_errors(notes, POINTS, assessed, metadata)
    assert any(message in error for error in errors), errors


def test_open_point_with_a_candidate_reading_passes():
    notes = {"O004": note("O004", [{"cp": "CP3", "outcome": "met", "siblings_checked": [], "implemented_reading": "reported",
                                    "results": [result("before first query"), result("after first query")]}])}
    assert checkpoint_result_errors(notes, POINTS, {"O004"}, metadata) == []


def test_adapter_inserts_the_batch_section_before_file_delivery(monkeypatch):
    state = {"batch_size": 3, "pending_target_ids": ["O002"], "accepted": [], "revisable_target_ids": [],
             "review_notes": {"output_root": "/r", "target_paths": [{"path": "notes/O002.md", "target_id": "O002"}]}}
    prompt = "TASK\nCONTROLLER STATE\n" + json.dumps(state) + "\n"
    monkeypatch.setattr(mini_v66_review.FrozenMiniReviewRunner, "_adapt_review_prompt",
                        lambda self, p: "BODY\n\nV6.3 FILE DELIVERY (rules)\nMORE")
    runner = object.__new__(mini_v66_review.V66FrozenReviewRunner)
    runner.v66_checkpoints = POINTS
    adapted = runner._adapt_review_prompt(prompt)
    assert adapted.index("CHECK POINTS FOR THIS BATCH") < adapted.index("V6.3 FILE DELIVERY")
    assert "- CP2 [analog]" in adapted and "CP1 [task]" not in adapted
    runner.v66_checkpoints = []
    assert runner._adapt_review_prompt(prompt) == "BODY\n\nV6.3 FILE DELIVERY (rules)\nMORE"
    with pytest.raises(ValueError):
        mini_v66_review.insert_batch_section("no marker", "SECTION")


def test_environment_withholds_a_batch_with_missing_results(tmp_path, monkeypatch):
    archive = tmp_path / "review-records"
    (archive / "notes").mkdir(parents=True)
    (archive / "notes/O002.md").write_text(note("O002"))

    def parent_execute(self, action):
        self.batch = {"assessments": [{"target_id": "O002", "review_note_path": "notes/O002.md"}]}
        return {"output": "ran\n", "returncode": 0, "exception_info": ""}

    monkeypatch.setattr(mini_v66_review.MiniFileReviewEnvironment, "execute", parent_execute)
    env = object.__new__(mini_v66_review.V66FileReviewEnvironment)
    env.archive, env.checkpoints, env.checkpoint_errors = archive, POINTS, []
    executed = env.execute({"command": "true"})
    assert env.batch is None and "[V6.6 checkpoint results] Fix current files" in executed["output"]
    assert json.loads((tmp_path / "checkpoint-results-errors.json").read_text())[0][0].startswith("CP2")
    (archive / "notes/O002.md").write_text(note("O002", [{"cp": "CP2", "outcome": "met", "siblings_checked": ["O003"],
                                                          "results": [result("normal"), result("absent")]}]))
    env.execute({"command": "true"})
    assert env.batch is not None


def test_siblings_checked_is_no_longer_required():
    notes = {"O002": note("O002", [{"cp": "CP2", "outcome": "met",
                                    "results": [result("normal"), result("empty")]}])}
    assert checkpoint_result_errors(notes, POINTS, {"O002"}, metadata) == []
