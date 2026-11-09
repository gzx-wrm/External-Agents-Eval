from __future__ import annotations

import copy
import importlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import pytest

from a_agent_finetune import run_f_inference

# The material-building scripts are also executable directly from their folder.
sys.path.insert(0, str(run_f_inference.HERE))
builder = importlib.import_module("build_f_observed")
DATA = run_f_inference.HERE / "data/f_teacher_observed_30_v4"


@pytest.mark.parametrize(
    "content,arguments,expected",
    [
        ("two\nthree", {"offset": 2, "limit": 2}, (2, 3)),
        ("two\nthree\n", {"offset": 2, "limit": 20}, (2, 3)),
        ("one\n\n[Showing lines 1-1 of 3 (50.0KB limit). Use offset=2 to continue.]", {}, (1, 1)),
        ("one\n\n[2 more lines in file. Use offset=2 to continue.]", {"limit": 1}, (1, 1)),
    ],
)
def test_read_ranges_follow_actual_return_including_truncation_and_eof(content, arguments, expected):
    assert builder.returned_span(content, arguments, "one\ntwo\nthree\n") == expected


def test_modified_text_is_not_silently_accepted_as_baseline():
    with pytest.raises(ValueError, match="differs from frozen baseline"):
        builder.returned_span("modified", {"limit": 1}, "original\n")


def test_source_removals_keep_original_indices_and_do_not_change_claims():
    first = {"kind": "task", "field": "description", "quote": "request"}
    wrong = {"kind": "repository", "path": "a.py", "line_start": 9, "line_end": 9}
    fixed = {**wrong, "line_start": 1, "line_end": 1}
    unrelated = {**wrong, "path": "b.py"}
    target = {"obligations": [{"statement": "request", "sources": [first, wrong, unrelated]}]}
    record = {"task_id": "sample", "target_original": copy.deepcopy(target), "target": target, "source_corrections": []}
    corrections = [
        {"field": "obligations", "entry_index": 0, "source_index": 1, "before": wrong, "after": fixed, "reason": "relocate"},
        {"field": "obligations", "entry_index": 0, "source_index": 2, "before": unrelated, "after": None, "reason": "unrelated"},
    ]
    builder.apply_corrections(record, corrections)
    assert record["target"]["obligations"][0] == {"statement": "request", "sources": [first, fixed]}
    assert record["target_original"]["obligations"][0]["sources"] == [first, wrong, unrelated]
    assert len(record["source_corrections"]) == 2


def test_all_thirty_inputs_bind_v4_prompt_to_visible_evidence_and_corrected_labels():
    manifest = json.loads((DATA / "manifest.json").read_text())
    samples, data_file = run_f_inference.load_samples(DATA, "full")
    rows = [json.loads(line) for line in data_file.read_text().splitlines()]
    assert len(samples) == len(rows) == 30
    assert manifest["label_source_counts"] == {"teacher_raw": 19, "human_annotation": 11}
    system = (run_f_inference.HERE / "f_prompt_v4.txt").read_text().strip()
    for sample, row in zip(samples, rows, strict=True):
        record = json.loads((DATA / "records" / f"{sample['task_id']}.json").read_text())
        assert sample["messages"][0]["content"] == system
        assert "completion" not in sample
        assert json.loads(row["completion"][0]["content"]) == record["target"]
        assert not record["target_validation"]["errors"]
        for field in builder.FIELDS:
            for before, after in zip(record["target_original"][field], record["target"][field], strict=True):
                assert {key: value for key, value in before.items() if key != "sources"} == {
                    key: value for key, value in after.items() if key != "sources"
                }
        for evidence in record["evidence"]:
            assert len(evidence["text"].splitlines()) == evidence["line_end"] - evidence["line_start"] + 1
            assert builder.sha256(evidence["text"].encode()) == evidence["sha256"]
        for ref in builder.reference_audit(record):
            assert ref["check"] in {"exact_task_quote", "visible_baseline_range", "removed_with_reason"}


def test_evidence_contains_exactly_observed_lines_without_gold_guided_expansion():
    for path in (DATA / "records").glob("*.json"):
        record = json.loads(path.read_text())
        expected = defaultdict(set)
        actual = defaultdict(set)
        exclusions = defaultdict(set)
        for excluded in record["environment_modified_lines_excluded"]:
            exclusions[excluded["tool_call_id"]].add(excluded["line"])
        for read in record["read_provenance"]:
            expected[read["path"]].update(
                set(range(read["line_start"], read["line_end"] + 1)) - exclusions[read["tool_call_id"]]
            )
        for evidence in record["evidence"]:
            lines = set(range(evidence["line_start"], evidence["line_end"] + 1))
            assert not actual[evidence["path"]] & lines
            actual[evidence["path"]].update(lines)
        assert dict(actual) == dict(expected)
        provenance = record["provenance"]
        for file_key, hash_key in (
            ("a_trajectory", "a_trajectory_sha256"),
            ("teacher_ledger", "teacher_ledger_sha256"),
            ("selected_label_ledger", "selected_label_ledger_sha256"),
        ):
            assert builder.sha256((builder.PROJECT_ROOT / provenance[file_key]).read_bytes()) == provenance[hash_key]
