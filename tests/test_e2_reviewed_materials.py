from __future__ import annotations

import hashlib
import json
import subprocess

import numpy as np
from jinja2 import Environment
from jsonschema import Draft202012Validator
from tokenizers import Tokenizer

from a_agent_finetune import revise_e2_materials as revision
from a_agent_finetune.build_e_teacher_trajectory import dump, sha


def load_samples(path):
    return [json.loads(line) for line in (path / "train.jsonl").read_text().splitlines()]


def test_reviewed_dataset_preserves_frozen_selection_and_all_original_context():
    data = revision.DEFAULT_OUTPUT
    parent = revision.DEFAULT_SOURCE
    manifest = json.loads((data / "manifest.json").read_text())
    original = json.loads((parent / "manifest.json").read_text())
    assert manifest["sample_count"] == 13
    assert manifest["label_source_counts"] == {"teacher_raw": 8, "human_annotation": 5}
    assert manifest["parent_manifest_sha256"] == sha(parent / "manifest.json")
    assert manifest["totals"]["label_review_edits"] == 11
    assert manifest["totals"]["repository_citations_checked"] == 206
    assert manifest["totals"]["tool_calls"] == original["totals"]["tool_calls"] + 3
    for root, record in ((data, manifest), (parent, original)):
        for path, expected in record["file_sha256"].items():
            assert sha(root / path) == expected
    assert sha(data / "provenance/e2_v1_train.jsonl") == sha(parent / "train.jsonl")
    assert sha(data / "provenance/e1_train.jsonl") == sha(parent / "provenance/e1_train.jsonl")
    assert [row["task_id"] for row in manifest["tasks"]] == [row["task_id"] for row in original["tasks"]]
    for row, before, after in zip(manifest["tasks"], load_samples(parent), load_samples(data), strict=True):
        revision.validate_conversation(after)
        assert after["tools"] == before["tools"]
        assert all(not message.get("reasoning_content") for message in after["messages"])
        assert all(message.get("tool_calls") for message in after["messages"][:-1] if message["role"] == "assistant")
        if row["task_id"] == "anko-typed-variable-bindings":
            assert after["messages"][:-5] == before["messages"][:-1]
            assert [message["role"] for message in after["messages"][-5:]] == ["assistant", "tool", "tool", "tool", "assistant"]
        else:
            assert after["messages"][:-1] == before["messages"][:-1]
        if row["label_review_edits"] == 0:
            assert after == before


def test_reviewed_labels_remain_source_valid_and_resolve_audited_conflicts():
    data = revision.DEFAULT_OUTPUT
    schema = Draft202012Validator(json.loads((data / "protocol/spec_output_schema.json").read_text()))
    manifest = json.loads((data / "manifest.json").read_text())
    labels = {}
    for row, sample in zip(manifest["tasks"], load_samples(data), strict=True):
        task = row["task_id"]
        record = json.loads((data / "records" / f"{task}.json").read_text())
        observed = json.loads((revision.ROOT / record["reference_evidence_record"]["path"]).read_text())
        target = json.loads(sample["messages"][-1]["content"])
        assert target == json.loads((data / "labels" / f"{task}.json").read_text())
        _, checks = revision.validate_output(dump(target), observed, schema)
        assert checks["schema_valid"] and checks["sources_valid"]
        revision.check_read_coverage(sample, target, observed, record["e2_review"]["added_reads"])
        labels[task] = target
    assert labels["termenv-preserve-ansi-resets"]["obligations"][0]["statement"] == "An importable `ansi` subpackage must be added to the module."
    assert len(labels["adaptix-name-mapping-aliases"]["open_questions"]) == 2
    assert "hidden" not in labels["valibot-recursive-schema-composition"]["open_questions"][0]["question"].lower()
    assert "excluding" not in labels["scriggo-method-declarations"]["obligations"][4]["statement"]
    assert len(labels["scriggo-method-declarations"]["open_questions"]) == 2
    preservation = labels["yjs-map-conflict-detection"]["preservation_obligations"][4]["statement"]
    assert "`allow`" in preservation and "`error`" in preservation and "may be rejected" in preservation
    for task in ("arktype-json-schema-refs-dependencies", "anko-typed-variable-bindings"):
        summary = labels[task]["summary"]
        assert "annotation" not in summary and "O004" not in summary and "P005" not in summary
    assert "finite nested data" in labels["arktype-json-schema-refs-dependencies"]["summary"]
    assert "nil result" in labels["anko-typed-variable-bindings"]["summary"]


def test_anko_reconstruction_uses_exact_frozen_baseline_lines_and_records_origin():
    data = revision.DEFAULT_OUTPUT
    task = "anko-typed-variable-bindings"
    record = json.loads((data / "records" / f"{task}.json").read_text())
    observed = json.loads((revision.ROOT / record["reference_evidence_record"]["path"]).read_text())
    review = record["e2_review"]
    sample = load_samples(data)[-1]
    assert "not an original teacher/model rollout" in review["assistant_message_origin"]
    calls = sample["messages"][-5]["tool_calls"]
    responses = sample["messages"][-4:-1]
    assert len(calls) == len(responses) == len(review["added_reads"]) == 3
    for read, call, response in zip(review["added_reads"], calls, responses, strict=True):
        body = subprocess.check_output(["git", "-C", str(revision.HERE / ".cache/baselines" / task),
                                        "show", f"{observed['base_commit']}:{read['path']}"])
        digest = hashlib.sha256(body).hexdigest()
        assert digest == read["full_file_sha256"]
        assert digest in {item["full_file_sha256"] for item in observed["evidence"] if item["path"] == read["path"]}
        lines = body.decode().splitlines()
        start, end = read["line_start"], read["line_end"]
        expected = "\n".join(lines[start - 1:end])
        if end < len(lines):
            expected += f"\n\n[{len(lines) - end} more lines in file. Use offset={end + 1} to continue.]"
        assert response["content"] == expected
        assert response["tool_call_id"] == call["id"] == read["tool_call_id"]
        assert call["function"]["arguments"] == read["arguments"]
        assert hashlib.sha256(expected.encode()).hexdigest() == read["returned_content_sha256"]


def test_reencoded_tokens_and_anko_loss_mask_match_non_thinking_inference():
    data = revision.DEFAULT_OUTPUT
    tokenizer = Tokenizer.from_file(str(data / "protocol/tokenizer.json"))
    native = (data / "protocol/native_chat_template.jinja").read_text()
    environment = Environment()
    environment.filters["tojson"] = lambda value, **kwargs: json.dumps(value, ensure_ascii=False, **kwargs)
    template = environment.from_string(native)
    manifest = json.loads((data / "manifest.json").read_text())
    for row, sample in zip(manifest["tasks"], load_samples(data), strict=True):
        text = template.render(**sample, add_generation_prompt=False, enable_thinking=False)
        prefix = template.render(messages=sample["messages"][:-1], tools=sample["tools"],
                                 add_generation_prompt=True, enable_thinking=False)
        assert text == prefix + sample["messages"][-1]["content"] + "<|im_end|>\n"
        encoded = tokenizer.encode(text, add_special_tokens=False)
        with np.load(data / "tokenized" / f"{row['task_id']}.npz") as arrays:
            assert np.array_equal(arrays["input_ids"], encoded.ids)
            mask = arrays["labels"] != -100
            assert len(encoded.ids) == row["total_tokens"]
            assert int(mask.sum()) == row["supervised_tokens"]
            assert np.array_equal(arrays["labels"][mask], arrays["input_ids"][mask])
            if row["task_id"] != "anko-typed-variable-bindings":
                continue
            for message in sample["messages"][-4:-1]:
                # The frozen native template trims the outer whitespace of every body.
                content = message["content"].strip()
                begin = text.rindex(content)
                indices = [i for i, (left, right) in enumerate(encoded.offsets)
                           if right > begin and left < begin + len(content)]
                assert indices and np.all(arrays["labels"][indices] == -100)
            action = sample["messages"][-5]["content"]
            begin = text.index(action)
            indices = [i for i, (left, right) in enumerate(encoded.offsets)
                       if right > begin and left < begin + len(action)]
            assert indices and np.all(arrays["labels"][indices] != -100)
            final_indices = [i for i, (left, right) in enumerate(encoded.offsets) if right > len(prefix)]
            assert np.all(arrays["labels"][final_indices] != -100)
