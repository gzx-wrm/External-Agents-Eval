from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from a_agent_finetune import build_e4_materials as e4
from a_agent_finetune.build_e_teacher_trajectory import dump, sha
from a_agent_finetune.run_f_inference import validate_output
from jsonschema import Draft202012Validator


def materials():
    root = e4.DEFAULT_OUTPUT
    manifest = json.loads((root / "manifest.json").read_text())
    samples = [json.loads(line) for line in (root / "train.jsonl").read_text().splitlines()]
    return root, manifest, samples


def test_conditional_loss_excludes_student_errors_observations_and_thinking_prefix():
    protocol = e4.DEFAULT_NORMAL / "protocol"
    native = (protocol / "native_chat_template.jinja").read_text()
    standard = (protocol / "training_chat_template.jinja").read_text()
    selected = e4.masked_template(standard)
    tokenizer = Tokenizer.from_file(str(protocol / "tokenizer.json"))
    tools = json.loads((protocol / "tools.json").read_text())

    def action(content, ident, path):
        return {"role": "assistant", "content": content, "reasoning_content": "",
                "tool_calls": [{"id": ident, "type": "function",
                                "function": {"name": "read", "arguments": {"path": path}}}]}

    sample = {"view": "full", "tools": tools, "messages": [
        {"role": "system", "content": "SYSTEM_CONTEXT_ONLY"},
        {"role": "user", "content": "USER_CONTEXT_ONLY"},
        action("STUDENT_ACTION_CONTEXT_ONLY", "student", "/work/base/STUDENT_FILE"),
        {"role": "tool", "content": "STUDENT_OBSERVATION_CONTEXT_ONLY", "tool_call_id": "student"},
        action("BAD_TEACHER_ACTION_CONTEXT_ONLY", "bad", "/work/base/BAD_FILE"),
        {"role": "tool", "content": "ERROR_OBSERVATION_CONTEXT_ONLY", "tool_call_id": "bad"},
        action("RECOVERY_ACTION_TARGET", "good", "/work/base/GOOD_FILE"),
        {"role": "tool", "content": "NEW_EVIDENCE_CONTEXT_ONLY", "tool_call_id": "good"},
        {"role": "assistant", "content": '{"summary":"LEDGER_TARGET"}', "reasoning_content": ""}],
        "trainable_message_mask": [False, False, False, False, False, False, True, False, True]}
    arrays, _ = e4.encode_masked_sample(sample, native, standard, selected, tokenizer)
    text = e4.environment().from_string(native).render(**sample, add_generation_prompt=False, enable_thinking=False)
    encoded = tokenizer.encode(text, add_special_tokens=False)
    assert np.array_equal(arrays["input_ids"], encoded.ids)
    for marker, supervised in (("SYSTEM_CONTEXT_ONLY", False), ("USER_CONTEXT_ONLY", False),
                               ("STUDENT_ACTION_CONTEXT_ONLY", False), ("STUDENT_FILE", False),
                               ("STUDENT_OBSERVATION_CONTEXT_ONLY", False),
                               ("BAD_TEACHER_ACTION_CONTEXT_ONLY", False), ("BAD_FILE", False),
                               ("ERROR_OBSERVATION_CONTEXT_ONLY", False),
                               ("RECOVERY_ACTION_TARGET", True), ("GOOD_FILE", True),
                               ("NEW_EVIDENCE_CONTEXT_ONLY", False), ("LEDGER_TARGET", True)):
        start = text.index(marker)
        indices = [i for i, (a, b) in enumerate(encoded.offsets) if b > start and a < start + len(marker)]
        assert indices
        assert np.all((arrays["labels"][indices] != -100) == supervised)
    for match in __import__("re").finditer(r"<think>|</think>|<\|im_start\|>assistant", text):
        indices = [i for i, (a, b) in enumerate(encoded.offsets) if b > match.start() and a < match.end()]
        assert np.all(arrays["labels"][indices] == -100)
    wrong = copy.deepcopy(sample)
    wrong["trainable_message_mask"][7] = True
    import pytest
    with pytest.raises(ValueError, match="Only assistant"):
        e4.encode_masked_sample(wrong, native, standard, selected, tokenizer)


def test_mixed_rows_keep_normal_tokens_and_student_prefixes_unchanged():
    root, manifest, samples = materials()
    assert manifest["sample_count"] == len(samples) == 25
    assert len({s["sample_id"] for s in samples}) == 25
    assert len({s["task_id"] for s in samples}) == 13
    assert manifest["normal_source_counts"] == {"teacher_raw": 8, "human_annotation": 5}
    assert [sum(s["view"] == view for s in samples) for view in ("normal", "short", "full")] == [13, 6, 6]
    assert not any("cliffy" in s["task_id"] for s in samples)
    normal = [json.loads(line) for line in (e4.DEFAULT_NORMAL / "train.jsonl").read_text().splitlines()]
    for old, sample in zip(normal, samples[:13], strict=True):
        assert sample["messages"] == old["messages"] and sample["tools"] == old["tools"]
        with np.load(e4.DEFAULT_NORMAL / "tokenized" / (sample["task_id"] + ".npz")) as a:
            with np.load(root / "tokenized" / (sample["sample_id"] + ".npz")) as b:
                assert all(np.array_equal(a[k], b[k]) for k in ("input_ids", "labels"))
    for sample in samples[13:]:
        seed = json.loads((root / "provenance" / sample["parent_seed_id"] / "seed.json").read_text())
        boundary = sample["teacher_start_message_index"]
        assert boundary == len(seed["messages"])
        assert [e4.normalize_message(m) for m in sample["messages"][:boundary]] == [e4.normalize_message(m) for m in seed["messages"]]
        assert not any(sample["trainable_message_mask"][:boundary])
        assert all(not m.get("reasoning_content") for m in sample["messages"])
        assert all(not flag for m, flag in zip(sample["messages"], sample["trainable_message_mask"]) if m["role"] != "assistant")
        if sample["view"] == "short":
            assert sample["messages"][-1]["role"] == "tool"
            assert not sample["trajectory_complete"]


def test_all_frozen_rows_match_native_ids_and_exact_selected_loss():
    root, manifest, samples = materials()
    native = (root / "protocol/native_chat_template.jinja").read_text()
    standard = (e4.DEFAULT_NORMAL / "protocol/training_chat_template.jinja").read_text()
    selected = (root / "protocol/training_chat_template.jinja").read_text()
    tokenizer = Tokenizer.from_file(str(root / "protocol/tokenizer.json"))
    for row, sample in zip(manifest["tasks"], samples, strict=True):
        arrays, stats = e4.encode_masked_sample(sample, native, standard, selected, tokenizer)
        with np.load(root / "tokenized" / (row["sample_id"] + ".npz")) as frozen:
            assert all(np.array_equal(arrays[k], frozen[k]) for k in arrays)
        assert stats["total_tokens"] == row["total_tokens"]
        assert stats["supervised_tokens"] == row["supervised_tokens"]
        assert np.array_equal(arrays["labels"][arrays["labels"] != -100], arrays["input_ids"][arrays["labels"] != -100])
        if sample["view"] != "normal":
            review = json.loads((root / "records" / (row["sample_id"] + ".json")).read_text())["recovery_review"]
            for masked in review["masked_teacher_turns"]:
                index = masked["message_index"]
                if index < len(sample["messages"]):
                    assert sample["messages"][index]["tool_calls"] == masked["calls"]
                    assert sample["trainable_message_mask"][index] is False
    config = json.loads((root / "training_config.json").read_text())
    assert config["max_seq_length"] >= max(row["total_tokens"] for row in manifest["tasks"])
    assert not config["truncation"] and not config["packing"]


def test_reviewed_recovery_ledgers_are_supported_by_retained_observations():
    root, _, samples = materials()
    validator = Draft202012Validator(json.loads((root / "protocol/spec_output_schema.json").read_text()))
    for sample in samples:
        if sample["view"] != "full":
            continue
        seed = json.loads((root / "provenance" / sample["parent_seed_id"] / "seed.json").read_text())
        task = json.loads(seed["messages"][1]["content"].split("SPEC INPUT\n", 1)[1])["task"]
        evidence = e4.visible_evidence(sample["messages"], e4.DEFAULT_SAMPLING / "baselines" / sample["task_id"])
        target, checks = validate_output(sample["messages"][-1]["content"], {"task": task, "evidence": evidence}, validator)
        assert checks["schema_valid"] and checks["sources_valid"]
        assert target == json.loads((root / "labels" / (sample["sample_id"] + ".json")).read_text())
        if "arktype" in sample["task_id"]:
            assert "finite" in dump(target) and "exported" in dump(target)
        elif "adaptix" in sample["task_id"]:
            assert "canonical/required entry" not in dump(target)
        elif "anko" in sample["task_id"]:
            assert "nil result" in dump(target) and "compound assignments" in dump(target)
        elif "termenv" in sample["task_id"]:
            assert target["preservation_obligations"][0]["statement"].startswith("When preserve-resets is not enabled")
        elif "langchain" in sample["task_id"]:
            assert "cannot receive" not in dump(target)


def test_anko_actual_supplement_is_bound_to_baseline_and_is_not_in_short_view():
    root, _, samples = materials()
    record = json.loads((root / "provenance/supplemental_reads/anko-vm-api.json").read_text())
    assert "not an original teacher" in record["assistant_message_origin"]
    assert "Actual Pi" in record["tool_observation_origin"]
    blob = (e4.DEFAULT_SAMPLING / "baselines" / record["task_id"] / record["path"]).read_bytes()
    text = "".join(b["text"] for b in record["result"]["content"])
    assert hashlib.sha256(blob).hexdigest() == record["full_file_sha256"]
    assert hashlib.sha256(text.encode()).hexdigest() == record["returned_content_sha256"]
    assert text.startswith("\n".join(blob.decode().splitlines()[81:89]))
    for sample in samples:
        if "anko" not in sample["task_id"] or sample["view"] == "normal":
            continue
        extra = [i for i, origin in enumerate(sample["message_origins"]) if origin == "expert_reconstruction"]
        if sample["view"] == "short":
            assert extra == []
        else:
            assert len(extra) == 1
            i = extra[0]
            assert sample["trainable_message_mask"][i] and not sample["trainable_message_mask"][i + 1]
            assert sample["messages"][i + 1]["content"] == text


def test_material_hashes_and_original_sources_are_unchanged():
    root, manifest, _ = materials()
    for path, expected in manifest["file_sha256"].items():
        assert sha(root / path) == expected
    original = json.loads((e4.DEFAULT_NORMAL / "manifest.json").read_text())
    for path, expected in original["file_sha256"].items():
        assert sha(e4.DEFAULT_NORMAL / path) == expected
    for archived in (root / "provenance").glob("recovery-*"):
        for p in archived.iterdir():
            assert p.read_bytes() == (e4.DEFAULT_SAMPLING / "samples" / archived.name / p.name).read_bytes()
