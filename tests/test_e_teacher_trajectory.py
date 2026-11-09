from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from jinja2 import Environment
from tokenizers import Tokenizer

from a_agent_finetune import build_e_teacher_trajectory as builder


def source_messages():
    return [
        {"role": "user", "content": [{"type": "text", "text": "SPEC INPUT\n" + json.dumps({
            "task": {"title": "test", "description": "request"}, "base_root": "/old/base",
        })}]},
        {"role": "assistant", "stopReason": "toolUse", "content": [
            {"type": "thinking", "thinking": "Inspect /old/base carefully."},
            {"type": "text", "text": "I will read both files."},
            {"type": "toolCall", "id": "one", "name": "read", "arguments": {"path": "/old/base/a.py"}},
            {"type": "toolCall", "id": "two", "name": "read", "arguments": {"path": "/old/base/b.py"}},
        ]},
        {"role": "toolResult", "toolCallId": "two", "toolName": "read", "isError": True,
         "content": [{"type": "text", "text": "File not found: /old/base/b.py"}]},
        {"role": "toolResult", "toolCallId": "one", "toolName": "read", "isError": False,
         "content": [{"type": "text", "text": "UNIQUE_BASELINE_EVIDENCE"}]},
        {"role": "assistant", "stopReason": "error", "errorMessage": "Connection error", "content": []},
        {"role": "assistant", "stopReason": "stop", "content": [
            {"type": "thinking", "thinking": "ORIGINAL_THINKING final decision."},
            {"type": "text", "text": "old final answer"},
        ]},
    ]


def tools():
    return [{"type": "function", "function": {"name": "read", "parameters": {
        "type": "object", "required": ["path"], "properties": {"path": {"type": "string"}},
    }}}]


def conversation():
    target = {"summary": "human", "obligations": [], "preservation_obligations": [], "open_questions": []}
    events = [{"type": "message", "message": message} for message in source_messages()]
    messages, stats = builder.convert_session(events, target, "system", tools())
    return {"messages": messages, "tools": tools()}, stats, target


def test_preserves_thinking_parallel_results_and_failures_while_replacing_only_final_text():
    sample, stats, target = conversation()
    messages = sample["messages"]
    assert [message["role"] for message in messages] == ["system", "user", "assistant", "tool", "tool", "assistant"]
    assert messages[2]["content"] == "I will read both files."
    assert messages[2]["reasoning_content"] == "Inspect /work/base carefully."
    assert messages[2]["tool_calls"][0]["function"]["arguments"] == {"path": "/work/base/a.py"}
    assert messages[3]["tool_call_id"] == "two"
    assert messages[3]["content"] == "File not found: /work/base/b.py"
    assert messages[-1]["reasoning_content"] == "ORIGINAL_THINKING final decision."
    assert json.loads(messages[-1]["content"]) == target
    assert stats["original_final_content"] == "old final answer"
    assert stats["failed_tool_observations_retained"] == [{"tool_call_id": "two", "name": "read"}]
    assert len(stats["empty_connection_failures_archived_only"]) == 1


def test_rejects_missing_tool_observation():
    messages = source_messages()
    del messages[2]
    with pytest.raises(ValueError, match="Missing tool observations"):
        builder.convert_session([{"type": "message", "message": message} for message in messages], {}, "system", tools())


def test_latest_manual_version_must_match_original_task_and_run(tmp_path):
    source = tmp_path / "source"
    annotations = tmp_path / "annotations"
    for version, modified in (("older", 100), ("newer", 200)):
        directory = annotations / version
        directory.mkdir(parents=True)
        ledger = directory / "ledger.json"
        ledger.write_text(json.dumps({"summary": version}))
        os.utime(ledger, ns=(modified, modified))
        (directory / "fork.json").write_text(json.dumps({
            "task_id": "task", "source_run": str(source), "reuse": {"A": str(ledger)},
        }))
    found = builder.discover_annotations({"task": source}, annotations)
    assert found["task"][0]["ledger"].parent.name == "newer"
    with pytest.raises(ValueError, match="mismatch"):
        builder.discover_annotations({"task": tmp_path / "different_run"}, annotations)


def test_native_qwen_template_preserves_thinking_and_masks_all_tool_observations():
    sample, _, _ = conversation()
    config = json.loads((builder.HERE / ".cache/tokenizer-config-qwen3.5-9b.json").read_text())
    native = config["chat_template"]
    tokenizer = Tokenizer.from_file(str(builder.HERE / ".cache/tokenizer-qwen3.5-9b.json"))
    arrays, stats = builder.encode_sample(sample, native, builder.patch_training_template(native), tokenizer)
    environment = Environment()
    environment.filters["tojson"] = lambda value: json.dumps(value, ensure_ascii=False)
    rendered = environment.from_string(native).render(**sample, add_generation_prompt=False, enable_thinking=True)
    encoded = tokenizer.encode(rendered, add_special_tokens=False)
    for text, supervise in (("UNIQUE_BASELINE_EVIDENCE", False), ("File not found:", False),
                            ("ORIGINAL_THINKING", True), ("Inspect /work/base carefully.", True),
                            ("<function=read>", True)):
        begin = rendered.index(text)
        indices = [i for i, (left, right) in enumerate(encoded.offsets) if right > begin and left < begin + len(text)]
        assert indices
        labels = arrays["labels"][indices]
        assert np.all(labels != -100) if supervise else np.all(labels == -100)
    assert stats["native_template_text_equivalence_verified"]
    assert stats["supervised_tokens"] < stats["total_tokens"]


def test_built_thirty_task_materials_match_sources_without_silent_label_corrections():
    data = builder.DEFAULT_DATA
    manifest = json.loads((data / "manifest.json").read_text())
    samples = [json.loads(line) for line in (data / "train.jsonl").read_text().splitlines()]
    assert len(samples) == manifest["sample_count"] == 30
    assert manifest["label_source_counts"] == {"teacher_raw": 19, "human_annotation": 11}
    for row, sample in zip(manifest["tasks"], samples, strict=True):
        task = row["task_id"]
        record = json.loads((data / "records" / f"{task}.json").read_text())
        target = builder.ledger_target(data / "raw" / task / "selected_ledger.json")
        assert json.loads(sample["messages"][-1]["content"]) == target
        assert json.loads((data / "labels" / f"{task}.json").read_text()) == target
        for name, source in record["sources"].items():
            assert builder.sha(data / "raw" / task / name) == source["sha256"]
            assert builder.sha(builder.ROOT / source["path"]) == source["sha256"]
        arrays = np.load(data / "tokenized" / f"{task}.npz")
        assert len(arrays["input_ids"]) == row["total_tokens"]
        mask = arrays["labels"] != -100
        assert int(np.count_nonzero(mask)) == row["supervised_tokens"]
        assert np.array_equal(arrays["labels"][mask], arrays["input_ids"][mask])
        assert len(row["assistant_turn_supervised_tokens"]) == row["assistant_messages"]
