from __future__ import annotations

import json

import numpy as np
from jinja2 import Environment
from tokenizers import Tokenizer

from a_agent_finetune import build_e2_materials as e2
from a_agent_finetune.build_e_teacher_trajectory import encode_sample, sha


def conversation():
    tools = [{"type": "function", "function": {"name": name, "parameters": {
        "type": "object", "required": ["path"], "properties": {"path": {"type": "string"}},
    }}} for name in ("grep", "read")]

    def call(name, ident):
        return {"id": ident, "type": "function", "function": {"name": name, "arguments": {"path": "/work/base/x"}}}

    return {"tools": tools, "messages": [
        {"role": "system", "content": "SYSTEM_CONTEXT_ONLY"},
        {"role": "user", "content": "USER_CONTEXT_ONLY"},
        {"role": "assistant", "content": "", "reasoning_content": "THINKING_MUST_DISAPPEAR",
         "tool_calls": [call("grep", "first")]},
        {"role": "tool", "name": "grep", "tool_call_id": "first", "content": e2.RG_UNAVAILABLE},
        {"role": "assistant", "content": "", "reasoning_content": "MORE_THINKING_MUST_DISAPPEAR",
         "tool_calls": [call("grep", "repeat"), call("read", "valid")]},
        {"role": "tool", "name": "read", "tool_call_id": "valid", "content": "TOOL_CONTEXT_ONLY"},
        {"role": "tool", "name": "grep", "tool_call_id": "repeat", "content": e2.RG_UNAVAILABLE},
        {"role": "assistant", "content": '{"summary":"FINAL_LEDGER_TARGET"}',
         "reasoning_content": "FINAL_THINKING_MUST_DISAPPEAR"},
    ]}


def test_removes_only_repeated_unavailable_calls_and_keeps_parallel_success():
    original = conversation()
    sample, changes = e2.transform_sample(original)
    e2.validate_conversation(sample)
    assert original["messages"][2]["reasoning_content"] == "THINKING_MUST_DISAPPEAR"
    assert all(not message.get("reasoning_content") for message in sample["messages"])
    assert [call["id"] for message in sample["messages"] for call in message.get("tool_calls", [])] == ["first", "valid"]
    assert [message["tool_call_id"] for message in sample["messages"] if message["role"] == "tool"] == ["first", "valid"]
    assert changes["removed_redundant_tool_calls"][0]["call"]["id"] == "repeat"
    assert sample["messages"][-1]["content"] == original["messages"][-1]["content"]


def test_supervises_actions_and_ledger_without_supervising_empty_think_or_observations():
    sample, _ = e2.transform_sample(conversation())
    native = (e2.DEFAULT_SOURCE / "protocol/native_chat_template.jinja").read_text()
    tokenizer = Tokenizer.from_file(str(e2.HERE / ".cache/tokenizer-qwen3.5-9b.json"))
    arrays, stats = encode_sample(sample, native, e2.no_thinking_template(native), tokenizer)
    environment = Environment()
    environment.filters["tojson"] = lambda value, **kwargs: json.dumps(value, ensure_ascii=False, **kwargs)
    text = environment.from_string(native).render(**sample, add_generation_prompt=False, enable_thinking=False)
    encoded = tokenizer.encode(text, add_special_tokens=False)
    assert "THINKING_MUST_DISAPPEAR" not in text
    for substring, supervised in (("SYSTEM_CONTEXT_ONLY", False), ("USER_CONTEXT_ONLY", False),
                                   ("TOOL_CONTEXT_ONLY", False), (e2.RG_UNAVAILABLE, False),
                                   ("<think>", False), ("</think>", False),
                                   ("<function=read>", True), ("FINAL_LEDGER_TARGET", True)):
        begin = text.index(substring)
        indices = [i for i, (left, right) in enumerate(encoded.offsets)
                   if right > begin and left < begin + len(substring)]
        assert indices
        assert np.all(arrays["labels"][indices] != -100) if supervised else np.all(arrays["labels"][indices] == -100)
    prefix = environment.from_string(native).render(messages=sample["messages"][:-1], tools=sample["tools"],
                                                    add_generation_prompt=True, enable_thinking=False)
    assert text == prefix + sample["messages"][-1]["content"] + "<|im_end|>\n"
    assert stats["native_template_text_equivalence_verified"]


def test_e2_materials_are_complete_and_original_e1_files_are_unchanged():
    data = e2.DEFAULT_OUTPUT
    manifest = json.loads((data / "manifest.json").read_text())
    original = json.loads((e2.DEFAULT_SOURCE / "manifest.json").read_text())
    samples = [json.loads(line) for line in (data / "train.jsonl").read_text().splitlines()]
    assert manifest["experiment"] == "E2" and manifest["enable_thinking"] is False
    assert len(samples) == 13
    assert manifest["label_source_counts"] == {"teacher_raw": 8, "human_annotation": 5}
    for path, expected in manifest["file_sha256"].items():
        assert sha(data / path) == expected
    for path, expected in original["file_sha256"].items():
        assert sha(e2.DEFAULT_SOURCE / path) == expected
    assert [row["task_id"] for row in manifest["tasks"]] == [row["task_id"] for row in original["tasks"]]
    assert manifest["totals"]["citation_corrections_applied"] == 9
    for row, sample in zip(manifest["tasks"], samples, strict=True):
        e2.validate_conversation(sample)
        assert all(not message.get("reasoning_content") for message in sample["messages"])
        assert json.loads(sample["messages"][-1]["content"]) == json.loads((data / "labels" / f"{row['task_id']}.json").read_text())
        with np.load(data / "tokenized" / f"{row['task_id']}.npz") as arrays:
            mask = arrays["labels"] != -100
            assert len(arrays["input_ids"]) == row["total_tokens"]
            assert int(mask.sum()) == row["supervised_tokens"]
            assert np.array_equal(arrays["labels"][mask], arrays["input_ids"][mask])
