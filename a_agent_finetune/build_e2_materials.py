"""Prepare E2 from the frozen E1 selection, without teacher thinking or model calls."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from jinja2 import Environment
from jsonschema import Draft202012Validator
from tokenizers import Tokenizer

try:
    from .build_e_teacher_trajectory import ROOT, discover_annotations, dump, encode_sample, sha, write_json
    from .run_f_inference import validate_output
except ImportError:
    from build_e_teacher_trajectory import ROOT, discover_annotations, dump, encode_sample, sha, write_json
    from run_f_inference import validate_output

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = HERE / "runs/lora-train/qwen35-9b-e13-teacher8-human5-a800-20261004-r1/materials"
DEFAULT_OUTPUT = HERE / "data/e2_no_thinking_success13_teacher8_human5_v1"
RG_UNAVAILABLE = "ripgrep (rg) is not available and could not be downloaded"


def no_thinking_template(native: str) -> str:
    """Keep native serialization; supervise content/calls/end, not the empty think prefix."""
    reasoning = "{{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content + '\\n</think>\\n\\n' + content }}"
    plain = "{{- '<|im_start|>' + message.role + '\\n' + content }}"
    branch = '{%- elif message.role == "assistant" %}'
    calls = '{%- if message.tool_calls and message.tool_calls is iterable and message.tool_calls is not mapping %}'
    end = '{%- elif message.role == "tool" %}'
    for text in (reasoning, plain, branch, calls, end):
        if native.count(text) != 1:
            raise ValueError("Pinned native template changed")
    patched = native.replace(branch, branch + "\n        {{- '<|im_start|>assistant\\n' }}")
    patched = patched.replace(reasoning, "{{- '<think>\\n' + reasoning_content + '\\n</think>\\n\\n' }}")
    patched = patched.replace(plain, "{{- '' }}")
    patched = patched.replace(calls, "{%- generation %}\n        {{- content }}\n        " + calls)
    return patched.replace(end, "{%- endgeneration %}\n    " + end)


def transform_sample(original: dict) -> tuple[dict, dict]:
    sample = copy.deepcopy(original)
    calls = {
        call["id"]: call
        for message in sample["messages"] if message["role"] == "assistant"
        for call in message.get("tool_calls", [])
    }
    removed = []
    first_failure = None
    for message in sample["messages"]:
        if message["role"] != "tool" or message["content"] != RG_UNAVAILABLE:
            continue
        call = calls[message["tool_call_id"]]
        if call["function"]["name"] != "grep":
            continue
        if first_failure is None:
            first_failure = call["id"]
        else:
            removed.append({"call": call, "observation": message,
                            "reason": "Repeated grep after this session established rg is unavailable"})
    removed_ids = {item["call"]["id"] for item in removed}
    messages, thinking_chars, empty_turns = [], 0, 0
    for message in sample["messages"]:
        if message["role"] == "tool" and message["tool_call_id"] in removed_ids:
            continue
        if message["role"] == "assistant":
            thinking_chars += len(message.get("reasoning_content", ""))
            message["reasoning_content"] = ""
            if "tool_calls" in message:
                message["tool_calls"] = [call for call in message["tool_calls"] if call["id"] not in removed_ids]
                if not message["tool_calls"]:
                    del message["tool_calls"]
            if not message["content"].strip() and not message.get("tool_calls"):
                empty_turns += 1
                continue
        messages.append(message)
    sample["messages"] = messages
    return sample, {"removed_thinking_characters": thinking_chars,
                    "removed_redundant_tool_calls": removed,
                    "removed_empty_assistant_turns": empty_turns,
                    "retained_first_rg_failure": first_failure}


def validate_conversation(sample: dict) -> None:
    validators = {tool["function"]["name"]: Draft202012Validator(tool["function"]["parameters"])
                  for tool in sample["tools"]}
    pending, seen = {}, set()
    for message in sample["messages"]:
        if message["role"] == "assistant":
            if pending or message.get("reasoning_content"):
                raise ValueError("Unresolved tool calls or retained thinking")
            if "<think>" in message["content"] or "</think>" in message["content"]:
                raise ValueError("Thinking embedded in assistant content")
            for call in message.get("tool_calls", []):
                name = call["function"]["name"]
                if call["id"] in seen:
                    raise ValueError("Duplicate tool call ID")
                validators[name].validate(call["function"]["arguments"])
                seen.add(call["id"])
                pending[call["id"]] = name
        elif message["role"] == "tool":
            if pending.pop(message["tool_call_id"], None) != message["name"]:
                raise ValueError("Orphan tool observation")
    if pending or sample["messages"][-1]["role"] != "assistant" or sample["messages"][-1].get("tool_calls"):
        raise ValueError("Incomplete trajectory")


def correct_citations(target: dict, corrections: list[dict], evidence: list[dict]) -> list[dict]:
    applied = []
    for correction in corrections:
        source = target[correction["field"]][correction["entry_index"]]["sources"][correction["source_index"]]
        if source != correction["before"]:
            raise ValueError("Citation correction does not match the frozen label")
        after = correction["after"]
        matching = [item for item in evidence if item["path"] == after["path"]
                    and item["line_start"] <= after["line_start"] and after["line_end"] <= item["line_end"]]
        if not matching:
            raise ValueError("Correction is outside teacher-visible baseline evidence")
        item = matching[0]
        if hashlib.sha256(item["text"].encode()).hexdigest() != item["sha256"]:
            raise ValueError("Evidence hash changed")
        text = "\n".join(item["text"].splitlines()[after["line_start"] - item["line_start"]:
                                                   after["line_end"] - item["line_start"] + 1])
        source.clear()
        source.update(after)
        applied.append({**correction, "verified_baseline_text": text, "evidence_sha256": item["sha256"]})
    return applied


def build(source: Path, output: Path) -> dict:
    original_manifest = json.loads((source / "manifest.json").read_text())
    for name, expected in original_manifest["file_sha256"].items():
        if sha(source / name) != expected:
            raise ValueError(f"Frozen E1 file changed: {name}")
    originals = [json.loads(line) for line in (source / "train.jsonl").read_text().splitlines()]
    if len(originals) != 13 or original_manifest["label_source_counts"] != {"teacher_raw": 8, "human_annotation": 5}:
        raise ValueError("Expected the frozen successful 13-task E1 selection")
    source_records = {row["task_id"]: json.loads((source / "records" / f"{row['task_id']}.json").read_text())
                      for row in original_manifest["tasks"]}
    annotations = discover_annotations({
        task: (ROOT / record["sources"]["teacher_ledger.json"]["path"]).parent.parent
        for task, record in source_records.items()
    }, ROOT / "results/deepswe/manual_annotations")
    for task, record in source_records.items():
        latest = annotations[task][0]["ledger"] if task in annotations else ROOT / record["sources"]["teacher_ledger.json"]["path"]
        if sha(latest) != record["sources"]["selected_ledger.json"]["sha256"]:
            raise ValueError(f"E1 selection no longer contains the latest Ledger: {task}")
    native = (source / "protocol/native_chat_template.jinja").read_text()
    template = no_thinking_template(native)
    tokenizer_path = HERE / ".cache/tokenizer-qwen3.5-9b.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    validator = Draft202012Validator(json.loads((source / "protocol/spec_output_schema.json").read_text()))
    output.mkdir(parents=True, exist_ok=False)
    for name in ("records", "labels", "tokenized", "provenance"):
        (output / name).mkdir()
    shutil.copytree(source / "protocol", output / "protocol")
    (output / "protocol/training_chat_template.jinja").write_text(template)
    shutil.copyfile(tokenizer_path, output / "protocol/tokenizer.json")
    shutil.copyfile(source / "train.jsonl", output / "provenance/e1_train.jsonl")
    shutil.copyfile(source / "manifest.json", output / "provenance/e1_manifest.json")
    # This standalone runner already verifies the generation mask against the frozen labels.
    runner = (HERE / "lora_train_trajectories.py").read_text().replace(
        "enable_thinking=True, add_generation_prompt=False,",
        'enable_thinking=manifest.get("enable_thinking", True), add_generation_prompt=False,')
    (output / "lora_train_trajectories.py").write_text(runner)
    environment = Environment()
    environment.filters["tojson"] = lambda value, **kwargs: json.dumps(value, ensure_ascii=False, **kwargs)
    rows, samples = [], []
    for old_row, original in zip(original_manifest["tasks"], originals, strict=True):
        task = old_row["task_id"]
        record = source_records[task]
        selected = record["sources"]["selected_ledger.json"]
        if sha(ROOT / selected["path"]) != selected["sha256"]:
            raise ValueError(f"Selected source Ledger changed: {task}")
        evidence_path = HERE / "data/f_teacher_observed_30_v4/records" / f"{task}.json"
        observed = json.loads(evidence_path.read_text())
        sample, changes = transform_sample(original)
        target = json.loads(sample["messages"][-1]["content"])
        changes["citation_corrections_applied"] = correct_citations(
            target, record["prior_f_citation_corrections_not_applied"], observed["evidence"])
        sample["messages"][-1]["content"] = dump(target)
        _, checks = validate_output(dump(target), observed, validator)
        if not checks["schema_valid"] or not checks["sources_valid"]:
            raise ValueError(f"Invalid Ledger: {task}: {checks}")
        validate_conversation(sample)
        arrays, lengths = encode_sample(sample, native, template, tokenizer)
        lengths["final_ledger_turn_tokens"] = lengths.pop("final_ledger_turn_tokens_including_original_thinking")
        # Non-thinking requests must start exactly where the supervised assistant body starts.
        prompt = environment.from_string(native).render(
            messages=sample["messages"][:-1], tools=sample["tools"],
            add_generation_prompt=True, enable_thinking=False)
        full = environment.from_string(native).render(**sample, add_generation_prompt=False, enable_thinking=False)
        if not full.startswith(prompt) or full[len(prompt):] != dump(target) + "<|im_end|>\n":
            raise ValueError("Final training/inference boundary differs")
        np.savez_compressed(output / "tokenized" / f"{task}.npz", **arrays)
        write_json(output / "labels" / f"{task}.json", target)
        row = {**old_row, **lengths, "e1_total_tokens": old_row["total_tokens"],
               "e1_supervised_tokens": old_row["supervised_tokens"],
               "removed_thinking_characters": changes["removed_thinking_characters"],
               "removed_redundant_tool_calls": len(changes["removed_redundant_tool_calls"]),
               "citation_corrections_applied": len(changes["citation_corrections_applied"]),
               "assistant_messages": sum(message["role"] == "assistant" for message in sample["messages"]),
               "tool_calls": sum(len(message.get("tool_calls", [])) for message in sample["messages"]),
               "training_ledger_sha256": hashlib.sha256(dump(target).encode()).hexdigest()}
        write_json(output / "records" / f"{task}.json", {
            "task_id": task, "label_source": old_row["label_source"],
            "e1_record": record, "e2_transform": changes, "lengths": lengths,
            "reference_evidence_record": {"path": str(evidence_path.relative_to(ROOT)), "sha256": sha(evidence_path)},
            "label_validation": checks, "thinking_in_training": False,
            "training_inference_boundary_verified": True,
        })
        rows.append(row)
        samples.append(sample)
        print(f"{task}: {lengths['total_tokens']} input / {lengths['supervised_tokens']} supervised", flush=True)
    (output / "train.jsonl").write_text("".join(dump(sample) + "\n" for sample in samples))
    config = json.loads((source.parent / "config.json").read_text())
    config["max_seq_length"] = ((max(row["total_tokens"] for row in rows) + 4095) // 4096) * 4096
    write_json(output / "training_config.json", config)
    write_json(output / "inference_config.json", {
        "chat_template_kwargs": {"enable_thinking": False},
        "restart_from_base_model": True,
        "note": "Every Pi/provider request and the inference server must use this non-thinking protocol.",
    })
    totals = {"tokens": sum(row["total_tokens"] for row in rows),
              "supervised_tokens": sum(row["supervised_tokens"] for row in rows),
              "maximum_tokens": max(row["total_tokens"] for row in rows),
              "removed_thinking_characters": sum(row["removed_thinking_characters"] for row in rows),
              "removed_redundant_tool_calls": sum(row["removed_redundant_tool_calls"] for row in rows),
              "citation_corrections_applied": sum(row["citation_corrections_applied"] for row in rows),
              "tool_calls": sum(row["tool_calls"] for row in rows)}
    manifest = {
        "schema_version": "e2_no_thinking_success13_v1", "experiment": "E2",
        "experiment_definitions": {"E0": "Base model, thinking enabled",
                                   "E1": "LoRA on full teacher thinking and trajectories",
                                   "E2": "LoRA on teacher tool interactions and Ledger, thinking removed and disabled"},
        "model": original_manifest["model"], "model_revision": original_manifest["model_revision"],
        "source_dataset": str(source.relative_to(ROOT)),
        "source_manifest_sha256": sha(source / "manifest.json"),
        "created_at": datetime.now(timezone.utc).isoformat(), "sample_count": len(rows),
        "label_source_counts": dict(Counter(row["label_source"] for row in rows)),
        "enable_thinking": False, "thinking_in_training": False,
        "latest_selected_ledgers_verified": True, "builder_sha256": sha(Path(__file__)),
        "loss_policy": "Assistant content, retained tool-call syntax and turn ends supervised; system/user/tool observations/headers/empty think prefixes -100",
        "length_policy": "No truncation, splitting, summarization or packing; only thinking and repeated unavailable-rg interactions removed",
        "split": original_manifest["split"], "e1_totals": original_manifest["totals"],
        "comparison_note": "E2 also cleans redundant rg failures and 9 documented citation anchors; it is not a thinking-only causal ablation. E0 non-thinking is needed to separate inference-mode effects.",
        "totals": totals, "tasks": rows,
    }
    write_json(output / "validation.json", {
        "status": "passed", "sample_count": len(rows),
        "latest_ledger_selection_verified": True,
        "assistant_thinking_characters": 0,
        "conversations_and_tool_arguments_valid": len(rows),
        "schema_and_source_checks_passed": len(rows),
        "native_template_text_equivalence_verified": len(rows),
        "non_thinking_training_inference_boundary_verified": len(rows),
        "tokenization_and_loss_spans_verified": len(rows),
        "full_semantic_label_audit_performed": False,
        "transformers_mask_crosscheck": "Performed by the bundled training runner before GPU model loading; not run locally",
    })
    (output / "examples").mkdir()
    for index in (0, 8):
        write_json(output / "examples" / f"{rows[index]['task_id']}.json", samples[index])
    table = "\n".join(f"| {row['task_id']} | {row['label_source']} | {row['e1_total_tokens']:,} | {row['total_tokens']:,} | {row['supervised_tokens']:,} |" for row in rows)
    (output / "README.md").write_text(f"""# E2：删除教师 thinking 的 13 题 LoRA 训练物料

E0 = 微调前 base + thinking；E1 = 完整教师 thinking 轨迹 LoRA；E2 = 删除教师 thinking、以工具调用及最终 Ledger 为监督目标，训练和推理均关闭 thinking。

沿用 E1 的成功 13 题，8 道教师来源标签 + 5 道最新人工标签。引用位置修正记录在 records 中；不修改义务含义或人工标注源文件。原 E1 轨迹（包含全部 thinking）归档在 provenance/e1_train.jsonl，仅作溯源，不能作为 E2 训练输入。

每轮输入 {totals['tokens']:,} tokens，监督 {totals['supervised_tokens']:,} tokens，最长 {totals['maximum_tokens']:,} tokens。Thinking 正文为零；原生空 think 前缀作为上下文，不计损失。首次 rg 不可用及其他失败观察保留，仅移除 {totals['removed_redundant_tool_calls']} 个相同环境故障后的重复调用及对应返回。全部 {totals['citation_corrections_applied']} 处引用修正已核对教师可见基线证据；结构与引用位置自动检查不等于完整语义金标准审查。

train.jsonl 每行一题，tokenized/*.npz 包含 input_ids 和 labels。labels/*.json 为最终无编号 Ledger。protocol/ 包含工具、schema、原生/训练模板及固定 revision 的 tokenizer。records/ 保留处理记录和校验结果，manifest.json 固定来源、token 统计和文件哈希。

training_config.json 沿用 E1 的 rank16/alpha32、BF16、学习率1e-4、3 epoch、batch1、seed3407；仅按 E2 最长输入收窄 max_seq_length。lora_train_trajectories.py 是独立训练入口，使用 manifest 中的 enable_thinking=False 并在载入时用 Transformers 重新校验全部 input_ids 和监督 mask。本地制作不加载模型、不启动 GPU 训练。

在已有 E1 训练环境中，从原始固定 revision 的 base 重新训练，不续训 E1 adapter：

```bash
python lora_train_trajectories.py --materials . --model-path /path/to/c202236235762e1c871ad0ccb60c8ee5ba337b9a --config training_config.json --output /path/to/new-e2-run
```

推理必须应用 inference_config.json 中的 chat_template_kwargs.enable_thinking=false；此文件是参数约定，不会自动修改正在运行的服务或 Pi 请求。

E2 同时改变 thinking、重复故障调用和引用锚点，不应把 E1/E2 差异全部归因为 thinking。E0 + 非 thinking 可作为推理模式对照。这 30 题已经用于开发/人工标注，余下 17 题只是未参与本轮训练的开发验证题，不自动构成论文独立测试集。

| 题目 | 标签来源 | E1 输入 tokens | E2 输入 tokens | E2 监督 tokens |
|---|---|---:|---:|---:|
{table}

重建：在项目根目录运行 python -m a_agent_finetune.build_e2_materials --output 新目录。依赖使用现有 numpy、jinja2、tokenizers、jsonschema；无需模型调用或网络。
""")
    manifest["file_sha256"] = {str(path.relative_to(output)): sha(path)
                               for path in sorted(output.rglob("*")) if path.is_file()}
    write_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = build(args.source.resolve(), args.output.resolve())
    print(json.dumps(manifest["totals"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
