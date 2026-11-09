"""Build full A-teacher conversations with the latest available manual Ledger.

No model calls, source edits, truncation, or train/eval splitting are performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

from jsonschema import Draft202012Validator

try:
    from .run_f_inference import validate_output
except ImportError:
    from run_f_inference import validate_output

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIELDS = ("obligations", "preservation_obligations", "open_questions")
MODEL = "Qwen/Qwen3.5-9B"
REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
DEFAULT_DATA = HERE / "data/e_teacher_trajectory_30_human_latest_v1"
REFERENCE_DATA = HERE / "data/f_teacher_observed_30_v4"
DEFAULT_REQUEST = (
    HERE / "runs/qwen35-9b-e0-thinking-native30-20261004-r3/tasks"
    / "fastapi-implicit-head-options/v5/specification/attempt-1/provider-requests.jsonl"
)


def dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ledger_target(path: Path) -> dict:
    ledger = json.loads(path.read_text())
    return {
        "summary": ledger["summary"],
        **{field: [entry.get("content", entry) for entry in ledger[field]] for field in FIELDS},
    }


def text_blocks(message: dict) -> str:
    blocks = message["content"]
    if any(block["type"] != "text" for block in blocks):
        raise ValueError("Unexpected non-text user/tool observation; needs explicit handling")
    return "\n".join(block["text"] for block in blocks)


def map_root(value: Any, old: str, new: str = "/work/base") -> Any:
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [map_root(item, old, new) for item in value]
    if isinstance(value, dict):
        return {key: map_root(item, old, new) for key, item in value.items()}
    return value


def discover_annotations(source_runs: dict[str, Path], directory: Path) -> dict[str, list[dict]]:
    found: dict[str, list[dict]] = {}
    for fork_path in sorted(directory.glob("*/fork.json")):
        fork = json.loads(fork_path.read_text())
        task_id = fork.get("task_id")
        if task_id not in source_runs or not fork.get("reuse", {}).get("A"):
            continue
        if Path(fork["source_run"]).resolve() != source_runs[task_id].resolve():
            raise ValueError(f"Manual annotation baseline/run mismatch: {fork_path}")
        ledger = Path(fork["reuse"]["A"])
        if not ledger.is_absolute():
            ledger = (ROOT / ledger).resolve()
        found.setdefault(task_id, []).append({
            "ledger": ledger, "fork": fork_path, "mtime_ns": ledger.stat().st_mtime_ns,
        })
    for candidates in found.values():
        candidates.sort(key=lambda item: (item["mtime_ns"], str(item["ledger"])), reverse=True)
        if len(candidates) > 1 and candidates[0]["mtime_ns"] == candidates[1]["mtime_ns"]:
            if sha(candidates[0]["ledger"]) != sha(candidates[1]["ledger"]):
                raise ValueError("Different manual versions have identical timestamps; choose explicitly")
    return found


def convert_session(
    events: list[dict], target: dict, system: str, tools: list[dict]
) -> tuple[list[dict], dict]:
    originals = [event["message"] for event in events if event["type"] == "message"]
    if sum(message["role"] == "user" for message in originals) != 1:
        raise ValueError("Expected one initial task query, without compaction or follow-up instructions")
    initial = text_blocks(originals[0])
    decoder = json.JSONDecoder()
    spec, _ = decoder.raw_decode(initial.split("SPEC INPUT\n", 1)[1].lstrip())
    base_root = spec["base_root"]
    user_input = {
        "task": spec["task"], "base_root": "/work/base", "patch_base": "base-commit",
        "baseline_patch_sha256": spec.get("baseline_patch_sha256"),
        "managed_delivery": spec.get("managed_delivery", []),
        "limits": {"max_tool_calls": None, "timeout_sec": 3600},
    }
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": "SPEC INPUT\n" + json.dumps(user_input, ensure_ascii=False, indent=2)},
    ]
    tool_validators = {
        tool["function"]["name"]: Draft202012Validator(tool["function"]["parameters"])
        for tool in tools
    }
    pending: dict[str, str] = {}
    call_ids: set[str] = set()
    error_events = []
    failures = []
    assistant_count = 0
    thinking_chars = 0
    for index, original in enumerate(originals[1:], 1):
        role = original["role"]
        if role == "assistant":
            if pending:
                raise ValueError(f"Missing tool observations before assistant message {index}")
            if original.get("stopReason") == "error" and not original["content"]:
                error_events.append({"source_message_index": index, "error": original.get("errorMessage")})
                continue
            content, reasoning, calls = [], [], []
            for block in original["content"]:
                if block["type"] == "thinking":
                    reasoning.append(block["thinking"])
                elif block["type"] == "text":
                    content.append(block["text"])
                elif block["type"] == "toolCall":
                    name, call_id = block["name"], block["id"]
                    if call_id in call_ids or name not in tool_validators:
                        raise ValueError(f"Duplicate or unknown tool call {call_id}: {name}")
                    arguments = map_root(block["arguments"], base_root)
                    tool_validators[name].validate(arguments)
                    call_ids.add(call_id)
                    pending[call_id] = name
                    calls.append({
                        "id": call_id, "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    })
                else:
                    raise ValueError(f"Unsupported assistant block: {block['type']}")
            message = {
                "role": "assistant", "content": map_root("\n".join(content), base_root),
                "reasoning_content": map_root("\n".join(reasoning), base_root),
            }
            thinking_chars += len(message["reasoning_content"])
            if calls:
                message["tool_calls"] = calls
            if index == len(originals) - 1:
                if calls or original.get("stopReason") != "stop":
                    raise ValueError("Teacher did not terminate with a normal final Ledger reply")
                message["content"] = dump(target)
            messages.append(message)
            assistant_count += 1
        elif role == "toolResult":
            call_id, name = original["toolCallId"], original["toolName"]
            if pending.pop(call_id, None) != name:
                raise ValueError(f"Orphan or mismatched tool result: {call_id}")
            messages.append({
                "role": "tool", "name": name, "tool_call_id": call_id,
                "content": map_root(text_blocks(original), base_root),
            })
            if original.get("isError"):
                failures.append({"tool_call_id": call_id, "name": name})
        else:
            raise ValueError(f"Unexpected role: {role}")
    if pending or messages[-1]["role"] != "assistant":
        raise ValueError("Conversation is incomplete")
    return messages, {
        "source_base_root": base_root, "task": spec["task"],
        "source_message_count": len(originals), "training_message_count": len(messages),
        "assistant_messages": assistant_count, "tool_calls": len(call_ids),
        "thinking_characters": thinking_chars,
        "empty_connection_failures_archived_only": error_events,
        "failed_tool_observations_retained": failures,
        "original_final_content": text_blocks({"content": [
            block for block in originals[-1]["content"] if block["type"] == "text"
        ]}),
    }


def patch_training_template(native: str) -> str:
    """Add assistant loss spans without changing the native serialized text."""
    old_reasoning = "{{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content + '\\n</think>\\n\\n' + content }}"
    old_plain = "{{- '<|im_start|>' + message.role + '\\n' + content }}"
    branch = '{%- elif message.role == "assistant" %}'
    end = '{%- elif message.role == "tool" %}'
    for expected in (old_reasoning, old_plain, branch, end):
        if native.count(expected) != 1:
            raise ValueError("Pinned Qwen template changed; inspect before adding loss masks")
    patched = native.replace(old_reasoning, "{{- '<think>\\n' + reasoning_content + '\\n</think>\\n\\n' + content }}")
    patched = patched.replace(old_plain, "{{- content }}")
    patched = patched.replace(branch, branch + "\n        {{- '<|im_start|>assistant\\n' }}\n        {%- generation %}")
    return patched.replace(end, "{%- endgeneration %}\n    " + end)


def encode_sample(sample: dict, native: str, training_template: str, tokenizer: Any) -> tuple[dict, dict]:
    import numpy as np
    from jinja2 import Environment

    # A marked render locates generation spans; the exported template uses the
    # standard {% generation %} markers supported by Transformers/TRL.
    start, end = "\x00GENERATION_START\x00", "\x00GENERATION_END\x00"
    if any(marker in dump(sample) for marker in (start, end)):
        raise ValueError("Loss-span marker collision")
    marked_template = training_template.replace("{%- generation %}", "{{- generation_start }}")
    marked_template = marked_template.replace("{%- endgeneration %}", "{{- generation_end }}")
    environment = Environment()
    # Match Transformers' tojson filter rather than Jinja's sorted/HTML-escaped default.
    environment.filters["tojson"] = lambda value, **kwargs: json.dumps(value, ensure_ascii=False, **kwargs)

    def fail(message: str) -> None:
        raise ValueError(message)

    kwargs = {**sample, "add_generation_prompt": False, "enable_thinking": True, "raise_exception": fail}
    original = environment.from_string(native).render(**kwargs)
    marked = environment.from_string(marked_template).render(
        **kwargs, generation_start=start, generation_end=end,
    )
    clean, spans = [], []
    remaining, position = marked, 0
    while start in remaining:
        prefix, body = remaining.split(start, 1)
        generated, remaining = body.split(end, 1)
        clean.extend((prefix, generated))
        position += len(prefix)
        spans.append((position, position + len(generated)))
        position += len(generated)
    clean.append(remaining)
    if "".join(clean) != original:
        raise ValueError("Training template changed the native Qwen serialization")
    assistants = [message for message in sample["messages"] if message["role"] == "assistant"]
    if len(spans) != len(assistants):
        raise ValueError("Lost assistant loss span")
    for message, (begin, finish) in zip(assistants, spans, strict=True):
        generated = original[begin:finish]
        if message["reasoning_content"].strip() not in generated or message["content"].strip() not in generated:
            raise ValueError("Reasoning or assistant text disappeared during serialization")
        if generated.count("<function=") != len(message.get("tool_calls", [])):
            raise ValueError("Tool calls disappeared during serialization")
    encoded = tokenizer.encode(original, add_special_tokens=False)
    input_ids = np.asarray(encoded.ids, dtype=np.int32)
    labels = np.full(len(input_ids), -100, dtype=np.int32)
    span_index = 0
    turn_tokens = [0] * len(spans)
    for index, (begin, finish) in enumerate(encoded.offsets):
        while span_index < len(spans) and begin >= spans[span_index][1]:
            span_index += 1
        if span_index == len(spans):
            break
        left, right = spans[span_index]
        if finish > left and begin < right:
            if begin < left or finish > right:
                raise ValueError("Tokenizer merged tokens across a loss-mask boundary")
            labels[index] = input_ids[index]
            turn_tokens[span_index] += 1
    supervised = int(np.count_nonzero(labels != -100))
    stats = {
        "total_tokens": len(input_ids), "supervised_tokens": supervised,
        "masked_context_tokens": len(input_ids) - supervised,
        "assistant_turn_supervised_tokens": turn_tokens,
        "final_ledger_turn_tokens_including_original_thinking": turn_tokens[-1],
        "fits_32k": len(input_ids) <= 32768, "fits_64k": len(input_ids) <= 65536,
        "fits_128k": len(input_ids) <= 131072,
        "rendered_text_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "native_template_text_equivalence_verified": True,
    }
    return {"input_ids": input_ids, "labels": labels}, stats


def build(output: Path, request_path: Path, reference: Path) -> dict:
    import numpy as np
    from tokenizers import Tokenizer

    reference_manifest = json.loads((reference / "manifest.json").read_text())
    reference_records = {
        row["task_id"]: json.loads((reference / "records" / f"{row['task_id']}.json").read_text())
        for row in reference_manifest["tasks"]
    }
    if len(reference_records) != 30:
        raise ValueError("Expected all 30 distinct source tasks")
    annotations = discover_annotations({
        task: ROOT / record["provenance"]["source_trial"] / "v6_review_repair"
        for task, record in reference_records.items()
    }, ROOT / "results/deepswe/manual_annotations")
    request = json.loads(request_path.open().readline())
    tools = request["tools"]
    system = (HERE / "e_prompt_v1.txt").read_text().strip() + "\n\nCurrent working directory: /work/base\n"
    if request["messages"][0]["content"] != system:
        raise ValueError("Current E system prompt differs from the captured E wire request")
    tokenizer_path = HERE / ".cache/tokenizer-qwen3.5-9b.json"
    config_path = HERE / ".cache/tokenizer-config-qwen3.5-9b.json"
    config = json.loads(config_path.read_text())
    native = config["chat_template"]
    training_template = patch_training_template(native)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    validator = Draft202012Validator(json.loads((HERE / "spec_output_schema.json").read_text()))
    output.mkdir(parents=True, exist_ok=False)
    for name in ("records", "labels", "raw", "tokenized", "protocol"):
        (output / name).mkdir()
    write_json(output / "protocol/tools.json", tools)
    (output / "protocol/system-prompt.txt").write_text(system)
    (output / "protocol/native_chat_template.jinja").write_text(native)
    (output / "protocol/training_chat_template.jinja").write_text(training_template)
    shutil.copyfile(config_path, output / "protocol/tokenizer_config.json")
    shutil.copyfile(HERE / "spec_output_schema.json", output / "protocol/spec_output_schema.json")
    samples, rows, audits = [], [], []
    for task_id, record in sorted(reference_records.items()):
        provenance = record["provenance"]
        trajectory = ROOT / provenance["a_trajectory"]
        if sha(trajectory) != provenance["a_trajectory_sha256"]:
            raise ValueError(f"Observed evidence inventory is stale: {task_id}")
        session = trajectory.parent / "session.jsonl"
        teacher_ledger = ROOT / provenance["teacher_ledger"]
        if sha(teacher_ledger) != provenance["teacher_ledger_sha256"]:
            raise ValueError(f"Original teacher Ledger changed: {task_id}")
        candidates = annotations.get(task_id, [])
        selected_ledger = candidates[0]["ledger"] if candidates else teacher_ledger
        target, teacher_target = ledger_target(selected_ledger), ledger_target(teacher_ledger)
        validator.validate(target)
        events = [json.loads(line) for line in session.read_text().splitlines()]
        if any(event["type"] not in {"session", "model_change", "thinking_level_change", "message"} for event in events):
            raise ValueError(f"Unexpected session branch/compaction: {task_id}")
        messages, session_stats = convert_session(events, target, system, tools)
        if session_stats["task"] != record["task"]:
            raise ValueError(f"Task identity mismatch: {task_id}")
        sample = {"messages": messages, "tools": tools}
        arrays, lengths = encode_sample(sample, native, training_template, tokenizer)
        np.savez_compressed(output / "tokenized" / f"{task_id}.npz", **arrays)
        _, checks = validate_output(dump(target), record, validator)
        if not checks["schema_valid"]:
            raise ValueError(f"Invalid final SpecOutput: {task_id}: {checks['errors']}")
        label_source = "human_annotation" if candidates else "teacher_raw"
        archive = output / "raw" / task_id
        archive.mkdir()
        sources = {
            "session.jsonl": session, "teacher_prompt.md": ROOT / provenance["spec_prompt"],
            "teacher_ledger.json": teacher_ledger, "selected_ledger.json": selected_ledger,
        }
        if candidates:
            sources["annotation_fork.json"] = candidates[0]["fork"]
        source_info = {}
        for name, path in sources.items():
            shutil.copyfile(path, archive / name)
            source_info[name] = {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}
        original_final = session_stats.pop("original_final_content")
        (archive / "teacher_final_reply.txt").write_text(original_final)
        write_json(output / "labels" / f"{task_id}.json", target)
        changed = target != teacher_target
        details = {
            "task_id": task_id, "split": "development_only", "label_source": label_source,
            "repository_url": record["repository_url"], "base_commit": record["base_commit"],
            "sources": source_info,
            "manual_candidates_newest_first": [
                {"ledger": str(item["ledger"].relative_to(ROOT)), "sha256": sha(item["ledger"]),
                 "mtime_ns": item["mtime_ns"]} for item in candidates
            ],
            "target_differs_from_original_teacher": changed,
            "reasoning_label_consistency": "not_reconciled_or_semantically_audited",
            "teacher_thinking_preserved": True,
            "path_mapping": {session_stats["source_base_root"]: "/work/base"},
            "prompt_adaptation": "Current E system/user protocol; original teacher prompt archived",
            "label_transform": "Unwrap controller IDs; canonical JSON; no semantic/citation corrections",
            "field_changes": {"summary": target["summary"] != teacher_target["summary"], **{
                field: {
                    "original_count": len(teacher_target[field]), "selected_count": len(target[field]),
                    "differs": target[field] != teacher_target[field],
                } for field in FIELDS
            }},
            "session": session_stats, "lengths": lengths, "label_validation": checks,
            "prior_f_citation_corrections_not_applied": record["source_corrections"],
            "prior_f_label_review_notes": record["label_review_notes"],
            "environment_modified_observations_retained": record["environment_modified_lines_excluded"],
        }
        write_json(output / "records" / f"{task_id}.json", details)
        audits.append({
            "task_id": task_id, "label_source": label_source,
            "target_differs_from_original_teacher": changed, "validation": checks,
            "prior_f_citation_corrections_not_applied": record["source_corrections"],
            "prior_f_label_review_notes": record["label_review_notes"],
            "environment_modified_observations_retained": record["environment_modified_lines_excluded"],
        })
        rows.append({
            "row_index": len(rows), "task_id": task_id, "label_source": label_source,
            "target_differs_from_original_teacher": changed,
            "selected_ledger_sha256": sha(selected_ledger),
            "schema_valid": checks["schema_valid"], "sources_valid": checks["sources_valid"],
            "source_issue_count": len(checks["errors"]),
            "assistant_messages": session_stats["assistant_messages"],
            "tool_calls": session_stats["tool_calls"],
            "failed_tool_observations": len(session_stats["failed_tool_observations_retained"]),
            "archived_empty_connection_failures": len(session_stats["empty_connection_failures_archived_only"]),
            **lengths,
        })
        samples.append(sample)
        print(f"{task_id}: {label_source}, {lengths['total_tokens']} tokens, {lengths['supervised_tokens']} supervised", flush=True)
    (output / "train.jsonl").write_text("".join(dump(sample) + "\n" for sample in samples))
    source_counts = dict(Counter(row["label_source"] for row in rows))
    manifest = {
        "schema_version": "e_full_teacher_trajectory_human_latest_v1",
        "model": MODEL, "model_revision": REVISION,
        "source_batch": reference_manifest["source_batch"], "sample_count": len(rows),
        "split": "development_only", "label_source_counts": source_counts,
        "label_selection": "Newest ledger mtime within matching-task/source-run manual annotation forks; otherwise original teacher A Ledger",
        "label_corrections_applied": False, "teacher_thinking_reconciled": False,
        "training_file": "train.jsonl", "training_file_sha256": sha(output / "train.jsonl"),
        "builder_sha256": sha(Path(__file__)),
        "protocol": {
            "system_sha256": sha(output / "protocol/system-prompt.txt"),
            "tools_sha256": sha(output / "protocol/tools.json"),
            "native_template_sha256": sha(output / "protocol/native_chat_template.jinja"),
            "training_template_sha256": sha(output / "protocol/training_chat_template.jinja"),
            "tool_definition_source": str(request_path.relative_to(ROOT)),
            "tokenizer_sha256": sha(tokenizer_path), "tokenizer_config_sha256": sha(config_path),
        },
        "loss_policy": "All assistant bodies, original thinking, tool-call syntax and turn terminators supervised; system/user/tool observations/assistant headers -100",
        "length_policy": "Full conversations, no truncation, splitting, summarization or packing",
        "totals": {
            "tokens": sum(row["total_tokens"] for row in rows),
            "supervised_tokens": sum(row["supervised_tokens"] for row in rows),
            "minimum_tokens": min(row["total_tokens"] for row in rows),
            "median_tokens": median(row["total_tokens"] for row in rows),
            "maximum_tokens": max(row["total_tokens"] for row in rows),
            "fits_32k": sum(row["fits_32k"] for row in rows),
            "fits_64k": sum(row["fits_64k"] for row in rows),
            "fits_128k": sum(row["fits_128k"] for row in rows),
            "tool_calls": sum(row["tool_calls"] for row in rows),
            "failed_tool_observations_retained": sum(row["failed_tool_observations"] for row in rows),
            "empty_connection_failures_archived_only": sum(row["archived_empty_connection_failures"] for row in rows),
            "schema_valid_tasks": sum(row["schema_valid"] for row in rows),
            "tasks_with_source_issues": sum(not row["sources_valid"] for row in rows),
        },
        "tasks": rows,
    }
    write_json(output / "manifest.json", manifest)
    write_json(output / "label_audit.json", {"scope": "Mechanical checks against previously frozen teacher-observed evidence; semantic support/completeness not audited; issues reported without changing labels", "tasks": audits})
    write_json(output / "training_contract.json", {
        "model": MODEL, "revision": REVISION, "thinking": True,
        "dataset": "train.jsonl", "format": "conversational messages + tools",
        "chat_template": "protocol/training_chat_template.jinja",
        "assistant_only_loss": True, "completion_only_loss": False, "packing": False,
        "truncation_allowed": False,
        "required_sequence_length_at_least": manifest["totals"]["maximum_tokens"],
        "suggested_max_length_for_complete_data": ((manifest["totals"]["maximum_tokens"] + 1023) // 1024) * 1024,
        "tokenized": "tokenized/{task_id}.npz: input_ids and labels int32; attention_mask is all ones; batch padding labels=-100; convert tensors to int64 for model input",
        "loss_shift": "labels are unshifted; causal LM shifts labels internally",
        "trainer_requirement": "Keep reasoning_content and dictionary tool arguments; use native tools-aware template; compare retokenized input_ids/labels to supplied NPZ before training",
        "split": "development_only: all 30 intended for this exploratory fit; not held-out evaluation",
        "hardware": "Full-sequence LoRA training capacity is not verified by inference capacity; do not silently truncate to fit memory",
    })
    write_readme(output, manifest)
    return manifest


def write_readme(output: Path, manifest: dict) -> None:
    totals = manifest["totals"]
    counts = manifest["label_source_counts"]
    audit = json.loads((output / "label_audit.json").read_text())
    correction_count = sum(len(row["prior_f_citation_corrections_not_applied"]) for row in audit["tasks"])
    rows = sorted(manifest["tasks"], key=lambda row: row["total_tokens"], reverse=True)
    table = "\n".join(
        f"| {row['task_id']} | {'人工' if row['label_source'] == 'human_annotation' else '教师'} | {row['total_tokens']:,} | {row['supervised_tokens']:,} | {row['source_issue_count']} |"
        for row in rows
    )
    text = f"""# 30 题原始教师轨迹 + 最新人工 Ledger 训练物料

本批次共 30 题：{counts.get('human_annotation', 0)} 题使用现有最新人工标注，{counts.get('teacher_raw', 0)} 题使用原始教师 Ledger。每题为一条完整多轮对话，保留原始 A 的思考、非最终文本、全部工具调用及真实返回。最终回复替换为选定 Ledger 的纯 SpecOutput JSON，去掉控制器分配的 ID。未启动训练，未自动划分训练／验证集。

## 文件

- `train.jsonl`：可用于工具调用 SFT 的 `messages` + `tools`，按 `manifest.json` 的 row_index 对应题目；正文没有运行 ID、用量统计或下游求解结果。
- `tokenized/*.npz`：使用固定 Qwen tokenizer 和实际工具模板生成的完整 `input_ids` / `labels`。标签未移位；模型内部完成 causal shift。系统、用户、工具返回及 assistant 角色头均为 -100；assistant 思考、文本、工具调用和结束标记参与损失。训练加载时转为 int64，attention_mask 为全 1，补齐处 labels=-100。
- `labels/`：选定 Ledger 的无 ID 版本。未应用此前 F 数据的引用修正。
- `raw/`：逐题保存完整原始 Pi session、原始提示词、教师 Ledger、选定 Ledger、原始最终文本及人工 fork（如有）。
- `records/`、`manifest.json`：来源路径、SHA256、标签选择、字段差异、工具失败和长度统计。
- `label_audit.json`：标签引用机械检查、历史修正建议及语义疑点。提示不代表标签已自动修正。
- `protocol/`：当前 E 提示词、4 个工具 schema、原生模板、带 generation 标记的训练模板、SpecOutput schema 和 tokenizer 配置。
- `training_contract.json`：加载、损失和长度要求；不是未经验证即可启动的完整训练脚本。

## 已执行检查

30 个最终 Ledger 均通过 JSON/schema 和空文本／重复检查。所有工具调用均绑定到对应真实返回，包含并行工具；保留 {totals['tool_calls']:,} 次调用和 {totals['failed_tool_observations_retained']} 次失败返回。{totals['empty_connection_failures_archived_only']} 次连接失败没有任何模型内容，仅保存在 raw 和 records，不生成空 assistant 训练回复。

每题使用 Qwen/Qwen3.5-9B 的固定 revision `{REVISION}`，验证训练模板与原生模板渲染文本完全一致，并逐 token 检查损失边界没有跨入上下文。没有截断、压缩证据、改写思考或分割对话。原仓库根路径统一为 `/work/base`；原始文本仍可在 raw 中追溯。任务和 managed_delivery 不变，系统／用户封装适配当前 E 推理协议。

## 长度与训练限制

完整数据共 **{totals['tokens']:,} tokens**，其中 **{totals['supervised_tokens']:,} tokens** 参与损失；单题最短 {totals['minimum_tokens']:,}，中位 {totals['median_tokens']:,.0f}，最长 **{totals['maximum_tokens']:,}**。32K 可容纳 {totals['fits_32k']}/30，64K 可容纳 {totals['fits_64k']}/30，128K 可容纳 {totals['fits_128k']}/30。必须按最长完整样本准备训练长度；设置 32K/64K 会丢失大量轨迹或最终 Ledger。

训练时保留 `reasoning_content` 和工具参数字典，传入 `tools`，使用带 generation 标记的模板和 assistant_only_loss。以随附 NPZ 为核对基准，避免训练框架丢掉 thinking、工具或改用普通文本模板。推理服务能接收长上下文不等于当前 24GB GPU 能训练这些完整序列；本批次没有做显存可行性验证。

## 保留的实验风险

人工 Ledger 与原教师思考没有重新对齐，逐题标明是否改变标签；这是用户指定的实验条件，不能据此声称思考推导支持人工标签。引用检查发现 {totals['tasks_with_source_issues']} 题仍有机械问题，另外存在此前 F 审计的 {correction_count} 条引用调整建议；全部保留原选定标签，并单独记录。Numba 的两次返回包含环境修改行，仍保留真实观察并标注其与冻结基线的差异。

全部 30 题是 development_only。若用它们训练，再在相同题目上验证，只能报告开发集拟合／下游变化，不能作为未见任务泛化结果。

## 按完整长度排序

| 题目 | 标签 | 总 tokens | 监督 tokens | 引用机械问题 |
| --- | --- | ---: | ---: | ---: |
{table}

## 重建

在项目根目录使用 `.venv-deepswe/bin/python a_agent_finetune/build_e_teacher_trajectory.py --output-dir <新的目录>`。构建过程只读取原实验与标注，不调用模型。默认目录已存在时不会覆盖；重新构建时使用新的输出目录。需要本地缓存中的固定 tokenizer、jinja2、tokenizers、numpy 和 jsonschema。
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--tool-request", type=Path, default=DEFAULT_REQUEST)
    parser.add_argument("--reference-data", type=Path, default=REFERENCE_DATA)
    args = parser.parse_args()
    build(args.output_dir, args.tool_request, args.reference_data)


if __name__ == "__main__":
    main()
