"""Apply the reviewed E2 label edits and one baseline-backed Anko retrieval turn."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from jinja2 import Environment
from jsonschema import Draft202012Validator
from tokenizers import Tokenizer

try:
    from .build_e2_materials import DEFAULT_OUTPUT as DEFAULT_SOURCE, HERE, validate_conversation
    from .build_e_teacher_trajectory import ROOT, dump, encode_sample, sha, write_json
    from .run_f_inference import validate_output
except ImportError:
    from build_e2_materials import DEFAULT_OUTPUT as DEFAULT_SOURCE, HERE, validate_conversation
    from build_e_teacher_trajectory import ROOT, dump, encode_sample, sha, write_json
    from run_f_inference import validate_output

DEFAULT_OUTPUT = HERE / "data/e2_no_thinking_success13_teacher8_human5_v2"


def verified_baseline(observed: dict, path: str) -> bytes:
    repository = HERE / ".cache/baselines" / observed["task_id"]
    body = subprocess.check_output(["git", "-C", str(repository), "show", f"{observed['base_commit']}:{path}"])
    expected = {item["full_file_sha256"] for item in observed["evidence"] if item["path"] == path}
    if hashlib.sha256(body).hexdigest() not in expected:
        raise ValueError(f"Cached baseline differs from the recorded evidence: {path}")
    return body


def review_label(task: str, target: dict, observed: dict) -> list[dict]:
    edits = []

    def edit(path: list, value, reason: str) -> None:
        parent = target
        for key in path[:-1]:
            parent = parent[key]
        previous = copy.deepcopy(parent[path[-1]])
        parent[path[-1]] = value
        edits.append({"path": path, "before": previous, "after": copy.deepcopy(value), "reason": reason})

    if task == "termenv-preserve-ansi-resets":
        edit(["obligations", 0, "statement"], "An importable `ansi` subpackage must be added to the module.",
             "Keep the package requirement; individual exported API obligations already cover the repeated list")
    elif task == "fastapi-deprecation-response-headers":
        verified_baseline(observed, "fastapi/applications.py")
        sources = target["obligations"][23]["sources"] + [
            {"kind": "repository", "path": "fastapi/applications.py", "line_start": 1015, "line_end": 1018},
            {"kind": "repository", "path": "fastapi/applications.py", "line_start": 1064, "line_end": 1067},
        ]
        edit(["obligations", 23, "sources"], sources,
             "Support the existing expert instance-isolation inference with the observed per-app middleware lifecycle; this is not an explicit baseline statistics contract")
    elif task == "arktype-json-schema-refs-dependencies":
        summary = target["summary"].split(" This manual annotation", 1)[0]
        summary += " Recursive schema conversion and validation of finite nested data must terminate and enforce the referenced constraints at every recursive occurrence, including within anyOf."
        edit(["summary"], summary, "Describe the required behavior directly; move human-edit history and controller IDs out of the training target")
    elif task == "anko-typed-variable-bindings":
        summary = target["summary"].split(" This annotation", 1)[0]
        summary += " Typed-binding failures must return a non-nil error with a nil result through the public VM API, including nested failures; successful result behavior remains unchanged."
        edit(["summary"], summary, "Keep the error-result behavior, without annotation history or controller IDs")
        sources = copy.deepcopy(target["obligations"][18]["sources"])
        for source in sources:
            if source.get("path") == "vm/main_test.go":
                source.update(line_start=201, line_end=218)
            elif source.get("path") == "vm/vmStmt.go" and source["line_start"] == 34:
                source["line_end"] = 49
        sources.append({"kind": "repository", "path": "vm/vm_test.go", "line_start": 82, "line_end": 89})
        edit(["obligations", 18, "sources"], sources,
             "Cite the actual return-value assertion and existing error cases with nil output, alongside the public execution API")
    elif task == "valibot-recursive-schema-composition":
        edit(["open_questions", 0, "question"],
             "The task does not specify the exact runtime representation or exported type name of the `Recur` placeholder. Is a particular public representation part of the API contract, or is its representation an implementation detail as long as the specified placeholder behavior is preserved?",
             "Ask about the public contract instead of speculating about hidden tests")
    elif task == "adaptix-name-mapping-aliases":
        edit(["open_questions"], target["open_questions"][1:],
             "Remove the replacement-versus-concatenation question already settled by first-wins-per-field and the corresponding obligation")
    elif task == "scriggo-method-declarations":
        edit(["obligations", 4, "statement"],
             "Method declarations must work across definable receiver base types, including defined integer, string, slice, map, function, channel and struct types, rather than only structs.",
             "Do not settle the pointer/interface receiver-category question inside an obligation while retaining that uncertainty in open_questions")
        edit(["summary"], target["summary"].replace("method sets spanning all valid receiver base types", "method sets across definable receiver base types"),
             "Align the summary with the receiver-category uncertainty retained in open_questions")
    elif task == "yjs-map-conflict-detection":
        for path in ("src/structs/ContentType.js", "src/structs/ContentDoc.js"):
            verified_baseline(observed, path)
        edit(["preservation_obligations", 4, "statement"],
             "Under the `allow` policy, or for conflict-free map writes and updates, Yjs nested shared types and subdocuments must remain valid map values, and their integration and associated events must continue to work. Conflicting writes under the `error` policy may be rejected as required by the task.",
             "Preserve successful shared-type/subdoc behavior without contradicting the task-authorized rejection of conflicting writes")
        sources = target["preservation_obligations"][4]["sources"] + [
            {"kind": "repository", "path": "src/structs/ContentType.js", "line_start": 78, "line_end": 80},
            {"kind": "repository", "path": "src/structs/ContentDoc.js", "line_start": 92, "line_end": 99},
            {"kind": "task", "field": "description", "quote": "In error mode, conflicting map writes throw MapConflictError"},
        ]
        edit(["preservation_obligations", 4, "sources"], sources,
             "Include the observed integration paths and the explicit error-policy exception")
    return edits


def add_anko_reads(sample: dict, observed: dict) -> list[dict]:
    reads, calls, results = [], [], []
    for number, (path, start, end) in enumerate((
        ("vm/vmStmt.go", 34, 49), ("vm/main_test.go", 198, 219), ("vm/vm_test.go", 80, 89),
    ), 1):
        body = verified_baseline(observed, path)
        lines = body.decode().splitlines()
        text = "\n".join(lines[start - 1:end])
        if end < len(lines):
            text += f"\n\n[{len(lines) - end} more lines in file. Use offset={end + 1} to continue.]"
        ident = f"call_e2_review_anko_{number:02d}"
        arguments = {"path": "/work/base/" + path, "offset": start, "limit": end - start + 1}
        calls.append({"id": ident, "type": "function", "function": {"name": "read", "arguments": arguments}})
        results.append({"role": "tool", "name": "read", "tool_call_id": ident, "content": text})
        reads.append({"tool_call_id": ident, "arguments": arguments, "path": path,
                      "line_start": start, "line_end": end, "base_commit": observed["base_commit"],
                      "full_file_sha256": hashlib.sha256(body).hexdigest(),
                      "returned_content_sha256": hashlib.sha256(text.encode()).hexdigest()})
    sample["messages"][-1:-1] = [{
        "role": "assistant", "content": "I'll verify the public VM API's result-and-error contract against its tests before finalizing the obligations.",
        "reasoning_content": "", "tool_calls": calls,
    }, *results]
    return reads


def check_read_coverage(sample: dict, target: dict, observed: dict, additions: list[dict]) -> int:
    ids = {call["id"] for message in sample["messages"] for call in message.get("tool_calls", [])}
    reads = [read for read in observed["read_provenance"] if read["tool_call_id"] in ids] + additions
    count = 0
    for field in ("obligations", "preservation_obligations", "open_questions"):
        for entry in target[field]:
            for source in entry["sources"]:
                if source["kind"] != "repository":
                    continue
                count += 1
                ranges = [(read["line_start"], read["line_end"]) for read in reads if read["path"] == source["path"]]
                if not all(any(start <= line <= end for start, end in ranges)
                           for line in range(source["line_start"], source["line_end"] + 1)):
                    raise ValueError(f"Citation is not covered by retained baseline reads: {source}")
    return count


def revise(source: Path, output: Path) -> dict:
    parent = json.loads((source / "manifest.json").read_text())
    for name, expected in parent["file_sha256"].items():
        if sha(source / name) != expected:
            raise ValueError(f"Frozen E2 v1 file changed: {name}")
    samples = [json.loads(line) for line in (source / "train.jsonl").read_text().splitlines()]
    shutil.copytree(source, output)
    shutil.copyfile(source / "manifest.json", output / "provenance/e2_v1_manifest.json")
    shutil.copyfile(source / "train.jsonl", output / "provenance/e2_v1_train.jsonl")
    native = (output / "protocol/native_chat_template.jinja").read_text()
    template = (output / "protocol/training_chat_template.jinja").read_text()
    tokenizer = Tokenizer.from_file(str(output / "protocol/tokenizer.json"))
    validator = Draft202012Validator(json.loads((output / "protocol/spec_output_schema.json").read_text()))
    environment = Environment()
    environment.filters["tojson"] = lambda value, **kwargs: json.dumps(value, ensure_ascii=False, **kwargs)
    changes, citations = [], 0
    for row, sample in zip(parent["tasks"], samples, strict=True):
        task = row["task_id"]
        record_path = output / "records" / f"{task}.json"
        record = json.loads(record_path.read_text())
        evidence_path = ROOT / record["reference_evidence_record"]["path"]
        if sha(evidence_path) != record["reference_evidence_record"]["sha256"]:
            raise ValueError(f"Reference evidence changed: {task}")
        observed = json.loads(evidence_path.read_text())
        target = json.loads(sample["messages"][-1]["content"])
        edits = review_label(task, target, observed)
        additions = add_anko_reads(sample, observed) if task == "anko-typed-variable-bindings" else []
        sample["messages"][-1]["content"] = dump(target)
        validate_conversation(sample)
        # Every intermediate assistant turn must continue through tools, not terminate early.
        if any(message["role"] == "assistant" and not message.get("tool_calls") for message in sample["messages"][:-1]):
            raise ValueError("An intermediate assistant turn has no tool continuation")
        _, checks = validate_output(dump(target), observed, validator)
        if not checks["schema_valid"] or not checks["sources_valid"]:
            raise ValueError(f"Invalid reviewed Ledger: {task}: {checks}")
        citations += check_read_coverage(sample, target, observed, additions)
        arrays, lengths = encode_sample(sample, native, template, tokenizer)
        lengths["final_ledger_turn_tokens"] = lengths.pop("final_ledger_turn_tokens_including_original_thinking")
        prompt = environment.from_string(native).render(messages=sample["messages"][:-1], tools=sample["tools"],
                                                       add_generation_prompt=True, enable_thinking=False)
        full = environment.from_string(native).render(**sample, add_generation_prompt=False, enable_thinking=False)
        if full != prompt + dump(target) + "<|im_end|>\n":
            raise ValueError("Non-thinking training/inference boundary differs")
        np.savez_compressed(output / "tokenized" / f"{task}.npz", **arrays)
        write_json(output / "labels" / f"{task}.json", target)
        row.update(lengths)
        row.update(assistant_messages=sum(message["role"] == "assistant" for message in sample["messages"]),
                   tool_calls=sum(len(message.get("tool_calls", [])) for message in sample["messages"]),
                   training_ledger_sha256=hashlib.sha256(dump(target).encode()).hexdigest(),
                   label_review_edits=len(edits), reconstructed_tool_calls=len(additions))
        review = {"task_id": task, "label_edits": edits, "added_reads": additions,
                  "assistant_message_origin": "Human-authored training reconstruction; not an original teacher/model rollout" if additions else None,
                  "tool_observation_origin": "Replayed from pinned public baseline Git objects, with native read-style line limits" if additions else None}
        record.update(lengths=lengths, label_validation=checks, e2_review=review)
        write_json(record_path, record)
        if edits or additions:
            changes.append(review)
        print(f"{task}: {lengths['total_tokens']} input / {lengths['supervised_tokens']} supervised", flush=True)
    (output / "train.jsonl").write_text("".join(dump(sample) + "\n" for sample in samples))
    write_json(output / "review_changes.json", changes)
    for index in (0, 8, 12):
        write_json(output / "examples" / f"{parent['tasks'][index]['task_id']}.json", samples[index])
    parent.update(schema_version="e2_no_thinking_success13_v2", materials_revision=2,
                  parent_materials=str(source.relative_to(ROOT)), parent_manifest_sha256=sha(source / "manifest.json"),
                  previous_builder_sha256=parent["builder_sha256"], builder_sha256=sha(Path(__file__)),
                  created_at=datetime.now(timezone.utc).isoformat())
    parent["totals"].update(tokens=sum(row["total_tokens"] for row in parent["tasks"]),
                            supervised_tokens=sum(row["supervised_tokens"] for row in parent["tasks"]),
                            maximum_tokens=max(row["total_tokens"] for row in parent["tasks"]),
                            tool_calls=sum(row["tool_calls"] for row in parent["tasks"]),
                            label_review_edits=sum(len(item["label_edits"]) for item in changes),
                            reconstructed_tool_calls=3, repository_citations_checked=citations)
    parent["length_policy"] += "; one human-authored Anko retrieval turn replayed from verified baseline objects"
    parent["comparison_note"] += " Revision 2 also reviews target labels and adds three baseline-backed Anko reads; reconstruction is explicitly recorded."
    config = json.loads((output / "training_config.json").read_text())
    config["max_seq_length"] = ((parent["totals"]["maximum_tokens"] + 4095) // 4096) * 4096
    write_json(output / "training_config.json", config)
    validation = json.loads((output / "validation.json").read_text())
    validation.update(targeted_label_review_applied=True, repository_citations_covered_by_retained_reads=citations,
                      reconstructed_baseline_reads=3, no_intermediate_assistant_stop_turns=True)
    write_json(output / "validation.json", validation)
    table = "\n".join(f"| {row['task_id']} | {row['label_source']} | {row['total_tokens']:,} | {row['supervised_tokens']:,} |" for row in parent["tasks"])
    (output / "README.md").write_text(f"""# E2 训练物料：审查修订版 v2

实验仍为 E2：删除教师 thinking 正文，工具调用和最终无编号 Ledger 为主要监督目标；训练与推理均关闭 thinking。E0 为 base + thinking，E1 为包含教师 thinking 的 LoRA。保持原 13 题、8 道教师来源标签与 5 道人工来源标签。

v1 完整保留在 ../{source.name}/；本版是根据逐题审查制作的训练标签修订，不修改原教师结果或人工标注源文件。逐项 before/after 及原因见 review_changes.json；原 E1 和 E2 v1 训练输入/manifest 保留在 provenance/，仅用于溯源，不加载进训练。

## 修订内容

- Termenv：首条义务仅保留新子包要求，具体 API 由后续义务分别覆盖。
- FastAPI：为实例统计隔离补上已有的实例化/生命周期代码引用；该规则仍是专家推导，代码引用不是现有统计实现的直接证明。
- Arktype、Anko：summary 直接描述行为，删除人工修订历史及内部编号。
- Anko：引用实际返回值断言及 nil 输出错误案例。最终 Ledger 前新增一条带三次 read 调用的简短行动说明，读取 vm/vmStmt.go:34–49、vm/main_test.go:198–219、vm/vm_test.go:80–89。
- Valibot：开放问题只讨论公开契约，不猜测隐藏测试。
- Adaptix：删除题面已明确 first-wins-per-field 的重复问题。
- Scriggo：义务不再提前排除开放问题中尚待确定的 receiver 类别。
- Yjs：共享类型/subdoc 的保留范围限定到 allow 或无冲突场景，并补齐 integration 引用，允许任务要求的 error 模式拒绝冲突。
- Optique、LangChain、TOMLKit、SQL Formatter、Mobly：维持 v1 标签与轨迹。

Anko 新增行动说明是人工撰写的训练用重建，不是原教师日志或新的模型采样。三条工具返回从公开仓库冻结 commit 的 Git 对象回放，逐文件 SHA-256 与原证据核对一致；没有写入答案提示、候选修复或隐藏评测内容。只增加一个工具交互回合，没有独立的 assistant 解释终止回合，也没有追加 user 指令。

## 训练和校验

每轮输入 {parent['totals']['tokens']:,} tokens；监督 {parent['totals']['supervised_tokens']:,} tokens；最长 {parent['totals']['maximum_tokens']:,} tokens。全部 tokenized 标签重新生成，空 thinking 前缀和工具观察继续屏蔽损失；简短行动说明、工具调用及 Ledger 参与监督。

检查包括 13 题的 schema、引用、调用/返回关系、无 thinking、无中间终止回合、{citations} 处引用的读取可见性，以及原生模板与非 thinking 推理前缀一致性。自动检查和针对性审查不代表整批已成为完整语义金标准。训练入口会在加载 GPU 模型前再以 Transformers 校验 token IDs 和监督 mask。

沿用 E1/E2 v1 的 rank16/alpha32、BF16、学习率1e-4、3 epoch、batch1、seed3407。配置见 training_config.json，必须从固定 revision 的原始 base 开始，不续训 E1 adapter。推理采用 inference_config.json 的 chat_template_kwargs.enable_thinking=false。

```bash
python lora_train_trajectories.py --materials . --model-path /path/to/c202236235762e1c871ad0ccb60c8ee5ba337b9a --config training_config.json --output /path/to/new-e2-run
```

本次只制作物料，未启动训练或修改远端服务。13 题仍属于已用于开发的 30 题；其余题不自动构成独立论文测试集。E2/v2 包含轨迹及标签清理，不能把与 E1 的所有差异归因于删除 thinking。

| 题目 | 原标签来源 | 输入 tokens | 监督 tokens |
|---|---|---:|---:|
{table}

在项目根目录重建：python -m a_agent_finetune.revise_e2_materials --output 新目录。使用已有固定 tokenizer、依赖及本地 Git 基线缓存，不调用模型或网络。
""")
    parent["file_sha256"] = {str(path.relative_to(output)): sha(path) for path in sorted(output.rglob("*"))
                             if path.is_file() and path != output / "manifest.json"}
    write_json(output / "manifest.json", parent)
    return parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = revise(args.source.resolve(), args.output.resolve())
    print(json.dumps(manifest["totals"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
