"""Build the E6 normal-only, loop-heavy 13-task training materials.

The source conversations and selected Ledgers are reused from the frozen
30-task teacher package.  This builder removes teacher thinking with the same
E2 transformation, applies the already recorded mechanical citation fixes,
and writes only normal training rows.  It does not call a model.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

from jsonschema import Draft202012Validator
from jinja2 import Environment
from tokenizers import Tokenizer

try:
    from .build_e2_materials import no_thinking_template, transform_sample, validate_conversation
    from .build_e_teacher_trajectory import encode_sample, sha, dump, write_json
    from .run_f_inference import validate_output
except ImportError:
    from build_e2_materials import no_thinking_template, transform_sample, validate_conversation
    from build_e_teacher_trajectory import encode_sample, sha, dump, write_json
    from run_f_inference import validate_output


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_SOURCE = HERE / "data/e_teacher_trajectory_30_human_latest_v1"
DEFAULT_OUTPUT = HERE / "data/e6_normal13_loopheavy_v1"
E2_PROTOCOL = HERE / "data/e2_no_thinking_success13_teacher8_human5_v2/protocol"
OBSERVED_DATA = HERE / "data/f_teacher_observed_30_v4"

TRAIN_TASKS = [
    # E-loop tasks selected for the loop-heavy training distribution.
    "aiomonitor-task-snapshots-diff",
    "expr-try-catch-errors",
    "testem-per-launcher-reports",
    "yaegi-go-embed-directives",
    "boa-hierarchical-evaluation-cancellation",
    "vulture-persistent-analysis-cache",
    "obsidian-linter-link-format-conversion",
    # Keep both Obsidian tasks in one split.
    "obsidian-linter-auto-table-of-contents",
    # Five normal tasks already used by the previous recovery experiments.
    "adaptix-name-mapping-aliases",
    "anko-typed-variable-bindings",
    "arktype-json-schema-refs-dependencies",
    "langchain-request-coalescing",
    "termenv-preserve-ansi-resets",
]
VALIDATION_TASKS = [
    "tomlkit-toml-table-converters",
    "valibot-recursive-schema-composition",
    "optique-conditional-option-dependencies",
]
TEST_TASKS = [
    "abs-stepped-slices",
    "cliffy-config-file-parsing",
    "fastapi-implicit-head-options",
    "gql-incremental-graphql-delivery",
    "meriyah-explicit-resource-declarations",
    "numba-stencil-boundary-modes",
    "sqlite-utils-safe-import-checkpoints",
    "tengo-callable-instance-isolation",
    "ts-pattern-match-each",
    "sql-formatter-bigquery-pipe-formatting",
    "mobly-grouped-test-barriers",
    "scriggo-method-declarations",
    "yjs-map-conflict-detection",
    "fastapi-deprecation-response-headers",
]


def apply_recorded_corrections(target: dict, record: dict) -> list[dict]:
    """Apply only corrections with a concrete replacement source.

    Some historical audit entries intentionally recorded ``after: null`` to
    explain why an old citation was retained.  Those are not edits.
    """

    applied = []
    for correction in record.get("prior_f_citation_corrections_not_applied", []):
        after = correction.get("after")
        if after is None:
            continue
        source = target[correction["field"]][correction["entry_index"]]["sources"][correction["source_index"]]
        if source != correction["before"]:
            raise ValueError(f"Citation correction does not match frozen Ledger: {record['task_id']}")
        source.clear()
        source.update(after)
        applied.append(correction)
    return applied


def copy_protocol(output: Path) -> tuple[str, str, Tokenizer]:
    import numpy as np  # noqa: F401  # keep the dependency check close to build

    (output / "protocol").mkdir(parents=True)
    for name in (
        "system-prompt.txt", "tools.json", "native_chat_template.jinja",
        "tokenizer_config.json", "spec_output_schema.json",
    ):
        shutil.copyfile(E2_PROTOCOL / name, output / "protocol" / name)
    native = (output / "protocol/native_chat_template.jinja").read_text()
    training = no_thinking_template(native)
    (output / "protocol/training_chat_template.jinja").write_text(training)
    tokenizer_path = HERE / ".cache/tokenizer-qwen3.5-9b.json"
    shutil.copyfile(tokenizer_path, output / "protocol/tokenizer.json")
    runner = (HERE / "lora_train_trajectories.py").read_text()
    runner = runner.replace(
        "enable_thinking=True, add_generation_prompt=False,",
        "enable_thinking=manifest.get(\"enable_thinking\", True), add_generation_prompt=False,",
    )
    (output / "lora_train_trajectories.py").write_text(runner)
    return native, training, Tokenizer.from_file(str(tokenizer_path))


def build(source: Path, output: Path) -> dict:
    import numpy as np

    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    source_manifest = json.loads((source / "manifest.json").read_text())
    # The 30-task source predates the per-file hash map used by E2/E5.  Its
    # manifest still pins the train file and every selected record is checked
    # again below, so absence of this optional map is expected.
    for name, expected in source_manifest.get("file_sha256", {}).items():
        path = source / name
        if path.exists() and sha(path) != expected:
            raise ValueError(f"Frozen source file changed: {name}")

    split_sets = {
        "train": TRAIN_TASKS,
        "validation": VALIDATION_TASKS,
        "test": TEST_TASKS,
    }
    all_split_tasks = [task for tasks in split_sets.values() for task in tasks]
    if len(TRAIN_TASKS) != 13 or len(VALIDATION_TASKS) != 3 or len(TEST_TASKS) != 14:
        raise ValueError("E6 split must be 13/3/14")
    if len(set(all_split_tasks)) != 30:
        raise ValueError("E6 split contains duplicate or missing tasks")

    source_rows = [json.loads(line) for line in (source / "train.jsonl").read_text().splitlines()]
    source_manifest_tasks = source_manifest["tasks"]
    if len(source_rows) != len(source_manifest_tasks):
        raise ValueError("Source train rows and manifest differ")
    row_by_task = {row["task_id"]: sample for row, sample in zip(source_manifest_tasks, source_rows, strict=True)}
    manifest_row_by_task = {row["task_id"]: row for row in source_manifest_tasks}
    if set(row_by_task) != set(all_split_tasks):
        raise ValueError("E6 split does not cover the frozen 30-task source")

    output.mkdir(parents=True)
    for name in ("records", "labels", "tokenized", "provenance", "examples"):
        (output / name).mkdir()
    native, training_template, tokenizer = copy_protocol(output)
    schema = Draft202012Validator(json.loads((output / "protocol/spec_output_schema.json").read_text()))
    tools = json.loads((output / "protocol/tools.json").read_text())
    observed_records = {
        task: json.loads((OBSERVED_DATA / "records" / f"{task}.json").read_text())
        for task in TRAIN_TASKS
    }
    source_records = {
        task: json.loads((source / "records" / f"{task}.json").read_text())
        for task in TRAIN_TASKS
    }

    shutil.copyfile(source / "train.jsonl", output / "provenance/source_30_train.jsonl")
    shutil.copyfile(source / "manifest.json", output / "provenance/source_30_manifest.json")
    shutil.copyfile(Path(__file__), output / "provenance/build_e6_materials.py")

    rows, samples, audits = [], [], []
    for task in TRAIN_TASKS:
        old_sample = row_by_task[task]
        source_record = source_records[task]
        observed = observed_records[task]
        sample, changes = transform_sample(copy.deepcopy(old_sample))
        target = json.loads(sample["messages"][-1]["content"])
        corrections = apply_recorded_corrections(target, source_record)
        sample["messages"][-1]["content"] = dump(target)
        _, checks = validate_output(dump(target), observed, schema)
        if not checks["schema_valid"] or not checks["sources_valid"]:
            raise ValueError(f"Invalid E6 Ledger for {task}: {checks}")
        validate_conversation(sample)

        arrays, lengths = encode_sample(sample, native, training_template, tokenizer)
        lengths["final_ledger_turn_tokens"] = lengths.pop("final_ledger_turn_tokens_including_original_thinking")
        env = Environment()
        env.filters["tojson"] = lambda value, **kwargs: json.dumps(value, ensure_ascii=False, **kwargs)
        prompt = env.from_string(native).render(
            messages=sample["messages"][:-1], tools=tools,
            add_generation_prompt=True, enable_thinking=False,
        )
        full = env.from_string(native).render(
            **sample, add_generation_prompt=False, enable_thinking=False,
        )
        if not full.startswith(prompt) or full[len(prompt):] != dump(target) + "<|im_end|>\n":
            raise ValueError(f"E6 training/inference boundary differs: {task}")

        sample.update({
            "sample_id": f"normal-{task}", "task_id": task, "view": "normal",
            "trajectory_complete": True, "label_source": source_record["label_source"],
        })
        np.savez_compressed(output / "tokenized" / f"{task}.npz", **arrays)
        write_json(output / "labels" / f"{task}.json", target)
        write_json(output / "records" / f"{task}.json", {
            "task_id": task,
            "sample_id": sample["sample_id"],
            "view": "normal",
            "label_source": source_record["label_source"],
            "source_row_index": manifest_row_by_task[task]["row_index"],
            "source_record": source_record,
            "e6_transform": changes,
            "citation_corrections_applied": corrections,
            "lengths": lengths,
            "label_validation": checks,
            "thinking_in_training": False,
            "training_inference_boundary_verified": True,
        })
        rows.append({
            "row_index": len(rows), "sample_id": sample["sample_id"], "task_id": task,
            "view": "normal", "trajectory_complete": True,
            "label_source": source_record["label_source"],
            "source_row_index": manifest_row_by_task[task]["row_index"],
            "citation_corrections_applied": len(corrections),
            "removed_thinking_characters": changes["removed_thinking_characters"],
            "removed_redundant_tool_calls": len(changes["removed_redundant_tool_calls"]),
            "assistant_messages": sum(m["role"] == "assistant" for m in sample["messages"]),
            "tool_calls": sum(len(m.get("tool_calls", [])) for m in sample["messages"]),
            **lengths,
        })
        audits.append({"task_id": task, "checks": checks, "transform": changes,
                       "citation_corrections_applied": corrections})
        samples.append(sample)
        if len(samples) in (1, len(TRAIN_TASKS)):
            write_json(output / "examples" / f"{sample['sample_id']}.json", sample)
        print(f"{task}: {lengths['total_tokens']} tokens / {lengths['supervised_tokens']} supervised")

    (output / "train.jsonl").write_text("".join(dump(sample) + "\n" for sample in samples))
    max_tokens = max(row["total_tokens"] for row in rows)
    max_seq_length = ((max_tokens + 4095) // 4096) * 4096
    config = {
        "epochs": 1, "learning_rate": 2e-5, "scheduler": "cosine", "warmup_steps": 3,
        "lora_rank": 16, "lora_alpha": 32, "lora_dropout": 0.0, "weight_decay": 0.0,
        "max_grad_norm": 1.0, "batch_size": 1, "gradient_accumulation_steps": 1,
        "max_seq_length": max_seq_length, "seed": 3407, "precision": "bf16",
        "optimizer": "AdamW8bit", "shuffle_each_epoch": True, "packing": False,
        "truncation": False, "assistant_only_loss": True, "ledger_loss_weight": 1.0,
        "checkpoint_each_epoch": True, "max_runtime_seconds": 21600,
        "checkpoint_at_steps": [len(TRAIN_TASKS)],
    }
    write_json(output / "training_config.json", config)
    write_json(output / "inference_config.json", {
        "chat_template_kwargs": {"enable_thinking": False},
        "restart_from_base_model": True,
        "note": "E6 uses the same non-thinking inference protocol as E5; no recovery rows are trained.",
    })

    split_manifest = {
        "schema_version": "e6_split_v1",
        "source_manifest": "provenance/source_30_manifest.json",
        "train": TRAIN_TASKS, "validation": VALIDATION_TASKS, "test": TEST_TASKS,
        "same_repository_grouping": {
            "obsidian": ["obsidian-linter-link-format-conversion", "obsidian-linter-auto-table-of-contents"],
            "fastapi": ["fastapi-implicit-head-options", "fastapi-deprecation-response-headers"],
        },
        "notes": "Development split for E6; all 30 tasks originated from the existing development pool.",
    }
    write_json(output / "split_manifest.json", split_manifest)
    totals = {
        "tokens": sum(row["total_tokens"] for row in rows),
        "supervised_tokens": sum(row["supervised_tokens"] for row in rows),
        "maximum_tokens": max_tokens,
        "minimum_tokens": min(row["total_tokens"] for row in rows),
        "median_tokens": median(row["total_tokens"] for row in rows),
        "removed_thinking_characters": sum(row["removed_thinking_characters"] for row in rows),
        "removed_redundant_tool_calls": sum(row["removed_redundant_tool_calls"] for row in rows),
        "citation_corrections_applied": sum(row["citation_corrections_applied"] for row in rows),
        "tool_calls": sum(row["tool_calls"] for row in rows),
        "fits_32k": sum(row["fits_32k"] for row in rows),
        "fits_64k": sum(row["fits_64k"] for row in rows),
        "fits_128k": sum(row["fits_128k"] for row in rows),
    }
    manifest = {
        "schema_version": "e6_normal_only_loopheavy_v1",
        "experiment": "E6",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": source_manifest["model"], "model_revision": source_manifest["model_revision"],
        "source_dataset": str(source.relative_to(ROOT)),
        "source_manifest_sha256": sha(source / "manifest.json"),
        "sample_count": len(rows), "train_task_count": 13,
        "validation_task_count": 3, "test_task_count": 14,
        "enable_thinking": False, "thinking_in_training": False,
        "recovery_rows": 0, "normal_rows": len(rows),
        "label_source_counts": dict(Counter(row["label_source"] for row in rows)),
        "label_selection": "Reuse selected 30-task Ledger; apply only recorded mechanical citation corrections",
        "split": "development_only",
        "loss_policy": "Assistant content, retained tool-call syntax and turn ends supervised; system/user/tool observations/headers/empty think prefixes -100",
        "length_policy": "No truncation, splitting, summarization or packing; teacher thinking and redundant unavailable-rg calls removed",
        "totals": totals,
        "tasks": rows,
        "validation": {
            "all_normal_rows": True, "no_recovery_rows": True,
            "schema_and_source_checks_passed": len(rows),
            "training_inference_boundary_verified": len(rows),
            "tokenization_loss_masks_verified": len(rows),
        },
        "split_manifest": "split_manifest.json",
    }
    write_json(output / "validation.json", manifest["validation"] | {"totals": totals})
    write_json(output / "label_audit.json", {"scope": "E6 mechanical validation; semantic completeness is not re-audited", "tasks": audits})
    write_json(output / "training_contract.json", {
        "model": source_manifest["model"], "revision": source_manifest["model_revision"],
        "thinking": False, "dataset": "train.jsonl", "format": "conversational messages + tools",
        "chat_template": "protocol/training_chat_template.jinja", "assistant_only_loss": True,
        "packing": False, "truncation_allowed": False,
        "required_sequence_length_at_least": max_tokens,
        "suggested_max_length_for_complete_data": max_seq_length,
        "recovery_rows": 0,
        "split": "development_only: 13 train / 3 validation / 14 test task assignment is recorded in split_manifest.json",
        "hardware": "Full-sequence LoRA feasibility still needs a GPU smoke test; do not silently truncate.",
    })
    (output / "README.md").write_text(f"""# E6：循环题偏置的 normal-only 训练物料

E6 用现有 30 题教师轨迹物料重新划分任务，训练集为 13 题，其中 7 题来自 E4/E4_1 中出现循环的题目；验证集 3 题，测试集 14 题。训练只包含 normal 教师轨迹，不包含任何 recovery、short recovery 或失败循环前缀。

本批次不重新调用教师模型。每题沿用 30 题物料中的选定 Ledger：有人工 Ledger 的题目保留人工版本，没有人工版本的题目保留教师原生版本。对历史记录中已经明确的机械引用修正执行了最小修正；没有重新做语义金标准审查。

E6 使用与 E5 相同的非 thinking 训练/推理约定，删除教师 thinking，并移除重复的 unavailable-rg 环境故障调用。完整 normal 轨迹不截断、不切分、不 packing。最长训练样本为 {max_tokens:,} tokens，训练配置使用 max_seq_length={max_seq_length:,}。

训练集不含 recovery 行，`recovery_rows=0`。本物料仍来自已经用于开发和人工标注的 30 题池，不能作为独立盲测集；具体 13/3/14 划分见 `split_manifest.json`。

训练入口：

```bash
python lora_train_trajectories.py --materials <本目录> --model-path /path/to/{source_manifest['model_revision']} --config <本目录>/training_config.json --output <新训练输出目录>
```

总输入 tokens：{totals['tokens']:,}；监督 tokens：{totals['supervised_tokens']:,}；引用机械修正：{totals['citation_corrections_applied']} 处。
""")

    manifest["file_sha256"] = {
        str(path.relative_to(output)): sha(path)
        for path in sorted(output.rglob("*")) if path.is_file()
    }
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
