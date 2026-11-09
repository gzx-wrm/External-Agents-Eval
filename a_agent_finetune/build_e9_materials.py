"""Package E9: the E6 13 normal trajectories plus teacher-written visible notes.

Input is the output of ``compress_teacher_thinking.py``.  Every row must match
the E6 row except for the visible content of non-final assistant turns and,
when closing notes were generated, a closing note before the unchanged final
Ledger.  The controller's own JSON extraction must recover exactly the E6
Ledger from that final turn.  Protocol files and training hyperparameters are
copied from E6 unchanged.  No model is called.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment
from tokenizers import Tokenizer

try:
    from .build_e2_materials import validate_conversation
    from .build_e_teacher_trajectory import dump, encode_sample, sha, write_json
except ImportError:
    from build_e2_materials import validate_conversation
    from build_e_teacher_trajectory import dump, encode_sample, sha, write_json
from external_agents_eval.workflows.solve_review_repair.v5_contracts import parse_json_object

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_NOTES = HERE / "data/e9_rationale_notes_v1"
DEFAULT_E6 = HERE / "data/e6_normal13_loopheavy_v1"
DEFAULT_OUTPUT = HERE / "data/e9_normal13_rationale_v1"
PROTOCOL_FILES = (
    "system-prompt.txt", "tools.json", "native_chat_template.jinja", "training_chat_template.jinja",
    "tokenizer_config.json", "tokenizer.json", "spec_output_schema.json",
)


def check_final_turn(task: str, content: str, reference: str) -> bool:
    """Return whether the final turn has a closing note; raise unless it yields the E6 Ledger."""

    if content == reference:
        return False
    note, separator, ledger = content.partition("\n\n{")
    if not separator or not note.strip() or "{" in note or "}" in note or "{" + ledger != reference:
        raise ValueError(f"{task}: final turn is neither the E6 Ledger nor closing note + E6 Ledger")
    if parse_json_object(content) != json.loads(reference):
        raise ValueError(f"{task}: controller JSON extraction does not recover the E6 Ledger")
    return True


def check_against_e6(sample: dict, reference: dict) -> tuple[int, bool]:
    """Return (changed notes, closing note present); raise on any other difference."""

    if sample["tools"] != reference["tools"] or len(sample["messages"]) != len(reference["messages"]):
        raise ValueError(f"{sample['task_id']}: tools or message count differ from E6")
    last, changed = len(sample["messages"]) - 1, 0
    closing = check_final_turn(sample["task_id"], sample["messages"][last]["content"],
                               reference["messages"][last]["content"])
    for index, (message, other) in enumerate(zip(sample["messages"], reference["messages"], strict=True)):
        if message["role"] != other["role"] or message.get("tool_calls") != other.get("tool_calls"):
            raise ValueError(f"{sample['task_id']}: message {index} structure differs from E6")
        if index == last:
            continue
        note = message["role"] == "assistant" and message.get("tool_calls")
        if message["content"] != other["content"]:
            if not note:
                raise ValueError(f"{sample['task_id']}: message {index} content differs from E6")
            changed += 1
        if note and not message["content"].strip():
            raise ValueError(f"{sample['task_id']}: message {index} has no visible note")
    return changed, closing


def build(notes: Path, e6: Path, output: Path, experiment: str = "E9") -> dict:
    import numpy as np

    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    e6_manifest = json.loads((e6 / "manifest.json").read_text())
    for name, expected in e6_manifest["file_sha256"].items():
        if sha(e6 / name) != expected:
            raise ValueError(f"E6 file changed: {name}")
    notes_run = json.loads((notes / "run.json").read_text())
    e6_rows = {row["task_id"]: row for row in map(json.loads, (e6 / "train.jsonl").read_text().splitlines())}
    samples = [json.loads(line) for line in (notes / "train.jsonl").read_text().splitlines()]
    tasks = [task["task_id"] for task in e6_manifest["tasks"]]
    if [sample["task_id"] for sample in samples] != tasks:
        raise ValueError("E9 rows must be the E6 tasks in E6 order")

    output.mkdir(parents=True)
    for name in ("protocol", "tokenized", "records", "provenance"):
        (output / name).mkdir()
    for name in PROTOCOL_FILES:
        shutil.copyfile(e6 / "protocol" / name, output / "protocol" / name)
    for name in ("lora_train_trajectories.py", "split_manifest.json", "inference_config.json"):
        shutil.copyfile(e6 / name, output / name)
    shutil.copyfile(e6 / "manifest.json", output / "provenance/e6_manifest.json")
    shutil.copyfile(notes / "run.json", output / "provenance/notes_run.json")
    shutil.copyfile(Path(__file__), output / "provenance/build_e9_materials.py")
    shutil.copyfile(HERE / "compress_teacher_thinking.py", output / "provenance/compress_teacher_thinking.py")
    native = (output / "protocol/native_chat_template.jinja").read_text()
    training_template = (output / "protocol/training_chat_template.jinja").read_text()
    tools = json.loads((output / "protocol/tools.json").read_text())
    tokenizer = Tokenizer.from_file(str(output / "protocol/tokenizer.json"))
    environment = Environment()
    environment.filters["tojson"] = lambda value, **kwargs: json.dumps(value, ensure_ascii=False, **kwargs)

    rows, written = [], []
    for e6_task, sample in zip(e6_manifest["tasks"], samples, strict=True):
        task = sample["task_id"]
        reference = e6_rows[task]
        sample = {"messages": sample["messages"], "tools": sample["tools"]}
        changed, closing = check_against_e6({**sample, "task_id": task}, reference)
        validate_conversation(sample)
        arrays, lengths = encode_sample(sample, native, training_template, tokenizer)
        lengths["final_ledger_turn_tokens"] = lengths.pop("final_ledger_turn_tokens_including_original_thinking")
        # Non-thinking inference must start exactly where the supervised Ledger starts.
        prompt = environment.from_string(native).render(
            messages=sample["messages"][:-1], tools=tools, add_generation_prompt=True, enable_thinking=False)
        full = environment.from_string(native).render(**sample, add_generation_prompt=False, enable_thinking=False)
        if not full.startswith(prompt) or full[len(prompt):] != sample["messages"][-1]["content"] + "<|im_end|>\n":
            raise ValueError(f"E9 training/inference boundary differs: {task}")
        np.savez_compressed(output / "tokenized" / f"{task}.npz", **arrays)
        sample.update({key: reference[key] for key in ("sample_id", "task_id", "view", "trajectory_complete",
                                                       "label_source")})
        written.append(sample)
        write_json(output / "records" / f"{task}.json", {
            "task_id": task, "notes_changed": changed, "closing_note": closing, "lengths": lengths,
            "e6_lengths": {key: e6_task[key] for key in ("total_tokens", "supervised_tokens")},
            "thinking_in_training": False, "visible_notes_in_training": True,
            "training_inference_boundary_verified": True,
        })
        rows.append({
            "row_index": len(rows), "sample_id": reference["sample_id"], "task_id": task,
            "view": "normal", "trajectory_complete": True, "label_source": reference["label_source"],
            "visible_notes": changed, "closing_note": closing,
            "e6_supervised_tokens": e6_task["supervised_tokens"], **lengths,
        })
        print(f"{task}: {lengths['total_tokens']} tokens / {lengths['supervised_tokens']} supervised "
              f"(E6 {e6_task['supervised_tokens']}), {changed} notes")

    (output / "train.jsonl").write_text("".join(dump(sample) + "\n" for sample in written))
    config = json.loads((e6 / "training_config.json").read_text())
    max_tokens = max(row["total_tokens"] for row in rows)
    if max_tokens > config["max_seq_length"]:
        raise ValueError(f"Longest E9 row {max_tokens} exceeds E6 max_seq_length")
    write_json(output / "training_config.json", config)
    totals = {
        "tokens": sum(row["total_tokens"] for row in rows),
        "supervised_tokens": sum(row["supervised_tokens"] for row in rows),
        "e6_supervised_tokens": sum(row["e6_supervised_tokens"] for row in rows),
        "maximum_tokens": max_tokens,
        "visible_notes": sum(row["visible_notes"] for row in rows),
        "closing_notes": sum(row["closing_note"] for row in rows),
    }
    manifest = {
        "schema_version": f"{experiment.lower()}_normal13_rationale_v1", "experiment": experiment,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": e6_manifest["model"], "model_revision": e6_manifest["model_revision"],
        "source_e6_manifest_sha256": sha(e6 / "manifest.json"),
        "notes_prompt_version": notes_run["prompt_version"], "notes_teacher": notes_run["teacher"],
        "sample_count": len(rows), "normal_rows": len(rows), "recovery_rows": 0,
        "train_task_count": 13, "validation_task_count": 3, "test_task_count": 14,
        "enable_thinking": False, "thinking_in_training": False, "visible_notes_in_training": True,
        "initialization": "original_base_plus_new_lora",
        "loss_policy": "E6 policy; teacher-written visible notes before tool calls"
                       + (" and closing notes before the final Ledger" if totals["closing_notes"] else "")
                       + " are additionally supervised",
        "length_policy": "No truncation, splitting or packing",
        "totals": totals, "tasks": rows, "split_manifest": "split_manifest.json",
    }
    write_json(output / "validation.json", {
        "rows_match_e6_except_notes": len(rows), "training_inference_boundary_verified": len(rows),
        "controller_extracts_e6_ledger": len(rows),
        "tokenization_loss_masks_verified": len(rows), "totals": totals,
    })
    closing_text = (
        f"最终回合在 Ledger JSON 前另有 2–3 句收尾说明（{totals['closing_notes']} 条），说明证据为何已足够、剩余不确定性记入 open_questions 并输出 Ledger；"
        "说明中不含花括号，控制器的 JSON 抽取仍得到与 E6 完全相同的 Ledger。"
        if totals["closing_notes"] else "最终回合仍是纯 Ledger JSON。")
    (output / "README.md").write_text(f"""# {experiment}：E6 + 教师可见决策说明

{experiment} 与 E6 使用相同的 13 道训练题、相同的工具调用、工具观察和最终 Ledger，以及相同的训练超参，并同样从原始 base 新建 LoRA。差别是每个非最终工具调用回合前加入 1–3 句由教师（{notes_run['teacher']['model']}，thinking 开启）根据自己原 thinking 压缩的可见决策说明，这些说明参与 loss。{closing_text}

说明生成见 `provenance/compress_teacher_thinking.py` 与 `provenance/notes_run.json`。共 {totals['visible_notes']} 条说明；总输入 {totals['tokens']:,} tokens，监督 {totals['supervised_tokens']:,} tokens（E6 为 {totals['e6_supervised_tokens']:,}）；最长 {max_tokens:,} tokens。推理仍关闭 thinking。

```bash
python lora_train_trajectories.py --materials <本目录> --model-path /path/to/{e6_manifest['model_revision']} --config <本目录>/training_config.json --output <新训练输出目录>
```
""")
    manifest["file_sha256"] = {
        str(path.relative_to(output)): sha(path) for path in sorted(output.rglob("*")) if path.is_file()
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notes", type=Path, default=DEFAULT_NOTES)
    parser.add_argument("--e6", type=Path, default=DEFAULT_E6)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--experiment", default="E9")
    args = parser.parse_args()
    manifest = build(args.notes.resolve(), args.e6.resolve(), args.output.resolve(), args.experiment)
    print(json.dumps(manifest["totals"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
