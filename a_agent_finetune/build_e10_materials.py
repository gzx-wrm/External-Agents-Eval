"""Package E10: the 13 E9_1 rows plus the 47 new train60 rows (60 rows).

The 13 E9_1 rows are reused unchanged.  Each new row comes from
``compress_teacher_thinking.py --closing-notes`` over the E10 source package
and must match that source (after the E2 thinking removal) except for the
visible notes and the closing note before the unchanged label Ledger; the
controller's JSON extraction must recover exactly that Ledger.  Protocol
files and hyperparameters come from E9_1 / its lr 1e-4 run; only
``max_seq_length`` grows to fit the longest complete row.  No truncation,
splitting or packing.  ``--only`` writes a subset (e.g. a memory smoke test
of the longest row).  No model is called.
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
    from .build_e2_materials import transform_sample, validate_conversation
    from .build_e9_materials import check_against_e6
    from .build_e_teacher_trajectory import dump, encode_sample, sha, write_json
except ImportError:
    from build_e2_materials import transform_sample, validate_conversation
    from build_e9_materials import check_against_e6
    from build_e_teacher_trajectory import dump, encode_sample, sha, write_json

HERE = Path(__file__).resolve().parent
DEFAULT_E9_1 = HERE / "data/e9_1_normal13_rationale_closing_v1"
DEFAULT_SOURCE = HERE / "data/e10_new47_source_v1"
DEFAULT_NOTES = HERE / "data/e10_new47_notes_v1"
DEFAULT_CONFIG = HERE / "runs/lora-train/qwen35-9b-e9_1-lr1e4-ep3-a800-20261013-r1/config.json"
DEFAULT_OUTPUT = HERE / "data/e10_train60_rationale_closing_v1"
PROTOCOL_FILES = (
    "system-prompt.txt", "tools.json", "native_chat_template.jinja", "training_chat_template.jinja",
    "tokenizer_config.json", "tokenizer.json", "spec_output_schema.json",
)


def verified(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, expected in manifest["file_sha256"].items():
        if sha(directory / name) != expected:
            raise ValueError(f"{directory.name}: file changed since packaging: {name}")
    return manifest


def build(e9_1: Path, source: Path, notes: Path, config_path: Path, output: Path,
          only: list[str] | None = None) -> dict:
    import numpy as np

    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    e9_1_manifest, source_manifest = verified(e9_1), verified(source)
    notes_run = json.loads((notes / "run.json").read_text())
    if notes_run["source_manifest_sha256"] != sha(source / "manifest.json"):
        raise ValueError("Notes were generated from a different source package")
    for name in ("system-prompt.txt", "tools.json", "spec_output_schema.json"):
        if (e9_1 / "protocol" / name).read_bytes() != (source / "protocol" / name).read_bytes():
            raise ValueError(f"E9_1 and E10 source protocols differ: {name}")

    old = [json.loads(line) for line in (e9_1 / "train.jsonl").read_text().splitlines()]
    old_rows = {row["task_id"]: row for row in e9_1_manifest["tasks"]}
    sources = {row["task_id"]: json.loads(line) for row, line in zip(
        source_manifest["tasks"], (source / "train.jsonl").read_text().splitlines(), strict=True)}
    noted = {row["task_id"]: row for row in map(json.loads, (notes / "train.jsonl").read_text().splitlines())}
    if set(noted) != set(sources):
        raise ValueError("Notes do not cover exactly the E10 source tasks")
    split = json.loads((e9_1 / "split_manifest.json").read_text())
    if set(split["validation"] + split["test"]) & set(sources):
        raise ValueError("A new training task is in the held-out split")

    entries = [("e9_1", row["task_id"], row) for row in old] + \
              [("train60", row["task_id"], noted[row["task_id"]]) for row in source_manifest["tasks"]]
    if only:
        entries = [entry for entry in entries if entry[1] in only]
        if len(entries) != len(set(only)):
            raise ValueError("--only names unknown tasks")

    output.mkdir(parents=True)
    for name in ("protocol", "tokenized", "records", "provenance"):
        (output / name).mkdir()
    for name in PROTOCOL_FILES:
        shutil.copyfile(e9_1 / "protocol" / name, output / "protocol" / name)
    shutil.copyfile(e9_1 / "lora_train_trajectories.py", output / "lora_train_trajectories.py")
    shutil.copyfile(e9_1 / "inference_config.json", output / "inference_config.json")
    for path, name in ((e9_1 / "manifest.json", "e9_1_manifest.json"), (source / "manifest.json", "e10_source_manifest.json"),
                       (notes / "run.json", "e10_notes_run.json"), (Path(__file__), "build_e10_materials.py")):
        shutil.copyfile(path, output / "provenance" / name)
    native = (output / "protocol/native_chat_template.jinja").read_text()
    training_template = (output / "protocol/training_chat_template.jinja").read_text()
    tools = json.loads((output / "protocol/tools.json").read_text())
    tokenizer = Tokenizer.from_file(str(output / "protocol/tokenizer.json"))
    environment = Environment()
    environment.filters["tojson"] = lambda value, **kwargs: json.dumps(value, ensure_ascii=False, **kwargs)

    rows, written = [], []
    for origin, task, sample in entries:
        sample = {"messages": sample["messages"], "tools": sample["tools"]}
        if origin == "train60":
            reference, _ = transform_sample(sources[task])
            reference = {"messages": reference["messages"], "tools": reference["tools"]}
            changed, closing = check_against_e6({**sample, "task_id": task}, reference)
            if not closing:
                raise ValueError(f"{task}: new rows must carry a closing note")
            label_source = source_manifest_row(source_manifest, task)["label_source"]
        else:
            changed, closing = old_rows[task]["visible_notes"], old_rows[task]["closing_note"]
            label_source = old_rows[task]["label_source"]
        validate_conversation(sample)
        arrays, lengths = encode_sample(sample, native, training_template, tokenizer)
        lengths["final_ledger_turn_tokens"] = lengths.pop("final_ledger_turn_tokens_including_original_thinking")
        prompt = environment.from_string(native).render(
            messages=sample["messages"][:-1], tools=tools, add_generation_prompt=True, enable_thinking=False)
        full = environment.from_string(native).render(**sample, add_generation_prompt=False, enable_thinking=False)
        if not full.startswith(prompt) or full[len(prompt):] != sample["messages"][-1]["content"] + "<|im_end|>\n":
            raise ValueError(f"E10 training/inference boundary differs: {task}")
        np.savez_compressed(output / "tokenized" / f"{task}.npz", **arrays)
        sample.update(sample_id=f"normal-{task}", task_id=task, view="normal",
                      trajectory_complete=True, label_source=label_source)
        written.append(sample)
        write_json(output / "records" / f"{task}.json", {
            "task_id": task, "origin": origin, "label_source": label_source, "visible_notes": changed,
            "closing_note": closing, "lengths": lengths, "thinking_in_training": False,
            "training_inference_boundary_verified": True,
        })
        rows.append({"row_index": len(rows), "sample_id": sample["sample_id"], "task_id": task, "origin": origin,
                     "view": "normal", "trajectory_complete": True, "label_source": label_source,
                     "visible_notes": changed, "closing_note": closing, **lengths})
        print(f"{task}: {lengths['total_tokens']} tokens / {lengths['supervised_tokens']} supervised ({origin})", flush=True)

    (output / "train.jsonl").write_text("".join(dump(sample) + "\n" for sample in written))
    max_tokens = max(row["total_tokens"] for row in rows)
    config = json.loads(config_path.read_text())
    config["max_seq_length"] = max(config["max_seq_length"], -(-max_tokens // 4096) * 4096)
    write_json(output / "training_config.json", config)
    train = list(dict.fromkeys(split["train"] + [row["task_id"] for row in source_manifest["tasks"]]))
    write_json(output / "split_manifest.json", {**split, "schema_version": "e10_split_v1", "train": train,
               "notes": "E6 validation/test unchanged; train = E6 13 + 47 train60 tasks (no repository shared with held-out)."})
    totals = {
        "rows": len(rows), "tokens": sum(row["total_tokens"] for row in rows),
        "supervised_tokens": sum(row["supervised_tokens"] for row in rows), "maximum_tokens": max_tokens,
        "visible_notes": sum(row["visible_notes"] for row in rows),
        "closing_notes": sum(row["closing_note"] for row in rows),
        "label_sources": {key: sum(row["label_source"] == key for row in rows)
                          for key in sorted({row["label_source"] for row in rows})},
    }
    manifest = {
        "schema_version": "e10_train60_rationale_closing_v1", "experiment": "E10",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": e9_1_manifest["model"], "model_revision": e9_1_manifest["model_revision"],
        "subset": only or None, "sample_count": len(rows), "normal_rows": len(rows), "recovery_rows": 0,
        "train_task_count": len(rows), "validation_task_count": len(split["validation"]),
        "test_task_count": len(split["test"]),
        "enable_thinking": False, "thinking_in_training": False, "visible_notes_in_training": True,
        "initialization": "original_base_plus_new_lora",
        "loss_policy": "E9_1 policy: assistant notes, tool-call syntax, closing note and Ledger supervised; context masked",
        "length_policy": f"No truncation, splitting or packing; max_seq_length={config['max_seq_length']}",
        "notes_prompt_version": notes_run["prompt_version"],
        "closing_prompt_version": notes_run.get("closing_prompt_version"),
        "notes_teacher": notes_run["teacher"], "totals": totals, "tasks": rows,
        "split_manifest": "split_manifest.json",
    }
    write_json(output / "validation.json", {
        "rows_match_source_except_notes": len(rows), "training_inference_boundary_verified": len(rows),
        "controller_extracts_label_ledger": len(rows), "tokenization_loss_masks_verified": len(rows),
        "totals": totals,
    })
    (output / "README.md").write_text(f"""# E10：60 题可见说明 + 收尾说明

E9_1 的 13 行原样复用；另加 train60 中选出的 47 题（选题见 `../train60_selection_v1/selection.json`，源数据见 `../e10_new47_source_v1/`）。新行与 E9_1 格式相同：删除教师 thinking，每个工具回合前是教师压缩的可见说明，最终回合是收尾说明 + 原样标签 Ledger。

训练超参沿用 E9_1 lr 1e-4 那次（3 epoch、rank 16），只把 `max_seq_length` 提到 {config['max_seq_length']:,} 以容纳最长的完整行（{max_tokens:,} tokens）。验证/测试 17 题不变，且与 60 道训练题无共享仓库。

共 {totals['rows']} 行；输入 {totals['tokens']:,} tokens，监督 {totals['supervised_tokens']:,} tokens；标签来源 {totals['label_sources']}。
""")
    manifest["file_sha256"] = {str(path.relative_to(output)): sha(path)
                               for path in sorted(output.rglob("*")) if path.is_file()}
    write_json(output / "manifest.json", manifest)
    return manifest


def source_manifest_row(manifest: dict, task: str) -> dict:
    return next(row for row in manifest["tasks"] if row["task_id"] == task)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--e9-1", type=Path, default=DEFAULT_E9_1)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--notes", type=Path, default=DEFAULT_NOTES)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--only", nargs="+", help="Package only these tasks (smoke tests)")
    args = parser.parse_args()
    manifest = build(args.e9_1.resolve(), args.source.resolve(), args.notes.resolve(), args.config.resolve(),
                     args.output.resolve(), args.only)
    print(json.dumps(manifest["totals"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
