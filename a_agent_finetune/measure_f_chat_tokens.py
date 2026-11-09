"""Check complete sample lengths using the pinned model's actual chat template."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "f_pilot_5"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-config", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--variants", nargs="+", choices=("full", "32k", "64k"), default=["full", "32k", "64k"])
    args = parser.parse_args()

    from jinja2 import Environment
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    config = json.loads(args.tokenizer_config.read_text())
    template = Environment().from_string(config["chat_template"])

    def fail(message: str) -> None:
        raise ValueError(message)

    task_ids = [task["task_id"] for task in json.loads((args.data_dir / "manifest.json").read_text())["tasks"]]
    rows = []
    for variant, suffix, limit in (("full", "", None), ("32k", "_32k", 32_768), ("64k", "_64k", 65_536)):
        if variant not in args.variants:
            continue
        samples = [json.loads(line) for line in (args.data_dir / f"training_candidates{suffix}.jsonl").read_text().splitlines()]
        if len(samples) != len(task_ids):
            raise ValueError(f"Unexpected sample count: {variant}")
        for task_id, sample in zip(task_ids, samples, strict=True):
            prompt = template.render(
                messages=sample["prompt"],
                add_generation_prompt=True,
                enable_thinking=False,
                raise_exception=fail,
            )
            full = template.render(
                messages=sample["prompt"] + sample["completion"],
                add_generation_prompt=False,
                enable_thinking=False,
                raise_exception=fail,
            )
            if not full.startswith(prompt):
                raise ValueError(f"Chat template prompt is not a prefix: {variant} {task_id}")
            prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False).ids)
            total_tokens = len(tokenizer.encode(full, add_special_tokens=False).ids)
            if limit is not None and total_tokens > limit:
                raise ValueError(f"Over {variant} context budget: {task_id}: {total_tokens}")
            rows.append(
                {
                    "variant": variant,
                    "task_id": task_id,
                    "prompt_tokens": prompt_tokens,
                    "total_tokens": total_tokens,
                    "assistant_suffix_tokens": total_tokens - prompt_tokens,
                    "fits_64k_with_8192_output_reserve": prompt_tokens + 8192 <= 65_536,
                    "fits_native_with_8192_output_reserve": prompt_tokens + 8192 <= config["model_max_length"],
                }
            )
    report = {
        "model": args.model,
        "model_revision": args.revision,
        "tokenizer_sha256": hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
        "tokenizer_config_sha256": hashlib.sha256(args.tokenizer_config.read_bytes()).hexdigest(),
        "chat_template_setting": "enable_thinking=False, add_generation_prompt=True for prompt only; complete sample rendered with add_generation_prompt=False",
        "model_max_length": config["model_max_length"],
        "rows": rows,
    }
    (args.data_dir / "chat_lengths.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for row in rows:
        print(f"{row['variant']} {row['task_id']}: {row['total_tokens']}")


if __name__ == "__main__":
    main()
