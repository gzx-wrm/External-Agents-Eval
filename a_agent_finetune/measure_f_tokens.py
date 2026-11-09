"""Measure pilot prompt/completion text with an explicitly supplied tokenizer.json."""

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
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()

    try:
        from tokenizers import Tokenizer, __version__ as tokenizers_version
    except ImportError as exc:
        raise SystemExit("Install the Python tokenizers package to measure lengths") from exc

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    manifest = json.loads((DATA / "manifest.json").read_text())
    samples = [json.loads(line) for line in (DATA / "training_candidates.jsonl").read_text().splitlines()]
    if len(samples) != len(manifest["tasks"]):
        raise ValueError("Manifest/sample count mismatch")
    rows = []
    for task, sample in zip(manifest["tasks"], samples, strict=True):
        prompt_tokens = sum(
            len(tokenizer.encode(message["content"], add_special_tokens=False).ids)
            for message in sample["prompt"]
        )
        completion_tokens = sum(
            len(tokenizer.encode(message["content"], add_special_tokens=False).ids)
            for message in sample["completion"]
        )
        rows.append(
            {
                "task_id": task["task_id"],
                "prompt_text_tokens": prompt_tokens,
                "completion_text_tokens": completion_tokens,
                "total_text_tokens": prompt_tokens + completion_tokens,
            }
        )
    result = {
        "model": args.model,
        "model_revision": args.revision,
        "tokenizer_sha256": hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
        "tokenizers_package_version": tokenizers_version,
        "counting_rule": "Sum tokenized message content, add_special_tokens=False. Chat-template overhead is not included.",
        "tasks": rows,
    }
    output = DATA / "lengths.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(output)
    for row in rows:
        print(f"{row['task_id']}: {row['total_text_tokens']} text tokens")


if __name__ == "__main__":
    main()
