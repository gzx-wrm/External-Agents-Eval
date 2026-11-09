"""Check E4 IDs and selected assistant loss with Transformers, without loading a model."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def verify(materials: Path, tokenizer_path: Path):
    import numpy as np
    import transformers
    from transformers import AutoTokenizer

    manifest = json.loads((materials / "manifest.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    template = (materials / "protocol/training_chat_template.jinja").read_text()
    lines = (materials / "train.jsonl").read_text().splitlines()
    input_paths = ["train.jsonl", "protocol/training_chat_template.jinja",
                   "protocol/tokenizer.json", "protocol/tokenizer_config.json"]
    checks = []
    for row, line in zip(manifest["tasks"], lines, strict=True):
        sample = json.loads(line)
        encoded = tokenizer.apply_chat_template(
            sample["messages"], tools=sample["tools"], chat_template=template,
            trainable_message_mask=sample["trainable_message_mask"],
            tokenize=True, return_dict=True, return_assistant_tokens_mask=True,
            enable_thinking=False, add_generation_prompt=False,
        )
        with np.load(materials / "tokenized" / (sample["sample_id"] + ".npz")) as arrays:
            if not np.array_equal(encoded["input_ids"], arrays["input_ids"]):
                raise ValueError(f"Transformers input IDs differ: {sample['sample_id']}")
            if not np.array_equal(encoded["assistant_masks"], arrays["labels"] != -100):
                raise ValueError(f"Transformers loss mask differs: {sample['sample_id']}")
            checks.append({"sample_id": sample["sample_id"], "tokens": len(arrays["input_ids"]),
                           "supervised_tokens": int(np.count_nonzero(arrays["labels"] != -100))})
        input_paths.append("tokenized/" + sample["sample_id"] + ".npz")
    return {"passed": True, "transformers_version": transformers.__version__,
            "tokenizer_class": type(tokenizer).__name__, "model_loaded": False,
            "verified_input_sha256": {path: hashlib.sha256((materials / path).read_bytes()).hexdigest()
                                      for path in input_paths},
            "rows_verified": len(checks), "checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materials", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.materials.resolve(), args.tokenizer_path or args.materials / "protocol")
    if args.output:
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("checks", "verified_input_sha256")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
