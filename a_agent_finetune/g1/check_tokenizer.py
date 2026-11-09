"""Verify the real pinned Qwen tokenizer and a synthetic tool trajectory without loading weights."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from .materials import load_rows
from .reward import DeliveryReward
from .rollout import Budgets, run_trajectory
from .tools import tool_schemas

HERE = Path(__file__).resolve().parent


def check(root: Path, tokenizer_path: Path) -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    original = tokenizer.chat_template
    template = (root / "protocol/training_chat_template.jinja").read_text()
    tokens = []
    for split in ("train", "validation", "test"):
        for row in load_rows(root, split):
            tokenizer.chat_template = original
            before = tokenizer.apply_chat_template(
                row["prompt"], tools=tool_schemas(), enable_thinking=False,
                add_generation_prompt=True, tokenize=True, return_dict=False,
            )
            tokenizer.chat_template = template
            after = tokenizer.apply_chat_template(
                row["prompt"], tools=tool_schemas(), enable_thinking=False,
                add_generation_prompt=True, tokenize=True, return_dict=False,
            )
            if before != after:
                raise ValueError(f"Initial Qwen prompt changes under the training template: {row['task_id']}")
            tokens.append(len(after))
    tokenizer.response_schema = json.loads((root / "protocol/response_schema.json").read_text())
    eos = tokenizer.eos_token
    call = (
        "<tool_call>\n<function=read>\n<parameter=path>\nsample.txt\n</parameter>\n"
        "<parameter=offset>\n2\n</parameter>\n<parameter=limit>\n1\n</parameter>\n"
        "</function>\n</tool_call>" + eos
    )
    call_ids = tokenizer.encode(call, add_special_tokens=False)
    parsed = tokenizer.parse_response(call_ids)
    function = parsed["tool_calls"][0]["function"]
    if function != {"name": "read", "arguments": {"path": "sample.txt", "offset": 2, "limit": 1}}:
        raise ValueError(f"Qwen tool parsing differs: {parsed}")
    ledger = {
        "summary": "Synthetic protocol check", "obligations": [{
            "statement": "Required behavior", "importance": "required",
            "sources": [{"kind": "task", "field": "description", "quote": "Required behavior"}],
        }], "preservation_obligations": [], "open_questions": [],
    }
    final_ids = tokenizer.encode(json.dumps(ledger) + eos, add_special_tokens=False)
    samples = iter([call_ids, final_ids])
    row = load_rows(root, "train")[0]
    with tempfile.TemporaryDirectory() as directory:
        repository = Path(directory)
        (repository / "sample.txt").write_text("first\npublic contract\nthird\n")
        result = run_trajectory(
            prompt=row["prompt"], repository=repository, tokenizer=tokenizer,
            generate=lambda context, maximum, deadline: next(samples),
            parse_response=lambda ids, prefix: tokenizer.parse_response(ids),
            reward=DeliveryReward(root / "protocol/spec_output_schema.json"), budgets=Budgets(),
        )
    if result.score["reward"] != 1 or result.tools[0]["output"] != "2: public contract":
        raise ValueError("Synthetic Qwen tool-to-Ledger trajectory failed")
    if result.completion_ids[:len(call_ids)] != call_ids or result.completion_ids[-len(final_ids):] != final_ids:
        raise ValueError("G1 changed sampled token IDs")
    if sum(result.env_mask) != len(call_ids) + len(final_ids):
        raise ValueError("G1 trains on tool observations or misses sampled tokens")
    return {
        "pinned_tokenizer_revision": tokenizer_path.name, "tasks_checked": len(tokens),
        "initial_prompt_ids_match_original": True,
        "initial_prompt_tokens_min": min(tokens), "initial_prompt_tokens_max": max(tokens),
        "xml_tool_arguments_parsed": True, "real_tokenizer_tool_suffix_valid": True,
        "sampled_tokens_preserved": True, "observation_tokens_masked": True,
        "synthetic_tool_to_ledger_reward": result.score["reward"],
        "model_weights_loaded": False, "model_inference_performed": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--materials", type=Path, default=HERE)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = check(args.materials, args.tokenizer_path)
    if args.report:
        args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
