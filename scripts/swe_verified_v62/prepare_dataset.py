"""Freeze a uniform, outcome-independent random100 from Verified's 500 test tasks."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset

DATASET = "princeton-nlp/SWE-bench_Verified"
REVISION = "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--grader-output", type=Path)
    args = parser.parse_args()
    dataset = load_dataset(DATASET, revision=REVISION, split="test")
    rows = {row["instance_id"]: row for row in dataset}
    population = sorted(rows)
    if len(population) != 500:
        raise ValueError("Expected the fixed Verified test population of 500")
    selected = sorted(random.Random(args.seed).sample(population, 100))
    manifest = {
        "dataset_name": DATASET, "dataset_revision": REVISION, "split": "test",
        "task_count": 100, "population_count": 500, "sample_seed": args.seed,
        "sampling": "random.Random(seed).sample(sorted(instance_ids), 100); no outcome filtering",
        "population_sha256": hashlib.sha256("\n".join(population).encode()).hexdigest(),
        "task_ids": selected, "repository_counts": dict(sorted(Counter(rows[i]["repo"] for i in selected).items())),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "verified_random100_seed20261026.json"
    text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text() != text:
        raise ValueError("Existing task selection differs; use another output directory")
    path.write_text(text)
    # Only these public fields may be used by generation. Gold data stays in the
    # host's dataset cache and the independent official evaluation harness.
    public_fields = ("instance_id", "repo", "base_commit", "version", "problem_statement")
    tasks = [{key: rows[i][key] for key in public_fields} for i in selected]
    (args.output / "public_tasks.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in tasks)
    )
    if args.grader_output:
        args.grader_output.parent.mkdir(parents=True, exist_ok=True)
        args.grader_output.write_text(json.dumps([rows[i] for i in selected], ensure_ascii=False))
        args.grader_output.chmod(0o600)
    print(json.dumps({"manifest": str(path), "task_count": 100, "repository_counts": manifest["repository_counts"]}))


if __name__ == "__main__":
    main()
