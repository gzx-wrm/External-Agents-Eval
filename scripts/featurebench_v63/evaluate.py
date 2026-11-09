"""Run a frozen prediction through the unmodified official FeatureBench scorer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from featurebench.harness.run_evaluation import run_instance
from featurebench.harness.utils import preprocess_hf_patch

import docker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--stage", choices=["gold", "control", "P0", "P1"], required=True)
    args = parser.parse_args()
    task = json.loads((args.root / "manifests/task-private.json").read_text())
    image = json.loads((args.root / "preflight/image.json").read_text())
    client = docker.from_env()
    try:
        if client.images.get(task["image_name"]).id != image["id"]:
            raise ValueError("Scoring image differs from the frozen generation image")
    finally:
        client.close()
    if args.stage == "gold":
        patch = preprocess_hf_patch(task["patch"], task["FAIL_TO_PASS"])
    elif args.stage == "control":
        patch = (
            "diff --git a/featurebench_smoke_control.txt b/featurebench_smoke_control.txt\n"
            "new file mode 100644\n--- /dev/null\n+++ b/featurebench_smoke_control.txt\n"
            "@@ -0,0 +1 @@\n+Environment validation only; no implementation changes.\n"
        )
    else:
        patch = (args.run / f"{args.stage}.patch").read_text()
    prediction = {
        "instance_id": task["instance_id"], "n_attempt": 1,
        "model_name_or_path": "v63-mini-deepseek-v4.1-flash", "model_patch": patch,
    }
    output = args.run / "scoring" / args.stage
    output.mkdir(parents=True, exist_ok=True)
    (output / "prediction.json").write_text(json.dumps(prediction, indent=2))
    result = run_instance(pd.Series(task), prediction, output, white=False)
    (output / "result.json").write_text(json.dumps(result, indent=2))
    report = result.get("report", {}).get(task["instance_id"], {})
    summary = {
        "stage": args.stage, "instance_id": task["instance_id"],
        "completed": result.get("completed", False), "resolved": result.get("resolved", False),
        "patch_applied": result.get("patch_applied", False),
        "pass_rate": report.get("pass_rate"), "error": result.get("error"), "report": report,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    if not summary["completed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
