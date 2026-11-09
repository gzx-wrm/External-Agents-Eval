"""Pinned official FeatureBench scoring, with explicit CPU container settings."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import pandas as pd
from common import cpu_reference_task, save
from featurebench.harness import run_evaluation
from featurebench.harness.container import EvalContainerManager
from featurebench.harness.utils import preprocess_hf_patch
from featurebench.infer.models import TaskInstance
from pristine_reference import run_pristine_reference

CONTROL_PATCH = (
    "diff --git a/native_featurebench_control.txt b/native_featurebench_control.txt\n"
    "new file mode 100644\n--- /dev/null\n+++ b/native_featurebench_control.txt\n"
    "@@ -0,0 +1 @@\n+Environment validation only.\n"
)


class CPUContainerManager(EvalContainerManager):
    cpu_environment: ClassVar[dict[str, str]] = {}
    network_mode = "none"
    iers_configuration = None
    iers_receipt_directory = None

    def create_container(self, *args, **kwargs):
        collection_class = type(self.client.containers)
        original = collection_class.run

        def controlled_run(collection, *positional, **options):
            options.update(network_mode=self.network_mode, nano_cpus=2_000_000_000, mem_limit="6g",
                           cap_drop=["ALL"], security_opt=["no-new-privileges:true"], pids_limit=512)
            options["environment"] = {**options.get("environment", {}), **self.cpu_environment}
            return original(collection, *positional, **options)

        with patch.object(collection_class, "run", controlled_run):
            container = super().create_container(*args, **kwargs)
        container.reload()
        host = container.attrs["HostConfig"]
        if host["NetworkMode"] != self.network_mode or host.get("Binds"):
            container.remove(force=True)
            raise RuntimeError("Official scorer container isolation differs from the approved configuration")
        if self.iers_configuration is not None:
            from iers_data import install_leap_seconds
            install_leap_seconds(container, self.iers_configuration,
                                 self.iers_receipt_directory / (container.name + ".json"))
        return container


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=["reference", "control", "model", "baseline"], required=True)
    parser.add_argument("--model-patch", type=Path)
    parser.add_argument("--cpu-config", type=Path)
    args = parser.parse_args()
    original = json.loads(args.task.read_text())
    task = copy.deepcopy(original)
    cpu = json.loads(args.cpu_config.read_text()) if args.cpu_config else {}
    if cpu.get("timeout_one") is not None:
        settings = json.loads(task["repo_settings"])
        settings["timeout_one"] = int(cpu["timeout_one"])
        task["repo_settings"] = json.dumps(settings)
    CPUContainerManager.cpu_environment = cpu.get("environment", {})
    CPUContainerManager.network_mode = cpu.get("network", "none")
    if args.cpu_config:
        candidate = args.cpu_config.parent / "iers-leap-seconds.json"
        CPUContainerManager.iers_configuration = candidate if candidate.exists() else None
        CPUContainerManager.iers_receipt_directory = args.output / "runtime-data"
    if CPUContainerManager.network_mode not in {"none", "host"}:
        raise ValueError("Unapproved evaluation network mode")
    if args.stage == "baseline" or (args.stage == "reference" and original["level"] == 2):
        if args.stage == "baseline":
            task["FAIL_TO_PASS"] = list(dict.fromkeys(task["FAIL_TO_PASS"] + task["PASS_TO_PASS"]))
        runtime = TaskInstance.from_dict(task).get_docker_runtime_config()
        runtime["need_gpu"] = False
        summary = run_pristine_reference(task, args.output, runtime, CPUContainerManager)
        summary["cpu_runtime_adjustments"] = cpu
        summary["network"] = CPUContainerManager.network_mode
        summary["stage"] = args.stage
        save(args.output / "summary.json", summary)
        print(json.dumps(summary), flush=True)
        return
    if args.stage == "reference":
        task = cpu_reference_task(task)
        model_patch = preprocess_hf_patch(task["patch"], task["FAIL_TO_PASS"])
    elif args.stage == "control":
        model_patch = CONTROL_PATCH
    else:
        model_patch = args.model_patch.read_text()
    runtime = TaskInstance.from_dict(task).get_docker_runtime_config()
    runtime["need_gpu"] = False  # Explicitly approved CPU validation/run mode.
    prediction = {"instance_id": task["instance_id"], "n_attempt": 1,
                  "model_name_or_path": "native-mini-deepseek-v4.1-flash", "model_patch": model_patch}
    args.output.mkdir(parents=True, exist_ok=True)
    save(args.output / "prediction.json", prediction)
    with patch.object(run_evaluation, "EvalContainerManager", CPUContainerManager):
        result = run_evaluation.run_instance(pd.Series(task), prediction, args.output,
                                             white=False, docker_runtime_config=runtime)
    save(args.output / "result.json", result)
    report = result.get("report", {}).get(task["instance_id"], {})
    statuses = report.get("tests_status", {})
    counts = {}
    for prefix, key in [("f2p", "FAIL_TO_PASS"), ("p2p", "PASS_TO_PASS")]:
        values = statuses.get(key, {})
        counts[prefix + "_passed"] = len(values.get("success", []))
        counts[prefix + "_total"] = counts[prefix + "_passed"] + len(values.get("failure", []))
    summary = {"instance_id": task["instance_id"], "stage": args.stage,
               "completed": result.get("completed", False), "resolved": result.get("resolved", False),
               "patch_applied": result.get("patch_applied", False), "pass_rate": report.get("pass_rate"),
               "error": result.get("error") or report.get("error"), **counts,
               "original_level": original["level"], "scored_level": task["level"],
               "cpu_reference_kind": ("official_gold" if original["level"] == 1
                                      else "pristine_original_tests"),
               "cpu_runtime_adjustments": cpu, "cpu_override": True,
               "network": CPUContainerManager.network_mode}
    save(args.output / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
