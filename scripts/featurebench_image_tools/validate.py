"""Compare a cache-only prototype with its original; never adopt it in a batch."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shlex
import subprocess
import time
from pathlib import Path


def inspect(engine, image):
    return json.loads(subprocess.check_output(["docker", "-H", engine, "image", "inspect", image]))[0]


def inventory(engine, image):
    result = {}
    python = ("import importlib.metadata as m,json,sys; "
              "print(json.dumps({'python':sys.version,'packages':sorted("
              "(d.metadata['Name'],d.version) for d in m.distributions())}))")
    for name in ["pytorch_base", "testbed"]:
        command = ("source /opt/miniconda3/etc/profile.d/conda.sh && conda activate "
                   + name + " && python -B -c " + shlex.quote(python))
        data = subprocess.check_output(["docker", "-H", engine, "run", "--rm", "--network=none",
                                        "--cpus=2", "--memory=6g", "--entrypoint=/bin/bash",
                                        image, "-lc", command], timeout=180)
        result[name] = json.loads(data.decode().splitlines()[-1])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    audit_file = Path("/mnt/featurebench/image-optimization-20261030-r1/repack-work/repack-audit.json")
    launch = json.loads((root / "repack-launch.json").read_text())
    while not audit_file.exists():
        if not Path("/proc/" + str(launch["pid"])).exists():
            raise RuntimeError("Repack stopped without a successful audit; inspect repack.log")
        time.sleep(2)
    audit = json.loads(audit_file.read_text())
    source = json.loads((root / "source.json").read_text())
    if audit["source_image_id"] != source["id"]:
        raise ValueError("Repacked archive is not the original frozen image")
    old = inspect(source["source_engine"], source["id"])
    new = inspect(source["candidate_engine"], audit["tag"])
    if new["Id"] != audit["candidate_image_id"]:
        raise ValueError("Loaded image does not match the repack audit")
    before, after = copy.deepcopy(old["Config"]), copy.deepcopy(new["Config"])
    before.pop("Labels", None)
    after.pop("Labels", None)
    if before != after:
        raise ValueError("Runtime image configuration changed")
    (root / "candidate-inspect.json").write_text(json.dumps(new, indent=2))
    (root / "repack-audit.json").write_text(json.dumps(audit, indent=2))
    # The immutable original image remains in the old daemon. This large export
    # is only a temporary input; the smaller loadable candidate archive is kept.
    (root / "source-image.tar").unlink()
    inventories = {"original": inventory(source["source_engine"], source["id"]),
                   "candidate": inventory(source["candidate_engine"], audit["tag"])}
    if inventories["original"] != inventories["candidate"]:
        raise ValueError("Installed Python packages changed")
    (root / "environment-inventories.json").write_text(json.dumps(inventories, indent=2))
    original_task = json.loads(Path("/root/native-mini-featurebench-lite-20261028-w3-r1/preflight/packaging-task.json").read_text())
    scores = {}
    for label, engine, image in [("original", source["source_engine"], source["image"]),
                                  ("candidate", source["candidate_engine"], audit["tag"])]:
        folder = root / "validation" / label
        folder.mkdir(parents=True, exist_ok=True)
        task = {**original_task, "image_name": image}
        task_file = folder / "task.json"
        task_file.write_text(json.dumps(task, ensure_ascii=False, indent=2))
        scores[label] = {}
        for stage in ["reference", "control"]:
            output = folder / stage
            env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
            env.update(DOCKER_HOST=engine, PYTHONPATH="/root/v63-featurebench-20261028/vendor/FeatureBench")
            cmd = ["/root/v63-featurebench-20261028/venv312/bin/python",
                   str(args.evaluation_root / "scripts/evaluate.py"), "--task", str(task_file),
                   "--output", str(output), "--stage", stage,
                   "--cpu-config", str(args.evaluation_root / "configs/cpu-evaluation.json")]
            with (folder / (stage + ".log")).open("wb") as log:
                subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
            scores[label][stage] = json.loads((output / "summary.json").read_text())
            print(json.dumps({"image": label, "stage": stage, "resolved": scores[label][stage]["resolved"],
                              "f2p": [scores[label][stage]["f2p_passed"], scores[label][stage]["f2p_total"]],
                              "p2p": [scores[label][stage]["p2p_passed"], scores[label][stage]["p2p_total"]]}), flush=True)
    for label in scores:
        if not scores[label]["reference"]["resolved"] or scores[label]["control"]["resolved"]:
            raise ValueError("Official gold/control validation failed for " + label)
    fields = ["resolved", "f2p_passed", "f2p_total", "p2p_passed", "p2p_total"]
    for stage in ["reference", "control"]:
        if any(scores["original"][stage][k] != scores["candidate"][stage][k] for k in fields):
            raise ValueError("Original and candidate test outcomes differ")
    artifact = Path(audit["candidate_archive"])
    digest = hashlib.sha256()
    with artifact.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    archive_hash = digest.hexdigest()
    result = {"passed": True, "source_size_bytes": old["Size"], "candidate_size_bytes": new["Size"],
              "size_reduction_bytes": old["Size"] - new["Size"],
              "size_reduction_percent": round(100 * (1 - new["Size"] / old["Size"]), 2),
              "original_layer_count": len(old["RootFS"]["Layers"]), "candidate_layer_count": len(new["RootFS"]["Layers"]),
              "runtime_configuration_equal": True, "both_python_environment_inventories_equal": True,
              "retained_files_verified_per_layer": True, "official_scores": scores,
              "candidate_archive": str(artifact), "candidate_archive_bytes": artifact.stat().st_size,
              "candidate_archive_sha256": archive_hash, "model_api_calls": 0,
              "active_experiment_adopted": False}
    # Keep the verified archive, freeing the prototype's uncompressed layers
    # before the active experiment moves to its next large image.
    subprocess.run(["docker", "-H", source["candidate_engine"], "image", "rm", audit["tag"]],
                   stdout=subprocess.DEVNULL, check=True)
    result["candidate_removed_from_active_engine_after_validation"] = True
    (root / "optimization-result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "official_scores"}), flush=True)


if __name__ == "__main__":
    main()
