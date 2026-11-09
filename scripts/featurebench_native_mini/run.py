"""Three native Mini solve/score slots, gated on per-task CPU validation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from common import classify_reference_failure, now, positive_reference_passed, save
from generate import configuration

import docker

NATIVE_PYTHON = "/root/v62-swe-verified100-20261026/venv312/bin/python"
SCORE_PYTHON = "/root/v63-featurebench-20261028/venv312/bin/python"
OFFICIAL_SOURCE = "/root/v63-featurebench-20261028/vendor/FeatureBench"


def registry_manifest(repository):
    with urllib.request.urlopen("https://auth.docker.io/token?service=registry.docker.io&scope=repository:"
                                + repository + ":pull", timeout=45) as response:
        token = json.loads(response.read())["token"]
    headers = {"Authorization": "Bearer " + token,
               "Accept": "application/vnd.docker.distribution.manifest.v2+json, application/vnd.oci.image.manifest.v1+json, "
                         "application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.oci.image.index.v1+json"}

    def read(reference):
        request = urllib.request.Request("https://registry-1.docker.io/v2/" + repository + "/manifests/" + reference,
                                         headers=headers)
        with urllib.request.urlopen(request, timeout=45) as response:
            payload = response.read()
            return json.loads(payload), "sha256:" + hashlib.sha256(payload).hexdigest()

    manifest, digest = read("latest")
    if "manifests" in manifest:
        entry = next(x for x in manifest["manifests"]
                     if x.get("platform", {}).get("architecture") == "amd64"
                     and x.get("platform", {}).get("os") == "linux")
        manifest, digest = read(entry["digest"])
        if digest != entry["digest"]:
            raise ValueError("Registry manifest content digest differs")
    return {"original_repository": repository, "original_digest": digest,
            "config_digest": manifest["config"]["digest"],
            "compressed_bytes": sum(layer["size"] for layer in manifest["layers"]),
            "mirror_reference": "docker.1ms.run/" + repository + "@" + digest}


def ensure_image(root, repository):
    if (root / "configs/image-cache.json").exists():
        from cached_images import ensure_cached_image
        client = docker.from_env(timeout=1800)
        try:
            return ensure_cached_image(root, repository, client)
        finally:
            client.close()
    directory = root / "preflight/images"
    directory.mkdir(exist_ok=True)
    metadata = directory / (repository.rsplit("/", 1)[-1] + ".json")
    first = root / "preflight/mlflow-registry.json"
    if metadata.exists():
        record = json.loads(metadata.read_text())
    elif first.exists() and json.loads(first.read_text())["original_repository"] == repository:
        record = json.loads(first.read_text())
    else:
        record = registry_manifest(repository)
    reference = record["original_repository"] + "@" + record["original_digest"]
    record.update(download_reference=reference, download_route="official-direct")
    save(metadata, record)
    client = docker.from_env(timeout=1800)
    try:
        try:
            image = client.images.get(reference)
        except docker.errors.ImageNotFound:
            with (root / "logs" / (repository.rsplit("/", 1)[-1] + "-pull.log")).open("ab") as log:
                subprocess.run(["docker", "pull", "--quiet", reference],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            image = client.images.get(reference)
        if image.id != record["config_digest"]:
            raise ValueError("Pulled image differs from its original Docker Hub configuration digest")
        image.tag(repository, tag="latest")
        record.update(image_id=image.id, verified_at=now())
        save(metadata, record)
        return image.id
    finally:
        client.close()


def subprocess_stage(root, task_path, output, stage, *, image_id=None):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = OFFICIAL_SOURCE
    if stage in {"generate", "native_offline"}:
        generation_folder = "generation" if stage == "generate" else "native-offline"
        command = [NATIVE_PYTHON, str(root / "scripts/generate.py"), "--root", str(root),
                   "--task", str(task_path), "--output", str(output / generation_folder), "--image-id", image_id]
        if stage == "native_offline":
            command.append("--offline")
    else:
        environment = {key: value for key, value in environment.items() if not key.endswith("_API_KEY")}
        command = [SCORE_PYTHON, str(root / "scripts/evaluate.py"), "--task", str(task_path),
                   "--output", str(output / stage), "--stage", stage]
        if (root / "configs/cpu-evaluation.json").exists():
            command += ["--cpu-config", str(root / "configs/cpu-evaluation.json")]
        if stage == "model":
            command += ["--model-patch", str(output / "generation/model.patch")]
    with (output / (stage + ".log")).open("ab") as log:
        subprocess.run(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    name = (generation_folder + "/generation.json" if stage in {"generate", "native_offline"}
            else stage + "/summary.json")
    return json.loads((output / name).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise ValueError("A transient host-side model credential is required")
    if os.environ.get("DOCKER_HOST") != "unix:///run/featurebench-lite-docker.sock":
        raise ValueError("This run must use its isolated Docker engine on the new disk")
    configuration(root)
    tasks = json.loads((root / "manifests/tasks-private.json").read_text())
    public = json.loads((root / "manifests/tasks-public.json").read_text())
    expected = {task["instance_id"]: task["private_task_sha256"] for task in public}
    for task in tasks:
        digest = hashlib.sha256(json.dumps(task, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if expected[task["instance_id"]] != digest:
            raise ValueError("A frozen task changed")
    states = {task["instance_id"]: {"phase": "queued"} for task in tasks}
    lock = threading.Lock()
    results = {}

    def update(task, phase, **fields):
        with lock:
            states[task["instance_id"]].update(phase=phase, updated_at=now(), **fields)
            save(root / "monitor-status.json", {"workers": 3, "task_count": len(tasks),
                 "updated_at": now(), "tasks": states})
        print(json.dumps({"instance_id": task["instance_id"], "phase": phase, **fields}), flush=True)

    def finish(task, value):
        value.update(instance_id=task["instance_id"], finished_at=now())
        save(root / "outputs" / task["instance_id"] / "result.json", value)
        with lock:
            results[task["instance_id"]] = value
            save(root / "result.json", {"task_count": len(tasks), "terminal_count": len(results),
                 "passed": sum(x.get("reward") == 1 for x in results.values()),
                 "valid_failed": sum(x.get("reward") == 0 for x in results.values()),
                 "results": list(results.values()), "updated_at": now()})
        update(task, value["status"], reward=value.get("reward"))
        return value

    def one(task, image_id):
        output = root / "outputs" / task["instance_id"]
        output.mkdir(exist_ok=True, parents=True)
        if (output / "result.json").exists():
            return finish(task, json.loads((output / "result.json").read_text()))
        task_path = output / "task-private.json"
        save(task_path, task)
        try:
            update(task, "cpu_reference_validation")
            reference = subprocess_stage(root, task_path, output, "reference")
            if not positive_reference_passed(reference):
                diagnosis = None
                if task["level"] == 1:
                    update(task, "baseline_diagnosis")
                    diagnosis = subprocess_stage(root, task_path, output, "baseline")
                logs = "\n".join(path.read_text(errors="replace")[-150000:]
                                 for path in (output / "reference").rglob("*.txt"))
                logs += (output / "reference.log").read_text(errors="replace")[-150000:]
                return finish(task, {"status": classify_reference_failure(logs), "reward": None,
                                    "cpu_reference": reference, "pristine_baseline": diagnosis,
                                    "model_api_calls": 0})
            update(task, "cpu_control_validation")
            control = subprocess_stage(root, task_path, output, "control")
            if control.get("error") or not control.get("patch_applied") or control.get("resolved"):
                return finish(task, {"status": "cpu_control_validation_failed", "reward": None,
                                    "cpu_control": control, "model_api_calls": 0})
            if task["level"] == 2:
                update(task, "native_offline_validation")
                protocol = subprocess_stage(root, task_path, output, "native_offline", image_id=image_id)
                if protocol.get("model_api_calls") != 0 or protocol["agent_info"]["exit_status"] != "Submitted":
                    raise RuntimeError("Native Level-2 offline protocol validation failed")
            save(output / "cpu-readiness.json", {"passed": True, "reference": reference,
                                                 "control": control, "model_api_calls": 0})
            update(task, "solving", started_at=now())
            generation = subprocess_stage(root, task_path, output, "generate", image_id=image_id)
            update(task, "official_evaluation")
            score = subprocess_stage(root, task_path, output, "model")
            if not score.get("completed") or score.get("error"):
                return finish(task, {"status": "scoring_infrastructure_error", "reward": None,
                                    "official": score, "generation": generation})
            if score.get("patch_applied") and score.get("f2p_total", 0) == 0:
                return finish(task, {"status": "scoring_no_test_results", "reward": None,
                                    "official": score, "generation": generation})
            return finish(task, {"status": "scored", "reward": int(bool(score["resolved"])),
                                "official": score, "generation": generation})
        except Exception as error:  # noqa: BLE001 -- preserve the failed task and continue siblings
            return finish(task, {"status": "infrastructure_or_stage_error", "reward": None,
                                "error_type": type(error).__name__, "error": str(error)})

    groups = {}
    for task in tasks:
        groups.setdefault(task["image_name"], []).append(task)
    flags = {task["image_name"]: task["runtime_config"]["need_gpu"] for task in public}
    ordered = sorted(groups, key=lambda name: flags[name])
    plan = root / "recovery-plan.json"
    if plan.exists():
        priority = {name: i for i, name in enumerate(json.loads(plan.read_text()).get("image_order", []))}
        ordered.sort(key=lambda name: priority.get(name, len(priority)))
    for repository in ordered:
        group = groups[repository]
        for task in group:
            update(task, "image_preparation", image=repository)
        try:
            image_id = ensure_image(root, repository)
        except Exception as error:  # noqa: BLE001 -- image failures are explicit per-task outcomes
            for task in group:
                finish(task, {"status": "image_infrastructure_error", "reward": None,
                              "error_type": type(error).__name__, "error": str(error), "model_api_calls": 0})
            continue
        with ThreadPoolExecutor(max_workers=3) as pool:
            for future in as_completed([pool.submit(one, task, image_id) for task in group]):
                future.result()
        # This engine contains only this experiment's images. All group tasks,
        # including immediate official scoring, have ended before image removal.
        client = docker.from_env()
        try:
            client.images.remove(image_id, force=True, noprune=False)
        except docker.errors.APIError as error:
            print(json.dumps({"image_cleanup_error": str(error), "image": repository}), flush=True)
        finally:
            client.close()
    save(root / "launcher-result.json", {"completed": True, "terminal_count": len(results), "finished_at": now()})


if __name__ == "__main__":
    main()
