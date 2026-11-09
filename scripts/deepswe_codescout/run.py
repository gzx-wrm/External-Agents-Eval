"""Preprocess one fixed DeepSWE task from a frozen native-Mini bundle.

No Solver, Reviewer, or official evaluation is started by this entry point.
It reads the bundle's *original* public-tasks.json, never M1 augmented inputs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import io
import json
import os
import sys
import tarfile
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from external_agents_eval.preprocessors.codescout.artifacts import ArtifactStore
from external_agents_eval.preprocessors.codescout.index import EXCLUDED, build_index
from external_agents_eval.preprocessors.codescout.pipeline import ChatClient, Config, Preprocessor


def augment_task(task, enhanced_issue):
    if not enhanced_issue.strip():
        raise ValueError("Enhanced problem statement is empty")
    return {**task, "instruction": enhanced_issue}


def extract_snapshot(data, destination):
    """Accept regular files only; no links, traversal, devices, or Git metadata."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        members = archive.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            if (path.is_absolute() or ".." in path.parts or ".git" in path.parts
                    or not (member.isfile() or member.isdir())):
                raise ValueError("Unexpected entry in isolated repository snapshot")
        archive.extractall(destination, members=members, filter="data")


def load_native(root):
    path = root / "scripts/m0_run.py"
    if not path.exists():
        raise ValueError("Use a frozen ft_m1 bundle containing corrected scripts/m0_run.py")
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("codescout_frozen_native", path)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    return native


def snapshot(root, task, record, output, native):
    store = ArtifactStore(output)
    source = output / "repository"
    if source.exists():
        receipt = json.loads((output / "snapshot.json").read_text())
        if (receipt["task_checksum"] != task["task_checksum"] or
                receipt["image_id"] != record["m0_image"]["id"]):
            raise ValueError("Existing snapshot belongs to another task/image")
        return source
    client, environment, container = native.docker.from_env(timeout=1800), None, None
    try:
        pinned = {**task, "image": record["m0_image"]["repo_digests"][0]}
        image, _ = native.get_image(client, pinned, output)
        if image.id != record["m0_image"]["id"]:
            raise ValueError("Source image differs from the frozen M0 image")
        import yaml

        config = yaml.safe_load((root / "configs/native-deepseek-v4.1-flash.yaml").read_text())
        options = copy.deepcopy(config["environment"])
        options["image"] = image.id
        options["run_args"] += ["--cpus=" + str(task["cpus"]),
                                "--memory=" + str(task["memory_mb"]) + "m"]
        environment = native.get_environment(options)
        container = client.containers.get(environment.container_id)
        isolation = native.prepare_repository(root, environment, container, task, output)
        script = (
            "import os,tarfile,stat\nfrom pathlib import Path\n"
            f"excluded=set({list(EXCLUDED)!r})\n"
            "with tarfile.open('/tmp/codescout-source.tar','w',dereference=True) as archive:\n"
            " for directory,dirs,names in os.walk('/app',followlinks=False):\n"
            "  dirs[:]=sorted(d for d in dirs if d not in excluded and "
            "not Path(directory,d).is_symlink())\n"
            "  for name in sorted(names):\n"
            "   p=Path(directory,name)\n"
            "   if name!='.git' and stat.S_ISREG(p.lstat().st_mode):\n"
            "    archive.add(p,arcname=p.relative_to('/app').as_posix(),recursive=False)\n"
        )
        native.upload(container, script.encode(), "/tmp/codescout-export.py")
        native.checked(environment, "python /tmp/codescout-export.py", timeout=600)
        chunks, _ = container.get_archive("/tmp/codescout-source.tar")
        outer = io.BytesIO(b"".join(chunks))
        with tarfile.open(fileobj=outer) as archive:
            data = archive.extractfile(archive.getmembers()[0]).read()
        extract_snapshot(data, source)
        store.write_json("snapshot.json", {
            "task_id": task["task_id"], "task_checksum": task["task_checksum"],
            "base_commit": task["base_commit"], "image_id": image.id,
            "image_digests": image.attrs.get("RepoDigests", []),
            "source_archive_sha256": hashlib.sha256(data).hexdigest(),
            "isolation": isolation, "excluded_directories": EXCLUDED,
            "native_source_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
            "original_public_instruction": True, "ledger_or_prior_trajectory_included": False,
        })
        return source
    finally:
        if container is not None:
            container.remove(force=True)
        if environment is not None:
            environment.container_id = None
        client.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["index", "run"])
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    root, output = args.native_root.resolve(), args.output.resolve()
    if root == output or root in output.parents:
        parser.error("Use a separate output directory outside the historical native bundle")
    config = Config(**(json.loads(args.config.read_text()) if args.config else {}))
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if args.mode == "run" and not key:
        parser.error("Inject DEEPSEEK_API_KEY into the host process environment")
    native = load_native(root)
    tasks = {task["task_id"]: task for task in native.load_tasks(root)}
    selected = json.loads((root / "selection.json").read_text())
    records = {record["task_id"]: record for record in selected["tasks"]}
    if args.task_id not in records:
        parser.error("Task is not pinned in this native bundle's selection.json")
    task, record = tasks[args.task_id], records[args.task_id]
    if task["task_checksum"] != record["task_checksum"]:
        raise ValueError("Native task checksum differs")
    output.mkdir(parents=True, exist_ok=True)
    store = ArtifactStore(output, secrets=(key,))
    store.write_json("original-task.json", task)
    source = snapshot(root, task, record, output, native)
    index = build_index(source, max_file_bytes=config.max_file_bytes)
    destination = output / "codescout"
    if args.mode == "index":
        outline, coverage = index.outline(config.max_outline_chars)
        store.write_json("codescout/repository-graph.json", index.graph())
        store.write_text("codescout/repository-outline.txt", outline)
        store.write_json("codescout/index-coverage.json", coverage)
        print(json.dumps({"mode": "index", "model_api_calls": 0,
                          "files": len(index.files), "entities": len(index.entities), **coverage}))
    else:
        enhanced = Preprocessor(config, ChatClient(config, key), destination, secrets=(key,)).run(
            task["instruction"], index)
        augmented = augment_task(task, enhanced)
        augmented["image"] = record["m0_image"]["repo_digests"][0]
        store.write_json("augmented-task.json", augmented)
        print(json.dumps({"status": "completed", "task_id": args.task_id,
                          "input": str(output / "augmented-task.json")}))


if __name__ == "__main__":
    main()
