"""Export original benchmark instructions and public Git blobs into a portable G1 dataset."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import subprocess
import tarfile
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_repository(repository: Path, revision: str, destination: Path) -> dict:
    entries = subprocess.check_output(
        ["git", "-C", str(repository), "ls-tree", "-rz", revision],
    )
    blobs, gitlinks = [], []
    for entry in entries.split(b"\0"):
        if not entry:
            continue
        metadata, name = entry.split(b"\t", 1)
        mode, kind, oid = metadata.split()
        if kind == b"blob":
            blobs.append((mode, oid, name.decode("utf-8")))
        elif kind == b"commit":
            gitlinks.append(name.decode("utf-8"))
    contents = subprocess.run(
        ["git", "-C", str(repository), "cat-file", "--batch"],
        input=b"\n".join(oid for _, oid, _ in blobs) + b"\n",
        capture_output=True, check=True,
    ).stdout
    offset, content_digest = 0, hashlib.sha256()
    with destination.open("wb") as target, gzip.GzipFile(fileobj=target, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w|", format=tarfile.PAX_FORMAT) as archive:
            for mode, oid, name in blobs:
                end = contents.index(b"\n", offset)
                actual, kind, size = contents[offset:end].split()
                if actual != oid or kind != b"blob":
                    raise ValueError(f"Git object mismatch: {revision}:{name}")
                offset = end + 1
                data = contents[offset:offset + int(size)]
                offset += int(size) + 1
                content_digest.update(name.encode() + b"\0" + mode + b"\0" + data + b"\0")
                info = tarfile.TarInfo(name)
                info.mode = 0o755 if mode == b"100755" else 0o644
                if mode == b"120000":
                    info.type = tarfile.SYMTYPE
                    info.linkname = data.decode("utf-8")
                    archive.addfile(info)
                else:
                    info.size = len(data)
                    archive.addfile(info, io.BytesIO(data))
            for name in gitlinks:
                info = tarfile.TarInfo(name)
                info.type, info.mode = tarfile.DIRTYPE, 0o755
                archive.addfile(info)
    return {
        "file_count": len(blobs), "gitlinks_not_materialized": gitlinks,
        "public_tree_sha256": content_digest.hexdigest(),
        "archive_sha256": sha256(destination), "archive_bytes": destination.stat().st_size,
    }


def build(task_root: Path, cache: Path, output: Path) -> None:
    selection = json.loads((HERE / "task_selection.json").read_text())
    ids = [task_id for split in ("train", "validation", "test") for task_id in selection[split]]
    if len(ids) != 30 or len(set(ids)) != 30:
        raise ValueError("G1 requires 30 distinct tasks in disjoint splits")
    output.mkdir(parents=True, exist_ok=False)
    (output / "archives").mkdir()
    schema = (HERE / "protocol/spec_output_schema.json").read_text()
    system = (HERE / "protocol/system_prompt.txt").read_text() + "\nOUTPUT JSON SCHEMA\n" + schema
    rows, inventory = [], []
    for split in ("train", "validation", "test"):
        split_rows = []
        for task_id in selection[split]:
            source = task_root / task_id
            metadata = tomllib.loads((source / "task.toml").read_text())["metadata"]
            description = (source / "instruction.md").read_text()
            revision = metadata["base_commit_hash"]
            archive_name = f"archives/{task_id}.tar.gz"
            info = export_repository(cache / task_id, revision, output / archive_name)
            user = {
                "task_id": task_id, "base_root": "/work/base",
                "task": {"title": task_id, "description": description},
                "managed_delivery": [],
            }
            row = {
                "task_id": task_id, "split": split,
                "repository_url": metadata["repository_url"], "base_commit": revision,
                "archive": archive_name,
                "prompt": [{"role": "system", "content": system}, {
                    "role": "user", "content": "SPEC INPUT\n" + json.dumps(user, ensure_ascii=False, indent=2),
                }],
            }
            split_rows.append(row)
            rows.append(row)
            inventory.append({
                "task_id": task_id, "split": split, "base_commit": revision,
                "repository_url": metadata["repository_url"], "archive": archive_name,
                "instruction_sha256": sha256(source / "instruction.md"), **info,
            })
        (output / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in split_rows),
            encoding="utf-8",
        )
    (output / "tasks.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8",
    )
    files = {
        path.relative_to(HERE).as_posix(): sha256(path)
        for path in sorted(output.rglob("*")) if path.is_file()
    }
    for name in ("task_selection.json", "config.json", "requirements-train.txt"):
        files[name] = sha256(HERE / name)
    for path in sorted((HERE / "protocol").iterdir()):
        files[path.relative_to(HERE).as_posix()] = sha256(path)
    for path in sorted(HERE.glob("*.py")) + sorted((HERE / "tests").glob("*.py")):
        files[path.relative_to(HERE).as_posix()] = sha256(path)
    for name in ("README.md", "requirements-check.txt", ".gitignore"):
        files[name] = sha256(HERE / name)
    manifest = {
        "experiment": "G1", "material_version": 5,
        "model": "Qwen/Qwen3.5-9B",
        "model_revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "initialization": "original_base_plus_new_lora",
        "algorithm": "GRPO", "reward": "binary_extracted_nonempty_ledger_delivery",
        "split_counts": {split: len(selection[split]) for split in ("train", "validation", "test")},
        "independent_tasks": 30, "teacher_answers_used": False, "teacher_trajectories_used": False,
        "hidden_tests_used": False, "solution_files_used": False,
        "repository_exports": "Git blobs at original base commit; benchmark tests/ and solution/ excluded",
        "source_task_root": str(task_root.resolve()),
        "split_notes": selection["notes"], "tasks": inventory, "files_sha256": files,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"experiment": "G1", "counts": manifest["split_counts"],
                      "archive_bytes": sum(t["archive_bytes"] for t in inventory)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, default=PROJECT / ".benchmarks/deep-swe-clean-v4.1-taskgraph-clone/tasks")
    parser.add_argument("--cache", type=Path, default=PROJECT / "a_agent_finetune/.cache/baselines")
    parser.add_argument("--output", type=Path, default=HERE / "data")
    args = parser.parse_args()
    build(args.task_root, args.cache, args.output)
