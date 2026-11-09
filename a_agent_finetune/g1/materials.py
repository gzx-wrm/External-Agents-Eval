"""Portable task loading, integrity checks, and extraction of public snapshots."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(root: Path, split: str) -> list[dict]:
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unknown G1 split: {split}")
    return [json.loads(line) for line in (root / "data" / f"{split}.jsonl").read_text().splitlines()]


def verify_materials(root: Path) -> dict:
    manifest = json.loads((root / "data/manifest.json").read_text())
    for name, expected in manifest["files_sha256"].items():
        if file_sha256(root / name) != expected:
            raise ValueError(f"G1 material hash differs: {name}")
    splits = {split: load_rows(root, split) for split in ("train", "validation", "test")}
    ids = []
    for split, rows in splits.items():
        if len(rows) != manifest["split_counts"][split]:
            raise ValueError(f"G1 split count differs: {split}")
        for row in rows:
            ids.append(row["task_id"])
            if row["split"] != split or [m["role"] for m in row["prompt"]] != ["system", "user"]:
                raise ValueError("G1 prompts must contain only a system message and original task")
            user = json.loads(row["prompt"][1]["content"].removeprefix("SPEC INPUT\n"))
            if set(user) != {"task_id", "task", "base_root", "managed_delivery"}:
                raise ValueError("Unexpected input field in G1 prompt")
            if set(user["task"]) != {"title", "description"} or user["task_id"] != row["task_id"]:
                raise ValueError("Malformed G1 task input")
    if len(ids) != len(set(ids)) or len(ids) != manifest["independent_tasks"]:
        raise ValueError("G1 splits overlap or task count differs")
    return {"experiment": "G1", "verified": True, "counts": manifest["split_counts"],
            "checked_files": len(manifest["files_sha256"])}


def repository_for(root: Path, row: dict) -> Path:
    repository = root / "data/repositories" / row["task_id"]
    marker = root / "data/repositories/.provenance" / f"{row['task_id']}.json"
    archive = root / "data" / row["archive"]
    digest = file_sha256(archive)
    manifest = json.loads((root / "data/manifest.json").read_text())
    expected = next(t for t in manifest["tasks"] if t["task_id"] == row["task_id"])
    if digest != expected["archive_sha256"] or row["base_commit"] != expected["base_commit"]:
        raise ValueError("G1 snapshot identity differs")
    if marker.exists():
        if json.loads(marker.read_text()) != {"archive_sha256": digest, "base_commit": row["base_commit"]}:
            raise ValueError("Existing G1 repository has different provenance")
        return repository
    repository.mkdir(parents=True, exist_ok=False)
    try:
        with tarfile.open(archive) as handle:
            handle.extractall(repository, filter="data")
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"archive_sha256": digest, "base_commit": row["base_commit"]}) + "\n")
    except BaseException:
        shutil.rmtree(repository)
        raise
    return repository
