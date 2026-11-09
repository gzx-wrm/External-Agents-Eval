"""Copy only ignored compiled/version metadata; compatible with old task-image Python."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def snapshot(source, destination):
    listing = subprocess.check_output([
        "git", "-C", source, "ls-files", "--others", "--ignored", "--exclude-standard", "-z",
    ])
    files = []
    for raw in listing.split(b"\0"):
        if not raw:
            continue
        relative = raw.decode("utf-8")
        parts = relative.split("/")
        if parts[0] in {"build", "dist", ".tox", ".venv", "venv"}:
            continue
        if not (
            relative.endswith((".so", ".pyd"))
            or any(part.endswith(".egg-info") for part in parts)
            or parts[-1] in {"_version.py", "version.py"}
        ):
            continue
        src = os.path.join(source, relative)
        if not os.path.isfile(src):
            continue
        dst = os.path.join(destination, relative)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        files.append({"path": relative, "sha256": digest(dst), "bytes": os.path.getsize(dst)})
    os.makedirs(destination, exist_ok=True)
    report = {"source": source, "files": files, "total_bytes": sum(item["bytes"] for item in files)}
    with open(os.path.join(destination, "manifest.json"), "w") as stream:
        json.dump(report, stream)
    return report


def seed(snapshot_root, worktree):
    with open(os.path.join(snapshot_root, "manifest.json")) as stream:
        report = json.load(stream)
    for item in report["files"]:
        source = os.path.join(snapshot_root, item["path"])
        target = os.path.join(worktree, item["path"])
        if digest(source) != item["sha256"]:
            raise ValueError("Runtime snapshot changed: " + item["path"])
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copy2(source, target)
    return {"worktree": worktree, "runtime_files": len(report["files"]), "bytes": report["total_bytes"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["snapshot", "seed"])
    parser.add_argument("--source")
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--worktree")
    args = parser.parse_args()
    result = snapshot(args.source, args.snapshot) if args.mode == "snapshot" else seed(args.snapshot, args.worktree)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
