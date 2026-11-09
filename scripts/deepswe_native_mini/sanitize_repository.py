"""Preserve DeepSWE's public submodule files while removing all Git history."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path


def public_snapshot(directory: Path) -> dict:
    """Hash the visible snapshot, including links/modes, excluding Git metadata."""
    digest = hashlib.sha256()
    counts = {"files": 0, "directories": 0, "symlinks": 0}
    if not directory.is_dir():
        return {"exists": False, "sha256": digest.hexdigest(), **counts}
    for current, directories, files in os.walk(str(directory), followlinks=False):
        directories[:] = sorted(name for name in directories if name != ".git")
        for name in sorted(directories + [name for name in files if name != ".git"]):
            path = Path(current) / name
            relative = str(path.relative_to(directory))
            mode = path.lstat().st_mode
            digest.update((relative + "\0" + str(mode) + "\0").encode())
            if path.is_symlink():
                counts["symlinks"] += 1
                digest.update(os.readlink(str(path)).encode())
            elif path.is_file():
                counts["files"] += 1
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
            else:
                counts["directories"] += 1
            digest.update(b"\0")
    return {"exists": True, "sha256": digest.hexdigest(), **counts}


def remove_git_metadata(directory: Path) -> int:
    removed = 0
    if not directory.is_dir():
        return removed
    for current, directories, files in os.walk(str(directory), followlinks=False):
        if ".git" in directories or ".git" in files:
            path = Path(current) / ".git"
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(str(path))
            else:
                path.unlink()
            directories[:] = [name for name in directories if name != ".git"]
            removed += 1
    return removed


def sanitize(repository: Path, revision: str) -> dict:
    def git(*args, input=None, env=None):
        return subprocess.run(  # noqa: UP022 - task images include Python 3.5/3.6
            ["git", "-C", str(repository), *args], input=input, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
        ).stdout

    tree = git("rev-parse", revision + "^{tree}").decode().strip()
    # Copy only the baseline's trees and blobs. Never copy a historical commit,
    # even an unreachable object, reflog, ref, remote or alternate object store.
    objects = {tree}
    submodules = []
    for row in git("ls-tree", "-r", "-t", "-z", revision).split(b"\0"):
        if row:
            raw_metadata, raw_path = row.split(b"\t", 1)
            metadata = raw_metadata.split()
            if metadata[1] == b"commit":
                path = raw_path.decode()
                directory = repository / path
                snapshot = public_snapshot(directory)
                removed = remove_git_metadata(directory)
                submodules.append({"path": path, "gitlink_sha": metadata[2].decode(),
                                   "snapshot": snapshot, "git_metadata_removed": removed})
                # Gitlinks refer to a separate repository. Preserve their tree
                # entry, but never pack the referenced commit or its history.
                continue
            objects.add(metadata[2].decode())
    git("-c", "submodule.recurse=false", "reset", "--hard", revision)
    pack = git("pack-objects", "--stdout", input=("\n".join(sorted(objects)) + "\n").encode())
    shutil.rmtree(str(repository / ".git"))
    git("init", "--template=")
    git("symbolic-ref", "HEAD", "refs/heads/eval-baseline")
    git("config", "core.autocrlf", "false")
    git("unpack-objects", "-r", input=pack)
    commit = git(
        "-c", "user.name=Evaluation", "-c", "user.email=eval@localhost",
        "commit-tree", tree, input=b"Frozen evaluation baseline\n",
        env={**os.environ, "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+00:00",
             "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+00:00"},
    ).decode().strip()
    git("update-ref", "refs/heads/eval-baseline", commit)
    git("-c", "submodule.recurse=false", "reset", "--hard", commit)
    git("reflog", "expire", "--expire=now", "--all")
    object_rows = git("cat-file", "--batch-all-objects", "--batch-check=%(objectname) %(objecttype)").decode().splitlines()
    retained = {row.split()[0]: row.split()[1] for row in object_rows}
    if (
        list(retained.values()).count("commit") != 1
        or set(retained) != objects | {commit}
        or git("rev-parse", "HEAD^{tree}").decode().strip() != tree
    ):
        raise ValueError("Repository isolation did not preserve a single baseline commit")
    if git("remote").strip():
        raise ValueError("Repository isolation left a remote configured")
    for item in submodules:
        if public_snapshot(repository / item["path"]) != item["snapshot"]:
            raise ValueError("Submodule public files changed during Git isolation: " + item["path"])
        if (repository / item["path"] / ".git").exists():
            raise ValueError("Submodule Git metadata remains visible")
    return {"synthetic_commit": commit, "tree_sha": tree, "commit_object_count": 1,
            "history_removed": True, "remotes_removed": True, "object_count": len(retained),
            "submodule_snapshots": submodules}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path("/testbed"))
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    print(json.dumps(sanitize(args.repository, args.revision), sort_keys=True))


if __name__ == "__main__":
    main()
