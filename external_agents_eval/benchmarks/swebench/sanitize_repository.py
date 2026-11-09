"""Run inside a disposable task container before exposing it to an agent."""
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


def sanitize(repository: Path, revision: str) -> dict:
    def git(*args, input=None, env=None):
        return subprocess.run(  # noqa: UP022 - task images include Python 3.5/3.6
            ["git", "-C", str(repository), *args], input=input, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
        ).stdout

    git("reset", "--hard", revision)
    tree = git("rev-parse", revision + "^{tree}").decode().strip()
    # Copy only the baseline's trees and blobs. Never copy a historical commit,
    # even an unreachable object, reflog, ref, remote or alternate object store.
    objects = {tree}
    for row in git("ls-tree", "-r", "-t", "-z", revision).split(b"\0"):
        if row:
            metadata = row.split(b"\t", 1)[0].split()
            if metadata[1] == b"commit":
                raise ValueError("Submodule requires an explicit isolated snapshot")
            objects.add(metadata[2].decode())
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
    git("reset", "--hard", commit)
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
    return {"synthetic_commit": commit, "tree_sha": tree, "commit_object_count": 1,
            "history_removed": True, "remotes_removed": True, "object_count": len(retained)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path("/testbed"))
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    print(json.dumps(sanitize(args.repository, args.revision), sort_keys=True))


if __name__ == "__main__":
    main()
