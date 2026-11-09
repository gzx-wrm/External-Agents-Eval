"""Extend the held-out split to 20 tasks with 3 of the unused passed train60 tasks.

The four passed train60 tasks that were kept out of E10 training are the only
candidates.  Two were excluded solely because they share a repository with an
existing test task, which does not matter once they are themselves test tasks.
The other two were excluded because their teacher trajectory does not fit the
training window; the longest of those is dropped so the test set gains only one
near-window task, and the choice is recorded rather than tuned.

For each selected task this writes the frozen baseline tree, the A task input
taken from the train60 trial prompt, and a metadata row carrying the base
commit.  Nothing in the existing held-out inputs or in tasks.jsonl is modified.
No model is called.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tomllib
from datetime import datetime, timezone
from pathlib import Path

try:
    from .build_f_pilot import BASELINE_CACHE, ensure_baseline, git, spec_input
    from .build_e_teacher_trajectory import sha, write_json
except ImportError:
    from build_f_pilot import BASELINE_CACHE, ensure_baseline, git, spec_input
    from build_e_teacher_trajectory import sha, write_json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_SELECTION = HERE / "data/train60_selection_v1/selection.json"
DEFAULT_MANIFEST = ROOT / "manifests/deepswe_v1.1_v6.1_train60_exclude30_seed20261014.json"
TASKS_ROOT = ROOT / ".benchmarks/deep-swe/tasks"
REPOSITORIES = HERE / "g1/data/repositories"
EXTRA_TASKS = HERE / "g1/data/tasks-extra.jsonl"
HELDOUT_SOURCE = HERE / "runs/e6-heldout17-source"
OUTPUT = HERE / "data/heldout20_selection_v1"
SELECT_COUNT = 3


def repository_key(url: str) -> str:
    return url.rstrip("/").removesuffix(".git").lower()


def export_tree(task: str, url: str, commit: str) -> dict:
    """Write the baseline tree exactly as the existing task repositories store it."""

    cache = ensure_baseline(task, url, commit)
    target = REPOSITORIES / task
    if target.exists():
        raise FileExistsError(f"Repository tree already exists: {target}")
    target.mkdir(parents=True)
    archive = subprocess.run(["git", "-C", str(cache), "archive", commit],
                             check=True, stdout=subprocess.PIPE).stdout
    subprocess.run(["tar", "-x", "-C", str(target)], check=True, input=archive)
    tracked = sorted(x for x in git(cache, "ls-tree", "-r", "--name-only", commit).decode().splitlines() if x)
    on_disk = sorted(str(p.relative_to(target)) for p in target.rglob("*") if p.is_file())
    if tracked != on_disk:
        raise ValueError(f"Exported tree differs from the baseline commit: {task}")
    return {"files": len(tracked), "baseline_cache": str(cache.relative_to(ROOT))}


def select(selection: dict, manifest: dict) -> tuple[list[str], dict]:
    candidates = sorted(selection["excluded_passed_tasks"])
    repo = {task["task_id"]: repository_key(task["repository_url"]) for task in manifest["tasks"]}
    frozen = {json.loads(line)["task_id"]: repository_key(json.loads(line)["repository_url"])
              for line in (HERE / "g1/data/tasks.jsonl").read_text().splitlines()}
    train = set(selection["existing_training_tasks"]) | {row["task_id"] for row in selection["new_training_tasks"]}
    train_repositories = {repo.get(task) or frozen[task] for task in train}
    leaking = [task for task in candidates if repo[task] in train_repositories]
    if leaking:
        raise ValueError(f"Candidate shares a repository with a training task: {leaking}")
    lengths = {row["task_id"]: row["estimated_training_tokens"]
               for row in selection["new_training_tasks"]}
    estimates = json.loads((OUTPUT.parent / "train60_selection_v1/provenance/scan.json").read_text())
    lengths |= {row["task_id"]: row.get("est_train_tokens", 0) for row in estimates}
    # Drop the longest candidates first, so the test set keeps at most one
    # task whose teacher trajectory already fills the inference window.
    ordered = sorted(candidates, key=lambda task: (lengths[task], task))
    chosen = sorted(ordered[:SELECT_COUNT])
    return chosen, {
        "candidates": candidates, "chosen": chosen, "dropped": ordered[SELECT_COUNT:],
        "rule": f"All four unused passed tasks are candidates; none shares a repository with a training task. "
                f"Keep the {SELECT_COUNT} with the shortest teacher trajectories so at most one near-window task enters the split.",
        "estimated_teacher_trajectory_tokens": {task: lengths[task] for task in candidates},
    }


def build(selection_path: Path, manifest_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    selection = json.loads(selection_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    chosen, reason = select(selection, manifest)
    by_task = {task["task_id"]: task for task in manifest["tasks"]}
    status = {row["task_id"]: row for row in json.loads(
        (HERE / "runs/v6.1-deepseek-train60-overview-20261016-r1/task-status.json").read_text())}

    output.mkdir(parents=True)
    (output / "tasks").mkdir()
    rows = []
    for task in chosen:
        if status[task]["status"] != "passed":
            raise ValueError(f"Candidate is not a passed task: {task}")
        metadata = tomllib.loads((TASKS_ROOT / task / "task.toml").read_text())["metadata"]
        url, commit = metadata["repository_url"], metadata["base_commit_hash"]
        if url != by_task[task]["repository_url"] or commit != by_task[task]["base_commit"]:
            raise ValueError(f"task.toml disagrees with the manifest: {task}")
        trial = Path(status[task]["selected_result"]["source_result"]).parent
        prompt = trial / "v6_review_repair/v5/specification/attempt-1/prompt.md"
        a_input = spec_input(prompt)
        write_json(output / "tasks" / f"{task}.json",
                   {"task": a_input["task"], "managed_delivery": a_input.get("managed_delivery", [])})
        tree = export_tree(task, url, commit)
        rows.append({"task_id": task, "split": "test", "repository_url": url, "base_commit": commit,
                     "teacher_trial": str(trial.relative_to(ROOT)), "spec_prompt_sha256": sha(prompt),
                     "repository_files": tree["files"], "source_group": status[task]["group"]})
        print(f"{task}: {tree['files']} files at {commit[:10]}", flush=True)

    EXTRA_TASKS.write_text("".join(json.dumps(
        {"task_id": row["task_id"], "split": "test", "repository_url": row["repository_url"],
         "base_commit": row["base_commit"]}, ensure_ascii=False) + "\n" for row in rows))
    for task in chosen:
        source = output / "tasks" / f"{task}.json"
        target = HELDOUT_SOURCE / "inputs/tasks" / f"{task}.json"
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError(f"Existing held-out input differs: {target}")
        target.write_bytes(source.read_bytes())

    e6 = json.loads((HERE / "data/e6_normal13_loopheavy_v1/split_manifest.json").read_text())
    result = {
        "schema_version": "heldout20_selection_v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Held-out split grows to 20 tasks: the E6 3 validation + 14 test tasks plus 3 unused passed train60 tasks",
        "source_selection": str(selection_path.relative_to(ROOT)),
        "source_manifest": str(manifest_path.relative_to(ROOT)),
        "validation": e6["validation"], "test": e6["test"] + chosen,
        "added_test_tasks": rows, "selection_reason": reason,
        "extra_task_metadata": str(EXTRA_TASKS.relative_to(ROOT)),
        "repository_trees": str(REPOSITORIES.relative_to(ROOT)),
        "limits": "These 3 tasks were used to develop the teacher pipeline and were verified by the official "
                  "verifier, so they are held out of training but are not a blind benchmark.",
    }
    write_json(output / "selection.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = build(args.selection.resolve(), args.manifest.resolve(), args.output.resolve())
    print(json.dumps({"validation": len(result["validation"]), "test": len(result["test"]),
                      "added": [row["task_id"] for row in result["added_test_tasks"]],
                      "dropped": result["selection_reason"]["dropped"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
