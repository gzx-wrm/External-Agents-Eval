"""Replay the A evidence collector over recorded A trajectories (no model calls).

For every task of a finished root, rebuild the sanitized baseline in an offline
container from the pinned image, run the current ``observed_evidence`` on the
recorded Pi trajectory, and compare the result with the recorded
``v6.2/evidence.json``.  Tasks whose A failed in evidence mapping are replayed too.
Results go to a separate output directory; the source root is read-only.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shlex
import sys
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run  # noqa: E402

from external_agents_eval.agents.pier.pi_role_runner import ContainerGitWorkspaceManager  # noqa: E402
from external_agents_eval.workflows.solve_review_repair.artifacts import sha256_json  # noqa: E402
from external_agents_eval.workflows.solve_review_repair.ft_evidence import observed_evidence  # noqa: E402

FIELDS = ("evidence", "read_provenance", "skipped_reads", "tool_counts",
          "repository_inventory_provenance", "environment_modified_lines_excluded")


def isolate(native, root, container, task):
    """The repository steps of run.prepare_a_container, without network or Pi."""
    patch = run.container_checked(container, native.snapshot_command(task["base_commit"]))
    native.upload(container, root / "scripts/sanitize_repository.py", "/tmp/native_sanitize_repository.py")
    audit = json.loads(run.container_checked(
        container, "python /tmp/native_sanitize_repository.py --repository /app --revision "
        + shlex.quote(task["base_commit"])).splitlines()[-1])
    run.container_checked(container, "git clean -fd -- .")
    if patch:
        native.upload(container, patch.encode(), "/tmp/native_environment.patch")
        run.container_checked(container, "git apply --binary /tmp/native_environment.patch && "
                              "rm /tmp/native_environment.patch")
    run.container_checked(container, "git config core.hooksPath /dev/null && git branch main HEAD")
    return audit["synthetic_commit"], patch


async def collect(container, task, trajectory, base_commit, patch, transfer):
    environment = run.PiContainerEnvironment(container, instance_id=task["task_id"], proxy_env={})
    workspace = ContainerGitWorkspaceManager(environment=environment, transfer_dir=transfer,
                                             repository="/app")
    paths = await workspace.prepare(base_commit=base_commit)
    if patch:
        await workspace.apply_patch(worktree=paths.spec_review, patch=patch.encode())
    try:
        return await observed_evidence(trajectory, task_id=task["task_id"], base_root=paths.spec_review,
                                       base_commit=base_commit, workspace=workspace)
    finally:
        await workspace.cleanup()


def replay(native, source, row, output):
    tid = row["task_id"]
    a = source / "a" / tid
    record = run.read(a / "a-result.json") if (a / "a-result.json").exists() else {}
    workflow = a / "v6_review_repair"
    trajectory = run.read(workflow / "v6.2/evidence-search/trajectory.json")
    stored_path = workflow / "v6.2/evidence.json"
    stored = run.read(stored_path) if stored_path.exists() else None
    client = native.docker.from_env(timeout=600)
    container = None
    try:
        image = run.require_image(client, row)
        container = client.containers.create(
            image.id, entrypoint=[], command=["sleep", "infinity"], working_dir="/app",
            name="v63h-replay-" + uuid.uuid4().hex[:12], network_mode="none", cap_drop=["ALL"],
            security_opt=["no-new-privileges:true"], labels={**run.LABEL, "stage": "evidence-replay"},
        )
        container.start()
        base_commit, patch = isolate(native, source, container, row["task"])
        if record.get("base_commit") and record["base_commit"] != base_commit:
            raise RuntimeError("Replay baseline differs from the recorded A baseline")
        try:
            part = asyncio.run(collect(container, row["task"], trajectory, base_commit, patch,
                                       output / "transfer" / tid))
        except ValueError as error:
            return {"task_id": tid, "a_status": record.get("status"), "replay": "error",
                    "error": str(error)}
    finally:
        if container is not None:
            container.remove(force=True)
        client.close()
    result = {"task_id": tid, "a_status": record.get("status"), "replay": "ok",
              "evidence_files": len({item["path"] for item in part["evidence"]}),
              "evidence_spans": len(part["evidence"]), "evidence_sha256": sha256_json(part["evidence"])}
    if stored is not None:
        differing = [field for field in FIELDS if part.get(field) != stored.get(field)]
        result.update(stored_evidence_sha256=sha256_json(stored["evidence"]),
                      identical=not differing, differing_fields=differing)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source, output = args.source_root.resolve(), args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit("Use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    native = run.load_native(source)
    selection = run.read(source / "selection.json")
    results = []
    for row in selection["tasks"]:
        if not (source / "a" / row["task_id"] / "v6_review_repair/v6.2/evidence-search/trajectory.json").exists():
            results.append({"task_id": row["task_id"], "replay": "no_trajectory"})
            continue
        try:
            results.append(replay(native, source, row, output))
        except Exception as error:  # noqa: BLE001 - one task must not stop the audit
            results.append({"task_id": row["task_id"], "replay": "infrastructure_error",
                            "error": f"{type(error).__name__}: {error}"})
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    recorded = [r for r in results if "identical" in r]
    summary = {
        "source_root": str(source), "collector": str(Path(run.PROJECT) / "external_agents_eval/"
                                                     "workflows/solve_review_repair/ft_evidence.py"),
        "collector_sha256": run.file_sha256(Path(run.PROJECT) / "external_agents_eval/workflows/"
                                            "solve_review_repair/ft_evidence.py"),
        "model_api_calls": 0, "tasks": len(results), "recorded_packets": len(recorded),
        "identical": sum(r["identical"] for r in recorded),
        "previously_failed_now_ok": [r["task_id"] for r in results
                                     if r.get("a_status") == "ledger_unavailable" and r["replay"] == "ok"],
        "results": results,
    }
    run.save(output / "replay.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
