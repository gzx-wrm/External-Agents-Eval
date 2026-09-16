from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from external_agents_eval.agents.config import AGENTS, DEFAULT_MODEL
from external_agents_eval.agents.runtime_build import RUNTIME_PLATFORMS, build_runtime_images
from external_agents_eval.benchmarks.deepswe.config import (
    DEEPSWE_AGENTS,
    DEFAULT_SAMPLE_SEED,
    DEFAULT_SAMPLE_SIZE,
)
from external_agents_eval.benchmarks.deepswe.config import (
    DEFAULT_CHECKOUT_DIR as DEFAULT_DEEPSWE_CHECKOUT,
)
from external_agents_eval.benchmarks.deepswe.config import (
    DEFAULT_JOBS_DIR as DEFAULT_DEEPSWE_JOBS,
)
from external_agents_eval.benchmarks.deepswe.config import (
    DEFAULT_MANIFEST as DEFAULT_DEEPSWE_MANIFEST,
)
from external_agents_eval.benchmarks.deepswe.manifest import (
    build_manifest_payload,
    ensure_checkout,
)
from external_agents_eval.benchmarks.deepswe.manifest import (
    load_manifest as load_deepswe_manifest,
)
from external_agents_eval.benchmarks.deepswe.manifest import (
    write_manifest as write_deepswe_manifest,
)
from external_agents_eval.benchmarks.deepswe.report import (
    collect_reviewer_experiment_summary,
    collect_summary,
)
from external_agents_eval.benchmarks.deepswe.reviewer_experiment_manifest import (
    build_reviewer_experiment_manifest,
    load_reviewer_experiment_manifest,
)
from external_agents_eval.benchmarks.deepswe.runner import run_pier
from external_agents_eval.benchmarks.swebench.config import (
    DEFAULT_MANIFEST,
    DEFAULT_RESULTS_DIR,
    DEFAULT_SWEBENCH_RUNNER,
)
from external_agents_eval.benchmarks.swebench.manifest import (
    atomic_write_json,
    load_task_manifest,
)
from external_agents_eval.benchmarks.swebench.report import run_official_report
from external_agents_eval.benchmarks.swebench.runner import generate, load_entries
from external_agents_eval.benchmarks.swebench.status import collect_status
from external_agents_eval.benchmarks.swebench.task_images import (
    audit_official_judge_dependencies,
    audit_task_images,
    ensure_task_image_audit,
    task_image_audit_payload,
)


def _add_build_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--agent", choices=[*AGENTS, "all"], default="all")
    parser.add_argument(
        "--architecture",
        choices=[*RUNTIME_PLATFORMS, "all"],
        default="all",
        help="Build runtimes matching prebuilt task-image architectures",
    )


def _add_swebench_audit_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--details", action="store_true")


def _add_swebench_generate_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--agent", choices=AGENTS, required=True)
    parser.add_argument("--task-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--agent-name", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--num-workers", type=int, default=3)
    parser.add_argument("--hours-per-task", type=int, default=5)
    parser.add_argument("--max-turns", type=int, default=10, help="Hermes turn limit")
    parser.add_argument("--resume", action="store_true")


def _add_swebench_report_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--agent-dir", type=Path, required=True)
    parser.add_argument("--task-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--num-workers", type=int, default=3)
    parser.add_argument("--report-retries", type=int, default=2)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--swebench-runner", type=Path, default=DEFAULT_SWEBENCH_RUNNER)


def _add_swebench_status_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--agent-dir", type=Path, required=True)
    parser.add_argument("--details", action="store_true")


def _add_swebench_commands(subparsers) -> None:
    audit = subparsers.add_parser(
        "audit-task-images",
        help="Verify exact prebuilt SWE-bench images without pulling or building",
    )
    _add_swebench_audit_arguments(audit)
    generate_parser = subparsers.add_parser("generate", help="Generate SWE-bench patches")
    _add_swebench_generate_arguments(generate_parser)
    report = subparsers.add_parser("report", help="Run the official SWE-bench judge")
    _add_swebench_report_arguments(report)
    status = subparsers.add_parser("status", help="Summarize SWE-bench generation")
    _add_swebench_status_arguments(status)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate external coding agents across pinned benchmarks"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    agents = subparsers.add_parser("agents", help="Manage shared agent runtimes")
    agent_commands = agents.add_subparsers(dest="agent_command", required=True)
    grouped_build = agent_commands.add_parser(
        "build-runtimes", help="Build pinned external-agent runtimes"
    )
    _add_build_arguments(grouped_build)

    swebench = subparsers.add_parser("swebench", help="SWE-bench backend")
    swebench_commands = swebench.add_subparsers(dest="swebench_command", required=True)
    _add_swebench_commands(swebench_commands)

    deepswe = subparsers.add_parser("deepswe", help="DeepSWE v1.1 backend")
    deepswe_commands = deepswe.add_subparsers(dest="deepswe_command", required=True)
    prepare = deepswe_commands.add_parser(
        "prepare", help="Clone/validate v1.1 and create an immutable task manifest"
    )
    prepare.add_argument("--checkout-dir", type=Path, default=DEFAULT_DEEPSWE_CHECKOUT)
    prepare.add_argument("--manifest", type=Path, default=DEFAULT_DEEPSWE_MANIFEST)
    prepare.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    prepare.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)

    review_prepare = deepswe_commands.add_parser(
        "prepare-review-manifest",
        help="Freeze an existing-P0 reviewer experiment manifest",
    )
    review_prepare.add_argument("--source-job-dir", type=Path, required=True)
    review_prepare.add_argument("--checkout-dir", type=Path, default=DEFAULT_DEEPSWE_CHECKOUT)
    review_prepare.add_argument("--output", type=Path, required=True)
    review_prepare.add_argument(
        "--scope",
        choices=("near-miss16", "random60"),
        default="near-miss16",
        help="Freeze the original near-miss set or every task in the source random60 manifest",
    )

    deep_run = deepswe_commands.add_parser(
        "run", help="Run the fixed manifest through official Pier"
    )
    deep_run.add_argument("--agent", choices=DEEPSWE_AGENTS, required=True)
    deep_run.add_argument("--checkout-dir", type=Path, default=DEFAULT_DEEPSWE_CHECKOUT)
    deep_run.add_argument("--manifest", type=Path, default=DEFAULT_DEEPSWE_MANIFEST)
    deep_run.add_argument("--jobs-dir", type=Path, default=DEFAULT_DEEPSWE_JOBS)
    deep_run.add_argument("--job-name", default=None)
    deep_run.add_argument("--model", default=DEFAULT_MODEL)
    deep_run.add_argument("--num-workers", type=int, default=1)
    deep_run.add_argument("--attempts", type=int, default=1)
    deep_run.add_argument(
        "--p0-manifest",
        type=Path,
        default=None,
        help="Frozen existing-P0 manifest for the pi-review-repair workflow",
    )
    deep_run.add_argument(
        "--workflow-role-timeout",
        type=int,
        default=None,
        help=(
            "Optional hard safety timeout in seconds for internal workflow roles; "
            "V1-V4 apply it to Solver/Repair and V5 also uses it for A/I/R/S "
            "invocations and persisted A/S stage deadlines"
        ),
    )
    deep_run.add_argument(
        "--few-shot-dir",
        type=Path,
        default=None,
        help=(
            "Frozen, provenance-backed artifacts matching the selected review contract "
            "(v1 reference-assisted, v2 synthetic, or v3 obligation-level synthetic)"
        ),
    )
    deep_run.add_argument(
        "--review-arm",
        choices=("E0", "E1", "E2"),
        default="E0",
        help=(
            "Reviewer control arm: E0=v1 baseline, E1=source-grounded review "
            "(v2 by default), E2=E1 plus mandatory Repair audit"
        ),
    )
    deep_run.add_argument(
        "--review-contract-version",
        choices=("1.0", "2.0", "3.0"),
        default=None,
        help=(
            "Explicit Reviewer contract. Defaults to v1 for E0 and v2 for E1/E2; "
            "select 3.0 to run the obligation-level V3 workflow."
        ),
    )
    deep_run.add_argument(
        "--review-prompt-version",
        choices=("4.0",),
        default=None,
        help=(
            "Optional Reviewer prompt override. Select 4.0 with contract 3.0 to run "
            "the counterexample-first V4 Stage B prompt."
        ),
    )
    deep_run.add_argument(
        "--task-name",
        dest="task_names",
        action="append",
        default=None,
        help=("Run only this task from the active manifest; repeat for a subset rerun"),
    )
    deep_run.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help=(
            "Optional Hermes turn limit; when omitted, the agent is limited only "
            "by the benchmark task timeout"
        ),
    )
    deep_run.add_argument("--environment", choices=["docker", "modal"], default="docker")
    deep_run.add_argument(
        "--trial-cpus",
        type=int,
        default=None,
        help="Optional CPU limit applied to each task and separate verifier container",
    )
    deep_run.add_argument(
        "--trial-memory-mb",
        type=int,
        default=None,
        help="Optional memory limit in MiB applied to each task and separate verifier container",
    )
    deep_run.add_argument("--pier-executable", default="pier")
    deep_run.add_argument("--dry-run", action="store_true")

    deep_summary = deepswe_commands.add_parser(
        "summary", aliases=["status"], help="Aggregate Pier reward.json/results.json files"
    )
    deep_summary.add_argument("--job-dir", type=Path, required=True)
    deep_summary.add_argument("--manifest", type=Path, default=DEFAULT_DEEPSWE_MANIFEST)
    deep_summary.add_argument("--details", action="store_true")
    deep_summary.add_argument("--output", type=Path, default=None)

    review_summary = deepswe_commands.add_parser(
        "review-summary",
        help="Summarize paired outcomes and protocol metrics for a reviewer experiment",
    )
    review_summary.add_argument("--job-dir", type=Path, required=True)
    review_summary.add_argument("--p0-manifest", type=Path, required=True)
    review_summary.add_argument("--details", action="store_true")
    review_summary.add_argument("--output", type=Path, default=None)

    # Deprecated flat aliases preserve all pre-refactor commands.
    build = subparsers.add_parser(
        "build-runtimes", help="Deprecated alias for 'agents build-runtimes'"
    )
    _add_build_arguments(build)
    _add_swebench_commands(subparsers)
    return parser


def _run_build(args: argparse.Namespace, repository_root: Path) -> int:
    agents = list(AGENTS) if args.agent == "all" else [args.agent]
    architectures = list(RUNTIME_PLATFORMS) if args.architecture == "all" else [args.architecture]
    build_runtime_images(repository_root, agents, architectures)
    return 0


def _run_swebench(command: str, args: argparse.Namespace) -> int:
    if command == "audit-task-images":
        import docker

        manifest = load_task_manifest(args.task_manifest)
        entries = load_entries(manifest)
        client = docker.from_env()
        try:
            payload = task_image_audit_payload(audit_task_images(client, entries))
        finally:
            client.close()
        if not args.details:
            payload.pop("images", None)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    if command == "generate":
        manifest = load_task_manifest(args.task_manifest)
        counts = generate(
            agent_id=args.agent,
            manifest=manifest,
            results_dir=args.results_dir,
            agent_name=args.agent_name,
            model=args.model,
            num_workers=args.num_workers,
            hours_per_task=args.hours_per_task,
            max_turns=args.max_turns,
            resume=args.resume,
        )
        print(json.dumps(counts, indent=2))
        return 0 if not counts.get("failed") and not counts.get("blocked") else 1
    if command == "report":
        import docker

        manifest = load_task_manifest(args.task_manifest)
        entries = load_entries(manifest)
        agent_dir = args.agent_dir.expanduser().resolve()
        client = docker.from_env()
        try:
            task_images = audit_task_images(client, entries)
            ensure_task_image_audit(agent_dir / "task_image_audit.json", task_images)
            judge_audit = audit_official_judge_dependencies(client, entries)
        finally:
            client.close()
        judge_audit["audited_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        atomic_write_json(agent_dir / "official_image_preflight.json", judge_audit)
        run_id = args.run_id or dt.datetime.now(dt.timezone.utc).strftime("external_%Y%m%d_%H%M%S")
        output = run_official_report(
            agent_dir=agent_dir,
            dataset_name=manifest.dataset_name,
            task_ids=manifest.task_ids,
            num_workers=args.num_workers,
            run_id=run_id,
            runner_path=args.swebench_runner,
            retries=args.report_retries,
        )
        print(f"Official report output: {output}")
        return 0
    if command == "status":
        status = collect_status(args.agent_dir.expanduser().resolve())
        if not args.details:
            status.pop("details", None)
        print(json.dumps(status, indent=2, ensure_ascii=False))
        return 0
    raise ValueError(f"Unknown SWE-bench command: {command}")


def _run_deepswe(command: str, args: argparse.Namespace) -> int:
    if command == "prepare":
        checkout = ensure_checkout(args.checkout_dir)
        payload = build_manifest_payload(
            checkout,
            sample_size=args.sample_size,
            seed=args.sample_seed,
        )
        manifest_path = write_deepswe_manifest(args.manifest, payload)
        print(
            json.dumps(
                {"checkout": str(checkout), "manifest": str(manifest_path), **payload},
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    if command == "run":
        manifest = load_deepswe_manifest(args.manifest)
        job_dir = run_pier(
            checkout=args.checkout_dir,
            manifest=manifest,
            agent_id=args.agent,
            jobs_dir=args.jobs_dir,
            job_name=args.job_name,
            model=args.model,
            num_workers=args.num_workers,
            attempts=args.attempts,
            max_turns=args.max_turns,
            pier_executable=args.pier_executable,
            environment=args.environment,
            dry_run=args.dry_run,
            p0_manifest=args.p0_manifest,
            workflow_timeout_sec=args.workflow_role_timeout,
            few_shot_dir=args.few_shot_dir,
            review_arm=args.review_arm,
            review_contract_version=args.review_contract_version,
            review_prompt_version=args.review_prompt_version,
            task_ids=tuple(args.task_names) if args.task_names else None,
            trial_cpus=args.trial_cpus,
            trial_memory_mb=args.trial_memory_mb,
        )
        print(f"Pier job directory: {job_dir}")
        return 0
    if command == "prepare-review-manifest":
        output = build_reviewer_experiment_manifest(
            source_job_dir=args.source_job_dir,
            checkout=args.checkout_dir,
            output=args.output,
            scope=args.scope,
        )
        print(f"Reviewer experiment manifest: {output}")
        return 0
    if command in {"summary", "status"}:
        manifest = load_deepswe_manifest(args.manifest)
        payload = collect_summary(args.job_dir, manifest)
        if args.output is not None:
            atomic_write_json(args.output.expanduser().resolve(), payload)
        if not args.details:
            payload.pop("details", None)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    if command == "review-summary":
        manifest = load_reviewer_experiment_manifest(args.p0_manifest)
        payload = collect_reviewer_experiment_summary(args.job_dir, manifest)
        if args.output is not None:
            atomic_write_json(args.output.expanduser().resolve(), payload)
        if not args.details:
            payload.pop("details", None)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    raise ValueError(f"Unknown DeepSWE command: {command}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    repository_root = Path(__file__).resolve().parents[1]

    if args.command == "agents":
        return _run_build(args, repository_root)
    if args.command == "swebench":
        return _run_swebench(args.swebench_command, args)
    if args.command == "deepswe":
        return _run_deepswe(args.deepswe_command, args)
    if args.command == "build-runtimes":
        return _run_build(args, repository_root)
    if args.command in {"audit-task-images", "generate", "report", "status"}:
        return _run_swebench(args.command, args)
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
