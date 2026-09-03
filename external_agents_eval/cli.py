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
from external_agents_eval.benchmarks.deepswe.report import collect_summary
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
        "--max-turns",
        type=int,
        default=None,
        help=(
            "Optional Hermes turn limit; when omitted, the agent is limited only "
            "by the benchmark task timeout"
        ),
    )
    deep_run.add_argument("--environment", choices=["docker", "modal"], default="docker")
    deep_run.add_argument("--pier-executable", default="pier")
    deep_run.add_argument("--dry-run", action="store_true")

    deep_summary = deepswe_commands.add_parser(
        "summary", aliases=["status"], help="Aggregate Pier reward.json/results.json files"
    )
    deep_summary.add_argument("--job-dir", type=Path, required=True)
    deep_summary.add_argument("--manifest", type=Path, default=DEFAULT_DEEPSWE_MANIFEST)
    deep_summary.add_argument("--details", action="store_true")
    deep_summary.add_argument("--output", type=Path, default=None)

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
        )
        print(f"Pier job directory: {job_dir}")
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
