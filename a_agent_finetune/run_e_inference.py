"""Read-only E A-agent inference; Qwen tools in Docker, no downstream agents."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from external_agents_eval.agents.pier.pi_model_config import PiStageModelConfig
from external_agents_eval.agents.pier.trajectory import (
    extract_pi_final_assistant_text,
    pi_jsonl_to_atif,
)
from external_agents_eval.workflows.solve_review_repair.artifacts import (
    ArtifactStore,
    sha256_bytes,
    sha256_json,
)
from external_agents_eval.workflows.solve_review_repair.models import (
    RoleName,
    RoleRunResult,
    TaskInput,
)
from external_agents_eval.workflows.solve_review_repair.v5_contracts import validate_spec_output
from external_agents_eval.workflows.solve_review_repair.v6_config import V6Config
from external_agents_eval.workflows.solve_review_repair.v6_controller import V6Controller
from external_agents_eval.workflows.solve_review_repair.v6_fork import write_stage
from external_agents_eval.workflows.solve_review_repair.validators import extract_single_json_object

HERE = Path(__file__).resolve().parent
BASE_ROOT = "/work/base"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


class FrozenWorkspace:
    """Bind sources to an already exported base revision, without creating worktrees."""

    def __init__(self, repository: Path) -> None:
        self.repository = repository.resolve()

    async def tracked_diff_digest(self, **_kwargs: Any) -> str:
        digest = hashlib.sha256()
        for path in sorted(self.repository.rglob("*")):
            if ".git" in path.relative_to(self.repository).parts:
                continue
            if path.is_file() or path.is_symlink():
                digest.update(path.relative_to(self.repository).as_posix().encode() + b"\0")
                data = os.readlink(path).encode() if path.is_symlink() else path.read_bytes()
                digest.update(data + b"\0")
        return digest.hexdigest()

    async def bind_repository_sources(
        self, *, base_commit: str, sources: list[dict[str, Any]], **_kwargs: Any
    ) -> tuple[list[dict[str, Any]], list[str]]:
        bound, errors = [], []
        for source in sources:
            relative = Path(source["path"])
            path = self.repository / relative
            label = source.get("id", "source")
            if relative.is_absolute() or ".." in relative.parts:
                errors.append(f"{label} repository source must be relative: {relative}")
                continue
            if (
                path.is_symlink()
                or not path.is_file()
                or not path.resolve().is_relative_to(self.repository)
            ):
                errors.append(f"{label} repository source is not a frozen regular file: {relative}")
                continue
            data = path.read_bytes()
            lines = data.decode("utf-8", errors="replace").splitlines()
            start, end = source["line_start"], source["line_end"]
            if start < 1 or end < start or end > len(lines):
                errors.append(f"{label} repository source line range is invalid: {relative}")
                continue
            bound.append(
                {
                    **source,
                    "quote": "\n".join(lines[start - 1 : end]),
                    "base_revision": base_commit,
                    "file_sha256": sha256_bytes(data),
                }
            )
        return bound, errors


def output_validity(raw: str) -> dict[str, Any]:
    result: dict[str, Any] = {"raw_json_schema_valid": False, "extracted_schema_valid": False}
    try:
        validate_spec_output(json.loads(raw))
        result["raw_json_schema_valid"] = True
    except (ValueError, TypeError):
        pass
    try:
        normalized, method = extract_single_json_object(raw)
        result["extraction_method"] = method
        validate_spec_output(json.loads(normalized))
        result["extracted_schema_valid"] = True
    except (ValueError, TypeError) as error:
        result["validation_error"] = str(error)
    return result


def transcript_metrics(transcript: str) -> dict[str, Any]:
    messages, executions = [], []
    for line in transcript.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if (
            event.get("type") == "message_end"
            and event.get("message", {}).get("role") == "assistant"
        ):
            messages.append(event["message"])
        if event.get("type") == "tool_execution_end":
            executions.append(event)
    return {
        "model_calls": len(messages),
        "tool_calls": len(executions),
        "tool_errors": sum(bool(e.get("isError")) for e in executions),
        "input_tokens": sum(m.get("usage", {}).get("input", 0) for m in messages),
        "output_tokens": sum(m.get("usage", {}).get("output", 0) for m in messages),
        "cached_tokens": sum(m.get("usage", {}).get("cacheRead", 0) for m in messages),
        "peak_input_tokens": max((m.get("usage", {}).get("input", 0) for m in messages), default=0),
        "stop_reason": messages[-1].get("stopReason") if messages else None,
    }


class DockerERoleRunner:
    """Adapt the V6 A/corrector interface to the isolated E Pi runtime."""

    def __init__(
        self,
        *,
        repository: Path,
        store: ArtifactStore,
        routing: PiStageModelConfig,
        spec_input: dict[str, Any],
        system_prompt: str,
        image: str,
        max_tool_calls: int,
        loop_detection: bool,
        loop_repeat_limit: int,
        loop_max_cycle_length: int,
        loop_action_repeat_limit: int,
        context_window: int,
        budget_soft_fraction: float,
        budget_hard_fraction: float,
        file_segment_limit: int,
        gate_block_limit: int,
        unknown_tool_turn_limit: int,
        sampling: dict[str, Any],
    ) -> None:
        self.repository, self.store, self.routing = repository, store, routing
        self.spec_input, self.system_prompt, self.image = spec_input, system_prompt, image
        self.max_tool_calls = max_tool_calls
        self.loop_detection = loop_detection
        self.loop_repeat_limit = loop_repeat_limit
        self.loop_max_cycle_length = loop_max_cycle_length
        self.loop_action_repeat_limit = loop_action_repeat_limit
        self.context_window = context_window
        self.budget_soft_fraction = budget_soft_fraction
        self.budget_hard_fraction = budget_hard_fraction
        self.file_segment_limit = file_segment_limit
        self.gate_block_limit = gate_block_limit
        self.unknown_tool_turn_limit = unknown_tool_turn_limit
        self.sampling = sampling
        self.invocations: list[dict[str, Any]] = []

    async def run_role(
        self,
        *,
        role: RoleName,
        prompt: str,
        tools: tuple[str, ...],
        timeout_sec: float,
        artifact_prefix: str,
        **_kwargs: Any,
    ) -> RoleRunResult:
        if role not in {RoleName.V5_SPECIFICATION_REVIEWER, RoleName.JSON_CORRECTOR}:
            raise ValueError(f"E inference cannot start downstream role: {role}")
        spec = self.routing.for_role(role)
        directory = self.store.root / artifact_prefix
        directory.mkdir(parents=True, exist_ok=False)
        config_dir = directory / "pi-config"
        config_dir.mkdir()
        (directory / "sessions").mkdir()
        (directory / "inference-config.json").write_text(json.dumps(self.sampling))
        models = self.routing.models_json(spec)
        (config_dir / "models.json").write_text(json.dumps(models, ensure_ascii=False))
        (config_dir / "settings.json").write_text(
            json.dumps(
                {
                    "defaultProjectTrust": "never",
                    "telemetry": False,
                    "compaction": {"enabled": False},
                    "httpIdleTimeoutMs": 0,
                }
            )
        )
        (config_dir / "auth.json").write_text("{}\n")
        (config_dir / "experiment_options.ts").write_bytes(
            (HERE / "e_smoke/experiment_options.ts").read_bytes()
        )
        (config_dir / "loop_detector.mjs").write_bytes(
            (HERE / "e_smoke/loop_detector.mjs").read_bytes()
        )
        if role == RoleName.V5_SPECIFICATION_REVIEWER:
            user_prompt = "SPEC INPUT\n" + json.dumps(self.spec_input, ensure_ascii=False, indent=2)
            system_prompt = self.system_prompt
        else:
            user_prompt = prompt
            system_prompt = (
                "You repair JSON presentation and structure only. Follow CORRECTION INPUT."
            )
        (directory / "prompt.txt").write_text(user_prompt, encoding="utf-8")
        (directory / "system-prompt.txt").write_text(system_prompt, encoding="utf-8")
        name = "e-qwen-" + hashlib.sha256(str(directory).encode()).hexdigest()[:16]
        command = [
            "docker",
            "run",
            "--rm",
            "-i",
            "--name",
            name,
            "--entrypoint",
            "/usr/bin/timeout",
            "--workdir",
            BASE_ROOT,
            "--mount",
            f"type=bind,source={self.repository},target={BASE_ROOT},readonly",
            "--mount",
            f"type=bind,source={directory},target=/work/output",
            "--env",
            "PI_CODING_AGENT_DIR=/work/output/pi-config",
            "--env",
            "E_ARTIFACT_DIR=/work/output",
            "--env",
            f"E_MAX_TOOL_CALLS={self.max_tool_calls}",
            "--env",
            f"E_LOOP_DETECTION={'1' if self.loop_detection else '0'}",
            "--env",
            f"E_LOOP_REPEAT_LIMIT={self.loop_repeat_limit}",
            "--env",
            f"E_LOOP_MAX_CYCLE_LENGTH={self.loop_max_cycle_length}",
            "--env",
            f"E_LOOP_ACTION_REPEAT_LIMIT={self.loop_action_repeat_limit}",
            "--env",
            f"E_CONTEXT_WINDOW={self.context_window}",
            "--env",
            f"E_BUDGET_SOFT_FRACTION={self.budget_soft_fraction}",
            "--env",
            f"E_BUDGET_HARD_FRACTION={self.budget_hard_fraction}",
            "--env",
            f"E_FILE_SEGMENT_LIMIT={self.file_segment_limit}",
            "--env",
            f"E_GATE_BLOCK_LIMIT={self.gate_block_limit}",
            "--env",
            f"E_UNKNOWN_TOOL_TURN_LIMIT={self.unknown_tool_turn_limit}",
            "--env",
            f"{spec.api_key_env}={os.environ.get(spec.api_key_env, 'EMPTY')}",
            "--env",
            "PI_OFFLINE=1",
            "--env",
            "PI_TELEMETRY=0",
            self.image,
            str(max(1, int(timeout_sec))),
            "/opt/pi-runtime/bin/node",
            "/opt/pi-runtime/node_modules/@earendil-works/pi-coding-agent/dist/cli.js",
            "--system-prompt",
            system_prompt,
            "--mode",
            "json",
            "--provider",
            spec.provider,
            "--model",
            spec.model,
            "--thinking",
            "high" if self.sampling.get("chat_template_kwargs", {}).get("enable_thinking") else "off",
            "--extension",
            "/work/output/pi-config/experiment_options.ts",
            "--no-approve",
            "--no-extensions",
            "--no-skills",
            "--no-prompt-templates",
            "--no-context-files",
            "--offline",
            "--session-dir",
            "/work/output/sessions",
        ]
        command.extend(["--tools", ",".join(tools)] if tools else ["--no-tools"])
        command.append("-p")
        started = datetime.now(timezone.utc)
        self.store.event(
            "E_ROLE_STARTED", role=role.value, container=name, artifact_prefix=artifact_prefix
        )
        transcript_path, stderr_path = directory / "pi.jsonl", directory / "pi.stderr.txt"
        with transcript_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=stdout,
                stderr=stderr,
            )
            try:
                await process.communicate(user_prompt.encode())
            except asyncio.CancelledError:
                cleanup = await asyncio.create_subprocess_exec(
                    "docker",
                    "rm",
                    "-f",
                    name,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await cleanup.wait()
                await process.wait()
                raise
        finished = datetime.now(timezone.utc)
        transcript = transcript_path.read_text(encoding="utf-8")
        (directory / "exit-code.txt").write_text(f"{process.returncode}\n")
        metrics = transcript_metrics(transcript)
        exception = None
        try:
            raw = extract_pi_final_assistant_text(transcript)
        except ValueError as error:
            raw, exception = "", str(error)
        (directory / "output.txt").write_text(raw, encoding="utf-8")
        trajectory_path = directory / "trajectory.json"
        trajectory_path.write_text(
            json.dumps(
                pi_jsonl_to_atif(
                    transcript,
                    instruction=user_prompt,
                    model=spec.model,
                    agent_version="0.84.2",
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        budget_exhausted = (directory / "budget-exhausted.json").is_file()
        loop_path = directory / "loop-detected.json"
        loop_detected = loop_path.is_file()
        loop_report = read_json(loop_path) if loop_detected else None
        successful = (
            process.returncode == 0
            and metrics["stop_reason"] == "stop"
            and not budget_exhausted
            and not loop_detected
            and exception is None
            and bool(raw.strip())
        )
        if not successful and exception is None:
            if loop_detected:
                exception = (
                    f"Loop detected: {loop_report.get('type')} "
                    f"after {loop_report.get('threshold')} repetitions"
                )
            else:
                exception = f"Pi exit={process.returncode}, stop={metrics['stop_reason']}, budget_exhausted={budget_exhausted}"
        invocation = {
            "role": role.value,
            "model": spec.to_metadata(),
            "artifact_prefix": artifact_prefix,
            "exit_code": process.returncode,
            "started_at": started.isoformat(),
            "finished_at": finished.isoformat(),
            "elapsed_sec": (finished - started).total_seconds(),
            "budget_exhausted": budget_exhausted,
            "loop_detected": loop_detected,
            "loop_report": loop_report,
            "termination_reason": (
                "loop_detected" if loop_detected
                else "tool_budget_exhausted" if budget_exhausted
                else "normal"
            ),
            "exception": exception,
            **metrics,
            **output_validity(raw),
        }
        self.invocations.append(invocation)
        self.store.write_json(f"{artifact_prefix}/result.json", invocation)
        self.store.event("E_ROLE_FINISHED", **invocation)
        return RoleRunResult(
            exit_code=0 if successful else process.returncode or 1,
            final_assistant_text=raw,
            transcript_path=transcript_path,
            trajectory_path=trajectory_path,
            started_at=started.isoformat(),
            finished_at=finished.isoformat(),
            resource_usage=metrics,
            timed_out=process.returncode == 124,
            exception=exception,
        )


async def prepare_e_ledger(
    *,
    task: TaskInput,
    runner: Any,
    workspace: FrozenWorkspace,
    store: ArtifactStore,
    config: V6Config,
) -> dict[str, Any]:
    """One A run and its existing format correction, followed by an A-stage export."""
    before = await workspace.tracked_diff_digest()
    controller = V6Controller(
        role_runner=runner, workspace=workspace, artifacts=store, config=config
    )
    try:
        ledger = await controller.prepare_ledger(task, base_worktree=BASE_ROOT)
        if ledger is not None:
            write_stage(
                store.root,
                stage="A",
                task_id=task.task_id,
                base_commit=task.base_commit,
                baseline_sha256=sha256_bytes(b""),
                ledger_sha256=sha256_json(ledger),
                quality=controller.ledger_quality,
            )
        after = await workspace.tracked_diff_digest()
        result = {
            "schema_version": "e-a-inference-1",
            "task_id": task.task_id,
            "base_commit": task.base_commit,
            "baseline_patch_sha256": sha256_bytes(b""),
            "baseline_files_sha256_before": before,
            "baseline_files_sha256_after": after,
            "baseline_unchanged": before == after,
            "final_state": "A_COMPLETED" if ledger is not None and before == after else "A_FAILED",
            "ledger_quality": controller.ledger_quality,
            "ledger_warnings": controller.ledger_warnings,
            "error": controller.ledger_error,
            "downstream_started": False,
            "counts": {
                key: len(ledger[key])
                for key in (
                    "obligations",
                    "preservation_obligations",
                    "open_questions",
                )
            }
            if ledger is not None
            else {},
            "invocations": runner.invocations,
            "correction_calls": sum(
                r["role"] == RoleName.JSON_CORRECTOR.value for r in runner.invocations
            ),
        }
        store.write_json("manifest.json", result)
        return result
    finally:
        controller.close()


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    repository = args.repository.resolve()
    if not repository.is_dir():
        raise ValueError("Provide a frozen public repository export")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    routing = PiStageModelConfig.from_value(args.stage_model_config)
    a, corrector = (
        routing.for_role(role)
        for role in (
            RoleName.V5_SPECIFICATION_REVIEWER,
            RoleName.JSON_CORRECTOR,
        )
    )
    if a != corrector:
        raise ValueError("E requires the A model itself for JSON correction")
    config_values = read_json(args.controller_config)
    config_values["role_timeout_sec"] = args.timeout_sec
    config = V6Config(**config_values)
    config.validate()
    if config.max_format_retries != 1 or config.max_semantic_retries != 0:
        raise ValueError("E uses at most one format correction and no complete A retries")
    original = read_json(args.task_input)
    # Whitelist only task inputs; never forward labels or teacher evidence.
    spec_input = {
        "task": original["task"],
        "base_root": BASE_ROOT,
        "patch_base": "base-commit",
        "baseline_patch_sha256": None,
        "managed_delivery": original.get("managed_delivery", []),
        "limits": {
            "max_tool_calls": args.max_tool_calls or None,
            "timeout_sec": args.timeout_sec,
        },
    }
    task = TaskInput(
        task_id=args.task_id,
        issue_title=spec_input["task"]["title"],
        issue_body=spec_input["task"]["description"],
        repository_name=args.task_id.split("-", 1)[0],
        base_commit=args.base_commit,
        managed_delivery=tuple(spec_input["managed_delivery"]),
    )
    store = ArtifactStore(
        output, secrets=tuple(os.environ.get(k, "") for k in routing.required_env_keys())
    )
    system_prompt = args.system_prompt.read_text(encoding="utf-8")
    sampling = read_json(args.inference_config)
    runner = DockerERoleRunner(
        repository=repository,
        store=store,
        routing=routing,
        spec_input=spec_input,
        system_prompt=system_prompt,
        image=args.image,
        max_tool_calls=args.max_tool_calls,
        loop_detection=args.loop_detection,
        loop_repeat_limit=args.loop_repeat_limit,
        loop_max_cycle_length=args.loop_max_cycle_length,
        loop_action_repeat_limit=args.loop_action_repeat_limit,
        context_window=args.context_window,
        budget_soft_fraction=args.budget_soft_fraction,
        budget_hard_fraction=args.budget_hard_fraction,
        file_segment_limit=args.file_segment_limit,
        gate_block_limit=args.gate_block_limit,
        unknown_tool_turn_limit=args.unknown_tool_turn_limit,
        sampling=sampling,
    )
    store.write_json("inputs/spec-input.json", spec_input)
    store.write_text("inputs/system-prompt.txt", system_prompt)
    store.write_json("inputs/stage-model-config.json", routing.to_metadata())
    store.write_json("inputs/controller-config.json", config_values)
    store.write_json("inputs/inference-config.json", sampling)
    store.write_json(
        "inputs/experiment.json",
        {
            "group": "E0",
            "task_id": args.task_id,
            "base_commit": args.base_commit,
            "system_prompt_sha256": sha256_bytes(system_prompt.encode()),
            "runner_sha256": sha256_bytes(Path(__file__).read_bytes()),
            "image": args.image,
            "sampling": sampling,
            "thinking": sampling.get("chat_template_kwargs", {}).get("enable_thinking", False),
            "no_compaction": True,
            "http_idle_timeout_ms": 0,
            "max_tool_calls": args.max_tool_calls,
            "loop_detection": args.loop_detection,
            "loop_repeat_limit": args.loop_repeat_limit,
            "loop_max_cycle_length": args.loop_max_cycle_length,
            "loop_action_repeat_limit": args.loop_action_repeat_limit,
            "budget_control": {
                "context_window": args.context_window,
                "soft_fraction": args.budget_soft_fraction,
                "hard_fraction": args.budget_hard_fraction,
                "file_segment_limit": args.file_segment_limit,
                "gate_block_limit": args.gate_block_limit,
                "unknown_tool_turn_limit": args.unknown_tool_turn_limit,
            },
            "timeout_sec": args.timeout_sec,
            "teacher_assistance": False,
            "preselected_evidence": False,
            "shared_controller": "V6Controller.prepare_ledger",
            "pi_version": "0.84.2",
        },
    )
    return await prepare_e_ledger(
        task=task, runner=runner, workspace=FrozenWorkspace(repository), store=store, config=config
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--task-input", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--system-prompt", type=Path, default=HERE / "e_prompt_v1.txt")
    parser.add_argument(
        "--stage-model-config", type=Path, default=HERE / "e_smoke/stage_model_config.json"
    )
    parser.add_argument(
        "--controller-config", type=Path, default=HERE / "e_smoke/controller_config.json"
    )
    parser.add_argument(
        "--inference-config", type=Path, default=HERE / "e_smoke/inference_config.json"
    )
    parser.add_argument("--image", default="external-agents-eval/pi-e-smoke:v0.84.2-arm64")
    parser.add_argument("--timeout-sec", type=int, default=3600)
    parser.add_argument("--max-tool-calls", type=int, default=0, help="0 means unlimited")
    parser.add_argument(
        "--loop-repeat-limit", type=int, default=8,
        help="Consecutive or periodic no-progress repetitions before stopping",
    )
    parser.add_argument(
        "--loop-max-cycle-length", type=int, default=8,
        help="Maximum action-cycle length checked by the loop detector",
    )
    parser.add_argument(
        "--loop-action-repeat-limit", type=int, default=0,
        help="Stop after this many identical calls even when results differ; 0 disables",
    )
    parser.add_argument(
        "--context-window", type=int, default=0,
        help="Context window used by the budget gates; 0 disables them",
    )
    parser.add_argument(
        "--budget-soft-fraction", type=float, default=0.0,
        help="Append a one-off wrap-up reminder at this fraction of the window; 0 disables",
    )
    parser.add_argument(
        "--budget-hard-fraction", type=float, default=0.0,
        help="Block further tool calls at this fraction so the role must deliver; 0 disables",
    )
    parser.add_argument(
        "--file-segment-limit", type=int, default=0,
        help="Block reading one file in more than this many ranges; 0 disables",
    )
    parser.add_argument(
        "--gate-block-limit", type=int, default=6,
        help="Terminate after this many gate-blocked attempts with no change",
    )
    parser.add_argument(
        "--unknown-tool-turn-limit", type=int, default=0,
        help="Abort after this many consecutive turns that only call unavailable tools; 0 disables",
    )
    parser.add_argument(
        "--no-loop-detection", dest="loop_detection", action="store_false",
        help="Disable online repeated-action detection",
    )
    parser.set_defaults(loop_detection=True)
    args = parser.parse_args()
    if (
        args.timeout_sec < 1
        or args.max_tool_calls < 0
        or args.loop_repeat_limit < 2
        or args.loop_max_cycle_length < 1
        or args.loop_action_repeat_limit < 0
        or args.context_window < 0
        or args.file_segment_limit < 0
        or args.gate_block_limit < 0
        or args.unknown_tool_turn_limit < 0
        or not 0.0 <= args.budget_soft_fraction < 1.0
        or not 0.0 <= args.budget_hard_fraction < 1.0
        or (args.budget_soft_fraction and args.budget_hard_fraction
            and args.budget_soft_fraction >= args.budget_hard_fraction)
        or ((args.budget_soft_fraction or args.budget_hard_fraction) and not args.context_window)
    ):
        parser.error("timeout/budget/loop limits are invalid")
    result = asyncio.run(main_async(args))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["final_state"] == "A_COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
