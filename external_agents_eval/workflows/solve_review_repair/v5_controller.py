"""First executable V5 review pipeline.

This controller is intentionally independent of the legacy V1--V4 contract
controller.  It provides a small vertical slice of the V5 protocol: A creates
the ledger, I/R submit durable Markdown-backed batches through ``V5StateStore``,
S adjudicates the frozen reports, and one unconstrained Repair may run.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import ArtifactStore, sha256_bytes, sha256_json
from .models import IndeterminateInvocationError, RoleName, RoleRunResult, TaskInput
from .v5_config import V5Config
from .v5_contracts import (
    V5ContractError,
    parse_json_object,
    validate_managed_delivery,
    validate_output_schema,
    validate_review_batch_shape,
    validate_spec_output,
    validate_summary_output,
)
from .v5_prompts import (
    REVIEW_BATCH_SCHEMA,
    SPEC_OUTPUT_SCHEMA,
    SUMMARY_OUTPUT_SCHEMA,
    correction_prompt,
    repair_prompt,
    reviewer_prompt,
    spec_prompt,
    summary_prompt,
)
from .v5_state import ImmutableBundleStore, V5StateStore


@dataclass(frozen=True)
class V5Result:
    run_id: str
    route: str
    p0_sha256: str
    p1_patch: bytes
    ledger: dict[str, Any]
    reports: dict[str, dict[str, Any]]
    summary: dict[str, Any]
    freeze_id: str
    error: str | None = None


class V5SnapshotMutationError(V5ContractError):
    """The next read-only V5 stage cannot be given a clean snapshot."""


class V5Controller:
    """Run V5 with injected role and workspace adapters."""

    def __init__(self, *, role_runner: Any, workspace: Any, artifacts: ArtifactStore, config: V5Config | None = None) -> None:
        self.role_runner = role_runner
        self.workspace = workspace
        self.artifacts = artifacts
        self.config = config or V5Config()
        self.config.validate()
        self.bundle_store = ImmutableBundleStore(
            self.artifacts.root / "v5", secrets=self.artifacts.secrets
        )
        self.state_store = V5StateStore(self.artifacts.root / "v5" / "workflow.sqlite")
        self._source_bindings: list[dict[str, Any]] = []
        self._active_run_id: str | None = None

    def close(self) -> None:
        self.state_store.close()

    async def run(
        self,
        task: TaskInput,
        *,
        candidate_patch: bytes,
        baseline_patch: bytes = b"",
        base_worktree: str,
        candidate_worktree: str,
        summary_worktree: str | None = None,
        regression_worktree: str | None = None,
        repair_worktree: str | None = None,
        run_id: str | None = None,
    ) -> V5Result:
        run_id = run_id or uuid.uuid4().hex
        self._active_run_id = run_id
        p0_sha = sha256_bytes(candidate_patch)
        task_json = {"title": task.issue_title, "description": task.issue_body}
        managed_delivery = validate_managed_delivery(
            list(task.managed_delivery), task=task_json
        )
        binding = {
            "task_id": task.task_id,
            "task_sha256": sha256_json(task_json),
            "repository_name": task.repository_name,
            "base_commit": task.base_commit,
            "patch_base": "environment-baseline" if baseline_patch else "base-commit",
            "baseline_patch_sha256": sha256_bytes(baseline_patch),
            "candidate_patch_sha256": p0_sha,
            "controller_version": self.config.controller_version,
            "prompt_version": self.config.prompt_version,
            "prompt_builder_sha256": sha256_bytes(
                Path(__file__).with_name("v5_prompts.py").read_bytes()
            ),
            "contract_validator_sha256": sha256_bytes(
                Path(__file__).with_name("v5_contracts.py").read_bytes()
            ),
            "output_schemas_sha256": sha256_json(
                {
                    "spec": SPEC_OUTPUT_SCHEMA,
                    "review": REVIEW_BATCH_SCHEMA,
                    "summary": SUMMARY_OUTPUT_SCHEMA,
                }
            ),
            "review_note_skill_sha256": self._review_skill_digest(),
            "model": self.config.model,
            "managed_delivery_sha256": sha256_json(managed_delivery),
            "limits": {
                "batch_size": self.config.batch_size,
                "max_rounds": self.config.max_rounds,
                "max_format_retries": self.config.max_format_retries,
                "max_semantic_retries": self.config.max_semantic_retries,
                "max_no_progress": self.config.max_no_progress,
                "lane_deadline_sec": self.config.lane_deadline_sec,
                "role_timeout_sec": self.config.role_timeout_sec,
                "no_compaction": self.config.no_compaction,
            },
        }
        created = self.state_store.create_run(run_id, binding)
        self.artifacts.write_bytes("outputs/v5-p0.patch", candidate_patch)
        if not created:
            cached = self._completed_result(run_id, p0_sha)
            if cached is not None:
                return cached
            if self.state_store.run_record(run_id)["phase"] == "completed":
                raise V5ContractError(
                    "completed V5 run is missing or has corrupt delivery artifacts"
                )
        diff_path = await self._materialize_candidate_patch(candidate_patch, p0_sha)
        skill_path = await self._materialize_review_skill()
        regression_worktree = regression_worktree or candidate_worktree
        summary_worktree = summary_worktree or candidate_worktree
        repository = {
            "name": task.repository_name,
            "base_commit": task.base_commit,
            "base_root": base_worktree,
            "diff_path": diff_path,
            "patch_base": "environment-baseline" if baseline_patch else "base-commit",
            "baseline_patch_sha256": sha256_bytes(baseline_patch),
        }
        try:
            ledger_path = self.artifacts.root / "v5" / "ledger.json"
            if ledger_path.is_file():
                ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
                self._validate_persisted_ledger(ledger)
                bindings_path = self.artifacts.root / "v5" / "source-bindings.json"
                if not bindings_path.is_file():
                    raise V5ContractError("persisted V5 ledger is missing source bindings")
                persisted_bindings = json.loads(
                    bindings_path.read_text(encoding="utf-8")
                )
                recomputed_bindings = await self._bind_ledger_sources(
                    ledger,
                    task=task_json,
                    worktree=base_worktree,
                    base_commit=task.base_commit,
                )
                if persisted_bindings != recomputed_bindings:
                    raise V5ContractError(
                        "persisted V5 ledger source bindings failed integrity check"
                    )
                self._source_bindings = persisted_bindings
            else:
                ledger = await self._build_ledger(
                    run_id,
                    task_json,
                    repository,
                    base_worktree,
                    deadline_at=self.state_store.stage_deadline(
                        run_id, "A", duration_sec=self.config.role_timeout_sec
                    ),
                    managed_delivery=managed_delivery,
                    baseline_patch=baseline_patch,
                )
            self._persist_role_artifact(
                "v5/exports/ledger.json",
                task_id=task.task_id,
                role="A",
                base_commit=task.base_commit,
                candidate_sha256=None,
                ledger_sha256=None,
                content=ledger,
            )
            if not ledger_path.is_file():
                # Publish the content-bound export first. If the process dies
                # between these writes, resume regenerates/replays A and checks
                # it against the immutable envelope before recreating this
                # convenience projection.
                self.artifacts.write_json("v5/ledger.json", ledger)
            self.state_store.initialize_lanes(
                run_id,
                implementation_targets=[item["id"] for item in ledger["obligations"]],
                regression_targets=[item["id"] for item in ledger["preservation_obligations"]],
                optional_regression_targets=[item["id"] for item in ledger["obligations"]],
                target_catalog={
                    item["id"]: item["content"]
                    for item in [
                        *ledger["obligations"],
                        *ledger["preservation_obligations"],
                    ]
                },
                batch_size=self.config.batch_size,
                lane_deadline_sec=self.config.lane_deadline_sec,
            )
            implementation_report, regression_report = await asyncio.gather(
                self._run_lane(
                    run_id=run_id,
                    role="I",
                    task=task_json,
                    repository=repository,
                    ledger=ledger,
                    worktree=candidate_worktree,
                    skill_path=skill_path,
                    managed_delivery=managed_delivery,
                    base_commit=task.base_commit,
                    baseline_patch=baseline_patch,
                ),
                self._run_lane(
                    run_id=run_id,
                    role="R",
                    task=task_json,
                    repository=repository,
                    ledger=ledger,
                    worktree=regression_worktree,
                    skill_path=skill_path,
                    managed_delivery=managed_delivery,
                    base_commit=task.base_commit,
                    baseline_patch=baseline_patch,
                ),
            )
            reports: dict[str, dict[str, Any]] = {
                "I": implementation_report,
                "R": regression_report,
            }
            # A caller that does not provide a dedicated S worktree must still
            # prove that any incidental reviewer edits were restored.  This is
            # an isolation guard for the dependent S stage, not a rejection of
            # the I/R batch itself.  The production adapter always supplies a
            # separate summary worktree.
            if summary_worktree == candidate_worktree:
                unrestored = [
                    event
                    for report in reports.values()
                    for event in report.get("review_integrity", {}).get("events", [])
                    if event.get("status") != "restored"
                ]
                if unrestored:
                    raise V5SnapshotMutationError(
                        "reviewer modified its frozen worktree and it could not be "
                        "restored before summary"
                    )
            for role, report in reports.items():
                self._persist_role_artifact(
                    f"v5/exports/review-{role}.json",
                    task_id=task.task_id,
                    role=role,
                    base_commit=task.base_commit,
                    candidate_sha256=p0_sha,
                    ledger_sha256=sha256_json(ledger),
                    content=report,
                )
            archived_review_artifacts = [
                artifact
                for role in ("I", "R")
                for artifact in self._review_artifacts(run_id, role)
            ]
            snapshot = {
                "run_id": run_id,
                "ledger_sha256": sha256_json(ledger),
                "source_bindings": self._source_bindings,
                "reports": reports,
                "review_artifacts": archived_review_artifacts,
                "lane_revisions": {role: self.state_store.lane(run_id, role)["revision"] for role in ("I", "R")},
            }
            freeze_id = self.state_store.freeze_summary(run_id, snapshot)
            artifact_root = str(self.bundle_store.bundles)
            decision_context = self._decision_context(ledger, reports)
            allowed_sources = self._allowed_summary_sources(
                ledger, [reports["I"], reports["R"]]
            )
            summary_input = {
                "task": task_json,
                "managed_delivery": managed_delivery,
                "ledger": ledger,
                "reviews": [reports["I"], reports["R"]],
                "review_notes": self._public_review_artifacts(
                    archived_review_artifacts, artifact_root=artifact_root
                ),
                "artifact_root": artifact_root,
                "required_decision_refs": self._required_decision_refs(ledger, reports),
                "allowed_supporting_refs": sorted(decision_context),
                "allowed_validation_record_refs": sorted(
                    reference
                    for reference, context in decision_context.items()
                    if context["kind"] == "assessment"
                ),
                "allowed_preserve_refs": sorted(
                    reference
                    for reference, context in decision_context.items()
                    if context["kind"] == "assessment"
                    and context.get("status") == "satisfied"
                ),
                "allowed_sources": allowed_sources,
            }
            model_summary_input = await self._materialize_summary_evidence(
                summary_input,
                archived_review_artifacts=archived_review_artifacts,
            )
            summary = await self._run_summary(
                model_summary_input,
                summary_worktree,
                decision_context=decision_context,
                freeze_id=freeze_id,
                base_commit=task.base_commit,
                baseline_patch=baseline_patch,
                deadline_at=self.state_store.stage_deadline(
                    run_id, "S", duration_sec=self.config.role_timeout_sec
                ),
            )
            self.artifacts.write_json("v5/summary.json", summary)
            self._persist_role_artifact(
                "v5/exports/summary.json",
                task_id=task.task_id,
                role="S",
                base_commit=task.base_commit,
                candidate_sha256=p0_sha,
                ledger_sha256=sha256_json(ledger),
                content=summary,
                extra_bindings={"summary_freeze_id": freeze_id},
            )
            route = self._route(ledger, reports, summary, decision_context)
            decision_artifact = {
                "route": route,
                "review_complete": all(
                    report["completion"] == "complete" for report in reports.values()
                ),
                "preserve_refs": self._preserve_refs(ledger, reports, summary),
                "plan": summary,
                "freeze_id": freeze_id,
            }
            self.artifacts.write_json("v5/decision.json", decision_artifact)
            repair_invocation_id = (
                f"{run_id}.repair.{freeze_id}" if route == "repair" else None
            )
            self.state_store.prepare_delivery(
                run_id,
                freeze_id=freeze_id,
                route=route,
                repair_invocation_id=repair_invocation_id,
            )
            p1 = candidate_patch
            repair_result: RoleRunResult | None = None
            if route == "repair" and repair_worktree is None:
                raise V5ContractError("V5 repair route requires a repair worktree")
            if route == "repair":
                assert repair_worktree is not None
                await self.workspace.apply_patch(worktree=repair_worktree, patch=candidate_patch)
                repair_result = await self._run_role(
                    role=RoleName.REPAIR,
                    prompt=repair_prompt(
                        task_json,
                        p0_sha,
                        summary,
                        review_context={
                            "freeze_id": freeze_id,
                            "review_notes": model_summary_input["review_notes"],
                            "artifact_root": model_summary_input["artifact_root"],
                            "preserve_refs": decision_artifact["preserve_refs"],
                            "reports": reports,
                        },
                    ),
                    worktree=repair_worktree,
                    prefix="v5/repair",
                    invocation_id=repair_invocation_id,
                )
                try:
                    capture_kwargs = {
                        "worktree": repair_worktree,
                        "base_commit": task.base_commit,
                    }
                    if baseline_patch:
                        capture_kwargs["baseline_patch"] = baseline_patch
                    p1 = await self.workspace.capture_patch(**capture_kwargs)
                    self.artifacts.event(
                        "V5_REPAIR_CAPTURED",
                        exit_code=repair_result.exit_code,
                        timed_out=repair_result.timed_out,
                        partial=(
                            repair_result.exit_code != 0 or repair_result.timed_out
                        ),
                        p1_sha256=sha256_bytes(p1),
                    )
                except Exception as error:  # noqa: BLE001 - patch capture must safely fall back
                    p1 = candidate_patch
                    self.artifacts.event("V5_REPAIR_FALLBACK_P0", error=str(error))
            self.artifacts.write_bytes("outputs/v5-p1.patch", p1)
            self.state_store.finish_delivery(
                run_id,
                p1_sha256=sha256_bytes(p1),
                result={
                    "route": route,
                    "repair_executed": repair_result is not None,
                    "repair_exit_code": (
                        repair_result.exit_code if repair_result is not None else None
                    ),
                    "repair_timed_out": (
                        repair_result.timed_out if repair_result is not None else False
                    ),
                    "partial_repair_delivery": bool(
                        repair_result is not None
                        and (repair_result.exit_code != 0 or repair_result.timed_out)
                    ),
                },
            )
            return V5Result(run_id, route, p0_sha, p1, ledger, reports, summary, freeze_id)
        except Exception as error:
            self.artifacts.event("V5_FAILED", error=str(error), run_id=run_id)
            raise

    async def _build_ledger(
        self,
        run_id: str,
        task: dict[str, Any],
        repository: dict[str, Any],
        worktree: str,
        deadline_at: str,
        managed_delivery: list[dict[str, Any]],
        baseline_patch: bytes = b"",
    ) -> dict[str, Any]:
        frozen_digest = await self._worktree_digest(
            worktree,
            repository["base_commit"],
            baseline_patch=baseline_patch,
        )
        base_prompt = spec_prompt(
            task,
            {
                "name": repository["name"],
                "base_commit": repository["base_commit"],
                "base_root": worktree,
            },
            managed_delivery=managed_delivery,
        )
        last_error: Exception | None = None
        ledger: dict[str, Any] | None = None
        source_bindings: list[dict[str, Any]] | None = None
        for semantic_attempt in range(1, self.config.max_semantic_retries + 2):
            remaining_sec = self._deadline_remaining(deadline_at)
            if remaining_sec <= 0:
                last_error = V5ContractError("V5 A deadline expired")
                break
            prompt = base_prompt
            if last_error is not None:
                prompt += (
                    "\nThe previous complete object was rejected. Recheck the original "
                    "sources and return a full replacement without weakening any requirement.\n"
                    f"VALIDATION ERROR\n{last_error}\n"
                )
            result = await self._run_role(
                role=RoleName.V5_SPECIFICATION_REVIEWER,
                prompt=prompt,
                worktree=worktree,
                prefix=f"v5/specification/attempt-{semantic_attempt}",
                session_role="A",
                invocation_id=f"{run_id}.A.{semantic_attempt}",
                timeout_sec=min(self.config.role_timeout_sec, remaining_sec),
            )
            await self._require_worktree_unchanged(
                worktree,
                repository["base_commit"],
                frozen_digest,
                role="V5 specification reviewer",
                baseline_patch=baseline_patch,
            )
            try:
                self._require_success(result, "V5 specification reviewer")
                payload = validate_spec_output(
                    await self._parse_with_format_corrections(
                        raw=result.final_assistant_text,
                        output_schema=SPEC_OUTPUT_SCHEMA,
                        worktree=worktree,
                        prefix=f"v5/specification/attempt-{semantic_attempt}",
                        deadline_at=deadline_at,
                    )
                )
                candidate_ledger = {
                    "summary": payload["summary"],
                    "obligations": self._assign_targets(
                        payload["obligations"], "O"
                    ),
                    "preservation_obligations": self._assign_targets(
                        payload["preservation_obligations"], "P"
                    ),
                    "open_questions": self._assign_questions(
                        payload["open_questions"]
                    ),
                }
                candidate_bindings = await self._bind_ledger_sources(
                    candidate_ledger,
                    task=task,
                    worktree=worktree,
                    base_commit=repository["base_commit"],
                )
                ledger = candidate_ledger
                source_bindings = candidate_bindings
                break
            except Exception as error:
                if isinstance(error, IndeterminateInvocationError):
                    raise
                last_error = error
        if ledger is None or source_bindings is None:
            assert last_error is not None
            raise last_error
        self._source_bindings = source_bindings
        self.artifacts.write_json("v5/source-bindings.json", self._source_bindings)
        return ledger

    async def _run_lane(
        self,
        *,
        run_id: str,
        role: str,
        task: dict[str, Any],
        repository: dict[str, Any],
        ledger: dict[str, Any],
        worktree: str,
        skill_path: str,
        managed_delivery: list[dict[str, Any]],
        base_commit: str,
        baseline_patch: bytes = b"",
    ) -> dict[str, Any]:
        # I/R are advisory readers over disposable candidate worktrees.  Capture
        # the state they started from so accidental edits can be diagnosed and
        # removed after the batch without turning them into protocol failures.
        initial_patch = await self._capture_worktree_patch(
            worktree=worktree,
            base_commit=base_commit,
            baseline_patch=baseline_patch,
        )
        initial_digest = await self._worktree_digest(
            worktree,
            base_commit,
            baseline_patch=baseline_patch,
        )
        integrity_events: list[dict[str, Any]] = []
        initial_state = self.state_store.lane(run_id, role)
        attempt = self.state_store.next_attempt(run_id, role, initial_state["round"])
        retry_errors: list[dict[str, str]] = []
        while self.state_store.lane(run_id, role)["phase"] == "running":
            state = self.state_store.lane(run_id, role)
            remaining_sec = self._deadline_remaining(state["deadline_at"])
            if state["round"] > self.config.max_rounds or remaining_sec <= 0:
                self.state_store.stop_lane(run_id, role, reason="deadline_or_round_limit")
                break
            artifact_key = (
                f"{run_id}/{role}/round-{state['round']:04d}/attempt-{attempt}"
            )
            host_output_root = (
                self.artifacts.root
                / "v5"
                / "attempts"
                / role
                / f"round-{state['round']:04d}"
                / f"attempt-{attempt}"
            )
            host_output_existed = host_output_root.is_dir()
            host_output_root.mkdir(parents=True, exist_ok=True)
            prepared = self.state_store.attempt_record(
                run_id, role, state["round"], attempt
            )
            if prepared is not None:
                if prepared["status"] != "PREPARED" or not isinstance(
                    prepared["input_payload"], dict
                ):
                    raise V5ContractError(
                        "recoverable V5 attempt lacks its prepared input payload"
                    )
                input_payload = prepared["input_payload"]
                try:
                    prompt = input_payload["prompt"]
                    output_root = input_payload["output_root"]
                    catalog_path = input_payload["catalog_path"]
                    round_state = input_payload["round_state"]
                    invocation_id = input_payload["invocation_id"]
                except KeyError as error:
                    raise V5ContractError(
                        "recoverable V5 attempt input is incomplete"
                    ) from error
                await self._resume_review_output_root(
                    output_root,
                    host_output_root,
                    host_output_existed=host_output_existed,
                )
            else:
                output_root = await self._prepare_review_output_root(
                    artifact_key, host_output_root
                )
                catalog_path = await self._materialize_target_catalog(
                    run_id=run_id,
                    role=role,
                    state=state,
                )
                round_state = self._round_state(
                    state,
                    ledger=ledger,
                    role=role,
                    output_root=output_root,
                    skill_path=skill_path,
                    catalog_path=catalog_path,
                    mode="retry" if attempt > 1 else "continue",
                    errors=retry_errors,
                    time_remaining_sec=remaining_sec,
                )
                prompt = reviewer_prompt(
                    task=task,
                    repository={
                        **repository,
                        "candidate_root": worktree,
                        "scratch_root": output_root,
                    },
                    ledger=ledger,
                    state=round_state,
                    role=role,
                    skill_path=skill_path,
                    output_root=output_root,
                    initial=state["round"] == 1 and attempt == 1,
                    managed_delivery=managed_delivery,
                    limits={
                        "batch_size": self.config.batch_size,
                        "max_rounds": self.config.max_rounds,
                        "max_format_retries": self.config.max_format_retries,
                        "max_semantic_retries": self.config.max_semantic_retries,
                        "max_no_progress": self.config.max_no_progress,
                        "time_budget_sec": int(self.config.lane_deadline_sec),
                    },
                )
                invocation_id = f"{run_id}.{role}.{state['round']}.{attempt}"
                input_payload = {
                    "role": role,
                    "round": state["round"],
                    "attempt": attempt,
                    "revision": state["revision"],
                    "ledger_sha256": sha256_json(ledger),
                    "round_state": round_state,
                    "review_batch_schema_sha256": sha256_json(
                        REVIEW_BATCH_SCHEMA
                    ),
                    "prompt": prompt,
                    "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
                    "skill_path": skill_path,
                    "catalog_path": catalog_path,
                    "output_root": output_root,
                    "invocation_id": invocation_id,
                    "deadline_at": state["deadline_at"],
                }
            self.state_store.prepare_attempt(
                run_id,
                role,
                state["round"],
                attempt,
                input_payload=input_payload,
            )
            preserve_output_root = False
            try:
                result = await self._run_role(
                    role=(
                        RoleName.V5_IMPLEMENTATION_REVIEWER
                        if role == "I"
                        else RoleName.V5_REGRESSION_REVIEWER
                    ),
                    prompt=prompt,
                    worktree=worktree,
                    prefix=(
                        f"v5/reviewer/{role}/round-{state['round']:04d}/"
                        f"attempt-{attempt}"
                    ),
                    session_role=role,
                    invocation_id=invocation_id,
                    timeout_sec=min(self.config.role_timeout_sec, remaining_sec),
                )
                self._require_success(result, f"V5 {role} reviewer")
                payload = validate_review_batch_shape(
                    await self._parse_with_format_corrections(
                        raw=result.final_assistant_text,
                        output_schema=REVIEW_BATCH_SCHEMA,
                        worktree=worktree,
                        prefix=(
                            f"v5/reviewer/{role}/round-{state['round']:04d}/"
                            f"attempt-{attempt}"
                        ),
                        deadline_at=state["deadline_at"],
                    )
                )
                await self._validate_review_batch_evidence(
                    payload,
                    task=task,
                    repository=repository,
                    worktree=worktree,
                    role=role,
                    lane_state=state,
                )
                notes = await self._collect_review_notes(
                    output_root=output_root,
                    host_output_root=host_output_root,
                    payload=payload,
                )
                extra_files: dict[str, bytes] = {}
                trace_files: list[dict[str, str]] = []
                if result.trajectory_path.is_file():
                    trajectory = result.trajectory_path.read_bytes()
                    extra_files["traces/trajectory.json"] = trajectory
                    trace_files.append(
                        {
                            "path": "traces/trajectory.json",
                            "sha256": sha256_bytes(trajectory),
                        }
                    )
                if result.session_file is not None and result.session_file.is_file():
                    session = result.session_file.read_bytes()
                    extra_files["sessions/session.jsonl"] = session
                    trace_files.append(
                        {
                            "path": "sessions/session.jsonl",
                            "sha256": sha256_bytes(session),
                        }
                    )
                material_files, material_entries = await self._collect_review_materials(
                    output_root=output_root,
                    host_output_root=host_output_root,
                )
                extra_files.update(material_files)
                trace = {
                    "role": role,
                    "invocation_id": result.invocation_id,
                    "files": trace_files,
                    "remote_session_file": result.remote_session_file,
                    "pre_session_sha256": result.pre_session_sha256,
                    "post_session_sha256": result.post_session_sha256,
                }
                materials = {
                    "role": role,
                    "files": material_entries,
                    "missing": [],
                    "empty_reason": (
                        None
                        if material_entries
                        else "reviewer created no probe or log material"
                    ),
                }
                batch_key = f"{run_id}:{role}:{state['round']}:{attempt}"
                bundle = self.bundle_store.stage(
                    batch_key=batch_key, normalized_json=payload,
                    raw_output=result.final_assistant_text, notes=notes,
                    trace_manifest=trace,
                    materials_manifest=materials,
                    extra_files=extra_files,
                )
                self.state_store.commit_batch(
                    run_id, role, state["round"], attempt, payload,
                    output_root=host_output_root,
                    bundle=bundle,
                    expected_revision=state["revision"],
                    observation_fingerprint=self._observation_fingerprint(result),
                )
            except Exception as error:
                if isinstance(error, IndeterminateInvocationError):
                    preserve_output_root = True
                    raise
                if isinstance(error, V5SnapshotMutationError):
                    raise
                self.state_store.reject_attempt(run_id, role, state["round"], attempt, str(error))
                retry_errors = [{"path": "$", "message": str(error)}]
                attempt += 1
                if attempt > self.config.max_semantic_retries + 1:
                    self.state_store.stop_lane(run_id, role, reason=str(error))
                    break
                continue
            finally:
                if not preserve_output_root:
                    integrity = await self._normalize_reviewer_worktree(
                        worktree=worktree,
                        base_commit=base_commit,
                        initial_patch=initial_patch,
                        initial_digest=initial_digest,
                        environment_patch=baseline_patch,
                        role=role,
                    )
                    if integrity is not None:
                        integrity_events.append(integrity)
                    await self._cleanup_review_output_root(output_root, host_output_root)
            attempt = 1
            retry_errors = []
            committed = self.state_store.lane(run_id, role)
            if (
                committed["phase"] == "running"
                and committed.get("no_progress", 0) >= self.config.max_no_progress
            ):
                self.state_store.stop_lane(run_id, role, reason="max_no_progress")
        return self._report(run_id, role, integrity_events=integrity_events)

    async def _run_summary(
        self,
        summary_input: dict[str, Any],
        worktree: str,
        *,
        decision_context: dict[str, dict[str, Any]],
        freeze_id: str,
        base_commit: str,
        deadline_at: str,
        baseline_patch: bytes = b"",
    ) -> dict[str, Any]:
        frozen_digest = await self._worktree_digest(
            worktree,
            base_commit,
            baseline_patch=baseline_patch,
        )
        base_prompt = summary_prompt(
            summary_input,
            freeze_id=freeze_id,
            source_bindings=self._source_bindings,
        )
        last_error: Exception | None = None
        for semantic_attempt in range(1, self.config.max_semantic_retries + 2):
            remaining_sec = self._deadline_remaining(deadline_at)
            if remaining_sec <= 0:
                last_error = V5ContractError("V5 S deadline expired")
                break
            prompt = base_prompt
            if last_error is not None:
                prompt += (
                    "\nThe previous summary object was rejected. Re-read the same frozen "
                    "evidence set and return a complete replacement. Do not change the "
                    "freeze scope or invent new observations.\n"
                    f"VALIDATION ERROR\n{last_error}\n"
                )
            result = await self._run_role(
                role=RoleName.V5_SUMMARY_REVIEWER,
                prompt=prompt,
                worktree=worktree,
                prefix=f"v5/summary/attempt-{semantic_attempt}",
                session_role="S",
                invocation_id=f"{freeze_id}.S.{semantic_attempt}",
                timeout_sec=min(self.config.role_timeout_sec, remaining_sec),
            )
            await self._require_worktree_unchanged(
                worktree,
                base_commit,
                frozen_digest,
                role="V5 summary reviewer",
                baseline_patch=baseline_patch,
            )
            try:
                self._require_success(result, "V5 summary reviewer")
                payload = await self._parse_with_format_corrections(
                    raw=result.final_assistant_text,
                    output_schema=SUMMARY_OUTPUT_SCHEMA,
                    worktree=worktree,
                    prefix=f"v5/summary/attempt-{semantic_attempt}",
                    deadline_at=deadline_at,
                )
                return validate_summary_output(
                    payload,
                    set(summary_input["required_decision_refs"]),
                    decision_context=decision_context,
                    allowed_sources=summary_input["allowed_sources"],
                )
            except Exception as error:
                if isinstance(error, IndeterminateInvocationError):
                    raise
                last_error = error
        assert last_error is not None
        raise last_error

    async def _run_role(
        self,
        *,
        role: RoleName,
        prompt: str,
        worktree: str,
        prefix: str,
        session_role: str | None = None,
        invocation_id: str | None = None,
        timeout_sec: float | None = None,
    ) -> RoleRunResult:
        if role == RoleName.JSON_CORRECTOR:
            tools: tuple[str, ...] = ()
        elif role in {
            RoleName.V5_SPECIFICATION_REVIEWER,
            RoleName.V5_SUMMARY_REVIEWER,
        }:
            tools = ("read", "grep", "find", "ls")
        else:
            tools = ("read", "bash", "grep", "find", "ls", "write", "edit")
        kwargs: dict[str, Any] = {
            "role": role, "prompt": prompt, "worktree": worktree, "tools": tools,
            "timeout_sec": (
                self.config.role_timeout_sec if timeout_sec is None else timeout_sec
            ),
            "artifact_prefix": prefix,
        }
        parameters = inspect.signature(self.role_runner.run_role).parameters
        if invocation_id is not None and "invocation_id" in parameters:
            kwargs["invocation_id"] = invocation_id
        if session_role and "session_id" in parameters:
            kwargs.update({
                "session_id": f"v5.{self.artifacts.root.name}.{session_role}",
                "session_dir": f"/external_eval/v5_sessions/{self.artifacts.root.name}/{session_role}",
                "no_compaction": self.config.no_compaction,
            })
        if invocation_id is not None and self._active_run_id is not None:
            self.state_store.prepare_invocation(
                invocation_id,
                run_id=self._active_run_id,
                role=role.value,
                prompt_sha256=sha256_bytes(prompt.encode("utf-8")),
                worktree=worktree,
            )
        # If transport raises, keep the invocation PREPARED: only the remote
        # completion record can prove whether that process ran.
        result = await self.role_runner.run_role(**kwargs)
        if invocation_id is not None and self._active_run_id is not None:
            status = (
                "COMPLETED"
                if result.exit_code == 0 and not result.timed_out
                else "FAILED"
            )
            self.state_store.finish_invocation(
                invocation_id,
                status=status,
                result={
                    "exit_code": result.exit_code,
                    "timed_out": result.timed_out,
                    "exception": result.exception,
                    "started_at": result.started_at,
                    "finished_at": result.finished_at,
                    "resource_usage": result.resource_usage,
                    "transcript_path": str(result.transcript_path),
                    "trajectory_path": str(result.trajectory_path),
                    "remote_session_file": result.remote_session_file,
                    "pre_session_sha256": result.pre_session_sha256,
                    "post_session_sha256": result.post_session_sha256,
                },
            )
        return result

    async def _parse_with_format_corrections(
        self,
        *,
        raw: str,
        output_schema: dict[str, Any],
        worktree: str,
        prefix: str,
        deadline_at: str | None = None,
    ) -> dict[str, Any]:
        current = raw
        last_error: V5ContractError | None = None
        for correction_index in range(self.config.max_format_retries + 1):
            try:
                return validate_output_schema(
                    parse_json_object(current), output_schema
                )
            except V5ContractError as error:
                last_error = error
            if correction_index >= self.config.max_format_retries:
                break
            correction_timeout = None
            if deadline_at is not None:
                remaining = self._deadline_remaining(deadline_at)
                if remaining <= 0:
                    raise V5ContractError(
                        "V5 lane deadline expired before format correction"
                    )
                correction_timeout = min(self.config.role_timeout_sec, remaining)
            correction = await self._run_role(
                role=RoleName.JSON_CORRECTOR,
                prompt=correction_prompt(
                    original_output=current,
                    output_schema=output_schema,
                    errors=[{"path": "$", "message": str(last_error)}],
                ),
                worktree=worktree,
                prefix=f"{prefix}/format-correction-{correction_index + 1}",
                invocation_id=uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{self.artifacts.root}:{prefix}:format:{correction_index + 1}",
                ).hex,
                timeout_sec=correction_timeout,
            )
            self._require_success(correction, "V5 JSON format corrector")
            current = correction.final_assistant_text
        assert last_error is not None
        raise last_error

    async def _materialize_candidate_patch(self, patch: bytes, digest: str) -> str:
        method = getattr(self.workspace, "materialize_file", None)
        if method is None:
            return f"sha256:{digest}"
        return await method(name=f"v5-p0-{digest[:16]}.patch", payload=patch, readonly=True)

    def _completed_result(self, run_id: str, p0_sha256: str) -> V5Result | None:
        if self.state_store.run_record(run_id)["phase"] != "completed":
            return None
        decision_path = self.artifacts.root / "v5" / "decision.json"
        ledger_path = self.artifacts.root / "v5" / "ledger.json"
        summary_path = self.artifacts.root / "v5" / "summary.json"
        p1_path = self.artifacts.root / "outputs" / "v5-p1.patch"
        if not all(path.is_file() for path in (decision_path, ledger_path, summary_path, p1_path)):
            return None
        delivery = self.state_store.delivery(run_id)
        if delivery is None or delivery["status"] != "COMPLETED":
            return None
        if sha256_bytes(p1_path.read_bytes()) != delivery["p1_sha256"]:
            return None
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        snapshot = self.state_store.frozen_snapshot(run_id)
        if snapshot is None:
            return None
        return V5Result(
            run_id=run_id,
            route=decision["route"],
            p0_sha256=p0_sha256,
            p1_patch=p1_path.read_bytes(),
            ledger=json.loads(ledger_path.read_text(encoding="utf-8")),
            reports=snapshot["reports"],
            summary=json.loads(summary_path.read_text(encoding="utf-8")),
            freeze_id=decision["freeze_id"],
        )

    def _persist_role_artifact(
        self,
        relative_path: str,
        *,
        task_id: str,
        role: str,
        base_commit: str,
        candidate_sha256: str | None,
        ledger_sha256: str | None,
        content: dict[str, Any],
        extra_bindings: dict[str, Any] | None = None,
    ) -> None:
        bindings = {
            "base_commit": base_commit,
            "candidate_sha256": candidate_sha256,
            "ledger_sha256": ledger_sha256,
            **(extra_bindings or {}),
        }
        content_sha256 = sha256_json(content)
        path = self.artifacts.root / relative_path
        if path.is_file():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if (
                existing.get("schema_version") != "5.0"
                or existing.get("task_id") != task_id
                or existing.get("role") != role
                or existing.get("bindings") != bindings
                or existing.get("content_sha256") != content_sha256
                or existing.get("content") != content
            ):
                raise V5ContractError(
                    f"persisted V5 role artifact conflicts with {relative_path}"
                )
            return
        self.artifacts.write_json(
            relative_path,
            {
                "schema_version": "5.0",
                "task_id": task_id,
                "role": role,
                "bindings": bindings,
                "content_sha256": content_sha256,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "content": content,
            },
        )

    async def _worktree_digest(
        self,
        worktree: str,
        base_commit: str,
        *,
        baseline_patch: bytes = b"",
    ) -> str | None:
        method = getattr(self.workspace, "tracked_diff_digest", None)
        if method is None:
            return None
        revision = base_commit or "HEAD"
        kwargs = {"worktree": worktree, "base_commit": revision}
        if baseline_patch:
            kwargs["baseline_patch"] = baseline_patch
        return await method(**kwargs)

    async def _capture_worktree_patch(
        self,
        *,
        worktree: str,
        base_commit: str,
        baseline_patch: bytes = b"",
    ) -> bytes | None:
        method = getattr(self.workspace, "capture_patch", None)
        if method is None:
            return None
        try:
            kwargs = {"worktree": worktree, "base_commit": base_commit}
            if baseline_patch:
                kwargs["baseline_patch"] = baseline_patch
            return await method(**kwargs)
        except Exception:  # noqa: BLE001 - integrity diagnostics are best effort
            # Integrity diagnostics must not hide the actual review result.  A
            # concrete workspace adapter still performs its own clean-up when
            # it can; callers without patch capture simply get no restore data.
            return None

    async def _normalize_reviewer_worktree(
        self,
        *,
        worktree: str,
        base_commit: str,
        initial_patch: bytes | None,
        initial_digest: str | None,
        environment_patch: bytes = b"",
        role: str,
    ) -> dict[str, Any] | None:
        """Record and remove incidental I/R worktree mutations.

        Reviewer mutations are diagnostics, not semantic protocol failures.  A
        real workspace restores the exact pre-review diff (usually P0), which
        keeps retries and any fallback consumer deterministic.  Adapters that do
        not expose restoration still return a warning and leave the decision to
        the caller; S is normally given a separate summary worktree.
        """
        if initial_digest is None:
            return None
        try:
            current_digest = await self._worktree_digest(
                worktree,
                base_commit,
                baseline_patch=environment_patch,
            )
        except Exception as error:  # noqa: BLE001 - diagnostics must be best effort
            event = {
                "role": role,
                "baseline_diff_sha256": initial_digest,
                "observed_diff_sha256": None,
                "status": "unavailable",
                "error": str(error),
            }
            self.artifacts.event("V5_REVIEW_INTEGRITY_WARNING", **event)
            return event
        if current_digest is None:
            event = {
                "role": role,
                "baseline_diff_sha256": initial_digest,
                "observed_diff_sha256": None,
                "status": "unavailable",
            }
            self.artifacts.event("V5_REVIEW_INTEGRITY_WARNING", **event)
            return event
        if current_digest == initial_digest:
            return None
        event: dict[str, Any] = {
            "role": role,
            "baseline_diff_sha256": initial_digest,
            "observed_diff_sha256": current_digest,
            "status": "observed",
        }
        restore = getattr(self.workspace, "restore_reviewer_worktree", None)
        apply_patch = getattr(self.workspace, "apply_patch", None)
        if restore is None or apply_patch is None or initial_patch is None:
            event["status"] = "unrestored"
            self.artifacts.event("V5_REVIEW_INTEGRITY_WARNING", **event)
            return event
        restore_kwargs = {
            "worktree": worktree,
            "base_commit": base_commit,
        }
        if environment_patch:
            restore_kwargs["baseline_patch"] = environment_patch
        await restore(**restore_kwargs)
        await apply_patch(worktree=worktree, patch=initial_patch)
        restored_digest = await self._worktree_digest(
            worktree,
            base_commit,
            baseline_patch=environment_patch,
        )
        event["restored_diff_sha256"] = restored_digest
        event["status"] = "restored" if restored_digest == initial_digest else "restore_mismatch"
        self.artifacts.event("V5_REVIEW_INTEGRITY_WARNING", **event)
        if restored_digest != initial_digest:
            raise V5ContractError(
                f"could not restore disposable {role} reviewer worktree"
            )
        return event

    async def _require_worktree_unchanged(
        self,
        worktree: str,
        base_commit: str,
        expected: str | None,
        *,
        role: str,
        baseline_patch: bytes = b"",
    ) -> None:
        if expected is None:
            return
        current = await self._worktree_digest(
            worktree,
            base_commit,
            baseline_patch=baseline_patch,
        )
        if current != expected:
            raise V5SnapshotMutationError(f"{role} modified its frozen worktree")

    async def _bind_ledger_sources(
        self,
        ledger: dict[str, Any],
        *,
        task: dict[str, Any],
        worktree: str,
        base_commit: str,
    ) -> list[dict[str, Any]]:
        bindings: list[dict[str, Any]] = []
        repository_sources: list[dict[str, Any]] = []
        entries = [
            *(item for item in ledger["obligations"]),
            *(item for item in ledger["preservation_obligations"]),
            *(item for item in ledger["open_questions"]),
        ]
        for entry in entries:
            sources = entry["content"].get("sources", [])
            for index, source in enumerate(sources, 1):
                source_id = f"{entry['id']}/S{index:03d}"
                if source["kind"] == "task":
                    field_text = task[source["field"]]
                    if source["quote"] not in field_text:
                        raise V5ContractError(
                            f"{source_id} quote is not contiguous text from task {source['field']}"
                        )
                    bindings.append(
                        {
                            "id": source_id,
                            **source,
                            "field_sha256": sha256_bytes(field_text.encode("utf-8")),
                        }
                    )
                else:
                    repository_sources.append({"id": source_id, **source})
        method = getattr(self.workspace, "bind_repository_sources", None)
        if repository_sources and method is not None:
            bound, errors = await method(
                worktree=worktree,
                base_commit=base_commit,
                sources=repository_sources,
            )
            if errors:
                raise V5ContractError("; ".join(errors))
            bindings.extend(bound)
        elif repository_sources:
            bindings.extend(repository_sources)
        return sorted(bindings, key=lambda item: item["id"])

    @staticmethod
    def _validate_persisted_ledger(ledger: Any) -> None:
        if not isinstance(ledger, dict):
            raise V5ContractError("persisted V5 ledger is not an object")
        try:
            spec_payload = {
                "summary": ledger["summary"],
                "obligations": [item["content"] for item in ledger["obligations"]],
                "preservation_obligations": [
                    item["content"] for item in ledger["preservation_obligations"]
                ],
                "open_questions": [
                    item["content"] for item in ledger["open_questions"]
                ],
            }
        except (KeyError, TypeError) as error:
            raise V5ContractError("persisted V5 ledger has invalid fields") from error
        validate_spec_output(spec_payload)
        expected_ids = {
            "obligations": [
                f"O{index:03d}" for index in range(1, len(ledger["obligations"]) + 1)
            ],
            "preservation_obligations": [
                f"P{index:03d}"
                for index in range(1, len(ledger["preservation_obligations"]) + 1)
            ],
            "open_questions": [
                f"Q{index:03d}"
                for index in range(1, len(ledger["open_questions"]) + 1)
            ],
        }
        for field, ids in expected_ids.items():
            if [item.get("id") for item in ledger[field]] != ids:
                raise V5ContractError(f"persisted V5 ledger has invalid {field} IDs")

    async def _validate_review_batch_evidence(
        self,
        payload: dict[str, Any],
        *,
        task: dict[str, Any],
        repository: dict[str, Any],
        worktree: str,
        role: str,
        lane_state: dict[str, Any],
    ) -> None:
        """Bind reviewer-supplied anchors before the batch becomes durable."""
        repository_sources: list[dict[str, Any]] = []
        next_number = int(lane_state.get("next_target_number", 1))
        for target_offset, target in enumerate(payload["new_targets"]):
            target_id = f"{role}{next_number + target_offset:03d}"
            for source_index, source in enumerate(target["sources"], 1):
                source_id = f"{target_id}/S{source_index:03d}"
                if source["kind"] == "task":
                    field_text = task[source["field"]]
                    if source["quote"] not in field_text:
                        raise V5ContractError(
                            f"{source_id} quote is not contiguous text from task "
                            f"{source['field']}"
                        )
                else:
                    repository_sources.append({"id": source_id, **source})

        bind_sources = getattr(self.workspace, "bind_repository_sources", None)
        if repository_sources and bind_sources is not None:
            _bound, errors = await bind_sources(
                worktree=repository["base_root"],
                base_commit=repository["base_commit"],
                sources=repository_sources,
            )
            if errors:
                raise V5ContractError("; ".join(errors))

        validate_refs = getattr(self.workspace, "validate_v5_code_references", None)
        if validate_refs is not None:
            errors = await validate_refs(
                base_worktree=repository["base_root"],
                candidate_worktree=worktree,
                base_commit=repository["base_commit"],
                assessments=payload["assessments"],
            )
            if errors:
                raise V5ContractError("; ".join(errors))

    async def _materialize_review_skill(self) -> str:
        root = Path(__file__).with_name("skills") / "v5-review-note"
        method = getattr(self.role_runner, "prepare_review_note_skill", None)
        if method is None:
            return str(root / "SKILL.md")
        return await method(root)

    @staticmethod
    def _review_skill_digest() -> str:
        root = Path(__file__).with_name("skills") / "v5-review-note"
        files = sorted(path for path in root.rglob("*") if path.is_file())
        if not files or not (root / "SKILL.md").is_file():
            raise V5ContractError("V5 review-note skill package is incomplete")
        payload = b"".join(
            path.relative_to(root).as_posix().encode("utf-8")
            + b"\0"
            + path.read_bytes()
            + b"\0"
            for path in files
        )
        return sha256_bytes(payload)

    async def _materialize_target_catalog(
        self,
        *,
        run_id: str,
        role: str,
        state: dict[str, Any],
    ) -> str:
        catalog = {
            "schema_version": "5.0",
            "run_id": run_id,
            "role": role,
            "revision": state["revision"],
            "targets": [
                {"id": target_id, "content": content}
                for target_id, content in sorted(state.get("targets", {}).items())
            ],
        }
        payload = json.dumps(
            catalog, ensure_ascii=False, indent=2, sort_keys=True
        ).encode("utf-8")
        digest = sha256_bytes(payload)
        relative = f"v5/catalogs/{role}-{digest}.json"
        local = self.artifacts.write_bytes(relative, payload)
        method = getattr(self.role_runner, "prepare_review_catalog", None)
        if method is None:
            return str(local)
        return await method(role=role, catalog=catalog)

    async def _prepare_review_output_root(
        self, artifact_key: str, host_output_root: Path
    ) -> str:
        method = getattr(self.role_runner, "prepare_review_output_root", None)
        if method is None:
            return str(host_output_root)
        return await method(artifact_key)

    async def _resume_review_output_root(
        self,
        output_root: str,
        host_output_root: Path,
        *,
        host_output_existed: bool,
    ) -> None:
        if output_root == str(host_output_root):
            if not host_output_existed:
                raise IndeterminateInvocationError(
                    "prepared local Reviewer output root was lost"
                )
            return
        method = getattr(self.role_runner, "resume_review_output_root", None)
        if method is None:
            raise IndeterminateInvocationError(
                "role runner cannot validate a prepared Reviewer output root"
            )
        resumed = await method(output_root)
        if resumed != output_root:
            raise IndeterminateInvocationError(
                "role runner resumed a different Reviewer output root"
            )

    async def _collect_review_notes(
        self,
        *,
        output_root: str,
        host_output_root: Path,
        payload: dict[str, Any],
    ) -> dict[str, bytes]:
        notes: dict[str, bytes] = {}
        method = getattr(self.role_runner, "download_review_file", None)
        for item in payload["assessments"]:
            relative = item["review_note_path"]
            if method is None:
                content = self._note_bytes(Path(output_root), relative)
            else:
                content = await method(
                    output_root=output_root,
                    relative_path=relative,
                    host_path=host_output_root / relative,
                )
            notes[relative] = content
        return notes

    async def _cleanup_review_output_root(
        self, output_root: str, host_output_root: Path
    ) -> None:
        if output_root == str(host_output_root):
            return
        method = getattr(self.role_runner, "cleanup_review_output_root", None)
        if method is not None:
            await method(output_root)

    async def _collect_review_materials(
        self,
        *,
        output_root: str,
        host_output_root: Path,
    ) -> tuple[dict[str, bytes], list[dict[str, Any]]]:
        method = getattr(self.role_runner, "download_review_materials", None)
        if method is not None:
            return await method(
                output_root=output_root,
                host_root=host_output_root / "downloaded-materials",
            )

        root = Path(output_root).resolve()
        files: dict[str, bytes] = {}
        manifest: list[dict[str, Any]] = []
        total = 0
        for category in ("probes", "logs"):
            category_root = root / category
            if not category_root.exists():
                continue
            if category_root.is_symlink() or not category_root.is_dir():
                raise V5ContractError(
                    f"review material root is not a directory: {category}"
                )
            for path in sorted(category_root.rglob("*")):
                if path.is_symlink():
                    # Dependency caches may contain symlink entries.  They are
                    # outside the declared artifact protocol and are skipped;
                    # a declared review note is still checked strictly by the
                    # path-local collector above.
                    continue
                if not path.is_file():
                    continue
                try:
                    path.resolve().relative_to(root)
                except ValueError as error:
                    raise V5ContractError(
                        f"review material escapes its output root: {path}"
                    ) from error
                relative = path.relative_to(root)
                content = path.read_bytes()
                if len(content) > 16 * 1024 * 1024:
                    raise V5ContractError(
                        f"review material exceeds per-file limit: {relative}"
                    )
                total += len(content)
                if len(files) >= 128 or total > 32 * 1024 * 1024:
                    raise V5ContractError("review materials exceed archive limits")
                archived_path = f"materials/{relative.as_posix()}"
                files[archived_path] = content
                manifest.append(
                    {
                        "original_path": str(path),
                        "archived_path": archived_path,
                        "sha256": sha256_bytes(content),
                        "material_type": category[:-1],
                        "version_evidence": "captured_after_reviewer_process_exit",
                    }
                )
        return files, manifest

    async def _materialize_summary_evidence(
        self,
        summary_input: dict[str, Any],
        *,
        archived_review_artifacts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        materialized = json.loads(json.dumps(summary_input))
        method = getattr(self.role_runner, "materialize_evidence_bundles", None)
        if method is None:
            return materialized
        artifact_root, artifacts = await method(archived_review_artifacts)
        materialized["artifact_root"] = artifact_root
        materialized["review_notes"] = self._public_review_artifacts(
            artifacts, artifact_root=artifact_root
        )
        return materialized

    @staticmethod
    def _public_review_artifacts(
        artifacts: list[dict[str, Any]],
        *,
        artifact_root: str,
    ) -> list[dict[str, Any]]:
        fields = (
            "assessment_ref",
            "version",
            "note_path",
            "note_sha256",
            "trace_manifest_path",
            "materials_manifest_path",
        )
        root = Path(artifact_root).resolve()
        public: list[dict[str, Any]] = []
        for artifact in artifacts:
            item = {field: artifact[field] for field in fields}
            for path_field in (
                "note_path",
                "trace_manifest_path",
                "materials_manifest_path",
            ):
                try:
                    item[path_field] = str(
                        Path(str(item[path_field])).resolve().relative_to(root)
                    )
                except ValueError as error:
                    raise V5ContractError(
                        f"V5 {path_field} escapes artifact_root"
                    ) from error
            public.append(item)
        return public

    def _round_state(
        self,
        state: dict[str, Any],
        *,
        ledger: dict[str, Any],
        role: str,
        output_root: str,
        skill_path: str,
        catalog_path: str,
        mode: str,
        errors: list[dict[str, str]],
        time_remaining_sec: float,
    ) -> dict[str, Any]:
        del ledger, role
        eligible = sorted(
            set(state["pending"]) | set(state["revisable"]) | set(state.get("optional", []))
        )
        return {
            "round": state["round"],
            "mode": mode,
            "batch_size": state["batch_size"],
            "registered_targets": [
                {"id": target_id, "content": content}
                for target_id, content in sorted(state.get("targets", {}).items())
            ],
            "accepted": [
                {
                    "target_id": target_id,
                    "status": record["assessment"]["status"],
                }
                for target_id, record in sorted(state["effective"].items())
            ],
            "pending_target_ids": state["pending"],
            "revisable_target_ids": state["revisable"],
            "review_notes": {
                "skill_path": skill_path,
                "output_root": output_root,
                "target_paths": [
                    {"target_id": target_id, "path": f"notes/{target_id}.md"}
                    for target_id in eligible
                ],
                "probe_root": f"{output_root}/probes",
                "log_root": f"{output_root}/logs",
            },
            "catalog_path": catalog_path,
            "errors": errors,
            "time_remaining_sec": max(0.0, time_remaining_sec),
            "rounds_remaining": max(0, self.config.max_rounds - state["round"] + 1),
        }

    @staticmethod
    def _deadline_remaining(deadline_at: str) -> float:
        try:
            deadline = datetime.fromisoformat(deadline_at)
        except (TypeError, ValueError) as error:
            raise V5ContractError("lane has an invalid persisted deadline") from error
        if deadline.tzinfo is None:
            raise V5ContractError("lane deadline must include a timezone")
        return max(0.0, (deadline - datetime.now(timezone.utc)).total_seconds())

    @staticmethod
    def _observation_fingerprint(result: RoleRunResult) -> str | None:
        try:
            trajectory = json.loads(result.trajectory_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        observations: list[dict[str, Any]] = []
        for step in trajectory.get("steps", []):
            calls = step.get("tool_calls") or []
            results = (step.get("observation") or {}).get("results") or []
            by_call = {item.get("source_call_id"): item for item in results}
            for call in calls:
                name = call.get("function_name")
                arguments = call.get("arguments") or {}
                serialized = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
                if name in {"read", "write", "edit", "ls", "find"} and (
                    "/v5_review_notes/" in serialized
                    or "/v5_resources/" in serialized
                ):
                    continue
                observation = by_call.get(call.get("tool_call_id")) or {}
                observations.append(
                    {
                        "function_name": name,
                        "arguments": arguments,
                        "content": observation.get("content"),
                    }
                )
        return sha256_json(observations) if observations else None

    @staticmethod
    def _require_success(result: RoleRunResult, label: str) -> None:
        if result.timed_out or result.exit_code != 0:
            raise RuntimeError(f"{label} failed: {result.exception or result.exit_code}")

    @staticmethod
    def _assign_targets(items: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
        return [{"id": f"{prefix}{index:03d}", "content": item} for index, item in enumerate(items, 1)]

    @staticmethod
    def _assign_questions(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{"id": f"Q{index:03d}", "content": item} for index, item in enumerate(items, 1)]

    @staticmethod
    def _note_bytes(root: Path, relative: str) -> bytes:
        path = (root / relative).resolve()
        if root.resolve() not in path.parents:
            raise V5ContractError("note path escapes output root")
        return path.read_bytes()

    def _report(
        self,
        run_id: str,
        role: str,
        *,
        integrity_events: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        state = self.state_store.lane(run_id, role)
        assessments = []
        for target_id in state["history"]:
            if target_id in state["effective"]:
                current = state["effective"][target_id]
                assessment_id = f"{role}/{target_id}"
                assessments.append({
                    "id": assessment_id,
                    "version": current["version"],
                    "content": current["assessment"],
                })
        return {
            "role": "implementation" if role == "I" else "regression",
            "completion": "complete" if state["phase"] == "complete" else "incomplete",
            "additional_targets": [
                {"id": target_id, "content": content}
                for target_id, content in sorted(state.get("targets", {}).items())
                if target_id.startswith(role)
            ],
            "assessments": assessments,
            "pending_target_ids": state["pending"],
            "stop_reason": state.get("stop_reason"),
            "review_integrity": {
                "status": "clean" if not integrity_events else "degraded",
                "events": integrity_events or [],
            },
        }

    def _review_artifacts(self, run_id: str, role: str) -> list[dict[str, Any]]:
        state = self.state_store.lane(run_id, role)
        artifacts: list[dict[str, Any]] = []
        for target_id, current in sorted(state["effective"].items()):
            bundle_sha256 = current.get("bundle_sha256")
            if not bundle_sha256:
                continue
            bundle_root = self.bundle_store.bundles / bundle_sha256
            note_path = current["assessment"]["review_note_path"]
            note_relative = (
                note_path
                if Path(note_path).parts[:1] == ("notes",)
                else str(Path("notes") / note_path)
            )
            artifacts.append(
                {
                    "assessment_ref": f"{role}/{target_id}",
                    "version": current["version"],
                    "bundle_sha256": bundle_sha256,
                    "bundle_path": str(bundle_root),
                    "note_path": str(bundle_root / note_relative),
                    "note_sha256": sha256_bytes((bundle_root / note_relative).read_bytes()),
                    "trace_manifest_path": str(bundle_root / "trace-manifest.json"),
                    "materials_manifest_path": str(bundle_root / "materials-manifest.json"),
                    "commit_id": current.get("commit_id"),
                }
            )
        return artifacts

    @staticmethod
    def _required_decision_refs(ledger: dict[str, Any], reports: dict[str, dict[str, Any]]) -> list[str]:
        refs = [f"A/{item['id']}" for item in ledger["open_questions"]]
        for report in reports.values():
            refs.extend(item["id"] for item in report["assessments"] if item["content"]["status"] != "satisfied")
        return refs

    @staticmethod
    def _decision_context(
        ledger: dict[str, Any], reports: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        targets = {
            item["id"]: item["content"]
            for item in [
                *ledger["obligations"],
                *ledger["preservation_obligations"],
                *(
                    target
                    for report in reports.values()
                    for target in report.get("additional_targets", [])
                ),
            ]
        }
        context: dict[str, dict[str, Any]] = {
            f"A/{item['id']}": {
                "kind": "question",
                "blocking": bool(item["content"].get("blocking")),
                "sources": item["content"].get("sources", []),
            }
            for item in ledger["open_questions"]
        }
        for report in reports.values():
            for item in report["assessments"]:
                target_id = item["content"]["target_id"]
                target = targets.get(target_id, {})
                context[item["id"]] = {
                    "kind": "assessment",
                    "status": item["content"]["status"],
                    "importance": target.get("importance", "required"),
                    "sources": target.get("sources", []),
                }
        return context

    @staticmethod
    def _allowed_summary_sources(
        ledger: dict[str, Any], reports: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        sources: list[dict[str, Any]] = []
        for entry in [
            *ledger["obligations"],
            *ledger["preservation_obligations"],
            *ledger["open_questions"],
            *(target for report in reports for target in report.get("additional_targets", [])),
        ]:
            sources.extend(entry["content"].get("sources", []))
        unique = {
            json.dumps(source, ensure_ascii=False, sort_keys=True): source
            for source in sources
        }
        return list(unique.values())

    @staticmethod
    def _route(
        ledger: dict[str, Any],
        reports: dict[str, dict[str, Any]],
        summary: dict[str, Any],
        decision_context: dict[str, dict[str, Any]],
    ) -> str:
        del ledger
        if summary["repair_requests"]:
            return "repair"
        if any(report["completion"] != "complete" for report in reports.values()):
            return "hold"
        decisions = {item["finding_ref"]: item for item in summary["decisions"]}
        for reference, context in decision_context.items():
            if context["kind"] == "assessment" and context.get("status") == "satisfied":
                challenge = decisions.get(reference)
                if (
                    challenge is not None
                    and challenge["disposition"] == "unresolved"
                    and context.get("importance") == "required"
                ):
                    return "hold"
                continue
            decision = decisions.get(reference)
            if decision is None:
                return "hold"
            if decision["disposition"] != "unresolved":
                continue
            if context["kind"] == "question" and context.get("blocking"):
                return "hold"
            if context["kind"] == "assessment" and context.get("importance") == "required":
                return "hold"
        if any(question.get("blocking") for question in summary["new_questions"]):
            return "hold"
        return "accept"

    @staticmethod
    def _preserve_refs(
        ledger: dict[str, Any],
        reports: dict[str, dict[str, Any]],
        summary: dict[str, Any],
    ) -> list[str]:
        target_importance = {
            item["id"]: item["content"]["importance"]
            for item in [
                *ledger["obligations"],
                *ledger["preservation_obligations"],
                *(
                    target
                    for report in reports.values()
                    for target in report.get("additional_targets", [])
                ),
            ]
        }
        challenged = {
            decision["finding_ref"]
            for decision in summary["decisions"]
            if decision["disposition"] == "unresolved"
        }
        assessment_targets = {
            item["id"]: item["content"]["target_id"]
            for report in reports.values()
            for item in report["assessments"]
        }
        repaired_targets = {
            assessment_targets[decision["finding_ref"]]
            for decision in summary["decisions"]
            if decision["disposition"] == "repair"
            and decision["finding_ref"] in assessment_targets
        }
        return sorted(
            item["id"]
            for report in reports.values()
            for item in report["assessments"]
            if item["content"]["status"] == "satisfied"
            and target_importance.get(item["content"]["target_id"]) == "required"
            and item["id"] not in challenged
            and item["content"]["target_id"] not in repaired_targets
        )


__all__ = ["V5Controller", "V5Result"]
