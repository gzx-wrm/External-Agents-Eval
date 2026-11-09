"""V6.5 review controller: review view (A), assumptions (C) and the D lane.

``run`` follows ``V5Controller.run`` step for step; V6.5 additions are marked.
V5/V6.4 code paths are not modified: the shared controller only gained the
``_lane_prompt``/``_summary_prompt`` extension points, whose defaults are the
original builders.

Information barrier: before any reviewer starts, the reviewer worktrees hold the
view instead of P0, the raw-P0 inputs the container received are deleted and
unreachable Git objects are pruned. X, the only role that reads the withheld text,
has no tools and its session is never uploaded to the container.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import uuid
from enum import Enum
from pathlib import Path
from typing import Any

from .artifacts import ArtifactStore, sha256_bytes, sha256_json
from .build_gate import FAIL as GATE_FAIL
from .build_gate import PASS as GATE_PASS
from .build_gate import build_check_for_repair
from .models import RoleName, RoleRunResult, TaskInput
from .v5_config import V5_ROLE_STEP_LIMIT
from .v5_contracts import V5ContractError, parse_json_object, validate_managed_delivery
from .v5_controller import V5Result, V5SnapshotMutationError
from .v5_prompts import (
    REVIEW_BATCH_SCHEMA,
    SPEC_OUTPUT_SCHEMA,
    SUMMARY_OUTPUT_SCHEMA,
    V56_PROMPT_VERSION,
    repair_prompt,
    reviewer_prompt,
)
from .v6_controller import V6Controller
from .v65_config import V65Config
from .v65_contracts import select_disambiguation_targets, validate_assumption_output
from .v65_prompts import (
    ASSUMPTION_OUTPUT_SCHEMA,
    assumption_prompt,
    disambiguation_prompt,
    insert_view_notice,
    summary_prompt_v65,
    view_notice,
)
from .v65_state import V65StateStore

D_WORKTREE = "v65-disambiguation-review"
HERE = Path(__file__).resolve().parent
# Strings whose presence in a reviewer command suggests reaching for withheld input.
# The raw P0's own file name (apply-<sha16>.patch) is added per run.
BARRIER_MARKERS = ("mini_sessions", "fsck", "--batch-all-objects", "lost-found",
                   "v65-review-view", "v65-view-input", "/tmp/v65-")


class V65Role(str, Enum):
    DISAMBIGUATION_REVIEWER = "v5_disambiguation_reviewer"
    ASSUMPTION_EXTRACTOR = "v5_assumption_extractor"


def _file_sha256(name: str) -> str:
    return sha256_bytes((HERE / name).read_bytes())


class V65Controller(V6Controller):
    def __init__(
        self,
        *,
        role_runner: Any,
        workspace: Any,
        artifacts: ArtifactStore,
        config: V65Config | None = None,
        build_gate: Any = None,
    ) -> None:
        super().__init__(
            role_runner=role_runner, workspace=workspace, artifacts=artifacts,
            config=config or V65Config(), build_gate=build_gate,
        )
        self.state_store.close()
        self.state_store = V65StateStore(self.artifacts.root / "v5" / "workflow.sqlite")
        self._view_level: str | None = None
        self._d_catalog: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # prompt and role extension points
    # ------------------------------------------------------------------

    def _lane_prompt(self, **kwargs: Any) -> str:
        notice = view_notice(self._view_level)
        if kwargs["role"] == "D":
            return disambiguation_prompt(**kwargs, view=notice)
        # I/R keep the V5-6 text; V6.5 inserts only the REVIEW VIEW notice.
        prompt = reviewer_prompt(**{**kwargs, "prompt_version": V56_PROMPT_VERSION})
        return insert_view_notice(prompt, notice)

    def _summary_prompt(self, summary_input: dict[str, Any], **kwargs: Any) -> str:
        return summary_prompt_v65(summary_input, **kwargs)

    async def _run_role(
        self,
        *,
        role: Any,
        prompt: str,
        worktree: str,
        prefix: str,
        session_role: str | None = None,
        invocation_id: str | None = None,
        timeout_sec: float | None = None,
    ) -> RoleRunResult:
        if session_role == "D" and role == RoleName.V5_REGRESSION_REVIEWER:
            role = V65Role.DISAMBIGUATION_REVIEWER
        if role != V65Role.ASSUMPTION_EXTRACTOR:
            return await super()._run_role(
                role=role, prompt=prompt, worktree=worktree, prefix=prefix,
                session_role=session_role, invocation_id=invocation_id, timeout_sec=timeout_sec,
            )
        # X is tool-less and sessionless: nothing it reads reaches the container.
        kwargs: dict[str, Any] = {
            "role": role, "prompt": prompt, "worktree": worktree, "tools": (),
            "timeout_sec": timeout_sec or self.config.assumption_timeout_sec,
            "artifact_prefix": prefix,
        }
        if invocation_id is not None and "invocation_id" in inspect.signature(
                self.role_runner.run_role).parameters:
            kwargs["invocation_id"] = invocation_id
        if invocation_id is not None and self._active_run_id is not None:
            self.state_store.prepare_invocation(
                invocation_id, run_id=self._active_run_id, role=role.value,
                prompt_sha256=sha256_bytes(prompt.encode("utf-8")), worktree=worktree,
            )
        result = await self.role_runner.run_role(**kwargs)
        if invocation_id is not None and self._active_run_id is not None:
            self.state_store.finish_invocation(
                invocation_id,
                status="COMPLETED" if result.exit_code == 0 and not result.timed_out else "FAILED",
                result={"exit_code": result.exit_code, "timed_out": result.timed_out,
                        "exception": result.exception, "started_at": result.started_at,
                        "finished_at": result.finished_at,
                        "resource_usage": result.resource_usage,
                        "transcript_path": str(result.transcript_path),
                        "trajectory_path": str(result.trajectory_path)},
            )
        return result

    def _report(self, run_id: str, role: str, *, integrity_events=None) -> dict[str, Any]:
        report = super()._report(run_id, role, integrity_events=integrity_events)
        if role == "D":
            report["role"] = "disambiguation"
        return report

    def _decision_context_v65(
        self, ledger: dict[str, Any], reports: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        context = self._decision_context(ledger, reports)
        # Q/X targets are advisory: only a derived D obligation can be repaired.
        for item in reports.get("D", {}).get("assessments", []):
            target = self._d_catalog.get(item["content"]["target_id"])
            if target is not None:
                context[item["id"]] = {**context[item["id"]], "importance": "advisory",
                                       "sources": target.get("sources", [])}
        return context

    # ------------------------------------------------------------------
    # A: review view and barrier
    # ------------------------------------------------------------------

    async def _build_review_view(
        self,
        *,
        task: TaskInput,
        candidate_patch: bytes,
        baseline_patch: bytes,
        p0_gate: str | None,
    ) -> tuple[bytes, dict[str, Any]]:
        p0_sha = sha256_bytes(candidate_patch)
        record_path = self.artifacts.root / "v65" / "review-view.json"
        patch_path = self.artifacts.root / "outputs" / "v65-view.patch"
        if record_path.is_file() and patch_path.is_file():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            view = patch_path.read_bytes()
            if record.get("p0_sha256") == p0_sha and record.get("view_sha256") == sha256_bytes(view):
                return view, record
        build = getattr(self.workspace, "build_review_view", None)
        if build is None:
            raise V5ContractError("V6.5 review view requires a workspace that can build it")
        attempts: list[dict[str, Any]] = []
        chosen: tuple[str, bytes, dict[str, Any]] | None = None
        try:
            for level in self.config.review_view_levels:
                view, manifest = await build(
                    base_commit=task.base_commit, baseline_patch=baseline_patch,
                    candidate_patch=candidate_patch, level=level,
                )
                attempt = {"level": level, "view_sha256": sha256_bytes(view),
                           "stripped_item_count": manifest.get("stripped_item_count", 0),
                           "skipped": manifest.get("skipped", [])}
                # Fall back only when the view breaks a build that P0 passes.
                if self._gate_active() and p0_gate == GATE_PASS and view != candidate_patch:
                    report = await self._run_build_gate(f"VIEW-{level}", view)
                    attempt["gate"] = report.get("verdict")
                    if report.get("verdict") == GATE_FAIL:
                        attempts.append(attempt)
                        continue
                attempts.append(attempt)
                chosen = (level, view, manifest)
                break
        finally:
            await self.workspace.discard_review_view(candidate_patch=candidate_patch)
        if chosen is None:
            level, view, manifest = None, candidate_patch, {}
            self.artifacts.event("V65_REVIEW_VIEW_FALLBACK_RAW", attempts=attempts)
        else:
            level, view, manifest = chosen
        record = {"schema_version": "v6.5-review-view-record-1", "p0_sha256": p0_sha,
                  "level": level, "view_sha256": sha256_bytes(view), "attempts": attempts,
                  "manifest": manifest}
        self.artifacts.write_bytes("outputs/v65-view.patch", view)
        self.artifacts.write_json("v65/review-view.json", record)
        self.artifacts.event("V65_REVIEW_VIEW", level=level, view_sha256=record["view_sha256"])
        return view, record

    async def _install_review_view(
        self,
        *,
        task: TaskInput,
        view: bytes,
        candidate_patch: bytes,
        baseline_patch: bytes,
        worktrees: list[str],
    ) -> str | None:
        for worktree in dict.fromkeys(worktrees):
            await self.workspace.reset_review_worktree(
                worktree=worktree, base_commit=task.base_commit,
                baseline_patch=baseline_patch, patch=view,
            )
        d_worktree = None
        if self.config.disambiguation_enabled:
            d_worktree = await self.workspace.prepare_review_worktree(
                name=D_WORKTREE, base_commit=task.base_commit,
                baseline_patch=baseline_patch, patch=view,
            )
        removed = await self.workspace.remove_candidate_inputs(candidate_patch=candidate_patch)
        pruned = await self.workspace.prune_unreachable_objects()
        self.artifacts.write_json("v65/barrier.json", {
            "reviewer_worktrees": list(dict.fromkeys(worktrees)),
            "disambiguation_worktree": d_worktree,
            "removed_inputs": removed, "prune": pruned,
        })
        return d_worktree

    def _barrier_audit(self, p0_sha256: str) -> dict[str, Any]:
        markers = (*BARRIER_MARKERS, f"apply-{p0_sha256[:16]}")
        hits: dict[str, list[dict[str, Any]]] = {}
        root = self.artifacts.root / "v5" / "reviewer"
        for path in sorted(root.rglob("trajectory.json")) if root.is_dir() else []:
            try:
                trajectory = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for step in trajectory.get("steps", []):
                for call in step.get("tool_calls") or []:
                    command = json.dumps(call.get("arguments") or {}, ensure_ascii=False)
                    found = [marker for marker in markers if marker in command]
                    if found:
                        role = path.relative_to(root).parts[0]
                        hits.setdefault(role, []).append(
                            {"trajectory": str(path.relative_to(self.artifacts.root)),
                             "markers": found, "command": command[:500]})
        audit = {"markers": list(markers), "hits": hits, "clean": not any(hits.values())}
        self.artifacts.write_json("v65/barrier-audit.json", audit)
        return audit

    # ------------------------------------------------------------------
    # C: assumptions
    # ------------------------------------------------------------------

    async def _extract_assumptions(
        self,
        *,
        run_id: str,
        task: dict[str, Any],
        ledger: dict[str, Any],
        candidate_patch: bytes,
        view_record: dict[str, Any],
        worktree: str,
        deadline_at: str,
    ) -> list[dict[str, Any]]:
        p0_sha = sha256_bytes(candidate_patch)
        path = self.artifacts.root / "v65" / "assumptions.json"
        if path.is_file():
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached.get("p0_sha256") == p0_sha:
                return cached["assumptions"]
        withheld = [
            {"path": entry["path"], "line": item["line"], "kind": item["kind"], "text": item["text"]}
            for entry in view_record.get("manifest", {}).get("stripped", [])
            for item in entry["items"]
        ]
        base_prompt = assumption_prompt(
            task=task, ledger=ledger,
            candidate_patch=candidate_patch.decode("utf-8", "replace"),
            withheld=withheld, max_items=self.config.assumption_max_items,
        )
        error: Exception | None = None
        for attempt in range(1, self.config.assumption_max_attempts + 1):
            remaining = self._deadline_remaining(deadline_at)
            if remaining <= 0:
                error = V5ContractError("assumption deadline expired")
                break
            prompt = base_prompt if error is None else (
                base_prompt + "\nThe previous object was rejected. Return a complete "
                f"replacement.\nVALIDATION ERROR\n{error}\n")
            result = await self._run_role(
                role=V65Role.ASSUMPTION_EXTRACTOR, prompt=prompt, worktree=worktree,
                prefix=f"v65/assumptions/attempt-{attempt}",
                invocation_id=f"{run_id}.X.{attempt}",
                timeout_sec=min(self.config.assumption_timeout_sec, remaining),
            )
            try:
                self._require_success(result, "V6.5 assumption extractor")
                items = validate_assumption_output(
                    parse_json_object(result.final_assistant_text), ledger=ledger,
                    withheld_texts=[item["text"] for item in withheld],
                    max_items=self.config.assumption_max_items,
                )
            except (V5ContractError, RuntimeError) as failure:
                error = failure
                continue
            self.artifacts.write_json("v65/assumptions.json", {
                "status": "completed", "p0_sha256": p0_sha, "attempts": attempt,
                "assumptions": items,
            })
            return items
        # X never blocks review: D proceeds with the ledger questions alone.
        self.artifacts.write_json("v65/assumptions.json", {
            "status": "failed", "p0_sha256": p0_sha, "error": str(error), "assumptions": [],
        })
        self.artifacts.event("V65_ASSUMPTIONS_FAILED", error=str(error))
        return []

    # ------------------------------------------------------------------
    # orchestration (V5Controller.run plus the V6.5 steps)
    # ------------------------------------------------------------------

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
        reused_reviews: dict[str, dict[str, Any]] | None = None,
        reused_summary: dict[str, Any] | None = None,
        export_v6_stages: bool = False,
    ) -> V5Result:
        if reused_reviews or reused_summary is not None:
            raise V5ContractError("V6.5 never reuses reviews or summaries from another run")
        run_id = run_id or uuid.uuid4().hex
        self._active_run_id = run_id
        p0_sha = sha256_bytes(candidate_patch)
        task_json = {"title": task.issue_title, "description": task.issue_body}
        managed_delivery = validate_managed_delivery(list(task.managed_delivery), task=task_json)
        config = self.config
        binding = {
            "task_id": task.task_id,
            "task_sha256": sha256_json(task_json),
            "repository_name": task.repository_name,
            "base_commit": task.base_commit,
            "patch_base": "environment-baseline" if baseline_patch else "base-commit",
            "baseline_patch_sha256": sha256_bytes(baseline_patch),
            "candidate_patch_sha256": p0_sha,
            "controller_version": config.controller_version,
            "prompt_version": config.prompt_version,
            "prompt_builder_sha256": _file_sha256("v5_prompts.py"),
            "contract_validator_sha256": _file_sha256("v5_contracts.py"),
            "output_schemas_sha256": sha256_json({
                "spec": SPEC_OUTPUT_SCHEMA, "review": REVIEW_BATCH_SCHEMA,
                "summary": SUMMARY_OUTPUT_SCHEMA, "assumptions": ASSUMPTION_OUTPUT_SCHEMA,
            }),
            "review_note_skill_sha256": self._review_skill_digest(self._review_skill_name()),
            "model": config.model,
            "managed_delivery_sha256": sha256_json(managed_delivery),
            "limits": {
                "batch_size": config.batch_size, "max_rounds": config.max_rounds,
                "max_format_retries": config.max_format_retries,
                "max_semantic_retries": config.max_semantic_retries,
                "max_no_progress": config.max_no_progress,
                "lane_deadline_sec": config.lane_deadline_sec,
                "role_timeout_sec": config.role_timeout_sec,
                "no_compaction": config.no_compaction,
            },
            # V6.5
            "v65": {
                "review_view": {"enabled": config.review_view_enabled,
                                "levels": list(config.review_view_levels),
                                "builder_sha256": _file_sha256("review_view.py")},
                "assumptions": {"enabled": config.assumptions_enabled,
                                "timeout_sec": config.assumption_timeout_sec,
                                "max_items": config.assumption_max_items,
                                "max_attempts": config.assumption_max_attempts},
                "disambiguation": {"enabled": config.disambiguation_enabled,
                                   "max_targets": config.disambiguation_max_targets},
                "prompts_sha256": _file_sha256("v65_prompts.py"),
                "contracts_sha256": _file_sha256("v65_contracts.py"),
                "controller_sha256": _file_sha256("v65_controller.py"),
            },
        }
        if self._gate_active():
            binding["build_gate"] = self.build_gate.config()
        created = self.state_store.create_run(run_id, binding)
        self.artifacts.write_bytes("outputs/v5-p0.patch", candidate_patch)
        if not created:
            cached = self._completed_result(run_id, p0_sha)
            if cached is not None:
                return cached
            if self.state_store.run_record(run_id)["phase"] == "completed":
                raise V5ContractError("completed V5 run is missing or has corrupt delivery artifacts")
        skill_path = await self._materialize_review_skill()
        regression_worktree = regression_worktree or candidate_worktree
        summary_worktree = summary_worktree or candidate_worktree
        try:
            ledger_path = self.artifacts.root / "v5" / "ledger.json"
            if not ledger_path.is_file():
                raise V5ContractError("V6.5 reviews a frozen ledger; none was supplied")
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            self._validate_persisted_ledger(ledger)
            bindings_path = self.artifacts.root / "v5" / "source-bindings.json"
            if not bindings_path.is_file():
                raise V5ContractError("persisted V5 ledger is missing source bindings")
            await self._load_persisted_ledger_bindings(
                ledger, json.loads(bindings_path.read_text(encoding="utf-8")),
                task=task_json, base_worktree=base_worktree, base_commit=task.base_commit,
            )
            self._persist_role_artifact(
                "v5/exports/ledger.json", task_id=task.task_id, role="A",
                base_commit=task.base_commit, candidate_sha256=None, ledger_sha256=None,
                content=ledger,
            )

            gate_reports: dict[str, dict[str, Any]] = {}
            if self._gate_active():
                # Deterministic and model-free; reviewers never see its result.
                gate_reports["P0"] = await self._run_build_gate("P0", candidate_patch)

            # V6.5 A: build the view and install it before any reviewer starts.
            view_record: dict[str, Any] = {"level": None, "manifest": {}}
            review_patch = candidate_patch
            d_worktree: str | None = None
            if config.review_view_enabled:
                review_patch, view_record = await self._build_review_view(
                    task=task, candidate_patch=candidate_patch, baseline_patch=baseline_patch,
                    p0_gate=gate_reports.get("P0", {}).get("verdict"),
                )
                self._view_level = view_record["level"]
            if config.review_view_enabled or config.disambiguation_enabled:
                d_worktree = await self._install_review_view(
                    task=task, view=review_patch, candidate_patch=candidate_patch,
                    baseline_patch=baseline_patch,
                    worktrees=[candidate_worktree, regression_worktree, summary_worktree],
                )
            diff_path = await self._materialize_candidate_patch(
                review_patch, sha256_bytes(review_patch))
            repository = {
                "name": task.repository_name, "base_commit": task.base_commit,
                "base_root": base_worktree, "diff_path": diff_path,
                "patch_base": "environment-baseline" if baseline_patch else "base-commit",
                "baseline_patch_sha256": sha256_bytes(baseline_patch),
            }

            self.state_store.initialize_lanes(
                run_id,
                implementation_targets=[item["id"] for item in ledger["obligations"]],
                regression_targets=[item["id"] for item in ledger["preservation_obligations"]],
                optional_regression_targets=[item["id"] for item in ledger["obligations"]],
                target_catalog={
                    item["id"]: item["content"]
                    for item in [*ledger["obligations"], *ledger["preservation_obligations"]]
                },
                batch_size=config.batch_size,
                lane_deadline_sec=config.lane_deadline_sec,
            )
            window_deadline = self.state_store.lane(run_id, "I")["deadline_at"]

            async def review(role: str, worktree: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
                report = await self._run_lane(
                    run_id=run_id, role=role, task=task_json, repository=repository,
                    ledger=ledger, worktree=worktree, skill_path=skill_path,
                    managed_delivery=managed_delivery, base_commit=task.base_commit,
                    baseline_patch=baseline_patch,
                )
                artifacts = self._review_artifacts(run_id, role)
                self._persist_role_artifact(
                    f"v5/exports/review-{role}.json", task_id=task.task_id, role=role,
                    base_commit=task.base_commit, candidate_sha256=p0_sha,
                    ledger_sha256=sha256_json(ledger), content=report,
                )
                if export_v6_stages:
                    from .v6_fork import write_stage

                    write_stage(
                        self.artifacts.root, stage=role, task_id=task.task_id,
                        base_commit=task.base_commit, baseline_sha256=sha256_bytes(baseline_patch),
                        ledger_sha256=sha256_json(ledger), p0_sha256=p0_sha,
                        report_sha256=sha256_json(report), review_artifacts=artifacts,
                    )
                return report, artifacts

            async def disambiguation() -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
                # V6.5 C then D, inside the same review window as I/R.
                assumptions: list[dict[str, Any]] = []
                if config.assumptions_enabled:
                    assumptions = await self._extract_assumptions(
                        run_id=run_id, task=task_json, ledger=ledger,
                        candidate_patch=candidate_patch, view_record=view_record,
                        worktree=base_worktree, deadline_at=window_deadline,
                    )
                if not config.disambiguation_enabled:
                    return None
                targets = select_disambiguation_targets(
                    ledger, assumptions, max_targets=config.disambiguation_max_targets)
                self.artifacts.write_json("v65/disambiguation-targets.json", {
                    "max_targets": config.disambiguation_max_targets, "targets": targets,
                    "assumption_count": len(assumptions),
                    "question_count": len(ledger.get("open_questions", [])),
                })
                if not targets or d_worktree is None:
                    return None
                self._d_catalog = {item["id"]: item["content"] for item in targets}
                self.state_store.initialize_extra_lane(
                    run_id, "D", targets=[item["id"] for item in targets],
                    target_catalog=self._d_catalog, batch_size=config.batch_size,
                    deadline_at=window_deadline,
                )
                return await review("D", d_worktree)

            results = await asyncio.gather(
                review("I", candidate_worktree),
                review("R", regression_worktree),
                disambiguation(),
                return_exceptions=export_v6_stages,
            )
            if export_v6_stages:
                for outcome in results:
                    if isinstance(outcome, BaseException):
                        raise outcome
            reports: dict[str, dict[str, Any]] = {"I": results[0][0], "R": results[1][0]}
            archived_review_artifacts = results[0][1] + results[1][1]
            if results[2] is not None:
                reports["D"] = results[2][0]
                archived_review_artifacts += results[2][1]
            self._barrier_audit(p0_sha)
            if summary_worktree == candidate_worktree:
                unrestored = [
                    event for report in reports.values()
                    for event in report.get("review_integrity", {}).get("events", [])
                    if event.get("status") != "restored"
                ]
                if unrestored:
                    raise V5SnapshotMutationError(
                        "reviewer modified its frozen worktree and it could not be restored before summary"
                    )
            snapshot = {
                "run_id": run_id, "ledger_sha256": sha256_json(ledger),
                "source_bindings": self._source_bindings, "reports": reports,
                "review_artifacts": archived_review_artifacts,
                "lane_revisions": {role: self.state_store.lane(run_id, role)["revision"]
                                   for role in reports},
            }
            freeze_id = self.state_store.freeze_summary(run_id, snapshot)
            artifact_root = str(self.bundle_store.bundles)
            decision_context = self._decision_context_v65(ledger, reports)
            report_list = [reports[role] for role in ("I", "R", "D") if role in reports]
            summary_input = {
                "task": task_json,
                "managed_delivery": managed_delivery,
                "ledger": ledger,
                "reviews": report_list,
                # V6.5: what the reviewers saw and which Q/X targets D resolved.
                "review_view": {"level": view_record["level"],
                                "notice": view_notice(view_record["level"])},
                "disambiguation_targets": [
                    {"id": target_id, "content": content}
                    for target_id, content in self._d_catalog.items()
                ],
                "review_notes": self._public_review_artifacts(
                    archived_review_artifacts, artifact_root=artifact_root),
                "artifact_root": artifact_root,
                "required_decision_refs": self._required_decision_refs(ledger, reports),
                "allowed_supporting_refs": sorted(decision_context),
                "allowed_validation_record_refs": sorted(
                    ref for ref, context in decision_context.items()
                    if context["kind"] == "assessment"),
                "allowed_preserve_refs": sorted(
                    ref for ref, context in decision_context.items()
                    if context["kind"] == "assessment" and context.get("status") == "satisfied"),
                "allowed_sources": self._allowed_summary_sources(ledger, report_list),
            }
            model_summary_input = await self._materialize_summary_evidence(
                summary_input, archived_review_artifacts=archived_review_artifacts)
            summary = await self._run_summary(
                model_summary_input, summary_worktree, decision_context=decision_context,
                freeze_id=freeze_id, base_commit=task.base_commit, baseline_patch=baseline_patch,
                deadline_at=self.state_store.stage_deadline(
                    run_id, "S", duration_sec=config.role_timeout_sec),
            )
            self.artifacts.write_json("v5/summary.json", summary)
            self._persist_role_artifact(
                "v5/exports/summary.json", task_id=task.task_id, role="S",
                base_commit=task.base_commit, candidate_sha256=p0_sha,
                ledger_sha256=sha256_json(ledger), content=summary,
                extra_bindings={"summary_freeze_id": freeze_id},
            )
            route = self._route(ledger, reports, summary, decision_context)
            gate_override = False
            if gate_reports.get("P0", {}).get("verdict") == GATE_FAIL and route != "repair":
                route, gate_override = "repair", True
            decision_artifact = {
                "route": route,
                "review_complete": all(r["completion"] == "complete" for r in reports.values()),
                "preserve_refs": self._preserve_refs(ledger, reports, summary),
                "plan": summary,
                "freeze_id": freeze_id,
            }
            if self._gate_active():
                decision_artifact["build_gate"] = {
                    "P0": gate_reports["P0"].get("verdict"), "route_override": gate_override}
            self.artifacts.write_json("v5/decision.json", decision_artifact)
            if export_v6_stages:
                from .v6_fork import write_stage

                write_stage(
                    self.artifacts.root, stage="S", task_id=task.task_id,
                    base_commit=task.base_commit, baseline_sha256=sha256_bytes(baseline_patch),
                    ledger_sha256=sha256_json(ledger), p0_sha256=p0_sha,
                    review_sha256={role: sha256_json(reports[role]) for role in reports},
                    summary_sha256=sha256_json(summary), route=route,
                )
            repair_invocation_id = f"{run_id}.repair.{freeze_id}" if route == "repair" else None
            self.state_store.prepare_delivery(
                run_id, freeze_id=freeze_id, route=route,
                repair_invocation_id=repair_invocation_id)
            p1 = candidate_patch
            repair_result: RoleRunResult | None = None
            if route == "repair" and repair_worktree is None:
                raise V5ContractError("V5 repair route requires a repair worktree")
            if route == "repair":
                assert repair_worktree is not None
                # Repair edits the real P0, comments and tests included.
                await self.workspace.apply_patch(worktree=repair_worktree, patch=candidate_patch)
                repair_result = await self._run_role(
                    role=RoleName.REPAIR,
                    prompt=repair_prompt(
                        task_json, p0_sha, summary,
                        review_context={
                            "freeze_id": freeze_id,
                            "review_notes": model_summary_input["review_notes"],
                            "artifact_root": model_summary_input["artifact_root"],
                            "preserve_refs": decision_artifact["preserve_refs"],
                            "reports": reports,
                        },
                        prompt_version=V56_PROMPT_VERSION,
                        budget={"minutes": max(1, int(config.role_timeout_sec) // 60),
                                "steps": V5_ROLE_STEP_LIMIT},
                        build_check=(build_check_for_repair(gate_reports["P0"])
                                     if "P0" in gate_reports else None),
                    ),
                    worktree=repair_worktree, prefix="v5/repair",
                    invocation_id=repair_invocation_id,
                )
                try:
                    capture_kwargs = {"worktree": repair_worktree, "base_commit": task.base_commit}
                    if baseline_patch:
                        capture_kwargs["baseline_patch"] = baseline_patch
                    p1 = await self.workspace.capture_patch(**capture_kwargs)
                    self.artifacts.event(
                        "V5_REPAIR_CAPTURED", exit_code=repair_result.exit_code,
                        timed_out=repair_result.timed_out,
                        partial=repair_result.exit_code != 0 or repair_result.timed_out,
                        p1_sha256=sha256_bytes(p1),
                    )
                except Exception as error:  # noqa: BLE001 - patch capture must safely fall back
                    p1 = candidate_patch
                    self.artifacts.event("V5_REPAIR_FALLBACK_P0", error=str(error))
            gate_summary: dict[str, Any] | None = None
            if self._gate_active():
                delivered = "P0" if p1 == candidate_patch else "P1"
                if route == "repair" and p1 != candidate_patch:
                    gate_reports["P1"] = await self._run_build_gate("P1", p1)
                    if (gate_reports["P0"].get("verdict") != GATE_FAIL
                            and gate_reports["P1"].get("verdict") == GATE_FAIL):
                        p1, delivered = candidate_patch, "P0"
                        self.artifacts.event("V64_BUILD_GATE_REVERT_TO_P0", run_id=run_id)
                gate_summary = {
                    "schema_version": self.build_gate.config()["schema_version"],
                    "P0": gate_reports["P0"].get("verdict"),
                    "P1": gate_reports.get("P1", {}).get("verdict"),
                    "route_override": gate_override,
                    "delivered": delivered,
                    "reports": {label: f"v5/build-gate/{label}.json" for label in gate_reports},
                }
                self.artifacts.write_json("v5/build-gate.json", gate_summary)
            self.artifacts.write_bytes("outputs/v5-p1.patch", p1)
            self.state_store.finish_delivery(
                run_id, p1_sha256=sha256_bytes(p1),
                result={
                    "route": route,
                    "repair_executed": repair_result is not None,
                    "repair_exit_code": repair_result.exit_code if repair_result else None,
                    "repair_timed_out": repair_result.timed_out if repair_result else False,
                    "partial_repair_delivery": bool(
                        repair_result is not None
                        and (repair_result.exit_code != 0 or repair_result.timed_out)),
                    **({"build_gate": gate_summary} if gate_summary is not None else {}),
                },
            )
            if export_v6_stages:
                from .v6_fork import write_stage

                write_stage(
                    self.artifacts.root, stage="P1", task_id=task.task_id,
                    base_commit=task.base_commit, baseline_sha256=sha256_bytes(baseline_patch),
                    ledger_sha256=sha256_json(ledger), p0_sha256=p0_sha,
                    summary_sha256=sha256_json(summary), patch_sha256=sha256_bytes(p1),
                    route=route,
                )
            return V5Result(run_id, route, p0_sha, p1, ledger, reports, summary, freeze_id,
                            build_gate=gate_summary)
        except Exception as error:
            self.artifacts.event("V5_FAILED", error=str(error), run_id=run_id)
            raise


__all__ = ["D_WORKTREE", "V65Controller", "V65Role"]
