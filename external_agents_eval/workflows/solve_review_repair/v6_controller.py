"""V6.1 orchestration over the unchanged V5 review controller.

V6.1 makes the A ledger available to Solver before P0 is produced.  Mechanical
source-anchor problems are recorded as degraded ledger quality and do not block
the run.  A structurally unparseable A result is the one terminal pre-solver
condition: no P0 is generated and the adapter records a terminal manifest.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from .artifacts import ArtifactStore, sha256_bytes, sha256_json
from .models import TaskInput
from .v5_contracts import validate_managed_delivery
from .v5_controller import V5Controller
from .v6_config import V6Config


class V6Controller(V5Controller):
    """Generate A before P0, then reuse the exact ledger in V5 review."""

    def __init__(
        self,
        *,
        role_runner: Any,
        workspace: Any,
        artifacts: ArtifactStore,
        config: V6Config | None = None,
        build_gate: Any = None,
    ) -> None:
        super().__init__(
            role_runner=role_runner,
            workspace=workspace,
            artifacts=artifacts,
            config=config or V6Config(),
            build_gate=build_gate,
        )
        self.ledger_quality = "unavailable"
        self.ledger_warnings: list[str] = []
        self.ledger_error: str | None = None

    def _warn_ledger(self, message: str) -> None:
        if message not in self.ledger_warnings:
            self.ledger_warnings.append(message)

    def _write_pre_solver_envelope(
        self,
        *,
        stage_id: str,
        task: TaskInput,
        baseline_patch: bytes,
        ledger: dict[str, Any] | None,
    ) -> None:
        self.artifacts.write_json(
            "v6/pre-solver-ledger.json",
            {
                "schema_version": f"{self.config.workflow_controller_version}-pre-solver-ledger-1",
                "stage_id": stage_id,
                "task_id": task.task_id,
                "base_commit": task.base_commit,
                "baseline_patch_sha256": sha256_bytes(baseline_patch),
                "ledger_quality": self.ledger_quality,
                "ledger_error": self.ledger_error,
                "ledger_warnings": list(self.ledger_warnings),
                "ledger_sha256": sha256_json(ledger) if ledger is not None else None,
                "source_bindings_sha256": sha256_json(self._source_bindings),
                "ledger": ledger,
                "source_bindings": self._source_bindings,
            },
        )

    def _mark_ledger_unavailable(
        self,
        *,
        stage_id: str,
        task: TaskInput,
        baseline_patch: bytes,
        error: Exception | str,
    ) -> None:
        self.ledger_quality = "unavailable"
        self.ledger_error = str(error)
        self._source_bindings = []
        self.artifacts.event(
            f"{self.config.workflow_controller_version.upper()}_SPECIFICATION_UNAVAILABLE",
            task_id=task.task_id,
            error=self.ledger_error,
        )
        self._write_pre_solver_envelope(
            stage_id=stage_id,
            task=task,
            baseline_patch=baseline_patch,
            ledger=None,
        )

    async def prepare_ledger(
        self,
        task: TaskInput,
        *,
        base_worktree: str,
        baseline_patch: bytes = b"",
        stage_id: str | None = None,
        imported_ledger: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Prepare the pre-solver ledger with non-blocking anchor checks.

        A schema/parse failure returns ``None`` and is terminal for the V6.1
        task.  A structurally valid ledger with source-anchor errors is returned
        with ``ledger_quality == "degraded"`` so Solver can still use it.
        """

        self.ledger_quality = "unavailable"
        self.ledger_warnings = []
        self.ledger_error = None
        self._source_bindings = []
        self._ledger_source_binding_errors = []

        task_json = {"title": task.issue_title, "description": task.issue_body}
        managed_delivery = validate_managed_delivery(
            list(task.managed_delivery), task=task_json
        )
        stage_id = stage_id or uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"{self.config.workflow_controller_version}-spec:{task.task_id}:{task.base_commit}:"
            f"{sha256_json(task_json)}:{sha256_bytes(baseline_patch)}:"
            f"{self.config.workflow_prompt_version}:"
            f"{self.config.workflow_controller_version}:"
            f"{self.config.prompt_version}:{self.config.controller_version}",
        ).hex
        ledger_path = self.artifacts.root / "v5" / "ledger.json"
        bindings_path = self.artifacts.root / "v5" / "source-bindings.json"

        if imported_ledger is not None:
            if ledger_path.is_file() and json.loads(
                ledger_path.read_text(encoding="utf-8")
            ) != imported_ledger:
                raise ValueError("V6 imported A ledger changed during this run")
            self._validate_persisted_ledger(imported_ledger)
            ledger = imported_ledger
            try:
                self._source_bindings = await self._bind_ledger_sources(
                    ledger,
                    task=task_json,
                    worktree=base_worktree,
                    base_commit=task.base_commit,
                )
            except Exception as error:  # noqa: BLE001 - preserve degraded source binding behavior
                self._warn_ledger(str(error))
            self.artifacts.write_json("v5/ledger.json", ledger)
        elif ledger_path.is_file():
            try:
                ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
                self._validate_persisted_ledger(ledger)
            except Exception as error:
                self._mark_ledger_unavailable(
                    stage_id=stage_id,
                    task=task,
                    baseline_patch=baseline_patch,
                    error=error,
                )
                return None

            if not bindings_path.is_file():
                self._warn_ledger("persisted V6.1 ledger is missing source bindings")
                self._source_bindings = []
            else:
                try:
                    bindings = json.loads(bindings_path.read_text(encoding="utf-8"))
                except Exception as error:
                    self._warn_ledger(f"source bindings could not be parsed: {error}")
                    bindings = []
                if not isinstance(bindings, list):
                    self._warn_ledger("persisted source bindings are not an array")
                    bindings = []
                try:
                    recomputed = await self._bind_ledger_sources(
                        ledger,
                        task=task_json,
                        worktree=base_worktree,
                        base_commit=task.base_commit,
                    )
                except Exception as error:
                    self._warn_ledger(str(error))
                    self._source_bindings = []
                else:
                    if bindings != recomputed:
                        self._warn_ledger(
                            "persisted V6.1 ledger source bindings differ from the "
                            "current base snapshot"
                        )
                    self._source_bindings = recomputed
        else:
            deadline_at = (
                datetime.now(timezone.utc)
                + timedelta(seconds=self.config.role_timeout_sec)
            ).isoformat()
            try:
                ledger = await self._build_ledger(
                    stage_id,
                    task_json,
                    {
                        "name": task.repository_name,
                        "base_commit": task.base_commit,
                        "base_root": base_worktree,
                    },
                    base_worktree,
                    deadline_at,
                    managed_delivery,
                    baseline_patch,
                    allow_source_binding_errors=True,
                )
            except Exception as error:
                # An unparseable/invalid A object cannot be safely injected into
                # Solver.  End this task before P0 rather than inventing a
                # synthetic ledger or silently changing the treatment arm.
                self._mark_ledger_unavailable(
                    stage_id=stage_id,
                    task=task,
                    baseline_patch=baseline_patch,
                    error=error,
                )
                return None
            self.ledger_warnings.extend(self._ledger_source_binding_errors)
            # Publish a structurally valid ledger even in degraded mode.  The
            # source-binding warning is carried separately in the V6.1 envelope.
            self.artifacts.write_json("v5/ledger.json", ledger)

        self.ledger_quality = "degraded" if self.ledger_warnings else "valid"
        self.artifacts.write_json("v5/source-bindings.json", self._source_bindings)
        self._write_pre_solver_envelope(
            stage_id=stage_id,
            task=task,
            baseline_patch=baseline_patch,
            ledger=ledger,
        )
        return ledger

    async def _load_persisted_ledger_bindings(
        self,
        ledger: dict[str, Any],
        persisted_bindings: Any,
        *,
        task: dict[str, Any],
        base_worktree: str,
        base_commit: str,
    ) -> None:
        """Keep V5 review running when A's anchor integrity is degraded."""
        if not isinstance(persisted_bindings, list):
            self._warn_ledger("persisted source bindings are not an array")
            persisted_bindings = []
        try:
            recomputed_bindings = await self._bind_ledger_sources(
                ledger,
                task=task,
                worktree=base_worktree,
                base_commit=base_commit,
            )
        except Exception as error:
            self._warn_ledger(str(error))
            self._source_bindings = []
            self.ledger_quality = "degraded"
            return
        if persisted_bindings != recomputed_bindings:
            self._warn_ledger(
                "persisted V6.1 ledger source bindings failed integrity check"
            )
            self.ledger_quality = "degraded"
        self._source_bindings = recomputed_bindings
