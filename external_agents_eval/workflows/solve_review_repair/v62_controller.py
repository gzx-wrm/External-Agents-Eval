"""Original A exploration → frozen FT evidence → FT ledger → unchanged V6.1 downstream."""
from __future__ import annotations

import json
from typing import Any

from .artifacts import sha256_bytes, sha256_json
from .ft_evidence import observed_evidence
from .ft_inference import FT_SYSTEM_PROMPT
from .ft_protocol import render_user
from .models import RoleName, TaskInput
from .v5_prompts import spec_prompt
from .v6_controller import V6Controller
from .v62_config import V62Config


class V62Controller(V6Controller):
    def __init__(self, *, ledger_client: Any, **kwargs: Any) -> None:
        kwargs.setdefault("config", V62Config())
        super().__init__(**kwargs)
        self.ledger_client = ledger_client

    async def prepare_ledger(self, task: TaskInput, **kwargs: Any) -> dict[str, Any] | None:
        self._task_id = task.task_id
        ledger = await super().prepare_ledger(task, **kwargs)
        validation_path = self.artifacts.root / "v6.2/ledger/validation.json"
        if ledger is not None and kwargs.get("imported_ledger") is None and validation_path.exists():
            # A citation may exist in Git but be absent from FT's supplied packet.
            # Keep that warning when the already-generated ledger is resumed.
            checks = json.loads(validation_path.read_text(encoding="utf-8"))
            for error in checks["errors"]:
                if error["check"] == "source":
                    self._warn_ledger(json.dumps(error, ensure_ascii=False, sort_keys=True))
            if self.ledger_warnings:
                self.ledger_quality = "degraded"
                envelope_path = self.artifacts.root / "v6/pre-solver-ledger.json"
                envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
                envelope.update(ledger_quality=self.ledger_quality, ledger_warnings=self.ledger_warnings)
                self.artifacts.write_json("v6/pre-solver-ledger.json", envelope)
        return ledger

    async def _build_ledger(
        self,
        run_id: str,
        task: dict[str, Any],
        repository: dict[str, Any],
        worktree: str,
        deadline_at: str,
        managed_delivery: list[dict[str, Any]],
        baseline_patch: bytes = b"",
        *,
        allow_source_binding_errors: bool = False,
    ) -> dict[str, Any]:
        packet_path = self.artifacts.root / "v6.2/evidence.json"
        if packet_path.exists():
            packet = json.loads(packet_path.read_text(encoding="utf-8"))
            if (
                packet["task_id"] != self._task_id
                or packet["task"] != task or packet["base_commit"] != repository["base_commit"]
                or packet["managed_delivery"] != managed_delivery
                or packet["baseline_patch_sha256"] != sha256_bytes(baseline_patch)
            ):
                raise ValueError("V6.2 evidence belongs to another task or baseline")
        else:
            frozen_digest = await self._worktree_digest(
                worktree, repository["base_commit"], baseline_patch=baseline_patch,
            )
            # Keep the historical A prompt, tool set and first exploration unchanged.
            # Its final draft is archived by the role runner but never supplied to FT.
            search = await self._run_role(
                role=RoleName.V5_SPECIFICATION_REVIEWER,
                prompt=spec_prompt(task, repository, managed_delivery=managed_delivery),
                worktree=worktree, prefix="v6.2/evidence-search", session_role="A",
                invocation_id=f"{run_id}.A.1",
                timeout_sec=min(self.config.role_timeout_sec, self._deadline_remaining(deadline_at)),
            )
            await self._require_worktree_unchanged(
                worktree, repository["base_commit"], frozen_digest,
                role="V6.2 evidence search", baseline_patch=baseline_patch,
            )
            self._require_success(search, "V6.2 evidence search")
            trajectory = json.loads(search.trajectory_path.read_text(encoding="utf-8"))
            packet = {
                "schema_version": "v6.2-ft-evidence-1", "task_id": self._task_id,
                "task": task, "managed_delivery": managed_delivery,
                "base_commit": repository["base_commit"],
                "baseline_patch_sha256": sha256_bytes(baseline_patch),
                "search_trajectory_sha256": sha256_bytes(search.trajectory_path.read_bytes()),
                **await observed_evidence(
                    trajectory, task_id=self._task_id, base_root=worktree,
                    base_commit=repository["base_commit"], workspace=self.workspace,
                ),
            }
            self.artifacts.write_json("v6.2/evidence.json", packet)
            self.artifacts.event("V6.2_EVIDENCE_FROZEN", evidence_sha256=sha256_json(packet))

        # Historical FT parsed sorted SPEC INPUT: description preceded title.
        # Use that same order for both fresh and resumed requests.
        packet.update(task={"description": task["description"], "title": task["title"]},
                      managed_delivery=managed_delivery)
        bash_audit = packet.get("bash_evidence_audit")
        if bash_audit is not None:
            self.artifacts.write_json("v6.2/bash-evidence-audit.json", bash_audit)
            if bash_audit["errors"]:
                calls = [item["tool_call_id"] for item in bash_audit["errors"]]
                raise ValueError(f"Bash evidence mapping is incomplete for {calls}; inspect the evidence audit")
            if not packet["evidence"]:
                raise ValueError("Bash search returned no verified frozen source evidence")
        messages = [
            {"role": "system", "content": FT_SYSTEM_PROMPT},
            {"role": "user", "content": render_user(packet, packet["evidence"])},
        ]
        sample = {**packet, "messages": messages, "messages_sha256": sha256_json(messages)}
        payload, checks = await self.ledger_client.generate(sample, artifacts=self.artifacts)
        ledger = {
            "summary": payload["summary"],
            "obligations": self._assign_targets(payload["obligations"], "O"),
            "preservation_obligations": self._assign_targets(payload["preservation_obligations"], "P"),
            "open_questions": self._assign_questions(payload["open_questions"]),
        }
        self._ledger_source_binding_errors = [
            json.dumps(error, ensure_ascii=False, sort_keys=True)
            for error in checks["errors"] if error["check"] == "source"
        ]
        try:
            self._source_bindings = await self._bind_ledger_sources(
                ledger, task=task, worktree=worktree, base_commit=repository["base_commit"],
            )
        except Exception as error:
            if not allow_source_binding_errors:
                raise
            self._ledger_source_binding_errors.append(str(error))
            self._source_bindings = []
        self.artifacts.write_json("v5/source-bindings.json", self._source_bindings)
        self.artifacts.event(
            "V6.2_FT_LEDGER_READY", ledger_sha256=sha256_json(ledger),
            evidence_sha256=sha256_json(packet), messages_sha256=sample["messages_sha256"],
        )
        return ledger
