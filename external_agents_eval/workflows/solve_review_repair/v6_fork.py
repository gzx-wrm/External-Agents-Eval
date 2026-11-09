"""Load frozen V6.1 stage outputs for a new experiment run."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import sha256_bytes, sha256_json

STAGES = ("A", "Solver", "I/R", "I", "R", "S", "Repair", "Verifier")
PREREQUISITES = {
    "A": (),
    "Solver": ("A",),
    "I/R": ("A", "Solver"),
    "I": ("A", "Solver", "R"),
    "R": ("A", "Solver", "I"),
    "S": ("A", "Solver", "I", "R"),
    "Repair": ("A", "Solver", "I", "R", "S"),
    "Verifier": ("P1",),
}


def _same_task_id(recorded: object, full: str) -> bool:
    return recorded == full or recorded == full[:32].rstrip("_-")


@dataclass(frozen=True)
class V6ForkPlan:
    path: Path
    task_id: str
    start_at: str
    sources: dict[str, Path]
    digest: str

    @classmethod
    def load(cls, path: str | Path) -> V6ForkPlan:
        path = Path(path).expanduser().resolve()
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("start_at") not in STAGES:
            raise ValueError("V6 fork plan needs a valid start_at stage")
        task_id = raw.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("V6 fork plan needs task_id")
        default = raw.get("source_run")
        overrides = raw.get("reuse", {})
        if not isinstance(overrides, dict) or any(
            key not in {"A", "Solver", "I", "R", "S", "P1"}
            for key in overrides
        ):
            raise ValueError("V6 fork plan has an invalid reuse stage")
        sources = {}
        for stage in PREREQUISITES[raw["start_at"]]:
            value = overrides.get(stage, default)
            if not isinstance(value, str) or not value:
                raise ValueError(f"V6 fork plan is missing the {stage} source")
            source = Path(value).expanduser()
            sources[stage] = (source if source.is_absolute() else path.parent / source).resolve()
        unused = set(overrides) - set(PREREQUISITES[raw["start_at"]])
        if unused:
            raise ValueError(f"V6 fork plan has unused reuse stages: {sorted(unused)}")
        return cls(path, task_id, raw["start_at"], sources, sha256_json(raw))

    def source(self, stage: str) -> Path:
        return self.sources[stage]


@dataclass(frozen=True)
class V6ForkPlans:
    path: Path
    plans: dict[str, V6ForkPlan]
    digest: str

    @classmethod
    def load(cls, path: str | Path) -> V6ForkPlans:
        path = Path(path).expanduser().resolve()
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or not raw:
            raise ValueError("V6 fork plans must map task IDs to plan paths")
        plans = {}
        for task_id, value in raw.items():
            if not isinstance(task_id, str) or not task_id or not isinstance(value, str) or not value:
                raise ValueError("V6 fork plans must map task IDs to plan paths")
            plan_path = Path(value).expanduser()
            plan = V6ForkPlan.load(plan_path if plan_path.is_absolute() else path.parent / plan_path)
            if plan.task_id != task_id:
                raise ValueError(f"V6 fork plan task_id does not match mapping key: {task_id}")
            plans[task_id] = plan
        return cls(path, plans, sha256_json(raw))


class V6StageSource:
    def __init__(
        self,
        root: Path,
        *,
        task_id: str,
        base_commit: str,
        baseline_sha256: str,
    ) -> None:
        self.root = root
        self.task_id = task_id
        self.base_commit = base_commit
        self.baseline_sha256 = baseline_sha256

    def _json(self, relative: str) -> dict[str, Any]:
        payload = json.loads((self.root / relative).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"V6 source {relative} is not an object")
        return payload

    def _stage(self, name: str) -> dict[str, Any] | None:
        path = self.root / "v6" / "stages" / f"{name}.json"
        if not path.is_file():
            return None
        stage = self._json(f"v6/stages/{name}.json")
        if (
            not _same_task_id(stage.get("task_id"), self.task_id)
            or stage.get("base_commit") != self.base_commit
            or stage.get("baseline_patch_sha256") != self.baseline_sha256
        ):
            raise ValueError(f"V6 {name} source belongs to another task or baseline")
        return stage

    def _run_manifest(self) -> dict[str, Any]:
        run = self._json("manifest.json")
        envelope = self._json("v6/pre-solver-ledger.json")
        if (
            not _same_task_id(envelope.get("task_id"), self.task_id)
            or envelope.get("base_commit") != self.base_commit
            or run.get("base_commit", envelope["base_commit"]) != self.base_commit
            or run.get("baseline_patch_sha256") != self.baseline_sha256
        ):
            raise ValueError("V6 source run belongs to another base or baseline")
        return run

    def ledger(self) -> dict[str, Any]:
        if self.root.is_file():
            return json.loads(self.root.read_text(encoding="utf-8"))
        envelope = self._json("v6/pre-solver-ledger.json")
        if (
            not _same_task_id(envelope.get("task_id"), self.task_id)
            or envelope.get("base_commit") != self.base_commit
            or envelope.get("baseline_patch_sha256") != self.baseline_sha256
        ):
            raise ValueError("V6 A source belongs to another task or baseline")
        ledger = envelope.get("ledger")
        if not isinstance(ledger, dict) or sha256_json(ledger) != envelope.get("ledger_sha256"):
            raise ValueError("V6 A source ledger is missing or changed")
        return ledger

    def patch(self, stage: str, *, ledger_sha256: str | None = None) -> bytes:
        relative = "outputs/v6-p0.patch" if stage == "Solver" else "outputs/v5-p1.patch"
        payload = (self.root / relative).read_bytes()
        stage_record = self._stage(stage)
        if stage_record is not None:
            expected = stage_record.get("patch_sha256")
            if ledger_sha256 is not None and stage_record.get("ledger_sha256") != ledger_sha256:
                raise ValueError("V6 Solver source was generated from another A ledger")
        else:
            run = self._run_manifest()
            expected = run.get("p0_sha256" if stage == "Solver" else "p1_sha256")
            if ledger_sha256 is not None and run.get("ledger_sha256") != ledger_sha256:
                raise ValueError("V6 Solver source was generated from another A ledger")
        if sha256_bytes(payload) != expected:
            raise ValueError(f"V6 {stage} source patch has changed")
        return payload

    def review(self, role: str, *, ledger_sha256: str, p0_sha256: str) -> dict[str, Any]:
        export = self._json(f"v5/exports/review-{role}.json")
        bindings = export.get("bindings", {})
        report = export.get("content")
        if (
            not _same_task_id(export.get("task_id"), self.task_id)
            or export.get("role") != role
            or bindings.get("base_commit") != self.base_commit
            or bindings.get("candidate_sha256") != p0_sha256
            or bindings.get("ledger_sha256") != ledger_sha256
            or not isinstance(report, dict)
            or export.get("content_sha256") != sha256_json(report)
        ):
            raise ValueError(f"V6 {role} source does not match the selected A/P0")
        stage_record = self._stage(role)
        if stage_record is not None:
            if stage_record.get("report_sha256") != sha256_json(report):
                raise ValueError(f"V6 {role} source report has changed")
            artifacts = stage_record.get("review_artifacts", [])
        else:
            self._run_manifest()
            with sqlite3.connect(f"file:{self.root / 'v5/workflow.sqlite'}?mode=ro", uri=True) as db:
                row = db.execute("SELECT snapshot_json FROM freezes LIMIT 1").fetchone()
            if row is None:
                raise ValueError(f"V6 {role} source has no frozen review evidence")
            artifacts = json.loads(row[0])["review_artifacts"]
        return {
            "report": report,
            "review_artifacts": [
                item for item in artifacts
                if item.get("assessment_ref", "").startswith(f"{role}/")
            ],
            "source_root": str(self.root),
        }

    def summary(
        self,
        *,
        ledger_sha256: str,
        p0_sha256: str,
        review_sha256: dict[str, str],
    ) -> dict[str, Any]:
        export = self._json("v5/exports/summary.json")
        bindings = export.get("bindings", {})
        summary = export.get("content")
        if (
            not _same_task_id(export.get("task_id"), self.task_id)
            or bindings.get("base_commit") != self.base_commit
            or bindings.get("candidate_sha256") != p0_sha256
            or bindings.get("ledger_sha256") != ledger_sha256
            or not isinstance(summary, dict)
            or export.get("content_sha256") != sha256_json(summary)
        ):
            raise ValueError("V6 S source does not match the selected A/P0")
        stage_record = self._stage("S")
        if stage_record is not None:
            if (
                stage_record.get("summary_sha256") != sha256_json(summary)
                or stage_record.get("review_sha256") != review_sha256
            ):
                raise ValueError("V6 S source summary has changed")
        else:
            self._run_manifest()
            with sqlite3.connect(f"file:{self.root / 'v5/workflow.sqlite'}?mode=ro", uri=True) as db:
                row = db.execute("SELECT snapshot_json FROM freezes LIMIT 1").fetchone()
            if row is None or {
                role: sha256_json(json.loads(row[0])["reports"][role])
                for role in ("I", "R")
            } != review_sha256:
                raise ValueError("V6 S source used different I/R reports")
        return summary

    def p1_route(self) -> str:
        stage_record = self._stage("P1")
        if stage_record is not None:
            return str(stage_record["route"])
        return str(self._run_manifest()["route"])


def write_stage(
    root: Path,
    *,
    stage: str,
    task_id: str,
    base_commit: str,
    baseline_sha256: str,
    **details: Any,
) -> None:
    path = root / "v6" / "stages" / f"{stage}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "v6.1-stage-1",
        "stage": stage,
        "task_id": task_id,
        "base_commit": base_commit,
        "baseline_patch_sha256": baseline_sha256,
        **details,
    }
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ValueError(f"V6 {stage} output changed during this run")
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
