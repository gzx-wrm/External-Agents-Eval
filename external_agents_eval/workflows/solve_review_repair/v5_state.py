"""Durable V5 state transitions and immutable review artifacts.

The state store is intentionally small: one controller owns writes, while Pi
and its tools only produce files.  A batch changes business state only inside a
single SQLite transaction after its immutable bundle has been staged.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .artifacts import canonical_json_bytes, sha256_bytes, sha256_json
from .v5_contracts import V5ContractError, validate_review_batch


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_dir(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class V5Bundle:
    path: Path
    content_sha256: str
    note_hashes: dict[str, str]
    manifest: dict[str, Any]


class ImmutableBundleStore:
    """Publish a staged bundle without ever overwriting an existing digest."""

    def __init__(
        self, root: str | Path, *, secrets: tuple[str, ...] = ()
    ) -> None:
        self.root = Path(root).resolve()
        self.secrets = tuple(
            secret.encode("utf-8") for secret in secrets if secret
        )
        self.root.mkdir(parents=True, exist_ok=True)
        self.staging = self.root / "staging"
        self.staging.mkdir(exist_ok=True)
        self.bundles = self.root / "bundles"
        self.bundles.mkdir(exist_ok=True)

    def stage(
        self,
        *,
        batch_key: str,
        normalized_json: dict[str, Any],
        raw_output: str,
        notes: dict[str, bytes],
        trace_manifest: dict[str, Any],
        materials_manifest: dict[str, Any],
        extra_files: dict[str, bytes] | None = None,
    ) -> V5Bundle:
        if not batch_key or "/" in batch_key or ".." in batch_key:
            raise V5ContractError("invalid batch key")
        stage_dir = Path(tempfile.mkdtemp(prefix=f"{batch_key}-", dir=self.staging))
        note_hashes: dict[str, str] = {}
        try:
            self._write(stage_dir / "response.normalized.json", canonical_json_bytes(normalized_json) + b"\n")
            self._write(stage_dir / "response.raw.txt", raw_output.encode("utf-8"))
            notes_dir = stage_dir / "notes"
            for relative, content in sorted(notes.items()):
                relative_path = Path(relative)
                if relative_path.is_absolute() or ".." in relative_path.parts:
                    raise V5ContractError(f"invalid archived note path: {relative}")
                target = stage_dir / relative_path if relative_path.parts[0] == "notes" else notes_dir / relative_path
                self._write(target, content)
                note_hashes[relative] = sha256_bytes(content)
            self._write(stage_dir / "trace-manifest.json", canonical_json_bytes(trace_manifest) + b"\n")
            self._write(stage_dir / "materials-manifest.json", canonical_json_bytes(materials_manifest) + b"\n")
            extra_hashes: dict[str, str] = {}
            for relative, content in sorted((extra_files or {}).items()):
                relative_path = Path(relative)
                if (
                    relative_path.is_absolute()
                    or ".." in relative_path.parts
                    or not relative_path.parts
                ):
                    raise V5ContractError(f"invalid supplemental artifact path: {relative}")
                target = stage_dir / relative_path
                if target.exists():
                    raise V5ContractError(f"supplemental artifact collides with bundle file: {relative}")
                self._write(target, content)
                extra_hashes[relative] = sha256_bytes(content)
            manifest = {
                "schema_version": "v5-bundle-1",
                "batch_key": batch_key,
                "files": {
                    "response.normalized.json": sha256_bytes(
                        (stage_dir / "response.normalized.json").read_bytes()
                    ),
                    "response.raw.txt": sha256_bytes((stage_dir / "response.raw.txt").read_bytes()),
                    "trace-manifest.json": sha256_bytes((stage_dir / "trace-manifest.json").read_bytes()),
                    "materials-manifest.json": sha256_bytes(
                        (stage_dir / "materials-manifest.json").read_bytes()
                    ),
                },
                "note_hashes": note_hashes,
                "extra_hashes": extra_hashes,
            }
            content_sha = sha256_json(manifest)
            self._write(stage_dir / "bundle-manifest.json", canonical_json_bytes(manifest) + b"\n")
            for path in stage_dir.rglob("*"):
                if path.is_file():
                    _fsync_file(path)
            _fsync_dir(stage_dir)
            destination = self.bundles / content_sha
            if destination.exists():
                if not (destination / "bundle-manifest.json").is_file():
                    raise V5ContractError(f"existing bundle is incomplete: {content_sha}")
                existing = json.loads((destination / "bundle-manifest.json").read_text(encoding="utf-8"))
                if existing != manifest:
                    raise V5ContractError(f"bundle digest collision: {content_sha}")
                self._verify_manifest_files(destination, existing)
                self._remove_tree(stage_dir)
            else:
                os.replace(stage_dir, destination)
                _fsync_dir(self.bundles)
            return V5Bundle(destination, content_sha, note_hashes, manifest)
        except Exception:
            if stage_dir.exists():
                self._remove_tree(stage_dir)
            raise

    def _write(self, path: Path, content: bytes) -> None:
        if any(secret in content for secret in self.secrets):
            raise V5ContractError(
                f"refusing to persist an unredacted secret in {path.name}"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    @staticmethod
    def _remove_tree(path: Path) -> None:
        for child in sorted(path.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
        path.rmdir()

    @staticmethod
    def _verify_manifest_files(root: Path, manifest: dict[str, Any]) -> None:
        hashes = {
            **manifest.get("files", {}),
            **manifest.get("note_hashes", {}),
            **manifest.get("extra_hashes", {}),
        }
        for relative, expected in hashes.items():
            path = root / relative
            if (
                not path.is_file()
                or path.is_symlink()
                or sha256_bytes(path.read_bytes()) != expected
            ):
                raise V5ContractError(f"existing bundle file is corrupt: {relative}")


class V5StateStore:
    """SQLite authority for V5 lanes, attempts, commits and summary freeze."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, isolation_level=None, timeout=5)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=DELETE")
        self.db.execute("PRAGMA synchronous=EXTRA")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA busy_timeout=5000")
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs(
              run_id TEXT PRIMARY KEY, binding_sha256 TEXT NOT NULL,
              binding_json TEXT NOT NULL, phase TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS lanes(
              run_id TEXT NOT NULL REFERENCES runs(run_id), role TEXT NOT NULL,
              revision INTEGER NOT NULL, phase TEXT NOT NULL, state_json TEXT NOT NULL,
              PRIMARY KEY(run_id, role)
            );
            CREATE TABLE IF NOT EXISTS attempts(
              run_id TEXT NOT NULL, role TEXT NOT NULL, logical_round INTEGER NOT NULL,
              attempt INTEGER NOT NULL, expected_revision INTEGER NOT NULL,
              status TEXT NOT NULL, input_sha256 TEXT NOT NULL, bundle_sha256 TEXT,
              content_sha256 TEXT, receipt_json TEXT, reason TEXT,
              PRIMARY KEY(run_id, role, logical_round, attempt),
              FOREIGN KEY(run_id, role) REFERENCES lanes(run_id, role)
            );
            CREATE TABLE IF NOT EXISTS commits(
              commit_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, role TEXT NOT NULL,
              logical_round INTEGER NOT NULL, attempt INTEGER NOT NULL,
              batch_key TEXT NOT NULL UNIQUE, content_sha256 TEXT NOT NULL,
              receipt_json TEXT NOT NULL, created_at TEXT NOT NULL,
              FOREIGN KEY(run_id, role) REFERENCES lanes(run_id, role)
            );
            CREATE TABLE IF NOT EXISTS invocations(
              invocation_id TEXT PRIMARY KEY, run_id TEXT NOT NULL,
              role TEXT NOT NULL, prompt_sha256 TEXT NOT NULL,
              worktree TEXT NOT NULL, status TEXT NOT NULL,
              started_at TEXT NOT NULL, finished_at TEXT,
              result_json TEXT,
              lease_owner TEXT, lease_expires_at TEXT, heartbeat_at TEXT,
              launcher_json TEXT, recovery_reason TEXT,
              FOREIGN KEY(run_id) REFERENCES runs(run_id)
            );
            CREATE TABLE IF NOT EXISTS freezes(
              freeze_id TEXT PRIMARY KEY, run_id TEXT NOT NULL UNIQUE,
              snapshot_json TEXT NOT NULL, created_at TEXT NOT NULL,
              FOREIGN KEY(run_id) REFERENCES runs(run_id)
            );
            CREATE TABLE IF NOT EXISTS stage_deadlines(
              run_id TEXT NOT NULL REFERENCES runs(run_id), stage TEXT NOT NULL,
              duration_sec REAL NOT NULL, deadline_at TEXT NOT NULL,
              PRIMARY KEY(run_id, stage)
            );
            CREATE TABLE IF NOT EXISTS deliveries(
              run_id TEXT PRIMARY KEY REFERENCES runs(run_id),
              freeze_id TEXT NOT NULL, route TEXT NOT NULL,
              repair_invocation_id TEXT, status TEXT NOT NULL,
              p1_sha256 TEXT, result_json TEXT,
              started_at TEXT NOT NULL, finished_at TEXT
            );
            """
        )
        attempt_columns = {
            row["name"]
            for row in self.db.execute("PRAGMA table_info(attempts)").fetchall()
        }
        if "input_payload_json" not in attempt_columns:
            self.db.execute("ALTER TABLE attempts ADD COLUMN input_payload_json TEXT")
        invocation_columns = {
            row["name"]
            for row in self.db.execute("PRAGMA table_info(invocations)").fetchall()
        }
        # These columns are appended through idempotent migrations so an
        # interrupted V5 run can be resumed with the same SQLite file.
        for name, definition in (
            ("lease_owner", "TEXT"),
            ("lease_expires_at", "TEXT"),
            ("heartbeat_at", "TEXT"),
            ("launcher_json", "TEXT"),
            ("recovery_reason", "TEXT"),
        ):
            if name not in invocation_columns:
                self.db.execute(f"ALTER TABLE invocations ADD COLUMN {name} {definition}")

    def close(self) -> None:
        self.db.close()

    def create_run(
        self,
        run_id: str,
        binding: dict[str, Any],
        *,
        roles: tuple[str, ...] = ("I", "R"),
    ) -> bool:
        binding_bytes = canonical_json_bytes(binding)
        binding_sha = sha256_bytes(binding_bytes)
        with self.db:
            existing = self.db.execute(
                "SELECT binding_sha256 FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()
            if existing is not None:
                if existing["binding_sha256"] != binding_sha:
                    raise V5ContractError("run ID is already bound to different inputs")
                return False
            self.db.execute(
                "INSERT INTO runs VALUES (?, ?, ?, 'running', ?)",
                (run_id, binding_sha, binding_bytes.decode("utf-8"), _now()),
            )
            for role in roles:
                initial = self._initial_lane(role)
                self.db.execute(
                    "INSERT INTO lanes VALUES (?, ?, 0, 'running', ?)",
                    (run_id, role, json.dumps(initial, sort_keys=True)),
                )
        return True

    def run_record(self, run_id: str) -> dict[str, Any]:
        row = self.db.execute(
            "SELECT binding_json, binding_sha256, phase, created_at FROM runs WHERE run_id=?",
            (run_id,),
        ).fetchone()
        if row is None:
            raise KeyError(run_id)
        return {
            "binding": json.loads(row["binding_json"]),
            "binding_sha256": row["binding_sha256"],
            "phase": row["phase"],
            "created_at": row["created_at"],
        }

    def stage_deadline(
        self, run_id: str, stage: str, *, duration_sec: float
    ) -> str:
        if stage not in {"A", "S"} or duration_sec <= 0:
            raise ValueError("invalid V5 stage deadline")
        with self.db:
            existing = self.db.execute(
                "SELECT duration_sec, deadline_at FROM stage_deadlines "
                "WHERE run_id=? AND stage=?",
                (run_id, stage),
            ).fetchone()
            if existing is not None:
                if existing["duration_sec"] != duration_sec:
                    raise V5ContractError(
                        f"V5 {stage} deadline was initialized with another duration"
                    )
                return existing["deadline_at"]
            deadline_at = (
                datetime.now(timezone.utc) + timedelta(seconds=duration_sec)
            ).isoformat()
            self.db.execute(
                "INSERT INTO stage_deadlines VALUES (?, ?, ?, ?)",
                (run_id, stage, duration_sec, deadline_at),
            )
            return deadline_at

    def initialize_lanes(
        self,
        run_id: str,
        *,
        implementation_targets: list[str],
        regression_targets: list[str],
        optional_regression_targets: list[str] = (),
        target_catalog: dict[str, dict[str, Any]] | None = None,
        batch_size: int = 3,
        lane_deadline_sec: float = 45 * 60,
    ) -> None:
        if batch_size < 1 or lane_deadline_sec <= 0:
            raise ValueError("batch_size and lane_deadline_sec must be positive")
        values = {
            "I": implementation_targets,
            "R": regression_targets,
        }
        with self.db:
            for role, targets in values.items():
                state = self.lane(run_id, role)
                expected_enrolled = list(dict.fromkeys(targets))
                expected_optional = (
                    list(dict.fromkeys(optional_regression_targets)) if role == "R" else []
                )
                if state.get("initialized"):
                    if (
                        state.get("initial_enrolled") != expected_enrolled
                        or state.get("initial_optional") != expected_optional
                        or state["batch_size"] != batch_size
                        or state.get("lane_deadline_sec") != lane_deadline_sec
                    ):
                        raise V5ContractError("cannot reinitialize a V5 lane with different inputs")
                    continue
                state.update(
                    {
                        "initialized": True,
                        "initial_enrolled": expected_enrolled,
                        "initial_optional": expected_optional,
                        "batch_size": batch_size,
                        "lane_started_at": _now(),
                        "lane_deadline_sec": lane_deadline_sec,
                        "deadline_at": (
                            datetime.now(timezone.utc)
                            + timedelta(seconds=lane_deadline_sec)
                        ).isoformat(),
                        "enrolled": expected_enrolled,
                        "optional": expected_optional,
                        "targets": {
                            target_id: dict((target_catalog or {}).get(target_id, {}))
                            for target_id in dict.fromkeys(
                                [
                                    *targets,
                                    *(optional_regression_targets if role == "R" else ()),
                                ]
                            )
                        },
                        "effective": {},
                        "history": {},
                        "revisable": [],
                        "reopened_target_ids": [],
                        "pending": list(dict.fromkeys(targets)),
                    }
                )
                state.pop("revision", None)
                state.pop("phase", None)
                self.db.execute(
                    "UPDATE lanes SET phase='running', state_json=? WHERE run_id=? AND role=?",
                    (json.dumps(state, sort_keys=True), run_id, role),
                )

    @staticmethod
    def _initial_lane(role: str) -> dict[str, Any]:
        target = "O001" if role == "I" else "P001"
        return {
            "phase": "running", "round": 1, "attempt": 1, "batch_size": 3,
            "initialized": False,
            "lane_started_at": None, "lane_deadline_sec": None, "deadline_at": None,
            "enrolled": [target], "optional": ["O001"] if role == "R" else [],
            "targets": {},
            "effective": {}, "history": {}, "revisable": [],
            "reopened_target_ids": [], "pending": [target],
            "next_target_number": 1, "no_progress": 0,
            "observation_fingerprints": [],
        }

    def lane(self, run_id: str, role: str) -> dict[str, Any]:
        row = self.db.execute(
            "SELECT revision, phase, state_json FROM lanes WHERE run_id=? AND role=?", (run_id, role)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown lane {run_id}/{role}")
        state = json.loads(row["state_json"])
        state["revision"] = row["revision"]
        state["phase"] = row["phase"]
        return state

    def next_attempt(self, run_id: str, role: str, logical_round: int) -> int:
        row = self.db.execute(
            "SELECT MAX(attempt) AS maximum FROM attempts "
            "WHERE run_id=? AND role=? AND logical_round=?",
            (run_id, role, logical_round),
        ).fetchone()
        if row is None or row["maximum"] is None:
            return 1
        latest = self.db.execute(
            "SELECT attempt, status FROM attempts "
            "WHERE run_id=? AND role=? AND logical_round=? "
            "ORDER BY attempt DESC LIMIT 1",
            (run_id, role, logical_round),
        ).fetchone()
        assert latest is not None
        return latest["attempt"] if latest["status"] == "PREPARED" else latest["attempt"] + 1

    def attempt_record(
        self, run_id: str, role: str, logical_round: int, attempt: int
    ) -> dict[str, Any] | None:
        row = self.db.execute(
            "SELECT * FROM attempts WHERE run_id=? AND role=? "
            "AND logical_round=? AND attempt=?",
            (run_id, role, logical_round, attempt),
        ).fetchone()
        if row is None:
            return None
        return {
            **dict(row),
            "input_payload": (
                json.loads(row["input_payload_json"])
                if row["input_payload_json"]
                else None
            ),
            "receipt": (
                json.loads(row["receipt_json"]) if row["receipt_json"] else None
            ),
        }

    def prepare_invocation(
        self,
        invocation_id: str,
        *,
        run_id: str,
        role: str,
        prompt_sha256: str,
        worktree: str,
        lease_owner: str | None = None,
        lease_ttl_sec: float = 15 * 60,
    ) -> bool:
        if lease_ttl_sec <= 0:
            raise ValueError("invocation lease TTL must be positive")
        now = datetime.now(timezone.utc)
        lease_expires_at = (
            (now + timedelta(seconds=lease_ttl_sec)).isoformat()
            if lease_owner
            else None
        )
        with self.db:
            existing = self.db.execute(
                "SELECT run_id, role, prompt_sha256, worktree, status "
                "FROM invocations WHERE invocation_id=?",
                (invocation_id,),
            ).fetchone()
            if existing is not None:
                identity = (
                    existing["run_id"],
                    existing["role"],
                    existing["prompt_sha256"],
                    existing["worktree"],
                )
                expected = (run_id, role, prompt_sha256, worktree)
                if identity != expected:
                    raise V5ContractError(
                        "invocation ID is already bound to different execution inputs"
                    )
                return False
            self.db.execute(
                "INSERT INTO invocations("
                "invocation_id, run_id, role, prompt_sha256, worktree, status, "
                "started_at, finished_at, result_json, lease_owner, "
                "lease_expires_at, heartbeat_at, launcher_json, recovery_reason "
                ") VALUES (?, ?, ?, ?, ?, 'PREPARED', ?, NULL, NULL, ?, ?, ?, NULL, NULL)",
                (
                    invocation_id,
                    run_id,
                    role,
                    prompt_sha256,
                    worktree,
                    _now(),
                    lease_owner,
                    lease_expires_at,
                    _now() if lease_owner else None,
                ),
            )
        return True

    def start_invocation(
        self,
        invocation_id: str,
        *,
        lease_owner: str,
        launcher: dict[str, Any] | None = None,
        lease_ttl_sec: float = 15 * 60,
    ) -> bool:
        """Claim a prepared invocation without ever creating a duplicate.

        ``False`` means another live owner already holds the lease.  An
        expired owner is moved to EXPIRED before the new owner claims it;
        terminal invocations remain immutable and return ``False``.
        """
        if not lease_owner or lease_ttl_sec <= 0:
            raise ValueError("invocation lease owner and TTL are required")
        now = datetime.now(timezone.utc)
        expires = (now + timedelta(seconds=lease_ttl_sec)).isoformat()
        with self.db:
            row = self.db.execute(
                "SELECT status, lease_owner, lease_expires_at FROM invocations "
                "WHERE invocation_id=?", (invocation_id,)
            ).fetchone()
            if row is None:
                raise KeyError(invocation_id)
            if row["status"] in {"COMPLETED", "FAILED"}:
                return False
            current_expiry = row["lease_expires_at"]
            try:
                expired = bool(
                    current_expiry
                    and datetime.fromisoformat(current_expiry) <= now
                )
            except ValueError:
                expired = True
            if row["status"] == "RUNNING" and row["lease_owner"] not in {None, lease_owner} and not expired:
                return False
            if expired and row["status"] == "RUNNING":
                self.db.execute(
                    "UPDATE invocations SET status='EXPIRED', recovery_reason=? "
                    "WHERE invocation_id=?",
                    ("lease_expired_before_recovery", invocation_id),
                )
            self.db.execute(
                "UPDATE invocations SET status='RUNNING', lease_owner=?, "
                "lease_expires_at=?, heartbeat_at=?, launcher_json=?, "
                "recovery_reason=NULL WHERE invocation_id=?",
                (
                    lease_owner,
                    expires,
                    now.isoformat(),
                    json.dumps(launcher, sort_keys=True) if launcher is not None else None,
                    invocation_id,
                ),
            )
        return True

    def heartbeat_invocation(
        self, invocation_id: str, *, lease_owner: str, lease_ttl_sec: float = 15 * 60
    ) -> bool:
        if not lease_owner or lease_ttl_sec <= 0:
            raise ValueError("invocation lease owner and TTL are required")
        now = datetime.now(timezone.utc)
        expires = (now + timedelta(seconds=lease_ttl_sec)).isoformat()
        with self.db:
            updated = self.db.execute(
                "UPDATE invocations SET heartbeat_at=?, lease_expires_at=? "
                "WHERE invocation_id=? AND status='RUNNING' AND lease_owner=?",
                (now.isoformat(), expires, invocation_id, lease_owner),
            ).rowcount
        return bool(updated)

    def expire_invocation(self, invocation_id: str, *, reason: str) -> bool:
        if not reason:
            raise ValueError("invocation expiry requires a reason")
        with self.db:
            updated = self.db.execute(
                "UPDATE invocations SET status='EXPIRED', finished_at=?, "
                "recovery_reason=? WHERE invocation_id=? "
                "AND status IN ('PREPARED','RUNNING','RECEIVED','STAGED')",
                (_now(), reason, invocation_id),
            ).rowcount
        return bool(updated)

    def record_invocation_stage(
        self,
        invocation_id: str,
        *,
        status: str,
        result: dict[str, Any] | None = None,
    ) -> None:
        """Advance the non-terminal receive/stage markers monotonically."""
        if status not in {"RECEIVED", "STAGED"}:
            raise ValueError(f"invalid invocation stage: {status}")
        order = {"PREPARED": 0, "RUNNING": 1, "RECEIVED": 2, "STAGED": 3}
        with self.db:
            row = self.db.execute(
                "SELECT status FROM invocations WHERE invocation_id=?", (invocation_id,)
            ).fetchone()
            if row is None:
                raise KeyError(invocation_id)
            current = row["status"]
            if current in {"COMPLETED", "FAILED", "INTERRUPTED", "EXPIRED"}:
                if current != status:
                    return
            elif order.get(status, -1) < order.get(current, -1):
                raise V5ContractError(
                    f"invocation stage cannot move backwards from {current} to {status}"
                )
            self.db.execute(
                "UPDATE invocations SET status=?, result_json=COALESCE(?, result_json), "
                "heartbeat_at=? WHERE invocation_id=?",
                (
                    status,
                    json.dumps(result, sort_keys=True) if result is not None else None,
                    _now(),
                    invocation_id,
                ),
            )

    def invocation_recovery_decision(
        self, invocation_id: str, *, now: datetime | None = None
    ) -> dict[str, Any]:
        """Return a stable, inspectable recovery classification."""
        record = self.invocation(invocation_id)
        if record is None:
            raise KeyError(invocation_id)
        current = now or datetime.now(timezone.utc)
        expiry = record.get("lease_expires_at")
        expired = False
        if expiry:
            try:
                expired = datetime.fromisoformat(expiry) <= current
            except ValueError:
                expired = True
        status = record["status"]
        if status in {"COMPLETED", "FAILED"}:
            action = "replay_terminal"
        elif status in {"RUNNING", "RECEIVED", "STAGED"} and not expired:
            action = "wait_or_attach"
        elif status in {"PREPARED", "EXPIRED"} or expired:
            action = "claim_or_new_attempt"
        else:
            action = "inspect"
        return {
            "invocation_id": invocation_id,
            "status": status,
            "lease_expired": expired,
            "action": action,
            "recovery_reason": record.get("recovery_reason"),
        }

    def finish_invocation(
        self,
        invocation_id: str,
        *,
        status: str,
        result: dict[str, Any],
    ) -> None:
        if status not in {"COMPLETED", "FAILED", "INTERRUPTED"}:
            raise ValueError(f"Invalid invocation status: {status}")
        with self.db:
            row = self.db.execute(
                "SELECT status, result_json FROM invocations WHERE invocation_id=?",
                (invocation_id,),
            ).fetchone()
            if row is None:
                raise KeyError(invocation_id)
            if row["status"] == "EXPIRED":
                raise V5ContractError(
                    "expired invocation cannot be completed; create a new attempt"
                )
            normalized = json.dumps(result, sort_keys=True)
            if row["status"] in {"COMPLETED", "FAILED", "INTERRUPTED"}:
                prior = json.loads(row["result_json"] or "{}")
                stable_fields = ("exit_code", "timed_out", "exception")
                if row["status"] != status or any(
                    prior.get(field) != result.get(field) for field in stable_fields
                ):
                    raise V5ContractError(
                        "completed invocation was replayed with a different result"
                    )
                return
            self.db.execute(
                "UPDATE invocations SET status=?, finished_at=?, result_json=?, "
                "lease_owner=NULL, lease_expires_at=NULL, heartbeat_at=NULL "
                "WHERE invocation_id=?",
                (status, _now(), normalized, invocation_id),
            )

    def invocation(self, invocation_id: str) -> dict[str, Any] | None:
        row = self.db.execute(
            "SELECT * FROM invocations WHERE invocation_id=?", (invocation_id,)
        ).fetchone()
        if row is None:
            return None
        return {
            **dict(row),
            "result": json.loads(row["result_json"]) if row["result_json"] else None,
        }

    def prepare_delivery(
        self,
        run_id: str,
        *,
        freeze_id: str,
        route: str,
        repair_invocation_id: str | None,
    ) -> dict[str, Any]:
        if route not in {"accept", "repair", "hold"}:
            raise ValueError(f"invalid V5 delivery route: {route}")
        if (route == "repair") != (repair_invocation_id is not None):
            raise ValueError("repair delivery must bind exactly one invocation")
        with self.db:
            existing = self.db.execute(
                "SELECT * FROM deliveries WHERE run_id=?", (run_id,)
            ).fetchone()
            if existing is not None:
                identity = (
                    existing["freeze_id"],
                    existing["route"],
                    existing["repair_invocation_id"],
                )
                if identity != (freeze_id, route, repair_invocation_id):
                    raise V5ContractError(
                        "V5 delivery was prepared with different inputs"
                    )
                return self._delivery_row(existing)
            self.db.execute(
                "INSERT INTO deliveries VALUES (?, ?, ?, ?, 'PREPARED', NULL, NULL, ?, NULL)",
                (run_id, freeze_id, route, repair_invocation_id, _now()),
            )
        delivery = self.delivery(run_id)
        assert delivery is not None
        return delivery

    def finish_delivery(
        self,
        run_id: str,
        *,
        p1_sha256: str,
        result: dict[str, Any],
    ) -> None:
        if not isinstance(p1_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", p1_sha256
        ):
            raise ValueError("invalid V5 P1 digest")
        normalized = json.dumps(result, sort_keys=True)
        with self.db:
            row = self.db.execute(
                "SELECT * FROM deliveries WHERE run_id=?", (run_id,)
            ).fetchone()
            if row is None:
                raise KeyError(run_id)
            if row["status"] == "COMPLETED":
                if (
                    row["p1_sha256"] != p1_sha256
                    or json.loads(row["result_json"] or "{}") != result
                ):
                    raise V5ContractError(
                        "completed V5 delivery was replayed with different output"
                    )
                return
            if row["status"] != "PREPARED":
                raise V5ContractError(
                    f"cannot finish V5 delivery in state {row['status']}"
                )
            self.db.execute(
                "UPDATE deliveries SET status='COMPLETED', p1_sha256=?, "
                "result_json=?, finished_at=? WHERE run_id=?",
                (p1_sha256, normalized, _now(), run_id),
            )
            self.db.execute(
                "UPDATE runs SET phase='completed' WHERE run_id=?", (run_id,)
            )

    def delivery(self, run_id: str) -> dict[str, Any] | None:
        row = self.db.execute(
            "SELECT * FROM deliveries WHERE run_id=?", (run_id,)
        ).fetchone()
        return self._delivery_row(row) if row is not None else None

    @staticmethod
    def _delivery_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            **dict(row),
            "result": json.loads(row["result_json"]) if row["result_json"] else None,
        }

    def prepare_attempt(
        self, run_id: str, role: str, logical_round: int, attempt: int, *, input_payload: dict[str, Any]
    ) -> str:
        state = self.lane(run_id, role)
        if state["phase"] != "running":
            raise V5ContractError("lane is not running")
        input_sha = sha256_json(input_payload)
        with self.db:
            existing = self.db.execute(
                "SELECT input_sha256, input_payload_json, status FROM attempts "
                "WHERE run_id=? AND role=? AND logical_round=? AND attempt=?",
                (run_id, role, logical_round, attempt),
            ).fetchone()
            if existing is not None:
                if existing["input_sha256"] != input_sha:
                    raise V5ContractError("attempt identity was prepared with different input")
                if (
                    existing["input_payload_json"] is not None
                    and json.loads(existing["input_payload_json"]) != input_payload
                ):
                    raise V5ContractError(
                        "attempt input payload conflicts with its stored digest"
                    )
                if existing["status"] in {"PREPARED", "STAGED", "RECEIVED", "COMMITTED"}:
                    return input_sha
                raise V5ContractError(
                    f"attempt {run_id}/{role}/{logical_round}/{attempt} is already "
                    f"{existing['status']}"
                )
            self.db.execute(
                "INSERT INTO attempts("
                "run_id, role, logical_round, attempt, expected_revision, status, "
                "input_sha256, bundle_sha256, content_sha256, receipt_json, reason, "
                "input_payload_json"
                ") VALUES (?, ?, ?, ?, ?, 'PREPARED', ?, NULL, NULL, NULL, NULL, ?)",
                (
                    run_id,
                    role,
                    logical_round,
                    attempt,
                    state["revision"],
                    input_sha,
                    json.dumps(input_payload, ensure_ascii=False, sort_keys=True),
                ),
            )
        return input_sha

    def reject_attempt(self, run_id: str, role: str, logical_round: int, attempt: int, reason: str) -> None:
        with self.db:
            self.db.execute(
                "UPDATE attempts SET status='REJECTED', reason=? WHERE run_id=? AND role=? AND logical_round=? AND attempt=?",
                (reason, run_id, role, logical_round, attempt),
            )

    def commit_batch(
        self,
        run_id: str,
        role: str,
        logical_round: int,
        attempt: int,
        payload: dict[str, Any],
        *,
        output_root: str | Path,
        bundle: V5Bundle,
        expected_revision: int | None = None,
        observation_fingerprint: str | None = None,
    ) -> dict[str, Any]:
        batch_key = f"{run_id}:{role}:{logical_round}:{attempt}"
        content_sha = sha256_json({"payload": payload, "bundle": bundle.content_sha256})
        self.db.execute("BEGIN IMMEDIATE")
        try:
            prior = self.db.execute(
                "SELECT content_sha256, receipt_json FROM commits WHERE batch_key=?", (batch_key,)
            ).fetchone()
            if prior is not None:
                if prior["content_sha256"] != content_sha:
                    raise V5ContractError("same batch identity was submitted with different content")
                self.db.execute("COMMIT")
                return json.loads(prior["receipt_json"])
            lane_row = self.db.execute(
                "SELECT revision, phase, state_json FROM lanes WHERE run_id=? AND role=?", (run_id, role)
            ).fetchone()
            if lane_row is None or lane_row["phase"] != "running":
                raise V5ContractError("lane is not running")
            state = json.loads(lane_row["state_json"])
            state["phase"] = lane_row["phase"]
            revision = lane_row["revision"]
            if expected_revision is not None and revision != expected_revision:
                raise V5ContractError("stale lane revision")
            attempt_row = self.db.execute(
                "SELECT expected_revision, status FROM attempts WHERE run_id=? AND role=? AND logical_round=? AND attempt=?",
                (run_id, role, logical_round, attempt),
            ).fetchone()
            if attempt_row is None or attempt_row["status"] not in {"PREPARED", "STAGED"}:
                raise V5ContractError("attempt is not prepared for commit")
            if attempt_row["expected_revision"] != revision:
                raise V5ContractError("attempt was prepared against a stale revision")
            validate_review_batch(
                payload,
                role=role,
                enrolled_target_ids=set(state["enrolled"]),
                effective_target_ids=set(state["effective"]),
                revisable_target_ids=set(state["revisable"]),
                optional_target_ids=set(state["optional"]),
                output_root=output_root,
                batch_size=state["batch_size"],
            )
            commit_id = uuid.uuid4().hex
            new_state, assigned = self._transition(
                state,
                payload,
                role,
                bundle_sha256=bundle.content_sha256,
                commit_id=commit_id,
                observation_fingerprint=observation_fingerprint,
            )
            receipt = {
                "commit_id": commit_id,
                "batch_key": batch_key,
                "role": role,
                "logical_round": logical_round,
                "attempt": attempt,
                "content_sha256": content_sha,
                "assigned_target_ids": assigned,
                "state_revision": revision + 1,
            }
            self.db.execute(
                "INSERT INTO commits VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (commit_id, run_id, role, logical_round, attempt, batch_key, content_sha, json.dumps(receipt, sort_keys=True), _now()),
            )
            self.db.execute(
                "UPDATE lanes SET revision=?, phase=?, state_json=? WHERE run_id=? AND role=?",
                (revision + 1, new_state["phase"], json.dumps(new_state, sort_keys=True), run_id, role),
            )
            self.db.execute(
                "UPDATE attempts SET status='COMMITTED', bundle_sha256=?, content_sha256=?, receipt_json=? WHERE run_id=? AND role=? AND logical_round=? AND attempt=?",
                (bundle.content_sha256, content_sha, json.dumps(receipt, sort_keys=True), run_id, role, logical_round, attempt),
            )
            self.db.execute("COMMIT")
            return receipt
        except Exception:
            self.db.execute("ROLLBACK")
            raise

    @staticmethod
    def _transition(
        state: dict[str, Any],
        payload: dict[str, Any],
        role: str,
        *,
        bundle_sha256: str | None = None,
        commit_id: str | None = None,
        observation_fingerprint: str | None = None,
    ) -> tuple[dict[str, Any], list[str]]:
        candidate = json.loads(json.dumps(state))
        candidate.setdefault("targets", {})
        candidate.setdefault("observation_fingerprints", [])
        candidate.setdefault("reopened_target_ids", [])
        assigned: list[str] = []
        for target_id in payload["revisit_target_ids"]:
            if target_id in candidate["reopened_target_ids"]:
                raise V5ContractError(
                    f"target {target_id} has already been reopened once"
                )
            candidate["effective"].pop(target_id, None)
            if target_id not in candidate["revisable"]:
                candidate["revisable"].append(target_id)
            candidate["reopened_target_ids"].append(target_id)
        for assessment in payload["assessments"]:
            target_id = assessment["target_id"]
            if target_id in candidate["effective"] and target_id not in candidate["revisable"]:
                raise V5ContractError(f"target {target_id} was not reopened")
            if target_id not in candidate["enrolled"]:
                candidate["enrolled"].append(target_id)
            candidate["optional"] = [item for item in candidate["optional"] if item != target_id]
            history = candidate["history"].setdefault(target_id, [])
            record = {
                "version": len(history) + 1,
                "assessment": assessment,
                **({"bundle_sha256": bundle_sha256} if bundle_sha256 else {}),
                **({"commit_id": commit_id} if commit_id else {}),
            }
            history.append(record)
            candidate["effective"][target_id] = record
            candidate["revisable"] = [item for item in candidate["revisable"] if item != target_id]
        prefix = "I" if role in {"I", "implementation"} else "R"
        for target in payload["new_targets"]:
            statement_key = " ".join(target["statement"].casefold().split())
            existing_keys = {
                " ".join(value.get("statement", "").casefold().split())
                for value in candidate["targets"].values()
            }
            if statement_key in existing_keys:
                raise V5ContractError("new target duplicates an existing target")
            candidate["next_target_number"] += 1
            target_id = f"{prefix}{candidate['next_target_number'] - 1:03d}"
            candidate["enrolled"].append(target_id)
            candidate["pending"].append(target_id)
            candidate["targets"][target_id] = target
            assigned.append(target_id)
        candidate["pending"] = sorted(set(candidate["enrolled"]) - set(candidate["effective"]))
        structural_progress = bool(
            payload["assessments"] or payload["new_targets"] or payload["revisit_target_ids"]
        )
        observation_progress = bool(
            observation_fingerprint
            and observation_fingerprint not in candidate["observation_fingerprints"]
        )
        if observation_progress:
            candidate["observation_fingerprints"].append(observation_fingerprint)
        candidate["no_progress"] = (
            0 if structural_progress or observation_progress else candidate.get("no_progress", 0) + 1
        )
        candidate["round"] += 1
        if payload["done"]:
            if candidate["pending"] or payload["new_targets"] or payload["revisit_target_ids"]:
                raise V5ContractError("done=true requires no pending target or new work")
            candidate["phase"] = "complete"
        return candidate, assigned

    def freeze_summary(self, run_id: str, snapshot: dict[str, Any]) -> str:
        freeze_id = uuid.uuid4().hex
        payload = {"freeze_id": freeze_id, **snapshot}
        self.db.execute("BEGIN IMMEDIATE")
        try:
            existing = self.db.execute(
                "SELECT freeze_id, snapshot_json FROM freezes WHERE run_id=?", (run_id,)
            ).fetchone()
            if existing is not None:
                prior = json.loads(existing["snapshot_json"])
                prior.pop("freeze_id", None)
                if prior != snapshot:
                    raise V5ContractError(
                        "summary freeze was replayed with a different snapshot"
                    )
                self.db.execute("COMMIT")
                return existing["freeze_id"]
            for row in self.db.execute("SELECT phase FROM lanes WHERE run_id=?", (run_id,)).fetchall():
                if row["phase"] not in {"complete", "incomplete"}:
                    raise V5ContractError("cannot freeze summary while a lane is running")
            self.db.execute(
                "INSERT INTO freezes VALUES (?, ?, ?, ?)",
                (freeze_id, run_id, json.dumps(payload, sort_keys=True), _now()),
            )
            self.db.execute("UPDATE runs SET phase='frozen' WHERE run_id=?", (run_id,))
            self.db.execute("COMMIT")
            return freeze_id
        except Exception:
            self.db.execute("ROLLBACK")
            raise

    def frozen_snapshot(self, run_id: str) -> dict[str, Any] | None:
        row = self.db.execute(
            "SELECT snapshot_json FROM freezes WHERE run_id=?", (run_id,)
        ).fetchone()
        return json.loads(row["snapshot_json"]) if row is not None else None

    def stop_lane(self, run_id: str, role: str, *, reason: str) -> None:
        with self.db:
            row = self.db.execute(
                "SELECT revision, phase, state_json FROM lanes WHERE run_id=? AND role=?",
                (run_id, role),
            ).fetchone()
            if row is None:
                raise KeyError(role)
            state = json.loads(row["state_json"])
            if row["phase"] != "running":
                if row["phase"] == "incomplete" and state.get("stop_reason") == reason:
                    return
                raise V5ContractError(f"cannot stop lane in phase {row['phase']}")
            state["phase"] = "incomplete"
            state["stop_reason"] = reason
            self.db.execute(
                "UPDATE lanes SET revision=?, phase='incomplete', state_json=? WHERE run_id=? AND role=?",
                (row["revision"] + 1, json.dumps(state, sort_keys=True), run_id, role),
            )


__all__ = ["ImmutableBundleStore", "V5Bundle", "V5StateStore"]
