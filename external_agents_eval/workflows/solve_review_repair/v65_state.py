"""V6.5 lane state: the D lane on top of the unchanged V5 SQLite store."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from .v5_contracts import V5ContractError
from .v5_state import V5StateStore, _now

EXTRA_LANES = ("D",)


class V65StateStore(V5StateStore):
    def initialize_extra_lane(
        self,
        run_id: str,
        role: str,
        *,
        targets: list[str],
        target_catalog: dict[str, dict[str, Any]],
        batch_size: int,
        deadline_at: str,
    ) -> bool:
        """Create the lane on first use with an absolute deadline; False when it exists.

        The lane row is created lazily so a run without D targets has no D lane and
        the summary freeze does not wait for one.
        """
        if role not in EXTRA_LANES:
            raise ValueError(f"unsupported V6.5 lane: {role}")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        enrolled = list(dict.fromkeys(targets))
        deadline = datetime.fromisoformat(deadline_at)
        if deadline.tzinfo is None:
            raise V5ContractError("lane deadline must include a timezone")
        with self.db:
            row = self.db.execute(
                "SELECT state_json FROM lanes WHERE run_id=? AND role=?", (run_id, role)
            ).fetchone()
            if row is not None:
                state = json.loads(row["state_json"])
                if (state.get("initial_enrolled") != enrolled
                        or state.get("deadline_at") != deadline_at
                        or state.get("batch_size") != batch_size):
                    raise V5ContractError("cannot reinitialize a V6.5 lane with different inputs")
                return False
            remaining = max(0.0, (deadline - datetime.now(timezone.utc)).total_seconds())
            state = {
                **self._initial_lane(role),
                "initialized": True,
                "initial_enrolled": enrolled,
                "initial_optional": [],
                "batch_size": batch_size,
                "lane_started_at": _now(),
                "lane_deadline_sec": remaining,
                "deadline_at": deadline_at,
                "enrolled": enrolled,
                "optional": [],
                "targets": {target: dict(target_catalog.get(target, {})) for target in enrolled},
                "pending": enrolled,
            }
            state.pop("phase", None)
            self.db.execute(
                "INSERT INTO lanes VALUES (?, ?, 0, 'running', ?)",
                (run_id, role, json.dumps(state, sort_keys=True)),
            )
        return True

    def has_lane(self, run_id: str, role: str) -> bool:
        return self.db.execute(
            "SELECT 1 FROM lanes WHERE run_id=? AND role=?", (run_id, role)
        ).fetchone() is not None

    @staticmethod
    def _transition(
        state: dict[str, Any],
        payload: dict[str, Any],
        role: str,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], list[str]]:
        if role != "D":
            return V5StateStore._transition(state, payload, role, **kwargs)
        # Reuse the V5 transition unchanged; it numbers new targets with an R prefix
        # outside I, so rename this batch's assignments to the D namespace.
        candidate, assigned = V5StateStore._transition(state, payload, "R", **kwargs)
        renamed = {old: "D" + old[1:] for old in assigned}
        if any(new in candidate["targets"] for new in renamed.values()):
            raise V5ContractError("D target numbering collided with an existing target")
        for field in ("enrolled", "pending"):
            candidate[field] = [renamed.get(item, item) for item in candidate[field]]
        candidate["targets"] = {
            renamed.get(key, key): value for key, value in candidate["targets"].items()
        }
        return candidate, [renamed[item] for item in assigned]


__all__ = ["EXTRA_LANES", "V65StateStore"]
