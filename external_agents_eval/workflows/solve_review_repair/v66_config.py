from __future__ import annotations

from dataclasses import dataclass

from .v62_config import V62Config

# v6.6.1: rules_out/basis fields, merged targets, implicit inputs, evidence-plan check,
# Pi API-error detection and delivery views (approved 2026-11-03).
# v6.6.2: sibling hard rule in C, per-batch review check points with checkpoint_results,
# probe-stage step allocation (approved 2026-11-04).
# v6.6.3: sibling rule withdrawn in C and review (user, 2026-11-04); C is the v6.6.1 prompt again.
# Kept: per-batch review check points with >=2 inputs from different partitions, probe step
# allocation with incremental results, probe-resolved points ranked first.
# v6.6.4: C in target batches (one corrective retry per batch; invalid points dropped, coverage
# rebuilt), 30k-character tool-output cap in I/R/S, added test files stripped before grading
# (user, 2026-11-04).
V66_PROMPT_VERSION = "v6.6.4"
V66_CONTROLLER_VERSION = "v6.6.4"


@dataclass(frozen=True)
class V66Config(V62Config):
    """V6.2 A plus an analog/contract evidence plan and a check-point stage C."""

    workflow_prompt_version: str = V66_PROMPT_VERSION
    workflow_controller_version: str = V66_CONTROLLER_VERSION
    checkpoint_max_per_target: int = 4
    checkpoint_max_attempts: int = 2
    checkpoint_batch_max_targets: int = 8
    checkpoint_timeout_sec: float = 1800
    solver_view_max_points: int = 12
    evidence_plan_continuations: int = 1
