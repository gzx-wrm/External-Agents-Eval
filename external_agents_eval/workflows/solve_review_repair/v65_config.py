"""V6.5: review view (A), assumption extraction (C) and disambiguation lane (D).

Everything else is inherited from V6.4 unchanged: models, I/R/Repair prompts (v5-6),
the build gate, lane limits and scoring. Each mechanism has its own switch so the
same code can run ablations; with all three off the review equals V6.4 except for
the version labels.

Time budget: X and D are counted inside the existing 45-minute review window. X
runs concurrently with I/R and has no container access; D starts when X ends and
shares the I/R lane deadline. Building the view happens before the window opens.
"""
from __future__ import annotations

from dataclasses import dataclass

from .review_view import LEVELS as REVIEW_VIEW_LEVELS
from .v64_config import V64Config

V65_PROMPT_VERSION = "v6.5"
V65_CONTROLLER_VERSION = "v6.5"
V57_REVIEW_PROMPT_VERSION = "v5-7"
V56_REVIEW_CONTROLLER_VERSION = "v5-6"
V65_REVIEW_SKILL = "v6.5-review-note-files"


@dataclass(frozen=True)
class V65Config(V64Config):
    workflow_prompt_version: str = V65_PROMPT_VERSION
    workflow_controller_version: str = V65_CONTROLLER_VERSION
    prompt_version: str = V57_REVIEW_PROMPT_VERSION
    controller_version: str = V56_REVIEW_CONTROLLER_VERSION
    review_skill: str = V65_REVIEW_SKILL
    # A: reviewers see P0 with added comments/docstrings and test changes withheld.
    review_view_enabled: bool = True
    review_view_levels: tuple[str, ...] = REVIEW_VIEW_LEVELS
    # C: one tool-less call turns P0's interpretive choices into neutral claims.
    assumptions_enabled: bool = True
    assumption_timeout_sec: float = 8 * 60
    assumption_max_items: int = 12
    assumption_max_attempts: int = 2
    # D: evidence-based resolution of ledger questions and extracted assumptions.
    disambiguation_enabled: bool = True
    disambiguation_max_targets: int = 6

    def validate(self) -> None:
        super().validate()
        if not set(self.review_view_levels) <= set(REVIEW_VIEW_LEVELS) or not self.review_view_levels:
            raise ValueError("V6.5 review_view_levels must be a non-empty subset of the view levels")
        if self.assumption_timeout_sec <= 0 or self.assumption_timeout_sec >= self.lane_deadline_sec:
            raise ValueError("V6.5 assumption timeout must fit inside the review window")
        if self.assumption_max_items < 1 or self.assumption_max_attempts < 1:
            raise ValueError("V6.5 assumption limits must be positive")
        if self.disambiguation_max_targets < 1:
            raise ValueError("V6.5 disambiguation_max_targets must be positive")
        if self.assumptions_enabled and not self.review_view_enabled:
            raise ValueError("V6.5 assumptions require the review view (they read withheld text)")
