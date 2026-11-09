"""V6.4: build gate, I probe rules and role-prompt revisions over the V6.3 Mini path.

Models, budgets, isolation, S adjudication and scoring are inherited unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

from .v63_config import V63Config

V64_PROMPT_VERSION = "v6.4"
V64_CONTROLLER_VERSION = "v6.4"
V56_REVIEW_PROMPT_VERSION = "v5-6"
V55_REVIEW_CONTROLLER_VERSION = "v5-5"
V64_REVIEW_SKILL = "v6.4-review-note-files"
V64_BUILD_GATE_TIMEOUT_SEC = 600


@dataclass(frozen=True)
class V64Config(V63Config):
    workflow_prompt_version: str = V64_PROMPT_VERSION
    workflow_controller_version: str = V64_CONTROLLER_VERSION
    prompt_version: str = V56_REVIEW_PROMPT_VERSION
    controller_version: str = V55_REVIEW_CONTROLLER_VERSION
    review_skill: str = V64_REVIEW_SKILL
    build_gate_enabled: bool = True
    build_gate_timeout_sec: float = V64_BUILD_GATE_TIMEOUT_SEC

    def validate(self) -> None:
        super().validate()
        if self.build_gate_timeout_sec <= 0:
            raise ValueError("V6.4 build gate timeout must be positive")
