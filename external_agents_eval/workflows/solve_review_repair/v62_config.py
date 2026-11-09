from __future__ import annotations

from dataclasses import dataclass

from .v6_config import V6Config

V62_PROMPT_VERSION = "v6.2"
V62_CONTROLLER_VERSION = "v6.2"


@dataclass(frozen=True)
class V62Config(V6Config):
    """Change A orchestration only; the FT and V5 protocols stay frozen."""

    workflow_prompt_version: str = V62_PROMPT_VERSION
    workflow_controller_version: str = V62_CONTROLLER_VERSION
