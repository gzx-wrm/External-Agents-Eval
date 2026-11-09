from __future__ import annotations

from dataclasses import dataclass

from .v5_config import V5Config

V6_PROMPT_VERSION = "v6.1"
V6_CONTROLLER_VERSION = "v6.1"


@dataclass(frozen=True)
class V6Config(V5Config):
    """V6.1 labels; review behavior and limits are inherited from V5."""

    workflow_prompt_version: str = V6_PROMPT_VERSION
    workflow_controller_version: str = V6_CONTROLLER_VERSION
