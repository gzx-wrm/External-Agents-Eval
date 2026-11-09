"""V6.3 identifies the mini file-delivery adapter; V6 review policy is inherited."""
from dataclasses import dataclass

from .v62_config import V62Config


@dataclass(frozen=True)
class V63Config(V62Config):
    workflow_prompt_version: str = "v6.3"
    workflow_controller_version: str = "v6.3"
