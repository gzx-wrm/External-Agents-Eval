from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from external_agents_eval.agents.config import AgentSpec


@dataclass(frozen=True)
class AgentRunContext:
    instance_id: str
    prompt: str
    model: str
    timeout_seconds: int
    api_key: str
    workdir: str = "/testbed"


@dataclass
class AgentRunResult:
    exit_code: int
    transcript: str
    metadata: dict[str, Any]


class AgentAdapter(Protocol):
    spec: AgentSpec

    def run(self, container, context: AgentRunContext) -> AgentRunResult: ...

    @property
    def transcript_suffix(self) -> str: ...
