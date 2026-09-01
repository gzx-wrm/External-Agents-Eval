"""Compatibility settings exported from the agent and SWE-bench layers."""

from external_agents_eval.agents.config import (
    AGENTS,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    AgentSpec,
    get_agent_spec,
)
from external_agents_eval.benchmarks.swebench.config import (
    DEFAULT_MANIFEST,
    DEFAULT_RESULTS_DIR,
    DEFAULT_SWEBENCH_RUNNER,
)

__all__ = [
    "AGENTS",
    "DEFAULT_BASE_URL",
    "DEFAULT_MANIFEST",
    "DEFAULT_MODEL",
    "DEFAULT_RESULTS_DIR",
    "DEFAULT_SWEBENCH_RUNNER",
    "AgentSpec",
    "get_agent_spec",
]
