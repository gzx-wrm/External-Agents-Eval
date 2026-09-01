from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"


@dataclass(frozen=True)
class AgentSpec:
    agent_id: str
    upstream_url: str
    upstream_ref: str
    upstream_commit: str
    runtime_image_repository: str
    runtime_image_tag: str
    runtime_root: str
    license_name: str = "MIT"

    def runtime_image(self, docker_architecture: str) -> str:
        if docker_architecture not in {"amd64", "arm64"}:
            raise ValueError(f"Unsupported Docker architecture: {docker_architecture}")
        return f"{self.runtime_image_repository}:{self.runtime_image_tag}-{docker_architecture}"


AGENTS = {
    "pi": AgentSpec(
        agent_id="pi",
        upstream_url="https://github.com/earendil-works/pi",
        upstream_ref="v0.84.2",
        upstream_commit="914cf1472e715297caa30db4b9535d534a9eb718",
        runtime_image_repository="external-agents-eval/pi-runtime",
        runtime_image_tag="v0.84.2",
        runtime_root="/opt/pi-runtime",
    ),
    "hermes": AgentSpec(
        agent_id="hermes",
        upstream_url="https://github.com/NousResearch/hermes-agent",
        upstream_ref="v2026.8.18",
        upstream_commit="e624e9fde561e1add9388384012b295fde669ade",
        runtime_image_repository="external-agents-eval/hermes-runtime",
        runtime_image_tag="v2026.8.18",
        runtime_root="/opt/hermes",
    ),
}


def get_agent_spec(agent_id: str) -> AgentSpec:
    try:
        return AGENTS[agent_id]
    except KeyError as error:
        raise ValueError(f"Unsupported agent: {agent_id}") from error
