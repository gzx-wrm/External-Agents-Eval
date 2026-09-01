from __future__ import annotations

import subprocess
from pathlib import Path

import docker
from external_agents_eval.agents.config import AGENTS

RUNTIME_PLATFORMS = {
    "amd64": "linux/amd64",
    "arm64": "linux/arm64/v8",
}


def build_runtime_images(
    repository_root: Path,
    agents: list[str],
    architectures: list[str],
) -> None:
    client = docker.from_env()
    try:
        for agent_id in agents:
            spec = AGENTS[agent_id]
            dockerfile = f"docker/Dockerfile.{agent_id}-runtime"
            for architecture in architectures:
                image_name = spec.runtime_image(architecture)
                platform = RUNTIME_PLATFORMS[architecture]
                print(
                    f"Building {image_name} ({platform}) from {spec.upstream_ref}...",
                    flush=True,
                )
                subprocess.run(
                    [
                        "docker",
                        "buildx",
                        "build",
                        "--load",
                        "--progress=plain",
                        "--platform",
                        platform,
                        "--file",
                        dockerfile,
                        "--tag",
                        image_name,
                        "--build-arg",
                        f"{agent_id.upper()}_REF={spec.upstream_ref}",
                        "--build-arg",
                        f"{agent_id.upper()}_COMMIT={spec.upstream_commit}",
                        str(repository_root),
                    ],
                    cwd=repository_root,
                    check=True,
                )
                image = client.images.get(image_name)
                image.reload()
                actual_architecture = image.attrs.get("Architecture")
                if actual_architecture != architecture:
                    raise RuntimeError(
                        f"Built {image_name} as {actual_architecture!r}, expected {architecture!r}"
                    )
                print(f"Built {image_name}: {image.id}", flush=True)
    finally:
        client.close()
