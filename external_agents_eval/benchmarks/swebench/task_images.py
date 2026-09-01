from __future__ import annotations

import datetime as dt
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import docker
from external_agents_eval.benchmarks.swebench.manifest import atomic_write_json

TEST_SPEC_TO_DOCKER_ARCHITECTURE = {
    "x86_64": "amd64",
    "arm64": "arm64",
}


class TaskImageAuditError(RuntimeError):
    pass


@dataclass(frozen=True)
class TaskImageRecord:
    instance_id: str
    image_name: str
    image_id: str
    repo_digests: tuple[str, ...]
    test_spec_architecture: str
    docker_architecture: str
    platform: str

    def as_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["repo_digests"] = list(self.repo_digests)
        return payload


def audit_task_images(
    client: docker.DockerClient,
    entries: list[dict[str, Any]],
    *,
    make_spec: Callable[[dict[str, Any]], Any] | None = None,
) -> list[TaskImageRecord]:
    """Resolve every task's exact prebuilt image without pulling or building anything."""
    if make_spec is None:
        try:
            from swebench.harness.test_spec import make_test_spec
        except ImportError as error:
            raise TaskImageAuditError(
                "The SWE-bench harness is not importable; run in the hgm Conda environment"
            ) from error
        make_spec = make_test_spec

    records: list[TaskImageRecord] = []
    missing: list[str] = []
    wrong_architecture: list[str] = []
    for entry in entries:
        test_spec = make_spec(entry)
        expected_architecture = TEST_SPEC_TO_DOCKER_ARCHITECTURE.get(test_spec.arch)
        if expected_architecture is None:
            wrong_architecture.append(
                f"{entry['instance_id']}: unsupported TestSpec architecture {test_spec.arch!r}"
            )
            continue
        try:
            image = client.images.get(test_spec.instance_image_key)
        except docker.errors.ImageNotFound:
            missing.append(f"{entry['instance_id']}: {test_spec.instance_image_key}")
            continue

        actual_architecture = image.attrs.get("Architecture")
        if actual_architecture != expected_architecture:
            wrong_architecture.append(
                f"{entry['instance_id']}: {test_spec.instance_image_key} is "
                f"{actual_architecture!r}, expected {expected_architecture!r}"
            )
            continue
        records.append(
            TaskImageRecord(
                instance_id=entry["instance_id"],
                image_name=test_spec.instance_image_key,
                image_id=image.id,
                repo_digests=tuple(image.attrs.get("RepoDigests") or ()),
                test_spec_architecture=test_spec.arch,
                docker_architecture=actual_architecture,
                platform=test_spec.platform,
            )
        )

    if missing or wrong_architecture:
        details = []
        if missing:
            details.append("missing prebuilt images:\n  " + "\n  ".join(missing))
        if wrong_architecture:
            details.append("invalid image architectures:\n  " + "\n  ".join(wrong_architecture))
        raise TaskImageAuditError(
            "Prebuilt task-image audit failed. No task images will be pulled or built.\n"
            + "\n".join(details)
        )
    return records


def task_image_audit_payload(records: list[TaskImageRecord]) -> dict[str, Any]:
    architecture_counts: dict[str, int] = {}
    for record in records:
        architecture_counts[record.docker_architecture] = (
            architecture_counts.get(record.docker_architecture, 0) + 1
        )
    return {
        "audit_version": 1,
        "reuse_only": True,
        "task_count": len(records),
        "architecture_counts": architecture_counts,
        "images": [record.as_json() for record in records],
    }


def ensure_task_image_audit(path: Path, records: list[TaskImageRecord]) -> None:
    """Persist immutable image identities and reject image drift on resume."""
    expected = task_image_audit_payload(records)
    if path.exists():
        try:
            current = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise TaskImageAuditError(f"Cannot read existing task-image audit {path}: {error}")
        comparable_current = dict(current)
        comparable_current.pop("audited_at", None)
        if comparable_current != expected:
            raise TaskImageAuditError(
                f"Prebuilt task image identities differ from existing audit {path}; "
                "refusing to mix images in one evaluation result"
            )
        return
    expected["audited_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    atomic_write_json(path, expected)


def audit_official_judge_dependencies(
    client: docker.DockerClient,
    entries: list[dict[str, Any]],
    *,
    make_spec: Callable[[dict[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    """Prove the official runner can reuse its whole image chain without building."""
    if make_spec is None:
        try:
            from swebench.harness.test_spec import make_test_spec
        except ImportError as error:
            raise TaskImageAuditError(
                "The SWE-bench harness is not importable; run in the hgm Conda environment"
            ) from error
        make_spec = make_test_spec

    task_records = audit_task_images(client, entries, make_spec=make_spec)
    dependencies: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    stale: list[str] = []
    for entry in entries:
        test_spec = make_spec(entry)
        images = {}
        for role, name in (
            ("base", test_spec.base_image_key),
            ("environment", test_spec.env_image_key),
            ("task", test_spec.instance_image_key),
        ):
            try:
                image = client.images.get(name)
            except docker.errors.ImageNotFound:
                missing.append(f"{entry['instance_id']}: missing {role} image {name}")
                continue
            images[role] = image
            dependencies.setdefault(
                name,
                {
                    "image_name": name,
                    "image_id": image.id,
                    "repo_digests": list(image.attrs.get("RepoDigests") or ()),
                    "architecture": image.attrs.get("Architecture"),
                    "created": image.attrs.get("Created"),
                },
            )
        if len(images) != 3:
            continue
        # These are the exact timestamp comparisons used by the pinned SWE-bench
        # harness to decide whether it must rebuild an env or task image.
        if images["environment"].attrs["Created"] < images["base"].attrs["Created"]:
            stale.append(f"{entry['instance_id']}: environment image is older than its base image")
        if images["task"].attrs["Created"] < images["environment"].attrs["Created"]:
            stale.append(f"{entry['instance_id']}: task image is older than its environment image")

    if missing or stale:
        details = []
        if missing:
            details.append("missing judge dependencies:\n  " + "\n  ".join(missing))
        if stale:
            details.append("stale judge dependencies:\n  " + "\n  ".join(stale))
        raise TaskImageAuditError(
            "Official judge reuse-only audit failed. The judge was not started because "
            "the pinned runner would build or rebuild images.\n" + "\n".join(details)
        )

    return {
        "audit_version": 1,
        "reuse_only": True,
        "task_count": len(task_records),
        "dependency_image_count": len(dependencies),
        "dependencies": sorted(dependencies.values(), key=lambda item: item["image_name"]),
    }
