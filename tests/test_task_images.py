from types import SimpleNamespace

import pytest

import docker
from external_agents_eval.task_images import (
    TaskImageAuditError,
    audit_official_judge_dependencies,
    audit_task_images,
)


class FakeImage:
    def __init__(self, image_id: str, architecture: str, created: str = "2026-01-02"):
        self.id = image_id
        self.attrs = {
            "Architecture": architecture,
            "RepoDigests": [],
            "Created": created,
        }


class FakeImages:
    def __init__(self, images):
        self.images = images

    def get(self, name):
        try:
            return self.images[name]
        except KeyError as error:
            raise docker.errors.ImageNotFound(name) from error


def make_spec(entry):
    architecture = entry["arch"]
    return SimpleNamespace(
        arch=architecture,
        instance_image_key=f"sweb.eval.{architecture}.{entry['instance_id']}:latest",
        platform="linux/x86_64" if architecture == "x86_64" else "linux/arm64/v8",
    )


def make_full_spec(entry):
    spec = make_spec(entry)
    spec.base_image_key = f"sweb.base.{entry['arch']}:latest"
    spec.env_image_key = f"sweb.env.{entry['arch']}.abc:latest"
    return spec


def test_audit_accepts_mixed_prebuilt_architectures():
    entries = [
        {"instance_id": "one", "arch": "x86_64"},
        {"instance_id": "two", "arch": "arm64"},
    ]
    client = SimpleNamespace(
        images=FakeImages(
            {
                "sweb.eval.x86_64.one:latest": FakeImage("sha256:one", "amd64"),
                "sweb.eval.arm64.two:latest": FakeImage("sha256:two", "arm64"),
            }
        )
    )
    records = audit_task_images(client, entries, make_spec=make_spec)
    assert [record.docker_architecture for record in records] == ["amd64", "arm64"]


def test_audit_blocks_missing_image_without_pull_or_build():
    client = SimpleNamespace(images=FakeImages({}))
    with pytest.raises(TaskImageAuditError, match="missing prebuilt images"):
        audit_task_images(
            client,
            [{"instance_id": "one", "arch": "x86_64"}],
            make_spec=make_spec,
        )


def test_audit_blocks_wrong_local_architecture():
    client = SimpleNamespace(
        images=FakeImages(
            {
                "sweb.eval.x86_64.one:latest": FakeImage("sha256:one", "arm64"),
            }
        )
    )
    with pytest.raises(TaskImageAuditError, match="invalid image architectures"):
        audit_task_images(
            client,
            [{"instance_id": "one", "arch": "x86_64"}],
            make_spec=make_spec,
        )


def test_official_audit_requires_fresh_reusable_dependency_chain():
    client = SimpleNamespace(
        images=FakeImages(
            {
                "sweb.base.x86_64:latest": FakeImage("sha256:base", "amd64", "2026-01-01"),
                "sweb.env.x86_64.abc:latest": FakeImage("sha256:env", "amd64", "2026-01-02"),
                "sweb.eval.x86_64.one:latest": FakeImage("sha256:task", "amd64", "2026-01-03"),
            }
        )
    )
    payload = audit_official_judge_dependencies(
        client,
        [{"instance_id": "one", "arch": "x86_64"}],
        make_spec=make_full_spec,
    )
    assert payload["dependency_image_count"] == 3


def test_official_audit_blocks_chain_that_would_trigger_rebuild():
    client = SimpleNamespace(
        images=FakeImages(
            {
                "sweb.base.x86_64:latest": FakeImage("sha256:base", "amd64", "2026-01-01"),
                "sweb.env.x86_64.abc:latest": FakeImage("sha256:env", "amd64", "2026-01-03"),
                "sweb.eval.x86_64.one:latest": FakeImage("sha256:task", "amd64", "2026-01-02"),
            }
        )
    )
    with pytest.raises(TaskImageAuditError, match="would build or rebuild"):
        audit_official_judge_dependencies(
            client,
            [{"instance_id": "one", "arch": "x86_64"}],
            make_spec=make_full_spec,
        )
