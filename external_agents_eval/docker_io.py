from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
import threading
from collections.abc import Iterable
from pathlib import Path

import docker


def redact_secret(text: str, secret: str | None) -> str:
    if secret:
        return text.replace(secret, "[REDACTED]")
    return text


def decode_output(output: bytes | str | None) -> str:
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return str(output)


def put_files(container, destination: str, files: dict[str, bytes | str]) -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for relative_name, value in files.items():
            data = value.encode() if isinstance(value, str) else value
            info = tarfile.TarInfo(relative_name)
            info.size = len(data)
            info.mode = 0o600
            archive.addfile(info, io.BytesIO(data))
    buffer.seek(0)
    if not container.put_archive(destination, buffer.getvalue()):
        raise RuntimeError(f"Failed to copy files into {container.name}:{destination}")


def read_file(container, path: str) -> bytes:
    stream, _ = container.get_archive(path)
    payload = b"".join(stream)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:*") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        if len(members) != 1:
            raise RuntimeError(f"Expected one file in archive for {path}, found {len(members)}")
        extracted = archive.extractfile(members[0])
        if extracted is None:
            raise RuntimeError(f"Cannot extract {path} from {container.name}")
        return extracted.read()


class RuntimeArchiveCache:
    """Materialize a pinned runtime image directory once and stream it to task containers."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def ensure(
        self,
        client: docker.DockerClient,
        image_name: str,
        runtime_root: str,
        *,
        expected_architecture: str,
    ) -> Path:
        image = client.images.get(image_name)
        architecture = image.attrs.get("Architecture")
        if architecture != expected_architecture:
            raise RuntimeError(
                f"Runtime image {image_name} has architecture {architecture!r}; "
                f"the selected task images require {expected_architecture!r}. Rebuild it with "
                "external-agents-eval build-runtimes."
            )
        image_id = image.id.removeprefix("sha256:")
        agent_label = image_name.split("/", 1)[-1].split(":", 1)[0]
        archive_path = self.cache_dir / f"{agent_label}-{image_id[:16]}.tar"
        metadata_path = archive_path.with_suffix(".json")
        with self._lock:
            if archive_path.is_file() and metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text())
                if metadata.get("image_id") == image.id and metadata.get("root") == runtime_root:
                    return archive_path

            temporary = archive_path.with_suffix(".tar.tmp")
            runtime_container = client.containers.create(image_name)
            try:
                stream, _ = runtime_container.get_archive(runtime_root)
                with temporary.open("wb") as handle:
                    for chunk in stream:
                        handle.write(chunk)
                temporary.replace(archive_path)
                metadata_path.write_text(
                    json.dumps(
                        {
                            "image": image_name,
                            "image_id": image.id,
                            "root": runtime_root,
                            "size": archive_path.stat().st_size,
                            "sha256": _sha256_file(archive_path),
                        },
                        indent=2,
                    )
                    + "\n"
                )
            finally:
                runtime_container.remove(force=True)
        return archive_path

    @staticmethod
    def install(container, archive_path: Path, destination: str = "/opt") -> None:
        with archive_path.open("rb") as handle:
            if not container.put_archive(destination, handle):
                raise RuntimeError(
                    f"Failed to install runtime archive into {container.name}:{destination}"
                )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def disk_free_bytes(path: Path) -> int:
    return os.statvfs(path).f_bavail * os.statvfs(path).f_frsize


def write_log(path: Path, chunks: Iterable[str], secret: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(redact_secret(chunk, secret))
            if chunk and not chunk.endswith("\n"):
                handle.write("\n")
