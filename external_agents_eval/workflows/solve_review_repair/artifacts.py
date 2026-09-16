from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(payload: Any) -> str:
    return sha256_bytes(canonical_json_bytes(payload))


class ArtifactStore:
    """Versioned, append-only workflow artifact writer."""

    def __init__(self, root: str | Path, *, secrets: tuple[str, ...] = ()) -> None:
        self.root = Path(root)
        self.secrets = tuple(secret for secret in secrets if secret)
        self.root.mkdir(parents=True, exist_ok=True)

    def redact(self, text: str) -> str:
        for secret in self.secrets:
            text = text.replace(secret, "[REDACTED]")
        return text

    def write_bytes(self, relative_path: str, payload: bytes) -> Path:
        leaked = next(
            (secret for secret in self.secrets if secret.encode("utf-8") in payload),
            None,
        )
        if leaked is not None:
            raise ValueError(f"Refusing to persist an unredacted secret in {relative_path}")
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path

    def write_text(self, relative_path: str, text: str) -> Path:
        return self.write_bytes(relative_path, self.redact(text).encode("utf-8"))

    def write_json(self, relative_path: str, payload: Any) -> Path:
        text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str)
        return self.write_text(relative_path, text + "\n")

    def event(self, state: str, **details: Any) -> None:
        payload = {
            "at": datetime.now(timezone.utc).isoformat(),
            "state": state,
            **details,
        }
        path = self.root / "events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(self.redact(json.dumps(payload, ensure_ascii=False, default=str)) + "\n")

    def digest_file(self, relative_path: str) -> str:
        return sha256_bytes((self.root / relative_path).read_bytes())
