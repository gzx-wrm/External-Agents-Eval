"""Small artifact writer, independent of the review/repair workflow."""
from __future__ import annotations

import json
from pathlib import Path

from .index import digest


def sha256_json(value):
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


class ArtifactStore:
    def __init__(self, root, *, secrets=()):
        self.root = Path(root)
        self.secrets = tuple(secret for secret in secrets if secret)
        self.root.mkdir(parents=True, exist_ok=True)

    def redact(self, text):
        for secret in self.secrets:
            text = text.replace(secret, "[REDACTED]")
        return text

    def write_text(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(self.redact(text), encoding="utf-8")
        temporary.replace(path)

    def write_json(self, name, value):
        self.write_text(name, json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
