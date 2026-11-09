"""Four read-only tools with Pi-compatible names, confined to a frozen repository."""

from __future__ import annotations

import fnmatch
import os
import subprocess
from pathlib import Path
from typing import Any


class RepositoryTools:
    def __init__(self, root: Path, max_output_chars: int = 12000) -> None:
        self.root = root.resolve()
        self.max_output_chars = max_output_chars
        self.calls: list[dict[str, Any]] = []

    def _path(self, path: str) -> Path:
        if path == "/work/base":
            path = "."
        elif path.startswith("/work/base/"):
            path = path[len("/work/base/"):]
        relative = Path(path)
        if relative.is_absolute():
            raise ValueError("Use a repository-relative path or /work/base")
        resolved = (self.root / relative).resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError("Inspect only the frozen repository")
        return resolved

    def _bounded(self, text: str) -> str:
        if len(text) > self.max_output_chars:
            return text[:self.max_output_chars] + "\n[output truncated; narrow the query]"
        return text

    def read(self, path: str, offset: int = 1, limit: int = 200) -> str:
        """Read numbered lines from a repository file.

        Args:
            path: Repository-relative file path.
            offset: First line, counting from one.
            limit: Maximum lines to read, up to 400.
        """
        if offset < 1 or limit < 1:
            raise ValueError("offset and limit must be positive")
        with self._path(path).open(encoding="utf-8", errors="replace") as stream:
            selected = []
            for number, line in enumerate(stream, 1):
                if number < offset:
                    continue
                if number >= offset + min(limit, 400):
                    break
                selected.append(f"{number}: {line.rstrip()}")
                if sum(map(len, selected)) >= self.max_output_chars:
                    break
        return self._bounded("\n".join(selected))

    def ls(self, path: str = ".", limit: int = 200) -> str:
        """List entries in a repository directory.

        Args:
            path: Repository-relative directory path.
            limit: Maximum entries to show, up to 400.
        """
        if limit < 1:
            raise ValueError("limit must be positive")
        entries = sorted(self._path(path).iterdir(), key=lambda p: p.name)
        return self._bounded("\n".join(
            p.name + ("/" if p.is_dir() else "") for p in entries[:min(limit, 400)]
        ))

    def find(self, pattern: str, path: str = ".", limit: int = 200) -> str:
        """Find repository files matching a glob, without following directory symlinks.

        Args:
            pattern: Glob matched against a relative file path or filename.
            path: Repository-relative search directory.
            limit: Maximum matches to show, up to 400.
        """
        if limit < 1:
            raise ValueError("limit must be positive")
        base = self._path(path)
        matches = []
        for parent, dirs, files in os.walk(base, followlinks=False):
            dirs.sort()
            for name in sorted(files):
                p = Path(parent) / name
                relative = p.relative_to(base).as_posix()
                if fnmatch.fnmatch(relative, pattern) or fnmatch.fnmatch(name, pattern):
                    matches.append(p.relative_to(self.root).as_posix())
                    if len(matches) >= min(limit, 400):
                        return self._bounded("\n".join(matches))
        return self._bounded("\n".join(matches))

    def grep(
        self, pattern: str, path: str = ".", glob: str | None = None,
        ignoreCase: bool = False, literal: bool = False, context: int = 0, limit: int = 100,
    ) -> str:
        """Search repository text with ripgrep and numbered matching lines.

        Args:
            pattern: Regex, or literal text when literal is true.
            path: Repository-relative file or directory to search.
            glob: Optional file glob.
            ignoreCase: Whether to ignore case.
            literal: Whether to interpret the pattern as literal text.
            context: Context lines around matches, up to five.
            limit: Maximum matching lines per file, up to 200.
        """
        target = self._path(path)
        if limit < 1 or context < 0:
            raise ValueError("limit must be positive and context nonnegative")
        command = [
            "rg", "--line-number", "--with-filename", "--color=never", "--hidden",
            "--glob", "!.git/**", "--max-count", str(min(limit, 200)),
            "--context", str(min(context, 5)),
        ]
        if ignoreCase:
            command.append("--ignore-case")
        if literal:
            command.append("--fixed-strings")
        if glob:
            command += ["--glob", glob]
        command += ["--", pattern, str(target.relative_to(self.root))]
        result = subprocess.run(
            command, cwd=self.root, capture_output=True, text=True, timeout=10,
        )
        if result.returncode not in (0, 1):
            raise ValueError(result.stderr.strip())
        return self._bounded(result.stdout)

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        if name not in {"read", "ls", "find", "grep"}:
            output = f"Tool error: unknown tool {name}"
            error = True
        else:
            try:
                output = getattr(self, name)(**arguments)
                error = False
            except (OSError, ValueError, TypeError, subprocess.TimeoutExpired) as exc:
                output, error = f"Tool error: {exc}", True
        self.calls.append({"name": name, "arguments": arguments, "output": output, "error": error})
        return output


def tool_schemas() -> list[dict[str, Any]]:
    """Load the frozen protocol; no Transformers dependency is needed by material checks."""
    import json
    return json.loads((Path(__file__).parent / "protocol/tools.json").read_text())
