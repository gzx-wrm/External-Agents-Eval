"""Strip test files that a model patch newly adds, before the patch is graded (v6.6.4).

Official verifiers bring their own test files. A model-added test file with the same name
breaks the build (for example a duplicate Go test function), so only *new* files that
match a test-file rule are removed. Source changes and edits to existing files are kept.
"""
from __future__ import annotations

import fnmatch
import posixpath
import re
from typing import Any

TEST_DIRECTORIES = frozenset({"test", "tests", "__tests__", "spec", "testdata"})
TEST_FILE_PATTERNS = ("*_test.go", "test_*.py", "*_test.py", "conftest.py", "*.test.*", "*.spec.*",
                      "*Test.java", "*_spec.rb")
_HEADER = re.compile(r"^diff --git (?:\"?a/)(?P<a>.+?)\"? (?:\"?b/)(?P<b>.+?)\"?$")


def is_test_path(path: str) -> bool:
    directories = path.split("/")[:-1]
    name = posixpath.basename(path)
    return (any(part in TEST_DIRECTORIES for part in directories)
            or any(fnmatch.fnmatchcase(name, pattern) for pattern in TEST_FILE_PATTERNS))


def _sections(patch: str) -> list[str]:
    """Split a git patch into per-file sections, keeping every byte."""
    sections: list[str] = []
    current: list[str] = []
    for line in patch.splitlines(keepends=True):
        if line.startswith("diff --git ") and current:
            sections.append("".join(current))
            current = []
        current.append(line)
    if current:
        sections.append("".join(current))
    return sections


def _new_file_path(section: str) -> str | None:
    lines = section.splitlines()
    if not lines or not lines[0].startswith("diff --git "):
        return None
    header: list[str] = []
    for line in lines[1:]:
        if line.startswith(("@@", "GIT binary patch", "Binary files ")):
            break
        header.append(line)
    if not any(line.startswith("new file mode") for line in header):
        return None
    for line in header:
        if line.startswith("+++ b/"):
            return line[len("+++ b/"):].rstrip("\t")
    match = _HEADER.match(lines[0])
    return match.group("b") if match else None


def strip_added_tests(patch: str) -> tuple[str, list[str]]:
    """Return (patch without newly added test files, removed paths)."""
    kept: list[str] = []
    removed: list[str] = []
    for section in _sections(patch):
        path = _new_file_path(section)
        if path is not None and is_test_path(path):
            removed.append(path)
        else:
            kept.append(section)
    return "".join(kept), removed


def filter_record(original: str, submitted: str, removed: list[str]) -> dict[str, Any]:
    from external_agents_eval.workflows.solve_review_repair.artifacts import sha256_bytes

    return {"rule": "v6.6.4 strip newly added test files", "removed": removed,
            "test_directories": sorted(TEST_DIRECTORIES), "test_file_patterns": list(TEST_FILE_PATTERNS),
            "original_sha256": sha256_bytes(original.encode()),
            "submitted_sha256": sha256_bytes(submitted.encode())}
