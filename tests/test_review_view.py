"""V6.5 review view: withheld comments/docstrings and test-file changes."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from external_agents_eval.workflows.solve_review_repair import review_view as rv


def lines(*numbers):
    return set(numbers)


def all_lines(text):
    return set(range(1, text.count("\n") + 2))


# ---------------------------------------------------------------------------
# patch parsing and test paths
# ---------------------------------------------------------------------------

PATCH = """diff --git a/src/a.ts b/src/a.ts
index 1111111..2222222 100644
--- a/src/a.ts
+++ b/src/a.ts
@@ -1,3 +1,5 @@
 export const a = 1;
+// new comment
+export const b = 2;
 export const c = 3;
-export const d = 4;
+export const e = 5;
diff --git a/test/a.test.ts b/test/a.test.ts
new file mode 100644
--- /dev/null
+++ b/test/a.test.ts
@@ -0,0 +1,2 @@
+import { b } from "../src/a";
+test("b", () => {});
diff --git a/old.py b/old.py
deleted file mode 100644
--- a/old.py
+++ /dev/null
@@ -1 +0,0 @@
-x = 1
diff --git a/img.png b/img.png
new file mode 100644
Binary files /dev/null and b/img.png differ
"""


def test_parse_patch_tracks_post_image_added_lines_and_status():
    changes = {change["path"]: change for change in rv.parse_patch(PATCH)}
    assert changes["src/a.ts"]["status"] == "modified"
    assert changes["src/a.ts"]["added_lines"] == {2, 3, 5}
    assert changes["test/a.test.ts"]["status"] == "added"
    assert changes["test/a.test.ts"]["added_lines"] == {1, 2}
    assert changes["old.py"]["status"] == "deleted"
    assert changes["img.png"]["binary"] is True


@pytest.mark.parametrize("path, expected", [
    ("tests/test_cache.py", True), ("pkg/test_utils.py", True), ("pkg/x_test.py", True),
    ("conftest.py", True), ("ansi/ansi_test.go", True), ("src/a.test.ts", True),
    ("src/a.spec.tsx", True), ("__tests__/auto-toc.test.ts", True),
    ("test/parser/miscellaneous/__snapshots__/commonjs.ts.snap", True),
    ("src/test/java/FooTest.java", True), ("cli_test.ts", True),
    ("src/parser.ts", False), ("gql/transport/aiohttp.py", False), ("truncate.go", False),
    ("src/lang/locale/en.ts", False), ("contest.py", False),
])
def test_is_test_path(path, expected):
    assert rv.is_test_path(path) is expected


# ---------------------------------------------------------------------------
# C-like languages
# ---------------------------------------------------------------------------

MERIYAH = """function isUsingBindingStart(token: Token): boolean {
  // `using [x] = y` is a valid member-expression assignment, so only an
  // identifier or an object pattern can unambiguously start a declaration.
  return (token & Token.IsIdentifier) === Token.IsIdentifier || token === Token.LeftBrace;
}
"""


def test_ts_rationale_comment_is_withheld():
    out, removed, reason = rv.strip_file("src/parser.ts", MERIYAH, all_lines(MERIYAH))
    assert reason is None
    assert "member-expression" not in out and "using [x]" not in out
    assert "return (token & Token.IsIdentifier)" in out
    assert out.count("\n") == MERIYAH.count("\n") - 2
    assert [item["kind"] for item in removed] == ["comment", "comment"]


def test_only_added_lines_are_touched():
    text = "// base comment\nconst a = 1; // base trailing\n// new\nconst b = 2;\n"
    out, removed, _ = rv.strip_file("a.ts", text, lines(3, 4))
    assert out == "// base comment\nconst a = 1; // base trailing\nconst b = 2;\n"
    assert len(removed) == 1


def test_ts_directives_strings_templates_and_regexes_survive():
    text = (
        '/// <reference types="node" />\n'
        "// @ts-expect-error intentional\n"
        "/* eslint-disable no-console */\n"
        'const url = "http://example.com"; // trailing note\n'
        "const t = `a // b ${f(/* c */ 1)} // d`;\n"
        "const re = /\\/\\/x/; // regex note\n"
        "const y = a / b; /* inline */ const z = 1;\n"
        "/**\n"
        " * JSDoc rationale.\n"
        " */\n"
        "export function f() { return 1; }\n"
    )
    out, removed, _ = rv.strip_file("a.ts", text, all_lines(text))
    assert '/// <reference types="node" />' in out
    assert "// @ts-expect-error intentional" in out
    assert "/* eslint-disable no-console */" in out
    assert 'const url = "http://example.com";\n' in out
    assert "const t = `a // b ${f(  1)} // d`;" in out
    assert "const re = /\\/\\/x/;\n" in out
    assert "const y = a / b;   const z = 1;" in out
    assert "JSDoc rationale" not in out and "trailing note" not in out
    assert "export function f() { return 1; }" in out
    assert len(removed) == 5


def test_jsx_only_removes_full_line_comments():
    text = "// rationale\nconst x = <a href={u}>http://site</a>; // note\n"
    out, removed, _ = rv.strip_file("a.tsx", text, all_lines(text))
    assert out == "const x = <a href={u}>http://site</a>; // note\n"
    assert len(removed) == 1


def test_go_directives_and_raw_strings():
    text = (
        "// Package x does things.\n"
        "package x\n\n"
        "//go:embed data.txt\n"
        "var data string\n\n"
        "// Truncate keeps resets because tests expect it.\n"
        "func Truncate() string { return `// not a comment` } // trailing\n"
    )
    out, removed, _ = rv.strip_file("truncate.go", text, all_lines(text))
    assert "//go:embed data.txt" in out
    assert "`// not a comment`" in out
    assert "because tests expect it" not in out and "Package x does" not in out
    assert "func Truncate() string { return `// not a comment` }\n" in out
    assert len(removed) == 3


def test_go_cgo_files_are_left_unchanged():
    text = '/*\n#include <stdio.h>\n*/\nimport "C"\n'
    out, removed, reason = rv.strip_file("c.go", text, all_lines(text))
    assert out == text and removed == [] and "cgo" in reason


def test_rust_doc_comments_lifetimes_raw_strings_and_nesting():
    text = (
        "/// Doc rationale.\n"
        "fn f<'a>(x: &'a str) -> char {\n"
        "    let s = r#\"// keep\"#;\n"
        "    /* outer /* inner */ still comment */\n"
        "    'x'\n"
        "}\n"
    )
    out, removed, _ = rv.strip_file("lib.rs", text, all_lines(text))
    assert "Doc rationale" not in out and "still comment" not in out
    assert "r#\"// keep\"#" in out and "fn f<'a>(x: &'a str)" in out and "'x'" in out
    assert len(removed) == 2


# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------

PY = '''"""Module docstring added by P0."""
import os  # noqa: F401

# Rationale comment.
CONSTANT = "a # not a comment"


class Cache:
    """Class docstring explaining the choice."""

    def keep(self):
        """Only a docstring."""

    def value(self):
        """Value docstring."""
        text = """not a docstring"""
        return text  # trailing rationale
'''


def test_python_comments_and_docstrings_are_withheld():
    out, removed, reason = rv.strip_file("pkg/cache.py", PY, all_lines(PY))
    assert reason is None
    compile(out, "cache.py", "exec")
    assert "Module docstring" not in out and "Class docstring" not in out
    assert "Value docstring" not in out and "Rationale comment" not in out
    assert "trailing rationale" not in out
    assert "# noqa: F401" in out and '"a # not a comment"' in out
    assert 'text = """not a docstring"""' in out
    assert "    def keep(self):\n        pass\n" in out
    kinds = sorted(item["kind"] for item in removed)
    assert kinds.count("docstring") == 4 and kinds.count("comment") == 2


def test_python_docstring_on_unchanged_lines_is_kept():
    text = 'def f():\n    """Base docstring."""\n    return 1  # new\n'
    out, removed, _ = rv.strip_file("a.py", text, lines(3))
    assert '"""Base docstring."""' in out and "# new" not in out
    assert len(removed) == 1


def test_python_tokenize_failure_leaves_file_unchanged():
    text = 'x = """unterminated\n'
    out, removed, reason = rv.strip_file("a.py", text, all_lines(text))
    assert out == text and removed == [] and reason.startswith("tokenize failed")


def test_unknown_extensions_are_untouched():
    text = "# config comment\nkey = 1\n"
    assert rv.strip_file("Cargo.toml", text, all_lines(text)) == (text, [], None)


# ---------------------------------------------------------------------------
# end to end in a git worktree
# ---------------------------------------------------------------------------

def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.email", "t@example.com")
    git(root, "config", "user.name", "t")
    (root / "src").mkdir()
    (root / "tests").mkdir()
    (root / "src/core.py").write_text("def f():\n    return 1\n")
    (root / "tests/test_core.py").write_text("from src.core import f\n\n\ndef test_f():\n    assert f() == 1\n")
    git(root, "add", "-A")
    git(root, "commit", "-qm", "base")
    (root / "src/core.py").write_text(
        'def f():\n    """Return one because the task says so."""\n    # why\n    return 1\n'
    )
    (root / "tests/test_core.py").write_text(
        "from src.core import f\n\n\ndef helper():\n    return 1\n\n\ndef test_f():\n    assert f() == helper()\n"
    )
    (root / "tests/test_new.py").write_text("from tests.test_core import helper\n")
    patch = git(root, "diff", "--binary")
    git(root, "add", "-N", "tests/test_new.py")
    patch = git(root, "diff", "--binary")
    path = tmp_path / "p0.patch"
    path.write_text(patch)
    return root, path


@pytest.mark.parametrize("level, kept, gone", [
    ("strip-all-tests", [], ["tests/test_new.py"]),
    ("strip-new-tests", ["tests/test_core.py"], ["tests/test_new.py"]),
    ("strip-comments", ["tests/test_core.py", "tests/test_new.py"], []),
])
def test_build_view_levels(repo, level, kept, gone):
    root, patch = repo
    manifest = rv.build_view(str(root), str(patch), level)
    assert manifest["schema_version"] == rv.SCHEMA_VERSION and manifest["level"] == level
    core = (root / "src/core.py").read_text()
    assert "because the task says so" not in core and "# why" not in core
    compile(core, "core.py", "exec")
    for path in gone:
        assert not (root / path).exists()
    assert manifest["kept_test_files"] == sorted(kept)
    test_core = (root / "tests/test_core.py").read_text()
    assert ("helper" in test_core) == ("tests/test_core.py" in kept)
    assert manifest["stripped_item_count"] == 2


def test_container_entry_point_prints_manifest(repo):
    root, patch = repo
    script = Path(rv.__file__)
    result = subprocess.run(
        [sys.executable, "-I", str(script), "--worktree", str(root), "--patch", str(patch),
         "--level", "strip-all-tests"],
        check=True, capture_output=True, text=True,
    )
    manifest = json.loads(result.stdout)
    assert manifest["removed_test_files"] == ["tests/test_new.py"]
    assert manifest["reverted_test_files"] == ["tests/test_core.py"]
