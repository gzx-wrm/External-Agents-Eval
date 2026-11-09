"""V6.5 review view: a behavioral projection of P0 for the I/R/D/S reviewers.

This file runs inside the task container (old task-image Python 3, stdlib only,
no f-strings or annotations) and is imported on the host by the controller tests.

Given a worktree where P0 has been applied on top of the environment baseline and
the P0 patch file itself, it withholds from P0's *added* lines:

- comments and docstrings that P0 added (comment spans entirely on added lines);
  directives that change tool behavior (go:embed, @ts-expect-error, noqa, ...) stay;
- depending on the level, P0's changes to test files, reverted with ``git apply -R``.

Levels (strongest first; the controller falls back when a level breaks the build):
  strip-all-tests  comments/docstrings withheld, every P0 test-file change reverted
  strip-new-tests  comments/docstrings withheld, only test files P0 added removed
  strip-comments   comments/docstrings withheld, test files kept

The worktree is edited in place; a JSON manifest is printed on stdout. The
controller captures the resulting worktree diff as the view patch.
"""
import argparse
import io
import json
import os
import re
import subprocess
import sys
import tokenize

SCHEMA_VERSION = "v6.5-review-view-1"
LEVELS = ("strip-all-tests", "strip-new-tests", "strip-comments")
MAX_ITEM_TEXT = 2000

TEST_DIRS = {"test", "tests", "__tests__", "__snapshots__", "testdata", "spec", "specs", "e2e"}
TEST_NAME_PATTERNS = (
    re.compile(r"^test_.*\.py$"),
    re.compile(r".*_test\.py$"),
    re.compile(r"^conftest\.py$"),
    re.compile(r".*_test\.go$"),
    re.compile(r".*\.(test|spec)\.[cm]?[jt]sx?$"),
    re.compile(r".*_test\.[cm]?[jt]sx?$"),
    re.compile(r".*\.snap$"),
    re.compile(r".*(Test|Tests|Spec)\.(java|kt|kts|scala)$"),
)

C_LIKE = {
    ".js": "js", ".mjs": "js", ".cjs": "js", ".ts": "js", ".mts": "js", ".cts": "js",
    ".jsx": "jsx", ".tsx": "jsx",
    ".go": "go", ".rs": "rust", ".java": "java", ".kt": "kotlin", ".kts": "kotlin",
    ".scala": "scala", ".swift": "swift", ".dart": "dart", ".cs": "csharp",
    ".c": "c", ".h": "c", ".cc": "c", ".cpp": "c", ".cxx": "c", ".hpp": "c", ".hh": "c",
}
PYTHON = {".py", ".pyi"}
NESTED_BLOCKS = {"rust", "kotlin", "swift", "scala", "dart"}
# Inline (non-full-line) comment removal is unsafe where JSX text can contain '//'.
FULL_LINE_ONLY = {"jsx"}

LINE_DIRECTIVE = re.compile(
    r"^(\+build|nolint|lint:|@ts-|eslint|prettier-ignore|istanbul |c8 |"
    r"deno-lint-ignore|deno-fmt-ignore|biome-ignore|#region|#endregion|<reference|<amd-|"
    r"NOSONAR|swiftlint:|ktlint|@flow|@jsx|@refresh|noinspection|language=|tslint:|jshint|"
    r"jscs:|webpack|@vite-ignore|oxlint|@generated|@license|@preserve)"
)
# Go compiler directives are written without a space after '//'.
GO_DIRECTIVE = re.compile(r"^//(go:|line |export |nolint|extern |cgo )")
BLOCK_DIRECTIVE = re.compile(
    r"^(eslint|istanbul|c8 |@__PURE__|#__PURE__|webpack|@vite-ignore|prettier-ignore|global |"
    r"jshint|@jsx|@flow|@license|@preserve|tslint:|@ts-|!)"
)
PY_DIRECTIVE = re.compile(
    r"^(type:|noqa|pragma|pylint:|fmt:|isort:|mypy:|pyright:|ruff:|nosec|pyre-|-\*-|!|"
    r"flake8:|yapf:|coverage:|doctest:)"
)
JS_REGEX_PRECEDERS = set("(,=:[!&|?{};+-*%<>~^")
JS_REGEX_KEYWORDS = {"return", "typeof", "case", "do", "else", "in", "of", "new", "delete",
                     "void", "throw", "yield", "await", "instanceof"}


# ---------------------------------------------------------------------------
# patch parsing
# ---------------------------------------------------------------------------

def _unquote(path):
    if len(path) >= 2 and path[0] == '"' and path[-1] == '"':
        body = path[1:-1]
        raw = bytearray()
        index = 0
        escapes = {"n": 10, "t": 9, '"': 34, "\\": 92, "a": 7, "b": 8, "f": 12, "v": 11, "r": 13}
        while index < len(body):
            char = body[index]
            if char == "\\" and index + 1 < len(body):
                following = body[index + 1]
                if following in escapes:
                    raw.append(escapes[following])
                    index += 2
                    continue
                if re.match(r"[0-7]{3}", body[index + 1:index + 4]):
                    raw.append(int(body[index + 1:index + 4], 8))
                    index += 4
                    continue
            raw.extend(char.encode("utf-8"))
            index += 1
        return raw.decode("utf-8", "replace")
    return path


def _strip_prefix(path):
    path = _unquote(path.strip())
    if path == "/dev/null":
        return None
    if path[:2] in ("a/", "b/"):
        return path[2:]
    return path


def parse_patch(text):
    """Return file changes with the post-image line numbers P0 added."""
    changes = []
    current = None
    new_line = None
    in_hunk = False
    for line in text.splitlines():
        if line.startswith("diff --git "):
            current = {"path": None, "old_path": None, "status": "modified",
                       "binary": False, "added_lines": set()}
            # Fallback for entries without ---/+++ lines (binary, mode-only).
            header = re.match(r'^diff --git ("a/.*?"|a/\S+) ("b/.*"|b/\S+)$', line)
            if header:
                current["header_path"] = _strip_prefix(header.group(2))
            changes.append(current)
            in_hunk = False
            continue
        if current is None:
            continue
        if line.startswith("@@ "):
            match = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
            new_line = int(match.group(1)) if match else None
            in_hunk = new_line is not None
            continue
        if in_hunk:
            if line.startswith("+"):
                current["added_lines"].add(new_line)
                new_line += 1
                continue
            if line.startswith(" ") or line == "":
                new_line += 1
                continue
            if line.startswith("-") or line.startswith("\\"):
                continue
            in_hunk = False
        if line.startswith("new file mode"):
            current["status"] = "added"
        elif line.startswith("deleted file mode"):
            current["status"] = "deleted"
        elif line.startswith("rename to "):
            current["status"] = "renamed"
            current["path"] = _unquote(line[len("rename to "):])
        elif line.startswith("rename from "):
            current["old_path"] = _unquote(line[len("rename from "):])
        elif line.startswith("Binary files ") or line.startswith("GIT binary patch"):
            current["binary"] = True
        elif line.startswith("--- "):
            current["old_path"] = current["old_path"] or _strip_prefix(line[4:])
        elif line.startswith("+++ "):
            path = _strip_prefix(line[4:])
            if path is not None:
                current["path"] = path
    for change in changes:
        header_path = change.pop("header_path", None)
        if change["path"] is None:
            change["path"] = change["old_path"] or header_path
    return [change for change in changes if change["path"]]


def is_test_path(path):
    parts = path.split("/")
    if any(part in TEST_DIRS for part in parts[:-1]):
        return True
    name = parts[-1]
    return any(pattern.match(name) for pattern in TEST_NAME_PATTERNS)


# ---------------------------------------------------------------------------
# C-like comment scanning
# ---------------------------------------------------------------------------

def _line_starts(text):
    starts = [0]
    for index, char in enumerate(text):
        if char == "\n":
            starts.append(index + 1)
    return starts


def _line_of(starts, offset):
    low, high = 0, len(starts) - 1
    while low < high:
        middle = (low + high + 1) // 2
        if starts[middle] <= offset:
            low = middle
        else:
            high = middle - 1
    return low + 1


def scan_c_like(text, lang):
    """Return comment spans (start, end, kind) outside strings, regexes and templates."""
    spans = []
    length = len(text)
    index = 0
    # Mode stack for JS template literals: ("code", brace_depth) or ("template",).
    stack = [["code", 0]]
    last_sig = ""
    last_word = ""
    while index < length:
        mode = stack[-1]
        char = text[index]
        following = text[index + 1] if index + 1 < length else ""
        if mode[0] == "template":
            if char == "\\":
                index += 2
                continue
            if char == "`":
                stack.pop()
                last_sig, last_word = "`", ""
                index += 1
                continue
            if char == "$" and following == "{":
                stack.append(["code", 0])
                index += 2
                continue
            index += 1
            continue
        # code mode
        if char == "/" and following == "/":
            end = text.find("\n", index)
            end = length if end < 0 else end
            spans.append((index, end, "line"))
            index = end
            continue
        if char == "/" and following == "*":
            depth = 1
            cursor = index + 2
            while cursor < length and depth:
                if lang in NESTED_BLOCKS and text.startswith("/*", cursor):
                    depth += 1
                    cursor += 2
                elif text.startswith("*/", cursor):
                    depth -= 1
                    cursor += 2
                else:
                    cursor += 1
            spans.append((index, cursor, "block"))
            index = cursor
            continue
        if lang in ("js", "jsx") and char == "`":
            stack.append(["template"])
            index += 1
            continue
        if lang in ("js", "jsx") and char == "/" and (
                last_sig in JS_REGEX_PRECEDERS or last_sig == "" or last_word in JS_REGEX_KEYWORDS):
            cursor = index + 1
            in_class = False
            while cursor < length and text[cursor] != "\n":
                current = text[cursor]
                if current == "\\":
                    cursor += 2
                    continue
                if current == "[":
                    in_class = True
                elif current == "]":
                    in_class = False
                elif current == "/" and not in_class:
                    break
                cursor += 1
            index = cursor + 1
            last_sig, last_word = "/", ""
            continue
        if char in ('"', "'") or (char == "`" and lang == "go"):
            if char == "'" and lang == "rust":
                if following == "\\" or (index + 2 < length and text[index + 2] == "'"):
                    pass  # character literal
                else:
                    index += 1  # lifetime or label
                    last_sig, last_word = "'", ""
                    continue
            triple = text.startswith(char * 3, index) and char == '"' and lang in (
                "kotlin", "java", "scala", "swift", "dart", "csharp")
            if triple:
                end = text.find('"""', index + 3)
                end = length if end < 0 else end + 3
                index = end
                last_sig, last_word = '"', ""
                continue
            raw = char == "`" or (lang == "csharp" and index > 0 and text[index - 1] == "@")
            cursor = index + 1
            while cursor < length:
                current = text[cursor]
                if raw:
                    if current == char:
                        if lang == "csharp" and text.startswith(char * 2, cursor):
                            cursor += 2
                            continue
                        break
                elif current == "\\":
                    cursor += 2
                    continue
                elif current == char or (current == "\n" and char != "`"):
                    break
                cursor += 1
            index = cursor + 1
            last_sig, last_word = char, ""
            continue
        if lang == "rust" and (char == "r" or text.startswith("br", index)):
            # Identifiers are consumed whole below, so this is a token start.
            match = re.match(r"(?:br|r)(#*)\"", text[index:index + 260])
            if match:
                hashes = match.group(1)
                closing = '"' + hashes
                end = text.find(closing, index + len(match.group(0)))
                index = length if end < 0 else end + len(closing)
                last_sig, last_word = '"', ""
                continue
        if mode[0] == "code" and len(stack) > 1:
            if char == "{":
                mode[1] += 1
            elif char == "}":
                if mode[1] == 0:
                    stack.pop()
                    index += 1
                    continue
                mode[1] -= 1
        if char.isalnum() or char in "_$":
            cursor = index
            while cursor < length and (text[cursor].isalnum() or text[cursor] in "_$"):
                cursor += 1
            last_word = text[index:cursor]
            last_sig = last_word[-1]
            index = cursor
            continue
        if not char.isspace():
            last_sig, last_word = char, ""
        index += 1
    return spans


def _comment_body(text, kind):
    if kind == "line":
        body = text.lstrip("/").lstrip("!")
    else:
        body = text[2:]
        if body.endswith("*/"):
            body = body[:-2]
        body = body.lstrip("*")
    return body.strip()


def _is_c_directive(text, kind, lang):
    body = _comment_body(text, kind)
    if kind == "line":
        if text.startswith("///") and body.startswith("<"):
            return True
        if lang == "go" and GO_DIRECTIVE.match(text):
            return True
        return bool(LINE_DIRECTIVE.match(body))
    if text.startswith("/*!"):
        return True
    return bool(BLOCK_DIRECTIVE.match(body))


def strip_c_like(text, added_lines, lang):
    """Remove comment spans entirely on added lines; return (text, removed)."""
    if lang == "go" and re.search(r'^\s*import\s+"C"', text, re.M):
        return text, [], "cgo file: comments may be compiler input"
    starts = _line_starts(text)
    lines = text.splitlines(True)
    delete_lines = set()
    cuts = []  # (start, end, replacement)
    removed = []
    for start, end, kind in scan_c_like(text, lang):
        first = _line_of(starts, start)
        last = _line_of(starts, max(start, end - 1))
        if not all(number in added_lines for number in range(first, last + 1)):
            continue
        comment = text[start:end]
        if _is_c_directive(comment, kind, lang):
            continue
        before = text[starts[first - 1]:start]
        line_end = starts[last] - 1 if last < len(starts) else len(text)
        after = text[end:line_end] if end <= line_end else ""
        full_line = not before.strip() and not after.strip()
        if full_line:
            for number in range(first, last + 1):
                delete_lines.add(number)
        elif lang in FULL_LINE_ONLY or first != last:
            # Cutting a multi-line inline comment would renumber lines; keep it.
            continue
        else:
            cuts.append((start, end, "" if kind == "line" else " "))
        removed.append({"line": first, "kind": "comment", "text": comment[:MAX_ITEM_TEXT]})
    if not removed:
        return text, [], None
    output = text
    for start, end, replacement in sorted(cuts, reverse=True):
        line_number = _line_of(starts, start)
        if line_number in delete_lines:
            continue
        output = output[:start] + replacement + output[end:]
    # Re-split after inline cuts; cuts never add or remove newlines.
    lines = output.splitlines(True)
    kept = []
    for number, line in enumerate(lines, 1):
        if number in delete_lines:
            continue
        if any(_line_of(starts, start) == number for start, _, _ in cuts):
            ending = "\r\n" if line.endswith("\r\n") else ("\n" if line.endswith("\n") else "")
            line = line.rstrip("\r\n").rstrip() + ending
        kept.append(line)
    return "".join(kept), removed, None


# ---------------------------------------------------------------------------
# Python comments and docstrings
# ---------------------------------------------------------------------------

def _python_tokens(text):
    reader = io.StringIO(text).readline
    return list(tokenize.generate_tokens(reader))


def strip_python(text, added_lines):
    """Remove added comments and docstrings; return (text, removed, skipped_reason)."""
    try:
        tokens = _python_tokens(text)
    except (tokenize.TokenError, IndentationError, SyntaxError) as error:
        return text, [], "tokenize failed: %s" % (error,)
    lines = text.splitlines(True)
    delete_lines = set()
    replace_lines = {}
    cuts = {}  # line -> column where a trailing comment starts
    removed = []
    noise = (tokenize.NL, tokenize.COMMENT)
    if hasattr(tokenize, "ENCODING"):
        noise = noise + (tokenize.ENCODING,)

    # Comments.
    for token in tokens:
        if token.type != tokenize.COMMENT:
            continue
        row, column = token.start
        if row not in added_lines:
            continue
        body = token.string[1:].strip()
        if row <= 2 and (token.string.startswith("#!") or "coding" in token.string):
            continue
        if PY_DIRECTIVE.match(body):
            continue
        prefix = lines[row - 1][:column] if row - 1 < len(lines) else ""
        if prefix.strip():
            cuts[row] = column
        else:
            delete_lines.add(row)
        removed.append({"line": row, "kind": "comment", "text": token.string[:MAX_ITEM_TEXT]})

    # Docstrings: a string-only statement that opens a module, class or def body.
    # expect: "module" (file start), "header" (after a def/class line ending in
    # ':'), "body" (first token of that body), or None.
    significant = [token for token in tokens if token.type not in noise]
    expect = "module"
    in_header = False
    index = 0
    while index < len(significant):
        token = significant[index]
        if token.type == tokenize.STRING and expect in ("module", "body"):
            cursor = index
            while cursor < len(significant) and significant[cursor].type == tokenize.STRING:
                cursor += 1
            following = significant[cursor] if cursor < len(significant) else None
            if following is not None and following.type in (tokenize.NEWLINE, tokenize.ENDMARKER):
                first_row = token.start[0]
                last_row = significant[cursor - 1].end[0]
                start_prefix = lines[first_row - 1][:token.start[1]]
                end_suffix = lines[last_row - 1][significant[cursor - 1].end[1]:]
                whole_lines = not start_prefix.strip() and (
                    not end_suffix.strip() or end_suffix.strip().startswith("#"))
                rows = range(first_row, last_row + 1)
                if whole_lines and all(row in added_lines for row in rows):
                    after = significant[cursor + 1] if cursor + 1 < len(significant) else None
                    empties_body = expect == "body" and (
                        after is None or after.type in (tokenize.DEDENT, tokenize.ENDMARKER))
                    for row in rows:
                        delete_lines.add(row)
                    if empties_body:
                        delete_lines.discard(first_row)
                        replace_lines[first_row] = start_prefix + "pass" + (
                            "\r\n" if lines[first_row - 1].endswith("\r\n") else "\n")
                    removed.append({"line": first_row, "kind": "docstring",
                                    "text": "".join(lines[row - 1] for row in rows)[:MAX_ITEM_TEXT]})
            expect = None
            index = cursor
            continue
        if token.type == tokenize.NAME and token.string in ("def", "class"):
            in_header = True
        if token.type == tokenize.NEWLINE:
            previous = significant[index - 1] if index > 0 else None
            expect = "header" if in_header and previous is not None and previous.string == ":" else None
            in_header = False
        elif token.type == tokenize.INDENT:
            expect = "body" if expect == "header" else None
        else:
            expect = None
        index += 1

    if not removed:
        return text, [], None
    output = []
    for number, line in enumerate(lines, 1):
        if number in replace_lines:
            output.append(replace_lines[number])
            continue
        if number in delete_lines:
            continue
        if number in cuts:
            ending = "\r\n" if line.endswith("\r\n") else ("\n" if line.endswith("\n") else "")
            line = line[:cuts[number]].rstrip() + ending
        output.append(line)
    return "".join(output), removed, None


def strip_file(path, text, added_lines):
    extension = os.path.splitext(path)[1].lower()
    if extension in PYTHON:
        return strip_python(text, added_lines)
    lang = C_LIKE.get(extension)
    if lang is None:
        return text, [], None
    return strip_c_like(text, added_lines, lang)


# ---------------------------------------------------------------------------
# container entry point
# ---------------------------------------------------------------------------

def _git(worktree, *args):
    process = subprocess.Popen(("git",) + args, cwd=worktree, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise RuntimeError("git %s failed: %s" % (" ".join(args), stderr.decode("utf-8", "replace")))
    return stdout


def build_view(worktree, patch_path, level):
    if level not in LEVELS:
        raise ValueError("unknown review-view level: %s" % (level,))
    with io.open(patch_path, encoding="utf-8", errors="replace") as handle:
        changes = parse_patch(handle.read())
    tests = [change for change in changes if is_test_path(change["path"])]
    if level == "strip-all-tests":
        reverted = tests
    elif level == "strip-new-tests":
        reverted = [change for change in tests if change["status"] == "added"]
    else:
        reverted = []
    if reverted:
        includes = []
        for change in reverted:
            for path in (change["path"], change["old_path"]):
                if path:
                    includes.append("--include=" + path)
        _git(worktree, "apply", "-R", "--whitespace=nowarn", *(includes + [patch_path]))
    reverted_paths = set(change["path"] for change in reverted)
    stripped, skipped = [], []
    for change in changes:
        if change["path"] in reverted_paths or change["status"] == "deleted" or change["binary"]:
            continue
        target = os.path.join(worktree, change["path"])
        if not os.path.isfile(target) or os.path.islink(target):
            continue
        try:
            with io.open(target, encoding="utf-8", newline="") as handle:
                text = handle.read()
        except (UnicodeDecodeError, OSError):
            skipped.append({"path": change["path"], "reason": "not utf-8 text"})
            continue
        new_text, removed, reason = strip_file(change["path"], text, change["added_lines"])
        if reason:
            skipped.append({"path": change["path"], "reason": reason})
        if removed and new_text != text:
            with io.open(target, "w", encoding="utf-8", newline="") as handle:
                handle.write(new_text)
            stripped.append({"path": change["path"], "items": removed})
    return {
        "schema_version": SCHEMA_VERSION,
        "level": level,
        "reverted_test_files": sorted(change["path"] for change in reverted
                                      if change["status"] != "added"),
        "removed_test_files": sorted(change["path"] for change in reverted
                                     if change["status"] == "added"),
        "kept_test_files": sorted(change["path"] for change in tests
                                  if change["path"] not in reverted_paths),
        "stripped": stripped,
        "stripped_item_count": sum(len(item["items"]) for item in stripped),
        "skipped": skipped,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--worktree", required=True)
    parser.add_argument("--patch", required=True)
    parser.add_argument("--level", required=True, choices=LEVELS)
    args = parser.parse_args(argv)
    manifest = build_view(args.worktree, args.patch, args.level)
    sys.stdout.write(json.dumps(manifest, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
