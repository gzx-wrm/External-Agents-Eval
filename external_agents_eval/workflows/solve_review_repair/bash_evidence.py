"""Map visible Bash output to frozen source lines without asking the model to select evidence."""
from __future__ import annotations

import fnmatch
import posixpath
import re
import shlex
from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise

from .artifacts import sha256_bytes
from .ft_protocol import FOOTER

PROTOCOL = "bash-observed-v1"
_READERS = {"cat", "head", "tail", "sed", "nl", "awk", "more", "less"}
_DISCOVERY = {"ls", "find", "grep", "rg", "wc", "pwd", "echo", "printf", "test",
              "true", "false", "sort", "uniq", "cut", "git"}
_NUMBERED = re.compile(r"^\s*(\d+)\t(.*)$")


@dataclass
class CommandInfo:
    paths: set[str]
    ranges: dict[str, tuple[int, int]]
    intervals: dict[str, list[tuple[int, int]]]
    discovery_only: bool
    reads: bool
    untracked_only: bool


def command_info(command: str, cwd: str, base_root: str, paths: set[str], lengths=None) -> CommandInfo:
    """Hints resolve ambiguous text; they never establish that text was actually returned."""
    lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>()")
    lexer.whitespace_split = True
    try:
        words = list(lexer)
    except ValueError:
        words = []
    segments, segment = [], []
    skip_redirect = False
    for word in words + [";"]:
        if skip_redirect:
            skip_redirect = False
            continue
        if word in {">", ">>", "2>", "2>>"}:
            if segment and segment[-1].isdigit():
                segment.pop()
            skip_redirect = True
            continue
        if word and all(char in ";&|<>()" for char in word):
            if segment:
                segments.append(segment)
            segment = []
        else:
            segment.append(word)
    selected, ranges, variables = set(), {}, {}
    intervals = defaultdict(list)
    def add_range(path, start, end):
        previous = ranges.get(path, (start, end))
        ranges[path] = (min(previous[0], start), max(previous[1], end))
        intervals[path].append((start, end))

    only_discovery, reads, untracked_only = bool(segments), False, False
    for words in segments:
        while words and re.fullmatch(r"[A-Za-z_]\w*=.*", words[0], re.DOTALL):
            key, value = words.pop(0).split("=", 1)
            variables[key] = value
        if not words:
            continue
        words = [re.sub(r"\$\{?(\w+)\}?", lambda m: variables.get(m[1], m[0]), w)
                 for w in words]
        name = posixpath.basename(words[0])
        if name in {"bash", "sh"} and len(words) >= 3 and words[1] in {"-c", "-lc"}:
            inner = command_info(words[2], cwd, base_root, paths, lengths)
            selected.update(inner.paths)
            for path, spans in inner.intervals.items():
                for start, end in spans:
                    add_range(path, start, end)
            only_discovery &= inner.discovery_only
            reads |= inner.reads
            untracked_only |= inner.untracked_only
            continue
        if name == "cd":
            if len(words) == 2:
                cwd = posixpath.normpath(posixpath.join(cwd, words[1]))
            continue
        operands = set()
        for word in words[1:]:
            absolute = posixpath.normpath(posixpath.join(cwd, word))
            if absolute.startswith(base_root.rstrip("/") + "/"):
                relative = absolute[len(base_root.rstrip("/")) + 1:]
                operands.update(path for path in paths if fnmatch.fnmatchcase(path, relative))
        if name in _READERS:
            selected.update(operands)
            if not operands and name in {"cat", "head", "tail", "sed", "nl"}:
                explicit = [w for w in words[1:] if not w.startswith("-") and not w.isdigit()
                            and "$" not in w and "`" not in w
                            and not re.fullmatch(r"\d+(?:,\d+|,\$)?p", w)]
                untracked_only |= bool(explicit)
            # head/tail used solely as a pipe filter are still discovery commands.
            if operands or name not in {"head", "tail"}:
                reads = True
                only_discovery = False
            for path in operands:
                if name in {"cat", "nl"}:
                    add_range(path, 1, 2**63)
                if name == "sed":
                    match = next((re.fullmatch(r"(\d+)(?:,(\d+|\$))?p", w)
                                  for w in words[1:] if re.fullmatch(r"(\d+)(?:,(\d+|\$))?p", w)), None)
                    if match:
                        start = int(match[1])
                        add_range(path, start, 2**63 if match[2] == "$" else int(match[2] or start))
                if name == "head":
                    count = 10
                    for i, word in enumerate(words[1:], 1):
                        if word in {"-n", "--lines"} and i + 1 < len(words):
                            count = int(words[i + 1]) if words[i + 1].isdigit() else count
                        elif re.fullmatch(r"-\d+", word):
                            count = int(word[1:])
                    add_range(path, 1, count)
                if name == "tail" and lengths is not None:
                    count = "10"
                    for i, word in enumerate(words[1:], 1):
                        if word in {"-n", "--lines"} and i + 1 < len(words):
                            count = words[i + 1]
                        elif re.fullmatch(r"-\d+", word):
                            count = word[1:]
                    if re.fullmatch(r"\+?\d+", count):
                        start = int(count[1:]) if count.startswith("+") else max(1, lengths[path] - int(count) + 1)
                        add_range(path, start, lengths[path])
        elif name not in _DISCOVERY:
            only_discovery = False
            reads |= bool(re.search(r"\bopen\s*\(|\.read_(?:text|bytes)\s*\(", command))
    # Heredocs, variables, and Python readers may have no statically resolvable operand.
    if not selected and not only_discovery:
        selected = {path for path in paths if path in command}
    if not only_discovery and re.search(r"\b(?:cat|head|tail|sed|nl|awk)\b", command):
        reads = True
    return CommandInfo(selected, ranges, dict(intervals), only_discovery, reads, untracked_only)


class BashSourceIndex:
    def __init__(self, files: dict[str, bytes]):
        self.files = files
        self.lines = {path: text.decode().splitlines() for path, text in files.items()}
        self.index = defaultdict(list)
        for path, lines in self.lines.items():
            for i, line in enumerate(lines):
                self.index[line].append((path, i))

    @staticmethod
    def _forms(line):
        forms = [(line, None)]
        numbered = _NUMBERED.fullmatch(line)
        if numbered:
            forms.append((numbered[2], int(numbered[1])))
        return forms

    def _direct_cat(self, command, cwd, base_root, visible):
        """Retain order when cat prints identical files or blank-only files."""
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>")
        lexer.whitespace_split = True
        try:
            words = list(lexer)
        except ValueError:
            return None
        segments, segment = [], []
        for word in words + [";"]:
            if word in {"|", "||", "&", ">", ">>", "<", "<<"}:
                return None
            if word in {";", "&&"}:
                if segment:
                    segments.append(segment)
                segment = []
            else:
                segment.append(word)
        plan = []
        for words in segments:
            if words[0] == "cd" and len(words) == 2:
                cwd = posixpath.normpath(posixpath.join(cwd, words[1]))
                continue
            if words[0] == "echo" and all(w not in {"-n", "-e", "-E"} for w in words[1:]):
                plan.append((None, 0, " ".join(words[1:]), None))
                continue
            if words[0] != "cat":
                return None
            number = 0 if any(w in {"-n", "--number"} for w in words) else None
            for word in words[1:]:
                if word in {"-n", "--number", "--"}:
                    continue
                if word.startswith("-"):
                    return None
                absolute = posixpath.normpath(posixpath.join(cwd, word))
                if not absolute.startswith(base_root.rstrip("/") + "/"):
                    return None
                relative = absolute[len(base_root.rstrip("/")) + 1:]
                paths = sorted(path for path in self.files if fnmatch.fnmatchcase(path, relative))
                if not paths:
                    return None
                for path in paths:
                    for row, text in enumerate(self.lines[path]):
                        if number is not None:
                            number += 1
                        plan.append((path, row, text, number))
        if not visible or len(visible) > len(plan):
            return None
        for i, line in enumerate(visible):
            if not any(text == plan[i][2] and number == plan[i][3]
                       for text, number in self._forms(line)):
                return None
        result = []
        for i, (path, row, _, _) in enumerate(plan[:len(visible)]):
            if path is None:
                continue
            if result and result[-1]["path"] == path and result[-1]["line_end"] == row:
                result[-1].update(line_end=row + 1, output_line_end=i + 1)
            else:
                result.append({"path": path, "line_start": row + 1, "line_end": row + 1,
                               "output_line_start": i + 1, "output_line_end": i + 1})
        return result

    def extract(self, command: str, output: str, *, cwd: str, base_root: str, returncode=0):
        info = command_info(command, cwd, base_root, set(self.files),
                            {path: len(lines) for path, lines in self.lines.items()})
        audit = {"command": command, "cwd": cwd, "returned_content_sha256": sha256_bytes(output.encode()),
                 "returncode": returncode,
                 "status": "discovery" if info.discovery_only else "no_source", "mapped_lines": 0}
        if output.lstrip().splitlines()[:1] == ["COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]:
            audit["status"] = "delivery"
            return [], audit
        if info.untracked_only and not info.paths:
            audit["status"] = "outside_or_untracked"
            return [], audit
        if info.discovery_only:
            return [], audit
        footer = FOOTER.search(output)
        body = output[:footer.start()] if footer else output
        visible = body.splitlines()
        direct = self._direct_cat(command, cwd, base_root, visible)
        if direct is not None:
            audit.update(status="mapped", mapped_lines=len(visible), unmapped_nonempty_output_lines=[])
            return direct, audit
        # Preserve Pi's EOF and trailing-newline conventions for an exact, located read.
        if len(info.paths) == 1:
            path = next(iter(info.paths))
            intervals = info.intervals.get(path, [])
            if len(intervals) == 1:
                start, end = intervals[0]
                if start > len(self.lines[path]):
                    audit.update(status="empty_read", unmapped_nonempty_output_lines=[])
                    return [], audit
                end = min(end, len(self.lines[path]))
                if footer:
                    end = min(end, int(re.search(r"Use offset=(\d+)", footer.group())[1]) - 1)
                expected = "\n".join(self.lines[path][start - 1:end])
                if start <= end and body.rstrip("\n") == expected.rstrip("\n"):
                    audit.update(status="mapped", mapped_lines=end - start + 1,
                                 unmapped_nonempty_output_lines=[])
                    return [{"path": path, "line_start": start, "line_end": end,
                             "output_line_start": 1, "output_line_end": len(visible)}], audit
        forms = [self._forms(line) for line in visible]
        result, ambiguous, covered = [], [], set()
        i = 0
        while i < len(visible):
            possibilities = set()
            for text, number in forms[i]:
                if not text.strip() and (not info.paths or result or i):
                    continue
                for path, row in self.index.get(text, []):
                    if number is None or row + 1 == number:
                        possibilities.add((path, row))
            preferred = {(path, row) for path, row in possibilities if path in info.paths}
            if preferred:
                possibilities = preferred
            candidates = []
            for path, row in possibilities:
                intervals = info.intervals.get(path, [(1, 2**63)])
                ends = [end for start, end in intervals if start <= row + 1 <= end]
                if not ends:
                    continue
                end = max(ends)
                source = self.lines[path]
                length = 0
                while i + length < len(visible) and row + length < len(source):
                    if row + length + 1 > end:
                        break
                    if not any(text == source[row + length] and
                               (number is None or number == row + length + 1)
                               for text, number in forms[i + length]):
                        break
                    length += 1
                if length:
                    candidates.append((length, path, row))
            if not candidates:
                i += 1
                continue
            longest = max(item[0] for item in candidates)
            best = [(path, row) for length, path, row in candidates if length == longest]
            # A known read start resolves repeated short lines without guessing by relevance.
            starts = [(path, row) for path, row in best
                      if any(row + 1 == start for start, _ in info.intervals.get(path, []))]
            if len(starts) == 1:
                best = starts
            if len(best) != 1:
                ambiguous.append({"output_line": i + 1, "length": longest,
                                  "candidates": [[p, r + 1] for p, r in best]})
                i += longest
                continue
            path, row = best[0]
            first = i
            lower = info.ranges.get(path, (1, 2**63))[0] - 1
            while first and row > lower and first - 1 not in covered:
                if not any(text == self.lines[path][row - 1] and
                           (number is None or number == row)
                           for text, number in forms[first - 1]):
                    break
                first -= 1
                row -= 1
            end = i + longest
            result.append({"path": path, "line_start": row + 1,
                           "line_end": row + end - first, "output_line_start": first + 1,
                           "output_line_end": end})
            covered.update(range(first, end))
            i = end
        audit["mapped_lines"] = len(covered)
        audit["unmapped_nonempty_output_lines"] = [i + 1 for i, line in enumerate(visible)
                                                 if line.strip() and i not in covered]
        if ambiguous:
            audit.update(status="ambiguous", ambiguities=ambiguous)
        elif result:
            audit["status"] = "mapped"
        elif info.reads and output.strip():
            audit["status"] = "failed_command" if returncode else "unmapped_read"
        # Unmatched lines between mapped portions of one read may be modified source.
        for left, right in pairwise(result):
            if (left["path"] == right["path"] and
                    right["output_line_start"] - left["output_line_end"] > 1 and
                    right["line_start"] - left["line_end"] ==
                    right["output_line_start"] - left["output_line_end"] and
                    any(start <= left["line_end"] and right["line_start"] <= end
                        for start, end in info.intervals.get(left["path"], []))):
                audit["status"] = "unmapped_read"
        return result, audit
