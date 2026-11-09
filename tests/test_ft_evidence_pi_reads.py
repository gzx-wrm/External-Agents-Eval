"""Pi read conventions the frozen F comparison did not cover (offset < 1, CRLF files)."""
from __future__ import annotations

import asyncio

import pytest

from external_agents_eval.workflows.solve_review_repair.ft_evidence import (
    observed_evidence,
    pi_read_view,
)
from external_agents_eval.workflows.solve_review_repair.ft_protocol import returned_span

LF = b"one\ntwo\nthree\nfour\nfive\n"
CRLF = b"from pathlib import Path\r\n\r\nROOT_DIR = Path(__file__)\r\n"


class Workspace:
    async def read_frozen_file(self, *, path, **kwargs):
        return {"lf.py": (path, LF), "crlf.py": (path, CRLF)}.get(path)


def step(number, path, content, **arguments):
    return {
        "step_id": number,
        "tool_calls": [{"tool_call_id": str(number), "function_name": "read",
                        "arguments": {"path": "/base/" + path, **arguments}}],
        "observation": {"results": [{"source_call_id": str(number), "content": content}]},
    }


def collect(*steps):
    return asyncio.run(observed_evidence(
        {"steps": list(steps)}, task_id="task", base_root="/base", base_commit="base",
        workspace=Workspace(),
    ))


# Exactly what Pi 0.84.2 returns: startLine = offset ? max(0, offset - 1) : 0, split("\n").
FIRST_THREE = "one\ntwo\nthree\n\n[2 more lines in file. Use offset=4 to continue.]"


def test_frozen_comparison_alone_rejects_both_pi_conventions():
    with pytest.raises(ValueError, match="lines 0-3"):
        returned_span(FIRST_THREE, {"offset": 0, "limit": 3}, LF.decode())
    with pytest.raises(ValueError, match="lines 1-3"):
        returned_span(CRLF.decode(), {}, CRLF.decode())


@pytest.mark.parametrize("offset", [0, -2])
def test_offset_below_one_is_read_from_line_one(offset):
    packet = collect(step(1, "lf.py", FIRST_THREE, offset=offset, limit=3))
    assert [(e["line_start"], e["line_end"], e["text"]) for e in packet["evidence"]] == [
        (1, 3, "one\ntwo\nthree\n")]
    # Provenance keeps the arguments the model actually sent.
    assert packet["read_provenance"][0]["arguments"]["offset"] == offset


def test_crlf_file_keeps_raw_bytes_in_evidence():
    packet = collect(step(1, "crlf.py", CRLF.decode()))
    assert [(e["line_start"], e["line_end"]) for e in packet["evidence"]] == [(1, 3)]
    assert packet["evidence"][0]["text"] == CRLF.decode()


def test_previously_accepted_reads_are_unchanged():
    reads = [step(1, "lf.py", "one\ntwo", limit=2),
             step(2, "lf.py", "three\nfour\n\n[1 more lines in file. Use offset=5 to continue.]",
                  offset=3, limit=2),
             step(3, "lf.py", LF.decode())]
    for read in reads:
        arguments = read["tool_calls"][0]["arguments"]
        content = read["observation"]["results"][0]["content"]
        assert pi_read_view(content, arguments, LF) == (content, arguments)
    packet = collect(*reads)
    assert [(e["line_start"], e["line_end"]) for e in packet["evidence"]] == [(1, 5)]


def test_genuine_differences_still_fail():
    with pytest.raises(ValueError, match="differs from frozen baseline"):
        collect(step(1, "lf.py", "one\nTWO\nthree", limit=3))
    with pytest.raises(ValueError, match="differs from frozen baseline"):
        collect(step(1, "crlf.py", "from pathlib import Path\r\n\r\nROOT = 1\r\n"))
    with pytest.raises(ValueError, match="differs from frozen baseline"):
        # A CRLF return for an LF baseline is not normalized.
        collect(step(1, "lf.py", "one\r\ntwo", limit=2))
