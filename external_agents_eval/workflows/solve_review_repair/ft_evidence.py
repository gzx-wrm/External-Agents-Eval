"""Build the historical FT read-observation packet from a live A trajectory."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .artifacts import sha256_bytes
from .bash_evidence import PROTOCOL, BashSourceIndex
from .ft_protocol import merge, returned_span


def pi_read_view(content: str, arguments: dict[str, Any], raw: bytes) -> tuple[str, dict[str, Any]]:
    """Map a Pi read onto the frozen F comparison without changing returned_span.

    Pi 0.84.2 starts at line 1 for any offset below 1 and splits text on "\n" only,
    so CRLF files come back with their "\r\n" intact.  The F protocol compares
    against LF-joined baseline lines.  Only reads that F would otherwise reject change.
    """
    view = dict(arguments)
    if "offset" in view and int(view["offset"]) < 1:
        view["offset"] = 1
    if b"\r\n" in raw:
        content = content.replace("\r\n", "\n")
    return content, view


async def observed_evidence(
    trajectory: dict[str, Any],
    *,
    task_id: str,
    base_root: str,
    base_commit: str,
    workspace: Any,
) -> dict[str, Any]:
    """Keep exactly the baseline lines returned by read, as in build_f_observed."""
    prefix = base_root.rstrip("/") + "/"
    files: dict[str, bytes] = {}
    resolved = {}
    spans = defaultdict(list)
    reads = []
    skipped = []
    tool_counts = Counter()
    inventories = []
    environment_exclusions = []
    bash_audits = []
    bash_index = None
    if trajectory.get("evidence_protocol") == PROTOCOL:
        bash_index = BashSourceIndex(await workspace.read_frozen_text_files(
            worktree=base_root, base_commit=base_commit,
        ))
    for step in trajectory["steps"]:
        results = {
            result["source_call_id"]: result["content"]
            for result in step.get("observation", {}).get("results", [])
            if "source_call_id" in result
        }
        returncodes = {result["source_call_id"]: result.get("exit_code", 0)
                       for result in step.get("observation", {}).get("results", [])
                       if "source_call_id" in result}
        text_returned = {result["source_call_id"]: result.get("source_text_returned", True)
                         for result in step.get("observation", {}).get("results", [])}
        for call in step.get("tool_calls", []):
            name = call["function_name"]
            tool_counts[name] += 1
            arguments = call["arguments"]
            content = results[call["tool_call_id"]]
            if name == "bash" and bash_index is not None:
                mapping_content = content
                excluded = None
                original = "_min_llvmlite_version = (0, 47, 0)"
                patched = "_min_llvmlite_version = (0, 46, 0)"
                numba_source = bash_index.files.get("numba/__init__.py", b"").decode()
                if (task_id == "numba-stencil-boundary-modes" and patched in content
                        and original in numba_source):
                    mapping_content = content.replace(patched, original)
                    excluded = numba_source.splitlines().index(original) + 1
                blocks, audit = bash_index.extract(
                    arguments["command"], mapping_content,
                    cwd=arguments.get("cwd", base_root), base_root=base_root,
                    returncode=returncodes.get(call["tool_call_id"], 0),
                )
                audit["returned_content_sha256"] = sha256_bytes(content.encode())
                if excluded is not None:
                    filtered = []
                    for block in blocks:
                        if block["path"] != "numba/__init__.py" or not (
                            block["line_start"] <= excluded <= block["line_end"]
                        ):
                            filtered.append(block)
                            continue
                        for first, last in ((block["line_start"], excluded - 1),
                                            (excluded + 1, block["line_end"])):
                            if first <= last:
                                filtered.append({**block, "line_start": first, "line_end": last,
                                                 "output_line_start": block["output_line_start"] + first - block["line_start"],
                                                 "output_line_end": block["output_line_start"] + last - block["line_start"]})
                        environment_exclusions.append({
                            "path": "numba/__init__.py", "line": excluded,
                            "baseline": original, "observed": patched,
                            "reason": "Environment dependency compatibility edit; excluded from frozen public evidence",
                            "tool_call_id": call["tool_call_id"],
                        })
                    blocks = filtered
                audit.update(tool_call_id=call["tool_call_id"], step_id=step["step_id"])
                bash_audits.append(audit)
                if audit["status"] == "discovery":
                    inventories.append({"tool": "bash", "arguments": arguments,
                                        "returned_content": content,
                                        "tool_call_id": call["tool_call_id"]})
                for block in blocks:
                    requested_path = block["path"]
                    if requested_path not in resolved:
                        resolved[requested_path] = await workspace.read_frozen_file(
                            worktree=base_root, base_commit=base_commit, path=requested_path,
                        )
                    frozen = resolved[requested_path]
                    if frozen is None:
                        raise ValueError(f"Mapped Bash source is not in the baseline: {requested_path}")
                    path, raw = frozen
                    if raw != bash_index.files[requested_path]:
                        raise ValueError(f"Bash source snapshot changed: {requested_path}")
                    files[path] = raw
                    start, end = block["line_start"], block["line_end"]
                    spans[path].append((start, end))
                    reads.append({
                        "path": path, "requested_path": requested_path,
                        "line_start": start, "line_end": end,
                        "arguments": arguments, "step_id": step["step_id"],
                        "tool_call_id": call["tool_call_id"],
                        "returned_content_sha256": sha256_bytes(content.encode()),
                        "extraction": PROTOCOL,
                        "output_line_start": block["output_line_start"],
                        "output_line_end": block["output_line_end"],
                    })
                continue
            if name in {"ls", "find"}:
                inventories.append({
                    "tool": name, "arguments": arguments,
                    "returned_content": content, "tool_call_id": call["tool_call_id"],
                })
            if name != "read":
                continue
            if returncodes.get(call["tool_call_id"], 0) or not text_returned.get(call["tool_call_id"], True):
                skipped.append({"path": arguments.get("path", ""),
                                "reason": "Read returned an error or a non-source notice, not source text",
                                "tool_call_id": call["tool_call_id"]})
                continue
            absolute = arguments.get("path", "")
            if not absolute.startswith(prefix):
                skipped.append({
                    "path": absolute, "reason": "Outside the declared frozen base_root",
                    "tool_call_id": call["tool_call_id"],
                })
                continue
            requested_path = absolute[len(prefix):]
            if requested_path not in resolved:
                resolved[requested_path] = await workspace.read_frozen_file(
                    worktree=base_root, base_commit=base_commit, path=requested_path,
                )
            frozen = resolved[requested_path]
            if frozen is None:
                skipped.append({
                    "path": requested_path, "reason": "Not a tracked file in the frozen baseline",
                    "tool_call_id": call["tool_call_id"],
                })
                continue
            path, raw = frozen
            files[path] = raw
            baseline = raw.decode()
            if int(arguments.get("offset", 1)) > len(baseline.splitlines()):
                skipped.append({
                    "path": path,
                    "reason": "Empty file or requested offset beyond EOF; no source lines returned",
                    "tool_call_id": call["tool_call_id"],
                })
                continue
            view_content, view_arguments = pi_read_view(content, arguments, raw)
            try:
                start, end = returned_span(view_content, view_arguments, baseline)
                spans[path].append((start, end))
            except ValueError:
                # Preserve the one documented compatibility exclusion in the FT builder.
                original = "_min_llvmlite_version = (0, 47, 0)"
                patched = "_min_llvmlite_version = (0, 46, 0)"
                if (
                    task_id != "numba-stencil-boundary-modes"
                    or path != "numba/__init__.py" or original not in baseline
                ):
                    raise
                start, end = returned_span(
                    view_content, view_arguments, baseline.replace(original, patched))
                changed_line = baseline.splitlines().index(original) + 1
                for first, last in ((start, changed_line - 1), (changed_line + 1, end)):
                    if first <= last:
                        spans[path].append((first, last))
                environment_exclusions.append({
                    "path": path, "line": changed_line, "baseline": original, "observed": patched,
                    "reason": "Environment dependency compatibility edit; excluded from frozen public evidence",
                    "tool_call_id": call["tool_call_id"],
                })
            reads.append({
                "path": path, "requested_path": requested_path, "line_start": start, "line_end": end,
                "arguments": arguments, "step_id": step["step_id"],
                "tool_call_id": call["tool_call_id"],
                "returned_content_sha256": sha256_bytes(content.encode()),
            })
    evidence = []
    for path, intervals in sorted(spans.items()):
        raw_lines = files[path].decode().splitlines(keepends=True)
        for start, end in merge(intervals):
            text = "".join(raw_lines[start - 1:end])
            evidence.append({
                "path": path, "line_start": start, "line_end": end, "text": text,
                "sha256": sha256_bytes(text.encode()), "full_file_sha256": sha256_bytes(files[path]),
            })
    packet = {
        "evidence": evidence, "read_provenance": reads, "skipped_reads": skipped,
        "tool_counts": dict(tool_counts), "repository_inventory_provenance": inventories,
        "environment_modified_lines_excluded": environment_exclusions,
    }
    if bash_index is not None:
        packet["bash_evidence_audit"] = {
            "protocol": PROTOCOL, "calls": bash_audits,
            "errors": [a for a in bash_audits if a["status"] in {"ambiguous", "unmapped_read"}],
            "mapped_lines": sum(a["mapped_lines"] for a in bash_audits),
        }
    return packet
