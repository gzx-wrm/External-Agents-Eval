"""Build the E10 source package for the 47 new train60 tasks.

Input is ``data/train60_selection_v1/selection.json``.  Each row is the
teacher's V6.1 A session converted exactly like the 30-task package
(E protocol system prompt, /work/base root, validated tool calls), with the
selected label Ledger as the final reply and teacher thinking kept in
``reasoning_content`` for ``compress_teacher_thinking.py``.

Annotated tasks with a ``trajectory.calibrated.json`` take each step's
reasoning from it, after checking that every tool call and observation is
unchanged; the calibrated reasoning is what explains the annotated Ledger.

Repository citations must fall inside lines that A's read calls actually
returned.  Three mechanical corrections are applied and recorded: an end
line past the returned range by at most ``MAX_END_OVERSHOOT`` lines is
narrowed, a range spanning an unread gap is split into its returned parts,
and a never-returned source is dropped when the entry keeps other sources.
Anything else fails the build.  A call to a tool outside the A tool set
that only returned "Tool <name> not found" is removed together with its
error, as E2 removed the repeated unavailable-rg calls.  No model is called.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from jsonschema import Draft202012Validator

try:
    from .build_e_teacher_trajectory import FIELDS, convert_session, dump, ledger_target, sha, write_json
except ImportError:
    from build_e_teacher_trajectory import FIELDS, convert_session, dump, ledger_target, sha, write_json

HERE = Path(__file__).resolve().parent
DEFAULT_SELECTION = HERE / "data/train60_selection_v1/selection.json"
DEFAULT_PROTOCOL = HERE / "data/e9_1_normal13_rationale_closing_v1/protocol"
DEFAULT_OUTPUT = HERE / "data/e10_new47_source_v1"
ANNOTATIONS = HERE.parent / "results/deepswe/manual_annotations"
FOOTER = re.compile(r"\n\n\[(?:Showing lines [^\n]+|\d+ more lines in file\. Use offset=\d+ to continue\.)\]$")
READ_ERRORS = ("Offset ", "EISDIR", "ENOENT", "Path not found")
MAX_END_OVERSHOOT = 3
UNKNOWN_TOOL = re.compile(r"Tool \S+ not found")


def calibrated_trajectory(task_id: str, label_ledger: Path) -> Path | None:
    """The calibrated trajectory of the annotation whose Ledger equals the selected label, if any.

    Passing reruns copy the annotation Ledger into the trial, so the match is
    by content; it also tells the annotation rounds of the same task apart.
    """

    label = ledger_target(label_ledger)
    matches = []
    for path in sorted(ANNOTATIONS.glob("**/trajectory.calibrated.json")):
        ledger = path.parent / "ledger.json"
        provenance = path.parent / "provenance.json"
        owner = json.loads(provenance.read_text()).get("task_id") if provenance.exists() else path.parent.name
        if owner == task_id and ledger.exists() and ledger_target(ledger) == label:
            matches.append(path)
    if len({sha(path) for path in matches}) > 1:
        raise ValueError(f"Several calibrated trajectories match the label of {task_id}: {matches}")
    return matches[0] if matches else None


def apply_calibrated_reasoning(events: list[dict], original: dict, calibrated: dict) -> list[dict]:
    """Replace per-step thinking with the calibrated text; tool calls and observations must match."""

    def agent(steps: list[dict]) -> list[dict]:
        return [step for step in steps if step.get("source") == "agent"]

    before, after = agent(original["steps"]), agent(calibrated["steps"])
    if len(before) != len(after):
        raise ValueError("Calibrated trajectory has a different number of agent steps")
    for old, new in zip(before, after, strict=True):
        if [(c["function_name"], c["arguments"]) for c in old.get("tool_calls") or []] != \
                [(c["function_name"], c["arguments"]) for c in new.get("tool_calls") or []]:
            raise ValueError(f"Calibrated tool calls differ at step {old['step_id']}")
        if old.get("observation") != new.get("observation"):
            raise ValueError(f"Calibrated observations differ at step {old['step_id']}")
    events = copy.deepcopy(events)
    assistants = [event["message"] for event in events
                  if event.get("type") == "message" and event["message"]["role"] == "assistant"]
    if len(assistants) != len(before):
        raise ValueError("Session assistant messages do not align with trajectory agent steps")
    changed = []
    for index, (message, old, new) in enumerate(zip(assistants, before, after, strict=True)):
        thinking = "\n".join(block["thinking"] for block in message["content"] if block["type"] == "thinking")
        if thinking != (old.get("reasoning_content") or ""):
            raise ValueError(f"Session thinking differs from the original trajectory at step {old['step_id']}")
        replacement = new.get("reasoning_content") or ""
        if replacement == thinking:
            continue
        others = [block for block in message["content"] if block["type"] != "thinking"]
        message["content"] = ([{"type": "thinking", "thinking": replacement}] if replacement else []) + others
        changed.append({"agent_step": index + 1, "step_id": old["step_id"],
                        "original_characters": len(thinking), "calibrated_characters": len(replacement)})
    return events, changed


def remove_unknown_tool_calls(events: list[dict], tools: list[dict]) -> tuple[list[dict], list[dict]]:
    """Drop calls to tools A does not have whose only result was the not-found error."""

    allowed = {tool["function"]["name"] for tool in tools}
    results = {event["message"]["toolCallId"]: event["message"] for event in events
               if event.get("type") == "message" and event["message"]["role"] == "toolResult"}
    removed, kept = [], []
    for event in events:
        message = event.get("message") if event.get("type") == "message" else None
        if message and message["role"] == "toolResult" and message["toolCallId"] in {item["id"] for item in removed}:
            continue
        if message and message["role"] == "assistant":
            blocks = []
            for block in message["content"]:
                if block["type"] == "toolCall" and block["name"] not in allowed:
                    result = results.get(block["id"])
                    text = "".join(part.get("text", "") for part in result["content"]) if result else ""
                    if not result or not result.get("isError") or not UNKNOWN_TOOL.fullmatch(text.strip()):
                        raise ValueError(f"Call to unavailable tool {block['name']} needs manual review")
                    removed.append({"id": block["id"], "name": block["name"], "arguments": block["arguments"],
                                    "result": text})
                    continue
                blocks.append(block)
            if not any(block["type"] in ("toolCall", "text") for block in blocks) and len(blocks) != len(message["content"]):
                raise ValueError("Removing an unavailable-tool call would empty an assistant turn")
            event = {**event, "message": {**message, "content": blocks}}
        kept.append(event)
    return kept, removed


def returned_ranges(messages: list[dict]) -> dict[str, list[list[int]]]:
    """Merged line ranges returned by successful read calls, keyed by repository-relative path."""

    calls = {call["id"]: call["function"] for message in messages if message["role"] == "assistant"
             for call in message.get("tool_calls", [])}
    ranges: dict[str, list[tuple[int, int]]] = {}
    for message in messages:
        if message["role"] != "tool" or message["name"] != "read":
            continue
        arguments, content = calls[message["tool_call_id"]]["arguments"], message["content"]
        if content.startswith(READ_ERRORS):
            continue
        path = arguments["path"]
        path = path.removeprefix("/work/base/") if path.startswith("/work/base/") else path.removeprefix("./")
        if path.startswith("/"):
            continue
        start = int(arguments.get("offset", 1))
        footer = FOOTER.search(content)
        end = (int(re.search(r"Use offset=(\d+)", footer.group()).group(1)) - 1 if footer
               else start + len(content.rstrip("\n").split("\n")) - 1)
        ranges.setdefault(path, []).append((start, end))
    merged = {}
    for path, intervals in ranges.items():
        result = []
        for start, end in sorted(intervals):
            if result and start <= result[-1][1] + 1:
                result[-1][1] = max(result[-1][1], end)
            else:
                result.append([start, end])
        merged[path] = result
    return merged


def correct_citations(target: dict, ranges: dict[str, list[list[int]]]) -> list[dict]:
    corrections = []
    for field in FIELDS:
        for entry_index, entry in enumerate(target[field]):
            sources = []
            for source_index, source in enumerate(entry["sources"]):
                if source["kind"] != "repository":
                    sources.append(source)
                    continue
                start, end = source["line_start"], source["line_end"]
                spans = ranges.get(source["path"], [])
                if any(a <= start and end <= b for a, b in spans):
                    sources.append(source)
                    continue
                parts = [[max(a, start), min(b, end)] for a, b in spans if a <= end and start <= b]
                record = {"field": field, "entry_index": entry_index, "source_index": source_index,
                          "before": copy.deepcopy(source)}
                if len(parts) == 1 and parts[0][0] == start and end - parts[0][1] <= MAX_END_OVERSHOOT:
                    after = [{**source, "line_end": parts[0][1]}]
                    reason = f"End line exceeded the returned read range by {end - parts[0][1]}; narrowed to the returned end"
                elif len(parts) > 1:
                    after = [{**source, "line_start": a, "line_end": b} for a, b in parts]
                    reason = "Range spanned lines no read returned; split into the returned parts"
                elif not parts and len(entry["sources"]) > 1:
                    after = []
                    reason = "No read returned these lines; removed because the entry keeps other sources"
                else:
                    raise ValueError(f"Citation needs manual review: {field}[{entry_index}] {source}")
                sources.extend(after)
                corrections.append({**record, "after": after, "reason": reason})
            entry["sources"] = sources
    return corrections


def check_target(target: dict, task: dict, ranges: dict, schema: Draft202012Validator) -> None:
    errors = sorted(schema.iter_errors(target), key=lambda error: list(error.path))
    if errors:
        raise ValueError(f"Label Ledger fails the SpecOutput schema: {errors[0].message}")
    for field in FIELDS:
        for entry in target[field]:
            for source in entry["sources"]:
                if source["kind"] == "task" and source["quote"] not in task[source["field"]]:
                    raise ValueError(f"Task quote not found: {source['quote'][:80]}")
                if source["kind"] == "repository" and not any(
                        a <= source["line_start"] and source["line_end"] <= b
                        for a, b in ranges.get(source["path"], [])):
                    raise ValueError(f"Repository source outside returned reads: {source}")


def build(selection_path: Path, protocol: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    selection = json.loads(selection_path.read_text())
    system = (protocol / "system-prompt.txt").read_text()
    tools = json.loads((protocol / "tools.json").read_text())
    schema = Draft202012Validator(json.loads((protocol / "spec_output_schema.json").read_text()))
    output.mkdir(parents=True)
    for name in ("records", "protocol", "provenance"):
        (output / name).mkdir()
    for name in ("system-prompt.txt", "tools.json", "spec_output_schema.json"):
        shutil.copyfile(protocol / name, output / "protocol" / name)
    shutil.copyfile(selection_path, output / "provenance/selection.json")
    shutil.copyfile(Path(__file__), output / "provenance/build_e10_source.py")

    rows, samples = [], []
    for item in selection["new_training_tasks"]:
        task_id = item["task_id"]
        session, label = Path(item["a_session"]), Path(item["label_ledger"])
        if sha(session) != item["a_session_sha256"] or sha(label) != item["label_ledger_sha256"]:
            raise ValueError(f"Selected source changed since selection: {task_id}")
        events = [json.loads(line) for line in session.read_text().splitlines()]
        calibrated = calibrated_trajectory(task_id, label)
        reasoning_changes = []
        if calibrated:
            events, reasoning_changes = apply_calibrated_reasoning(
                events, json.loads((session.parent / "trajectory.json").read_text()),
                json.loads(calibrated.read_text()))
        events, unknown_calls = remove_unknown_tool_calls(events, tools)
        target = ledger_target(label)
        messages, report = convert_session(events, target, system, tools)
        ranges = returned_ranges(messages)
        corrections = correct_citations(target, ranges)
        check_target(target, report["task"], ranges, schema)
        messages[-1]["content"] = dump(target)
        samples.append({"messages": messages, "tools": tools})
        record = {
            "task_id": task_id, "label_source": item["label_source"],
            "reasoning_source": "calibrated" if reasoning_changes else "teacher_session",
            "calibrated_trajectory": str(calibrated) if calibrated else None,
            "calibrated_trajectory_sha256": sha(calibrated) if calibrated else None,
            "calibrated_reasoning_steps": reasoning_changes,
            "a_session": str(session), "a_session_sha256": item["a_session_sha256"],
            "label_ledger": str(label), "label_ledger_sha256": item["label_ledger_sha256"],
            "citation_corrections": corrections, "removed_unknown_tool_calls": unknown_calls,
            "conversion": report,
        }
        write_json(output / "records" / f"{task_id}.json", record)
        rows.append({"row_index": len(rows), "task_id": task_id, "label_source": item["label_source"],
                     "reasoning_source": record["reasoning_source"],
                     "citation_corrections": len(corrections), "removed_unknown_tool_calls": len(unknown_calls),
                     "assistant_messages": report["assistant_messages"], "tool_calls": report["tool_calls"],
                     "thinking_characters": report["thinking_characters"]})
        print(f"{task_id}: {report['tool_calls']} calls, {len(corrections)} citation corrections, "
              f"reasoning={record['reasoning_source']}", flush=True)

    (output / "train.jsonl").write_text("".join(dump(sample) + "\n" for sample in samples))
    manifest = {
        "schema_version": "e10_new47_source_v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_sha256": sha(selection_path), "sample_count": len(rows),
        "format": "30-task source format: E protocol messages with teacher reasoning_content; final reply is the label Ledger",
        "citation_policy": f"Repository sources inside returned read ranges; end overshoot <= {MAX_END_OVERSHOOT} narrowed, "
                           "unread gaps split, never-read sources dropped only when the entry keeps other sources",
        "tasks": rows,
        "totals": {"citation_corrections": sum(row["citation_corrections"] for row in rows),
                   "removed_unknown_tool_calls": sum(row["removed_unknown_tool_calls"] for row in rows),
                   "calibrated_reasoning_tasks": sum(row["reasoning_source"] == "calibrated" for row in rows)},
    }
    manifest["file_sha256"] = {str(path.relative_to(output)): sha(path)
                               for path in sorted(output.rglob("*")) if path.is_file()}
    write_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = build(args.selection.resolve(), args.protocol.resolve(), args.output.resolve())
    print(json.dumps(manifest["totals"], ensure_ascii=False))


if __name__ == "__main__":
    main()
