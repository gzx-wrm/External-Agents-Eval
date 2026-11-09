"""Build reviewed E4 recovery views and normal replay, without starting training."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from jinja2 import Environment
from jsonschema import Draft202012Validator
from tokenizers import Tokenizer

from .build_e_teacher_trajectory import dump, sha, write_json
from .build_e2_materials import validate_conversation
from .revise_e2_materials import check_read_coverage
from .run_f_inference import strict_json, validate_output

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_NORMAL = HERE / "data/e2_no_thinking_success13_teacher8_human5_v2"
DEFAULT_SAMPLING = HERE / "runs/e4-deepseek-recovery6-20261005-r1"
DEFAULT_SEEDS = HERE / "data/e_recovery_pilot_v1"
DEFAULT_OUTPUT = HERE / "data/e4_no_thinking_recovery6_normal13_v1"
FIELDS = ("obligations", "preservation_obligations", "open_questions")
BAD_TURNS = {2: {7: "Read offset beyond known end of file"},
             3: {2: "Invalid unescaped regex", 3: "Escaped regex used with literal=true"},
             5: {5: "Search of nonexistent directory"}}
SHORT_TURNS = {1: 3, 2: 3, 3: 1, 4: 3, 5: 3, 6: 3}


def environment():
    env = Environment()
    env.filters["tojson"] = lambda value, **kwargs: json.dumps(value, ensure_ascii=False, **kwargs)
    return env


def masked_template(e2_template: str) -> str:
    """Conditional generation spans keep the native text unchanged."""
    start, end = "{%- generation %}", "{%- endgeneration %}"
    if e2_template.count(start) != 1 or e2_template.count(end) != 1:
        raise ValueError("Expected the pinned E2 assistant-body generation block")
    head, rest = e2_template.split(start)
    body, tail = rest.split(end)
    condition = "trainable_message_mask is not defined or trainable_message_mask[loop.index0]"
    return (head + "{%- if " + condition + " %}" + start + body + end
            + "{%- else %}" + body + "{%- endif %}" + tail)


def render_spans(template, sample):
    start, end = "\x00E4_GENERATION_START\x00", "\x00E4_GENERATION_END\x00"
    marked = template.replace("{%- generation %}", "{{- span_start }}")
    marked = marked.replace("{%- endgeneration %}", "{{- span_end }}")
    text = environment().from_string(marked).render(
        **sample, add_generation_prompt=False, enable_thinking=False,
        span_start=start, span_end=end,
        raise_exception=lambda message: (_ for _ in ()).throw(ValueError(message)))
    pieces, spans, position = [], [], 0
    while start in text:
        before, rest = text.split(start, 1)
        body, text = rest.split(end, 1)
        pieces.extend((before, body))
        position += len(before)
        spans.append((position, position + len(body)))
        position += len(body)
    pieces.append(text)
    return "".join(pieces), spans


def encode_masked_sample(sample, native, e2_template, e4_template, tokenizer):
    messages, mask = sample["messages"], sample["trainable_message_mask"]
    if len(mask) != len(messages) or any(type(v) is not bool for v in mask):
        raise ValueError("One boolean loss flag is required per message")
    if any(flag and m["role"] != "assistant" for flag, m in zip(mask, messages)):
        raise ValueError("Only assistant messages can receive loss")
    original = environment().from_string(native).render(
        **sample, add_generation_prompt=False, enable_thinking=False)
    all_text, all_spans = render_spans(e2_template, sample)
    selected_text, selected_spans = render_spans(e4_template, sample)
    positions = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    if all_text != original or selected_text != original or len(all_spans) != len(positions):
        raise ValueError("Generation markers changed the native serialization")
    expected_spans = [span for i, span in zip(positions, all_spans) if mask[i]]
    if selected_spans != expected_spans:
        raise ValueError("Conditional generation spans do not match the message mask")
    encoded = tokenizer.encode(original, add_special_tokens=False)
    ids = np.asarray(encoded.ids, dtype=np.int32)
    labels = np.full(len(ids), -100, dtype=np.int32)
    counts, span_index = [0] * len(all_spans), 0
    for token, (left, right) in enumerate(encoded.offsets):
        while span_index < len(all_spans) and left >= all_spans[span_index][1]:
            span_index += 1
        if span_index == len(all_spans):
            break
        begin, finish = all_spans[span_index]
        if right > begin and left < finish:
            if left < begin or right > finish:
                raise ValueError("A token crosses an assistant-body loss boundary")
            counts[span_index] += 1
            if mask[positions[span_index]]:
                labels[token] = ids[token]
    rows = [{"message_index": i, "body_tokens": count, "supervised": mask[i]}
            for i, count in zip(positions, counts)]
    supervised = int(np.count_nonzero(labels != -100))
    if supervised == 0:
        raise ValueError("Training row has no supervised tokens")
    final_tokens = counts[-1] if sample["view"] != "short" else 0
    if sample["view"] != "short":
        prefix = environment().from_string(native).render(
            messages=messages[:-1], tools=sample["tools"],
            add_generation_prompt=True, enable_thinking=False)
        if original != prefix + messages[-1]["content"].strip() + "<|im_end|>\n":
            raise ValueError("Non-thinking Ledger boundary differs from inference")
    stats = {"total_tokens": len(ids), "supervised_tokens": supervised,
             "masked_context_tokens": len(ids) - supervised,
             "final_ledger_turn_tokens": final_tokens,
             "action_supervised_tokens": supervised - final_tokens,
             "assistant_message_spans": rows,
             "native_template_text_equivalence_verified": True,
             "rendered_text_sha256": hashlib.sha256(original.encode()).hexdigest()}
    return {"input_ids": ids, "labels": labels}, stats


def normalize_message(message):
    text = message.get("content")
    if not isinstance(text, str):
        text = "".join(b.get("text", "") for b in (text or []))
    result = {"role": message["role"], "content": text}
    if message.get("tool_call_id"):
        result["tool_call_id"] = message["tool_call_id"]
    if message.get("tool_calls"):
        result["tool_calls"] = [{"id": c["id"], "name": c["function"]["name"],
            "arguments": json.loads(c["function"]["arguments"])
            if isinstance(c["function"]["arguments"], str) else c["function"]["arguments"]}
            for c in message["tool_calls"]]
    return result


def convert_continuation(seed, session, number):
    messages = copy.deepcopy(seed["messages"])
    origins = ["student_context"] * len(messages)
    flags = [False] * len(messages)
    names = {c["id"]: c["function"]["name"] for m in messages for c in m.get("tool_calls", [])}
    for m in messages:
        if m["role"] == "assistant":
            m["reasoning_content"] = ""
        elif m["role"] == "tool":
            m["name"] = names[m["tool_call_id"]]
    turn, thinking_chars, masked = 0, 0, []
    for m in session[len(seed["messages"]) - 1:]:
        if m["role"] == "assistant":
            turn += 1
            thinking_chars += sum(len(b.get("thinking", "")) for b in m["content"])
            out = {"role": "assistant", "content": "".join(b.get("text", "") for b in m["content"]),
                   "reasoning_content": ""}
            calls = [{"id": b["id"], "type": "function", "function": {
                "name": b["name"], "arguments": b["arguments"]}}
                for b in m["content"] if b["type"] == "toolCall"]
            if calls:
                out["tool_calls"] = calls
                names.update({c["id"]: c["function"]["name"] for c in calls})
            reason = BAD_TURNS.get(number, {}).get(turn)
            flags.append(reason is None)
            if reason:
                masked.append({"teacher_turn": turn, "message_index": len(messages),
                               "reason": reason, "calls": calls})
            origins.append("teacher" if calls else "reviewed_teacher_ledger")
        elif m["role"] == "toolResult":
            out = {"role": "tool", "name": m["toolName"], "tool_call_id": m["toolCallId"],
                   "content": "".join(b.get("text", "") for b in m["content"])}
            flags.append(False)
            origins.append("tool_observation")
        else:
            raise ValueError("Unexpected new message after the teacher takeover")
        messages.append(out)
    if [normalize_message(m) for m in messages[:len(seed["messages"])]] != [normalize_message(m) for m in seed["messages"]]:
        raise ValueError("Conversion changed the student prefix")
    return messages, flags, origins, {"removed_thinking_characters": thinking_chars,
                                     "masked_teacher_turns": masked}


def revise_ledger(number, target, reference):
    target = copy.deepcopy(target)
    edits = []

    def edit(path, value, reason):
        parent = target
        for key in path[:-1]:
            parent = parent[key]
        before = copy.deepcopy(parent[path[-1]])
        parent[path[-1]] = copy.deepcopy(value)
        edits.append({"path": path, "before": before, "after": copy.deepcopy(value), "reason": reason})

    def append_obligation(entry, reason):
        edit(["obligations"], target["obligations"] + [entry], reason)

    if number in (1, 4):
        for index in (0, 1):
            edit(["obligations", index, "statement"], reference["obligations"][index]["statement"],
                 "Align dependency variants and absent-trigger behavior with the reviewed task standard")
        recursion = 3 if number == 1 else 2
        statement = reference["obligations"][3]["statement"]
        if number == 4:
            statement = (reference["obligations"][2]["statement"] + " " + statement + " "
                         + reference["obligations"][4]["statement"])
        edit(["obligations", recursion, "statement"], statement,
             "Cover recursive conversion and finite-data validation, not just parse-time resolution")
        edit(["obligations", 19 if number == 1 else 18, "statement"],
             reference["obligations"][20]["statement"], "Retain recursive constraints within anyOf at runtime")
        append_obligation(reference["obligations"][21], "Promote exported schema-type support from uncertainty to an obligation")
        keep = [1] if number == 1 else [1, 3]
        edit(["open_questions"], [target["open_questions"][i] for i in keep],
             "Remove resolved dependency/type-support questions and implementation-permission questions")
        edit(["summary"], target["summary"] + " Recursive conversion and validation of finite nested values must terminate and enforce constraints at every nesting level; exported schema types must admit the new keywords.",
             "Align summary with the latest human-reviewed recursive contract")
        if number == 1:
            for path, end in ((["obligations", 0, "sources", 1, "line_end"], 274),
                              (["preservation_obligations", 0, "sources", 1, "line_end"], 274),
                              (["preservation_obligations", 3, "sources", 0, "line_end"], 65)):
                edit(path, end, "Trim the citation to the actual last baseline line")
        else:
            edit(["preservation_obligations", 0, "sources", 0, "line_end"], 52,
                 "Trim the citation to the actual last baseline line")
            edit(["preservation_obligations", 0, "statement"],
                 "Except for behavior explicitly changed by this task, " + target["preservation_obligations"][0]["statement"],
                 "Allow the task-authorized enum/conditional changes while preserving legacy behavior")
    elif number == 2:
        edit(["obligations", 14, "statement"],
             "Input JSON Schema generation must expose each field's aliases as additional object properties typed with the same per-field schema as the primary key.",
             "Do not invent a requirement that the primary key remains mandatory in alias input schemas")
    elif number == 3:
        append_obligation(reference["obligations"][17], "Cover all typed-target assignment forms consistently with the reviewed annotation")
        append_obligation(reference["obligations"][18], "Preserve the manually reviewed public VM failure-result contract")
        edit(["open_questions"], target["open_questions"][1:],
             "Assignment-form coverage was settled by the reviewed task standard")
        edit(["preservation_obligations", 3], reference["preservation_obligations"][3],
             "Include nil-options behavior and the default disabled state alongside Debug compatibility")
        edit(["summary"], target["summary"] + " Typed-binding failures must return a non-nil error with a nil result through the public VM API, including nested failures; successful result behavior remains unchanged.",
             "Align the summary with the latest manual error-result annotation")
    elif number == 5:
        edit(["open_questions"], target["open_questions"][:3],
             "Synchronous code can raise CancelledError; do not train the opposite premise")
        edit(["open_questions", 2, "question"],
             "How should failed underlying executions contribute to the coalesced and total statistics?",
             "Keep the statistics uncertainty without questioning propagation of an execution's error to joiners")
        edit(["obligations", 5, "statement"],
             reference["obligations"][10]["statement"] + " " + reference["obligations"][11]["statement"],
             "Clarify batch-local duplicate coalescing independent of internal scheduling")
    elif number == 6:
        edit(["obligations", 0, "statement"], reference["obligations"][0]["statement"],
             "Avoid duplicating the API list already covered by the following obligations")
        edit(["obligations", 11, "statement"],
             "Style.Truncate must respect a style marked via Style.PreserveResets() when applying preserve-resets behavior.",
             "The task specifies an OR rule for Output.Truncate, not explicitly for Style.Truncate")
        edit(["obligations", 14, "statement"], reference["obligations"][17]["statement"],
             "Do not restrict the stated preserve-resets rule to truncation only")
        for i in (0, 2):
            edit(["preservation_obligations", i, "statement"],
                 "When preserve-resets is not enabled, " + target["preservation_obligations"][i]["statement"],
                 "Permit the new flag to change rendering while retaining the old default behavior")
        questions = target["open_questions"][1:] + [reference["open_questions"][5]]
        edit(["open_questions"], questions,
             "Drop the package-versus-method question already settled by the wrapper obligation; retain the Style option-combination uncertainty")
    return target, edits


def replay_anko_read(sampling, destination, seed_row):
    """Execute Pi's actual read tool; the action text is a documented expert reconstruction."""
    task = seed_row["task_id"]
    path = "vm/vm_test.go"
    baseline = sampling / "baselines" / task
    args = {"path": "/work/base/" + path, "offset": 82, "limit": 8}
    ident = "call_e4_review_anko_01"
    code = ("import {createReadTool} from '/opt/pi-runtime/node_modules/@earendil-works/pi-coding-agent/dist/core/tools/index.js';"
            + "console.log(JSON.stringify(await createReadTool('/work/base').execute("
            + json.dumps(ident) + "," + json.dumps(args) + ")));" )
    executed = subprocess.run(["docker", "run", "--rm", "--entrypoint", "/opt/pi-runtime/bin/node",
        "--volume", f"{baseline}:/work/base:ro", "--workdir", "/work/base",
        "external-agents-eval/pi-e-smoke:v0.84.2-arm64", "--input-type=module", "-e", code],
        capture_output=True, text=True, check=True)
    result = json.loads(executed.stdout)
    blob = subprocess.check_output(["git", "-C", str(HERE / ".cache/baselines" / task),
                                    "show", f"{seed_row['base_commit']}:{path}"])
    if blob != (baseline / path).read_bytes():
        raise ValueError("Anko supplemental baseline differs from the pinned Git object")
    text = "".join(b["text"] for b in result["content"])
    if not text.startswith("\n".join(blob.decode().splitlines()[81:89])):
        raise ValueError("Actual read did not return the requested baseline lines")
    provenance = {"executed_at": datetime.now(timezone.utc).isoformat(), "task_id": task,
        "base_commit": seed_row["base_commit"], "path": path, "arguments": args,
        "tool_call_id": ident, "result": result,
        "full_file_sha256": hashlib.sha256(blob).hexdigest(),
        "returned_content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "assistant_message_origin": "Human-authored expert reconstruction, not an original teacher rollout",
        "tool_observation_origin": "Actual Pi 0.84.2 read tool execution on the pinned read-only baseline"}
    write_json(destination, provenance)
    messages = [{"role": "assistant", "content": "Before finalizing, I'll verify the public VM API's failure-result contract against its baseline tests.",
                 "reasoning_content": "", "tool_calls": [{"id": ident, "type": "function", "function": {"name": "read", "arguments": args}}]},
                {"role": "tool", "name": "read", "tool_call_id": ident, "content": text}]
    return messages, provenance


def visible_evidence(messages, baseline):
    """Bind citations to the exact read/grep text present in this conversation."""
    calls, seen, coverage = {}, set(), defaultdict(set)
    for m in messages:
        if m["role"] == "assistant":
            for c in m.get("tool_calls", []):
                if c["id"] in calls:
                    raise ValueError("Duplicate tool-call ID")
                calls[c["id"]] = c["function"]
        elif m["role"] == "tool":
            if m["tool_call_id"] in seen:
                raise ValueError("Duplicate tool result")
            seen.add(m["tool_call_id"])
            c = calls[m["tool_call_id"]]
            a, text = c["arguments"], m["content"]
            rel = a.get("path", "/work/base").removeprefix("/work/base/")
            rel = "" if rel == "/work/base" else rel
            p = baseline / rel
            if c["name"] == "read" and p.is_file():
                lines = p.read_text(errors="replace").splitlines()
                start, count = a.get("offset", 1) - 1, 0
                for actual, expected in zip(text.splitlines(), lines[start:]):
                    if actual != expected:
                        break
                    count += 1
                coverage[rel].update(range(start + 1, start + count + 1))
            elif c["name"] == "grep":
                for line in text.splitlines():
                    match = re.match(r"^(.*?)([:\-])(\d+)([:\-])(.*)$", line)
                    if not match:
                        continue
                    name, _, number, _, body = match.groups()
                    file = p if p.is_file() else p / name
                    if file.is_file():
                        lines = file.read_text(errors="replace").splitlines()
                        number = int(number)
                        if 1 <= number <= len(lines) and body == lines[number - 1]:
                            coverage[file.relative_to(baseline).as_posix()].add(number)
    if seen != set(calls):
        raise ValueError("Tool-call/result pairing is incomplete")
    evidence = []
    for path, numbers in sorted(coverage.items()):
        if not numbers:
            continue
        lines = (baseline / path).read_text(errors="replace").splitlines()
        runs = []
        for n in sorted(numbers):
            if not runs or n != runs[-1][1] + 1:
                runs.append([n, n])
            else:
                runs[-1][1] = n
        for a, b in runs:
            evidence.append({"path": path, "line_start": a, "line_end": b,
                             "text": "\n".join(lines[a - 1:b])})
    return evidence


def short_view(full, turns):
    actions = [i for i, (m, origin) in enumerate(zip(full["messages"], full["message_origins"]))
               if origin == "teacher" and m.get("tool_calls")]
    last = actions[turns - 1]
    end = next((i for i in range(last + 1, len(full["messages"]))
                if full["messages"][i]["role"] == "assistant"), len(full["messages"]))
    sample = copy.deepcopy(full)
    sample.update(sample_id=full["parent_seed_id"] + "--short", view="short", trajectory_complete=False)
    for key in ("messages", "trainable_message_mask", "message_origins"):
        sample[key] = sample[key][:end]
    # Full validator requires a final Ledger; appending a dummy stop message only validates pairing here.
    validate_conversation({"messages": sample["messages"] + [{"role": "assistant", "content": "{}", "reasoning_content": ""}],
                           "tools": sample["tools"]})
    if sample["messages"][-1]["role"] != "tool" or not all(sample["trainable_message_mask"][i] for i in actions[:turns]):
        raise ValueError("Short recovery view includes an unapproved action or dangling calls")
    return sample


def frozen_trainer(e2_text):
    """Use the established trainer with E4 IDs, message masks and one intermediate checkpoint."""
    text = e2_text.replace('"full_trajectories": True', '"full_trajectories": manifest.get("all_rows_complete", True)')
    text = text.replace('enable_thinking=manifest.get("enable_thinking", True), add_generation_prompt=False,',
                        'enable_thinking=manifest.get("enable_thinking", True), add_generation_prompt=False,\n'
                        '                trainable_message_mask=sample["trainable_message_mask"],')
    text = text.replace("f\"{task['task_id']}.npz\"", "f\"{task['sample_id']}.npz\"")
    text = text.replace('task_id=task["task_id"]', 'sample_id=task["sample_id"], task_id=task["task_id"]')
    text = text.replace('"task_id": task["task_id"],', '"sample_id": task["sample_id"], "task_id": task["task_id"],')
    text = text.replace('task_order=[tasks[index]["task_id"] for index in order]',
                        'sample_order=[tasks[index]["sample_id"] for index in order]')
    anchor = '                del output, loss, input_ids, labels, attention_mask\n'
    mid = '''                if global_step in config.get("checkpoint_at_steps", []):
                    checkpoint = args.output / f"checkpoint-step-{global_step}"
                    checkpoint.mkdir()
                    mid_adapter = checkpoint / "adapter"
                    event("saving_intermediate_checkpoint", epoch=epoch, step=global_step)
                    model.save_pretrained(str(mid_adapter), safe_serialization=True)
                    tokenizer.save_pretrained(str(mid_adapter))
                    torch.save({"epoch": epoch, "global_step": global_step,
                                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                                "config": config,
                                "dataset_manifest_sha256": report["dataset_manifest_sha256"]},
                               checkpoint / "training-state.pt")
                    report["checkpoints"].append({"kind": "intermediate", "epoch": epoch,
                                                  "step": global_step, "path": checkpoint.name})
                    event("intermediate_checkpoint_saved", step=global_step)
'''
    if text.count(anchor) != 1 or 'trainable_message_mask=sample' not in text:
        raise ValueError("Pinned trainer layout differs")
    return text.replace(anchor, anchor + mid)


def build(normal, sampling, seeds, output):
    source = json.loads((normal / "manifest.json").read_text())
    for path, expected in source["file_sha256"].items():
        if sha(normal / path) != expected:
            raise ValueError(f"Frozen E2 source changed: {path}")
    seed_manifest = json.loads((seeds / "manifest.json").read_text())
    batch = json.loads((sampling / "batch.json").read_text())
    if batch["status"] != "SAMPLING_FINISHED_REVIEW_PENDING" or batch["errors"]:
        raise ValueError("Teacher sampling is not complete")
    output.mkdir(parents=True, exist_ok=False)
    for name in ("labels", "records", "tokenized", "examples", "provenance", "provenance/supplemental_reads"):
        (output / name).mkdir(exist_ok=True)
    shutil.copytree(normal / "protocol", output / "protocol")
    shutil.copy2(normal / "inference_config.json", output / "inference_config.json")
    for name, path in (("e2_manifest.json", normal / "manifest.json"),
                       ("e2_train.jsonl", normal / "train.jsonl"),
                       ("seed_manifest.json", seeds / "manifest.json"),
                       ("sampling_batch.json", sampling / "batch.json")):
        shutil.copy2(path, output / "provenance" / name)
    native = (normal / "protocol/native_chat_template.jinja").read_text()
    e2_template = (normal / "protocol/training_chat_template.jinja").read_text()
    e4_template = masked_template(e2_template)
    (output / "protocol/training_chat_template.jinja").write_text(e4_template)
    tokenizer = Tokenizer.from_file(str(normal / "protocol/tokenizer.json"))
    schema = Draft202012Validator(json.loads((normal / "protocol/spec_output_schema.json").read_text()))
    samples, rows, reviews = [], [], []

    def add(sample, metadata, record):
        arrays, stats = encode_masked_sample(sample, native, e2_template, e4_template, tokenizer)
        ident = sample["sample_id"]
        np.savez_compressed(output / "tokenized" / f"{ident}.npz", **arrays)
        row = {"row_index": len(samples), "sample_id": ident, "task_id": sample["task_id"],
               "view": sample["view"], "parent_seed_id": sample.get("parent_seed_id"),
               "trajectory_complete": sample["trajectory_complete"], **metadata, **stats}
        if sample["view"] != "short":
            target = strict_json(sample["messages"][-1]["content"])
            schema.validate(target)
            write_json(output / "labels" / f"{ident}.json", target)
            row["training_ledger_sha256"] = hashlib.sha256(dump(target).encode()).hexdigest()
        record.update(sample_id=ident, task_id=sample["task_id"], lengths=stats)
        write_json(output / "records" / f"{ident}.json", record)
        samples.append(sample)
        rows.append(row)
        print(f"{ident}: {stats['total_tokens']:,} input / {stats['supervised_tokens']:,} supervised", flush=True)
        return arrays

    for parent_row, line in zip(source["tasks"], (normal / "train.jsonl").read_text().splitlines(), strict=True):
        sample = json.loads(line)
        task = parent_row["task_id"]
        record = json.loads((normal / "records" / f"{task}.json").read_text())
        evidence_path = ROOT / record["reference_evidence_record"]["path"]
        observed = json.loads(evidence_path.read_text())
        target = strict_json(sample["messages"][-1]["content"])
        _, checks = validate_output(dump(target), observed, schema)
        if not checks["sources_valid"]:
            raise ValueError(f"Normal replay source invalid: {task}: {checks}")
        checked = check_read_coverage(sample, target, observed, record["e2_review"]["added_reads"])
        validate_conversation(sample)
        sample.update(sample_id="normal-" + task, task_id=task, view="normal", trajectory_complete=True,
                      trainable_message_mask=[m["role"] == "assistant" for m in sample["messages"]])
        record.update(source_materials=str(normal), original_sample_unchanged=True,
                      repository_citations_covered=checked)
        arrays = add(sample, {"label_source": parent_row["label_source"], "source_row_index": parent_row["row_index"]}, record)
        with np.load(normal / "tokenized" / f"{task}.npz") as old:
            if not all(np.array_equal(arrays[k], old[k]) for k in arrays):
                raise ValueError("Normal replay changed token IDs or loss masks")

    for number, seed_row in enumerate(seed_manifest["samples"], 1):
        directory = sampling / "samples" / seed_row["sample_id"]
        seed = json.loads((directory / "seed.json").read_text())
        if sha(directory / "seed.json") != seed_row["seed_sha256"] or sha(seeds / seed_row["seed_file"]) != seed_row["seed_sha256"]:
            raise ValueError("Seed hash differs")
        result = json.loads((directory / "result.json").read_text())
        if result["exit_code"] != 0 or not result["baseline_unchanged"] or not result["raw_json_schema_valid"]:
            raise ValueError("Teacher continuation did not finish normally")
        request_count = 0
        for line in (directory / "provider-requests.jsonl").open():
            request = json.loads(line)
            if [normalize_message(m) for m in request["messages"][:len(seed["messages"])]] != [normalize_message(m) for m in seed["messages"]]:
                raise ValueError("Provider payload changed the prefix")
            if [{k: t["function"][k] for k in ("name", "description", "parameters")} for t in request["tools"]] != [{k: t["function"][k] for k in ("name", "description", "parameters")} for t in seed["tools"]]:
                raise ValueError("Provider payload changed the tools")
            request_count += 1
        archive = output / "provenance" / seed_row["sample_id"]
        archive.mkdir()
        for name in ("seed.json", "native-session.json", "output.txt", "result.json", "config.json", "prefix-check.json"):
            shutil.copy2(directory / name, archive / name)
        session = json.loads((directory / "native-session.json").read_text())
        if session[-1]["stopReason"] != "stop":
            raise ValueError("Teacher Ledger was truncated or interrupted")
        messages, flags, origins, cleanup = convert_continuation(seed, session, number)
        raw = strict_json((directory / "output.txt").read_text())
        reference_path = normal / "labels" / (seed_row["task_id"] + ".json")
        target, edits = revise_ledger(number, raw, json.loads(reference_path.read_text()))
        addition = None
        if number == 3:
            extra, addition = replay_anko_read(sampling, output / "provenance/supplemental_reads/anko-vm-api.json", seed_row)
            messages[-1:-1] = extra
            flags[-1:-1] = [True, False]
            origins[-1:-1] = ["expert_reconstruction", "tool_observation"]
        messages[-1]["content"] = dump(target)
        full = {"sample_id": seed["sample_id"] + "--full", "task_id": seed["task_id"],
                "parent_seed_id": seed["sample_id"], "view": "full", "trajectory_complete": True,
                "teacher_start_message_index": len(seed["messages"]),
                "messages": messages, "tools": seed["tools"], "trainable_message_mask": flags,
                "message_origins": origins}
        validate_conversation(full)
        task = json.loads(seed["messages"][1]["content"].split("SPEC INPUT\n", 1)[1])["task"]
        evidence = visible_evidence(messages, sampling / "baselines" / seed["task_id"])
        _, checks = validate_output(dump(target), {"task": task, "evidence": evidence}, schema)
        if not checks["sources_valid"]:
            raise ValueError(f"Reviewed recovery target source invalid: {seed['sample_id']}: {checks}")
        short = short_view(full, SHORT_TURNS[number])
        review = {"parent_seed_id": seed["sample_id"], "task_id": seed["task_id"], "base_commit": seed_row["base_commit"],
                  "accepted": True, "short_action_turns": SHORT_TURNS[number], "ledger_edits": edits,
                  **cleanup, "supplemental_read": addition, "validation": checks,
                  "provider_requests_prefix_and_tools_verified": request_count,
                  "source_seed_sha256": seed_row["seed_sha256"], "reference_label_sha256": sha(reference_path),
                  "sampling_status_note": "Original batch misread schema_valid; raw_json_schema_valid is true. Original metadata preserved."}
        reviews.append(review)
        source_meta = {"label_source": "teacher_continuation_expert_reviewed", "base_commit": seed_row["base_commit"],
                       "student_prefix_messages": len(seed["messages"]), "recovery_category": seed_row["category"]}
        add(short, source_meta, {"recovery_review": review, "ledger_in_training_row": False})
        add(full, source_meta, {"recovery_review": review, "ledger_in_training_row": True})

    (output / "train.jsonl").write_text("".join(dump(sample) + "\n" for sample in samples))
    write_json(output / "review_changes.json", reviews)
    for sample in samples[13:]:
        write_json(output / "examples" / f"{sample['sample_id']}.json", sample)
    (output / "lora_train_trajectories.py").write_text(frozen_trainer((normal / "lora_train_trajectories.py").read_text()))
    shutil.copy2(Path(__file__), output / "build_e4_materials.py")
    shutil.copy2(HERE / "verify_e4_tokenization.py", output / "verify_e4_tokenization.py")
    totals = {"tokens": sum(row["total_tokens"] for row in rows),
              "supervised_tokens": sum(row["supervised_tokens"] for row in rows),
              "maximum_tokens": max(row["total_tokens"] for row in rows),
              "removed_recovery_thinking_characters": sum(review["removed_thinking_characters"] for review in reviews),
              "masked_teacher_turns": sum(len(review["masked_teacher_turns"]) for review in reviews)}
    by_view = {view: {"rows": sum(row["view"] == view for row in rows),
                      "tokens": sum(row["total_tokens"] for row in rows if row["view"] == view),
                      "supervised_tokens": sum(row["supervised_tokens"] for row in rows if row["view"] == view),
                      "action_supervised_tokens": sum(row["action_supervised_tokens"] for row in rows if row["view"] == view),
                      "ledger_supervised_tokens": sum(row["final_ledger_turn_tokens"] for row in rows if row["view"] == view)}
               for view in ("normal", "short", "full")}
    config = json.loads((normal / "training_config.json").read_text())
    config.update(epochs=1, learning_rate=2e-5, max_seq_length=((totals["maximum_tokens"] + 4095) // 4096) * 4096,
                  checkpoint_at_steps=[13], max_runtime_seconds=21600)
    write_json(output / "training_config.json", config)
    validation = {"trainable": True, "training_started": False,
        "normal_rows_token_ids_and_loss_unchanged": 13,
        "all_rows_native_serialization_equivalent": 25, "all_masks_match_message_flags": True,
        "all_student_prefixes_context_only": True, "all_error_turns_context_only": True,
        "all_tool_observations_context_only": True, "thinking_removed": True,
        "no_truncation_or_packing": True, "all_full_rows_valid_unnumbered_ledgers": 19,
        "recovery_ledgers_sources_verified": 6, "short_rows_have_paired_tools": 6,
        "expert_supplemental_reads": 1, "source_sampling_files_unchanged": True,
        "semantic_review_note": "Targeted expert review against task, visible baseline and existing reviewed annotations; not a complete semantic gold standard or a GPU memory test."}
    write_json(output / "validation.json", validation)
    manifest = {"schema_version": "e4_recovery_normal_mixed_v1", "experiment": "E4",
        "created_at": datetime.now(timezone.utc).isoformat(), "model": source["model"], "model_revision": source["model_revision"],
        "sample_count": len(samples), "independent_training_tasks": len(source["tasks"]),
        "independent_recovery_tasks": 5, "recovery_seeds": 6, "normal_source_counts": source["label_source_counts"],
        "enable_thinking": False, "thinking_in_training": False, "all_rows_complete": False,
        "trainable": True, "training_started": False, "tasks": rows, "totals": totals, "by_view": by_view,
        "normal_materials": str(normal), "normal_manifest_sha256": sha(normal / "manifest.json"),
        "sampling_run": str(sampling), "sampling_batch_sha256": sha(sampling / "batch.json"),
        "seed_manifest_sha256": sha(seeds / "manifest.json"), "builder_sha256": sha(Path(__file__)),
        "loss_policy": "Only selected assistant bodies/calls/turn ends; all student-prefix, error-turn, system/user/tool/thinking/header tokens -100",
        "length_policy": "Full prefixes and full continuations preserved; six explicit short recovery views, no arbitrary truncation",
        "split_policy": "Development training only. All same-task seeds, views and normal replay are grouped together; Cliffy remains outside this dataset."}
    table = "\n".join(f"| {row['sample_id']} | {row['total_tokens']:,} | {row['supervised_tokens']:,} |" for row in rows if row["view"] != "normal")
    (output / "README.md").write_text(f'''# E4 训练物料：恢复续写与正常轨迹混合 v1

已完成数据制作和本地校验，尚未启动训练。共 25 行：13 条 E2 v2 正常轨迹（教师来源标签 8、人工来源标签 5）、6 条短恢复视图、6 条完整恢复视图。仍为原 13 道训练题；异常来自 6 个起点、5 道题。短/完整视图不是独立样本，Cliffy 不进入训练。

## 数据与监督

`train.jsonl` 每行包含 sample_id、task_id、view、messages、tools 和逐消息 `trainable_message_mask`。正常轨迹的输入 IDs、标签和对话均与冻结 E2 v2 相同；恢复轨迹完整保留学生前缀和真实观察，只监督有效教师动作与修订 Ledger。系统、用户、工具返回、空 thinking 前缀、学生所有消息、4 个问题教师回合的 loss 均为 -100。发生混合工具调用的错误回合整体屏蔽，成功返回仍留作后续上下文。

短恢复视图选前 3 个教师动作回合，Anko 选第一回合（包含 3 次有效 read），避开随后两个错误回合。短视图带完整对应工具返回，末尾为 tool，是明确的局部训练窗口，不伪造最终 Ledger。完整视图包含全部续写和审查后的最终无编号 Ledger。

采用 E2 非 thinking 原生协议；`protocol/training_chat_template.jinja` 按消息标记生成 loss 区间，序列文本及 IDs 与原生模板完全相同。`tokenized/` 按 sample_id 保存 input_ids/labels；不能直接使用 E2 的“所有 assistant 监督”入口或按 task_id 覆盖同题文件。

## 审查与来源

六条 Ledger 已作针对性修订：Arktype 修正 4 处末行越界并明确递归运行时/公开类型义务；Adaptix 删除无依据的 primary-required 条件；Anko 对齐复合赋值和公开 VM 失败返回规范；LangChain 清除同步 CancelledError 的错误前提；Termenv 限定旧渲染保留范围并避免额外推定 Style 的选项 OR 规则。逐项 before/after 见 `review_changes.json`。

Anko 在最终 Ledger 前补充一个专家重建的 read 回合，实际执行 Pi 0.84.2 工具读取冻结仓库 vm/vm_test.go:82–89。行动说明由人工编写，工具观察来自实际执行，不是原教师采样。其 Git commit、文件/返回哈希、执行时间和原始结果见 `provenance/supplemental_reads/anko-vm-api.json`。审查引用只使用原任务、已保留观察及这次补读，没有后续补丁或隐藏评测信息。

`provenance/` 保留 E2 manifest/原 train、seed manifest、原批次信息和六条原始 seed/session/Ledger。采样元数据曾误读 schema_valid 字段，本物料依据实际 raw_json_schema_valid 和重新证据校验；原始文件不改写。所有生成文件哈希见 manifest.json。

## 规模与训练入口

每轮输入 {totals['tokens']:,} tokens，监督 {totals['supervised_tokens']:,} tokens；最长 {totals['maximum_tokens']:,} tokens。配置 max_seq_length={config['max_seq_length']:,}，不截断、不 packing。98,304 的 E2 上限不足以容纳 Anko 完整恢复视图。序列上限已配置，但 GPU 显存是否充足须在训练时实测。

正常/恢复更新行数为 13:12；短/完整恢复为 6:6。具体动作/最终 Ledger 监督 token 占比见 manifest.json 的 by_view，行数比例不等于 token 比例。

默认从固定 revision 的原始 Qwen3.5-9B base 新建 BF16 LoRA，rank16/alpha32、lr=2e-5、1 epoch、batch1。训练第 13 步保存中间 adapter，第 25 步保存最终 epoch checkpoint。冻结入口 `lora_train_trajectories.py` 支持 sample_id 和逐消息 mask，在加载模型前核对本地 Transformers 序列和监督范围。超长序列的计算/显存尚未验证；本数据不是已完成微调的模型。

无需加载模型即可使用 `verify_e4_tokenization.py --materials <物料目录>` 核对原生 token IDs 和监督范围。正式训练入口参数为 `--materials <物料目录> --model-path <固定 revision 的本地模型快照> --config <物料目录>/training_config.json --output <新训练输出目录>`。快照目录末段必须为 manifest 中的 revision。构建脚本快照用于记录制作逻辑，重新构建需要项目中的 E2 与采样源文件。

推理继续关闭 thinking，使用 inference_config.json；不得在恢复数据中重新监督教师思考。论文比较需另补相同训练参数的“仅正常轨迹”对照。本批次是已有训练题上的离线恢复可行性实验，不能作为未见仓库泛化的独立证据。

| 恢复训练行 | 输入 tokens | 监督 tokens |
|---|---:|---:|
{table}
''')
    manifest["file_sha256"] = {p.relative_to(output).as_posix(): sha(p)
                              for p in sorted(output.rglob("*")) if p.is_file() and p.name != "manifest.json"}
    write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal-data", type=Path, default=DEFAULT_NORMAL)
    parser.add_argument("--sampling", type=Path, default=DEFAULT_SAMPLING)
    parser.add_argument("--seeds", type=Path, default=DEFAULT_SEEDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build(args.normal_data.resolve(), args.sampling.resolve(), args.seeds.resolve(), args.output.resolve())
    print(json.dumps({"sample_count": result["sample_count"], "totals": result["totals"], "by_view": result["by_view"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
