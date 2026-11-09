"""V6.7 generation-only experiment: ledger + check points on frozen V6.6 A evidence.

No Solver, review, probe or scoring runs. For every selected task the frozen V6.6 A
evidence packet and the exact V6.6 ledger user message are reused; only the tool-free
ledger request (V6.7 fidelity rules) and the check-point stage (V6.7 C) are new.

Arms differ only in the model:
- deepseek: DeepSeek V4.1 Flash, chat completions, thinking enabled, reasoning high,
  max_tokens 65536, seed 42 (the V6.6.4 A settings);
- gpt: GPT-6.1 Sol through the two host-side Responses gateways, reasoning xhigh,
  max_output_tokens 65536 (the setting of the approved GPT-6.1 Sol runs).

Each request gets transport retries only. Ledger and every check-point batch get at most
one corrective retry when the output violates the contract. prepare and check make no
model calls; run is explicit. Outputs go to a new root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT))

from jsonschema import Draft202012Validator  # noqa: E402

from external_agents_eval.workflows.solve_review_repair import v67_generation as v67  # noqa: E402
from external_agents_eval.workflows.solve_review_repair.ft_inference import FT_SCHEMA  # noqa: E402
from external_agents_eval.workflows.solve_review_repair.ft_protocol import (  # noqa: E402
    render_user,
    strict_json,
    validate_output,
)
from external_agents_eval.workflows.solve_review_repair.v66_checkpoints import (  # noqa: E402
    BatchReport,
    target_batches,
)
from external_agents_eval.workflows.solve_review_repair.v66_prompts import checkpoint_user  # noqa: E402
from external_agents_eval.workflows.solve_review_repair.v66_transport import open_with_retry  # noqa: E402

EXPERIMENT_ID = "v6.7-generation"
# v6.7.1: GPT stream stall timeout (transport only; generation logic stays v6.7.0).
# v6.7.2: httpx transport errors retryable, sticky working gateway, resume from stored responses.
# v6.7.3: in-stream API errors without an HTTP status are retryable too.
# v6.7.4: three gateways (user-added third, 2026-11-06) in round-robin; lower concurrency.
# v6.7.5: GPT check-point batches of 3 targets (user, 2026-11-06), so that one streamed call fits the
#         gateways; DeepSeek keeps 8. Prompts, limits and validation are unchanged.
RUNNER_VERSION = "v6.7.5"
FROZEN_SOURCES = (
    "scripts/deepswe_v67_gen/run.py",
    "external_agents_eval/workflows/solve_review_repair/v67_generation.py",
    "external_agents_eval/workflows/solve_review_repair/v66_checkpoints.py",
    "external_agents_eval/workflows/solve_review_repair/v66_prompts.py",
    "external_agents_eval/workflows/solve_review_repair/v66_transport.py",
    "external_agents_eval/workflows/solve_review_repair/ft_inference.py",
    "external_agents_eval/workflows/solve_review_repair/ft_protocol.py",
    "external_agents_eval/workflows/solve_review_repair/ft/prompt.txt",
)
ARMS = {
    "deepseek": {"model": "deepseek-flash", "base_url": "https://api.deepseek.com/v1",
                 "parameters": {"thinking": {"type": "enabled"}, "reasoning_effort": "high",
                                "max_tokens": 65536, "seed": 42}},
    "gpt": {"model": "gpt-6.1-sol", "parameters": {"reasoning": {"effort": "xhigh"},
                                                   "max_output_tokens": 65536, "store": False}},
}
ENDPOINTS_FILE = Path("/root/.config/external-agents-eval/v67-gpt-endpoints-20261106/endpoints.json")
BATCH_MAX_TARGETS = 8  # V6.6.4 and the DeepSeek arm
ARM_BATCH_MAX_TARGETS = {"deepseek": 8, "gpt": 3}
MAX_ATTEMPTS = 2
TIMEOUT_SEC = 1800
GPT_ROUNDS = 3  # each round tries every gateway once; waits between rounds
GPT_STALL_SEC = 600  # no stream data for this long: abort and fail over
GPT_WAITS_SEC = (30, 90)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def sha256(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode()).hexdigest()


def code_hashes():
    return {name: sha256((PROJECT / name).read_bytes()) for name in FROZEN_SOURCES}


# --- frozen inputs ----------------------------------------------------------------------------

def frozen_packet(a_dir):
    """The V6.6 evidence packet in the field order the FT request and stage C use."""
    packet = read(Path(a_dir) / "v6_review_repair/v6.2/evidence.json")
    task = packet["task"]
    packet["task"] = {"description": task["description"], "title": task["title"]}
    return packet


def frozen_ledger_user(a_dir):
    request = read(Path(a_dir) / "v6_review_repair/v6.2/ledger/request.json")
    return request["messages"][1]["content"]


def rebuilt_ledger_user(packet):
    """The FT request used the live managed_delivery entries (kind, field, quote order)."""
    live = {**packet, "managed_delivery": [{"kind": e["kind"], "field": e["field"], "quote": e["quote"]}
                                           for e in packet["managed_delivery"]]}
    return render_user(live, live["evidence"])


def frozen_c_first_batch(a_dir):
    """V6.6.4 stage C batch 1 input, when the A directory holds it."""
    path = Path(a_dir) / "v6_review_repair/v6.6/checkpoints/batch-01/attempt-1/messages.json"
    ledger = Path(a_dir) / "ledger.json"
    if not (path.exists() and ledger.exists()):
        return None, None
    return read(ledger), read(path)[1]["content"]


def prepare(root, sources, task_ids, note):
    if root.exists() and any(root.iterdir()):
        raise ValueError("Use a new, empty experiment directory")
    rows = []
    for spec in sources:
        split, batch, a_root = spec.split(":", 2)
        for a_dir in sorted((Path(a_root) / "a").iterdir()):
            tid = a_dir.name
            if task_ids and tid not in task_ids:
                continue
            if not (a_dir / "v6_review_repair/v6.2/evidence.json").exists():
                continue
            a_dir = a_dir.resolve()
            rows.append({"task_id": tid, "split": split, "batch": batch, "a_dir": str(a_dir),
                         "evidence_sha256": sha256((a_dir / "v6_review_repair/v6.2/evidence.json").read_bytes()),
                         "ledger_request_sha256": sha256(
                             (a_dir / "v6_review_repair/v6.2/ledger/request.json").read_bytes())})
    missing = sorted(set(task_ids or ()) - {row["task_id"] for row in rows})
    if missing:
        raise ValueError("No frozen A evidence for: " + ", ".join(missing))
    root.mkdir(parents=True, exist_ok=True)
    selection = {
        "experiment_id": EXPERIMENT_ID, "version": v67.VERSION, "runner_version": RUNNER_VERSION,
        "note": note, "created_at": now(),
        "task_count": len(rows), "tasks": rows, "code_sha256": code_hashes(),
        "arms": ARMS, "config": {"batch_max_targets": ARM_BATCH_MAX_TARGETS, "max_attempts": MAX_ATTEMPTS,
                                 "max_requirements_per_target": v67.MAX_REQUIREMENTS_PER_TARGET,
                                 "max_questions_per_target": v67.MAX_QUESTIONS_PER_TARGET,
                                 "solver_view_max_chars": v67.SOLVER_VIEW_MAX_CHARS,
                                 "timeout_sec": TIMEOUT_SEC},
        "stages": ["frozen V6.6 A evidence packet and ledger user message (no new exploration)",
                   "ledger: FT system prompt + V6.7 fidelity rules; one corrective retry",
                   "C: V6.7 requirement/question check points in target batches; one corrective retry",
                   "mechanical fidelity flags; V6.7 Solver view (requirements first, questions without answers)",
                   "no probe, Solver, review or scoring"],
    }
    save(root / "selection.json", selection)
    return selection


def validate(root):
    selection = read(root / "selection.json")
    if (selection["experiment_id"] != EXPERIMENT_ID or selection["version"] != v67.VERSION
            or selection.get("runner_version") != RUNNER_VERSION):
        raise ValueError("Wrong experiment or version")
    if code_hashes() != selection["code_sha256"]:
        changed = [k for k, v in code_hashes().items() if selection["code_sha256"].get(k) != v]
        raise ValueError("Frozen experiment code changed: " + ", ".join(changed))
    for row in selection["tasks"]:
        a_dir = Path(row["a_dir"])
        if (sha256((a_dir / "v6_review_repair/v6.2/evidence.json").read_bytes()) != row["evidence_sha256"]
                or sha256((a_dir / "v6_review_repair/v6.2/ledger/request.json").read_bytes())
                != row["ledger_request_sha256"]):
            raise ValueError(f"Frozen A input changed for {row['task_id']}")
    return selection


def check(root, selection):
    """Zero-API: the rebuilt ledger message equals the frozen one; prompts and validators load."""
    problems, rows = [], []
    for row in selection["tasks"]:
        packet = frozen_packet(row["a_dir"])
        user = rebuilt_ledger_user(packet)
        same = user == frozen_ledger_user(row["a_dir"])
        if not same:
            problems.append(f"{row['task_id']}: rebuilt ledger user message differs from the frozen request")
        old_ledger, old_c = frozen_c_first_batch(row["a_dir"])
        c_same = None
        if old_c is not None:
            batches = target_batches(old_ledger, BATCH_MAX_TARGETS)
            c_same = checkpoint_user(packet, old_ledger, (batches[0], 1, len(batches))) == old_c
            if not c_same:
                problems.append(f"{row['task_id']}: rebuilt stage C input differs from the frozen V6.6.4 batch 1")
        rows.append({"task_id": row["task_id"], "ledger_user_identical": same, "c_user_identical": c_same,
                     "evidence_spans": len(packet["evidence"]), "user_chars": len(user)})
    system = v67.ledger_system_prompt()
    c_system = v67.checkpoint_system_prompt()
    synthetic = {"checkpoints": [], "coverage": []}
    report = v67.check_batch(synthetic, ledger={"obligations": [], "preservation_obligations": [],
                                                "open_questions": []},
                             packet={"task": {"title": "t", "description": "d"}, "evidence": []}, batch=[])
    readiness = {"passed": not problems, "model_api_calls": 0, "checked_at": now(), "problems": problems,
                 "tasks": rows, "ledger_system_sha256": sha256(system), "c_system_sha256": sha256(c_system),
                 "synthetic_validation_errors": report.messages(),
                 "gpt_endpoints_file_private": (ENDPOINTS_FILE.stat().st_mode & 0o077) == 0
                 if ENDPOINTS_FILE.exists() else None}
    save(root / "preflight/readiness.json", readiness)
    write(root / "preflight/ledger-system.txt", system)
    write(root / "preflight/c-system.txt", c_system)
    return readiness


# --- model clients ------------------------------------------------------------------------------

class DeepSeekClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.arm = ARMS["deepseek"]

    def complete(self, messages, transport):
        request = {"model": self.arm["model"], "messages": messages, **self.arm["parameters"], "stream": False}
        http = Request(self.arm["base_url"] + "/chat/completions",
                       data=json.dumps(request, ensure_ascii=False).encode(),
                       headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.api_key})
        with open_with_retry(http, timeout=TIMEOUT_SEC, record=transport.append) as response:
            body = json.loads(response.read().decode())
        choice = body["choices"][0]
        return {"content": choice["message"].get("content") or "",
                "reasoning": choice["message"].get("reasoning_content") or "",
                "finish_reason": choice.get("finish_reason"), "usage": body.get("usage") or {},
                "model": body.get("model"), "response_id": body.get("id")}


class GPTClient:
    """Streamed Responses; transport failures move to the other gateway, then wait and retry."""

    def __init__(self, endpoints_file, initial=0):
        self.endpoints_file = Path(endpoints_file)
        if self.endpoints_file.stat().st_mode & 0o077:
            raise ValueError("The GPT endpoints file must be owner-only")
        self.initial = initial
        self.arm = ARMS["gpt"]

    def _request(self, endpoint, instructions, inputs):
        import openai

        import httpx

        timeout = httpx.Timeout(TIMEOUT_SEC, connect=30, read=GPT_STALL_SEC)
        with openai.OpenAI(base_url=endpoint["base_url"], api_key=endpoint["api_key"],
                           max_retries=0, timeout=timeout) as client:
            with client.responses.create(model=self.arm["model"], instructions=instructions, input=inputs,
                                         stream=True, **self.arm["parameters"]) as events:
                final = None
                for event in events:
                    if event.type in {"response.completed", "response.incomplete", "response.failed"}:
                        final = event.response
        if final is None or final.status == "failed":
            raise RuntimeError("Responses stream ended without a successful terminal response")
        return final

    def complete(self, messages, transport):
        import openai

        systems = [m for m in messages if m["role"] == "system"]
        instructions = systems[0]["content"]
        inputs = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]
        endpoints = json.loads(self.endpoints_file.read_text())
        import httpx

        retryable = (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError,
                     openai.InternalServerError, httpx.TransportError, RuntimeError)
        last = None
        for round_index in range(GPT_ROUNDS):
            order = [(self.initial + k) % len(endpoints) for k in range(len(endpoints))]
            for index in order:
                endpoint = endpoints[index]
                entry = {"round": round_index + 1, "endpoint": endpoint.get("id", index), "started_at": now()}
                try:
                    final = self._request(endpoint, instructions, inputs)
                except Exception as error:  # noqa: BLE001 - record, then retry or raise
                    transport.append({**entry, "status": "error", "error_type": type(error).__name__,
                                      "http_status": getattr(error, "status_code", None)})
                    last = error
                    in_stream = isinstance(error, openai.APIError) and getattr(error, "status_code", None) is None
                    if not (isinstance(error, retryable) or in_stream):
                        raise
                    continue
                if not final.model.startswith(self.arm["model"]):
                    raise ValueError("The gateway returned a different model: " + final.model)
                content = "".join(part.text for item in final.output if item.type == "message"
                                  for part in item.content if getattr(part, "type", "") == "output_text")
                usage = final.usage.model_dump(mode="json") if final.usage else {}
                transport.append({**entry, "status": "success", "model": final.model,
                                  "response_status": final.status})
                self.initial = index  # stay on the gateway that works
                reason = "stop" if final.status == "completed" else (
                    "length" if getattr(final.incomplete_details, "reason", "") == "max_output_tokens"
                    else final.status)
                return {"content": content, "reasoning": "", "finish_reason": reason, "usage": usage,
                        "model": final.model, "response_id": final.id}
            if round_index < len(GPT_WAITS_SEC):
                time.sleep(GPT_WAITS_SEC[round_index])
        raise RuntimeError(f"GPT gateways failed after {GPT_ROUNDS} rounds: {type(last).__name__}")


# --- stages ---------------------------------------------------------------------------------------

def call(client, messages, out, transport, usage):
    """One model call with its full record. A stored response for identical messages is reused
    (resume); a stored response for different messages is an error (never resampled)."""
    if (out / "response.json").exists():
        if read(out / "messages.json") != messages:
            raise RuntimeError(f"{out} holds a response for different messages; not resampled")
        stored = read(out / "response.json")
        result = {**stored, "content": (out / "output.txt").read_text(),
                  "reasoning": (out / "reasoning.txt").read_text() if (out / "reasoning.txt").exists() else ""}
        transport.append({"status": "resumed", "path": str(out)})
        usage.append(result.get("usage") or {})
        return result
    save(out / "messages.json", messages)
    result = client.complete(messages, transport)
    save(out / "response.json", {k: v for k, v in result.items() if k not in ("content", "reasoning")})
    write(out / "output.txt", result["content"])
    if result["reasoning"]:
        write(out / "reasoning.txt", result["reasoning"])
    usage.append(result["usage"])
    return result


def correction(errors):
    return ("The output violates the contract:\n- " + "\n- ".join(errors[:40])
            + "\nReturn the complete corrected JSON object only.")


def build_ledger(client, packet, a_dir, out, transport, usage):
    user = frozen_ledger_user(a_dir)
    messages = [{"role": "system", "content": v67.ledger_system_prompt()}, {"role": "user", "content": user}]
    sample = {**packet, "messages": messages}
    validator = Draft202012Validator(FT_SCHEMA)
    attempts = []
    payload, checks = None, None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        result = call(client, messages, out / f"ledger/attempt-{attempt}", transport, usage)
        payload, checks = validate_output(result["content"], sample, validator)
        save(out / f"ledger/attempt-{attempt}/validation.json", checks)
        structural = [e for e in checks["errors"] if e["check"] != "source"]
        attempts.append({"attempt": attempt, "finish_reason": result["finish_reason"],
                         "structural_errors": len(structural),
                         "source_errors": sum(1 for e in checks["errors"] if e["check"] == "source")})
        if checks["schema_valid"] and result["finish_reason"] == "stop":
            break
        errors = [f"{e.get('path', '')}: {e['message']}" for e in structural] or ["finish: output incomplete"]
        messages = [*messages, {"role": "assistant", "content": result["content"]},
                    {"role": "user", "content": correction(errors)}]
    if not checks or not checks["schema_valid"]:
        return None, {"status": "invalid_output", "attempts": attempts}
    ledger = {"summary": payload["summary"],
              "obligations": [{"id": f"O{i:03d}", "content": x} for i, x in enumerate(payload["obligations"], 1)],
              "preservation_obligations": [{"id": f"P{i:03d}", "content": x}
                                           for i, x in enumerate(payload["preservation_obligations"], 1)],
              "open_questions": [{"id": f"Q{i:03d}", "content": x} for i, x in enumerate(payload["open_questions"], 1)]}
    warnings = [e for e in checks["errors"] if e["check"] == "source"]
    return ledger, {"status": "passed" if not warnings else "degraded", "attempts": attempts,
                    "source_warnings": warnings}


def build_checkpoints(client, packet, ledger, out, transport, usage, batch_max_targets):
    batches = target_batches(ledger, batch_max_targets)
    points, coverage, records = [], [], []
    for index, scope in enumerate(batches, 1):
        messages = [{"role": "system", "content": v67.checkpoint_system_prompt()},
                    {"role": "user", "content": checkpoint_user(packet, ledger, (scope, index, len(batches)))}]
        payload, report = None, BatchReport(structure=["no attempt"])
        record = {"batch": index, "targets": scope, "attempts": []}
        for attempt in range(1, MAX_ATTEMPTS + 1):
            prefix = out / f"checkpoints/batch-{index:02d}/attempt-{attempt}"
            result = call(client, messages, prefix, transport, usage)
            try:
                payload = strict_json(result["content"])
                report = v67.check_batch(payload, ledger=ledger, packet=packet, batch=scope)
            except ValueError as error:
                payload, report = None, BatchReport(structure=[f"json: {error}"])
            errors = report.messages()
            if result["finish_reason"] != "stop":
                errors = [f"finish: {result['finish_reason']}", *errors]
            save(prefix / "validation.json", {"errors": errors})
            record["attempts"].append({"attempt": attempt, "finish_reason": result["finish_reason"],
                                       "errors": len(errors)})
            if not errors:
                break
            messages = [*messages, {"role": "assistant", "content": result["content"]},
                        {"role": "user", "content": correction(errors)}]
        kept, batch_coverage, outcome = v67.finalize_batch(payload, report, scope=scope)
        record.update(outcome, kept=len(kept))
        save(out / f"checkpoints/batch-{index:02d}/final.json", {**record, "checkpoints": kept})
        points += kept
        coverage += batch_coverage
        records.append(record)
    points, merged = v67.merge_duplicates(points)
    points, trimmed = v67.enforce_limit(points)
    return {"checkpoints": points, "coverage": coverage}, {
        "batches": records, "cross_batch_merged": merged, "cross_batch_trimmed": trimmed,
        "failed_batches": [r["batch"] for r in records if r["status"] == "failed"],
        "degraded_batches": [r["batch"] for r in records if r["status"] == "degraded"]}


def run_task(root, arm, row, client):
    out = root / arm / row["task_id"]
    if (out / "result.json").exists():
        return read(out / "result.json")
    # A partial directory is resumed call by call; stored responses are reused, never resampled.
    started, transport, usage = now(), [], []
    record = {"task_id": row["task_id"], "arm": arm, "version": v67.VERSION, "started_at": started}
    try:
        packet = frozen_packet(row["a_dir"])
        ledger, ledger_status = build_ledger(client, packet, row["a_dir"], out, transport, usage)
        record["ledger"] = ledger_status
        if ledger is not None:
            save(out / "ledger.json", ledger)
            payload, c_status = build_checkpoints(client, packet, ledger, out, transport, usage,
                                                  ARM_BATCH_MAX_TARGETS[arm])
            record["checkpoints"] = c_status
            flags = v67.fidelity_flags(ledger, payload["checkpoints"], packet)
            views = v67.delivery_views(payload, flags)
            save(out / "checkpoints.json", payload)
            save(out / "flags.json", flags)
            write(out / "solver-view.md", views["solver_text"] + "\n")
            save(out / "review-targets.json", views["review_targets"])
            record["summary"] = {**v67.summarize(payload, ledger),
                                 **{k: views[k] for k in ("requirements", "questions_shown",
                                                          "questions_dropped_from_solver_view", "chars")},
                                 "ledger_counts": {f: len(ledger[f]) for f in
                                                   ("obligations", "preservation_obligations", "open_questions")},
                                 "ledger_interface_flags": len(flags["ledger"]),
                                 "checkpoint_flags": len(flags["checkpoints"])}
            record["status"] = "passed" if payload["checkpoints"] else "no_checkpoints"
        else:
            record["status"] = "ledger_invalid"
    except Exception as error:  # noqa: BLE001 - keep per-task failures explicit
        record.update(status="stage_error", error_type=type(error).__name__, error=str(error)[:500])
    record.update(finished_at=now(), transport=transport, usage=usage)
    save(out / "result.json", record)
    return record


def run(root, selection, arm, workers, task_filter):
    if arm == "deepseek":
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise ValueError("Supply DEEPSEEK_API_KEY through the host process environment")
        make = lambda i: DeepSeekClient(key)  # noqa: E731
    else:
        make = lambda i: GPTClient(ENDPOINTS_FILE, initial=i)  # noqa: E731
    readiness = read(root / "preflight/readiness.json")
    if not readiness["passed"]:
        raise ValueError("Run the zero-API check first")
    rows = [r for r in selection["tasks"] if not task_filter or r["task_id"] in task_filter]
    lock = threading.Lock()

    def one(i, row):
        result = run_task(root, arm, row, make(i))
        with lock:
            print(json.dumps({"arm": arm, "task_id": row["task_id"], "status": result["status"]}), flush=True)
        return result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(one, i, row) for i, row in enumerate(rows)]):
            future.result()


def report(root, selection):
    out = {"experiment_id": EXPERIMENT_ID, "version": v67.VERSION, "generated_at": now(), "arms": {}}
    for arm in ARMS:
        rows = []
        for row in selection["tasks"]:
            path = root / arm / row["task_id"] / "result.json"
            if path.exists():
                r = read(path)
                tokens = {"input": 0, "output": 0}
                for u in r.get("usage", []):
                    tokens["input"] += u.get("prompt_tokens", u.get("input_tokens", 0)) or 0
                    tokens["output"] += u.get("completion_tokens", u.get("output_tokens", 0)) or 0
                rows.append({"task_id": row["task_id"], "split": row["split"], "status": r["status"],
                             "summary": r.get("summary"), "tokens": tokens,
                             "failed_batches": (r.get("checkpoints") or {}).get("failed_batches"),
                             "degraded_batches": (r.get("checkpoints") or {}).get("degraded_batches")})
        out["arms"][arm] = rows
    save(root / "report.json", out)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "check", "run", "report"])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--source", action="append", default=[], help="SPLIT:BATCH:A_ROOT")
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--note")
    parser.add_argument("--arm", choices=list(ARMS))
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == "prepare":
        selection = prepare(root, args.source, args.task_id, args.note)
        print(json.dumps({"mode": "prepare", "task_count": selection["task_count"]}))
        return
    selection = validate(root)
    if args.mode == "check":
        readiness = check(root, selection)
        print(json.dumps({"mode": "check", "passed": readiness["passed"], "problems": readiness["problems"]}))
    elif args.mode == "run":
        if not args.arm:
            parser.error("run requires --arm")
        run(root, selection, args.arm, args.workers, set(args.task_id or ()))
    else:
        result = report(root, selection)
        print(json.dumps({arm: len(rows) for arm, rows in result["arms"].items()}))


if __name__ == "__main__":
    main()
