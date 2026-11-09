"""V6.8.1 selection step (generation-only experiment; no Solver, review or scoring).

Per task, one tool-free DeepSeek call (the V6.7 DeepSeek arm settings: V4.1 Flash, thinking
enabled, reasoning high) reads the task text, the targeted pool (V6.8.1 agent items, T1..; Tn is
CPn of the V6.8.1 checkpoints.md) and the generic pool (V6.7 DeepSeek check points, G1..; Gn is
CPn of the V6.7 checkpoints) and returns two rankings: the TOP most valuable targeted items and the
TOP most valuable generic items. The list given to an implementer for a targeted share r (user's
default 7:3) and size k is the first round(k*r) targeted plus the first k-round(k*r) generic items
(mix()), so the ratio can be changed without new model calls. One corrective retry on an invalid
answer; a stored response is reused, never resampled.

usage: rank_select.py --v68-root ROOT --out NEW_DIR [--task-id T ...] [--workers N]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
VERSION = "v6.8.1-select"
TOP = 10
DEFAULT_SHARE = 0.7  # targeted : generic = 7 : 3 (user, 2026-11-07); adjustable in mix()
MAX_ATTEMPTS = 2


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SYSTEM = """\
You select check points for an engineer who is about to implement the software task below. The
engineer will see the task text and only the check points you select, so pick the ones most likely
to prevent a mistake.

Hidden acceptance tests will be run on the implementation. The value of a check point is how likely
it prevents a mistake those tests would catch: a behavior that a competent engineer who reads the
task carefully could still miss or get wrong (other code paths that must behave the same,
lifecycle/state interactions, edge values, the exact form of a new API, rules of an external
standard), that a test plausibly exercises, stated concretely enough to act on. Rank lower: items
that only restate what the task text plainly says, generic advice, internals no test can observe,
and near-duplicates of a higher-ranked item. Questions are fine: the engineer resolves them while
implementing; readings and observations are not answers.

You get two lists, TARGETED (ids T1, T2, ...) and GENERIC (ids G1, G2, ...). Rank each list
separately. The engineer's checklist mixes the two rankings by a configurable ratio (for example
the first 7 targeted and the first 3 generic ids), so each ranking must be useful on its own.

Return exactly one JSON object and nothing else:
{"targeted": ["T..", ...], "generic": ["G..", ...]}
with exactly %d distinct ids in each list (all ids if a list has fewer), most valuable first.
""" % TOP


def _sq(text):
    return " ".join(str(text or "").split())


def render_targeted(index, item, entities):
    parts = [f"T{index} [{item.get('kind')}] given: {_sq(item.get('given'))}; when: {_sq(item.get('when'))}; "
             f"then: {_sq(item.get('then'))}"]
    entity = entities.get(item.get("entity"))
    if entity:
        parts.append(f"object: {entity.get('symbol')} ({entity.get('path')}:{entity.get('line')})")
    if item.get("alternatives"):
        parts.append("readings: " + " | ".join(_sq(a) for a in item["alternatives"]))
    observations = [f"{o.get('path')}:{o.get('line')} {_sq(o.get('text'))}"
                    for o in item.get("observations") or [] if isinstance(o, dict)]
    if observations:
        parts.append("observations: " + "; ".join(observations))
    spec = item.get("spec") or {}
    if isinstance(spec, dict) and spec.get("name"):
        parts.append(f"spec: {_sq(spec.get('name'))}: {_sq(spec.get('rule'))}")
    if item.get("task_quote"):
        parts.append("task quote: " + _sq(item["task_quote"])[:300])
    return "\n  ".join(parts)


def render_generic(index, point):
    parts = [f"G{index} [{point.get('kind')}] given: {_sq(point.get('given'))}; when: {_sq(point.get('when'))}; "
             f"then: {_sq(point.get('then'))}"]
    if point.get("rules_out"):
        parts.append("excluded: " + _sq(point["rules_out"]))
    if point.get("alternatives"):
        parts.append("readings: " + " | ".join(_sq(a) for a in point["alternatives"]))
    observations = []
    for observation in point.get("observations") or []:
        where = ", ".join(f"{s['path']}:{s['line_start']}" for s in observation.get("sources") or []
                          if s.get("kind") == "repository")
        observations.append(f"{where} {_sq(observation.get('observation'))}".strip())
    if observations:
        parts.append("observations: " + "; ".join(observations))
    quotes = [_sq(s.get("quote"))[:300] for s in point.get("sources") or [] if s.get("kind") == "task"]
    if quotes:
        parts.append("task quotes: " + " || ".join(quotes))
    return "\n  ".join(parts)


def messages_for(task_text, targeted, entities, generic):
    entity_map = {e.get("id"): e for e in entities if isinstance(e, dict)}
    user = ("TASK\n" + task_text.strip() + "\n\nTARGETED (" + str(len(targeted)) + " items)\n"
            + "\n".join(render_targeted(i, item, entity_map) for i, item in enumerate(targeted, 1))
            + "\n\nGENERIC (" + str(len(generic)) + " items)\n"
            + "\n".join(render_generic(i, point) for i, point in enumerate(generic, 1))
            + "\n\nReturn the JSON object with both rankings.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def check(payload, n_targeted, n_generic):
    if not isinstance(payload, dict):
        return ["output is not a JSON object"]
    errors = []
    for name, prefix, size in (("targeted", "T", n_targeted), ("generic", "G", n_generic)):
        ids = payload.get(name)
        want = min(TOP, size)
        valid = {f"{prefix}{i}" for i in range(1, size + 1)}
        if not isinstance(ids, list):
            errors.append(f"{name} must be a list of ids")
            continue
        bad = [i for i in ids if i not in valid]
        if bad:
            errors.append(f"{name}: unknown ids {bad[:10]} (valid: {prefix}1..{prefix}{size})")
        if len(set(ids)) != len(ids):
            errors.append(f"{name}: ids must be distinct")
        if len(ids) != want:
            errors.append(f"{name}: exactly {want} ids are required, got {len(ids)}")
    return errors


def mix(selection, k, share=DEFAULT_SHARE):
    """Final list of size k: first round(k*share) targeted ids, then generic ids (fills up if short)."""
    targeted, generic = selection["targeted"], selection["generic"]
    n_targeted = min(len(targeted), int(k * share + 0.5))
    chosen = targeted[:n_targeted] + generic[:k - n_targeted]
    if len(chosen) < k:
        chosen += targeted[n_targeted:n_targeted + k - len(chosen)]
    return chosen


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_one(v68run, client, v68_root, out_root, task_id):
    out = out_root / task_id
    if (out / "selection.json").exists():
        return v68run.read(out / "selection.json")
    data_path = v68_root / "tasks" / task_id / "agent" / "data.json"
    generic_path = next(Path(r) / "deepseek" / task_id / "checkpoints.json" for r in v68run.V67_ROOTS
                        if (Path(r) / "deepseek" / task_id / "checkpoints.json").exists())
    row = next(r for r in v68run.read(v68_root / "selection.json")["tasks"] if r["task_id"] == task_id)
    data = v68run.read(data_path)
    targeted, generic = data["items"], v68run.read(generic_path)["checkpoints"]
    messages = messages_for(row["task"]["instruction"], targeted, data.get("entities") or [], generic)
    transport, usage, attempts = [], [], []
    payload = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        result = v68run.v67run.call(client, messages, out / f"attempt-{attempt}", transport, usage)
        try:
            payload = v68run.strict_json(result["content"])
            errors = check(payload, len(targeted), len(generic))
        except ValueError as error:
            errors = [f"json: {error}"]
        if result["finish_reason"] != "stop":
            errors = [f"finish: {result['finish_reason']}", *errors]
        attempts.append({"attempt": attempt, "finish_reason": result["finish_reason"], "errors": errors})
        if not errors:
            break
        messages = [*messages, {"role": "assistant", "content": result["content"]},
                    {"role": "user", "content": v68run.v67run.correction(errors)}]
    ok = not attempts[-1]["errors"]
    selection = {"task_id": task_id, "version": VERSION, "ok": ok,
                 "targeted": payload.get("targeted") if ok else [], "generic": payload.get("generic") if ok else [],
                 "pool_sizes": {"targeted": len(targeted), "generic": len(generic)},
                 "inputs": {"targeted": str(data_path), "targeted_sha256": sha(data_path),
                            "generic": str(generic_path), "generic_sha256": sha(generic_path)},
                 "attempts": attempts, "usage": usage, "transport": transport}
    v68run.save(out / "selection.json", selection)
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v68-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    v68run = _load("v68_select_runner", HERE / "run.py")
    v68_root, out_root = args.v68_root.resolve(), args.out.resolve()
    manifest = {"version": VERSION, "top": TOP, "default_share": DEFAULT_SHARE, "v68_root": str(v68_root),
                "select_sha256": sha(__file__), "model": v68run.v67run.ARMS["deepseek"]}
    out_root.mkdir(parents=True, exist_ok=True)
    if (out_root / "manifest.json").exists():
        if v68run.read(out_root / "manifest.json") != manifest:
            raise ValueError("This output directory belongs to a different selection setup")
    else:
        v68run.save(out_root / "manifest.json", manifest)
    tasks = args.task_id or sorted(p.name for p in (v68_root / "tasks").iterdir()
                                   if (p / "agent" / "data.json").exists())
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise ValueError("Supply DEEPSEEK_API_KEY through the host process environment")
    client = v68run.v67run.DeepSeekClient(os.environ["DEEPSEEK_API_KEY"])
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {t: pool.submit(run_one, v68run, client, v68_root, out_root, t) for t in tasks}
        for task_id, future in futures.items():
            try:
                result = future.result()
                print(json.dumps({"task_id": task_id, "ok": result["ok"], "pool": result["pool_sizes"]}), flush=True)
            except Exception as error:  # noqa: BLE001 - keep per-task failures explicit
                v68run.save(out_root / "errors" / f"{task_id}.json", {"error_type": type(error).__name__,
                                                                      "error": str(error)[:2000]})
                print(json.dumps({"task_id": task_id, "error": type(error).__name__}), flush=True)


if __name__ == "__main__":
    main()
