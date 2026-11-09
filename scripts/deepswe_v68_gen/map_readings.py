"""Map recorded behavior descriptions to reading numbers (tool-free; user, 2026-11-08).

For each pair (a review root's candidate rows, or an observation root's reference rows), one
DeepSeek call (V6.7 DeepSeek arm settings) reads each question's condition, numbered readings and
the recorded description of what the implementation does, and returns the matching reading
number (0 = none matches, -1 = the description does not tell). The mapper does not judge which
reading is correct. One corrective retry; a stored response is reused, never resampled.

usage: map_readings.py --source ROOT --kind review|observe --v681-root ROOT --out NEW_DIR [--workers N]
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
VERSION = "v6.8-map-1"
MAX_ATTEMPTS = 2

SYSTEM = """\
You map recorded observations of a software implementation to the readings of questions.
For each question you get its condition, its numbered readings (possible behaviors) and a recorded
description of what the implementation does for that condition, with evidence. Decide which reading the
described behavior matches:
- the reading number, if the description matches that reading;
- 0, if the description clearly matches none of the readings;
- -1, if the description does not say enough to tell.
Use only the description and its evidence; do not judge which reading is correct or desirable.
Return exactly one JSON object and nothing else:
{"labels": [{"id": "CP1", "reading": 2}, ...]}
covering every question id exactly once.
"""


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sq(text):
    return " ".join(str(text or "").split())


def describe(row, kind):
    if kind == "review":
        parts = [f"status: {row.get('status')}", f"behavior: {_sq(row.get('candidate_behavior'))}"]
    else:
        parts = [f"status: {row.get('status')}", f"behavior: {_sq(row.get('behavior'))}"]
    for evidence in row.get("evidence") or []:
        if isinstance(evidence, dict):
            parts.append(f"evidence ({evidence.get('kind')}): {_sq(evidence.get('detail'))[:400]}")
    return "\n  ".join(parts)


def messages_for(data, rows, kind):
    blocks = []
    for row in rows:
        item = data["items"][int(row["id"][2:]) - 1]
        readings = "\n  ".join(f"{n}. {_sq(r)}" for n, r in enumerate(item.get("alternatives") or [], 1))
        blocks.append(f"{row['id']}\n  condition: given {_sq(item.get('given'))}; when {_sq(item.get('when'))}; "
                      f"then {_sq(item.get('then'))}\n  readings:\n  {readings}\n  recorded: {describe(row, kind)}")
    user = "QUESTIONS (" + str(len(rows)) + ")\n\n" + "\n\n".join(blocks) + "\n\nReturn the JSON object."
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def check(payload, data, ids):
    if not isinstance(payload, dict) or not isinstance(payload.get("labels"), list):
        return ["output must be an object with a labels list"]
    errors, seen = [], set()
    for entry in payload["labels"]:
        if not isinstance(entry, dict) or entry.get("id") not in ids:
            errors.append(f"unknown id: {entry!r}"[:200])
            continue
        count = len(data["items"][int(entry["id"][2:]) - 1].get("alternatives") or [])
        reading = entry.get("reading")
        if not isinstance(reading, int) or isinstance(reading, bool) or not -1 <= reading <= count:
            errors.append(f"{entry['id']}: reading must be -1..{count}")
        seen.add(entry["id"])
    missing = [i for i in ids if i not in seen]
    if missing:
        errors.append(f"{len(missing)} ids missing: {', '.join(missing[:30])}")
    return errors


def source_rows(source, kind, pair):
    name = "review.json" if kind == "review" else "labels.json"
    rows = json.loads((source / "pairs" / pair / name).read_text())["rows"]
    if kind == "observe":
        rows = [r for r in rows if r.get("status") == "observed"]
    return rows


def run_one(base, client, source, kind, v681_root, out_root, pair):
    out = out_root / pair
    if (out / "mapping.json").exists():
        return json.loads((out / "mapping.json").read_text())
    task = pair.split("__")[0]
    data = json.loads((v681_root / "tasks" / task / "agent" / "data.json").read_text())
    rows = source_rows(source, kind, pair)
    ids = [r["id"] for r in rows]
    messages = messages_for(data, rows, kind)
    transport, usage, attempts, payload = [], [], [], None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        result = base.v67run.call(client, messages, out / f"attempt-{attempt}", transport, usage)
        try:
            payload = base.strict_json(result["content"])
            errors = check(payload, data, ids)
        except ValueError as error:
            errors = [f"json: {error}"]
        if result["finish_reason"] != "stop":
            errors = [f"finish: {result['finish_reason']}", *errors]
        attempts.append({"attempt": attempt, "finish_reason": result["finish_reason"], "errors": errors})
        if not errors:
            break
        messages = [*messages, {"role": "assistant", "content": result["content"]},
                    {"role": "user", "content": base.v67run.correction(errors)}]
    ok = not attempts[-1]["errors"]
    labels = {e["id"]: e["reading"] for e in payload["labels"]} if ok else {}
    mapping = {"pair": pair, "kind": kind, "version": VERSION, "ok": ok, "labels": labels, "rows": len(rows),
               "attempts": attempts, "usage": usage, "transport": transport}
    base.save(out / "mapping.json", mapping)
    return mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--kind", required=True, choices=["review", "observe"])
    parser.add_argument("--v681-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    base = _load("v68_map_base", HERE / "run.py")
    out_root = args.out.resolve()
    manifest = {"version": VERSION, "source": str(args.source.resolve()), "kind": args.kind,
                "map_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "model": base.v67run.ARMS["deepseek"]}
    out_root.mkdir(parents=True, exist_ok=True)
    if (out_root / "manifest.json").exists():
        if json.loads((out_root / "manifest.json").read_text()) != manifest:
            raise ValueError("This output directory belongs to a different mapping setup")
    else:
        base.save(out_root / "manifest.json", manifest)
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise ValueError("Supply DEEPSEEK_API_KEY through the host process environment")
    client = base.v67run.DeepSeekClient(os.environ["DEEPSEEK_API_KEY"])
    pairs = sorted(p.name for p in (args.source / "pairs").iterdir() if (p / "result.json").exists())
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {p: pool.submit(run_one, base, client, args.source.resolve(), args.kind, args.v681_root.resolve(),
                                  out_root, p) for p in pairs}
        for pair, future in futures.items():
            try:
                result = future.result()
                print(json.dumps({"pair": pair, "ok": result["ok"], "rows": result["rows"]}), flush=True)
            except Exception as error:  # noqa: BLE001 - keep per-pair failures explicit
                base.save(out_root / "errors" / f"{pair}.json", {"error_type": type(error).__name__,
                                                                 "error": str(error)[:2000]})
                print(json.dumps({"pair": pair, "error": type(error).__name__}), flush=True)


if __name__ == "__main__":
    main()
