"""Blind reading resolution (step 2; user, 2026-11-08): decide before seeing any patch.

Prompt "blind" is direction B: plain judgment. Prompt "evidence" is direction C1: the same inputs, plus
each answer must name its evidence type (task text read in context, external spec rule, repository
precedent, or none), and the prompt says the author wrote the task knowing the reference implementation.

Each V6.8.1 generation batch is split into near-equal contiguous chunks of at most CHUNK questions.
For each chunk, one tool-free DeepSeek call (V6.7 DeepSeek arm settings) reads the task text and the
questions: obligation, condition, numbered readings, code object, baseline observations, spec and
task quote. It never sees an implementation, a patch, a review or a label. It returns, for each
question, the reading the task intends (0 = none of them, plus what instead), a confidence (0-100)
and a short reason.

Order "original" keeps the generator's reading order. Order "shuffled" presents a seeded permutation
in which the generator's first reading is never first; answers are mapped back to the original
numbering, and the permutation is recorded. One corrective retry; a stored response is reused,
never resampled.

Model "gpt" (user, 2026-11-09): the same prompt and inputs answered by GPT-6.1 Sol (V6.7 GPT arm
settings: streamed Responses, xhigh reasoning) through the owner-only gateway list; each chunk starts
at the next gateway in turn and fails over to the others. --questions FILE ({task_id: [question id]})
restricts the run to those questions; per task they keep generation-batch order and are cut into
near-equal chunks of at most CHUNK. The file holds ids only, never a label.

usage: resolve.py --v681-root ROOT --out NEW_DIR --order original|shuffled [--prompt blind|evidence]
                  [--model deepseek|gpt] [--questions FILE] [--task-id T ...] [--workers N]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
VERSION = "v6.8-resolve-3"  # -1: blind prompt only; -2: + evidence prompt; -3: + gpt model, question subsets
SHUFFLE_SEED = "v6.8-resolve-1"  # keeps the direction B permutations reproducible
CHUNK = 35
MAX_ATTEMPTS = 2
ORDERS = ("original", "shuffled")
BASES = ("task_text", "external_spec", "repo_precedent", "none")
WHY_MAX_WORDS = 40

BLIND = """\
You resolve open questions about a software task before any implementation of it exists.

Each question names a condition and numbered readings: behaviors an implementation could have under
that condition. For each question decide which reading the task intends, i.e. the behavior a correct
implementation of the task has (hidden acceptance tests written for the task will check it):
- the number of that reading; or
- 0 if none of the readings is what the task intends; then say in "instead" what the behavior should be.

Decide from the task text, the existing repository as described by the observations (facts about the
code before the change), and any cited external specification. Decide each question on its own merits.

For each question also give "confidence": your probability (0-100) that your answer is the behavior
the task intends, and "why": the decisive evidence in at most 30 words.

Return exactly one JSON object and nothing else:
{"answers": [{"id": "CP1", "reading": 2, "confidence": 70, "why": "..."},
             {"id": "CP2", "reading": 0, "confidence": 55, "why": "...", "instead": "..."}]}
covering every question id exactly once.
"""

EVIDENCE = """\
You resolve open questions about a software task before any implementation of it exists.

The task text was written by an author who already has the reference implementation; hidden acceptance
tests check that implementation's behavior. The author wrote each sentence to steer an implementer toward
that implementation, and left unsaid what they expected an implementer to get right from the existing code
or from the standards the code follows.

Each question names a condition and numbered readings: behaviors an implementation could have under that
condition. For each question decide which reading the reference implementation follows:
- the number of that reading; or
- 0 if it follows none of them; then say in "instead" what it does.

HOW TO DECIDE
1. Consider every reading, not only the most natural one: for each, ask what in the task text, the existing
   repository or a cited standard would make it the author's intent.
2. Weigh the evidence by type:
   - "task_text": a passage of the task, read in context: what the sentence is for, what it covers and
     what it does not, and what the author would have written if they meant another reading. A sentence
     written to settle one point is not a rule for other conditions; a behavior the author states
     explicitly is meant even if it seems unusual.
   - "external_spec": a named standard, format or protocol and the specific rule in it that settles the
     question.
   - "repo_precedent": how the existing code already behaves in the analogous situation (a similar feature,
     a convention of the code being extended, an existing test), as the observations show. New behavior
     usually follows the precedents of the code it extends.
   - "none": nothing of the above settles it; you are going by what seems most natural.
3. Answer with the reading the strongest evidence supports. When no evidence settles it, still answer,
   with basis "none" and a confidence that reflects the uncertainty.

For each question give "reading"; "basis" (task_text, external_spec, repo_precedent or none); "evidence":
the quoted passage, the rule, or the observation with its file:line that you rely on (at most 40 words;
empty string for basis none); "confidence": your probability (0-100) that the reference implementation
does what you answered; and "why": at most 30 words.

Return exactly one JSON object and nothing else:
{"answers": [{"id": "CP1", "reading": 2, "basis": "repo_precedent", "evidence": "...", "confidence": 70, "why": "..."},
             {"id": "CP2", "reading": 0, "basis": "task_text", "evidence": "...", "confidence": 55, "why": "...",
              "instead": "..."}]}
covering every question id exactly once.
"""

PROMPTS = {"blind": BLIND, "evidence": EVIDENCE}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def chunks(ids, size=CHUNK):
    """Near-equal contiguous parts of at most size ids."""
    parts = max(1, math.ceil(len(ids) / size))
    step = len(ids) / parts
    return [ids[round(i * step):round((i + 1) * step)] for i in range(parts)]


def permutation(task_id, cid, count, order):
    """Presented position p (1-based) shows original reading perm[p-1] (1-based)."""
    perm = list(range(1, count + 1))
    if order == "original" or count < 2:
        return perm
    rng = random.Random(hashlib.sha256(f"{task_id}:{cid}:{SHUFFLE_SEED}".encode()).hexdigest())
    while perm[0] == 1:
        rng.shuffle(perm)
    return perm


def question_rows(rv, data, ids, task_id, order):
    rows, perms = [], {}
    for line in rv.question_lines(data, ids).decode().splitlines():
        row = json.loads(line)
        readings = row.get("readings") or []
        perm = permutation(task_id, row["id"], len(readings), order)
        perms[row["id"]] = perm
        row["readings"] = [f"{p}. {readings[o - 1]}" for p, o in enumerate(perm, 1)]
        rows.append(json.dumps(row, ensure_ascii=False))
    return rows, perms


def messages_for(task_text, rows, prompt="blind"):
    user = ("TASK\n" + task_text.strip() + "\n\nQUESTIONS (" + str(len(rows)) + ", one JSON object per line)\n"
            + "\n".join(rows) + "\n\nReturn the JSON object with one answer per question.")
    return [{"role": "system", "content": PROMPTS[prompt]}, {"role": "user", "content": user}]


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def check(payload, perms, prompt="blind"):
    if not isinstance(payload, dict) or not isinstance(payload.get("answers"), list):
        return ["output must be an object with an answers list"]
    errors, seen = [], set()
    for entry in payload["answers"]:
        if not isinstance(entry, dict) or entry.get("id") not in perms:
            errors.append(f"unknown id: {entry!r}"[:200])
            continue
        cid, count = entry["id"], len(perms[entry["id"]])
        if cid in seen:
            errors.append(f"{cid}: answered more than once")
        seen.add(cid)
        reading, confidence = entry.get("reading"), entry.get("confidence")
        if not isinstance(reading, int) or isinstance(reading, bool) or not 0 <= reading <= count:
            errors.append(f"{cid}: reading must be an integer 0-{count}")
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= confidence <= 100:
            errors.append(f"{cid}: confidence must be a number 0-100")
        if not _text(entry.get("why")):
            errors.append(f"{cid}: why is empty")
        if reading == 0 and not _text(entry.get("instead")):
            errors.append(f"{cid}: reading 0 needs instead (what the behavior should be)")
        if prompt == "evidence":
            if entry.get("basis") not in BASES:
                errors.append(f"{cid}: basis must be one of " + ", ".join(BASES))
            elif entry["basis"] != "none" and not _text(entry.get("evidence")):
                errors.append(f"{cid}: basis {entry['basis']} needs evidence (the passage, rule or observation)")
    missing = [i for i in perms if i not in seen]
    if missing:
        errors.append(f"{len(missing)} ids missing: {', '.join(missing[:30])}")
    return errors


def to_original(entry, perm):
    presented = entry["reading"]
    answer = {"reading": perm[presented - 1] if presented > 0 else 0, "presented": presented,
              "confidence": entry["confidence"], "why": " ".join(str(entry.get("why")).split()[:WHY_MAX_WORDS])}
    if presented == 0:
        answer["instead"] = entry.get("instead")
    if "basis" in entry:
        answer["basis"] = entry["basis"]
        answer["evidence"] = " ".join(str(entry.get("evidence") or "").split()[:60])
    return answer


def run_chunk(base, client, out, task_text, rows, perms, prompt):
    if (out / "chunk.json").exists():
        return base.read(out / "chunk.json")
    messages = messages_for(task_text, rows, prompt)
    transport, usage, attempts, payload = [], [], [], None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        result = base.v67run.call(client, messages, out / f"attempt-{attempt}", transport, usage)
        try:
            payload = base.strict_json(result["content"])
            errors = check(payload, perms, prompt)
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
    answers = {e["id"]: to_original(e, perms[e["id"]]) for e in payload["answers"]} if ok else {}
    record = {"ok": ok, "ids": list(perms), "permutations": perms, "answers": answers,
              "attempts": attempts, "usage": usage, "transport": transport}
    base.save(out / "chunk.json", record)
    return record


def plan_chunks(rv, data, wanted=None):
    """[(directory name, ids)]: per generation batch, or, for a question subset, over the whole task."""
    if wanted is None:
        return [(f"b{b:02d}-c{c}", ids) for b, batch in enumerate(rv.batches(data), 1)
                for c, ids in enumerate(chunks(batch), 1)]
    ids = [i for batch in rv.batches(data) for i in batch if i in wanted]
    if len(ids) != len(set(wanted)):
        raise ValueError(f"{len(set(wanted)) - len(ids)} requested question ids are not in the task")
    return [(f"s-c{c:02d}", part) for c, part in enumerate(chunks(ids), 1)] if ids else []


def run_task(base, rv, make_client, pool, v681_root, out_root, order, prompt, task_id, task_text, wanted=None):
    data = base.read(v681_root / "tasks" / task_id / "agent" / "data.json")
    jobs = []
    for name, ids in plan_chunks(rv, data, wanted):
        rows, perms = question_rows(rv, data, ids, task_id, order)
        jobs.append(pool.submit(run_chunk, base, make_client(), out_root / task_id / name,
                                task_text, rows, perms, prompt))
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v681-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--order", required=True, choices=ORDERS)
    parser.add_argument("--prompt", default="blind", choices=sorted(PROMPTS))
    parser.add_argument("--model", default="deepseek", choices=("deepseek", "gpt"))
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    base = _load("v68_resolve_base", HERE / "run.py")
    from external_agents_eval.workflows.solve_review_repair import v68_review as rv

    v681_root, out_root = args.v681_root.resolve(), args.out.resolve()
    manifest = {"version": VERSION, "order": args.order, "prompt": args.prompt, "chunk": CHUNK,
                "v681_root": str(v681_root),
                "resolve_sha256": base.sha(__file__), "model": base.v67run.ARMS[args.model]}
    wanted = None
    if args.questions:
        wanted = {t: set(ids) for t, ids in base.read(args.questions).items()}
        manifest["questions"] = {"path": str(args.questions.resolve()), "sha256": base.sha(args.questions),
                                 "tasks": len(wanted), "count": sum(map(len, wanted.values()))}
    out_root.mkdir(parents=True, exist_ok=True)
    if (out_root / "manifest.json").exists():
        if base.read(out_root / "manifest.json") != manifest:
            raise ValueError("This output directory belongs to a different resolution setup")
    else:
        base.save(out_root / "manifest.json", manifest)
    if args.model == "gpt":
        endpoints_file = base.v67run.ENDPOINTS_FILE
        gateways = len(base.read(endpoints_file))
        turn = iter(range(10 ** 9))
        make_client = lambda: base.v67run.GPTClient(endpoints_file, initial=next(turn) % gateways)  # noqa: E731
    else:
        if not os.environ.get("DEEPSEEK_API_KEY"):
            raise ValueError("Supply DEEPSEEK_API_KEY through the host process environment")
        shared = base.v67run.DeepSeekClient(os.environ["DEEPSEEK_API_KEY"])
        make_client = lambda: shared  # noqa: E731
    tasks = {r["task_id"]: r["task"]["instruction"] for r in base.read(v681_root / "selection.json")["tasks"]}
    selected = sorted(args.task_id or [t for t in tasks if (v681_root / "tasks" / t / "agent" / "data.json").exists()])
    if wanted is not None:
        unknown = sorted(set(wanted) - set(tasks))
        if unknown:
            raise ValueError(f"--questions names unknown tasks: {unknown}")
        selected = [t for t in selected if wanted.get(t)]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {t: run_task(base, rv, make_client, pool, v681_root, out_root, args.order, args.prompt,
                            t, tasks[t], None if wanted is None else wanted[t]) for t in selected}
        for task_id, futures in jobs.items():
            answers, failed, usage = {}, [], []
            for future in futures:
                try:
                    record = future.result()
                except Exception as error:  # noqa: BLE001 - keep per-chunk failures explicit
                    failed.append(f"{type(error).__name__}: {str(error)[:300]}")
                    continue
                answers.update(record["answers"])
                usage += record["usage"]
                if not record["ok"]:
                    failed.append(f"chunk {record['ids'][0]}..{record['ids'][-1]}: {record['attempts'][-1]['errors'][:3]}")
            base.save(out_root / task_id / "resolution.json",
                      {"task_id": task_id, "version": VERSION, "order": args.order, "prompt": args.prompt,
                       "model": args.model, "answers": answers,
                       "chunks": len(futures), "failed": failed, "usage": usage})
            print(json.dumps({"task": task_id, "answers": len(answers), "failed": len(failed)}), flush=True)


if __name__ == "__main__":
    main()
