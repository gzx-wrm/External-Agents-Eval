"""Compress teacher thinking into short visible decision notes.

Each non-final assistant turn that issues tool calls gets a 1-3 sentence
first-person note, written by the teacher model from its own thinking, the
observations it was reacting to and the action it actually took.  The note
replaces the turn's visible ``content``; thinking is then removed with the
same E2/E6 transformation.

By default the final Ledger turn is left unchanged so its content stays an
exact JSON object.  With ``--closing-notes`` it instead starts with a 2-3
sentence closing note (why the evidence is sufficient and the remaining
uncertainty goes to open_questions), a blank line, then the unchanged Ledger.
The note contains no braces, so the controller's single-object extraction
still recovers exactly the Ledger.

Results are cached per turn, so an interrupted run resumes where it stopped.
Tokenization and packaging into training materials are a separate step.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    from .build_e2_materials import transform_sample, validate_conversation
    from .build_e6_materials import TRAIN_TASKS
    from .build_e_teacher_trajectory import dump, sha, write_json
except ImportError:
    from build_e2_materials import transform_sample, validate_conversation
    from build_e6_materials import TRAIN_TASKS
    from build_e_teacher_trajectory import dump, sha, write_json

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = HERE / "data/e_teacher_trajectory_30_human_latest_v1"
DEFAULT_OUTPUT = HERE / "data/e9_rationale_notes_v1"
DEFAULT_LABELS = HERE / "data/e6_normal13_loopheavy_v1"
DEFAULT_MODEL_CONFIG = HERE.parent / "configs/v6.1-deepseek-v4.1-flash.json"
SOURCE_INDEX = "_source_index"
PROMPT_VERSION = "rationale-note-v2"
CLOSING_PROMPT_VERSION = "closing-note-v1"

INSTRUCTIONS = """You are rewriting one step of your own earlier repository-inspection session.
At this step you thought privately and then issued tool calls. Write the short
visible note you would say aloud right before those tool calls.

Your role in that session is specification reviewer: you inspect a frozen
repository to extract the obligations a future change must satisfy, then
return a Ledger JSON. You never implement, edit or plan code changes yourself.

Requirements:
- 1 to 3 sentences, at most {max_words} words, first person, plain prose.
- State what the latest tool results established, ruled out or got wrong (if
  there are any), then why the next action is the useful one. After an error,
  say how the next action corrects it.
- Frame the purpose as gathering evidence for obligations, not as preparing
  to implement, add, wire or change code.
- Only use facts present in the private reasoning, the tool results or the
  next action. Do not introduce new file names, symbols or line numbers.
- Stay consistent with the next action; do not predict its results.
- No code blocks, JSON, markdown headings, lists or tool-call syntax. Do not
  mention private reasoning, notes, a teacher or a rewrite.
- Do not repeat an earlier note verbatim.

Return only the note text."""

CLOSING_INSTRUCTIONS = """You are rewriting the last step of your own earlier repository-inspection session.
At this step you stopped calling tools and returned the Ledger JSON shown below.
Write the short visible note you would say aloud right before that JSON.

Your role in that session is specification reviewer: you inspect a frozen
repository to extract the obligations a future change must satisfy, then
return a Ledger JSON. You never implement, edit or plan code changes yourself.

Requirements:
- 2 to 3 sentences, at most {max_words} words, first person, plain prose.
- Say what the latest tool results confirmed, then why the inspected evidence
  is now sufficient to state the requested and preservation obligations.
- Say that the remaining uncertainty is recorded as open questions rather than
  searched further, and that you now return the Ledger.
- Do not propose or announce any further tool call, search or read.
- Only use facts present in the private reasoning, the tool results, the
  earlier notes or the Ledger. Do not introduce new file names, symbols or
  line numbers. Do not quote the Ledger or count its entries.
- No braces, code blocks, JSON, markdown headings, lists or tool-call syntax.
  Do not mention private reasoning, notes, a teacher or a rewrite.

Return only the note text."""

PATH_PATTERN = re.compile(r"(?<![\w/.-])(?:[\w.-]+/)+[\w.-]+|(?<![\w/.-])[\w-]+\.[A-Za-z]{1,5}(?![\w])")
BACKTICK_PATTERN = re.compile(r"`([^`\n]+)`")
LINE_PATTERN = re.compile(r"\blines?\s+(\d+)(?:\s*[-–]\s*(\d+))?", re.IGNORECASE)
FORBIDDEN = ("```", "<tool_call", "<function=", "<parameter=", "<think", "</think", "<|im_")
FORBIDDEN_WORDS = re.compile(r"\b(private reasoning|my notes?|teacher|rewrit\w*)\b", re.IGNORECASE)
# The reviewer role must not drift into implementation planning.
IMPLEMENTER = re.compile(r"\bI(?:'ll| will| need to| can| must|'m going to)?\s+(?:then\s+)?"
                         r"(?:implement|add|wire|write|modify|edit|change|make (?:any |the )?changes)\b"
                         r"|\bbefore (?:I |we )?(?:implement|add|wir|writ|mak|chang)\w*", re.IGNORECASE)
# A closing note must end the inspection, not announce another tool call.
CONTINUES = re.compile(r"\b(?:I(?:'ll| will| need to|'m going to| should| must)|let me|next,? I)\s+(?:also\s+|now\s+|then\s+)?"
                       r"(?:read|search|grep|list|inspect|check|look|open|find|explore|verify|examine|confirm)\b",
                       re.IGNORECASE)
STOPS = re.compile(r"\b(?:return|output|produce|deliver|finali[sz]e|emit|provide)\w*\b[^.]*\bledger\b"
                   r"|\bledger\b[^.]*\b(?:now|ready)\b", re.IGNORECASE)


def clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [{len(text) - limit} more characters]"


def describe_call(call: dict) -> str:
    function = call["function"]
    return f"{function['name']}({json.dumps(function['arguments'], ensure_ascii=False)})"


def task_text(sample: dict, limit: int) -> str:
    user = next(message for message in sample["messages"] if message["role"] == "user")
    return clip(user["content"], limit)


def plan_turns(source: dict) -> tuple[dict, dict, list[dict], dict]:
    """Return the transformed sample, change report, one record per note and the final turn."""

    marked = copy.deepcopy(source)
    for index, message in enumerate(marked["messages"]):
        if message["role"] == "assistant":
            message[SOURCE_INDEX] = index
    sample, changes = transform_sample(marked)
    assistants = [index for index, message in enumerate(sample["messages"]) if message["role"] == "assistant"]
    turns, observations = [], []
    calls = {}
    for index, message in enumerate(sample["messages"]):
        if message["role"] == "tool":
            observations.append((calls[message["tool_call_id"]], message["content"]))
            continue
        if message["role"] != "assistant":
            continue
        for call in message.get("tool_calls", []):
            calls[call["id"]] = call
        original = source["messages"][message[SOURCE_INDEX]]
        if index == assistants[-1]:
            final = {
                "message_index": index,
                "source_message_index": message[SOURCE_INDEX],
                "thinking": original.get("reasoning_content") or "",
                "observations": observations,
            }
        elif message.get("tool_calls"):
            turns.append({
                "message_index": index,
                "source_message_index": message[SOURCE_INDEX],
                "thinking": original.get("reasoning_content") or "",
                "visible_content": original.get("content") or "",
                "observations": observations,
                "action": message["tool_calls"],
            })
        observations = []
    return sample, changes, turns, final


def build_prompt(task: str, turn: dict, previous: list[str], args: argparse.Namespace) -> str:
    observed = "\n\n".join(
        f"{describe_call(call)} returned:\n{clip(content, args.observation_chars)}"
        for call, content in turn["observations"]
    ) or "(none; this is the first step)"
    earlier = "\n".join(f"- {note}" for note in previous[-args.previous_notes:]) or "(none)"
    action = "\n".join(describe_call(call) for call in turn["action"])
    parts = [
        INSTRUCTIONS.format(max_words=args.max_words),
        f"TASK\n{task}",
        f"EARLIER VISIBLE NOTES\n{earlier}",
        f"LATEST TOOL RESULTS\n{observed}",
        f"PRIVATE REASONING AT THIS STEP\n{turn['thinking'] or '(empty)'}",
    ]
    if turn["visible_content"].strip():
        parts.append(f"TEXT ALREADY SHOWN AT THIS STEP\n{turn['visible_content'].strip()}")
    parts.append(f"NEXT ACTION\n{action}")
    return "\n\n".join(parts)


def check_note(note: str, grounding: str, previous: list[str], args: argparse.Namespace,
               max_words: int | None = None) -> list[str]:
    problems = []
    if not note:
        return ["empty note"]
    words, max_words = len(note.split()), max_words or args.max_words
    if words > max_words:
        problems.append(f"{words} words exceeds {max_words}")
    sentences = len([part for part in re.split(r"(?<=[.!?])\s+", note) if part.strip()])
    if sentences > 3:
        problems.append(f"{sentences} sentences exceeds 3")
    if note.count("\n") > 2:
        problems.append("too many line breaks")
    problems += [f"contains {token!r}" for token in FORBIDDEN if token in note]
    if note.lstrip().startswith(("{", "[", "#", "-", "*")):
        problems.append("not plain prose")
    if match := FORBIDDEN_WORDS.search(note):
        problems.append(f"mentions {match.group(0)!r}")
    if match := IMPLEMENTER.search(note):
        problems.append(f"implementer framing {match.group(0)!r}; the role only gathers evidence")
    for token in PATH_PATTERN.findall(note) + BACKTICK_PATTERN.findall(note):
        token = token.strip(".,;:()'\"")
        if token and token not in grounding:
            problems.append(f"ungrounded reference {token!r}")
    for start, end in LINE_PATTERN.findall(note):
        for number in filter(None, (start, end)):
            if not re.search(rf"(?<!\d){number}(?!\d)", grounding):
                problems.append(f"ungrounded line number {number}")
    if note in previous:
        problems.append("repeats an earlier note")
    return problems


def build_closing_prompt(task: str, final: dict, previous: list[str], ledger: str,
                         args: argparse.Namespace) -> str:
    observed = "\n\n".join(
        f"{describe_call(call)} returned:\n{clip(content, args.observation_chars)}"
        for call, content in final["observations"]
    ) or "(none)"
    earlier = "\n".join(f"- {note}" for note in previous[-args.closing_previous_notes:]) or "(none)"
    return "\n\n".join([
        CLOSING_INSTRUCTIONS.format(max_words=args.closing_max_words),
        f"TASK\n{task}",
        f"EARLIER VISIBLE NOTES\n{earlier}",
        f"LATEST TOOL RESULTS\n{observed}",
        f"PRIVATE REASONING AT THIS STEP\n{clip(final['thinking'], args.closing_thinking_chars) or '(empty)'}",
        f"LEDGER RETURNED AT THIS STEP\n{ledger}",
    ])


def check_closing_note(note: str, grounding: str, ledger: dict, args: argparse.Namespace) -> list[str]:
    problems = check_note(note, grounding, [], args, max_words=args.closing_max_words)
    sentences = len([part for part in re.split(r"(?<=[.!?])\s+", note) if part.strip()])
    if note and sentences < 2:
        problems.append("needs 2 to 3 sentences")
    if "{" in note or "}" in note:
        problems.append("contains braces; the controller must extract exactly one JSON object")
    if match := CONTINUES.search(note):
        problems.append(f"announces another tool action {match.group(0)!r}; the inspection is over")
    if note and not STOPS.search(note):
        problems.append("does not state that the Ledger is returned now")
    if ledger.get("open_questions") and not re.search(r"\bopen questions?\b", note, re.IGNORECASE):
        problems.append("does not say the remaining uncertainty is recorded as open questions")
    return problems


class Teacher:
    def __init__(self, args: argparse.Namespace):
        config = json.loads(args.model_config.read_text())["default"]
        self.url = (args.base_url or config["base_url"]).rstrip("/") + "/chat/completions"
        self.model = args.model or config["model"]
        self.key = os.environ.get(args.api_key_env)
        if not self.key:
            raise RuntimeError(f"Set {args.api_key_env} before calling the teacher")
        # Thinking mode ignores temperature, so it is only sent when explicitly given.
        sampling = {} if args.temperature is None else {"temperature": args.temperature}
        self.body = {"max_tokens": args.max_tokens, **sampling,
                     "thinking": {"type": args.thinking},
                     **({"reasoning_effort": args.reasoning_effort} if args.thinking == "enabled" else {}),
                     **json.loads(args.extra_body)}
        self.timeout = args.timeout
        self.attempts = args.transport_attempts

    def complete(self, prompt: str) -> dict:
        payload = {"model": self.model, "messages": [{"role": "user", "content": prompt}], **self.body}
        request = Request(self.url, data=json.dumps(payload).encode(), method="POST", headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
        for attempt in range(1, self.attempts + 1):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read())
            except HTTPError as error:
                if error.code < 500 and error.code != 429 or attempt == self.attempts:
                    raise RuntimeError(f"Teacher HTTP {error.code}: {error.read()[:500]!r}") from error
            except (URLError, TimeoutError) as error:
                if attempt == self.attempts:
                    raise RuntimeError(f"Teacher transport failed: {error}") from error
            time.sleep(min(60, 5 * 2 ** attempt))
        raise AssertionError("unreachable")


def compress_task(task_id: str, source: dict, labels: dict | None, teacher: Teacher, output: Path,
                  args: argparse.Namespace) -> dict:
    sample, changes, turns, final = plan_turns(source)
    task = task_text(source, args.task_chars)
    cache = output / "cache" / task_id
    cache.mkdir(parents=True, exist_ok=True)
    notes, records = [], []
    for turn in turns:
        prompt = build_prompt(task, turn, notes, args)
        grounding = "\n".join([
            task, turn["thinking"], turn["visible_content"],
            *(describe_call(call) + "\n" + content for call, content in turn["observations"]),
            *(describe_call(call) for call in turn["action"]),
        ])
        key = hashlib.sha256((PROMPT_VERSION + self_model(teacher) + prompt).encode()).hexdigest()
        path = cache / f"{turn['source_message_index']:04d}.json"
        record = json.loads(path.read_text()) if path.exists() else None
        if not record or record["request_sha256"] != key or record["problems"]:
            note, problems, attempts = generate(
                teacher, prompt, lambda text: check_note(text, grounding, notes, args), args)
            record = {"request_sha256": key, "prompt_version": PROMPT_VERSION,
                      "source_message_index": turn["source_message_index"],
                      "message_index": turn["message_index"], "note": note,
                      "problems": problems, "attempts": attempts,
                      "thinking_characters": len(turn["thinking"])}
            write_json(path, record)
        if record["problems"]:
            raise ValueError(f"{task_id} message {turn['source_message_index']}: {record['problems']}")
        sample["messages"][turn["message_index"]]["content"] = record["note"]
        notes.append(record["note"])
        records.append({**record, "thinking": turn["thinking"],
                        "action": [describe_call(call) for call in turn["action"]]})
    for message in sample["messages"]:
        message.pop(SOURCE_INDEX, None)
    sample["task_id"] = task_id
    if labels is not None:
        adopt_labels(sample, labels)
    closing = None
    if args.closing_notes:
        closing = write_closing_note(task_id, sample, final, task, notes, teacher, cache, args)
    validate_conversation(sample)
    write_json(output / "rationales" / f"{task_id}.json", {
        "task_id": task_id, "notes": records, "closing": closing, "transform": changes})
    print(f"{task_id}: {len(records)} notes" + (", closing note" if closing else ""), flush=True)
    return sample


def generate(teacher: "Teacher", prompt: str, check, args: argparse.Namespace) -> tuple[str, list[str], list[dict]]:
    attempts, feedback = [], ""
    for _ in range(args.attempts):
        response = teacher.complete(prompt + feedback)
        message = response["choices"][0]["message"]
        note = " ".join((message.get("content") or "").split())
        finish = response["choices"][0].get("finish_reason")
        problems = check(note)
        if finish != "stop":
            problems.append(f"finish_reason {finish!r}")
        attempts.append({"note": note, "problems": problems, "finish_reason": finish,
                         "reasoning_characters": len(message.get("reasoning_content") or ""),
                         "usage": response.get("usage")})
        if not problems:
            break
        feedback = ("\n\nYour previous answer was rejected: " + "; ".join(problems)
                    + ". Write a corrected note.")
    return note, problems, attempts


def write_closing_note(task_id: str, sample: dict, final: dict, task: str, notes: list[str],
                       teacher: "Teacher", cache: Path, args: argparse.Namespace) -> dict:
    """Prefix the final Ledger turn with a teacher-written closing note."""

    message = sample["messages"][final["message_index"]]
    ledger_text = message["content"]
    ledger = json.loads(ledger_text)
    prompt = build_closing_prompt(task, final, notes, ledger_text, args)
    grounding = "\n".join([
        task, final["thinking"], ledger_text, *notes,
        *(describe_call(call) + "\n" + content for call, content in final["observations"]),
        # Earlier observations are visible in the trajectory and may be referenced too.
        *(item["content"] for item in sample["messages"] if item["role"] == "tool"),
    ])
    key = hashlib.sha256((CLOSING_PROMPT_VERSION + self_model(teacher) + prompt).encode()).hexdigest()
    path = cache / f"closing-{final['source_message_index']:04d}.json"
    record = json.loads(path.read_text()) if path.exists() else None
    if not record or record["request_sha256"] != key or record["problems"]:
        note, problems, attempts = generate(
            teacher, prompt, lambda text: check_closing_note(text, grounding, ledger, args), args)
        record = {"request_sha256": key, "prompt_version": CLOSING_PROMPT_VERSION,
                  "source_message_index": final["source_message_index"],
                  "message_index": final["message_index"], "note": note,
                  "problems": problems, "attempts": attempts,
                  "thinking_characters": len(final["thinking"]),
                  "ledger_sha256": hashlib.sha256(ledger_text.encode()).hexdigest()}
        write_json(path, record)
    if record["problems"]:
        raise ValueError(f"{task_id} closing note: {record['problems']}")
    message["content"] = record["note"] + "\n\n" + ledger_text
    return {**record, "thinking": final["thinking"]}


def self_model(teacher: Teacher) -> str:
    return teacher.model + dump(teacher.body)


def adopt_labels(sample: dict, reference: dict) -> None:
    """Take the final Ledger from the reference materials (with their citation fixes).

    Everything except the new notes must already match the reference row.
    """

    messages, expected = sample["messages"], reference["messages"]
    if len(messages) != len(expected):
        raise ValueError(f"{sample['task_id']}: message count differs from label reference")
    last = len(messages) - 1
    for index, (message, other) in enumerate(zip(messages, expected, strict=True)):
        if message["role"] != other["role"] or message.get("tool_calls") != other.get("tool_calls"):
            raise ValueError(f"{sample['task_id']}: message {index} differs from label reference")
        if message["role"] != "assistant" and message["content"] != other["content"]:
            raise ValueError(f"{sample['task_id']}: observation {index} differs from label reference")
    messages[last]["content"] = expected[last]["content"]


def write_review(output: Path, task_ids: list[str], thinking_chars: int) -> None:
    lines = ["# Thinking → visible note review", ""]
    for task_id in task_ids:
        data = json.loads((output / "rationales" / f"{task_id}.json").read_text())
        lines += [f"## {task_id}", ""]
        for note in data["notes"]:
            lines += [
                f"**message {note['source_message_index']}** · thinking {note['thinking_characters']} chars",
                "", "> " + clip(note["thinking"], thinking_chars).replace("\n", "\n> "), "",
                f"Note: {note['note']}", "", "Action: " + "; ".join(note["action"]), "",
            ]
        if closing := data.get("closing"):
            lines += [
                f"**closing (message {closing['source_message_index']})** · thinking "
                f"{closing['thinking_characters']} chars",
                "", "> " + clip(closing["thinking"], thinking_chars).replace("\n", "\n> "), "",
                f"Closing note: {closing['note']}", "", "Then: Ledger JSON", "",
            ]
    (output / "review.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS,
                        help="Materials whose final Ledgers (with citation fixes) are reused; 'none' keeps the source")
    parser.add_argument("--tasks", nargs="+", default=TRAIN_TASKS, help="Default: the E6 13 training tasks")
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--model", help="Override the model in --model-config")
    parser.add_argument("--base-url", help="Override the base URL in --model-config")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--thinking", choices=("enabled", "disabled"), default="enabled")
    parser.add_argument("--reasoning-effort", choices=("low", "high", "max"), default="high")
    parser.add_argument("--extra-body", default="{}", help="JSON merged into each request")
    parser.add_argument("--temperature", type=float, help="Only effective with --thinking disabled")
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--transport-attempts", type=int, default=4)
    parser.add_argument("--attempts", type=int, default=3, help="Generations per turn when validation fails")
    parser.add_argument("--max-words", type=int, default=60)
    parser.add_argument("--observation-chars", type=int, default=1500)
    parser.add_argument("--task-chars", type=int, default=4000)
    parser.add_argument("--previous-notes", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4, help="Tasks processed in parallel; turns stay sequential")
    parser.add_argument("--review-thinking-chars", type=int, default=600)
    parser.add_argument("--closing-notes", action="store_true",
                        help="Prefix each final Ledger turn with a closing note")
    parser.add_argument("--closing-max-words", type=int, default=75)
    parser.add_argument("--closing-previous-notes", type=int, default=8)
    parser.add_argument("--closing-thinking-chars", type=int, default=40000)
    args = parser.parse_args()

    source_manifest = json.loads((args.source / "manifest.json").read_text())
    rows = (args.source / "train.jsonl").read_text().splitlines()
    by_task = {task["task_id"]: json.loads(line) for task, line in zip(source_manifest["tasks"], rows, strict=True)}
    missing = [task for task in args.tasks if task not in by_task]
    if missing:
        raise ValueError(f"Tasks not in source: {missing}")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "rationales").mkdir(exist_ok=True)
    labels = {}
    if str(args.labels) != "none":
        labels = {row["task_id"]: row for row in map(json.loads, (args.labels / "train.jsonl").read_text().splitlines())}
        if missing := [task for task in args.tasks if task not in labels]:
            raise ValueError(f"Tasks not in label reference: {missing}")
    teacher = Teacher(args)
    with ThreadPoolExecutor(args.workers) as pool:
        samples = list(pool.map(
            lambda task: compress_task(task, by_task[task], labels.get(task), teacher, args.output, args),
            args.tasks))

    (args.output / "train.jsonl").write_text("".join(dump(sample) + "\n" for sample in samples))
    write_review(args.output, args.tasks, args.review_thinking_chars)
    rationales = [json.loads((args.output / "rationales" / f"{task}.json").read_text()) for task in args.tasks]
    notes = [note for data in rationales for note in data["notes"]]
    closings = [data["closing"] for data in rationales if data.get("closing")]
    write_json(args.output / "run.json", {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prompt_version": PROMPT_VERSION, "instructions": INSTRUCTIONS,
        "source": str(args.source), "source_manifest_sha256": sha(args.source / "manifest.json"),
        "teacher": {"url": teacher.url, "model": teacher.model, "body": teacher.body},
        "tasks": args.tasks, "notes": len(notes),
        "closing_notes": len(closings),
        **({"closing_prompt_version": CLOSING_PROMPT_VERSION, "closing_instructions": CLOSING_INSTRUCTIONS}
           if args.closing_notes else {}),
        "thinking_characters": sum(note["thinking_characters"] for note in notes),
        "note_characters": sum(len(note["note"]) for note in notes),
        "final_ledger_source": "source" if str(args.labels) == "none" else str(args.labels),
        "final_ledger_sha256": {data["task_id"]: hashlib.sha256(
            sample["messages"][-1]["content"].encode()).hexdigest() if not data.get("closing")
            else data["closing"]["ledger_sha256"] for data, sample in zip(rationales, samples, strict=True)},
        "settings": {key: value for key, value in vars(args).items()
                     if key not in {"source", "output", "model_config", "tasks", "labels"}},
    })
    print(f"Wrote {len(samples)} trajectories, {len(notes)} notes and {len(closings)} closing notes to {args.output}")


if __name__ == "__main__":
    main()
