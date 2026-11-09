"""V6.6 stage C: validate and request observable check points for a frozen ledger."""
from __future__ import annotations

import asyncio
import copy
import difflib
import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from jsonschema import Draft202012Validator

from .artifacts import ArtifactStore, sha256_json
from .ft_protocol import strict_json
from .v66_prompts import CHECKPOINT_SCHEMA, FACETS, checkpoint_system_prompt, checkpoint_user
from .v66_transport import RETRY_WAITS_SEC, open_with_retry

STRENGTH = {"task": "required", "analog": "inferred", "contract": "inferred",
            "convention": "inferred", "open": "open"}
INFERRED = {"analog", "contract", "convention"}
# "unset or empty", "empty or missing", "null or empty" ... merge separate D3 cases.
MERGED_ABSENCE = re.compile(
    r"\b(?:unset|omitted|missing|absent|null|undefined|none)\s*(?:,|/|or)\s*empty\b"
    r"|\bempty\s*(?:,|/|or)\s*(?:unset|omitted|missing|absent|null|undefined|none)\b",
    re.IGNORECASE)
DUPLICATE_RATIO = 0.9


def _squash(text: str) -> str:
    return " ".join(text.split())


def _rule_text(point: dict[str, Any]) -> str:
    return _squash(" | ".join((point["given"], point["when"], point["then"]))).lower()


def _evidence_lines(packet: dict[str, Any]) -> dict[str, dict[int, str]]:
    lines: dict[str, dict[int, str]] = defaultdict(dict)
    for item in packet["evidence"]:
        for number, line in enumerate(item["text"].splitlines(), item["line_start"]):
            lines[item["path"]][number] = line
    return lines


def _source_text(source: dict[str, Any], task_text: dict[str, str],
                 lines: dict[str, dict[int, str]]) -> str | None:
    if source["kind"] == "task":
        return task_text[source["field"]]
    numbers = range(source["line_start"], source["line_end"] + 1)
    available = lines.get(source["path"], {})
    if not all(number in available for number in numbers):
        return None
    return "\n".join(available[number] for number in numbers)


ITEM_SCHEMA = CHECKPOINT_SCHEMA["properties"]["checkpoints"]["items"]
COVERAGE_ITEM_SCHEMA = CHECKPOINT_SCHEMA["properties"]["coverage"]["items"]


def ledger_targets(ledger: dict[str, Any]) -> dict[str, Any]:
    return {item["id"]: item["content"] for field in ("obligations", "preservation_obligations")
            for item in ledger[field]}


def target_batches(ledger: dict[str, Any], max_targets: int) -> list[list[str]]:
    """Ledger order, split into the fewest even batches of at most max_targets targets."""
    ids = list(ledger_targets(ledger))
    if not ids:
        return []
    count = -(-len(ids) // max_targets)
    size, extra = divmod(len(ids), count)
    batches, start = [], 0
    for index in range(count):
        end = start + size + (1 if index < extra else 0)
        batches.append(ids[start:end])
        start = end
    return batches


@dataclass
class BatchReport:
    """structure: the output is unusable; points: per-point violations (the point is dropped
    after the last attempt); batch: limit, duplicate and coverage problems (resolved by rules)."""

    structure: list[str] = field(default_factory=list)
    points: dict[int, list[str]] = field(default_factory=lambda: defaultdict(list))
    batch: list[str] = field(default_factory=list)

    def messages(self) -> list[str]:
        return [*self.structure, *(e for i in sorted(self.points) for e in self.points[i]), *self.batch]


def check_batch(
    payload: Any, *, ledger: dict[str, Any], packet: dict[str, Any], max_per_target: int,
    batch: list[str] | None = None,
) -> BatchReport:
    """Classify every contract violation of one output; batch=None checks all ledger targets."""
    report = BatchReport()
    for error in Draft202012Validator(CHECKPOINT_SCHEMA).iter_errors(payload):
        path = list(error.absolute_path)
        message = f"schema: {'/'.join(map(str, path)) or '(root)'}: {error.message}"
        if len(path) >= 2 and path[0] == "checkpoints" and isinstance(path[1], int):
            report.points[path[1]].append(message)
        else:
            report.batch.append(message)
    if not isinstance(payload, dict) or not isinstance(payload.get("checkpoints"), list):
        # Only an output without a checkpoints array is unusable as a whole.
        report.structure, report.batch = report.batch, []
        return report
    targets = ledger_targets(ledger)
    scope = list(targets) if batch is None else list(batch)
    task_text = {"title": packet["task"]["title"], "description": packet["task"]["description"]}
    lines = _evidence_lines(packet)
    counts: dict[str, int] = defaultdict(int)
    covered: dict[tuple[str, str], int] = defaultdict(int)
    points = [(index, point) for index, point in enumerate(payload["checkpoints"])
              if index not in report.points]
    for index, point in points:
        errors = report.points[index]
        where = f"checkpoints/{index} ({','.join(point['targets'])}/{point['facet']})"
        unknown = [target for target in point["targets"] if target not in targets]
        if unknown:
            errors.append(f"{where}: unknown target {', '.join(unknown)}")
            continue
        outside = [target for target in point["targets"] if target not in scope]
        if outside:
            errors.append(f"{where}: targets outside this batch: {', '.join(outside)}")
            continue
        for target in point["targets"]:
            counts[target] += 1
            covered[(target, point["facet"])] += 1
        resolution = point["resolution"]
        if point["strength"] != STRENGTH[resolution]:
            errors.append(f"{where}: resolution {resolution} requires strength {STRENGTH[resolution]}")
        kinds = {source["kind"] for source in point["sources"]}
        if resolution == "task" and "task" not in kinds:
            errors.append(f"{where}: a task resolution needs an exact task quote")
        if resolution in {"analog", "contract"} and "repository" not in kinds:
            errors.append(f"{where}: an {resolution} resolution needs repository lines")
        if resolution == "convention" and not (point["convention"] or "").strip():
            errors.append(f"{where}: name the convention")
        if resolution == "open":
            if len(point["alternatives"]) < 2:
                errors.append(f"{where}: an open check point needs at least two alternatives")
            if point["favored"] is not None and point["favored"] not in point["alternatives"]:
                errors.append(f"{where}: favored must be one of the alternatives or null")
            if point["rules_out"] is not None:
                errors.append(f"{where}: an open check point lists alternatives instead of rules_out")
        else:
            if point["alternatives"] or point["favored"] is not None:
                errors.append(f"{where}: only open check points carry alternatives/favored")
            if not (point["rules_out"] or "").strip():
                errors.append(f"{where}: name in rules_out the reasonable implementation it excludes, "
                              "or mark the facet not_applicable")
        source_texts = []
        for source in point["sources"]:
            text = _source_text(source, task_text, lines)
            if source["kind"] == "task":
                if source["quote"] not in text:
                    errors.append(f"{where}: task quote is not exact: {source['quote'][:80]}")
                else:
                    source_texts.append(source["quote"])
            elif text is None:
                errors.append(f"{where}: repository lines are outside the supplied evidence: "
                              f"{source['path']} {source['line_start']}-{source['line_end']}")
            else:
                source_texts.append(text)
        needs_basis = resolution in {"analog", "contract"} or (
            resolution == "open" and point["favored"] is not None and not (point["convention"] or "").strip())
        if needs_basis or resolution == "convention" or (resolution == "open" and point["favored"] is not None):
            if not (point["basis_behavior"] or "").strip():
                errors.append(f"{where}: state in basis_behavior what the basis requires")
        if needs_basis:
            quote = _squash(point["basis_quote"] or "")
            if not quote:
                errors.append(f"{where}: copy the decisive text into basis_quote")
            elif not any(quote in _squash(text) for text in source_texts):
                errors.append(f"{where}: basis_quote is not exact text from the cited sources")
        for name in ("given", "when", "then"):
            if MERGED_ABSENCE.search(point[name]):
                errors.append(f"{where}: {name} merges omitted/null/empty; write separate cases")
    report.points = defaultdict(list, {index: errors for index, errors in report.points.items() if errors})
    for target, count in counts.items():
        if count > max_per_target:
            report.batch.append(f"{target}: {count} check points exceed the limit of {max_per_target}")
    for first in range(len(points)):
        for second in range(first + 1, len(points)):
            (i, a), (j, b) = points[first], points[second]
            if _same_rule(a, b):
                report.batch.append(f"checkpoints/{i} and checkpoints/{j}: the same rule; merge them "
                                    f"into one check point listing targets {sorted(set(a['targets']) | set(b['targets']))}")
    coverage = payload.get("coverage") if isinstance(payload.get("coverage"), list) else []
    seen: dict[tuple[str, str], str] = {}
    for entry in coverage:
        if not Draft202012Validator(COVERAGE_ITEM_SCHEMA).is_valid(entry):
            continue
        key = (entry["target"], entry["facet"])
        if key in seen:
            report.batch.append(f"coverage: duplicate {key[0]}/{key[1]}")
        seen[key] = entry["status"]
        if entry["status"] == "not_applicable" and not (entry["reason"] or "").strip():
            report.batch.append(f"coverage: {key[0]}/{key[1]} needs a not_applicable reason")
    for target in scope:
        for facet in FACETS:
            status = seen.get((target, facet))
            if status is None:
                report.batch.append(f"coverage: missing {target}/{facet}")
            elif (status == "covered") != bool(covered.get((target, facet))):
                report.batch.append(f"coverage: {target}/{facet} is {status} but has "
                                    f"{covered.get((target, facet), 0)} check points")
    for key in seen:
        if key[0] not in targets:
            report.batch.append(f"coverage: unknown target {key[0]}")
        elif key[0] not in scope:
            report.batch.append(f"coverage: target outside this batch {key[0]}")
    return report


def validate_checkpoints(
    payload: Any, *, ledger: dict[str, Any], packet: dict[str, Any], max_per_target: int,
    batch: list[str] | None = None,
) -> list[str]:
    """Return every contract violation; an empty list means the output is usable."""
    return check_batch(payload, ledger=ledger, packet=packet, max_per_target=max_per_target,
                       batch=batch).messages()


def _same_rule(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return (a["facet"] == b["facet"] and a["resolution"] == b["resolution"]
            and difflib.SequenceMatcher(None, _rule_text(a), _rule_text(b)).ratio() >= DUPLICATE_RATIO)


def merge_duplicates(points: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep the earlier of two near-identical rules and give it the later one's targets."""
    kept: list[dict[str, Any]] = []
    merged: list[dict[str, Any]] = []
    for point in points:
        for earlier in kept:
            if _same_rule(earlier, point):
                added = [target for target in point["targets"] if target not in earlier["targets"]]
                earlier["targets"] = earlier["targets"] + added
                merged.append({"into_targets": list(earlier["targets"]), "facet": point["facet"],
                               "from_targets": list(point["targets"])})
                break
        else:
            kept.append(point)
    return kept, merged


def enforce_limit(points: list[dict[str, Any]], limit: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep each target's first `limit` check points in output order."""
    counts: dict[str, int] = defaultdict(int)
    kept: list[dict[str, Any]] = []
    trimmed: list[dict[str, Any]] = []
    for point in points:
        allowed = [target for target in point["targets"] if counts[target] < limit]
        removed = [target for target in point["targets"] if target not in allowed]
        if removed:
            trimmed.append({"facet": point["facet"], "targets_removed": removed, "point_dropped": not allowed})
        if not allowed:
            continue
        point["targets"] = allowed
        for target in allowed:
            counts[target] += 1
        kept.append(point)
    return kept, trimmed


def rebuild_coverage(points: list[dict[str, Any]], model_coverage: Any,
                     scope: list[str]) -> list[dict[str, Any]]:
    """covered from the kept points; not_applicable only where the model gave a reason."""
    covered = {(target, point["facet"]) for point in points for target in point["targets"]}
    reasons: dict[tuple[str, str], str] = {}
    for entry in model_coverage if isinstance(model_coverage, list) else []:
        if (Draft202012Validator(COVERAGE_ITEM_SCHEMA).is_valid(entry)
                and entry["status"] == "not_applicable" and (entry["reason"] or "").strip()):
            reasons.setdefault((entry["target"], entry["facet"]), entry["reason"])
    rows = []
    for target in scope:
        for facet in FACETS:
            if (target, facet) in covered:
                rows.append({"target": target, "facet": facet, "status": "covered", "reason": None})
            elif (target, facet) in reasons:
                rows.append({"target": target, "facet": facet, "status": "not_applicable",
                             "reason": reasons[(target, facet)]})
            else:
                rows.append({"target": target, "facet": facet, "status": "uncovered", "reason": None})
    return rows


def finalize_batch(payload: Any, report: BatchReport, *, scope: list[str],
                   max_per_target: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """After the last attempt: drop invalid points, merge duplicates, trim, rebuild coverage."""
    if report.structure:
        coverage = [{"target": t, "facet": f, "status": "uncovered", "reason": None}
                    for t in scope for f in FACETS]
        return [], coverage, {"status": "failed", "errors": report.structure}
    dropped = [{"index": index, "errors": errors} for index, errors in sorted(report.points.items())]
    points = [copy.deepcopy(point) for index, point in enumerate(payload["checkpoints"])
              if index not in report.points]
    points, merged = merge_duplicates(points)
    points, trimmed = enforce_limit(points, max_per_target)
    coverage = rebuild_coverage(points, payload.get("coverage"), scope)
    clean = not report.messages()
    return points, coverage, {"status": "valid" if clean else "degraded", "dropped": dropped,
                              "merged": merged, "trimmed": trimmed, "unresolved": report.batch}


def _rank(point: dict[str, Any]) -> tuple[int, int]:
    new_behavior = any(target.startswith("O") for target in point["targets"])
    if point.get("probe"):
        # Open points decided by a verified probe are the most important references.
        return (-1 if new_behavior else 4), -len(point["targets"])
    if point["resolution"] == "open":
        tier = 0 if new_behavior else 4
    elif point["resolution"] in {"analog", "contract"}:
        tier = 1 if new_behavior else 4
    elif point["resolution"] == "convention":
        tier = 2 if new_behavior else 4
    else:
        tier = 3 if new_behavior else 5
    return tier, -len(point["targets"])


def _line(point: dict[str, Any]) -> str:
    head = (f"[{point['resolution']}] {','.join(point['targets'])}/{point['facet']}: "
            f"given {point['given']}; when {point['when']}; then {point['then']}")
    if point["resolution"] == "open":
        head += " Candidate readings: " + " | ".join(point["alternatives"])
        if point["favored"]:
            head += f" (favored: {point['favored']}; basis: {point['basis_behavior']})"
    else:
        head += f" Excludes: {point['rules_out']}"
        if point["resolution"] in INFERRED:
            head += f" (reference, {point['resolution']}: {point['basis_behavior']})"
    return head


def delivery_views(payload: dict[str, Any], max_solver_points: int) -> dict[str, Any]:
    """Solver sees the ranked decision points; reviewers get every check point."""
    ranked = sorted(payload["checkpoints"], key=_rank)
    solver = ranked[:max_solver_points]
    return {
        "solver_points": solver,
        "solver_text": "\n".join(f"- {_line(point)}" for point in solver),
        "review_targets": ranked,
        "dropped_from_solver_view": len(ranked) - len(solver),
    }


def summarize(payload: dict[str, Any], ledger: dict[str, Any]) -> dict[str, Any]:
    points = payload["checkpoints"]

    def tally(key: str) -> dict[str, int]:
        values = sorted({point[key] for point in points})
        return {value: sum(1 for point in points if point[key] == value) for value in values}

    return {
        "targets": len(ledger["obligations"]) + len(ledger["preservation_obligations"]),
        "checkpoints": len(points), "by_resolution": tally("resolution"), "by_facet": tally("facet"),
        "multi_target_checkpoints": sum(1 for point in points if len(point["targets"]) > 1),
        "coverage_cells": len(payload["coverage"]),
        "covered_cells": sum(1 for entry in payload["coverage"] if entry["status"] == "covered"),
        "uncovered_cells": sum(1 for entry in payload["coverage"] if entry["status"] == "uncovered"),
    }


class CheckpointClient:
    """Tool-free requests with the FT model parameters, one per target batch; each batch gets
    one corrective retry, then its invalid points are dropped (v6.6.4)."""

    PREFIX = "v6.6/checkpoints"

    def __init__(self, *, model: str, base_url: str, api_key: str, max_per_target: int,
                 max_attempts: int, timeout_sec: float, waits: tuple[float, ...] = RETRY_WAITS_SEC,
                 sleep=time.sleep, opener=urlopen, max_solver_points: int = 12,
                 batch_max_targets: int = 8) -> None:
        self.model, self.base_url, self.api_key = model, base_url.rstrip("/"), api_key
        self.max_per_target, self.max_attempts = max_per_target, max_attempts
        self.max_solver_points = max_solver_points
        self.batch_max_targets = batch_max_targets
        self.timeout_sec = timeout_sec
        self._waits, self._sleep, self._opener = waits, sleep, opener
        self.transport: list[dict[str, Any]] = []

    async def generate(self, *, packet: dict[str, Any], ledger: dict[str, Any],
                       artifacts: ArtifactStore) -> dict[str, Any]:
        return await asyncio.to_thread(self._generate, packet, ledger, artifacts)

    def _post(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        request = {"model": self.model, "messages": messages, "thinking": {"type": "enabled"},
                   "reasoning_effort": "high", "max_tokens": 65536, "seed": 42, "stream": False}
        http = Request(self.base_url + "/chat/completions",
                       data=json.dumps(request, ensure_ascii=False).encode(),
                       headers={"Content-Type": "application/json",
                                "Authorization": "Bearer " + self.api_key})
        # Transport retry only; a delivered response is never resampled.
        with open_with_retry(http, timeout=self.timeout_sec, record=self.transport.append,
                             waits=self._waits, sleep=self._sleep, opener=self._opener) as response:
            return json.loads(response.read().decode())

    def _run_batch(self, packet: dict[str, Any], ledger: dict[str, Any], artifacts: ArtifactStore,
                   scope: list[str], index: int, total: int,
                   result: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        messages = [{"role": "system", "content": checkpoint_system_prompt(self.max_per_target)},
                    {"role": "user", "content": checkpoint_user(packet, ledger, (scope, index, total))}]
        record: dict[str, Any] = {"batch": index, "targets": scope, "input_sha256": sha256_json(messages),
                                  "attempts": []}
        payload: Any = None
        report = BatchReport(structure=["no attempt"])
        for attempt in range(1, self.max_attempts + 1):
            prefix = f"{self.PREFIX}/batch-{index:02d}/attempt-{attempt}"
            artifacts.write_json(f"{prefix}/messages.json", messages)
            response = self._post(messages)
            artifacts.write_json(f"{prefix}/response.json", response)
            choice = response["choices"][0]
            content = choice["message"].get("content") or ""
            artifacts.write_text(f"{prefix}/reasoning.txt", choice["message"].get("reasoning_content") or "")
            artifacts.write_text(f"{prefix}/output.txt", content)
            try:
                payload = strict_json(content)
                report = check_batch(payload, ledger=ledger, packet=packet,
                                     max_per_target=self.max_per_target, batch=scope)
            except ValueError as error:
                payload, report = None, BatchReport(structure=[f"json: {error}"])
            errors = report.messages()
            artifacts.write_json(f"{prefix}/validation.json", {"errors": errors})
            summary = {"batch": index, "attempt": attempt, "finish_reason": choice.get("finish_reason"),
                       "usage": response.get("usage") or {}, "errors": len(errors)}
            record["attempts"].append(summary)
            result["attempts"].append(summary)
            if not errors:
                break
            messages = [*messages, {"role": "assistant", "content": content},
                        {"role": "user", "content": "The output violates the contract:\n- "
                         + "\n- ".join(errors[:40])
                         + "\nReturn the complete corrected JSON object only."}]
        points, coverage, outcome = finalize_batch(payload, report, scope=scope,
                                                   max_per_target=self.max_per_target)
        record.update(outcome, kept=len(points))
        artifacts.write_json(f"{self.PREFIX}/batch-{index:02d}/final.json",
                             {**record, "checkpoints": points, "coverage": coverage})
        return points, coverage, record

    def _generate(self, packet: dict[str, Any], ledger: dict[str, Any],
                  artifacts: ArtifactStore) -> dict[str, Any]:
        if (artifacts.root / self.PREFIX / "result.json").exists():
            raise RuntimeError("Check points already requested; use a new run instead of resampling")
        batches = target_batches(ledger, self.batch_max_targets)
        result: dict[str, Any] = {"status": "running", "attempts": [], "batches": [],
                                  "batch_max_targets": self.batch_max_targets,
                                  "started_at": datetime.now(timezone.utc).isoformat()}
        started = time.monotonic()
        points: list[dict[str, Any]] = []
        coverage: list[dict[str, Any]] = []
        try:
            for index, scope in enumerate(batches, 1):
                batch_points, batch_coverage, record = self._run_batch(
                    packet, ledger, artifacts, scope, index, len(batches), result)
                points += batch_points
                coverage += batch_coverage
                result["batches"].append(record)
            # Near-identical rules from different batches become one point with both targets.
            points, merged = merge_duplicates(points)
            points, trimmed = enforce_limit(points, self.max_per_target)
            result.update(cross_batch_merged=merged, cross_batch_trimmed=trimmed)
            result["status"] = "passed" if points else "invalid_output"
        except HTTPError as error:
            result.update(status="request_error", http_status=error.code)
        except Exception as error:  # noqa: BLE001 - record the outcome, never resample silently
            result.update(status="request_error", error_type=type(error).__name__,
                          error=artifacts.redact(str(error)))
        result.update(finished_at=datetime.now(timezone.utc).isoformat(),
                      elapsed_seconds=round(time.monotonic() - started, 3), transport=self.transport,
                      degraded_batches=[r["batch"] for r in result["batches"] if r["status"] == "degraded"],
                      failed_batches=[r["batch"] for r in result["batches"] if r["status"] == "failed"])
        if result["status"] == "passed":
            payload = {"checkpoints": points, "coverage": coverage}
            artifacts.write_json(f"{self.PREFIX}/checkpoints.json", payload)
            result["summary"] = summarize(payload, ledger)
            result["summary"].update(
                batches=len(batches), dropped_points=sum(len(r.get("dropped", [])) for r in result["batches"]),
                failed_batches=len(result["failed_batches"]), degraded_batches=len(result["degraded_batches"]))
            views = delivery_views(payload, self.max_solver_points)
            artifacts.write_text(f"{self.PREFIX}/solver-view.md", views["solver_text"] + "\n")
            artifacts.write_json(f"{self.PREFIX}/review-targets.json", views["review_targets"])
            result["summary"]["solver_view_points"] = len(views["solver_points"])
        artifacts.write_json(f"{self.PREFIX}/result.json", result)
        return result
