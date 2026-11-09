"""CodeScout's fixed scope/analyze/filter/synthesize controller, without tools."""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from urllib.request import Request, urlopen

from . import prompts
from .artifacts import ArtifactStore, sha256_json
from .index import RepositoryIndex, digest


@dataclass(frozen=True)
class Config:
    relevance_threshold: float = 4
    max_targets: int = 15
    max_outline_chars: int = 240_000
    max_source_chars: int = 240_000
    max_file_bytes: int = 4_000_000
    max_seconds: int = 3600
    request_timeout: int = 600
    model: str = "deepseek-flash"
    base_url: str = "https://api.deepseek.com/v1"
    reasoning_effort: str = "high"
    max_tokens: int = 65536

    def __post_init__(self):
        if not 0 <= self.relevance_threshold <= 10 or not 1 <= self.max_targets <= 15:
            raise ValueError("Invalid relevance threshold or target limit")
        for name in ("max_outline_chars", "max_source_chars", "max_file_bytes", "max_seconds",
                     "request_timeout", "max_tokens"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class Target:
    target_type: str
    target_name: str
    reasoning: str


FIELDS = ("ROLE_IN_ISSUE", "FIX_LOCATION_HINTS", "EXPLORATION_INSIGHTS", "ALTERNATIVE_HYPOTHESES")
HEADINGS = ("Issue Description", "Reproduction Steps", "Expected Behavior",
            "Exploration Hints", "Fix Hints")


def _json_output(text):
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    if stripped.startswith(("{", "[")):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return None
    return None


def parse_targets(text, max_targets=15):
    value = _json_output(text)
    if isinstance(value, dict):
        value = value.get("targets")
    if not isinstance(value, list):
        starts = list(re.finditer(r"\btarget_type\s*:\s*", text, re.IGNORECASE))
        value = []
        for number, match in enumerate(starts):
            segment = text[match.end():starts[number + 1].start() if number + 1 < len(starts)
                           else len(text)]
            fields = re.match(
                r"(file|class|function)\s*,?\s*target_name\s*:\s*(.*?)\s*,?\s*"
                r"reasoning\s*:\s*(.*)", segment, re.IGNORECASE | re.DOTALL,
            )
            if not fields:
                raise ValueError("Malformed CodeScout exploration target")
            kind, name, reason = fields.groups()
            value.append({"target_type": kind.lower(), "target_name": name.rstrip(", \n"),
                          "reasoning": re.sub(r"\s*\n\s*\d+[.)]\s*$", "", reason).strip()})
    if not value or len(value) > max_targets:
        raise ValueError(f"Expected 1..{max_targets} exploration targets")
    targets, seen = [], set()
    for record in value:
        kind = str(record["target_type"]).lower().strip()
        name, reason = str(record["target_name"]).strip(), str(record["reasoning"]).strip()
        if kind not in {"file", "class", "function"} or not name or not reason:
            raise ValueError("Invalid target type, name, or missing reasoning")
        if (kind, name) not in seen:
            targets.append(Target(kind, name, reason))
            seen.add((kind, name))
    return targets


def parse_analysis(text):
    value = _json_output(text)
    if isinstance(value, dict):
        value = {key.upper().replace(" ", "_"): content for key, content in value.items()}
        score = str(value.get("RELEVANCE", "")).removesuffix("/10").strip()
        result = {name: str(value.get(name, "")).strip() for name in FIELDS}
    else:
        match = re.search(r"\bRELEVANCE\s*:\s*(\d+(?:\.\d+)?)\s*/\s*10\b", text, re.IGNORECASE)
        if match is None:
            raise ValueError("Missing numeric RELEVANCE: X/10")
        score, result = match.group(1), {}
        for number, name in enumerate(FIELDS):
            label = r"[_ \t]+".join(name.split("_"))
            stop = "|".join(r"[_ \t]+".join(field.split("_"))
                            for field in FIELDS[number + 1:]) or r"(?!)"
            section = re.search(r"\b" + label + r"\s*:\s*(.*?)(?=\b(?:" + stop +
                                r")\s*:|\Z)", text, re.DOTALL | re.IGNORECASE)
            result[name] = section.group(1).strip() if section else ""
    relevance = float(score)
    if not 0 <= relevance <= 10 or not all(result.values()):
        raise ValueError("Analysis requires a 0..10 score and all four nonempty sections")
    return {"relevance": relevance, **{key.lower(): content for key, content in result.items()}}


def validate_synthesis(text):
    found = []
    for heading in HEADINGS:
        match = re.search(r"^#{1,6}\s+" + re.escape(heading) + r"\b[^\n]*", text, re.MULTILINE | re.IGNORECASE)
        if match is None:
            raise ValueError("Missing synthesized section: " + heading)
        found.append(match)
    if [match.start() for match in found] != sorted(match.start() for match in found):
        raise ValueError("Synthesized sections are out of order")
    for number, match in enumerate(found):
        end = found[number + 1].start() if number + 1 < len(found) else len(text)
        inline = re.sub(r"^#{1,6}\s+" + re.escape(HEADINGS[number]) + r"\s*:?\s*", "",
                        match.group(), flags=re.IGNORECASE)
        if not (inline.strip() or text[match.end():end].strip()):
            raise ValueError("Empty synthesized section: " + HEADINGS[number])


class ChatClient:
    """One independent, tool-free OpenAI-compatible request per paper stage.

    No hidden transport/format retry; a failed request is a recorded failed stage.
    Credentials are only in request headers, never in request artifacts.
    """

    backend = "openai-compatible-http"

    def __init__(self, config: Config, api_key: str):
        self.config, self.api_key = config, api_key

    def __call__(self, messages, timeout):
        request = {
            "model": self.config.model, "messages": messages, "stream": False,
            "reasoning_effort": self.config.reasoning_effort, "max_tokens": self.config.max_tokens,
            "thinking": {"type": "enabled"},
        }
        http = Request(
            self.config.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(request).encode(),
            headers={"Authorization": "Bearer " + self.api_key, "Content-Type": "application/json"},
        )
        with urlopen(http, timeout=timeout) as response:
            return json.loads(response.read())


class Preprocessor:
    def __init__(self, config: Config, client, output, *, secrets=()):
        self.config, self.client = config, client
        self.artifacts = ArtifactStore(output, secrets=secrets)

    def _call(self, stage, prompt, deadline):
        store = self.artifacts
        request = {"messages": [{"role": "user", "content": prompt}],
                   "config": asdict(self.config)}
        request_file = store.root / stage / "request.json"
        response_file = store.root / stage / "response.json"
        call = {"stage": stage, "cached": False, "usage": {}, "request_attempted": False}
        self.calls.append(call)
        if request_file.exists():
            if json.loads(request_file.read_text()) != request:
                raise ValueError("Stage input/config changed; use a new output directory")
            if not response_file.exists():
                raise RuntimeError("Previous request has no response; use a new run to retry explicitly")
            raw = json.loads(response_file.read_text())
            call["cached"] = True
        else:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("CodeScout preprocessing deadline exceeded")
            store.write_json(f"{stage}/request.json", request)
            call["request_attempted"] = True
            raw = self.client(request["messages"], min(self.config.request_timeout, remaining))
            store.write_json(f"{stage}/response.json", raw)
        call["usage"] = raw.get("usage", {})
        call["response_received"] = True
        choice = raw["choices"][0]
        text = choice["message"].get("content")
        if choice.get("finish_reason") != "stop" or not isinstance(text, str) or not text.strip():
            raise ValueError("Incomplete/non-text CodeScout model response")
        store.write_text(f"{stage}/output.txt", text)
        return text, raw.get("usage", {})

    def run(self, issue: str, index: RepositoryIndex):
        if not issue.strip() or not index.files:
            raise ValueError("CodeScout requires an issue and a nonempty repository")
        store, started = self.artifacts, time.monotonic()
        manifest = {
            "method": "codescout-reimplementation-v1", "paper": "2603.05744v2",
            "issue_sha256": digest(issue.encode()), "repository_sha256": index.fingerprint,
            "config": asdict(self.config),
            "prompts_sha256": sha256_json([prompts.SCOPE, prompts.ANALYZE, prompts.SYNTHESIZE]),
            "excluded_directories": index.excluded_dirs,
            "request_backend": getattr(self.client, "backend", "custom"),
            "parser_versions": index.parser_versions,
        }
        previous = store.root / "manifest.json"
        if previous.exists() and json.loads(previous.read_text()) != json.loads(json.dumps(manifest)):
            raise ValueError("Repository/issue/config changed; use a new output directory")
        store.write_json("manifest.json", manifest)
        store.write_text("original-issue.md", issue)
        store.write_json("repository-graph.json", index.graph())
        retrievals, analyses, self.calls = [], [], []
        status = {"status": "running", "model_calls_in_protocol": 0}
        stage = "index"
        try:
            outline, coverage = index.outline(self.config.max_outline_chars)
            store.write_text("repository-outline.txt", outline)
            store.write_json("index-coverage.json", coverage)
            deadline = started + self.config.max_seconds
            stage = "scope"
            text, _ = self._call(stage, prompts.SCOPE.format(
                problem_statement=issue, file_tree=outline), deadline)
            targets = parse_targets(text, self.config.max_targets)
            store.write_json("targets.json", [asdict(target) for target in targets])
            for number, target in enumerate(targets):
                stage = f"analysis/{number:02d}"
                retrieved = index.retrieve(target.target_type, target.target_name,
                                           max_chars=self.config.max_source_chars)
                store.write_json(f"{stage}/retrieval.json", retrieved)
                retrievals.append({"target": asdict(target), "status": retrieved["status"]})
                if retrieved["status"] != "retrieved":
                    continue
                text, _ = self._call(stage, prompts.ANALYZE.format(
                    problem_statement=issue, content=retrieved["text"],
                    target_type=target.target_type, target_name=target.target_name), deadline)
                analysis = {"target": asdict(target), **parse_analysis(text)}
                store.write_json(f"{stage}/analysis.json", analysis)
                analyses.append(analysis)
            retained = [a for a in analyses if a["relevance"] >= self.config.relevance_threshold]
            store.write_json("analyses.json", analyses)
            store.write_json("retained-analyses.json", retained)
            # Do not call an original-issue-only rewrite a successful CodeScout
            # analysis if the index never retrieved any selected target.
            if not analyses:
                raise ValueError("No exploration target was retrieved and analyzed")
            insights = []
            for number, analysis in enumerate(retained, 1):
                target = analysis["target"]
                lines = [f"{number}. TARGET: {target['target_type']} - {target['target_name']}",
                         f"RELEVANCE: {analysis['relevance']}/10"]
                lines.extend(f"{field.replace('_', ' ')}: {analysis[field.lower()]}" for field in FIELDS)
                insights.append("\n".join(lines))
            stage = "synthesis"
            text, _ = self._call(stage, prompts.SYNTHESIZE.format(
                original_ps=issue, insights="\n\n".join(insights) or "No analyses pass the threshold."),
                deadline)
            validate_synthesis(text)
            store.write_text("enhanced-issue.md", text)
            status.update(status="completed", targets=len(targets), analyzed=len(analyses),
                          retained=len(retained), enhanced_issue_sha256=digest(text.encode()),
                          index_coverage=coverage)
            return text
        except Exception as error:
            status.update(status="failed", failed_stage=stage, error_type=type(error).__name__,
                          error=store.redact(str(error)))
            raise
        finally:
            usage = [call["usage"] for call in self.calls if call.get("response_received")]
            new_usage = [call["usage"] for call in self.calls
                         if call.get("response_received") and not call["cached"]]
            status.update(elapsed_seconds=round(time.monotonic() - started, 3),
                          model_calls_in_protocol=len(usage), usage_by_call=usage,
                          new_request_attempts=sum(call["request_attempted"] for call in self.calls),
                          calls=self.calls,
                          retrievals=retrievals,
                          prompt_tokens=sum(item.get("prompt_tokens", 0) for item in usage),
                          completion_tokens=sum(item.get("completion_tokens", 0) for item in usage),
                          new_prompt_tokens=sum(item.get("prompt_tokens", 0) for item in new_usage),
                          new_completion_tokens=sum(item.get("completion_tokens", 0)
                                                    for item in new_usage))
            store.write_json("result.json", status)
