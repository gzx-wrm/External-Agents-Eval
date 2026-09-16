from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from referencing import Registry, Resource

from . import contracts_v2, contracts_v3

_JSON_FENCE = re.compile(
    r"```[ \t]*(?:json)?[ \t]*\r?\n(?P<body>.*?)\r?\n```",
    flags=re.IGNORECASE | re.DOTALL,
)
_JSON_OBJECT_START = re.compile(r"\{")
JsonNormalizationMethod = Literal["fenced_json", "embedded_json_object"]


def extract_single_json_object(
    raw_text: str,
) -> tuple[str, JsonNormalizationMethod | None]:
    """Extract one unambiguous JSON object using conservative fallbacks.

    The extraction order is exact JSON, one valid fenced JSON object, then one
    valid embedded top-level JSON object. Embedded extraction uses
    ``JSONDecoder.raw_decode`` rather than a greedy brace regex so nested
    objects and braces in JSON strings are handled correctly. Ambiguous or
    malformed responses remain untouched for the regular correction path.
    """
    stripped = raw_text.strip()
    try:
        exact = json.loads(stripped)
    except json.JSONDecodeError:
        pass
    else:
        return (stripped, None) if isinstance(exact, dict) else (raw_text, None)

    candidates: list[str] = []
    for match in _JSON_FENCE.finditer(raw_text):
        candidate = match.group("body").strip()
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            candidates.append(candidate)
    if len(candidates) == 1:
        return candidates[0], "fenced_json"

    decoder = json.JSONDecoder()
    embedded: list[tuple[int, int, str]] = []
    for match in _JSON_OBJECT_START.finditer(raw_text):
        start = match.start()
        try:
            payload, end = decoder.raw_decode(raw_text, start)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            embedded.append((start, end, raw_text[start:end]))

    # A complete outer object also exposes parseable nested objects. Keep only
    # candidates not contained by another valid candidate before checking
    # uniqueness across the response.
    top_level = [
        candidate
        for candidate in embedded
        if not any(
            other_start <= candidate[0]
            and candidate[1] <= other_end
            and (other_start, other_end) != candidate[:2]
            for other_start, other_end, _ in embedded
        )
    ]
    if len(top_level) == 1:
        start, end, candidate = top_level[0]
        outside = raw_text[:start] + raw_text[end:]
        if "{" not in outside and "}" not in outside:
            return candidate, "embedded_json_object"
    return raw_text, None


def unwrap_single_fenced_json_object(raw_text: str) -> tuple[str, bool]:
    """Backward-compatible wrapper for the original boolean API."""
    extracted, method = extract_single_json_object(raw_text)
    return extracted, method is not None


@dataclass
class ContractValidationError(ValueError):
    kind: str
    errors: tuple[str, ...]

    def __str__(self) -> str:
        return f"{self.kind} validation failed: " + "; ".join(self.errors)

    def as_json(self) -> list[dict[str, str]]:
        return [{"kind": self.kind, "message": error} for error in self.errors]


class ContractValidator:
    SCHEMAS: ClassVar[dict[str, str]] = {
        "build_input": "build_obligations_input.schema.json",
        "ledger": "review_obligations.schema.json",
        "assessment_input": "assess_patch_input.schema.json",
        "assessment": "patch_assessment.schema.json",
        "repair_input": "repair_input.schema.json",
    }

    SCHEMA_BUNDLES: ClassVar[dict[str, dict[str, str]]] = {
        "1.0": SCHEMAS,
        "2.0": {
            "source_policy": "source_policy_v1.schema.json",
            "build_input": "build_obligations_input_v2.schema.json",
            "ledger": "review_obligations_v2.schema.json",
            "assessment_input": "assess_patch_input_v2.schema.json",
            "assessment": "patch_assessment_v2.schema.json",
            "effective_review_set": "effective_review_set_v2.schema.json",
            "repair_input": "repair_input_v2.schema.json",
            "repair_audit": "repair_audit_v2.schema.json",
        },
        "3.0": {
            "build_input": "build_obligations_input_v3.schema.json",
            "ledger": "review_obligations_v3.schema.json",
            "source_bindings": "source_bindings_v3.schema.json",
            "assessment_input": "assess_patch_input_v3.schema.json",
            "assessment": "patch_assessment_v3.schema.json",
            "effective_review_set": "effective_review_set_v3.schema.json",
            "repair_input": "repair_input_v3.schema.json",
            "repair_audit": "repair_audit_v3.schema.json",
        },
    }

    def __init__(self, schema_dir: str | Path, *, contract_version: str = "1.0") -> None:
        self.schema_dir = Path(schema_dir)
        self.contract_version = contract_version
        try:
            self.SCHEMAS = dict(self.SCHEMA_BUNDLES[contract_version])
        except KeyError as error:
            raise ValueError(f"Unsupported contract version: {contract_version}") from error
        self._schemas: dict[str, dict[str, Any]] = {}
        self._validators: dict[str, Draft202012Validator] = {}
        for name, filename in self.SCHEMAS.items():
            schema = json.loads((self.schema_dir / filename).read_text(encoding="utf-8"))
            try:
                Draft202012Validator.check_schema(schema)
            except SchemaError as error:
                raise ValueError(f"Invalid bundled schema {filename}: {error.message}") from error
            self._schemas[name] = schema
        registry = Registry().with_resources(
            (schema["$id"], Resource.from_contents(schema)) for schema in self._schemas.values()
        )
        for name, schema in self._schemas.items():
            self._validators[name] = Draft202012Validator(schema, registry=registry)

    def schema(self, name: str) -> dict[str, Any]:
        return self._schemas[name]

    def parse(self, raw_text: str) -> dict[str, Any]:
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as error:
            raise ContractValidationError(
                "schema",
                (f"JSON parse error at line {error.lineno}, column {error.colno}: {error.msg}",),
            ) from error
        if not isinstance(payload, dict):
            raise ContractValidationError("schema", ("The final response must be one JSON object",))
        return payload

    def validate_schema(self, name: str, payload: dict[str, Any]) -> None:
        errors = sorted(
            self._validators[name].iter_errors(payload), key=lambda item: list(item.path)
        )
        if errors:
            messages = tuple(
                f"{_json_path(error.absolute_path)}: {error.message}" for error in errors
            )
            raise ContractValidationError("schema", messages)

    def validate_ledger(
        self,
        payload: dict[str, Any],
        *,
        task_id: str,
        source_policy_sha256: str | None = None,
    ) -> None:
        self.validate_schema("ledger", payload)
        if self.contract_version == "2.0":
            errors = contracts_v2.validate_ledger(
                payload,
                task_id=task_id,
                source_policy_sha256=source_policy_sha256,
            )
            errors.extend(_forbidden_evidence_errors(payload))
            if errors:
                raise ContractValidationError("semantic", tuple(dict.fromkeys(errors)))
            return
        if self.contract_version == "3.0":
            errors = contracts_v3.validate_ledger(payload, task_id=task_id)
            errors.extend(_forbidden_evidence_errors(payload))
            if errors:
                raise ContractValidationError("semantic", tuple(dict.fromkeys(errors)))
            return
        errors: list[str] = []
        if payload["task_id"] != task_id:
            errors.append(f"task_id {payload['task_id']!r} does not match {task_id!r}")
        obligations = payload["obligations"]
        ids = [item["id"] for item in obligations]
        if len(ids) != len(set(ids)):
            errors.append("obligation IDs must be unique")
        known = set(ids)
        graph: dict[str, tuple[str, ...]] = {}
        for obligation in obligations:
            obligation_id = obligation["id"]
            dependencies = tuple(obligation["dependencies"])
            graph[obligation_id] = dependencies
            for dependency in dependencies:
                if dependency not in known:
                    errors.append(f"{obligation_id} has dangling dependency {dependency}")
                if dependency == obligation_id:
                    errors.append(f"{obligation_id} cannot depend on itself")
        cycle = _find_cycle(graph)
        if cycle:
            errors.append("dependency graph contains cycle: " + " -> ".join(cycle))
        errors.extend(_forbidden_evidence_errors(payload))
        if errors:
            raise ContractValidationError("semantic", tuple(dict.fromkeys(errors)))

    def validate_assessment(
        self,
        payload: dict[str, Any],
        *,
        task_id: str,
        patch_sha256: str,
        ledger_sha256: str,
        ledger: dict[str, Any],
    ) -> None:
        self.validate_schema("assessment", payload)
        if self.contract_version == "2.0":
            errors, protocol_errors = contracts_v2.validate_assessment(
                payload,
                task_id=task_id,
                patch_sha256=patch_sha256,
                ledger_sha256=ledger_sha256,
                ledger=ledger,
            )
            errors.extend(_forbidden_evidence_errors(payload))
            if protocol_errors:
                raise ContractValidationError("protocol", tuple(dict.fromkeys(protocol_errors)))
            if errors:
                raise ContractValidationError("semantic", tuple(dict.fromkeys(errors)))
            return
        if self.contract_version == "3.0":
            errors, protocol_errors = contracts_v3.validate_assessment(
                payload,
                task_id=task_id,
                patch_sha256=patch_sha256,
                ledger_sha256=ledger_sha256,
                ledger=ledger,
            )
            errors.extend(_forbidden_evidence_errors(payload))
            if protocol_errors:
                raise ContractValidationError("protocol", tuple(dict.fromkeys(protocol_errors)))
            if errors:
                raise ContractValidationError("semantic", tuple(dict.fromkeys(errors)))
            return
        errors: list[str] = []
        protocol_errors: list[str] = []
        expected = [item["id"] for item in ledger["obligations"]]
        criticality = {item["id"]: item["criticality"] for item in ledger["obligations"]}
        actual = [item["obligation_id"] for item in payload["obligation_assessments"]]
        if len(actual) != len(set(actual)):
            errors.append("each obligation must be assessed exactly once; duplicates found")
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        if missing:
            errors.append("missing obligation assessments: " + ", ".join(missing))
        if extra:
            errors.append("unknown obligation assessments: " + ", ".join(extra))
        for assessment in payload["obligation_assessments"]:
            obligation_id = assessment["obligation_id"]
            status = assessment["status"]
            if status == "covered":
                if not assessment["implementation_evidence"]:
                    errors.append(f"{obligation_id} is covered without implementation evidence")
                if not assessment["test_evidence"] and not _nonempty(assessment["validation_note"]):
                    errors.append(
                        f"{obligation_id} is covered without test evidence or validation_note"
                    )
                if _nonempty(assessment["gap"]):
                    errors.append(f"{obligation_id} is covered but has a gap")
            elif not _nonempty(assessment["gap"]):
                errors.append(f"{obligation_id} status {status} requires a non-empty gap")
            for evidence in assessment["implementation_evidence"]:
                start, end = evidence["line_start"], evidence["line_end"]
                if (start is None) != (end is None):
                    errors.append(
                        f"{obligation_id} evidence line_start and line_end must both be set"
                    )
                elif start is not None and end < start:
                    errors.append(f"{obligation_id} evidence line_end precedes line_start")
        if payload["task_id"] != task_id:
            protocol_errors.append("assessment task_id does not match Controller input")
        if payload["patch_sha256"] != patch_sha256:
            protocol_errors.append("assessment patch_sha256 does not match P0")
        if payload["ledger_sha256"] != ledger_sha256:
            protocol_errors.append("assessment ledger_sha256 does not match frozen ledger")
        blocking_regression = any(item["blocking"] for item in payload["unexpected_regressions"])
        must_gaps = [
            item["obligation_id"]
            for item in payload["obligation_assessments"]
            if criticality.get(item["obligation_id"]) == "must" and item["status"] != "covered"
        ]
        if payload["verdict"] == "approve" and must_gaps:
            errors.append(
                "approve conflicts with uncovered must obligations: " + ", ".join(must_gaps)
            )
        if payload["verdict"] == "approve" and blocking_regression:
            errors.append("approve conflicts with a blocking unexpected regression")
        known_findings = set(expected)
        known_findings.update(item["id"] for item in payload["unexpected_regressions"])
        known_findings.update(item["id"] for item in payload["late_obligation_candidates"])
        priorities = [item["priority"] for item in payload["repair_requests"]]
        if priorities != list(range(1, len(priorities) + 1)):
            errors.append(
                "repair request priorities must be unique, ordered, and continuous from 1"
            )
        for request in payload["repair_requests"]:
            for finding in request["finding_refs"]:
                if finding not in known_findings:
                    errors.append(f"repair request references unknown finding {finding}")
        if payload["verdict"] == "approve" and payload["repair_requests"]:
            errors.append("approve verdict cannot include repair requests")
        errors.extend(_forbidden_evidence_errors(payload))
        if protocol_errors:
            raise ContractValidationError("protocol", tuple(dict.fromkeys(protocol_errors)))
        if errors:
            raise ContractValidationError("semantic", tuple(dict.fromkeys(errors)))

    def validate_source_policy(self, payload: dict[str, Any], *, task_id: str) -> None:
        if self.contract_version != "2.0":
            raise ValueError("Source policy is only available in contract v2.0")
        self.validate_schema("source_policy", payload)
        errors: list[str] = []
        if payload["task_id"] != task_id:
            errors.append("source policy task_id does not match Controller input")
        errors.extend(_forbidden_evidence_errors(payload))
        if errors:
            raise ContractValidationError("semantic", tuple(dict.fromkeys(errors)))

    @staticmethod
    def validate_issue_source_anchors(
        sources: list[dict[str, Any]], *, title: str, body: str
    ) -> None:
        errors = contracts_v2.validate_issue_source_anchors(
            sources,
            title=title,
            body=body,
        )
        if errors:
            raise ContractValidationError("source", tuple(dict.fromkeys(errors)))

    @staticmethod
    def validate_task_source_anchors(
        sources: list[dict[str, Any]], *, title: str, description: str
    ) -> None:
        errors = contracts_v3.validate_task_source_anchors(
            sources,
            title=title,
            description=description,
        )
        if errors:
            raise ContractValidationError("source", tuple(dict.fromkeys(errors)))

    def build_effective_review_set(
        self,
        *,
        task_id: str,
        ledger_sha256: str,
        assessment_sha256: str,
        ledger: dict[str, Any],
        assessment: dict[str, Any],
    ) -> dict[str, Any]:
        if self.contract_version == "2.0":
            payload = contracts_v2.build_effective_review_set(
                task_id=task_id,
                ledger_sha256=ledger_sha256,
                assessment_sha256=assessment_sha256,
                ledger=ledger,
                assessment=assessment,
            )
        elif self.contract_version == "3.0":
            payload = contracts_v3.build_effective_review_set(
                task_id=task_id,
                ledger_sha256=ledger_sha256,
                assessment_sha256=assessment_sha256,
                ledger=ledger,
                assessment=assessment,
            )
        else:
            raise ValueError("Effective review sets require contract v2.0 or v3.0")
        self.validate_schema("effective_review_set", payload)
        return payload

    def build_request_contexts(
        self,
        *,
        ledger: dict[str, Any],
        assessment: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if self.contract_version != "3.0":
            raise ValueError("Expanded request contexts are available only in contract v3.0")
        return contracts_v3.build_request_contexts(ledger=ledger, assessment=assessment)

    def validate_source_bindings(self, payload: dict[str, Any]) -> None:
        if self.contract_version != "3.0":
            raise ValueError("Source bindings are available only in contract v3.0")
        self.validate_schema("source_bindings", payload)
        errors = _forbidden_evidence_errors(payload)
        if errors:
            raise ContractValidationError("semantic", tuple(dict.fromkeys(errors)))

    def validate_repair_audit(
        self,
        payload: dict[str, Any],
        *,
        task_id: str,
        p0_sha256: str,
        assessment_sha256: str,
        effective_review_set_sha256: str,
        ledger: dict[str, Any],
        assessment: dict[str, Any],
        effective_review_set: dict[str, Any],
        actual_changed_paths: set[str],
    ) -> None:
        if self.contract_version not in {"2.0", "3.0"}:
            raise ValueError("Repair audits require contract v2.0 or v3.0")
        self.validate_schema("repair_audit", payload)
        if self.contract_version == "2.0":
            errors, protocol_errors = contracts_v2.validate_repair_audit(
                payload,
                task_id=task_id,
                p0_sha256=p0_sha256,
                assessment_sha256=assessment_sha256,
                effective_review_set_sha256=effective_review_set_sha256,
                ledger=ledger,
                assessment=assessment,
                effective_review_set=effective_review_set,
                actual_changed_paths=actual_changed_paths,
            )
        else:
            errors, protocol_errors = contracts_v3.validate_repair_audit(
                payload,
                task_id=task_id,
                candidate_patch_sha256=p0_sha256,
                assessment_sha256=assessment_sha256,
                effective_review_set_sha256=effective_review_set_sha256,
                assessment=assessment,
                effective_review_set=effective_review_set,
                actual_changed_paths=actual_changed_paths,
            )
        errors.extend(_forbidden_evidence_errors(payload))
        if protocol_errors:
            raise ContractValidationError("protocol", tuple(dict.fromkeys(protocol_errors)))
        if errors:
            raise ContractValidationError("semantic", tuple(dict.fromkeys(errors)))


def _json_path(parts: Iterable[Any]) -> str:
    result = "$"
    for part in parts:
        result += f"[{part}]" if isinstance(part, int) else f".{part}"
    return result


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _find_cycle(graph: dict[str, tuple[str, ...]]) -> tuple[str, ...] | None:
    active: list[str] = []
    visited: set[str] = set()

    def visit(node: str) -> tuple[str, ...] | None:
        if node in active:
            start = active.index(node)
            return tuple(active[start:] + [node])
        if node in visited:
            return None
        active.append(node)
        for dependency in graph.get(node, ()):
            if dependency in graph:
                cycle = visit(dependency)
                if cycle:
                    return cycle
        active.pop()
        visited.add(node)
        return None

    for node in graph:
        cycle = visit(node)
        if cycle:
            return cycle
    return None


def _forbidden_evidence_errors(payload: Any) -> list[str]:
    forbidden = (
        "solution/solution.patch",
        "solution.patch",
        "reference patch",
        "reference solution",
        "hidden test",
        "verifier output",
        "verifier report",
        "/tests/test.patch",
        "tests/test.patch",
    )
    errors: list[str] = []

    def walk(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                walk(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")
        elif isinstance(value, str):
            lowered = value.lower()
            for marker in forbidden:
                if marker in lowered:
                    errors.append(f"{path} contains forbidden evidence marker {marker!r}")
                    break

    walk(payload, "$")
    return errors
