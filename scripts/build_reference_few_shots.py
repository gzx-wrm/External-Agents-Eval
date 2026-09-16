#!/usr/bin/env python3
"""Build the explicitly authorized, reference-assisted development few-shots.

Only the two permanently excluded development tasks are supported. Reference
materials are used offline to audit the labels; their contents and identifiers
are not embedded in the examples shown to Reviewer roles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from external_agents_eval.workflows.solve_review_repair.artifacts import sha256_bytes
from external_agents_eval.workflows.solve_review_repair.config import DEFAULT_SCHEMA_DIR
from external_agents_eval.workflows.solve_review_repair.validators import ContractValidator

ROOT = Path(__file__).resolve().parents[1]
SOURCE_JOB = (
    ROOT
    / "results/deepswe/deepswe-v1.1-pi-random60-seed20260901-tools7-w5-cleanup"
)

TASKS: dict[str, dict[str, Any]] = {
    "abs-stepped-slices": {
        "base_commit": "cb1b3b671d0ee9fa9da9f7b02f86967953ffd10a",
        "repository": "abs",
        "title": "Extend indexing ranges so arrays and strings support a third slice component:",
        "instruction": ROOT / ".benchmarks/deep-swe/tasks/abs-stepped-slices/instruction.md",
        "p0": SOURCE_JOB / "abs-stepped-slices__7bahsX7/artifacts/model.patch",
        "solution": ROOT
        / ".benchmarks/deep-swe/tasks/abs-stepped-slices/solution/solution.patch",
        "test_patch": ROOT / ".benchmarks/deep-swe/tasks/abs-stepped-slices/tests/test.patch",
        "source_sha256": {
            "instruction": "0381637972fc77c6c27bbc9f999af79ce57e9fc78d9ac43cf1ced51c9f8ea3dd",
            "p0": "8c5340752da74639063271e8bca5e8db0e1aa5ca9419ab98c1d4a366b07038eb",
            "solution": "3006ff58e945289a5defb1fb314ba3bcbb8a2c7df278f4678b760e53073fb82c",
            "test_patch": "bf9eab2e681531d83c458e7ad45e4a2fecf622c0ccaee6a9581d32875c45ecc9",
        },
    },
    "effect-sse-httpapi-streaming": {
        "base_commit": "9245bc59ebfa688e8c92dd691296ee69d0815e59",
        "repository": "effect",
        "title": "The HttpApi framework should support endpoints that produce typed event streams via SSE.",
        "instruction": ROOT
        / ".benchmarks/deep-swe/tasks/effect-sse-httpapi-streaming/instruction.md",
        "p0": SOURCE_JOB / "effect-sse-httpapi-streaming__KKoJkiV/artifacts/model.patch",
        "solution": ROOT
        / ".benchmarks/deep-swe/tasks/effect-sse-httpapi-streaming/solution/solution.patch",
        "test_patch": ROOT
        / ".benchmarks/deep-swe/tasks/effect-sse-httpapi-streaming/tests/test.patch",
        "source_sha256": {
            "instruction": "b9fc8403f53c0aed7e53cf7c2ca3f5df4249de5af8ac5ffc1dfc8500d21ce205",
            "p0": "988f22a5d858643fbb1426abde9749af82606b0461d90b942ff15d4edcf39901",
            "solution": "38260088472ebcccefc84ae784dcefcf977beb3b403dd80b0d556a21a3b245c9",
            "test_patch": "ce032707d6f1bfefb225d3b888dfac2991a50e4bcdcaa6cae0bef3f3518f5599",
        },
    },
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _origin(kind: str, evidence: str, rationale: str) -> dict[str, str]:
    return {"type": kind, "evidence": evidence, "rationale": rationale}


def _obligation(
    obligation_id: str,
    statement: str,
    *,
    origin: dict[str, str],
    category: str,
    criteria: list[str],
    surfaces: list[str],
    dependencies: list[str] | None = None,
    criticality: str = "must",
    confidence: str = "high",
) -> dict[str, Any]:
    return {
        "id": obligation_id,
        "statement": statement,
        "origins": [origin],
        "category": category,
        "criticality": criticality,
        "acceptance_criteria": criteria,
        "affected_surfaces": surfaces,
        "dependencies": dependencies or [],
        "confidence": confidence,
    }


def _implementation(
    path: str, symbol: str, line_start: int, line_end: int, explanation: str
) -> dict[str, Any]:
    return {
        "path": path,
        "symbol": symbol,
        "line_start": line_start,
        "line_end": line_end,
        "explanation": explanation,
    }


def _assessment(
    obligation_id: str,
    status: str,
    evidence: list[dict[str, Any]],
    *,
    validation_note: str | None,
    gap: str | None = None,
    confidence: str = "high",
) -> dict[str, Any]:
    return {
        "obligation_id": obligation_id,
        "status": status,
        "implementation_evidence": evidence,
        "test_evidence": [],
        "validation_note": validation_note,
        "gap": gap,
        "confidence": confidence,
    }


def _abs_ledger() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "task_id": "abs-stepped-slices",
        "feature_summary": (
            "Add parser, read, and assignment support for two- and three-part slices "
            "over arrays and Unicode strings without changing existing index behavior."
        ),
        "scope": {
            "in_scope": [
                "Index-range parsing and AST rendering",
                "Forward and reverse slicing for arrays and strings",
                "Array and string index/range assignment",
                "Unicode rune correctness and exact diagnostics",
            ],
            "out_of_scope": ["Syntax outside index brackets", "Hash indexing behavior"],
            "assumptions": [
                "Slice ends remain exclusive and omitted bounds follow the established range semantics."
            ],
        },
        "obligations": [
            _obligation(
                "O01",
                "Parse start:end:step with any permitted omitted component and preserve each omission in AST stringification.",
                origin=_origin(
                    "explicit_issue",
                    "Accept omitted components for stepped slices and preserve stepped ranges in AST stringification.",
                    "The syntax and its observable AST rendering are explicit requirements.",
                ),
                category="public_api",
                criteria=[
                    "start:end:step, :end:step, start::step, and ::step all parse.",
                    "Rendering does not replace an omitted start or end with a numeric literal.",
                    "Negative step expressions retain their expression form.",
                ],
                surfaces=["ast.IndexExpression", "parser.parseIndexExpression"],
            ),
            _obligation(
                "O02",
                "Read array slices with positive or negative steps and omitted bounds while retaining single-index and two-part behavior.",
                origin=_origin(
                    "explicit_issue",
                    "New stepped range behavior must work in both directions for arrays.",
                    "Array read semantics are a primary feature surface.",
                ),
                category="core_behavior",
                criteria=[
                    "Positive steps select forward indexes and negative steps select reverse indexes.",
                    "Bounds are normalized safely and zero step is rejected.",
                    "Existing array index and two-part ranges remain compatible.",
                ],
                surfaces=["evaluator.evalArrayIndexExpression", "evaluator.evalSliceIndices"],
                dependencies=["O01"],
            ),
            _obligation(
                "O03",
                "Index and slice strings by Unicode runes with the same stepped-bound semantics as arrays.",
                origin=_origin(
                    "explicit_issue",
                    "String indexing and range slicing must operate on Unicode characters (runes), not raw bytes.",
                    "Rune correctness is explicit for single indexes and both range forms.",
                ),
                category="core_behavior",
                criteria=[
                    "Single indexing returns one rune rather than one byte.",
                    "Two- and three-part string ranges select runes in both directions.",
                ],
                surfaces=["evaluator.evalStringIndexExpression", "evaluator.evalSliceIndices"],
                dependencies=["O01"],
            ),
            _obligation(
                "O04",
                "Assign array ranges using identical index selection, exact-length array replacement, or scalar broadcast.",
                origin=_origin(
                    "explicit_issue",
                    "Support assigning to two- and three-part array ranges with exact array length or scalar broadcast.",
                    "Assignment behavior and mismatch handling are explicitly specified.",
                ),
                category="data_flow",
                criteria=[
                    "Array replacements exactly match the number of selected indexes.",
                    "Non-array values broadcast to every selected index.",
                    "Empty selections enforce the documented size rules.",
                ],
                surfaces=["evaluator.evalArrayRangeAssignment"],
                dependencies=["O02"],
            ),
            _obligation(
                "O05",
                "Support rune-safe string single-index and range assignment with exact replacement and one-rune broadcast rules.",
                origin=_origin(
                    "explicit_issue",
                    "String replacement lengths are measured in runes and range broadcast applies only to non-empty selections.",
                    "The issue defines distinct single-index and range replacement contracts.",
                ),
                category="data_flow",
                criteria=[
                    "Single-index replacement accepts exactly one rune.",
                    "Range replacement accepts exact rune count or a one-rune broadcast for non-empty targets.",
                    "A non-empty replacement for an empty selection reports a size mismatch.",
                ],
                surfaces=["evaluator.evalIndexAssignment", "evaluator.evalStringRangeAssignment"],
                dependencies=["O03"],
            ),
            _obligation(
                "O06",
                "Preserve the required error families and existing non-stepped behavior across reads and assignments.",
                origin=_origin(
                    "explicit_issue",
                    "Keep compatibility with current error style and evaluator behavior.",
                    "Exact diagnostics and backward compatibility are explicit acceptance constraints.",
                ),
                category="backward_compatibility",
                criteria=[
                    "Zero step, nonnumeric bounds, and nonnumeric start use their specified messages.",
                    "Assignment type and size errors match the specified strings.",
                    "Single indexes and two-part ranges retain prior results.",
                ],
                surfaces=["evaluator.evalIndexAssignment", "evaluator.evalSliceIndices"],
                dependencies=["O02", "O03", "O04", "O05"],
            ),
        ],
    }


def _abs_assessment(patch_sha: str, ledger_sha: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "task_id": "abs-stepped-slices",
        "patch_sha256": patch_sha,
        "ledger_sha256": ledger_sha,
        "obligation_assessments": [
            _assessment(
                "O01",
                "partial",
                [
                    _implementation(
                        "ast/ast.go",
                        "IndexExpression.String",
                        572,
                        613,
                        "The AST stores and renders an optional Step and renders nil range components as empty text.",
                    ),
                    _implementation(
                        "parser/parser.go",
                        "parseIndexExpression",
                        1004,
                        1079,
                        "The parser accepts a second colon and optional step, but materializes some omitted starts as a NumberLiteral(0).",
                    ),
                ],
                validation_note="The stepped forms are parsed, but omission is not represented consistently.",
                gap=(
                    "For :end:step, parser lines 1017-1018 replace the omitted start with 0, so AST "
                    "stringification emits [0:end:step] instead of preserving [:end:step]."
                ),
            ),
            _assessment(
                "O02",
                "covered",
                [
                    _implementation(
                        "evaluator/evaluator.go",
                        "evalArrayIndexExpression",
                        1536,
                        1574,
                        "Array ranges delegate to one index-selection routine and preserve single-index behavior.",
                    ),
                    _implementation(
                        "evaluator/evaluator.go",
                        "evalSliceIndices",
                        1581,
                        1670,
                        "The shared routine implements forward/reverse defaults, normalization, and zero-step rejection.",
                    ),
                ],
                validation_note="Source inspection covers positive, negative, omitted-bound, and legacy paths.",
            ),
            _assessment(
                "O03",
                "covered",
                [
                    _implementation(
                        "evaluator/evaluator.go",
                        "evalStringIndexExpression",
                        1493,
                        1534,
                        "The string path converts to []rune before single-index or range selection.",
                    )
                ],
                validation_note="Both read paths operate on rune indexes and use the shared slice selection.",
            ),
            _assessment(
                "O04",
                "covered",
                [
                    _implementation(
                        "evaluator/evaluator.go",
                        "evalArrayRangeAssignment",
                        528,
                        568,
                        "Array assignment uses shared indexes, exact array-length checks, and scalar broadcast.",
                    )
                ],
                validation_note="The selected target count controls exact replacement and mismatch behavior.",
            ),
            _assessment(
                "O05",
                "covered",
                [
                    _implementation(
                        "evaluator/evaluator.go",
                        "evalIndexAssignment",
                        463,
                        525,
                        "Single string assignment is rune-based and enforces a one-rune replacement.",
                    ),
                    _implementation(
                        "evaluator/evaluator.go",
                        "evalStringRangeAssignment",
                        571,
                        620,
                        "Range assignment implements exact rune replacement, conditional broadcast, and empty-target mismatch.",
                    ),
                ],
                validation_note="The single and range branches enforce their distinct rune-count contracts.",
            ),
            _assessment(
                "O06",
                "covered",
                [
                    _implementation(
                        "evaluator/evaluator.go",
                        "evalIndexAssignment",
                        438,
                        525,
                        "Assignment branches retain legacy index behavior and emit the requested type errors.",
                    ),
                    _implementation(
                        "evaluator/evaluator.go",
                        "evalSliceIndices",
                        1581,
                        1670,
                        "Range validation emits the requested zero-step and numeric-range diagnostics.",
                    ),
                ],
                validation_note="The compatibility and exact-error branches are explicit in the implementation.",
            ),
        ],
        "unexpected_regressions": [],
        "late_obligation_candidates": [],
        "verdict": "request_changes",
        "verdict_rationale": (
            "Runtime slicing and assignment obligations are covered, but a must-level observable parser/AST "
            "contract still rewrites an omitted start as zero."
        ),
        "repair_requests": [
            {
                "priority": 1,
                "finding_refs": ["O01"],
                "problem": "Omitted starts in :end:step are not preserved by AST stringification.",
                "requested_outcome": (
                    "Represent the start as omitted while retaining the runtime default, so :end:step renders "
                    "without an inserted zero and all other stepped forms remain unchanged."
                ),
                "suggested_validation": [
                    "Parse myArray[:5:2] and assert that String() returns (myArray[:5:2]).",
                    "Re-run parser and evaluator range tests for omitted bounds and negative steps.",
                ],
            }
        ],
    }


def _effect_ledger() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "task_id": "effect-sse-httpapi-streaming",
        "feature_summary": (
            "Add typed SSE endpoint declaration, server streaming, schema-aware codecs, client streaming, "
            "and OpenAPI integration across the HttpApi platform."
        ),
        "scope": {
            "in_scope": [
                "Endpoint and schema APIs for SSE",
                "Builder handler registration and stream context",
                "SSE wire codecs and response/stream adapters",
                "Typed client and OpenAPI behavior",
            ],
            "out_of_scope": ["Non-HttpApi streaming protocols", "Transport reconnection policy"],
            "assumptions": [
                "New public constructors follow existing platform conventions for direct response values and default success status."
            ],
        },
        "obligations": [
            _obligation(
                "O01",
                "Expose HttpApiEndpoint.sse and isSSE so only endpoint construction marks SSE behavior.",
                origin=_origin(
                    "explicit_issue",
                    "HttpApiEndpoint provides an sse constructor and isSSE guard; only sse() marks an endpoint as SSE.",
                    "Endpoint identity must be distinct from schema annotations.",
                ),
                category="public_api",
                criteria=[
                    "sse constructs a GET endpoint marked as SSE.",
                    "isSSE is true only for endpoint-level SSE identity.",
                    "Applying schema withSSE alone does not mark an endpoint.",
                ],
                surfaces=["HttpApiEndpoint.sse", "HttpApiEndpoint.isSSE"],
            ),
            _obligation(
                "O02",
                "Expose HttpApiSchema.withSSE and getSSE over schema AST annotations.",
                origin=_origin(
                    "explicit_issue",
                    "HttpApiSchema provides withSSE and getSSE (operates on AST nodes).",
                    "The schema annotation API is explicitly required.",
                ),
                category="public_api",
                criteria=["withSSE annotates schemas and getSSE reads the AST annotation."],
                surfaces=["HttpApiSchema.withSSE", "HttpApiSchema.getSSE"],
            ),
            _obligation(
                "O03",
                "Register direct stream handlers and auto-detect Stream values returned by ordinary handlers on SSE endpoints.",
                origin=_origin(
                    "explicit_issue",
                    "Handlers provide handleStream, and a Stream returned from handle on an SSE endpoint is auto-detected.",
                    "Both registration paths must reach the same SSE response conversion.",
                ),
                category="integration",
                criteria=[
                    "handleStream accepts a handler returning Stream directly.",
                    "handle preserves raw responses and plain success values.",
                    "Stream values are converted only through the SSE path.",
                ],
                surfaces=["HttpApiBuilder.Handlers", "HttpApiBuilder.handlerToRoute"],
                dependencies=["O01"],
            ),
            _obligation(
                "O04",
                "Capture the current Effect context and provide it to the stream so services remain available during consumption.",
                origin=_origin(
                    "explicit_issue",
                    "Capture the current Effect context and provide it to the stream before building the response.",
                    "Streaming outlives handler evaluation and must retain services.",
                ),
                category="data_flow",
                criteria=["The stream receives the active handler context before it is placed in the response."],
                surfaces=["HttpApiBuilder.handlerToRoute", "HttpApiSSE.toResponse"],
                dependencies=["O03"],
            ),
            _obligation(
                "O05",
                "Make toResponse a direct HttpServerResponse constructor with the normal success status and SSE headers.",
                origin=_origin(
                    "repository_contract",
                    "The public API lists toResponse(stream, encoder) alongside direct platform response constructors.",
                    "Callers must receive a response value with the standard status field rather than a deferred wrapper.",
                ),
                category="public_api",
                criteria=[
                    "toResponse returns HttpServerResponse directly.",
                    "The response exposes the normal default success status.",
                    "Content-Type is text/event-stream with no-cache and keep-alive headers.",
                ],
                surfaces=["HttpApiSSE.toResponse"],
                dependencies=["O04"],
            ),
            _obligation(
                "O06",
                "Format SSEMessage fields and multiline data into valid SSE wire records, with JSON data convenience formatting.",
                origin=_origin(
                    "explicit_issue",
                    "formatMessage returns SSE wire format with multi-line data support and formatDataMessage JSON-encodes any value.",
                    "Wire formatting is an explicit module contract.",
                ),
                category="serialization",
                criteria=[
                    "event, id, retry, and each data line are formatted correctly.",
                    "Every message terminates with a blank line.",
                    "formatDataMessage JSON-encodes its input.",
                ],
                surfaces=["HttpApiSSE.SSEMessage", "HttpApiSSE.formatMessage"],
            ),
            _obligation(
                "O07",
                "Encode single schemas and tagged unions, deriving event from _tag through wrapped, transformed, or suspended members.",
                origin=_origin(
                    "explicit_issue",
                    "Tagged union success schemas set event from _tag and wrapped, transformed, or suspended union members are supported.",
                    "Union dispatch and wrapper traversal are explicit requirements.",
                ),
                category="serialization",
                criteria=[
                    "makeEventEncoder produces Effect<string>.",
                    "makeUnionEventEncoder emits _tag as event for tagged unions.",
                    "Non-union schemas fall back to data-only formatting.",
                ],
                surfaces=["HttpApiSSE.makeEventEncoder", "HttpApiSSE.makeUnionEventEncoder"],
                dependencies=["O06"],
            ),
            _obligation(
                "O08",
                "Decode JSON data and union SSE messages into typed values with a non-union fallback.",
                origin=_origin(
                    "explicit_issue",
                    "makeEventDecoder and makeUnionEventDecoder decode typed values via Effect with non-union fallback.",
                    "Typed client consumption depends on symmetric decoding.",
                ),
                category="serialization",
                criteria=[
                    "Invalid JSON fails through the schema parse channel.",
                    "Union event names select the matching member decoder.",
                    "Non-unions ignore event dispatch and decode data directly.",
                ],
                surfaces=["HttpApiSSE.makeEventDecoder", "HttpApiSSE.makeUnionEventDecoder"],
                dependencies=["O06"],
            ),
            _obligation(
                "O09",
                "Convert typed streams to SSE bytes and parse response chunks across message boundaries without losing fields.",
                origin=_origin(
                    "explicit_issue",
                    "fromStream, toResponse, and toStream are required; toStream buffers partial chunks across double-newline boundaries.",
                    "Streaming adapters must preserve framing under arbitrary chunking.",
                ),
                category="data_flow",
                criteria=[
                    "fromStream emits encoded Uint8Array chunks.",
                    "toStream buffers partial frames and parses event, id, retry, and multiline data.",
                    "Decoder failures propagate through the stream error channel.",
                ],
                surfaces=["HttpApiSSE.fromStream", "HttpApiSSE.toStream"],
                dependencies=["O06", "O08"],
            ),
            _obligation(
                "O10",
                "Return typed Stream values from SSE clients while validating status before starting stream decoding.",
                origin=_origin(
                    "explicit_issue",
                    "SSE endpoints return a Stream and the client validates response status before streaming.",
                    "Outer Effect failures must remain distinguishable from stream decoding failures.",
                ),
                category="integration",
                criteria=[
                    "Successful SSE endpoints return a typed Stream.",
                    "Error status responses fail the outer Effect before their bodies are parsed as SSE.",
                    "Regular endpoints retain existing decoding behavior.",
                ],
                surfaces=["HttpApiClient.makeClient"],
                dependencies=["O01", "O08", "O09"],
            ),
            _obligation(
                "O11",
                "Describe SSE endpoint successes in OpenAPI using text/event-stream and the event schema.",
                origin=_origin(
                    "explicit_issue",
                    "SSE endpoints use text/event-stream content type with schema referencing the event type.",
                    "The generated contract must match runtime negotiation.",
                ),
                category="documentation",
                criteria=[
                    "SSE success content uses text/event-stream.",
                    "The generated schema still references the declared success event type.",
                    "Regular endpoints keep their existing content types and parameters.",
                ],
                surfaces=["OpenApi.fromApi"],
                dependencies=["O01"],
            ),
            _obligation(
                "O12",
                "Export HttpApiSSE through the platform barrel without regressing raw responses, plain values, or regular endpoints.",
                origin=_origin(
                    "compatibility_inferred",
                    "The feature is a new public HttpApi module that must coexist with existing handler and endpoint behavior.",
                    "Discoverability and compatibility are required for usable integration.",
                ),
                category="backward_compatibility",
                criteria=[
                    "HttpApiSSE is exported from the platform index.",
                    "Raw HttpServerResponse and plain success values retain prior handling.",
                    "Non-SSE client and OpenAPI paths remain unchanged.",
                ],
                surfaces=["platform/index", "HttpApiBuilder.handlerToRoute", "HttpApiClient.makeClient"],
                dependencies=["O03", "O10", "O11"],
            ),
        ],
    }


def _effect_assessment(patch_sha: str, ledger_sha: str) -> dict[str, Any]:
    items = [
        _assessment(
            "O01",
            "covered",
            [_implementation("packages/platform/src/HttpApiEndpoint.ts", "sse / isSSE", 1031, 1046, "Endpoint-level SSE identity, guard, and GET constructor are explicit.")],
            validation_note="The endpoint flag, rather than success-schema annotation, drives endpoint identity.",
        ),
        _assessment(
            "O02",
            "covered",
            [_implementation("packages/platform/src/HttpApiSchema.ts", "getSSE / withSSE", 155, 162, "The AST annotation accessor and schema combinator are both exported.")],
            validation_note="The annotation API operates on AST metadata and remains separate from endpoint identity.",
        ),
        _assessment(
            "O03",
            "covered",
            [
                _implementation("packages/platform/src/HttpApiBuilder.ts", "Handlers.handleStream", 259, 280, "The public handler interface accepts direct Stream handlers."),
                _implementation("packages/platform/src/HttpApiBuilder.ts", "HandlersProto.handleStream", 445, 460, "Registration records the stream-handler variant."),
                _implementation("packages/platform/src/HttpApiBuilder.ts", "handlerToRoute", 750, 760, "Raw responses pass through, Stream values are detected, and plain values use encodeSuccess."),
            ],
            validation_note="All three server result variants have distinct, reachable branches.",
        ),
        _assessment(
            "O04",
            "covered",
            [
                _implementation("packages/platform/src/HttpApiBuilder.ts", "handlerToRoute", 723, 757, "The handler route obtains the active fiber context before converting Stream values."),
                _implementation("packages/platform/src/HttpApiSSE.ts", "toResponse", 264, 278, "The stream is provided the captured context before response construction."),
            ],
            validation_note="Service context is explicitly attached to the stream before it is consumed.",
        ),
        _assessment(
            "O05",
            "partial",
            [_implementation("packages/platform/src/HttpApiSSE.ts", "toResponse", 264, 278, "Headers are correct, but the function returns Effect<HttpServerResponse> instead of a response value.")],
            validation_note="Wire headers are present, but the exported constructor shape is inconsistent.",
            gap=(
                "toResponse wraps response construction in Effect.map, so direct callers receive an Effect and cannot "
                "observe the normal response status field. Capture/provide context in the handler path and keep "
                "toResponse as a direct HttpServerResponse constructor."
            ),
        ),
        _assessment(
            "O06",
            "covered",
            [_implementation("packages/platform/src/HttpApiSSE.ts", "SSEMessage / formatMessage", 19, 73, "The message schema and formatter handle optional fields, multiline data, JSON values, and record termination.")],
            validation_note="Every defined wire field is serialized and data-only convenience formatting delegates consistently.",
        ),
        _assessment(
            "O07",
            "covered",
            [
                _implementation("packages/platform/src/HttpApiSSE.ts", "makeEventEncoder", 120, 125, "Single schemas encode to JSON SSE data messages through Effect."),
                _implementation("packages/platform/src/HttpApiSSE.ts", "makeUnionEventEncoder", 127, 173, "Union tags select member encoders and non-unions use the data-only fallback."),
            ],
            validation_note="Schema AST property extraction supports wrapped members and the runtime _tag selects the encoder.",
        ),
        _assessment(
            "O08",
            "covered",
            [
                _implementation("packages/platform/src/HttpApiSSE.ts", "makeEventDecoder", 179, 207, "JSON parsing is composed with schema decoding."),
                _implementation("packages/platform/src/HttpApiSSE.ts", "makeUnionEventDecoder", 219, 238, "Event names dispatch union members with a non-union/simple fallback."),
            ],
            validation_note="Invalid JSON and schema failures stay in the ParseError channel.",
        ),
        _assessment(
            "O09",
            "covered",
            [
                _implementation("packages/platform/src/HttpApiSSE.ts", "fromStream", 246, 250, "Encoded records become Uint8Array stream chunks."),
                _implementation("packages/platform/src/HttpApiSSE.ts", "toStream", 289, 311, "The parser normalizes line endings, accumulates partial chunks, splits complete records, and decodes them.")
            ],
            validation_note="Chunk boundaries are independent of message boundaries and trailing buffered text is flushed.",
        ),
        _assessment(
            "O10",
            "covered",
            [_implementation("packages/platform/src/HttpApiClient.ts", "makeClient endpointFn", 171, 263, "Status decoding is constructed first and awaited before the SSE response is converted to a typed Stream.")],
            validation_note="The SSE branch validates response status before stream decoding and leaves regular decoding intact.",
        ),
        _assessment(
            "O11",
            "covered",
            [_implementation("packages/platform/src/OpenApi.ts", "processResponseMap", 338, 360, "SSE success maps switch only the content type while processing the declared success AST.")],
            validation_note="The response schema remains derived from the event AST and regular maps keep their encoding content type.",
        ),
        _assessment(
            "O12",
            "covered",
            [
                _implementation("packages/platform/src/index.ts", "HttpApiSSE export", 87, 89, "The new module is exported from the platform barrel."),
                _implementation("packages/platform/src/HttpApiBuilder.ts", "handlerToRoute", 750, 760, "Raw responses and plain values retain their established branches."),
                _implementation("packages/platform/src/HttpApiClient.ts", "makeClient endpointFn", 247, 263, "Only SSE endpoints take the stream branch; regular endpoint response transforms remain unchanged."),
            ],
            validation_note="The new behavior is guarded by SSE identity and existing alternatives remain reachable.",
        ),
    ]
    return {
        "schema_version": "1.0",
        "task_id": "effect-sse-httpapi-streaming",
        "patch_sha256": patch_sha,
        "ledger_sha256": ledger_sha,
        "obligation_assessments": items,
        "unexpected_regressions": [],
        "late_obligation_candidates": [],
        "verdict": "request_changes",
        "verdict_rationale": (
            "The end-to-end SSE feature is otherwise complete, but the public toResponse helper exposes a deferred "
            "Effect instead of the direct response value expected from the platform constructor contract."
        ),
        "repair_requests": [
            {
                "priority": 1,
                "finding_refs": ["O04", "O05"],
                "problem": "toResponse has the wrong public return shape because context capture is embedded in the helper.",
                "requested_outcome": (
                    "Make toResponse return HttpServerResponse directly with default success status and SSE headers; "
                    "capture and provide the active context at the handler conversion site before calling it."
                ),
                "suggested_validation": [
                    "Construct a response with toResponse and inspect its status and SSE headers without running an Effect.",
                    "Stream an event that reads a service from the handler context.",
                    "Re-run regular builder and OpenAPI regression tests.",
                ],
            }
        ],
    }


def _build_input(task_id: str, meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "task_id": task_id,
        "issue": {
            "title": meta["title"],
            "body": meta["instruction"].read_text(encoding="utf-8"),
        },
        "repository": {
            "name": meta["repository"],
            "base_commit": meta["base_commit"],
            "worktree": "/tmp/external_eval/worktrees/spec-review",
            "state": "base_only",
        },
        "policy": {
            "max_obligations": 32,
            "allowed_origin_types": [
                "explicit_issue",
                "compatibility_inferred",
                "repository_contract",
            ],
            "forbid_patch_access": True,
            "forbid_hidden_evidence": True,
        },
    }


def _assessment_input(
    task_id: str, meta: dict[str, Any], patch_sha: str, ledger_sha: str
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "task_id": task_id,
        "issue": {
            "title": meta["title"],
            "body": meta["instruction"].read_text(encoding="utf-8"),
        },
        "repository": {
            "name": meta["repository"],
            "base_commit": meta["base_commit"],
            "worktree": "/tmp/external_eval/worktrees/patch-review",
            "state": "base_plus_p0",
        },
        "patch": {"path": "/tmp/external_eval/contracts/p0.patch", "sha256": patch_sha},
        "obligation_ledger": {
            "path": "/tmp/external_eval/contracts/review_obligations.json",
            "sha256": ledger_sha,
        },
        "public_test_results": [],
    }


def build(output: Path) -> None:
    validator = ContractValidator(DEFAULT_SCHEMA_DIR)
    for task_id, meta in TASKS.items():
        for label in ("instruction", "p0", "solution", "test_patch"):
            actual = _digest(meta[label])
            expected = meta["source_sha256"][label]
            if actual != expected:
                raise ValueError(f"{task_id} {label} digest changed: {actual} != {expected}")

    provenance = {
        "schema_version": "1.0",
        "mode": "reference_assisted_development_only",
        "development_task_ids": list(TASKS),
        "user_authorized": True,
        "authorization_date": "2026-09-04",
        "reference_material_scope": list(TASKS),
        "prompt_exposure": (
            "Reference materials were used only to audit labels. Their paths, test names, output, "
            "and patch contents are not embedded in the eight Reviewer example artifacts."
        ),
        "source_sha256": {
            task_id: dict(meta["source_sha256"]) for task_id, meta in TASKS.items()
        },
    }
    _write_json(output / "provenance.json", provenance)

    ledgers = {
        "abs-stepped-slices": _abs_ledger(),
        "effect-sse-httpapi-streaming": _effect_ledger(),
    }
    assessment_factories = {
        "abs-stepped-slices": _abs_assessment,
        "effect-sse-httpapi-streaming": _effect_assessment,
    }
    for task_id, meta in TASKS.items():
        task_dir = output / task_id
        build_input = _build_input(task_id, meta)
        ledger = ledgers[task_id]
        validator.validate_schema("build_input", build_input)
        validator.validate_ledger(ledger, task_id=task_id)
        _write_json(task_dir / "stage_a_input.json", build_input)
        ledger_path = task_dir / "stage_a_output.json"
        _write_json(ledger_path, ledger)
        ledger_sha = sha256_bytes(ledger_path.read_bytes())
        patch_sha = meta["source_sha256"]["p0"]
        assess_input = _assessment_input(task_id, meta, patch_sha, ledger_sha)
        assessment = assessment_factories[task_id](patch_sha, ledger_sha)
        validator.validate_schema("assessment_input", assess_input)
        validator.validate_assessment(
            assessment,
            task_id=task_id,
            patch_sha256=patch_sha,
            ledger_sha256=ledger_sha,
            ledger=ledger,
        )
        _write_json(task_dir / "stage_b_input_summary.json", assess_input)
        _write_json(task_dir / "stage_b_output.json", assessment)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "few_shots/reference_assisted_v1",
    )
    args = parser.parse_args()
    build(args.output.expanduser().resolve())
    print(args.output.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
