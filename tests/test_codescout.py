from __future__ import annotations

import importlib.util
import io
import json
import tarfile
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("tree_sitter")
for grammar in ("python", "javascript", "typescript", "go", "rust"):
    pytest.importorskip("tree_sitter_" + grammar)

from external_agents_eval.preprocessors.codescout.index import build_index
from external_agents_eval.preprocessors.codescout.pipeline import (
    ChatClient,
    Config,
    Preprocessor,
    parse_analysis,
    parse_targets,
)


@pytest.fixture
def repository(tmp_path):
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg/a.py").write_text(
        "from .base import Base\nclass C(Base):\n @deco\n def run(self, x):\n  return x\n")
    (root / "README.md").write_text("Public documentation.\n")
    return root


@pytest.mark.parametrize(("path", "code", "name"), [
    ("a.py", "class C:\n @deco\n def run(self): return 1\n", "C.run"),
    ("a.ts", "export const run = (x: number): number => x;", "run"),
    ("a.tsx", "export const View = () => <div />;", "View"),
    ("a.js", "const obj = {run(x) {return x}};", "obj.run"),
    ("a.go", "package p\nfunc (p *Parser) Run(s string) int { return 1 }", "Parser.Run"),
    ("a.rs", "struct Engine; impl Engine { pub fn run(&self) {} }", "Engine.run"),
])
def test_multilanguage_entity_retrieval_keeps_complete_source(tmp_path, path, code, name):
    (tmp_path / path).write_text(code)
    index = build_index(tmp_path)
    retrieved = index.retrieve("function", name)
    assert retrieved["status"] == "retrieved"
    assert len(retrieved["matches"]) == 1
    match = retrieved["matches"][0]
    assert match["file_sha256"] == index.files[path].sha256
    assert match["text"].strip() in code
    if path.endswith(".py"):
        assert "@deco" in match["text"]


def test_import_inheritance_scope_and_all_eligible_paths_are_recorded(repository):
    index = build_index(repository)
    assert {edge["relation"] for edge in index.edges} >= {"imports", "inherits", "contains"}
    outline, coverage = index.outline()
    assert all(path in outline for path in index.files)
    assert "C.run" in outline
    assert coverage["complete_file_tree"]
    assert not coverage["omitted_entities"]
    assert any(edge["target"] == "module:.base" for edge in index.edges)


def test_go_struct_interface_and_field_scopes(tmp_path):
    (tmp_path / "a.go").write_text(
        "package p\ntype Parser struct { Value string }; type Runner interface { Run() error }")
    index = build_index(tmp_path)
    assert index.retrieve("class", "Parser")["status"] == "retrieved"
    assert index.retrieve("class", "Runner")["status"] == "retrieved"
    assert index.retrieve("function", "Runner.Run")["status"] == "retrieved"
    assert any(entity.qualified_name == "Parser.Value" for entity in index.entities)


def test_large_file_ranges_match_frozen_source_bytes(tmp_path):
    code = "\n".join(f"def function_{i}():\n    return {i}\n" for i in range(1200))
    (tmp_path / "large.py").write_text(code)
    index = build_index(tmp_path)
    for entity in index.entities:
        assert entity.line_start == code.encode()[:entity.byte_start].count(b"\n") + 1
        assert entity.line_end == code.encode()[:entity.byte_end].count(b"\n") + 1
    assert index.retrieve("function", "function_1199")["matches"][0]["line_start"] == 3598


def test_multibyte_signature_boundary_does_not_change_retrieved_source(tmp_path):
    prefix = 'value = "'
    code = prefix + "x" * (239 - len(prefix.encode())) + "汉字" * 30 + '"\n'
    # Reproduces the former failure: the 240-byte preview ends inside 汉.
    with pytest.raises(UnicodeDecodeError):
        code.encode()[:240].decode("utf-8")
    (tmp_path / "unicode.py").write_text(code)
    index = build_index(tmp_path)
    entity = next(entity for entity in index.entities if entity.name == "value")
    assert entity.signature == code.encode()[:239].decode("utf-8")
    retrieved = index.retrieve("file", "unicode.py")
    assert retrieved["status"] == "retrieved"
    assert retrieved["matches"][0]["text"] == code
    assert retrieved["matches"][0]["file_sha256"] == index.files["unicode.py"].sha256


def test_public_qualified_names_and_ambiguous_methods(repository):
    (repository / "b.py").write_text("class D:\n def run(self): return 2\n")
    index = build_index(repository)
    assert index.retrieve("function", "public.pkg.C.run")["status"] == "retrieved"
    ambiguous = index.retrieve("function", "run")
    assert ambiguous["ambiguous"]
    assert len(ambiguous["matches"]) == 2
    assert {record["path"] for record in ambiguous["matches"]} == {"pkg/a.py", "b.py"}
    exact = index.retrieve("function", "pkg.a.C.run")
    assert len(exact["matches"]) == 1


def test_source_is_frozen_and_budget_failure_is_explicit(repository):
    index = build_index(repository)
    original = index.retrieve("file", "/app/pkg/a.py")
    (repository / "pkg/a.py").write_text("CHANGED_AFTER_INDEX")
    assert index.retrieve("file", "pkg/a.py") == original
    limited = index.retrieve("file", "pkg/a.py", max_chars=320)
    assert limited["status"] == "unavailable"
    assert limited["matches"][0]["status"] == "source_budget_exceeded"
    assert limited["text"] == ""


def test_no_history_dependencies_or_outside_symlink_reads(repository, tmp_path):
    (repository / ".git").mkdir()
    (repository / ".git/config").write_text("HISTORY_SECRET")
    (repository / "node_modules").mkdir()
    (repository / "node_modules/package.js").write_text("DEPENDENCY")
    (tmp_path / "outside.py").write_text("OUTSIDE")
    (repository / "escape.py").symlink_to(tmp_path / "outside.py")
    index = build_index(repository)
    assert set(index.files) == {"pkg/a.py", "README.md"}
    assert index.retrieve("file", "../outside.py")["status"] == "invalid_path"
    assert index.retrieve("file", "escape.py")["status"] == "not_found"
    assert len(index.skipped) == 3


def test_parser_errors_and_non_code_files_remain_visible(repository):
    (repository / "broken.ts").write_text("export function broken( {")
    (repository / "image.bin").write_bytes(b"\x00\xff")
    index = build_index(repository)
    assert index.files["broken.ts"].parse_errors
    assert index.files["image.bin"].unavailable == "non_utf8_or_binary"
    assert "broken.ts" in index.outline()[0]


def test_prompt_formats_and_json_are_both_parsed():
    targets = parse_targets(
        "1. target_type:file,\ntarget_name:pkg/a.py,\nreasoning:entry point\n"
        "2. target_type:function, target_name:C.run, reasoning:public behavior")
    assert [target.target_name for target in targets] == ["pkg/a.py", "C.run"]
    assert targets[0].reasoning == "entry point"
    assert parse_targets(json.dumps([vars(target) for target in targets])) == targets
    with pytest.raises(ValueError):
        parse_targets("No structured targets")
    with pytest.raises(ValueError):
        parse_analysis("RELEVANCE: 99/10\nROLE_IN_ISSUE: bad")


def analysis(score=8, label="GOOD"):
    return (f"RELEVANCE: {score}/10\nROLE_IN_ISSUE: {label}\nFIX_LOCATION_HINTS: change entry\n"
            "EXPLORATION_INSIGHTS: preserve baseline\nALTERNATIVE_HYPOTHESES: inspect caller")


def test_analysis_accepts_space_and_underscore_field_headings():
    text = analysis(2).replace("ALTERNATIVE_HYPOTHESES", "ALTERNATIVE HYPOTHESES")
    parsed = parse_analysis(text)
    assert parsed["relevance"] == 2
    assert parsed["alternative_hypotheses"] == "inspect caller"
    assert "ALTERNATIVE HYPOTHESES" not in parsed["exploration_insights"]


SYNTHESIS = "\n".join(
    "## " + heading + "\n" + content for heading, content in [
        ("Issue Description", "Support the requested public behavior."),
        ("Reproduction Steps", "Call the existing public interface."),
        ("Expected Behavior", "Return the requested result."),
        ("Exploration Hints", "Inspect pkg/a.py and C.run."),
        ("Fix Hints", "Preserve the existing default behavior."),
    ])


class ScriptedClient:
    backend = "offline-scripted-model"

    def __init__(self, texts):
        self.texts, self.calls = list(texts), []

    def __call__(self, messages, timeout):
        self.calls.append(messages)
        assert len(messages) == 1
        assert timeout > 0
        return {"choices": [{"message": {"content": self.texts.pop(0),
                                         "reasoning_content": "Raw reasoning is preserved."},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10}}


@pytest.fixture
def model():
    targets = [
        {"target_type": kind, "target_name": name, "reasoning": "Relevant to the issue"}
        for kind, name in [("file", "pkg/a.py"), ("class", "C"), ("function", "C.run"),
                           ("file", "missing.py"), ("file", "README.md")]
    ]
    return ScriptedClient([json.dumps(targets), analysis(), analysis(2, "LOW_RELEVANCE_MARKER"),
                           analysis(), analysis(4), SYNTHESIS])


def test_complete_protocol_filters_and_reuses_calls(repository, tmp_path, model):
    index, output = build_index(repository), tmp_path / "output"
    preprocessor = Preprocessor(Config(), model, output)
    assert preprocessor.run("ORIGINAL_ISSUE", index) == SYNTHESIS
    result = json.loads((output / "result.json").read_text())
    assert result["status"] == "completed"
    assert (result["targets"], result["analyzed"], result["retained"]) == (5, 4, 3)
    assert result["model_calls_in_protocol"] == 6
    assert result["prompt_tokens"] == 600
    assert result["completion_tokens"] == 60
    assert result["retrievals"][3]["status"] == "not_found"
    assert "LOW_RELEVANCE_MARKER" not in model.calls[-1][0]["content"]
    assert "ORIGINAL_ISSUE" in model.calls[-1][0]["content"]
    assert json.loads((output / "scope/response.json").read_text())["choices"][0][
        "message"]["reasoning_content"]
    # Full rerun against the same immutable inputs consumes no new model calls.
    assert preprocessor.run("ORIGINAL_ISSUE", index) == SYNTHESIS
    assert len(model.calls) == 6
    assert json.loads((output / "result.json").read_text())["new_request_attempts"] == 0
    with pytest.raises(ValueError, match="changed"):
        Preprocessor(replace(Config(), relevance_threshold=5), model, output).run("ORIGINAL_ISSUE", index)
    with pytest.raises(ValueError, match="changed"):
        preprocessor.run("DIFFERENT_ISSUE", index)


def test_invalid_response_records_usage_and_does_not_repeat(repository, tmp_path):
    output = tmp_path / "invalid"
    model = ScriptedClient(["unstructured output"])
    processor = Preprocessor(Config(), model, output)
    with pytest.raises(ValueError):
        processor.run("issue", build_index(repository))
    result = json.loads((output / "result.json").read_text())
    assert result["status"] == "failed" and result["failed_stage"] == "scope"
    assert result["prompt_tokens"] == 100
    with pytest.raises(ValueError):
        processor.run("issue", build_index(repository))
    assert len(model.calls) == 1


def test_request_without_response_does_not_silently_resample(repository, tmp_path):
    def interrupted(messages, timeout):
        raise TimeoutError("Interrupted request")

    output = tmp_path / "interrupted"
    processor = Preprocessor(Config(), interrupted, output)
    with pytest.raises(TimeoutError):
        processor.run("issue", build_index(repository))
    with pytest.raises(RuntimeError, match="Previous request"):
        processor.run("issue", build_index(repository))


def test_http_client_is_tool_free_and_does_not_include_key_in_json(monkeypatch):
    captured = {}

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def urlopen(request, timeout):
        captured.update(body=json.loads(request.data), headers=dict(request.header_items()))
        return Response(b'{"choices": []}')

    monkeypatch.setattr("external_agents_eval.preprocessors.codescout.pipeline.urlopen", urlopen)
    ChatClient(Config(), "TEST_SECRET")([{"role": "user", "content": "issue"}], 20)
    assert "tools" not in captured["body"] and "tool_choice" not in captured["body"]
    assert captured["body"]["thinking"] == {"type": "enabled"}
    assert "TEST_SECRET" not in json.dumps(captured["body"])
    assert captured["headers"]["Authorization"] == "Bearer TEST_SECRET"


def test_deepswe_adapter_only_changes_instruction_and_rejects_snapshot_escape(tmp_path):
    path = Path(__file__).parents[1] / "scripts/deepswe_codescout/run.py"
    spec = importlib.util.spec_from_file_location("codescout_deepswe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    task = {"task_id": "x", "instruction": "original", "image": "pinned", "base_commit": "abc"}
    augmented = module.augment_task(task, SYNTHESIS)
    assert task["instruction"] == "original"
    assert augmented == {**task, "instruction": SYNTHESIS}
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        entry = tarfile.TarInfo("../escape")
        entry.size = 4
        archive.addfile(entry, io.BytesIO(b"oops"))
    with pytest.raises(ValueError, match="Unexpected entry"):
        module.extract_snapshot(stream.getvalue(), tmp_path / "snapshot")
    assert not (tmp_path / "escape").exists()
