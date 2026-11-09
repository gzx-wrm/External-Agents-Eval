import importlib.util
import json
from pathlib import Path

from external_agents_eval.workflows.solve_review_repair import v68_review as rv
from external_agents_eval.workflows.solve_review_repair import v68_review_lint as rlint

DATA = {"obligations": [{"id": "O1", "statement": "retry calls"}],
        "entities": [{"id": "E1", "symbol": "retry", "path": "src/a.py", "line": 3}],
        "items": [{"obligation": "O1", "entity": "E1", "batch": 2, "given": "g1", "when": "w1", "then": "t1",
                   "alternatives": ["a", "b"], "observations": [{"text": "fact", "path": "src/a.py", "line": 2}]},
                  {"obligation": "O1", "entity": None, "batch": 1, "given": "g2", "when": "w2", "then": "t2",
                   "alternatives": ["c", "d"], "task_quote": "retried"}]}


def _write(tmp_path, ids, lines):
    (tmp_path / "scope.json").write_text(json.dumps({"ids": ids}))
    (tmp_path / "review.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src/a.py").write_text("a\nb\nc\n")
    return str(repo)


def test_review_lint_requires_every_id_and_evidence_for_defects(tmp_path):
    repo = _write(tmp_path, ["CP1", "CP2", "CP3"], [
        {"id": "CP1", "status": "defect", "candidate_behavior": "keeps tab", "expected": "one space", "severity": "high",
         "evidence": [{"kind": "code", "path": "src/a.py", "line": 9, "detail": "x"}]},
        {"id": "CP2", "status": "maybe", "candidate_behavior": "x"},
        {"id": "CP9", "status": "ok", "candidate_behavior": "x"}])
    ids, rows, parse = rlint.load(str(tmp_path))
    errors, stats = rlint.lint(ids, rows, repo)
    text = "\n".join(errors)
    assert parse == [] and "line must be an integer within src/a.py" in text and "status must be one of" in text
    assert "not a question id" in text and "1 question ids have no review line: CP3" in text


def test_review_lint_last_line_wins_and_passes(tmp_path):
    repo = _write(tmp_path, ["CP1", "CP2"], [
        {"id": "CP1", "status": "unsure", "candidate_behavior": "x"},
        {"id": "CP2", "status": "ok", "candidate_behavior": "y"},
        {"id": "CP1", "status": "defect", "candidate_behavior": "crashes", "expected": "returns []", "severity": "high",
         "evidence": [{"kind": "probe", "detail": "python -c ... -> TypeError"},
                      {"kind": "code", "path": "src/a.py", "line": 2, "detail": "b"}]}])
    ids, rows, _ = rlint.load(str(tmp_path))
    errors, stats = rlint.lint(ids, rows, repo)
    assert errors == [] and stats == {"in_scope": 2, "reviewed": 2, "defects": 1, "unsure": 0}
    assert [r["status"] for r in rlint.final_rows(ids, rows)] == ["defect", "ok"]


def test_questions_batches_and_report():
    assert rv.batches(DATA) == [["CP2"], ["CP1"]]
    rows = [json.loads(line) for line in rv.question_lines(DATA, ["CP1", "CP2"]).decode().splitlines()]
    assert rows[0]["id"] == "CP1" and rows[0]["readings"] == ["a", "b"] and "baseline" in rows[0]["code_object"]
    assert rows[1]["task_quote"] == "retried" and "code_object" not in rows[1]
    report = rv.defects_markdown(DATA, [
        {"id": "CP2", "status": "defect", "severity": "low", "candidate_behavior": "b2", "expected": "e2", "evidence": []},
        {"id": "CP1", "status": "defect", "severity": "high", "candidate_behavior": "b1", "expected": "e1",
         "evidence": [{"kind": "code", "path": "src/a.py", "line": 2, "detail": "d"}]},
        {"id": "CP1", "status": "ok", "candidate_behavior": "x"}])
    assert report.index("given g1") < report.index("given g2") and "(2 defects)" in report
    assert "{{task}}" in rv.header() and "review.jsonl" in rv.header()


def test_candidate_patch_drops_environment_sections(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "v68_review_runner", Path(__file__).resolve().parents[1] / "scripts/deepswe_v68_gen/review.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    env = "diff --git a/x b/x\n+env\n"
    (tmp_path / "model.patch").write_text(env + "diff --git a/y b/y\n+mine\n")
    (tmp_path / "environment-baseline.patch").write_text(env)
    text, source = module.candidate_patch(tmp_path)
    assert text == "diff --git a/y b/y\n+mine\n" and source.endswith("model.patch")
