import json

from external_agents_eval.workflows.solve_review_repair import v68_repair as rp
from external_agents_eval.workflows.solve_review_repair import v68_repair_lint as plint

DATA = {"items": [{"given": f"g{i}", "when": "w", "then": f"q{i}?", "task_quote": "quote" if i == 1 else None,
                   "alternatives": [f"r{i}a", f"r{i}b", f"r{i}c"]} for i in range(1, 6)]}
REVIEW = {"CP1": {"status": "defect", "candidate_behavior": "does a",
                  "evidence": [{"kind": "code", "path": "x.py", "line": 3, "detail": "d"}]},
          "CP2": {"status": "defect", "candidate_behavior": "does b"},
          "CP3": {"status": "ok", "candidate_behavior": "does c"},
          "CP4": {"status": "defect", "candidate_behavior": "does d"},
          "CP5": {"status": "defect", "candidate_behavior": "does e"}}
CAND = {"CP1": 1, "CP2": 2, "CP3": 1, "CP4": -1, "CP5": 1}
ANSWERS = {"CP1": {"reading": 2, "confidence": 70, "why": "w1", "basis": "task_text", "evidence": "quote"},
           "CP2": {"reading": 2, "confidence": 99, "why": "same as candidate"},
           "CP3": {"reading": 3, "confidence": 99, "why": "review said ok"},
           "CP4": {"reading": 2, "confidence": 99, "why": "candidate unknown"},
           "CP5": {"reading": 0, "confidence": 90, "why": "w5", "instead": "does f", "basis": "none", "evidence": ""}}


def test_select_issues_needs_defect_and_disagreement_and_sorts_by_confidence(monkeypatch):
    issues = rp.select_issues(DATA, REVIEW, CAND, ANSWERS)
    assert [(i["id"], i["question"]) for i in issues] == [("I1", "CP5"), ("I2", "CP1")]
    assert issues[0]["expected"] == "does f" and issues[1]["expected"] == "r1b"
    monkeypatch.setattr(rp, "MAX_ISSUES", 1)
    assert [i["question"] for i in rp.select_issues(DATA, REVIEW, CAND, ANSWERS)] == ["CP5"]


def test_issues_markdown_and_scope():
    issues = rp.select_issues(DATA, REVIEW, CAND, ANSWERS)
    text = rp.issues_markdown(issues)
    assert "## I2" in text and "Task passage: \"quote\"" in text and "code x.py:3: d" in text
    assert "Expected behavior: r1b" in text and "Reasoning (task_text): quote w1" in text
    assert "CP1" not in text and "r1a" not in text  # no question ids, no other readings
    assert json.loads(rp.scope(issues)) == {"ids": ["I1", "I2"]}
    assert "{{task}}" in rp.header() and rp.SUBMIT in rp.header()


def test_repair_lint(tmp_path):
    (tmp_path / "scope.json").write_text(json.dumps({"ids": ["I1", "I2"]}))
    ids, rows, errors = plint.load(str(tmp_path))
    assert errors == ["decisions.jsonl is missing"]
    (tmp_path / "decisions.jsonl").write_text(
        json.dumps({"id": "I1", "decision": "fixed", "note": ""}) + "\n" + json.dumps({"id": "I9", "decision": "kept"}) + "\n")
    ids, rows, errors = plint.load(str(tmp_path))
    problems, _ = plint.lint(ids, rows)
    text = "\n".join(problems)
    assert "decision must be changed or kept" in text and "note" in text and "'I9' is not an issue id" in text
    assert "1 issue ids have no decision: I2" in text
    with open(tmp_path / "decisions.jsonl", "a") as handle:
        handle.write(json.dumps({"id": "I1", "decision": "changed", "note": "fixed x"}) + "\n")
        handle.write(json.dumps({"id": "I2", "decision": "kept", "note": "already right"}) + "\n")
    ids, rows, _ = plint.load(str(tmp_path))
    problems, stats = plint.lint([i for i in ids], [r for r in rows if r.get("id") != "I9"])
    assert problems == [] and stats == {"issues": 2, "decided": 2, "changed": 1, "kept": 1}
