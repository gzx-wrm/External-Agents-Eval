import json

from external_agents_eval.workflows.solve_review_repair import v68_observe as ob
from external_agents_eval.workflows.solve_review_repair import v68_observe_lint as olint

DATA = {"obligations": [{"id": "O1", "statement": "retry"}], "entities": [],
        "items": [{"obligation": "O1", "batch": 1, "given": "g", "when": "w", "then": "t", "alternatives": ["a", "b"]},
                  {"obligation": "O1", "batch": 1, "given": "g2", "when": "w", "then": "t", "alternatives": ["c", "d", "e"]}]}


def _setup(tmp_path, lines):
    (tmp_path / "scope.json").write_text(json.dumps({"ids": ["CP1", "CP2"]}))
    (tmp_path / "questions.jsonl").write_bytes(ob.question_lines(DATA, ["CP1", "CP2"]))
    (tmp_path / "labels.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("x\ny\n")
    return str(repo)


def test_numbered_readings():
    rows = [json.loads(line) for line in ob.question_lines(DATA, ["CP1", "CP2"]).decode().splitlines()]
    assert rows[1]["readings"] == ["1. c", "2. d", "3. e"] and "{{task}}" in ob.header()


def test_observe_lint_checks_reading_range_and_evidence(tmp_path):
    repo = _setup(tmp_path, [{"id": "CP1", "status": "observed", "reading": 3, "behavior": "b",
                              "evidence": [{"kind": "code", "path": "a.py", "line": 2, "detail": "d"}]},
                             {"id": "CP2", "status": "observed", "reading": 0, "behavior": "other"}])
    ids, counts, rows, parse = olint.load(str(tmp_path))
    errors, _ = olint.lint(ids, counts, rows, repo)
    text = "\n".join(errors)
    assert parse == [] and "reading must be an integer 1-2" in text and "observed needs evidence" in text


def test_observe_lint_passes_and_last_line_wins(tmp_path):
    repo = _setup(tmp_path, [{"id": "CP1", "status": "undetermined"},
                             {"id": "CP2", "status": "not_applicable", "behavior": "no such API"},
                             {"id": "CP1", "status": "observed", "reading": 2, "behavior": "b",
                              "evidence": [{"kind": "probe", "detail": "python p.py -> 2"}]}])
    ids, counts, rows, _ = olint.load(str(tmp_path))
    errors, stats = olint.lint(ids, counts, rows, repo)
    assert errors == [] and stats["observed"] == 1 and stats["not_applicable"] == 1


def test_mapping_check_and_prompt():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "v68_map", Path(__file__).resolve().parents[1] / "scripts/deepswe_v68_gen/map_readings.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.check({"labels": [{"id": "CP1", "reading": 2}, {"id": "CP2", "reading": -1}]}, DATA, ["CP1", "CP2"]) == []
    errors = module.check({"labels": [{"id": "CP1", "reading": 3}]}, DATA, ["CP1", "CP2"])
    assert any("-1..2" in e for e in errors) and any("missing" in e for e in errors)
    rows = [{"id": "CP2", "status": "ok", "candidate_behavior": "does d", "evidence": [{"kind": "probe", "detail": "x"}]}]
    user = module.messages_for(DATA, rows, "review")[1]["content"]
    assert "2. d" in user and "behavior: does d" in user


BLIND_SHA = "5054d86e5230"  # sha256 prefix of the v6.8-resolve-1 (direction B) system prompt


def _resolve():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts/deepswe_v68_gen/resolve.py"
    spec = importlib.util.spec_from_file_location("v68_resolve", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_chunks_and_shuffle_maps_back():
    from external_agents_eval.workflows.solve_review_repair import v68_review as rv

    rs = _resolve()
    assert [len(c) for c in rs.chunks(list(range(71)))] == [24, 23, 24]
    assert [len(c) for c in rs.chunks(list(range(35)))] == [35]
    rows, perms = rs.question_rows(rv, DATA, ["CP1", "CP2"], "t", "shuffled")
    assert perms["CP1"] == [2, 1] and perms["CP2"][0] != 1 and sorted(perms["CP2"]) == [1, 2, 3]
    shown = json.loads(rows[1])["readings"]
    assert shown[0] == "1. " + "cde"[perms["CP2"][0] - 1]
    assert rs.to_original({"reading": 1, "confidence": 80, "why": "w"}, perms["CP2"])["reading"] == perms["CP2"][0]
    assert rs.to_original({"reading": 0, "confidence": 60, "why": "w", "instead": "x"}, [2, 1])["reading"] == 0
    _, same = rs.question_rows(rv, DATA, ["CP1", "CP2"], "t", "original")
    assert same == {"CP1": [1, 2], "CP2": [1, 2, 3]}


def test_resolve_check():
    rs = _resolve()
    perms = {"CP1": [1, 2], "CP2": [1, 2, 3]}
    good = {"answers": [{"id": "CP1", "reading": 2, "confidence": 70, "why": "w"},
                        {"id": "CP2", "reading": 0, "confidence": 55, "why": "w", "instead": "x"}]}
    assert rs.check(good, perms) == []
    bad = {"answers": [{"id": "CP1", "reading": 3, "confidence": 170, "why": ""},
                       {"id": "CP1", "reading": 0, "confidence": 5, "why": "w"}]}
    text = "\n".join(rs.check(bad, perms))
    assert "reading must be an integer 0-2" in text and "confidence must be" in text and "why is empty" in text
    assert "answered more than once" in text and "needs instead" in text and "1 ids missing: CP2" in text


def test_resolve_evidence_prompt_check_and_blind_prompt_unchanged():
    import hashlib

    rs = _resolve()
    perms = {"CP1": [1, 2], "CP2": [1, 2, 3]}
    good = {"answers": [{"id": "CP1", "reading": 2, "basis": "repo_precedent", "evidence": "a.py:3 x", "confidence": 70,
                         "why": "w"},
                        {"id": "CP2", "reading": 1, "basis": "none", "evidence": "", "confidence": 55, "why": "w"}]}
    assert rs.check(good, perms, "evidence") == [] and rs.check(good, perms) == []
    bad = {"answers": [{"id": "CP1", "reading": 2, "basis": "hunch", "confidence": 70, "why": "w"},
                       {"id": "CP2", "reading": 1, "basis": "task_text", "evidence": " ", "confidence": 55, "why": "w"}]}
    text = "\n".join(rs.check(bad, perms, "evidence"))
    assert "basis must be one of" in text and "basis task_text needs evidence" in text
    assert rs.to_original(good["answers"][0], [2, 1])["basis"] == "repo_precedent"
    # direction B prompt and shuffle seed are the ones of v6.8-resolve-1
    assert hashlib.sha256(rs.BLIND.encode()).hexdigest()[:12] == BLIND_SHA
    assert rs.messages_for("t", ["{}"], "evidence")[0]["content"] == rs.EVIDENCE


def test_resolve_question_subset_chunks_over_task_in_batch_order():
    import pytest

    from external_agents_eval.workflows.solve_review_repair import v68_review as rv

    rs = _resolve()
    item = {"obligation": "O1", "given": "g", "when": "w", "then": "t", "alternatives": ["a", "b"]}
    data = {"items": [{**item, "batch": 2}, {**item, "batch": 1}, {**item, "batch": 2}, {**item, "batch": 1}]}
    # full run: per generation batch, unchanged directory names
    assert rs.plan_chunks(rv, data) == [("b01-c1", ["CP2", "CP4"]), ("b02-c1", ["CP1", "CP3"])]
    # subset: batch order kept, chunks span batches
    assert rs.plan_chunks(rv, data, {"CP3", "CP4", "CP1"}) == [("s-c01", ["CP4", "CP1", "CP3"])]
    many = {"items": [{**item, "batch": 1 + i // 30} for i in range(80)]}
    plan = rs.plan_chunks(rv, many, {f"CP{i}" for i in range(1, 81)})
    assert [len(ids) for _, ids in plan] == [27, 26, 27] and plan[0][1][0] == "CP1"
    with pytest.raises(ValueError, match="not in the task"):
        rs.plan_chunks(rv, data, {"CP9"})
