import json

from external_agents_eval.workflows.solve_review_repair import v68_generation as v68
from external_agents_eval.workflows.solve_review_repair import v68_lint as lint

TASK = "Calls are retried at most 3 times. Use `retry(fn)` to wrap calls."
EVIDENCE = [{"path": "src/a.py", "line_start": 10, "line_end": 11, "text": "def backoff(n):\n    return min(2 ** n, 30)"}]
OBLIGATIONS = [{"id": "O1", "statement": "retry", "task_quote": "Calls are retried at most 3 times."},
               {"id": "O2", "statement": "wrap", "task_quote": "Use retry(fn) to wrap calls."}]


def batch(**over):
    data = {"obligations": OBLIGATIONS,
            "entities": [{"id": "E1", "kind": "analog", "symbol": "backoff", "path": "src/a.py", "line": 10,
                          "obligations": ["O1"], "why": "nearest analog"}],
            "items": [{"ext": "entity", "obligation": "O1", "entity": "E1", "kind": "question", "given": "g",
                       "when": "w", "then": "t", "alternatives": ["a", "b"],
                       "observations": [{"text": "caps at 30", "path": "src/a.py", "line": 11}]},
                      {"ext": "boundary", "obligation": "O1", "entity": None, "kind": "question", "given": "g2",
                       "when": "w", "then": "t", "task_quote": "retried at most 3 times",
                       "alternatives": ["3 retries after the first call", "3 calls in total"]}],
            "not_applicable": [{"obligation": "O1", "ext": e, "reason": "does not apply here"}
                               for e in ("analog_form", "external_spec")]}
    data.update(over)
    return data


def test_batch_scope_passes_and_requires_targets_only():
    errors, _, stats = lint.lint(TASK, batch(), lint.EvidenceRepo(EVIDENCE), {"mode": "batch", "targets": ["O1"]})
    assert errors == [] and stats["items"] == 2


def test_questions_need_readings_and_requirements_are_rejected():
    bad = batch(items=[{"ext": "boundary", "obligation": "O1", "kind": "question", "given": "g", "when": "w",
                        "then": "t", "alternatives": ["a"], "favored": "a"},
                       {"ext": "entity", "obligation": "O1", "entity": "E1", "kind": "requirement", "given": "x",
                        "when": "w", "then": "t", "task_quote": "not in the task text"}])
    errors, _, _ = lint.lint(TASK, bad, lint.EvidenceRepo(EVIDENCE), {"mode": "batch", "targets": ["O1"]})
    text = "\n".join(errors)
    assert "two distinct alternatives" in text and "must not carry favored" in text
    assert 'kind must be "question"' in text


def test_entities_must_point_at_real_symbols_and_items_stay_in_scope():
    bad = batch(entities=[{"id": "E1", "kind": "analog", "symbol": "retry_all", "path": "src/a.py", "line": 10,
                           "obligations": ["O2"], "why": "x"}])
    errors, _, _ = lint.lint(TASK, bad, lint.EvidenceRepo(EVIDENCE), {"mode": "batch", "targets": ["O1"]})
    text = "\n".join(errors)
    assert "does not appear" in text and "your target obligation ids" in text


def test_merge_renumbers_entities_across_batches():
    first = {"batch": 1, "entities": [{"id": "E1"}], "items": [{"entity": "E1", "ext": "entity"}], "not_applicable": []}
    second = {"batch": 2, "entities": [{"id": "E1"}], "items": [{"entity": "E1", "ext": "entity"}], "not_applicable": []}
    merged = v68.merge_batches(OBLIGATIONS, [first, second])
    assert [i["entity"] for i in merged["items"]] == ["E1", "E2"]


def test_headers_keep_the_mini_task_placeholder_and_batches_of_three():
    assert "{{task}}" in v68.plan_header() and "{{task}}" in v68.batch_header()
    assert "Your target obligations" in v68.batch_header() and "Your target obligations" not in v68.plan_header()
    ids = [{"id": f"O{i}"} for i in range(1, 8)]
    assert v68.target_batches(ids) == [["O1", "O2", "O3"], ["O4", "O5", "O6"], ["O7"]]
    data, errors = v68.control_data(json.loads('{"entities": [], "items": [{"ext": "boundary"}], "not_applicable": []}'))
    assert errors == [] and data["items"][0]["_line"] == 1


def test_plan_caps_obligations_and_runner_keeps_the_first_twelve():
    many = [{"id": f"O{i}", "statement": "s", "task_quote": "Calls are retried at most 3 times."} for i in range(1, 14)]
    errors, _, _ = lint.lint(TASK, {"obligations": many}, lint.EvidenceRepo(EVIDENCE), {"mode": "plan"})
    assert any("at most 12" in e for e in errors)
    errors, _, _ = lint.lint(TASK, {"obligations": many[:12]}, lint.EvidenceRepo(EVIDENCE), {"mode": "plan"})
    assert errors == []
    assert [o["id"] for o in v68.capped(many)] == [f"O{i}" for i in range(1, 13)]
    assert "AT MOST 12" in v68.plan_header() and "requirement|question" not in v68.batch_header()


def _rank_select():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts/deepswe_v68_gen/rank_select.py"
    spec = importlib.util.spec_from_file_location("v68_rank_select", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selection_check_and_ratio_mix():
    rs = _rank_select()
    good = {"targeted": [f"T{i}" for i in range(1, 11)], "generic": [f"G{i}" for i in range(10, 0, -1)]}
    assert rs.check(good, 30, 12) == []
    assert any("exactly 10" in e for e in rs.check({"targeted": ["T1"], "generic": good["generic"]}, 30, 12))
    assert any("unknown" in e for e in rs.check({"targeted": good["targeted"], "generic": ["G99"] * 10}, 30, 12))
    assert rs.check({"targeted": ["T2", "T1"], "generic": ["G1"]}, 2, 1) == []
    assert rs.mix(good, 10) == good["targeted"][:7] + good["generic"][:3]
    assert rs.mix(good, 5) == good["targeted"][:4] + good["generic"][:1]
    assert rs.mix(good, 8, share=0.5) == good["targeted"][:4] + good["generic"][:4]
    short = {"targeted": ["T1"], "generic": ["G1", "G2"]}
    assert rs.mix(short, 3) == ["T1", "G1", "G2"]
