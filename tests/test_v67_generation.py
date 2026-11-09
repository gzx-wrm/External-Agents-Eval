from external_agents_eval.workflows.solve_review_repair import v67_generation as v67

PACKET = {
    "task": {"title": "Add retry", "description": "Calls are retried at most 3 times. A call that "
             "was not explicitly cancelled keeps its handle. Use `retry(fn)`."},
    "evidence": [{"path": "src/a.py", "line_start": 10, "line_end": 12,
                  "text": "def backoff(n):\n    return min(2 ** n, 30)\n# done"}],
}
LEDGER = {"summary": "s", "obligations": [{"id": "O001", "content": {"statement": "Add `retry(fn)`.",
                                                                     "importance": "required", "sources": []}}],
          "preservation_obligations": [], "open_questions": []}


def requirement(**over):
    point = {"targets": ["O001"], "facet": "D4", "kind": "requirement", "given": "a failing call",
             "when": "it keeps failing", "then": "it is retried at most 3 times",
             "sources": [{"kind": "task", "field": "description", "quote": "Calls are retried at most 3 times."}],
             "rules_out": "retrying forever", "alternatives": [], "observations": []}
    return {**point, **over}


def question(**over):
    point = {"targets": ["O001"], "facet": "D6", "kind": "question", "given": "a retry", "when": "it waits",
             "then": "the delay between attempts is not stated",
             "sources": [], "rules_out": None, "alternatives": ["no delay", "exponential backoff"],
             "observations": [{"kind": "analog", "observation": "backoff doubles up to 30",
                               "basis_quote": "return min(2 ** n, 30)", "convention": None,
                               "sources": [{"kind": "repository", "path": "src/a.py",
                                            "line_start": 11, "line_end": 11}]}]}
    return {**point, **over}


def coverage(covered):
    return [{"target": "O001", "facet": f, "status": "covered" if f in covered else "not_applicable",
             "reason": None if f in covered else "n/a"} for f in v67.FACETS]


def test_ledger_prompt_adds_fidelity_rules_before_the_output_contract():
    prompt = v67.ledger_system_prompt()
    assert prompt.index("V6.7 FIDELITY RULES") < prompt.index("Return exactly one JSON object")
    assert "OUTPUT JSON SCHEMA" in prompt


def test_valid_requirement_and_question_pass():
    payload = {"checkpoints": [requirement(), question()], "coverage": coverage({"D4", "D6"})}
    assert v67.check_batch(payload, ledger=LEDGER, packet=PACKET, batch=["O001"]).messages() == []


def test_questions_cannot_exclude_and_requirements_need_task_quotes():
    bad_q = question(rules_out="no delay")
    bad_r = requirement(sources=[], alternatives=["a", "b"])
    payload = {"checkpoints": [bad_r, bad_q], "coverage": coverage({"D4", "D6"})}
    messages = v67.check_batch(payload, ledger=LEDGER, packet=PACKET, batch=["O001"]).messages()
    assert any("rules_out must be null" in m for m in messages)
    assert any("needs the exact task quote" in m for m in messages)
    assert any("no alternatives or observations" in m for m in messages)


def test_observation_quotes_must_come_from_cited_lines():
    obs = question()["observations"][0] | {"basis_quote": "return 2 ** n"}
    payload = {"checkpoints": [question(observations=[obs])], "coverage": coverage({"D6"})}
    messages = v67.check_batch(payload, ledger=LEDGER, packet=PACKET, batch=["O001"]).messages()
    assert any("basis_quote is not exact" in m for m in messages)


def test_limits_are_per_kind():
    points = [question(given=f"case {i}") for i in range(v67.MAX_QUESTIONS_PER_TARGET + 1)]
    kept, trimmed = v67.enforce_limit([*points, requirement()])
    assert sum(p["kind"] == "question" for p in kept) == v67.MAX_QUESTIONS_PER_TARGET
    assert sum(p["kind"] == "requirement" for p in kept) == 1 and trimmed[0]["point_dropped"]


def test_fidelity_flags_find_invented_signatures_and_lost_qualifiers():
    ledger = {**LEDGER, "obligations": [{"id": "O001", "content": {"statement": "Call `retry(fn, times=3)`."}}]}
    lost = requirement(then="the call keeps its handle",
                       sources=[{"kind": "task", "field": "description",
                                 "quote": "A call that was not explicitly cancelled keeps its handle."}])
    flags = v67.fidelity_flags(ledger, [requirement(), lost], PACKET)
    assert flags["ledger"] == [{"id": "O001", "unquoted_interfaces": ["retry(fn, times=3)"]}]
    assert flags["checkpoints"] == [{"cp": "CP2", "dropped_qualifiers": ["not", "explicitly"]}]
    assert v67.unquoted_interfaces("Use `retry(fn)`", PACKET["task"]["description"], "") == []


def test_solver_view_lists_requirements_first_and_questions_without_answers():
    payload = {"checkpoints": [question(), requirement()], "coverage": []}
    views = v67.delivery_views(payload, {"ledger": [], "checkpoints": []})
    text = views["solver_text"]
    assert text.index("CP2") < text.index("CP1")
    assert "Excludes" not in text and "Resolved" not in text
    assert "Readings: (1) no delay | (2) exponential backoff" in text
    assert 'task: "Calls are retried at most 3 times."' in text
