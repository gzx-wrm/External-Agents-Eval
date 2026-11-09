from __future__ import annotations

from types import SimpleNamespace

from a_agent_finetune import compress_teacher_thinking as c

ARGS = SimpleNamespace(max_words=60, closing_max_words=75)


def call(name, ident, path):
    return {"id": ident, "type": "function", "function": {"name": name, "arguments": {"path": path}}}


def conversation():
    tools = [{"type": "function", "function": {"name": name, "parameters": {
        "type": "object", "required": ["path"], "properties": {"path": {"type": "string"}},
    }}} for name in ("ls", "read")]
    return {"tools": tools, "messages": [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "TASK TEXT"},
        {"role": "assistant", "content": "", "reasoning_content": "List the root first.",
         "tool_calls": [call("ls", "a", "/work/base")]},
        {"role": "tool", "name": "ls", "tool_call_id": "a", "content": "expr.go\nvm/"},
        {"role": "assistant", "content": "", "reasoning_content": "expr.go is a file, read it.",
         "tool_calls": [call("read", "b", "/work/base/expr.go")]},
        {"role": "tool", "name": "read", "tool_call_id": "b", "content": "package expr"},
        {"role": "assistant", "content": '{"summary":"LEDGER"}', "reasoning_content": "DRAFT"},
    ]}


def test_plan_turns_skips_final_ledger_and_keeps_observations():
    sample, _, turns, final = c.plan_turns(conversation())
    assert [turn["source_message_index"] for turn in turns] == [2, 4]
    assert turns[0]["observations"] == []
    assert turns[1]["observations"][0][1] == "expr.go\nvm/"
    assert turns[1]["thinking"] == "expr.go is a file, read it."
    assert sample["messages"][-1]["content"] == '{"summary":"LEDGER"}'
    assert final["source_message_index"] == 6 and final["thinking"] == "DRAFT"
    assert final["observations"][0][1] == "package expr"


def test_check_note_rejects_ungrounded_and_malformed_notes():
    grounding = "read /work/base/expr.go lines 10-20"
    assert c.check_note("expr.go is a file, so I read it instead of listing it.", grounding, [], ARGS) == []
    assert any("ungrounded reference" in p for p in c.check_note("I read vm/vm.go next.", grounding, [], ARGS))
    assert any("line number" in p for p in c.check_note("Lines 40-50 matter.", grounding, [], ARGS))
    assert any("<tool_call" in p for p in c.check_note("<tool_call>", grounding, [], ARGS))
    assert c.check_note("Same.", grounding, ["Same."], ARGS) == ["repeats an earlier note"]
    assert any("sentences" in p for p in c.check_note("A. B. C. D.", grounding, [], ARGS))


def test_check_note_rejects_implementer_framing():
    for note in ("Reading go.mod first, before I make any changes.",
                 "I'll add the helper next.", "This shows where to wire the option before wiring it."):
        assert any("implementer framing" in p for p in c.check_note(note, "go.mod", [], ARGS)), note
    assert c.check_note("This shows how the option is added today, so I read go.mod.", "go.mod", [], ARGS) == []


LEDGER = {"summary": "S", "open_questions": [{"question": "Q"}]}
CLOSING = ("The read of expr.go confirmed the public entry points the change must keep. "
           "That covers the requested and preservation obligations, so the remaining uncertainty goes "
           "into open questions and I return the Ledger now.")


def test_check_closing_note_accepts_a_stop_decision():
    assert c.check_closing_note(CLOSING, "expr.go", LEDGER, ARGS) == []


def test_check_closing_note_rejects_continuing_braces_and_missing_decision():
    def problems(note, ledger=LEDGER):
        return " ".join(c.check_closing_note(note, "expr.go", ledger, ARGS))

    assert "another tool action" in problems(CLOSING + " Next I will read expr.go again.")
    assert "braces" in problems(CLOSING.replace("Ledger now", "Ledger {now}"))
    assert "Ledger is returned" in problems("The evidence is sufficient. Remaining open questions are recorded.")
    assert "open questions" in problems(CLOSING.replace("open questions", "the output"))
    assert "open questions" not in problems(CLOSING.replace("open questions", "the output"), {"summary": "S"})
    assert "2 to 3 sentences" in problems("I return the Ledger now with open questions recorded.")
