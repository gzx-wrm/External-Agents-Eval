import json

from external_agents_eval.agents.pier.trajectory import (
    hermes_json_to_atif,
    pi_jsonl_to_atif,
)


def test_pi_jsonl_converts_messages_tools_and_metrics():
    events = [
        {"type": "session", "id": "session-1"},
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "model": "deepseek-v4-flash",
                "content": [
                    {"type": "thinking", "thinking": "inspect"},
                    {
                        "type": "toolCall",
                        "id": "call-1",
                        "name": "bash",
                        "arguments": {"command": "git status"},
                    },
                ],
                "usage": {
                    "input": 100,
                    "output": 20,
                    "cacheRead": 40,
                    "cost": {"total": 0.01},
                },
            },
        },
        {
            "type": "message_end",
            "message": {
                "role": "toolResult",
                "toolCallId": "call-1",
                "content": [{"type": "text", "text": "clean"}],
            },
        },
    ]
    trajectory = pi_jsonl_to_atif(
        "\n".join(json.dumps(event) for event in events),
        instruction="Fix it",
        model="deepseek-v4-flash",
        agent_version="v0.84.2",
    )
    agent_step = trajectory["steps"][1]
    assert trajectory["schema_version"] == "ATIF-v1.7"
    assert agent_step["tool_calls"][0]["tool_call_id"] == "call-1"
    assert agent_step["observation"]["results"][0]["content"] == "clean"
    assert trajectory["final_metrics"]["total_prompt_tokens"] == 100


def test_hermes_json_converts_openai_messages():
    payload = {
        "completed": True,
        "api_calls": 1,
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "reasoning": "inspect",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path":"/app/a.py"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "source"},
            {"role": "assistant", "content": "Done"},
        ],
    }
    trajectory = hermes_json_to_atif(
        payload,
        instruction="Fix it",
        model="deepseek-v4-flash",
        agent_version="v2026.8.18",
    )
    assert trajectory["agent"]["name"] == "hermes"
    assert trajectory["steps"][1]["observation"]["results"][0]["content"] == "source"
    assert trajectory["steps"][-1]["message"] == "Done"
