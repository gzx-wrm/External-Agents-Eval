import json

import pytest

from external_agents_eval.agents.pier.trajectory import extract_pi_final_assistant_text


def test_extracts_last_completed_pi_assistant_text_only():
    events = [
        {
            "type": "message_end",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "first"}]},
        },
        {
            "type": "message_end",
            "message": {"role": "toolResult", "content": [{"type": "text", "text": "tool JSON"}]},
        },
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "private"},
                    {"type": "text", "text": '{"verdict":"approve"}'},
                ],
            },
        },
    ]
    assert (
        extract_pi_final_assistant_text("\n".join(map(json.dumps, events)))
        == '{"verdict":"approve"}'
    )


def test_final_message_extractor_does_not_scan_non_json_stdout():
    with pytest.raises(ValueError, match="no completed assistant"):
        extract_pi_final_assistant_text('noise {"verdict":"approve"}')
