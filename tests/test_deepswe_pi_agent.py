from external_agents_eval.agents.pier.pi import PI_BUILTIN_TOOLS, _build_pi_command


def test_pi_pier_enables_every_builtin_tool():
    assert PI_BUILTIN_TOOLS == (
        "read",
        "bash",
        "edit",
        "write",
        "grep",
        "find",
        "ls",
    )
    command = _build_pi_command(model="deepseek-test", prompt_path="/tmp/instruction.txt")
    assert "--tools read,bash,edit,write,grep,find,ls" in command
