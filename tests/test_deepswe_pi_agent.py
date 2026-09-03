from external_agents_eval.agents.pier.pi import (
    PI_BUILTIN_TOOLS,
    PiPierAgent,
    _build_pi_command,
)


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


def test_pi_pier_reads_prompt_from_stdin():
    command = _build_pi_command(
        model="deepseek-test",
        prompt_path="/tmp/instruction with spaces.txt",
    )
    assert "-p < '/tmp/instruction with spaces.txt'" in command
    assert "$(cat" not in command


def test_pi_pier_loads_nvm_without_auto_using_project_version():
    agent = object.__new__(PiPierAgent)
    install_spec = agent.install_spec()
    install_script = "\n".join(step.run for step in install_spec.steps)
    assert "curl --retry 5 --retry-all-errors --retry-delay 2" in install_script
    assert '. "$NVM_DIR/nvm.sh" --no-use' in install_script
