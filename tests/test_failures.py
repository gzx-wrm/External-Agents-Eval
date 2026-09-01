from external_agents_eval.failures import FailureKind, classify_failure


def test_deepseek_timeout_is_llm_service():
    assert classify_failure("DeepSeek connection timed out") == FailureKind.LLM_SERVICE


def test_docker_download_failure_is_external_network():
    assert classify_failure("Docker pull download failed: TLS EOF") == FailureKind.EXTERNAL_NETWORK


def test_agent_process_failure_is_agent_failure():
    assert classify_failure("Agent process exited with status 2") == FailureKind.AGENT
