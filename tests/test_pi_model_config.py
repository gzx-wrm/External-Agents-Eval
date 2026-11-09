import json
from pathlib import Path

from pier.models.agent.network import NetworkAllowlist

from external_agents_eval.agents.pier.pi import _build_pi_command
from external_agents_eval.agents.pier.pi_model_config import (
    PiModelSpec,
    PiStageModelConfig,
)
from external_agents_eval.benchmarks.deepswe.runner import build_pier_command
from external_agents_eval.benchmarks.deepswe.manifest import DeepSWEManifest


def _manifest(tmp_path: Path) -> DeepSWEManifest:
    return DeepSWEManifest(
        path=tmp_path / "manifest.json",
        task_ids=("task-a",),
        sample_seed=1,
        sample_size=1,
        population_count=1,
        population_fingerprint="a" * 64,
        raw={},
    )


def test_stage_config_defaults_to_deepseek_and_supports_provider_prefix():
    config = PiStageModelConfig.from_value(
        {
            "solver": {"model": "deepseek-v4-flash"},
            "A": {
                "model": "openai/gpt-5.6-sol",
                "base_url": "https://proxy.example.test/v1/",
            },
        }
    )

    assert config.for_stage("solver") == PiModelSpec(
        provider="deepseek", model="deepseek-v4-flash", base_url=None
    )
    assert config.for_stage("A").provider == "openai"
    assert config.for_stage("A").model == "gpt-5.6-sol"
    assert config.for_stage("A").base_url == "https://proxy.example.test/v1"
    assert config.for_role("v5_specification_reviewer") == config.for_stage("A")
    assert config.required_env_keys() == ("DEEPSEEK_API_KEY", "OPENAI_API_KEY")
    assert "proxy.example.test" in config.domains()


def test_models_json_is_secret_free_and_provider_scoped():
    config = PiStageModelConfig.from_value(
        {"A": {"model": "openai/gpt-5.6-sol", "base_url": "https://proxy.test/v1"}}
    )
    payload = config.models_json(config.for_stage("A"))
    assert payload == {"providers": {"openai": {"baseUrl": "https://proxy.test/v1"}}}
    assert "API_KEY" not in json.dumps(payload)


def test_custom_model_metadata_registers_a_new_native_provider_model():
    config = PiStageModelConfig.from_value(
        {
            "default": {
                "provider": "deepseek",
                "model": "deepseek-v4.1-flash",
                "model_metadata": {
                    "name": "DeepSeek V4.1 Flash",
                    "reasoning": True,
                    "contextWindow": 1000000,
                    "maxTokens": 384000,
                    "compat": {
                        "supportsStore": False,
                        "supportsDeveloperRole": False,
                        "maxTokensField": "max_tokens",
                    },
                },
            }
        }
    )

    payload = config.models_json(config.default)
    assert payload == {
        "providers": {
            "deepseek": {
                "baseUrl": "https://api.deepseek.com/v1",
                "api": "openai-completions",
                "models": [
                    {
                        "id": "deepseek-v4.1-flash",
                        "name": "DeepSeek V4.1 Flash",
                        "api": "openai-completions",
                        "reasoning": True,
                        "input": ["text"],
                        "contextWindow": 1000000,
                        "maxTokens": 384000,
                        "compat": {
                            "supportsStore": False,
                            "supportsDeveloperRole": False,
                            "maxTokensField": "max_tokens",
                        },
                    }
                ],
            }
        }
    }
    assert config.required_env_keys() == ("DEEPSEEK_API_KEY",)
    assert config.domains() == ("api.deepseek.com",)


def test_custom_model_metadata_is_inherited_by_stage_overrides():
    config = PiStageModelConfig.from_value(
        {
            "default": {
                "model": "deepseek-v4.1-flash",
                "model_metadata": {"contextWindow": 1000000},
            },
            "repair": {"model": "deepseek-v4.1-flash"},
        }
    )
    assert config.for_stage("repair").model_metadata == {"contextWindow": 1000000}


def test_custom_model_metadata_does_not_imply_disabled_compaction():
    config = PiStageModelConfig.from_value(
        {
            "default": {
                "model": "deepseek-v4.1-flash",
                "model_metadata": {"contextWindow": 1000000},
            }
        }
    )
    assert config.needs_custom_model_config(config.default)


def test_custom_ip_base_url_is_forwarded_to_pier_network_allowlist():
    config = PiStageModelConfig.from_value(
        {
            "A": {
                "model": "openai/gpt-5.6-sol",
                "base_url": "http://43.134.83.232:3000/v1",
            }
        }
    )

    allowlist = NetworkAllowlist(domains=list(config.domains()))

    assert "43.134.83.232" in allowlist.domains
    assert "api.deepseek.com" in allowlist.domains


def test_fully_overridden_openai_stages_do_not_require_unused_default_key():
    config = PiStageModelConfig.from_value(
        {
            stage: {"model": "openai/gpt-5.6-sol"}
            for stage in ("solver", "A", "I", "R", "S", "repair")
        }
    )
    assert config.required_env_keys() == ("OPENAI_API_KEY",)


def test_zai_default_routes_every_stage_through_native_provider():
    config = PiStageModelConfig.from_value(
        {
            "default": {
                "model": "zai/glm-5.3",
                "base_url": "https://open.bigmodel.cn/api/paas/v4",
            }
        }
    )

    assert config.default == PiModelSpec(
        provider="zai",
        model="glm-5.3",
        base_url="https://open.bigmodel.cn/api/paas/v4",
    )
    assert all(
        config.for_stage(stage) == config.default
        for stage in ("solver", "A", "I", "R", "S", "repair", "json_corrector")
    )
    assert config.required_env_keys() == ("ZAI_API_KEY",)
    assert config.domains() == ("open.bigmodel.cn",)
    assert config.models_json(config.default) == {
        "providers": {
            "zai": {"baseUrl": "https://open.bigmodel.cn/api/paas/v4"}
        }
    }


def test_pi_command_accepts_provider_without_changing_default_shape():
    command = _build_pi_command(
        provider="openai",
        model="gpt-5.6-sol",
        prompt_path="/tmp/prompt.md",
    )
    assert "--provider openai" in command
    assert "--model gpt-5.6-sol" in command


def test_pi_command_uses_native_zai_provider_for_glm():
    command = _build_pi_command(
        provider="zai",
        model="glm-5.3",
        prompt_path="/tmp/prompt.md",
    )
    assert "--provider zai" in command
    assert "--model glm-5.3" in command


def test_pi_command_infers_provider_from_provider_prefixed_model():
    command = _build_pi_command(model="openai/gpt-5.6-sol", prompt_path="/tmp/prompt.md")
    assert "--provider openai" in command
    assert "--model gpt-5.6-sol" in command


def test_pier_command_injects_only_required_provider_keys_and_stage_config(tmp_path):
    command = build_pier_command(
        pier_executable="pier",
        checkout=tmp_path / "checkout",
        manifest=_manifest(tmp_path),
        agent_id="pi-v5-review-repair",
        jobs_dir=tmp_path / "jobs",
        job_name="v5-models",
        stage_model_config={
            "solver": {"model": "deepseek-v4-flash"},
            "A": {"model": "openai/gpt-5.6-sol"},
        },
    )
    assert "DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY}" in command
    assert "OPENAI_API_KEY=${OPENAI_API_KEY}" in command
    config_arg = command[command.index("--agent-kwarg") + 1]
    assert config_arg.startswith("stage_model_config=")
    assert "gpt-5.6-sol" in config_arg


def test_pier_command_injects_only_zai_key_for_all_glm_stages(tmp_path):
    command = build_pier_command(
        pier_executable="pier",
        checkout=tmp_path / "checkout",
        manifest=_manifest(tmp_path),
        agent_id="pi-v6-review-repair",
        jobs_dir=tmp_path / "jobs",
        job_name="v6-glm-5.3",
        model="zai/glm-5.3",
        stage_model_config={"default": {"model": "zai/glm-5.3"}},
    )

    assert "ZAI_API_KEY=${ZAI_API_KEY}" in command
    assert not any("DEEPSEEK_API_KEY=" in part for part in command)
    assert not any("OPENAI_API_KEY=" in part for part in command)
    config_arg = command[command.index("--agent-kwarg") + 1]
    assert '"provider":"zai"' in config_arg
    assert '"model":"glm-5.3"' in config_arg
