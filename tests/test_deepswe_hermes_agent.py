from external_agents_eval.agents.pier.hermes import HermesPierAgent


def test_hermes_install_uses_pinned_uv_compatible_with_upstream_lockfile():
    agent = object.__new__(HermesPierAgent)
    install_spec = agent.install_spec()
    install_script = "\n".join(step.run for step in install_spec.steps)

    assert "/uv/0.9.28/install.sh" in install_script
    assert 'uv_bin=/usr/local/bin/uv' in install_script
    assert '"$uv_bin" --version' in install_script
    assert '"$uv_bin" sync --locked --no-dev' in install_script
    assert "uv-0.9.28" in install_spec.cache_key
