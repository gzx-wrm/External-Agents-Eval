import pytest

from external_agents_eval.agents.hermes_entry import (
    PRACTICALLY_UNLIMITED_MAX_ITERATIONS,
    resolve_max_iterations,
)


def test_omitted_turn_limit_maps_to_practically_unlimited_iterations():
    assert resolve_max_iterations(None) == PRACTICALLY_UNLIMITED_MAX_ITERATIONS


def test_explicit_turn_limit_is_preserved():
    assert resolve_max_iterations(12) == 12
    assert resolve_max_iterations("12") == 12


@pytest.mark.parametrize("value", [0, -1, "0", "-1"])
def test_non_positive_turn_limit_is_rejected(value):
    with pytest.raises(ValueError, match="max_turns must be positive"):
        resolve_max_iterations(value)
