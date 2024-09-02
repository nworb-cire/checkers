import pytest

from src.ai.self_play import action_to_move


@pytest.mark.parametrize("action", range(32 * 32))
def test_action_to_move(action):
    action_to_move(action)
