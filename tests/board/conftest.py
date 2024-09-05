import pytest

from src.game.board import GameBoard


@pytest.fixture(scope="function")
def game_board():
    return GameBoard()
