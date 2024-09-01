import numpy as np
import pytest

from src.game.board import GameBoard, Player, Move, BoardState


@pytest.fixture(scope="module")
def game_board():
    return GameBoard()


def test_board_setup(game_board):
    assert game_board.current_player == Player.RED
    assert np.allclose(game_board.board.board, BoardState().board)


def test_only_even_squares(game_board):
    for i in range(8):
        for j in range(8):
            if (i + j) % 2 == 1:
                assert game_board.board.board[i, j] == 0
