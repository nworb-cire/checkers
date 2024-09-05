import numpy as np

from src.game.board import BoardState
from src.game.player import Player
from tests.board.conftest import game_board


def test_board_setup(game_board):
    assert game_board.current_player == Player.RED
    assert np.allclose(game_board.board.board, BoardState().board)


def test_only_even_squares(game_board):
    for i in range(8):
        for j in range(8):
            if (i + j) % 2 == 1:
                assert game_board.board.board[i, j] == 0


def test_flip_init(game_board):
    assert game_board.board.flip() == game_board.board


def test_flip_single():
    arr = np.zeros((8, 8))
    arr[0, 0] = 1
    board = BoardState(arr)
    flipped = board.flip()
    assert flipped.board[7, 7] == -1


def test_double_flip(game_board):
    board = game_board.board
    flipped = board.flip().flip()
    assert flipped == board
