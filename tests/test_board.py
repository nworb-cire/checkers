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


@pytest.mark.parametrize("row, col, player, king, expected", [
    # Red player
    # bottom left corner
    (0, 0, Player.RED, False, [Move((0, 0), (1, 1))]),
    # bottom right corner
    (0, 6, Player.RED, False, [Move((0, 6), (1, 5)), Move((0, 6), (1, 7))]),
    # top left corner
    (7, 1, Player.RED, False, []),
    # top right corner
    (7, 7, Player.RED, False, []),
    # top left corner, king
    (7, 1, Player.RED, True, [Move((7, 1), (6, 0)), Move((7, 1), (6, 2))]),
    # top right corner, king
    (7, 7, Player.RED, True, [Move((7, 7), (6, 6))]),
])
def test_moves_at_extremities(row: int, col: int, player: Player, king: bool, expected: list):
    arr = np.zeros((8, 8))
    arr[row, col] = 1 if player == Player.RED else -1
    if king:
        arr[row, col] *= 2
    board = GameBoard(BoardState(arr))
    moves = board.get_moves(row, col, player)
    assert moves == expected
