import contextlib

import numpy as np
import pytest

from src.game.board import GameBoard, BoardState
from src.game.errors import UnableToMoveError
from src.game.moves import Move
from src.game.player import Player


@pytest.mark.parametrize(
    "row, col, player, king, expected",
    [
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
        # Black player
        # top left corner
        (7, 1, Player.BLACK, False, [Move((7, 1), (6, 0)), Move((7, 1), (6, 2))]),
        # top right corner
        (7, 7, Player.BLACK, False, [Move((7, 7), (6, 6))]),
        # bottom left corner
        (0, 0, Player.BLACK, False, []),
        # bottom right corner
        (0, 6, Player.BLACK, False, []),
        # bottom left corner, king
        (0, 0, Player.BLACK, True, [Move((0, 0), (1, 1))]),
        # bottom right corner, king
        (0, 6, Player.BLACK, True, [Move((0, 6), (1, 5)), Move((0, 6), (1, 7))]),
    ],
)
def test_moves_at_extremities(
    row: int, col: int, player: Player, king: bool, expected: list
):
    arr = np.zeros((8, 8))
    arr[row, col] = 1 if player == Player.RED else -1
    if king:
        arr[row, col] *= 2
    board = GameBoard(BoardState(arr))
    moves = board.board.get_moves(row, col, player)
    assert moves == expected


def test_first_turn(game_board):
    assert game_board.current_player == Player.RED
    game_board.make_move(Move((2, 0), (3, 1)))
    assert game_board.current_player == Player.BLACK
    expected = np.array(
        [
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
            [0, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, -1, 0, -1, 0, -1, 0, -1],
            [-1, 0, -1, 0, -1, 0, -1, 0],
            [0, -1, 0, -1, 0, -1, 0, -1],
        ]
    )
    assert np.allclose(game_board.board.board, expected)


def test_king_me():
    arr = np.zeros((8, 8))
    arr[6, 2] = 1
    board = GameBoard(BoardState(arr))
    with contextlib.suppress(UnableToMoveError):
        board.make_move(Move((6, 2), (7, 1)))
    assert board.board.board[7, 1] == 2
