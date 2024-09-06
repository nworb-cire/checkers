import numpy as np
import pytest

from src.game.board import GameBoard, BoardState
from src.game.errors import InvalidMoveError
from src.game.moves import Move
from src.game.player import Player


def test_jump():
    arr = np.zeros((8, 8))
    arr[0, 0] = 1
    arr[1, 1] = -1
    board = GameBoard(BoardState(arr))
    moves = board.board.get_jump_moves(0, 0)
    assert moves == [Move((0, 0), (2, 2))]
    moves = board.board.get_jump_moves(1, 1)
    assert moves == []


def test_jump_blocked():
    arr = np.zeros((8, 8))
    arr[0, 0] = 1
    arr[1, 1] = -1
    arr[2, 2] = 1
    board = GameBoard(BoardState(arr))
    moves = board.board.get_jump_moves(0, 0)
    assert moves == []
    moves = board.board.get_jump_moves(1, 1)
    assert moves == []


def test_turn_with_jump(game_board):
    game_board.board = BoardState(
        np.array(
            [
                [1, 0, 1, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0, 1, 0, 1],
                [0, 0, 1, 0, 1, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, -1, 0, 0, 0, 0, 0],
                [0, 0, 0, -1, 0, -1, 0, -1],
                [-1, 0, -1, 0, -1, 0, -1, 0],
                [0, -1, 0, -1, 0, -1, 0, -1],
            ]
        )
    )
    game_board.current_player = Player.BLACK
    game_board.make_move(Move((4, 2), (2, 0)))
    assert game_board.current_player == Player.RED
    expected = np.array(
        [
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
            [-1, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, -1, 0, -1, 0, -1],
            [-1, 0, -1, 0, -1, 0, -1, 0],
            [0, -1, 0, -1, 0, -1, 0, -1],
        ]
    )
    assert np.allclose(game_board.board.board, expected)


def test_turn_with_multiple_jumps(game_board):
    game_board.board = BoardState(
        np.array(
            [
                [1, 0, 0, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0, 1, 0, 1],
                [0, 0, 1, 0, 1, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, -1, 0, 0, 0, 0, 0],
                [0, 0, 0, -1, 0, -1, 0, -1],
                [-1, 0, -1, 0, -1, 0, -1, 0],
                [0, -1, 0, -1, 0, -1, 0, -1],
            ]
        )
    )
    game_board.current_player = Player.BLACK
    game_board.make_move(Move((4, 2), (2, 0)))
    assert game_board.current_player == Player.BLACK


def test_double_jump():
    game_board = GameBoard(
        BoardState(
            np.array(
                [
                    [1, 0, 0, 0, 0, 0, 0, 0],
                    [0, -1, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0],
                    [0, -1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, -1],
                ]
            )
        )
    )
    game_board.current_player = Player.RED
    game_board.make_move(Move((0, 0), (2, 2)))
    assert game_board.current_player == Player.RED
    assert game_board.restrict_moves == (2, 2)
    with pytest.raises(InvalidMoveError):
        game_board.make_move(Move((2, 0), (4, 2)))
    game_board.make_move(Move((2, 2), (4, 0)))
    assert game_board.current_player == Player.BLACK
