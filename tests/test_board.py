import numpy as np
import pytest

from src.game.board import GameBoard, Player, BoardState
from src.game.moves import Move


@pytest.fixture(scope="function")
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
    moves = board.get_moves(row, col, player)
    assert moves == expected


def test_jump():
    arr = np.zeros((8, 8))
    arr[0, 0] = 1
    arr[1, 1] = -1
    board = GameBoard(BoardState(arr))
    moves = board.get_jump_moves(0, 0, Player.RED)
    assert moves == [Move((0, 0), (2, 2))]
    moves = board.get_jump_moves(1, 1, Player.BLACK)
    assert moves == []


def test_jump_blocked():
    arr = np.zeros((8, 8))
    arr[0, 0] = 1
    arr[1, 1] = -1
    arr[2, 2] = 1
    board = GameBoard(BoardState(arr))
    moves = board.get_jump_moves(0, 0, Player.RED)
    assert moves == []
    moves = board.get_jump_moves(1, 1, Player.BLACK)
    assert moves == []


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


def test_king_me():
    arr = np.zeros((8, 8))
    arr[6, 2] = 1
    board = GameBoard(BoardState(arr))
    board.make_move(Move((6, 2), (7, 1)))
    assert board.board.board[7, 1] == 2


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
