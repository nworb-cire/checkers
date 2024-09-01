import enum

import numpy as np


class Player(enum.IntEnum):
    RED = 1
    BLACK = -1


class Move:
    def __init__(self, start: tuple[int, int], end: tuple[int, int]):
        assert len(start) == 2
        assert len(end) == 2
        assert 0 <= start[0] < 8
        assert 0 <= start[1] < 8
        assert 0 <= end[0] < 8
        assert 0 <= end[1] < 8
        self.start = start
        self.end = end

    def __str__(self):
        return f"Move from {self.start} to {self.end}"

    def __eq__(self, other):
        return self.start == other.start and self.end == other.end

    def __repr__(self):
        return f"{self.__class__.__name__}({self.start} -> {self.end})"


class BoardState:
    board: np.ndarray

    def __init__(self, board: np.ndarray | None = None):
        if board is None:
            board = self.setup_board()
        assert board.shape == (8, 8)
        self.board = board

    def __getitem__(self, key):
        return self.board[key]

    @staticmethod
    def setup_board():
        board = np.array([
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, -1, 0, -1, 0, -1, 0, -1],
            [-1, 0, -1, 0, -1, 0, -1, 0],
            [0, -1, 0, -1, 0, -1, 0, -1]
        ])
        return board


class GameBoard:
    def __init__(self, board: BoardState | None = None):
        if board is None:
            board = BoardState()
        self.board = board
        self.current_player = Player.RED

    def get_board(self):
        return self.board

    def get_current_player(self):
        return self.current_player

    def switch_player(self):
        self.current_player = -self.current_player

    def get_available_moves(self, player: Player):
        moves = []
        jump_moves = []
        for row in range(8):
            for col in range(8):
                if np.sign(self.board[row, col]) == player:
                    moves.extend(self.get_moves(row, col, player))
                    # jump_moves.extend(self.get_jump_moves(row, col, player))
        return moves + jump_moves

    @staticmethod
    def max_row_for_player(player: Player):
        return 7 if player == Player.RED else 0

    def get_moves(self, row: int, col: int, player: Player):
        moves = []
        if self.board[row, col] == 0:
            return moves
        if col > 0 and row != self.max_row_for_player(player) and self.board[row - player, col - 1] == 0:
            moves.append(Move((row, col), (row - player, col - 1)))
        if col < 7 and row != self.max_row_for_player(player) and self.board[row - player, col + 1] == 0:
            moves.append(Move((row, col), (row - player, col + 1)))
        if np.abs(self.board[row, col]) == 2:
            if col > 0 and row != self.max_row_for_player(-player) and self.board[row + player, col - 1] == 0:
                moves.append(Move((row, col), (row + player, col - 1)))
            if col < 7 and row != self.max_row_for_player(-player) and self.board[row + player, col + 1] == 0:
                moves.append(Move((row, col), (row + player, col + 1)))
        return moves
