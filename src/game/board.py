import enum

import numpy as np

from src.game.scores import Score


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

    def __setitem__(self, key, value):
        assert value in (-2, -1, 0, 1, 2)
        assert 0 <= key[0] < 8
        assert 0 <= key[1] < 8
        self.board[key] = value

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.board)})"

    def flip(self):
        return BoardState(-self.board[::-1, ::-1])

    def __eq__(self, other):
        return np.allclose(self.board, other.board)

    @staticmethod
    def setup_board():
        board = np.array(
            [
                [1, 0, 1, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0, 1, 0, 1],
                [1, 0, 1, 0, 1, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, -1, 0, -1, 0, -1, 0, -1],
                [-1, 0, -1, 0, -1, 0, -1, 0],
                [0, -1, 0, -1, 0, -1, 0, -1],
            ]
        )
        return board


class GameBoard:
    def __init__(self, board: BoardState | None = None):
        if board is None:
            board = BoardState()
        self.board = board
        self.current_player = Player.RED
        self.game_over = False
        self.scores = {Player.RED: 0, Player.BLACK: 0}

    def switch_player(self):
        self.current_player = -self.current_player

    def get_available_moves(self, player: Player) -> tuple[list[Move], list[Move]]:
        moves = []
        jump_moves = []
        for row in range(8):
            for col in range(8):
                if np.sign(self.board[row, col]) == player:
                    moves.extend(self.get_moves(row, col, player))
                    jump_moves.extend(self.get_jump_moves(row, col, player))
        return moves, jump_moves

    @staticmethod
    def max_row_for_player(player: Player):
        return 7 if player == Player.RED else 0

    @staticmethod
    def second_last_row_for_player(player: Player):
        return 6 if player == Player.RED else 1

    def get_moves(self, row: int, col: int, player: Player):
        moves = []
        if self.board[row, col] == 0:
            return moves
        if (
            col > 0
            and row != self.max_row_for_player(player)
            and self.board[row + player, col - 1] == 0
        ):
            moves.append(Move((row, col), (row + player, col - 1)))
        if (
            col < 7
            and row != self.max_row_for_player(player)
            and self.board[row + player, col + 1] == 0
        ):
            moves.append(Move((row, col), (row + player, col + 1)))
        if np.abs(self.board[row, col]) == 2:
            if (
                col > 0
                and row != self.max_row_for_player(-player)
                and self.board[row - player, col - 1] == 0
            ):
                moves.append(Move((row, col), (row - player, col - 1)))
            if (
                col < 7
                and row != self.max_row_for_player(-player)
                and self.board[row - player, col + 1] == 0
            ):
                moves.append(Move((row, col), (row - player, col + 1)))
        return moves

    def get_jump_moves(self, row: int, col: int, player: Player):
        moves = []
        if self.board[row, col] == 0:
            return moves
        if (
            col > 1
            and row
            not in (
                self.max_row_for_player(player),
                self.second_last_row_for_player(player),
            )
            and np.sign(self.board[row + player, col - 1]) == -player
            and self.board[row + 2 * player, col - 2] == 0
        ):
            moves.append(Move((row, col), (row + 2 * player, col - 2)))
        if (
            col < 6
            and row
            not in (
                self.max_row_for_player(player),
                self.second_last_row_for_player(player),
            )
            and np.sign(self.board[row + player, col + 1]) == -player
            and self.board[row + 2 * player, col + 2] == 0
        ):
            moves.append(Move((row, col), (row + 2 * player, col + 2)))
        if np.abs(self.board[row, col]) == 2:
            if (
                col > 1
                and row
                not in (
                    self.max_row_for_player(-player),
                    self.second_last_row_for_player(-player),
                )
                and np.sign(self.board[row - player, col - 1]) == -player
                and self.board[row - 2 * player, col - 2] == 0
            ):
                moves.append(Move((row, col), (row - 2 * player, col - 2)))
            if (
                col < 6
                and row
                not in (
                    self.max_row_for_player(-player),
                    self.second_last_row_for_player(-player),
                )
                and np.sign(self.board[row - player, col + 1]) == -player
                and self.board[row - 2 * player, col + 2] == 0
            ):
                moves.append(Move((row, col), (row - 2 * player, col + 2)))
        return moves

    def make_move(self, move: Move):
        moves, jump_moves = self.get_available_moves(self.current_player)
        if move not in moves and move not in jump_moves:
            raise ValueError("Invalid move")
        self.board[move.end] = self.board[move.start]
        self.board[move.start] = 0
        if move in jump_moves:
            jumped_space = (move.start[0] + move.end[0]) // 2, (
                move.start[1] + move.end[1]
            ) // 2
            jumped_king = np.abs(self.board[jumped_space]) == 2
            if jumped_king:
                self.scores[self.current_player] += Score.KING_CAPTURE
            else:
                self.scores[self.current_player] += Score.REGULAR_CAPTURE
            self.board[jumped_space] = 0
            jump_moves = self.get_jump_moves(
                move.end[0], move.end[1], self.current_player
            )
            if jump_moves:
                return
        # king me
        if (
            move.end[0] == self.max_row_for_player(self.current_player)
            and np.abs(self.board[move.end]) == 1
        ):
            self.board[move.end] *= 2
            self.scores[self.current_player] += Score.KING

        # end game if other player has no moves
        other, other_jump = self.get_available_moves(-self.current_player)
        if len(other) == 0 and len(other_jump) == 0:
            self.scores[self.current_player] += Score.WIN
            self.game_over = True
            return

        self.switch_player()
