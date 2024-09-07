import numpy as np
import torch

from src.ai.actions import MOVES
from src.game.errors import InvalidMoveError, Stalemate, UnableToMoveError
from src.game.moves import Move
from src.game.player import Player
from src.game.scores import Score
from src.game.settings import MUST_JUMP


class BoardState:
    """
    Represents a state of the board, with some functions for getting available moves. The board is represented as
    an 8x8 numpy array, +1 representing a red piece, +2 representing a red king, -1 representing a black piece, and -2
    representing a black king. 0 represents an empty space.
    """

    board: np.ndarray

    def __init__(self, board: np.ndarray | None = None):
        """
        Initialize the board state. If no board is provided, the default starting board is used.
        :param board: An 8x8 numpy array representing the board state.
        """
        if board is None:
            board = self.setup_board()
        assert board.shape == (8, 8)
        self.board = board
        self.restrict_moves = None

    def __getitem__(self, key):
        return self.board[key]

    def __setitem__(self, key, value):
        """
        Place or remove a piece from the board. The key is a tuple representing the row and column of the piece, and the
        value must be one of -2, -1, 0, 1, or 2.
        :param key: Tuple representing the row and column of the piece.
        :param value: Integer representing the piece to place or remove. Must be one of -2, -1, 0, 1, or 2.
        """
        assert value in (-2, -1, 0, 1, 2)
        assert 0 <= key[0] < 8
        assert 0 <= key[1] < 8
        assert sum(key) % 2 == 0  # only even squares
        self.board[key] = value

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.board)})"

    def flip(self):
        """
        Flip the board, effectively rotating it 180 degrees and reversing the players (red becomes black and vice versa).
        This is used to simplify AI implementations, so that the AI can always play as red.
        :return: A new BoardState object representing the flipped board.
        """
        return BoardState(-self.board[::-1, ::-1])

    def visualize(self):
        """
        Visualize the board as a string, with red pieces represented by "r" and black pieces represented by "b".
        Kings are represented by captial letters. This is useful for debugging or for a CLI based game.
        :return: A string representation of the board.
        """
        s = ""
        for row in range(8):
            for col in range(8):
                match self.board[row, col]:
                    case 0:
                        s += " "
                    case 1:
                        s += "r"
                    case -1:
                        s += "b"
                    case 2:
                        s += "R"
                    case -2:
                        s += "B"
            s += "\n"
        return s

    def __eq__(self, other):
        """
        Check if two board states are equal. This is used for unit testing.
        :param other: A BoardState object to compare to.
        :return: True if the board states are equal, False otherwise.
        """
        if not isinstance(other, BoardState):
            return False
        return np.allclose(self.board, other.board)

    def get_available_moves(self, player: Player) -> tuple[list[Move], list[Move]]:
        """
        Get all available moves for a player. Returns a tuple of two lists, the first containing regular moves and the
        second containing jump moves. If the game settings require that a player must jump if possible, only jump moves
        will be returned if available.
        :param player: The player to get moves for.
        :return: A tuple of (moves, jump_moves).
        """
        moves = []
        jump_moves = []
        for row in range(8):
            for col in range(8):
                if np.sign(self.board[row, col]) == player:
                    moves.extend(self.get_moves(row, col))
                    jump_moves.extend(self.get_jump_moves(row, col))
        if jump_moves and MUST_JUMP:
            moves = []
        return moves, jump_moves

    @staticmethod
    def max_row_for_player(player: Player):
        return 7 if player == Player.RED else 0

    @staticmethod
    def second_last_row_for_player(player: Player):
        return 6 if player == Player.RED else 1

    def get_moves(self, row: int, col: int):
        """
        Get all available standard (non-jump) moves for the piece at the given row and column.
        :param row: The row of the piece.
        :param col: The column of the piece.
        :return: A list of Move objects representing the available moves.
        """
        moves = []
        if self.board[row, col] == 0:
            return moves
        player = Player(np.sign(self.board[row, col]))
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

    def get_jump_moves(self, row: int, col: int):
        """
        Get all available jump moves for the piece at the given row and column.
        :param row: The row of the piece.
        :param col: The column of the piece.
        :return: A list of Move objects representing the available jump moves.
        """
        moves = []
        if self.board[row, col] == 0:
            return moves
        player = Player(np.sign(self.board[row, col]))
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

    def get_moves_mask(self, player: Player):
        """
        Get a boolean mask representing the available moves for a player. This is used to mask out invalid moves for AI.
        :param player: The player to get moves for.
        :return: A list of booleans representing the available moves. This has the same shape as the MOVES dictionary.
        """
        moves, jump_moves = self.get_available_moves(player)
        mask = torch.tensor(
            [move in moves + jump_moves for move in MOVES.values()], dtype=torch.bool
        ).unsqueeze(0)
        return mask

    def is_able_to_move(self, player: Player) -> bool:
        """
        Check if a player is able to move. This is used to determine if the game is over.
        :param player: The player to check.
        :return: True if the player has valid moves, False otherwise.
        """
        moves, jump_moves = self.get_available_moves(player)
        return len(moves) + len(jump_moves) > 0

    def is_stalemate(self) -> bool:
        """
        Check if the game is in a stalemate, meaning that neither player has any valid moves.
        :return: True if the game is in a stalemate, False otherwise.
        """
        return not self.is_able_to_move(Player.RED) and not self.is_able_to_move(
            Player.BLACK
        )

    def is_game_over(self) -> bool:
        """
        Check if the game is over, meaning that at least one player has no valid moves.
        :return: True if the game is over, False otherwise.
        """
        return not self.is_able_to_move(Player.RED) or not self.is_able_to_move(
            Player.BLACK
        )

    @staticmethod
    def setup_board():
        """
        Create the default starting board state.
        :return: A np.ndarray representing the default starting board state.
        """
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

    def to_tensor(self):
        """
        Convert the board state to a tensor for use with PyTorch models.
        :return: A tensor of shape (1, 64) representing the board state.
        """
        return torch.tensor(self.board.flatten(), dtype=torch.float32).unsqueeze(0)


class GameBoard:
    def __init__(self, board: BoardState | None = None):
        if board is None:
            board = BoardState()
        self.board = board
        self.current_player = Player.RED
        self.game_over = False
        self.scores = {Player.RED: 0, Player.BLACK: 0}
        self.turn_number = 0

    @property
    def restrict_moves(self):
        return self.board.restrict_moves

    def switch_player(self):
        self.current_player = -self.current_player

    def make_move(self, move: Move):
        """
        Make a move on the board. Returns True if the turn is complete and should switch players, False if the current
        player has another move available (i.e. a jump move) or has won the game.
        :param move:
        :return: True if the turn is complete, False otherwise
        """
        moves, jump_moves = self.board.get_available_moves(self.current_player)
        if (move not in moves and move not in jump_moves) or (
            self.restrict_moves and move.start != self.restrict_moves
        ):
            raise InvalidMoveError(move)

        self.turn_number += 1
        self.board[move.end] = self.board[move.start]
        self.board[move.start] = 0
        if move in jump_moves:
            jumped_space = (move.start[0] + move.end[0]) // 2, (
                move.start[1] + move.end[1]
            ) // 2
            jumped_king = np.abs(self.board[jumped_space]) == 2
            if jumped_king:
                self.scores[self.current_player] += Score.KING_CAPTURE
                self.scores[-self.current_player] -= Score.KING_CAPTURE
            else:
                self.scores[self.current_player] += Score.REGULAR_CAPTURE
                self.scores[-self.current_player] -= Score.REGULAR_CAPTURE
            self.board[jumped_space] = 0
            jump_moves = self.board.get_jump_moves(*move.end)
            if jump_moves:
                self.board.restrict_moves = move.end
                return
        # king me
        if (
            move.end[0] == self.board.max_row_for_player(self.current_player)
            and np.abs(self.board[move.end]) == 1
        ):
            self.board[move.end] *= 2
            self.scores[self.current_player] += Score.KING

        # end game if other player has no moves
        if self.board.is_stalemate():
            raise Stalemate()

        if not self.board.is_able_to_move(-self.current_player):
            self.scores[self.current_player] += Score.WIN
            self.scores[-self.current_player] -= Score.WIN
            raise UnableToMoveError(winner=self.current_player)

        self.board.restrict_moves = None
        self.switch_player()
