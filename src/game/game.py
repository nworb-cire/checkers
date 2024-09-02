import contextlib

import numpy as np

from src.game.board import GameBoard, Player, Move


class Game:
    def __init__(self):
        self.game_board = GameBoard()
        self.selected_square = None

    def current_player(self):
        return self.game_board.current_player

    def is_game_over(self):
        red, red_jump = self.game_board.get_available_moves(Player.RED)
        black, black_jump = self.game_board.get_available_moves(Player.BLACK)
        return (len(red) == 0 and len(red_jump) == 0) or (len(black) == 0 and len(black_jump) == 0)

    def get_winner(self):
        if self.is_game_over():
            return "red" if self.game_board.current_player == Player.BLACK else "black"
        return None

    def current_player_str(self):
        return "red" if self.current_player() == Player.RED else "black"

    def count_pieces(self, player: Player) -> tuple[int, int]:
        return np.sum(self.game_board.board.board == player), np.sum(self.game_board.board.board == 2 * player)

    def on_square_click(self, x, y):
        if self.selected_square is None and np.sign(self.game_board.board[x, y]) == self.current_player():
            self.selected_square = (x, y)
        elif self.selected_square is not None:
            with contextlib.suppress(ValueError):
                self.game_board.make_move(Move(self.selected_square, (x, y)))
            self.selected_square = None
        else:
            self.selected_square = None
