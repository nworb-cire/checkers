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
        return self.game_board.game_over

    def get_winner(self):
        if self.is_game_over():
            return self.current_player_str()
        return None

    def current_player_str(self):
        return "red" if self.current_player() == Player.RED else "black"

    def score_str(self):
        return f"Red: {self.game_board.scores[Player.RED]} Black: {self.game_board.scores[Player.BLACK]}"

    def count_pieces(self, player: Player) -> tuple[int, int]:
        return np.sum(self.game_board.board.board == player), np.sum(
            self.game_board.board.board == 2 * player
        )

    def take_turn(self, move: Move):
        switch_turns = self.game_board.make_move(move)
        if switch_turns:
            self.game_board.switch_player()

    def on_square_click(self, x, y):
        if (
            self.selected_square is None
            and np.sign(self.game_board.board[x, y]) == self.current_player()
        ):
            self.selected_square = (x, y)
        elif self.selected_square is not None:
            self.take_turn(Move(self.selected_square, (x, y)))
            self.selected_square = None
        else:
            self.selected_square = None
