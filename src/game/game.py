import contextlib
import time

import numpy as np

from src.ai.ai import CheckersAI
from src.game.board import GameBoard
from src.game.player import Player
from src.game.errors import InvalidMoveError, GameOver
from src.game.moves import Move


class Game:
    def __init__(self):
        self.game_board = GameBoard()
        self.from_square = None
        self.to_square = None
        self.winner = None

    def current_player(self):
        return self.game_board.current_player

    def is_game_over(self):
        return self.game_board.game_over

    def current_player_str(self):
        return "red" if self.current_player() == Player.RED else "black"

    def score_str(self):
        return f"Red: {self.game_board.scores[Player.RED]} Black: {self.game_board.scores[Player.BLACK]}"

    def count_pieces(self, player: Player) -> tuple[int, int]:
        return np.sum(self.game_board.board.board == player), np.sum(
            self.game_board.board.board == 2 * player
        )

    def take_turn(self, move: Move):
        try:
            switch_turns = self.game_board.make_move(move)
            if switch_turns:
                self.game_board.switch_player()
        except GameOver as e:
            self.winner = e.winner
            raise e

    def on_square_click(self, x, y):
        if self.from_square is None:
            self.from_square = (x, y)
        else:
            self.to_square = (x, y)

    def human_turn(self):
        if self.from_square is not None and self.to_square is not None:
            move = Move(self.from_square, self.to_square)
            with contextlib.suppress(InvalidMoveError):
                self.take_turn(move)
            self.from_square = None
            self.to_square = None

    def tick(self):
        self.human_turn()


class AIGame(Game):
    def __init__(self, ai: CheckersAI | None = None):
        super().__init__()
        self.human_player = Player.RED
        if ai is None:
            ai = CheckersAI.init(-self.human_player, "models/ppo.pt")
        self.ai = ai

    def ai_turn(self):
        time.sleep(0.25 + np.random.rand() * 0.5)
        move, _ = self.ai.select_action(self.game_board)
        super().take_turn(move)

    def tick(self):
        if self.current_player() == self.human_player:
            self.human_turn()
        else:
            self.ai_turn()
