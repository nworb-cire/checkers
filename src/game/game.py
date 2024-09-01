import numpy as np

from src.game.board import GameBoard, Player, Move


class Game:
    def __init__(self):
        self.game_board = GameBoard()
        self.selected_square = None
        self.game_over = False

    def current_player(self):
        return self.game_board.current_player

    def current_player_str(self):
        return "red" if self.current_player() == Player.RED else "black"

    def count_pieces(self, player: Player) -> tuple[int, int]:
        return np.sum(self.game_board.board.board == player), np.sum(self.game_board.board.board == 2 * player)

    def on_square_click(self, x, y):
        if self.selected_square is None and np.sign(self.game_board.board[x, y]) == self.current_player():
            self.selected_square = (x, y)
        elif self.selected_square is not None:
            try:
                self.game_board.make_move(Move(self.selected_square, (x, y)))
            except ValueError:
                print(f"Invalid move: {self.selected_square} -> {(x, y)}")
            self.selected_square = None
        else:
            self.selected_square = None
