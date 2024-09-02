from src.game.player import Player
from src.game.moves import Move


class InvalidMoveError(Exception):
    def __init__(self, move: Move):
        self.move = move
        super().__init__(f"Invalid move: {move}")


class GameOver(Exception):
    def __init__(self, winner: Player | None = None):
        self.winner = winner
        if winner is None:
            super().__init__("Game over!")
        else:
            super().__init__(f"Game over! {winner} wins!")


class Stalemate(GameOver):
    pass


class UnableToMoveError(GameOver):
    pass
