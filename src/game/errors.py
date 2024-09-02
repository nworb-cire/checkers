from src.game.moves import Move


class InvalidMoveError(Exception):
    def __init__(self, move: Move):
        self.move = move
        super().__init__(f"Invalid move: {move}")
