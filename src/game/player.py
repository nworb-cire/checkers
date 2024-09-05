import enum


class Player(enum.IntEnum):
    RED = 1
    BLACK = -1

    def __str__(self):
        return "Red" if self == Player.RED else "Black"
