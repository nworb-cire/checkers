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

    def flip(self):
        return Move(
            (7 - self.start[0], 7 - self.start[1]), (7 - self.end[0], 7 - self.end[1])
        )
