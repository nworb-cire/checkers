from src.game.board import Move


def action_to_move(action) -> Move:
    # unpack into ((from_x, from_y), (to_x, to_y))
    start, end = action // 32, action % 32
    # go from (start) play square to (end) play square
    start_x, start_y = start // 4, start % 4
    end_x, end_y = end // 4, end % 4
    # parity check, should be even
    start_y = start_y * 2 + (start_x % 2)
    end_y = end_y * 2 + (end_x % 2)
    return Move((start_x, start_y), (end_x, end_y))


MOVES = {action: action_to_move(action) for action in range(32 * 32)}
