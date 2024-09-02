import torch
from torch import nn

from src.ai.actions import MOVES
from src.game.board import GameBoard, Player, Move


class CheckersAI:
    def __init__(
        self,
        player: Player,
        policy_model: nn.Module,
        value_model: nn.Module,
    ):
        self.player = player
        self.policy_model = policy_model
        self.value_model = value_model

    def select_action(self, game_board: GameBoard) -> tuple[Move, torch.Tensor]:
        board = game_board.board
        if self.player == Player.BLACK:
            # Model expects to be playing as player 1 (red).
            # If the AI is playing as player 2 (black), the board needs to be flipped.
            board = board.flip()
        board = torch.tensor(board.board, dtype=torch.float32).flatten()

        logits = self.policy_model(board)
        moves, jumps = game_board.get_available_moves(Player.RED)
        valid_actions = torch.tensor([move in moves + jumps for move in MOVES.values()])
        logits[~valid_actions] = float("-inf")
        probabilities = torch.softmax(logits, dim=-1)
        move = torch.multinomial(probabilities, 1).item()
        move = MOVES[move]
        if self.player == Player.BLACK:
            move = move.flip()
        return move, probabilities

    @classmethod
    def init(cls, player: Player):
        policy_model = nn.Sequential(
            nn.Linear(8 * 8, 32 * 32),
        )
        value_model = nn.Sequential(
            nn.Linear(8 * 8, 1),
        )
        return cls(player, policy_model, value_model)
