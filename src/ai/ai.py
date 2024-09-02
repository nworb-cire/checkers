import torch
from torch import nn
from torch.distributions import Categorical

from src.ai.actions import MOVES
from src.game.board import GameBoard, Player


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

    @torch.no_grad()
    def select_action(self, game_board: GameBoard):
        is_p2 = game_board.current_player == Player.BLACK
        board = game_board.board
        if is_p2:
            # Model expects to be playing as player 1 (red).
            # If the AI is playing as player 2 (black), the board needs to be flipped.
            board = board.flip()
        state = torch.tensor(board.board, dtype=torch.float32).flatten()
        mask = torch.tensor(game_board.get_moves_mask(), dtype=torch.bool)
        action_probs, _ = self.policy_model(state, mask)
        dist = Categorical(action_probs)
        action = dist.sample()
        move = MOVES[action.item()]
        if is_p2:
            move = move.flip()
        return move, dist.log_prob(action).item()

    @classmethod
    def init(cls, player: Player):
        policy_model = nn.Sequential(
            nn.Linear(8 * 8, 32 * 32),
        )
        value_model = nn.Sequential(
            nn.Linear(8 * 8, 1),
        )
        return cls(player, policy_model, value_model)
