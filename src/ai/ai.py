import torch
from torch.distributions import Categorical

from src.ai.actions import MOVES
from src.ai.policy import PolicyNetwork
from src.game.board import GameBoard, Player


class CheckersAI:
    def __init__(
        self,
        player: Player,
        policy_model: PolicyNetwork,
    ):
        self.player = player
        self.policy = policy_model

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
        action_probs, _ = self.policy(state, mask)
        dist = Categorical(action_probs)
        action = dist.sample()
        move = MOVES[action.item()]
        if is_p2:
            move = move.flip()
        return move, dist.log_prob(action).item()

    @classmethod
    def init(cls, player: Player, path: str | None = None):
        policy_model = PolicyNetwork()
        if path is not None:
            policy_model.load_state_dict(torch.load(path))
        return cls(player, policy_model)
