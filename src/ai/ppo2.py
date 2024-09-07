import pytorch_lightning as pl
import torch
from torch import nn as nn
from torch.distributions import Categorical

from src.game.board import BoardState
from src.game.board import GameBoard
from src.game.errors import GameOver
from src.game.player import Player


class ResidualBlock(nn.Module):
    def __init__(self, d_in: int, d_hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_in)

    def forward(self, x):
        x_in = x
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x_in + x


class PolicyNetwork(nn.Module):
    """
    Policy network for PPO algorithm.
    """

    def __init__(
        self,
        state_dim: int = 8 * 8,
        action_dim: int = 32 * 32,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.resnet = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            ResidualBlock(hidden_dim, hidden_dim),
            ResidualBlock(hidden_dim, hidden_dim),
        )
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(
        self, state: BoardState, mask=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = state.to_tensor()
        x = self.resnet(x)
        value = self.value_head(x)

        logits = self.action_head(x)
        if mask is not None:
            logits[~mask] = -float("inf")
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class CheckersPPOAgent(pl.LightningModule):
    def __init__(
        self,
        lr=3e-4,
        gamma=0.99,
        eps_clip=0.2,
        K_epochs=4,
        games_per_batch=32,
        max_game_length=200,
    ):
        super().__init__()
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.games_per_batch = games_per_batch
        self.max_game_length = max_game_length
        self.save_hyperparameters()

        self.policy_network_red = PolicyNetwork()
        self.policy_network_black = PolicyNetwork()
        self.value_network_black.load_state_dict(self.value_network_red.state_dict())

        self.elo_red = self.elo_black = 1200

        self.game_board = GameBoard()
        self.memory = Memory()

    def configure_optimizers(self):
        return torch.optim.Adam(self.policy_network_red.parameters(), lr=self.lr)

    def play_game(self):
        self.game_board = GameBoard()
        i = 0
        while not self.game_board.game_over and i < self.max_game_length:
            reward_prev = self.game_board.scores[Player.RED]
            action, log_prob = self.get_action(
                self.game_board.board, self.game_board.current_player
            )
            self.game_board.move(action)
            reward_next = self.game_board.scores[Player.RED]
            reward = reward_next - reward_prev
            done = self.game_board.game_over
            self.memory.states.append(self.game_board.board.to_tensor())
            self.memory.actions.append(action)
            self.memory.logprobs.append(log_prob)
            self.memory.rewards.append(reward)
            self.memory.is_terminals.append(done)
            i += 1

    def elo_update(self, winner: Player):
        raise NotImplementedError

    def training_step(self, *args):
        # play several games
        for _ in range(self.games_per_batch):
            try:
                self.play_game()
            except GameOver as e:
                self.elo_update(e.winner)
        # collect data
        # train
        loss = self.loss()
        self.log("loss", loss)
        return loss

    def get_action(self, state: BoardState, player: Player):
        if player == Player.BLACK:
            state = state.flip()
        state_tensor = state.to_tensor()
        mask = state.get_moves_mask(Player.RED)  # AI always plays as RED
        action_probs, _ = self.agent(state_tensor, mask)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        return action, dist.log_prob(action)
