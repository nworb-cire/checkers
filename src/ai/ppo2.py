import pytorch_lightning as pl
import torch
from torch import nn as nn
from torch.distributions import Categorical

from src.ai.actions import MOVES, ACTIONS
from src.game.board import BoardState
from src.game.board import GameBoard
from src.game.errors import GameOver
from src.game.moves import Move
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
        self,
        state: BoardState,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = state.get_moves_mask(Player.RED)  # AI always plays as RED
        x = state.to_tensor()
        x = self.resnet(x)
        value = self.value_head(x)

        logits = self.action_head(x)
        logits[~mask] = -float("inf")
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value


class Memory:
    actions: list[Move]
    states: list[BoardState]
    logprobs: list[torch.Tensor]
    rewards: list[float]
    is_terminals: list[bool]

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
            move = MOVES[action.item()]
            try:
                self.game_board.make_move(move)
            except GameOver:
                self.memory.is_terminals[-1] = True
                # TODO: handle score?
                raise

            reward_next = self.game_board.scores[Player.RED]
            reward = reward_next - reward_prev
            done = self.game_board.game_over

            # Write to history only on red turns
            if self.game_board.current_player == Player.RED:
                self.memory.states.append(self.game_board.board.to_tensor())
                self.memory.actions.append(move)
                self.memory.logprobs.append(log_prob)
                self.memory.rewards.append(reward)
                self.memory.is_terminals.append(done)
            i += 1

    def elo_update(self, winner: Player, k: int = 32):
        """
        Update the ELO scores for the agents after a game.
        :param winner: The Player who won the game, or None if it was a draw.
        :param k: The K-factor for the ELO update.
        """
        if winner is None:
            return
        P_red = 1 / (1 + 10 ** ((self.elo_black - self.elo_red) / 400))
        P_black = 1 / (1 + 10 ** ((self.elo_red - self.elo_black) / 400))

        if winner == Player.RED:
            self.elo_red += k * (1 - P_red)
            self.elo_black -= k * P_black
        else:
            self.elo_red -= k * P_red
            self.elo_black += k * (1 - P_black)
        self.log("ELO_red", self.elo_red)
        self.log("ELO_black", self.elo_black)

    def loss(self, states, actions, logprobs, rewards):
        """Single iteration of PPO update."""
        # Evaluating old actions and values:
        action, log_prob, entropy, state_value = self.policy_network_red(
            states, actions
        )

        # Finding the ratio (pi_theta / pi_theta__old):
        ratios = torch.exp(log_prob - logprobs)

        # Finding Surrogate Loss:
        advantages = rewards - state_value.detach()
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        loss = (
            -torch.min(surr1, surr2)
            + 0.5 * nn.functional.mse_loss(state_value, rewards)
            - 0.01 * entropy
        )
        return loss

    def on_train_epoch_start(self):
        # play several games
        for _ in range(self.games_per_batch):
            try:
                self.play_game()
            except GameOver as e:
                self.elo_update(e.winner)

    def on_train_epoch_end(self):
        self.memory.clear_memory()
        self.policy_network_black.load_state_dict(self.policy_network_red.state_dict())

    def training_step(self, *args):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(
            reversed(self.memory.rewards), reversed(self.memory.is_terminals)
        ):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            rewards.insert(0, discounted_reward)

        loss = self.loss(
            states=torch.stack([state.to_tensor() for state in self.memory.states]),
            actions=torch.tensor([ACTIONS[action] for action in self.memory.actions]),
            logprobs=torch.stack(self.memory.logprobs),
            rewards=torch.tensor(rewards, dtype=torch.float32),
        )
        self.log("loss", loss)
        return loss

    def get_action(
        self,
        state: BoardState,
        player: Player,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if player == Player.BLACK:
            network = self.policy_network_black
            state = state.flip()
        else:
            network = self.policy_network_red
        state_tensor = state.to_tensor()
        mask = state.get_moves_mask(Player.RED)  # AI always plays as RED
        action, log_prob, _, _ = network(state_tensor)
        return action, log_prob
