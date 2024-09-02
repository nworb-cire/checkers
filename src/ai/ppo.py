import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from src.ai.actions import MOVES
from src.game.board import GameBoard, Player


class PPOAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        eps_clip=0.2,
        K_epochs=4,
    ):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy_red = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy_red.parameters(), lr=lr)
        self.policy_black = PolicyNetwork(state_dim, action_dim)
        self.policy_black.load_state_dict(self.policy_red.state_dict())
        self.MseLoss = nn.MSELoss()

    @torch.no_grad()
    def select_action(self, game_board: GameBoard):
        if game_board.current_player == Player.RED:
            return self.select_action_red(game_board)
        return self.select_action_black(game_board)

    def select_action_red(self, game_board: GameBoard):
        state = torch.tensor(game_board.board.board, dtype=torch.float32).flatten()
        mask = torch.tensor(game_board.get_moves_mask(), dtype=torch.bool)
        action_probs, _ = self.policy_red(state, mask)
        dist = Categorical(action_probs)
        action = dist.sample()
        move = MOVES[action.item()]
        return move, dist.log_prob(action).item()

    def select_action_black(self, game_board: GameBoard):
        # Model expects to be playing as player 1 (red).
        # If the AI is playing as player 2 (black), the board needs to be flipped.
        state = torch.tensor(
            game_board.board.flip().board, dtype=torch.float32
        ).flatten()
        mask = torch.tensor(game_board.get_moves_mask(), dtype=torch.bool)
        action_probs, _ = self.policy_black(state, mask)
        dist = Categorical(action_probs)
        action = dist.sample()
        move = MOVES[action.item()].flip()
        return move, dist.log_prob(action).item()

    def update(self, memory):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(
            reversed(memory.rewards), reversed(memory.is_terminals)
        ):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32)
        old_states = torch.tensor(memory.states, dtype=torch.float32)
        old_actions = torch.tensor(memory.actions, dtype=torch.int64)
        old_logprobs = torch.tensor(memory.logprobs, dtype=torch.float32)

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy_red.evaluate(
                old_states, old_actions
            )
            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )
            loss = (
                -torch.min(surr1, surr2)
                + 0.5 * self.MseLoss(state_values, rewards)
                - 0.01 * dist_entropy
            )

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_black.load_state_dict(self.policy_red.state_dict())


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int = 8 * 8, action_dim: int = 32 * 32):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.action_head = nn.Linear(128, action_dim)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x, mask=None):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        logits = self.action_head(x)
        if mask is not None:
            logits[~mask] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        value = self.value_head(x)
        return probs, value

    def evaluate(self, state, action):
        action_probs, state_value = self.forward(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprobs, state_value, dist_entropy


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
