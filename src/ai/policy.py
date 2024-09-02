import torch
from torch import nn as nn
from torch.distributions import Categorical


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
