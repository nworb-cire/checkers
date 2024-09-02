from typing import SupportsFloat, Any

import gymnasium as gym
from gymnasium import spaces, register
from gymnasium.core import ActType, ObsType

from src.ai.actions import action_to_move
from src.game.board import GameBoard, Player


class CheckersEnv(gym.Env):
    MOVES = {action: action_to_move(action) for action in range(32 * 32)}

    def __init__(self):
        self.action_space = spaces.Discrete(32 * 32)  # (from, to)
        self.observation_space = spaces.Box(low=0, high=2, shape=(8, 8))

        self.game_board = GameBoard()
        self.player = Player.BLACK

    def reset(self, seed=None, options=None):
        self.game_board = GameBoard()
        return self.game_board.board

    def is_valid_action(self, action):
        move = self.MOVES[action]
        moves, jumps = self.game_board.get_available_moves(
            self.game_board.current_player
        )
        return move in moves + jumps

    def get_valid_action(self):
        moves, jumps = self.game_board.get_available_moves(
            self.game_board.current_player
        )
        while True:
            action = self.action_space.sample()
            move = self.MOVES[action]
            if move in moves + jumps:
                return action

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        move = self.MOVES[action]
        self.game_board.make_move(move)

        terminated = self.game_board.game_over
        reward = self.game_board.scores[self.player]
        observation = self.game_board
        info = {}
        return observation, reward, terminated, False, info


register(
    id="Checkers-v0",
    entry_point="src.ai.self_play:CheckersEnv",
    max_episode_steps=100,
    reward_threshold=1.0,
)
