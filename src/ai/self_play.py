from typing import SupportsFloat, Any

import gymnasium as gym
from gymnasium import spaces, register
from gymnasium.core import ActType, ObsType

from src.ai.actions import action_to_move
from src.ai.ai import CheckersAI
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
        self.game_board.make_move(action)

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


if __name__ == "__main__":
    env = CheckersEnv()
    observation = env.reset()
    terminated = False

    agents = {
        Player.RED: CheckersAI.init(Player.RED),
        Player.BLACK: CheckersAI.init(Player.BLACK),
    }
    current_player = Player.RED

    while not terminated:
        action, _ = agents[current_player].select_action(env.game_board)
        print(f"Player {current_player}: {action}")
        print(env.game_board.board)
        next_observation, reward, terminated, _, _ = env.step(action)
        current_player = -current_player
        observation = next_observation

    print("Game over!")
    print(env.game_board.scores)
