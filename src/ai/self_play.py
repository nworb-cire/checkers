from typing import SupportsFloat, Any

import gymnasium as gym
from gymnasium import spaces, register
from gymnasium.core import ActType, ObsType

from src.ai.actions import action_to_move
from src.ai.ai import CheckersAI
from src.game.board import GameBoard, Player


class CheckersEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(32 * 32)  # (from, to)
        self.observation_space = spaces.Box(low=0, high=2, shape=(8, 8))

        self.game_board = GameBoard()
        self.player = Player.RED

    def reset(self, seed=None, options=None) -> GameBoard:
        self.game_board = GameBoard()
        return self.game_board

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        turn_over = self.game_board.make_move(action)
        if turn_over:
            self.game_board.switch_player()

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
