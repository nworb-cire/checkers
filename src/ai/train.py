from tqdm import trange

from src.ai.actions import MOVES, ACTIONS
from src.ai.ppo import PPOAgent, Memory
from src.ai.self_play import CheckersEnv
from src.game.board import Player


def train_ppo(
    env,
    agent,
    memory,
    max_episodes=1000,
    max_timesteps=50,
    update_timestep=10,
):
    timestep = 0
    for _ in trange(max_episodes, desc="Games"):
        game_board = env.reset()
        for t in range(max_timesteps):
            timestep += 1
            while game_board.current_player == Player.BLACK:
                move, logprob = agent.select_action(game_board)
                next_state, reward, done, _, _ = env.step(move)
            move, logprob = agent.select_action(game_board)
            next_state, reward, done, _, _ = env.step(move)

            memory.states.append(game_board.board.board.flatten())
            memory.actions.append(ACTIONS[move])
            memory.logprobs.append(logprob)
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            game_board = next_state

            if timestep % update_timestep == 0:
                agent.update(memory)
                memory.clear_memory()
                timestep = 0

            if done:
                break


if __name__ == "__main__":
    env = CheckersEnv()
    state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_dim = env.action_space.n

    ppo_agent = PPOAgent(state_dim, action_dim)
    memory = Memory()

    train_ppo(env, ppo_agent, memory)
