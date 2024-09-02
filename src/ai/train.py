from src.ai.actions import MOVES
from src.ai.ppo import PPOAgent, Memory
from src.ai.self_play import CheckersEnv


def train_ppo(
    env,
    agent,
    memory,
    max_episodes=1000,
    max_timesteps=3000,
    update_timestep=2000,
):
    timestep = 0
    for episode in range(max_episodes):
        state = env.reset()
        for t in range(max_timesteps):
            timestep += 1
            action, logprob = agent.select_action(state)
            move = MOVES[action]
            next_state, reward, done, _, _ = env.step(move)

            memory.states.append(state.board.board.flatten())
            memory.actions.append(action)
            memory.logprobs.append(logprob)
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            state = next_state

            if timestep % update_timestep == 0:
                agent.update(memory)
                memory.clear_memory()
                timestep = 0

            if done:
                break

        print(f"Episode {episode} completed")


if __name__ == "__main__":
    env = CheckersEnv()
    state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_dim = env.action_space.n

    ppo_agent = PPOAgent(state_dim, action_dim)
    memory = Memory()

    train_ppo(env, ppo_agent, memory)
