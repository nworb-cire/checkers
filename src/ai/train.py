from tqdm import trange

from src.ai.actions import ACTIONS
from src.ai.ppo import PPOAgent, Memory
from src.ai.self_play import CheckersEnv
from src.game.board import Player, BoardState


def train_ppo(
    env,
    agent,
    memory,
    max_episodes=5_000,
    max_timesteps=100,
    update_timestep=50,
):
    timestep = 0
    for _ in trange(max_episodes, desc="Games"):
        game_board = env.reset()
        for t in range(max_timesteps):
            timestep += 1
            while (
                game_board.current_player == Player.BLACK and not game_board.game_over
            ):
                board_prev = BoardState(game_board.board.board.copy())
                move, logprob = agent.select_action_black(game_board)
                next_state, reward, done, _, _ = env.step(move.flip())
                assert next_state.board != board_prev
            board_prev = BoardState(game_board.board.board.copy())
            move, logprob = agent.select_action_red(game_board)
            next_state, reward, done, _, _ = env.step(move)
            assert next_state.board != board_prev

            memory.states.append(game_board.board.board.flatten())
            memory.actions.append(ACTIONS[move])
            memory.logprobs.append(logprob)
            if memory.rewards:
                reward -= memory.rewards[-1]
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            game_board = next_state

            if timestep % update_timestep == 0:
                agent.update(memory)
                memory.clear_memory()
                timestep = 0

            if done:
                if game_board.board.is_stalemate():
                    print("Stalemate.")
                elif winner := game_board.board.is_game_over():
                    print(
                        f"Game over after {game_board.turn_number} turns, winner: {winner}"
                    )
                    print(f"Scores: {game_board.scores}")
                break

    print("Training finished.")
    print("Saving model...")
    agent.save("ppo.pt")
    print("Done.")


if __name__ == "__main__":
    env = CheckersEnv()
    state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_dim = env.action_space.n

    ppo_agent = PPOAgent(state_dim, action_dim)
    memory = Memory()

    train_ppo(env, ppo_agent, memory)
