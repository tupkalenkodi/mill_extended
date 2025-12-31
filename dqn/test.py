from famnit_gym.envs import mill
from dqn.agent import DQNAgent


def test_dqn_vs_random(model_path, num_games=1000):
    """Test trained DQN agent against random opponent"""

    env = mill.env(render_mode="none")

    # Load trained agent
    dqn_agent = DQNAgent(player_id=1)
    dqn_agent.load(model_path)
    dqn_agent.epsilon = 0.0  # No exploration during testing

    wins = 0
    losses = 0
    draws = 0

    for game in range(num_games):
        env.reset()

        for agent_name in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()

            current_player = 1 if agent_name == "player_1" else 2
            legal_moves = info['legal_moves']

            if termination or truncation:
                if truncation:
                    draws += 1
                elif current_player == 1:  # DQN agent lost
                    losses += 1
                else:  # DQN agent won
                    wins += 1
                break

            # DQN agent's turn
            if current_player == 1:
                action = dqn_agent.choose_move(observation, legal_moves)
            # Random opponent
            else:
                action = legal_moves[np.random.choice(len(legal_moves))]

            env.step(action)

        print(f"Game {game + 1}: Wins={wins}, Losses={losses}, Draws={draws}")

    print(f"\nFinal Results: Wins={wins}, Losses={losses}, Draws={draws}")
    print(f"Win Rate: {wins / num_games * 100:.1f}%")


if __name__ == "__main__":
    import numpy as np

    test_dqn_vs_random('dqn_models/agent1_final.pth', num_games=10)