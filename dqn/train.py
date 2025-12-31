import numpy as np
from famnit_gym.envs import mill
from dqn.agent import DQNAgent


# In your training script, add this helper
def calculate_shaped_reward(old_state, new_state, player_id, capture_made=False):
    """
    Calculate sophisticated reward using minimax evaluation
    """
    # Your existing evaluate_state function (adapted)
    def evaluate_state(state_model, player):
        opponent = 3 - player
        p1_pieces = state_model.count_pieces(player)
        p2_pieces = state_model.count_pieces(opponent)
        piece_advantage = (p1_pieces - p2_pieces) * 30

        # Position evaluation
        position_advantage = 0
        board_state = state_model.get_state()

        position_values = {
            4: 4, 5: 4, 6: 4, 14: 4, 21: 4, 20: 4, 19: 4, 11: 4,
            1: 3, 2: 3, 3: 3, 15: 3, 24: 3, 23: 3, 22: 3, 10: 3,
            7: 3, 8: 3, 9: 3, 13: 3, 18: 3, 17: 3, 16: 3, 12: 3
        }

        for pos, value in position_values.items():
            if board_state[pos - 1] == player:
                position_advantage += value
            elif board_state[pos - 1] == opponent:
                position_advantage -= value

        total_score = piece_advantage + position_advantage
        normalized_score = total_score / 240.0  # Normalize to [-1, 1]
        return normalized_score

    # Calculate scores
    old_score = evaluate_state(old_state, player_id)
    new_score = evaluate_state(new_state, player_id)

    # Improvement
    total_reward = new_score - old_score

    return total_reward


def train_dqn(num_episodes=1000, target_update_freq=10, save_freq=100):
    """Train DQN agent through self-play"""

    # Create environment
    env = mill.env(render_mode=None)

    # Create two agents (self-play)
    agent1 = DQNAgent(player_id=1)
    agent2 = DQNAgent(player_id=2)

    episode_rewards = []

    for episode in range(num_episodes):
        env.reset()

        episode_reward_1 = 0
        episode_reward_2 = 0
        step_count = 0

        for agent_name in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()

            # Get current player and state
            current_player = 1 if agent_name == "player_1" else 2
            state = observation  # Board state

            # Choose agent
            agent = agent1 if current_player == 1 else agent2

            # Episode done
            if termination or truncation:
                # Store final transition with terminal reward
                if current_player == 1:
                    episode_reward_1 += reward
                    agent1.store_transition(state, [0, 0, 0], reward, state, True)
                else:
                    episode_reward_2 += reward
                    agent2.store_transition(state, [0, 0, 0], reward, state, True)
                break

            # Get legal moves and choose action
            legal_moves = info['legal_moves']
            action = agent.choose_move(state, legal_moves)

            # Take step
            env.step(action)

            # Get next state and reward
            next_observation, next_reward, _, _, _ = env.last()

            # Store transition
            if current_player == 1:
                episode_reward_1 += next_reward
                agent1.store_transition(state, action, next_reward, next_observation, False)
            else:
                episode_reward_2 += next_reward
                agent2.store_transition(state, action, next_reward, next_observation, False)

            # Train both agents
            loss1 = agent1.train()
            loss2 = agent2.train()

            step_count += 1

        # Decay epsilon
        agent1.decay_epsilon()
        agent2.decay_epsilon()

        # Update target networks
        if episode % target_update_freq == 0:
            agent1.update_target_network()
            agent2.update_target_network()

        # Save models
        if episode % save_freq == 0 and episode > 0:
            agent1.save(f'dqn_models/agent1_episode_{episode}.pth')
            agent2.save(f'dqn_models/agent2_episode_{episode}.pth')

        # Logging
        episode_rewards.append((episode_reward_1, episode_reward_2))

        if episode % 10 == 0:
            avg_reward_1 = np.mean([r[0] for r in episode_rewards[-10:]])
            avg_reward_2 = np.mean([r[1] for r in episode_rewards[-10:]])
            print(f"Episode {episode}/{num_episodes}")
            print(f"  Avg Reward P1: {avg_reward_1:.2f}, P2: {avg_reward_2:.2f}")
            print(f"  Epsilon: {agent1.epsilon:.3f}")
            print(f"  Steps: {step_count}")

    # Final save
    agent1.save('dqn_models/agent1_final.pth')
    agent2.save('dqn_models/agent2_final.pth')

    print("Training complete!")

    return agent1, agent2, episode_rewards


if __name__ == "__main__":
    import os

    os.makedirs('dqn_models', exist_ok=True)

    train_dqn(num_episodes=1000, target_update_freq=10, save_freq=100)
