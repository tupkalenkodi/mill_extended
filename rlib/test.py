"""
RL Agent for Mill Game using Stable Baselines3

This module implements:
1. A Gym-compatible wrapper for the Mill environment
2. Training against opponents of various difficulties
3. Custom reward structure based on game outcome
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import os
from typing import Optional, Tuple, Dict, Any
from famnit_gym.envs import mill
from minimax_usages_tests.second.ai_player_with_difficulty import AiPlayerWithDifficulty
import random


class MillGymWrapper(gym.Env):
    """
    Gym-compatible wrapper for the Mill environment.

    WHAT THIS DOES:
    - Converts the PettingZoo Mill environment to work with Stable Baselines3
    - Handles opponent moves automatically (random or AI)
    - Implements your custom reward structure
    - Provides proper observation and action spaces

    KEY CONCEPTS:
    - Observation: What the agent "sees" (board state)
    - Action: What the agent can do (moves)
    - Reward: Feedback signal for learning
    """

    def __init__(self, opponent_type: str = "random", opponent_difficulty: str = "knight"):
        """
        Initialize the wrapper.

        Args:
            opponent_type: "random" or "ai"
            opponent_difficulty: One of ["apprentice", "adventurer", "knight", "champion", "legend"]
                                Only used if opponent_type is "ai"
        """
        super().__init__()

        # Create the Mill environment
        self.env = mill.env()
        self.opponent_type = opponent_type
        self.opponent_difficulty = opponent_difficulty

        # Our agent is always player 1, opponent is player 2
        self.agent_player = 1
        self.opponent_player = 2

        # Track moves for reward calculation
        self.num_moves = 0

        # OBSERVATION SPACE EXPLANATION:
        # The board has 24 positions, each can be: 0 (empty), 1 (player 1), 2 (player 2)
        # We represent this as a Box with values 0-2
        self.observation_space = spaces.Box(
            low=0,
            high=2,
            shape=(24,),
            dtype=np.uint8
        )

        # ACTION SPACE EXPLANATION:
        # Actions are [from, to, capture] where each is 0-24
        # 0 means "ignore this part" (e.g., from=0 for placing phase)
        # This is a MultiDiscrete space with 3 dimensions
        self.action_space = spaces.MultiDiscrete([25, 25, 25])

        # Initialize opponent AI if needed
        self.opponent_ai = None
        if self.opponent_type == "ai":
            self.opponent_ai = AiPlayerWithDifficulty(
                player_id=self.opponent_player,
                difficulty=self.opponent_difficulty
            )

    def _get_observation(self) -> np.ndarray:
        """
        Get the current board state as observation.

        WHAT THIS DOES:
        - Extracts the board state (24 positions)
        - Returns it as a numpy array
        """
        state = mill.transition_model(self.env)
        return np.array(state.get_state(), dtype=np.uint8)

    def _get_legal_moves(self, player: int) -> np.ndarray:
        """Get legal moves for a player."""
        state = mill.transition_model(self.env)
        return np.array(state.legal_moves(player))

    def _is_game_over(self) -> Tuple[bool, Optional[int]]:
        """
        Check if game is over and who won.

        Returns:
            (is_over, winner) where winner is 1, 2, or None (draw)
        """
        state = mill.transition_model(self.env)

        if state.game_over():
            # Check who lost
            if state.get_phase(self.agent_player) == 'lost':
                return True, self.opponent_player
            elif state.get_phase(self.opponent_player) == 'lost':
                return True, self.agent_player
            else:
                return True, None  # Draw (shouldn't happen with game_over())

        return False, None

    def _opponent_move(self):
        """
        Make the opponent's move.

        WHAT THIS DOES:
        - If opponent is random: chooses a random legal move
        - If opponent is AI: uses the minimax agent to choose
        - Executes the move in the environment
        """
        state = mill.transition_model(self.env)

        if self.opponent_type == "random":
            # Random opponent - choose any legal move
            legal_moves = self._get_legal_moves(self.opponent_player)
            if len(legal_moves) > 0:
                move = legal_moves[np.random.choice(len(legal_moves))]
            else:
                return  # No legal moves (game over)
        else:
            # AI opponent - use minimax
            move = self.opponent_ai.choose_move(state, self.num_moves)

        # Execute the move
        self.env.step(move)
        self.num_moves += 1

    def _calculate_reward(self, winner: Optional[int]) -> float:
        """
        Calculate reward based on game outcome.

        YOUR REWARD STRUCTURE:
        - Win: 200 / num_moves (faster wins = higher reward)
        - Loss: -200 / num_moves (faster losses = larger penalty)
        - Draw: 0

        WHY THIS MATTERS:
        - Encourages the agent to win quickly
        - Penalizes losses proportionally
        - This is the learning signal for the agent
        """
        if winner is None:
            return 0.0  # Draw
        elif winner == self.agent_player:
            return 200.0 / max(1, self.num_moves)  # Win
        else:
            return -200.0 / max(1, self.num_moves)  # Loss

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to start a new episode.

        WHAT THIS DOES:
        - Resets the Mill game
        - Resets move counter
        - Returns initial observation

        WHY THIS MATTERS:
        - Each training episode starts here
        - Agent learns from multiple episodes
        """
        super().reset(seed=seed)
        self.env.reset()
        self.num_moves = 0

        observation = self._get_observation()
        info = {"legal_moves": self._get_legal_moves(self.agent_player)}

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.

        WHAT THIS DOES:
        1. Agent makes a move
        2. Check if game is over
        3. If not, opponent makes a move
        4. Check if game is over again
        5. Calculate reward
        6. Return observation, reward, done flags, info

        Args:
            action: The agent's chosen action [from, to, capture]

        Returns:
            observation: New board state
            reward: Reward for this step
            terminated: True if game ended naturally (win/loss)
            truncated: True if game hit max moves
            info: Additional information
        """
        # Convert action to list format
        action_list = action.tolist() if isinstance(action, np.ndarray) else action

        # Get legal moves for validation
        legal_moves = self._get_legal_moves(self.agent_player)

        # Check if action is legal
        is_legal = False
        for legal_move in legal_moves:
            if np.array_equal(action_list, legal_move):
                is_legal = True
                break

        # If illegal move, choose a random legal move instead
        if not is_legal and len(legal_moves) > 0:
            action_list = legal_moves[np.random.choice(len(legal_moves))].tolist()

        # Execute agent's move
        self.env.step(action_list)
        self.num_moves += 1

        # Check if game is over after agent's move
        is_over, winner = self._is_game_over()

        if not is_over:
            # Opponent's turn
            self._opponent_move()

            # Check if game is over after opponent's move
            is_over, winner = self._is_game_over()

        # Check for truncation (max 200 moves)
        truncated = self.num_moves >= 200

        # Calculate reward
        if is_over or truncated:
            reward = self._calculate_reward(winner if not truncated else None)
        else:
            reward = 0.0  # No reward for intermediate steps

        # Get new observation
        observation = self._get_observation()

        # Prepare info
        info = {
            "legal_moves": self._get_legal_moves(self.agent_player) if not is_over else [],
            "winner": winner,
            "num_moves": self.num_moves
        }

        return observation, reward, is_over, truncated, info

    def render(self):
        """Render the environment (optional)."""
        state = mill.transition_model(self.env)
        print(state)


def make_env(opponent_type: str = "random", opponent_difficulty: str = "knight"):
    """
    Factory function to create wrapped environment.

    WHAT THIS DOES:
    - Creates a MillGymWrapper
    - Wraps it in Monitor for logging
    - Returns the environment

    WHY THIS MATTERS:
    - Monitor tracks episode rewards and lengths
    - Helps with debugging and evaluation
    """

    def _init():
        env = MillGymWrapper(opponent_type, opponent_difficulty)
        env = Monitor(env)
        return env

    return _init


def train_rl_agent(
        opponent_type: str = "random",
        opponent_difficulty: str = "knight",
        total_timesteps: int = 100_000,
        save_dir: str = "rl_models",
        model_name: str = "mill_ppo"
):
    """
    Train an RL agent using PPO algorithm.

    WHAT THIS DOES:
    1. Creates training and evaluation environments
    2. Initializes PPO agent
    3. Sets up callbacks for saving and evaluation
    4. Trains the agent
    5. Saves the final model

    KEY CONCEPTS:
    - PPO (Proximal Policy Optimization): A popular RL algorithm
    - Policy: What the agent learns (mapping from states to actions)
    - Training: Agent plays many games and learns from outcomes
    - Evaluation: Periodically test agent's performance

    Args:
        opponent_type: "random" or "ai"
        opponent_difficulty: Difficulty level if opponent_type is "ai"
        total_timesteps: How many steps to train for
        save_dir: Where to save models
        model_name: Name for the saved model
    """

    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    log_dir = os.path.join(save_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    print("=" * 70)
    print("TRAINING RL AGENT FOR MILL GAME")
    print("=" * 70)
    print(f"Opponent Type: {opponent_type}")
    if opponent_type == "ai":
        print(f"Opponent Difficulty: {opponent_difficulty}")
    print(f"Total Timesteps: {total_timesteps:,}")
    print(f"Save Directory: {save_dir}")
    print("=" * 70)

    # STEP 1: CREATE TRAINING ENVIRONMENT
    print("\n[1/5] Creating training environment...")
    train_env = DummyVecEnv([make_env(opponent_type, opponent_difficulty)])
    print("✓ Training environment created")

    # STEP 2: CREATE EVALUATION ENVIRONMENT
    print("\n[2/5] Creating evaluation environment...")
    eval_env = DummyVecEnv([make_env(opponent_type, opponent_difficulty)])
    print("✓ Evaluation environment created")

    # STEP 3: INITIALIZE PPO AGENT
    print("\n[3/5] Initializing PPO agent...")
    print("PPO Hyperparameters:")
    print("  - Learning Rate: 0.0003")
    print("  - Batch Size: 64")
    print("  - Number of Steps: 2048")
    print("  - Policy: MlpPolicy (Multi-Layer Perceptron)")

    model = PPO(
        "MlpPolicy",  # Neural network policy
        train_env,
        learning_rate=3e-4,
        n_steps=2048,  # Steps to collect before update
        batch_size=64,  # Batch size for training
        n_epochs=10,  # Training epochs per update
        gamma=0.99,  # Discount factor
        verbose=1,  # Print training info
        tensorboard_log=log_dir
    )
    print("✓ PPO agent initialized")

    # STEP 4: SETUP CALLBACKS
    print("\n[4/5] Setting up callbacks...")

    # Evaluation callback - tests agent every 10k steps
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_dir, "best_model"),
        log_path=log_dir,
        eval_freq=10_000,  # Evaluate every 10k steps
        deterministic=True,
        render=False,
        n_eval_episodes=10  # Test on 10 episodes
    )

    # Checkpoint callback - saves model every 20k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=20_000,
        save_path=os.path.join(save_dir, "checkpoints"),
        name_prefix=model_name
    )

    callbacks = [eval_callback, checkpoint_callback]
    print("✓ Callbacks configured")

    # STEP 5: TRAIN THE AGENT
    print("\n[5/5] Starting training...")
    print("=" * 70)
    print("Training in progress... This may take a while.")
    print("Monitor progress in TensorBoard:")
    print(f"  tensorboard --logdir {log_dir}")
    print("=" * 70)

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True
    )

    print("\n" + "=" * 70)
    print("Training completed!")
    print("=" * 70)

    # Save final model
    final_model_path = os.path.join(save_dir, f"{model_name}_final")
    model.save(final_model_path)
    print(f"✓ Final model saved to: {final_model_path}")

    return model


def evaluate_agent(
        model_path: str,
        opponent_type: str = "random",
        opponent_difficulty: str = "knight",
        n_episodes: int = 100
):
    """
    Evaluate a trained agent.

    WHAT THIS DOES:
    - Loads a trained model
    - Plays n_episodes games
    - Reports win rate, average reward, etc.

    Args:
        model_path: Path to the saved model
        opponent_type: Type of opponent
        opponent_difficulty: Difficulty if opponent is AI
        n_episodes: Number of games to play
    """
    print("=" * 70)
    print("EVALUATING RL AGENT")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Opponent Type: {opponent_type}")
    if opponent_type == "ai":
        print(f"Opponent Difficulty: {opponent_difficulty}")
    print(f"Number of Episodes: {n_episodes}")
    print("=" * 70)

    # Load model
    model = PPO.load(model_path)

    # Create environment
    env = MillGymWrapper(opponent_type, opponent_difficulty)

    # Evaluation metrics
    wins = 0
    losses = 0
    draws = 0
    total_reward = 0
    total_moves = 0

    print("\nRunning evaluation...")
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0

        while not done:
            # Agent chooses action
            action, _ = model.predict(obs, deterministic=True)

            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        # Record outcome
        total_reward += episode_reward
        total_moves += info["num_moves"]

        if info.get("winner") == env.agent_player:
            wins += 1
        elif info.get("winner") == env.opponent_player:
            losses += 1
        else:
            draws += 1

        # Progress
        if (episode + 1) % 10 == 0:
            print(f"  Completed {episode + 1}/{n_episodes} episodes...")

    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Wins: {wins}/{n_episodes} ({wins / n_episodes * 100:.1f}%)")
    print(f"Losses: {losses}/{n_episodes} ({losses / n_episodes * 100:.1f}%)")
    print(f"Draws: {draws}/{n_episodes} ({draws / n_episodes * 100:.1f}%)")
    print(f"Average Reward: {total_reward / n_episodes:.2f}")
    print(f"Average Moves per Game: {total_moves / n_episodes:.1f}")
    print("=" * 70)

    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / n_episodes,
        "avg_reward": total_reward / n_episodes,
        "avg_moves": total_moves / n_episodes
    }


# EXAMPLE USAGE
if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                   RL AGENT TRAINER FOR MILL GAME                     ║
    ║                                                                      ║
    ║  This script trains a Reinforcement Learning agent to play Mill    ║
    ║  using Proximal Policy Optimization (PPO) from Stable Baselines3.  ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)

    # Configuration
    OPPONENT_TYPE = "random"  # Change to "ai" for minimax opponent
    OPPONENT_DIFFICULTY = "knight"  # Only matters if OPPONENT_TYPE is "ai"
    TOTAL_TIMESTEPS = 100_000  # Increase for better results (e.g., 500k-1M)

    print("\nConfiguration:")
    print(f"  Opponent: {OPPONENT_TYPE}")
    if OPPONENT_TYPE == "ai":
        print(f"  Difficulty: {OPPONENT_DIFFICULTY}")
    print(f"  Training Steps: {TOTAL_TIMESTEPS:,}")

    # Train
    print("\nStarting training...")
    trained_model = train_rl_agent(
        opponent_type=OPPONENT_TYPE,
        opponent_difficulty=OPPONENT_DIFFICULTY,
        total_timesteps=TOTAL_TIMESTEPS,
        save_dir="rl_models",
        model_name="mill_ppo"
    )

    # Evaluate
    print("\nEvaluating trained agent...")
    results = evaluate_agent(
        model_path="rl_models/mill_ppo_final",
        opponent_type=OPPONENT_TYPE,
        opponent_difficulty=OPPONENT_DIFFICULTY,
        n_episodes=50
    )

    print("\n✓ Training and evaluation complete!")
    print("\nNext steps:")
    print("  1. Train for more timesteps for better performance")
    print("  2. Try different opponent difficulties")
    print("  3. Tune hyperparameters (learning rate, batch size, etc.)")
    print("  4. Compare against minimax agents")
