import gymnasium as gym
import numpy as np
from famnit_gym.envs import mill
from minimax_usages_tests.second.ai_player_with_difficulty import AiPlayerWithDifficulty


class MillSingleEnv(gym.Env):
    metadata = {'render_modes': ['human', 'ansi']}

    def __init__(self, render_mode=None, difficulty=None):
        super().__init__()
        self.render_mode = render_mode

        # Action space - must be single Discrete space for SB3
        # MultiDiscrete is problematic for SB3, convert to single Discrete
        # 25 * 25 * 25 = 15625 possible actions
        self.action_space = gym.spaces.Discrete(15625)

        # Observation space - convert Dict to Box for SB3 compatibility
        # Dict spaces are problematic for most SB3 algorithms
        # Let's flatten everything into a single vector
        observation_dim = 24 + 1 + 1 + 1  # board + turn + phase1 + phase2
        self.observation_space = gym.spaces.Box(
            low=0,
            high=3,  # max phase value
            shape=(observation_dim,),
            dtype=np.float32
        )

        # Internal PettingZoo environment
        self._pettingzoo_env = mill.env(render_mode=render_mode)
        self._current_player = "player_1"  # Must be string, not int

        if difficulty != "random":
            self.opponent = AiPlayerWithDifficulty(player_id=2, difficulty=difficulty)
        else:
            self.opponent = None

        self.total_moves_in_game = 0

        # Action mapping: store legal moves for decoding
        self.legal_moves = None

    def _encode_action(self, action_idx):
        """Convert single Discrete action back to MultiDiscrete"""
        if self.legal_moves is None or len(self.legal_moves) == 0:
            return np.array([0, 0, 0])

        # Use modulo to ensure valid index
        idx = action_idx % len(self.legal_moves)
        return self.legal_moves[idx]

    def _decode_observation(self, pettingzoo_obs):
        """Convert PettingZoo observation to flattened vector"""
        board = np.array(self._pettingzoo_env.observe(self._current_player), dtype=np.float32)

        # Determine whose turn it is
        current_agent = self._pettingzoo_env.agent_selection
        player_turn = 0.0 if current_agent == self._current_player else 1.0

        # Get phases
        player_1_phase = float(self._pettingzoo_env.infos["player_1"]['phase'])
        player_2_phase = float(self._pettingzoo_env.infos["player_2"]['phase'])

        # Flatten everything
        return np.concatenate([
            board,
            np.array([player_turn, player_1_phase, player_2_phase], dtype=np.float32)
        ])

    def reset(self, seed=None, options=None):
        # Reset the underlying multi-agent env
        self._pettingzoo_env.reset(seed=seed, options=options)

        # Reset counters
        self.total_moves_in_game = 0

        # Update legal moves
        self.legal_moves = self._pettingzoo_env.legal_moves[self._current_player]

        # Get initial observation
        obs = self._decode_observation(None)
        info = self._get_info()

        return obs, info

    def step(self, action):
        # Decode the Discrete action to MultiDiscrete
        decoded_action = self._encode_action(action)

        # Execute action for current player
        self._pettingzoo_env.step(decoded_action)
        self.total_moves_in_game += 1

        # Get opponent's move
        if not self._pettingzoo_env.terminations[self._current_player]:
            opponent = "player_2"
            legal_moves = self._pettingzoo_env.legal_moves[opponent]

            if self.opponent is None:
                opponent_action = legal_moves[np.random.choice(legal_moves.shape[0])]
            else:
                state = mill.transition_model(self._pettingzoo_env)
                opponent_action = self.opponent.choose_move(state, self.total_moves_in_game)

            self._pettingzoo_env.step(opponent_action)
            self.total_moves_in_game += 1

        # Update legal moves for next step
        self.legal_moves = self._pettingzoo_env.legal_moves[self._current_player]

        # Get observation, reward, done status
        obs = self._decode_observation(None)
        reward = float(self._pettingzoo_env.rewards[self._current_player])  # Convert to float
        terminated = bool(self._pettingzoo_env.terminations[self._current_player])  # Convert to bool
        truncated = bool(self._pettingzoo_env.truncations[self._current_player])  # Convert to bool
        info = self._get_info()

        # Add legal moves to info for action masking
        info['legal_moves'] = self.legal_moves
        info['action_mask'] = self._create_action_mask()

        return obs, reward, terminated, truncated, info

    def _create_action_mask(self):
        """Create action mask for valid actions"""
        mask = np.zeros(self.action_space.n, dtype=np.float32)
        if self.legal_moves is not None:
            for move in self.legal_moves:
                # Simple hash to map MultiDiscrete to Discrete
                idx = move[0] * 625 + move[1] * 25 + move[2]
                idx = idx % self.action_space.n  # Ensure within bounds
                mask[idx] = 1.0
        return mask

    def _get_info(self):
        return {
            'legal_moves': self._pettingzoo_env.legal_moves[self._current_player],
            'phase': self._pettingzoo_env.infos[self._current_player]['phase']
        }

    def render(self):
        self._pettingzoo_env.render()

    def close(self):
        self._pettingzoo_env.close()