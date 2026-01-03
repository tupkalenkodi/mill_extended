import numpy as np
from gymnasium import spaces
from pettingzoo.utils import aec_to_parallel
from ray.rllib.env.multi_agent_env import MultiAgentEnv

PHASE_TO_INT = {
    "placing": 0,
    "moving": 1,
    "flying": 2,
    "lost": 3
}

ACTION_DIM = 25 * 25 * 25  # flattened MultiDiscrete

def flatten_action(a):
    src, dst, cap = a
    return src * 25 * 25 + dst * 25 + cap

def unflatten_action(i):
    src = i // (25 * 25)
    rem = i % (25 * 25)
    dst = rem // 25
    cap = rem % 25
    return np.array([src, dst, cap], dtype=np.int64)


class MillRLlibEnv(MultiAgentEnv):
    def __init__(self, env_creator):
        self.env = aec_to_parallel(env_creator())
        self.agents = self.env.possible_agents

        self.observation_space = spaces.Dict({
            "obs": spaces.Box(low=0, high=9, shape=(30,), dtype=np.int32),
            "action_mask": spaces.Box(0, 1, shape=(ACTION_DIM,), dtype=np.int8),
        })

        self.action_space = spaces.Discrete(ACTION_DIM)

    def reset(self, *, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        return self._build_obs(obs), infos

    def step(self, action_dict):
        decoded = {
            agent: unflatten_action(a)
            for agent, a in action_dict.items()
        }

        obs, rewards, terms, truncs, infos = self.env.step(decoded)
        return (
            self._build_obs(obs),
            rewards,
            terms,
            truncs,
            infos
        )

    def _build_obs(self, obs_dict):
        out = {}
        for agent, board in obs_dict.items():
            info = self.env.infos[agent]
            opp = info["agent"]

            model = self.env.unwrapped._model
            agent_idx = self.env.unwrapped.agent_index[agent]
            opp_idx = 3 - agent_idx

            obs = np.concatenate([
                board.astype(np.int32),
                np.array([
                    agent_idx,
                    model._player[agent_idx]["pieces_holding"],
                    model._player[opp_idx]["pieces_holding"],
                    PHASE_TO_INT[model._player[agent_idx]["phase"]],
                    PHASE_TO_INT[model._player[opp_idx]["phase"]],
                ], dtype=np.int32)
            ])

            mask = np.zeros(ACTION_DIM, dtype=np.int8)
            for m in info["legal_moves"]:
                mask[flatten_action(m)] = 1

            out[agent] = {
                "obs": obs,
                "action_mask": mask
            }

        return out
