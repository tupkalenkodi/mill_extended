import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from mill_env import env as mill_env_creator
from mill_rllib_env import MillRLlibEnv

ENV_NAME = "mill-rllib"

def env_creator(config=None):
    return MillRLlibEnv(mill_env_creator)

if __name__ == "__main__":
    ray.init()

    register_env(ENV_NAME, env_creator)

    config = (
        PPOConfig()
        .environment(ENV_NAME)
        .framework("torch")
        .rollouts(num_rollout_workers=4)
        .training(
            gamma=0.99,
            lr=3e-4,
            train_batch_size=4000,
            model={
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            }
        )
        .multi_agent(
            policies={"shared"},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared",
        )
        .resources(num_gpus=0)
    )

    algo = config.build()

    for i in range(50):  # MVP training
        result = algo.train()
        print(f"Iter {i}: reward {result['episode_reward_mean']}")

    checkpoint = algo.save("checkpoints")
    print("Saved checkpoint:", checkpoint)

    ray.shutdown()
