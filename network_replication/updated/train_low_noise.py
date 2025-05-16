import gymnasium as gym
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise # Gaussian noise
from my_metanet_mpc_env import MetanetMPCEnv
from stable_baselines3.common.callbacks import BaseCallback

class NoiseDecayCallback(BaseCallback):
    """
        After every env step, reduce the action_noise.sigma by decay_rate,
        but never below zero. We do this manually, since there's no in-built function for it.
        """

    def __init__(self, decay_rate: float, verbose=0):
        super().__init__(verbose)
        self.decay_rate = decay_rate  # e.g. 5e-6 per step

    def _on_step(self) -> bool:
        # access your model's exploration noise
        noise = self.model.action_noise
        # subtract the decay (vectorized)
        new_sigma = np.maximum(noise._sigma - self.decay_rate, 0.0)
        noise.sigma = new_sigma
        return True


def main():
    env = MetanetMPCEnv()
    n_actions = env.action_space.shape[-1] # 1 action
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.3 * np.ones(n_actions)) # Gaussian noise

    # DDPG agent (values taken from Table 4)
    model = DDPG(
        policy="MlpPolicy",
        env=env,
        action_noise=action_noise,
        verbose=1,
        batch_size=512,
        buffer_size=int(2e5),
        learning_rate=0.001,
        gamma= 0.99,
        tau=0.01,
        policy_kwargs=dict(net_arch=[256,256],)
    )

    # Training the agent
    # M = 3000 episodes, each episode is 9000 s/ 10 s = 900 steps
    # total time steps = 900 * 3000
    total_timesteps = 900 * 10
    decay_cb = NoiseDecayCallback(decay_rate=5e-6)
    model.learn(total_timesteps=total_timesteps, callback=decay_cb)

    # Saving the model
    model.save("ddpg_low_noise")
    print("Training complete.")

    # Run the test (after training)
    test_env = MetanetMPCEnv()
    obs, _ = test_env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = test_env.step(action)
        total_reward += reward
        done = terminated or truncated

    print(f"test episode total reward: {total_reward}")


if __name__ == "__main__":
    main()

# currently, the reward after 900*10 steps is -811806.6990715223.



