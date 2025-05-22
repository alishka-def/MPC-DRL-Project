import gymnasium as gym
import os
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise # Gaussian noise
from stable_baselines3.common.monitor import Monitor, load_results, get_monitor_files
from my_metanet_mpc_env import MetanetMPCEnv
from stable_baselines3.common.callbacks import BaseCallback

class CheckpointCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "ddpg_checkpoint", verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            checkpoint_file = os.path.join(self.save_path, f"{self.name_prefix}_{self.n_calls}_steps")
            self.model.save(checkpoint_file)
            if self.verbose > 0:
                print(f"Checkpoint saved: {checkpoint_file}")
        return True

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

def train_and_test_for_noise(label: str, var: np.ndarray, total_timesteps: int = 900 * 3000):
    # Train one DDPG agent under a given noise variance

    # preparing separate logging directory
    logdir = os.path.join("logs", label)
    os.makedirs(logdir, exist_ok= True)

    # building and wrapping env
    env = MetanetMPCEnv()
    # overriding the environment's internal noise before training
    env.noise_var = var.astype(np.float32)
    env.noise_std = np.sqrt(env.noise_var)

    env = Monitor(env, filename=os.path.join(logdir, "monitor.csv"), allow_early_resets=True)

    # building agent
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=np.ones(n_actions)*0.3
    )

    model = DDPG(
        "MlpPolicy",
        env=env,
        action_noise=action_noise,
        batch_size=512,
        buffer_size=int(2e5),
        learning_rate=1e-3,
        gamma=0.99,
        tau=0.01,
        policy_kwargs=dict(net_arch=[256,256]),
        verbose=1,
    )

    # callback for saving checkpoints along the way
    checkpoint_cb = CheckpointCallback(
        save_freq=100_000,  # adjust as needed
        save_path=os.path.join("logs", label, "checkpoints"),
        name_prefix=f"{label}_model",
        verbose=1
    )

    # training with decay
    decay_cb = NoiseDecayCallback(decay_rate=5e-6)
    model.learn(total_timesteps=total_timesteps, callback=[decay_cb, checkpoint_cb])

    # saving the agent
    model_path = f"ddpg_{label}"
    model.save(model_path)

    # test the trained model
    test_env = MetanetMPCEnv()
    obs, _ = test_env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = test_env.step(action)
        total_reward += reward
        done = terminated or truncated

    print(f"[{label}] test episode total reward: {total_reward}")

if __name__ == "__main__":
    # defining noise settings
    scenarios = {
        "low": np.array([75.0, 30.0]),
        "medium": np.array([150.0, 60.0]),
        "high": np.array([225.0, 90.0])
    }

    # train one agent per noise scenario
    for label, var in scenarios.items():
        train_and_test_for_noise(label, var)


# currently, the reward after 900*10 steps is -811806.6990715223.



