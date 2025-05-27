import os
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.monitor import Monitor
from my_metanet_mpc_env import MetanetMPCEnv
from stable_baselines3.common.callbacks import BaseCallback
import pandas as pd

BASE_LOG_DIR = "updated_logs"
class EpisodeCheckpointCallback(BaseCallback):
    def __init__(self, save_every: int, save_path: str, name_prefix: str = "ddpg_ep", verbose: int = 0):
        super().__init__(verbose)
        self.save_every = save_every
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.episode_count = 0
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if info.get("episode") is not None:
                self.episode_count += 1
                if self.episode_count % self.save_every == 0:
                    fname = f"{self.name_prefix}_{self.episode_count}_episodes"
                    path = os.path.join(self.save_path, fname)
                    self.model.save(path)
                    self.model.save_replay_buffer(path + "_buffer")
                    if self.verbose:
                        print(f"✔️  Saved checkpoint at episode {self.episode_count}: {path}.zip")
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
        current = getattr(noise, "sigma", None)
        if current is None:
            current = noise._sigma
        new_sigma = np.maximum(current - self.decay_rate, 0.0)
        if hasattr(noise, "sigma"):
            noise.sigma = new_sigma
        else:
            noise._sigma = new_sigma
        return True

def train_and_test_for_noise(label: str, var: np.ndarray, total_timesteps: int = 150 * 1500):
    # Train one DDPG agent under a given noise variance

    # preparing separate logging directory
    logdir = os.path.join(BASE_LOG_DIR, label)
    os.makedirs(logdir, exist_ok= True)

    # building and wrapping env
    env = MetanetMPCEnv()
    # overriding the environment's internal noise before training
    env.noise_var = var.astype(np.float32)
    env.noise_std = np.sqrt(env.noise_var)

    # Save ACTUAL (noisy) demands to CSV for later plotting
    demands_df = pd.DataFrame(
        data=env.demands_actual,
        columns=["Mainline_Demand", "Onramp_Demand"]
    )
    demands_df["Time [h]"] = env.time
    demands_df.to_csv(os.path.join(logdir, "actual_demands.csv"), index=False)

    env = Monitor(env, filename=os.path.join(logdir, "monitor.csv"), allow_early_resets=True)

    # building agent
    n_actions = env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions),
        sigma=np.ones(n_actions)*0.3, # initial σ=0.3
        theta=0.15,
        dt=1.0
    )

    initial_lr = 1e-4
    final_lr = 1e-5
    lr_schedule = lambda progress: final_lr + (initial_lr - final_lr) * progress
    model = DDPG(
        "MlpPolicy",
        env=env,
        action_noise=action_noise,
        batch_size=256,
        buffer_size=int(2e5),
        learning_rate=lr_schedule,
        gamma=0.99,
        tau=0.005,
        policy_kwargs=dict(net_arch=[256,256]),
        verbose=1,
    )

    # callback for saving checkpoints along the way
    checkpoint_cb = EpisodeCheckpointCallback(
        save_every=100,  # save every 100 episodes
        save_path=os.path.join(BASE_LOG_DIR, label, "checkpoints"),
        name_prefix=f"{label}_ep",
        verbose=1
    )

    # training with decay
    decay_cb = NoiseDecayCallback(decay_rate=5e-6)
    model.learn(total_timesteps=total_timesteps, callback=[decay_cb, checkpoint_cb])

    # saving the agent
    final_path = os.path.join(BASE_LOG_DIR, label, f"ddpg_{label}_final")
    model.save(final_path)
    print(f"Final model saved to {final_path}.zip")

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
        "low": np.array([75.0, 30.0])
        #"medium": np.array([150.0, 60.0]),
        #"high": np.array([225.0, 90.0])
    }

    # train one agent per noise scenario
    for label, var in scenarios.items():
        train_and_test_for_noise(label, var)




