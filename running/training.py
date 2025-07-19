import os
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.monitor import Monitor
from my_metanet_mpc_env import MetanetMPCEnv
from stable_baselines3.common.callbacks import BaseCallback
import datetime
import uuid

########################################################################
# Configurations
########################################################################
BASE_LOG_DIR = "../logs"
TOTAL_EPISODES = 3000
EXPECTED_EPISODE_LENGTH = 150 # DRL steps per episode during main simulation only

# Episodes timing breakdown:
# Warm-up: 30 minutes (180 simulation steps, no DRL)
# Main simulation: 150 DRL steps (150*6*10 = 9000s = 2.5h)
# Total episode: 3 hours including warm-up

class EpisodeCheckpointCallback(BaseCallback):
    """Save model checkpoints every N episodes"""
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
                        print(f"Saved checkpoint at episode {self.episode_count}: {path}.zip")
        return True

class NoiseDecayCallback(BaseCallback):
    """This callback reduces the exploration noise (sigma) linearly from
    initial_sigma to final_sigma over the course of training to encourage exploitation
    as the agent becomes more experienced."""

    def __init__(self, initial_sigma: float = 0.3, final_sigma: float = 0.01,
                 total_timesteps: int = None, verbose: int = 0):
        super().__init__(verbose)
        self.initial_sigma = initial_sigma # Starting exploration noise level
        self.final_sigma = final_sigma # Ending exploration noise level
        self.total_timesteps = total_timesteps # Total training time steps for decay schedule
        self.current_timestep = 0 # Counter for current time steps


    def _on_step(self) -> bool:
        # Increment timestep counter each time this callback is called
        self.current_timestep += 1

        # Only apply decay if total time steps is specified
        if self.total_timesteps is not None:
            # calculate training progress as fraction from 0.0 to 1.0
            progress = min(self.current_timestep / self.total_timesteps, 1.0)

            # Linear interpolation: start at initial_sigma, end at final_sigma
            new_sigma = self.initial_sigma * (1 - progress) + self.final_sigma * progress

            # Access the model's action noise object and update its sigma parameter
            noise = self.model.action_noise
            if hasattr(noise, 'sigma'):
                noise.sigma = new_sigma
            elif hasattr(noise, '_sigma'):
                noise._sigma = new_sigma

        return True

def create_environment(noise_variance: np.ndarray) -> MetanetMPCEnv:
    env = MetanetMPCEnv(
        M_drl=6, # drl frequency: every 6 simulation steps
        w_u=0.4, # action scaling parameter
        w_p=10.0 # queue violation penalty weight
    )
    # setting demand noise variance
    # [mainline_variance, onramp_variance]
    env.noise_var = noise_variance.astype(np.float32)
    env.noise_std = np.sqrt(env.noise_var)

    return env

def create_ddpg_agent(env: MetanetMPCEnv, total_timesteps: int) -> DDPG:
    # action space ( should be 3: 1 ramp + 2 VSL controls)
    n_actions = env.action_space.shape[-1]

    # Ornstein-Uhlenbeck noise for continuous action exploration
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions),
        sigma=np.ones(n_actions) * 0.3,  # initial σ=0.3
        theta=0.15, # mean reversion rate (how quickly noise returns to mean)
        dt=1.0 # time step for noise process
    )

    # Learning rate schedule: start high for exploration
    initial_lr = 1e-4 # starting learning rate
    final_lr = 1e-5 # ending learning rate
    # Lambda function for linear decay based on training progress
    lr_schedule = lambda progress: final_lr + (initial_lr - final_lr) * (1-progress)

    # Creating ddpg agent
    model = DDPG(
        "MlpPolicy", # multi-layer perceptron policy network
        env=env, # training environment
        action_noise=action_noise, # exploration noise for actions
        batch_size=256, # number of samples per gradient update
        buffer_size=int(2e5), # replay buffer size
        learning_rate=lr_schedule, # learning rate schedule
        gamma=0.99, # discount factor for future rewards
        n_steps=10,  # using 10 steps returns for TD update
        tau=0.015, # increased soft update factor for target networks
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1, # print training progress
    )
    return model

def train_agent(label: str, noise_variance: np.ndarray, total_timesteps: int):
    print(f"Training DDPG agent: {label.upper()}")
    print(f"Noise variance: {noise_variance}")
    print(f"Total time steps: {total_timesteps:,}")
    print(f"Expected episodes: {total_timesteps // EXPECTED_EPISODE_LENGTH}")

    # create a unique timestamp and identifier for this training run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{label}_{timestamp}_{uuid.uuid4().hex[:8]}"

    # creating separate logging directory
    logdir = os.path.join(BASE_LOG_DIR, run_id)
    os.makedirs(logdir, exist_ok=True)

    # create environment
    # the environment automatically handles warm-up before each episode
    env = create_environment(noise_variance)
    env = Monitor(env, filename=os.path.join(logdir, "monitor.csv"), allow_early_resets=True)
    model = create_ddpg_agent(env, total_timesteps)

    # Checkpoint callback: saves model and replay buffer every N episodes)
    checkpoint_cb = EpisodeCheckpointCallback(
        save_every=500,
        save_path=os.path.join(logdir, "checkpoints"),
        name_prefix=f"{run_id}_ep",
        verbose=1 # print when checkpoints are saved
    )
    # noise decay callback
    noise_decay_cb = NoiseDecayCallback(
        initial_sigma=0.3,
        final_sigma=0.01,
        total_timesteps=total_timesteps,
        verbose=0
    )

    # training the agent with callbacks
    model.learn(
        total_timesteps=total_timesteps,
        callback=[noise_decay_cb, checkpoint_cb],
        progress_bar=True
    )

    # save final trained model and its replay buffer
    final_path = os.path.join(logdir, f"ddpg_{run_id}_final")
    model.save(final_path)
    model.save_replay_buffer(final_path + "_buffer")

    # test the trained agent on a fresh episode
    test_reward = test_agent(model, noise_variance)
    print(f"[{label}] Final test reward: {test_reward:.2f}")

    env.close()
    return model, run_id

def evaluate_agent(model: DDPG, noise_variance: np.ndarray) -> float:
    """
    This creates a new environment instance and rusn one complete
    episode to evaluate the agent's performance.
    Returns the total reward accumulated during the main
    simulation period.
    """

    # create fresh test environment with same noise characteristics
    test_env = create_environment(noise_variance)

    # reset environment
    obs, _ = test_env.reset()

    # initializing test episode tracking variables
    total_reward = 0.0
    done = False
    steps = 0

    # run one complete episode with trained policy
    while not done and steps <= EXPECTED_EPISODE_LENGTH :
        # get deterministic action from trained policy (no exploration noise)
        action, _ = model.predict(obs, deterministic=True)

        # execute action in environment and observe results
        obs, reward, terminated, truncated, _ = test_env.step(action)

        # accumulate total reward
        total_reward += reward
        done = terminated or truncated
        steps += 1

    # clean up test environment
    test_env.close()
    return total_reward

def main():
    # main function that coordinates the entire training process
    # 150 DRL steps per episode x 3000 episodes = 450,000 time steps
    # warm-up period is automatic, but doesn't count towards the total time steps
    total_timesteps = EXPECTED_EPISODE_LENGTH * TOTAL_EPISODES
    # only displaying low scenario
    scenario = {
        "low": np.array([75.0, 30.0])
    }
    trained_model = {}
    for label, noise_var in scenario.items():
        model, run_id = train_agent(label, noise_var, total_timesteps)
        trained_model[run_id] = model

    return trained_model

if __name__ == "__main__":
    trained_model = main()