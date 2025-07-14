import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ————— Parameters —————
monitor_file = "../logs/low/monitor.csv"
output_plot  = "../plots/no_noise_rewards_per_10_episodes_low.png"
block_size   = 10  # average every 10 episodes

# ————— 1) Load the CSV —————
# Note: stable-baselines Monitor CSV has a header line we skip
df = pd.read_csv(monitor_file, skiprows=1)
# each row in this file corresponds to one episode
df['episode'] = np.arange(1, len(df) + 1)

# ————— 2) Assign each episode to a 10-episode block —————
# block 1: episodes 1–10, block 2: 11–20, etc.
df['block'] = ((df['episode'] - 1) // block_size) + 1

# ————— 3) Compute mean reward in each block —————
grouped = (
    df
    .groupby('block', as_index=False)
    .agg(mean_reward=('r', 'mean'))
)
# for plotting it’s often nice to label by the *last* episode in the block:
grouped['block_end_ep'] = grouped['block'] * block_size

# ————— 4) Plot —————
plt.figure(figsize=(10, 6))
plt.plot(
    grouped['block_end_ep'],
    grouped['mean_reward'],
    marker='o',
    linestyle='-',
    label=f"Mean reward per {block_size} eps."
)
plt.title(f"Average Reward Every {block_size} Episodes")
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(output_plot, dpi=300)
plt.show()
