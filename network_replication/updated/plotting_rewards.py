import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ————— Parameters —————
monitor_file = "from_euler/logs/low/monitor.csv"
output_plot  = "plots/low_noise_rewards_per_episode.png"

# ————— 1) Load the CSV —————
# skiprows=1 drops only the JSON-comment line,
# the very next line becomes the header (r, l, t)
df = pd.read_csv(monitor_file, skiprows=1)

# ————— 2) Create an 'episode' column  —————
# one‐based episode numbering
df['episode'] = np.arange(1, len(df) + 1)

# ————— 3) Plot reward vs. episode —————
plt.figure(figsize=(10, 6))
plt.plot(df['episode'], df['r'], marker='.', linestyle='-', alpha=0.7, label='Reward per episode')
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Training Rewards per Episode")
plt.grid(True)
plt.legend()
plt.tight_layout()

# # ————— 4) Save & show —————
plt.savefig(output_plot, dpi=300)
plt.show()

