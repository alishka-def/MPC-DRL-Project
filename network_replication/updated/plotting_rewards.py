import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ————— Parameters —————
monitor_file = "updated_logs/low/monitor.csv"
output_plot  = "plots/low_updated_2_noise_rewards_per_episode.png"

# ————— 1) Load the CSV —————
df = pd.read_csv(monitor_file, skiprows=1)
df['episode'] = np.arange(1, len(df) + 1)

# ————— 2) Compute linear trend —————
# fit r = m * episode + b
m, b = np.polyfit(df['episode'], df['r'], deg=1)
trend = m * df['episode'] + b

# ————— 3) Plot rewards and trend line —————
plt.figure(figsize=(10, 6))
plt.plot(df['episode'], df['r'],
         marker='.', linestyle='-', alpha=0.6, label='Reward per episode')
plt.plot(df['episode'], trend,
         color='magenta', linewidth=2, label=f'Trend: y={m:.2e}·x+{b:.1f}')

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Training Rewards per Episode with Linear Trend")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(output_plot, dpi=300)
plt.show()
