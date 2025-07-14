import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ————— Parameters —————
monitor_file = "../logs/low/monitor.csv"
output_plot  = "../plots/low_noise_rewards_per_episode_low.png"

# ————— 1) Load the CSV —————
df = pd.read_csv(monitor_file, skiprows=1)
df['episode'] = np.arange(1, len(df) + 1)

x = df['episode'].values
y = df['r'].values

# ————— 2) Fit a polynomial of degree 2 (quadratic) —————
deg = 2
coeffs = np.polyfit(x, y, deg=deg)        # returns [a, b, c] for ax^2 + bx + c
poly   = np.poly1d(coeffs)
y_fit  = poly(x)

# If you wanted cubic, just do deg=3 above.

# ————— 3) Plot —————
plt.figure(figsize=(10, 6))
plt.plot(x, y,         marker='.', linestyle='-', alpha=0.6, label='Reward per episode')
plt.plot(x, y_fit,     color='magenta', linewidth=2, label=f'Poly‐deg{deg} fit')

# Optional: show the equation
a, b, c = coeffs
plt.title(f"Rewards with quadratic fit: y = {a:.2e} x² + {b:.2e} x + {c:.2f}")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(output_plot, dpi=300)
plt.show()
