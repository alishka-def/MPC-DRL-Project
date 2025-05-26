import pandas as pd
import matplotlib.pyplot as plt

logdir = "logs/low"  # or medium, high
df = pd.read_csv(f"{logdir}/actual_demands.csv")

plt.figure()
plt.plot(df["Time [h]"], df["Mainline_Demand"], label="Mainline Demand")
plt.plot(df["Time [h]"], df["Onramp_Demand"], label="On-Ramp Demand")
plt.xlabel("Time [h]")
plt.ylabel("Demand [veh/h]")
plt.title("Actual Demands vs. Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"{logdir}/actual_demands_plot.png")
plt.show()