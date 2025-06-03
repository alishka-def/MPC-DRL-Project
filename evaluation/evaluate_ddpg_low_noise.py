import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DDPG
from my_metanet_mpc_env import MetanetMPCEnv

if __name__ == "__main__":
    # Instantiate env
    env = MetanetMPCEnv()
    # Load trained policy
    model = DDPG.load("../updated_logs/low/ddpg_low_final.zip")

    # Reset once, get initial observation
    obs, _ = env.reset()
    Num_Steps = 150

    # Rollout using the learned policy instead of random
    for _ in range(Num_Steps):
        a, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action=a)
        if truncated or done:
            break

    # Stack up results
    env.sim_results["Density"] = np.stack(env.sim_results["Density"], axis=-1)
    env.sim_results["Speed"] = np.stack(env.sim_results["Speed"], axis=-1)
    env.sim_results["Queue_Length"] = np.stack(env.sim_results["Queue_Length"], axis=-1)
    env.sim_results["Flow"] = np.stack(env.sim_results["Flow"], axis=-1)
    env.sim_results["Origin_Flow"] = np.stack(env.sim_results["Origin_Flow"], axis=-1)
    env.sim_results["VSL"] = np.stack(env.sim_results["VSL"], axis=-1)
    env.sim_results["Ramp_Metering_Rate"] = np.stack(env.sim_results["Ramp_Metering_Rate"], axis=-1)
    env.sim_results["u_MPC"] = np.stack(env.sim_results["u_MPC"], axis=-1)
    env.sim_results["u_DRL"] = np.stack(env.sim_results["u_DRL"], axis=-1)

    # Plots
    time = env.time[:env.sim_results["Density"].shape[1]]

    plt.figure()
    plt.plot(time, env.sim_results["Density"].T)
    plt.xlabel("Time [h]")
    plt.ylabel("Density [veh/km/lane]")
    plt.savefig("plots/euler_density.png")

    plt.figure()
    plt.plot(time, env.sim_results["u_MPC"][0, :], label="MPC Baseline")
    plt.plot(time, env.sim_results["Ramp_Metering_Rate"].T, label="Combined Input")
    plt.legend()
    plt.xlabel("Time [h]")
    plt.ylabel("Ramp Metering Rate [-]")
    plt.savefig("plots/amp_metering_rate.png")

    plt.show()

