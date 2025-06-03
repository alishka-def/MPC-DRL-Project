########################################################################
# Imports
########################################################################
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

import casadi as cs
import sym_metanet as metanet

from csnlp import Nlp
from csnlp.wrappers import Mpc
from stable_baselines3 import DDPG
from matplotlib.patches import Rectangle
from typing import Tuple

########################################################################
# Global: Parameters
########################################################################
RUN_NO_CTRL = True
RUN_MPC_CTRL = True
RUN_MPC_DRL_CTRL = True

DEMAND_NOISE = "LOW" # options: None or "LOW" or "MEDIUM" or "HIGH"

########################################################################
# Methods
########################################################################
def create_demands(time: np.ndarray) -> np.ndarray:
    return np.stack(
        (
            np.interp(time, (0, 15/60, 2.50, 2.75, 3.00, 3.25), (0, 3500, 3500, 1000, 1000, 0)), 
            np.interp(time, (0, 15/60, 30/60, 0.60, 0.85, 1.0, 3.00, 3.25), (0, 500, 500, 1500, 1500, 500, 500, 0))
        )
    )


def compute_metrics(density: np.ndarray, flow: np.ndarray, queue_lengths, num_lanes: int, L: float, T: float) -> Tuple[float, float]:
    vkt = T * sum((q * L * num_lanes).sum() for q in flow)
    vht = T * sum(w.sum() + (rho * L * num_lanes).sum() for rho, w in zip(density, queue_lengths))
    return vkt, vht


def normalize_observations(rho: np.ndarray, v: np.ndarray, w: np.ndarray, 
                           u_mpc: np.ndarray, u_prev: np.ndarray, demands: np.ndarray) -> np.ndarray:
        """
        Normalize all components into [0,1]:
            rho_norm = rho/ rho_max
            v_norm = v/ v_free
            w_norm = w/ max_queue
            u_mpc_norm = [ramp_rate, vsl/ v_free]
            u_prev_norm = [prev_ramo, prev_vsl/ v_free]
            d_norm = demands/ max_demands
        Returns concentrated observation vector.
        """
        rho_max, v_free= 180.0, 102.0
        rho_norm = np.asarray(rho).flatten() / rho_max
        v_norm = np.asarray(v).flatten() / v_free
        w_norm = np.asarray(w).flatten() / np.array([200.0, 100.0])

        u_mpc_norm = u_mpc / np.array([1, v_free, v_free])
        u_prev_norm = u_prev / np.array([1, v_free, v_free])

        d_norm = demands / np.array([3500.0, 1500.0])
        return np.concatenate([rho_norm, v_norm, w_norm, u_mpc_norm, d_norm, u_prev_norm], dtype=np.float32)

########################################################################
# Main: Parameters
########################################################################
T = 10.0 / 3600 # temporal discretization (hours)
T_warmup = 30.0 / 60 # initial warm-up period of 30 minutes
T_sim = 2.5 # simulation time of 2.5 hours
T_cooldown = 45.0 / 60 # final cool-down period of 45 minutes
time = np.arange(0, T_warmup+T_sim+T_cooldown+T, T)
demands_forecast = create_demands(time).T # size = (900, 2)

if DEMAND_NOISE is None:
    demand_noise = np.zeros_like(demands_forecast)
else:
    if DEMAND_NOISE == "LOW":
        noise_std = np.array([75.0, 30.0])
    elif DEMAND_NOISE == "MEDIUM":
        noise_std = np.array([150.0, 60.0])
    elif DEMAND_NOISE == "HIGH":
        noise_std = np.array([225.0, 90.0])
    else:
        raise NotImplementedError
    demand_noise = np.random.normal(loc=0.0, scale=noise_std, size=demands_forecast.shape)
    demand_noise[:int(T_warmup/T), :] = 0
    demand_noise[-int(T_cooldown/T):, :] = 0

demands = demands_forecast + demand_noise

plt.figure()
plt.plot(time, demands[:,0], alpha=0.5, color="tab:blue") # label="Mainline (with noise)"
plt.plot(time, demands[:,1], alpha=0.5, color="tab:orange") # label="Onramp (with noise)"
plt.plot(time, demands_forecast[:,0], color="tab:blue", label="Mainline")
plt.plot(time, demands_forecast[:,1], color="tab:orange", label="Onramp")
plt.gca().add_patch(
    Rectangle(xy=[0,-50], width=T_warmup, height=4500, alpha=0.25, color="gray")
)
plt.text(T_warmup/18, 3800, "Warm Up", fontstyle="italic", fontsize=9)
plt.gca().add_patch(
    Rectangle(xy=[T_warmup+T_sim,-50], width=T_cooldown, height=4500, alpha=0.25, color="gray")
)
plt.text(T_warmup+T_sim+T_cooldown/6, 3800, "Cool Down", fontstyle="italic", fontsize=9)
plt.ylim([-50, 4000])
plt.xlabel("Time [h]")
plt.ylabel("Demands [veh/h]")
plt.legend()
plt.tight_layout()
#plt.show()
#sys.exit(1)

L = 1.0 # spatial discretization (km)
num_lanes = 2
tau = 18.0 / 3600
kappa = 40.0
eta = 60.0
delta = 0.0122
a = 1.867
rho_max = 180.0
rho_crit = 33.5
capacity_lane = 2000.0
v_free = 102.0

########################################################################
# Main: Network
########################################################################
N1 = metanet.Node(name="N1")
N2 = metanet.Node(name="N2")
N3 = metanet.Node(name="N3")

O1 = metanet.MainstreamOrigin[cs.SX](name="O1")
O2 = metanet.MeteredOnRamp[cs.SX](capacity_lane, name="O2")

D1 = metanet.Destination[cs.SX](name="D1")

L1 = metanet.Link[cs.SX](4, num_lanes, L, rho_max, rho_crit, v_free, a, name="L1") # num_segments = 4
L2 = metanet.Link[cs.SX](2, num_lanes, L, rho_max, rho_crit, v_free, a, name="L1") # num_segments = 2

net = (
    metanet.Network(name="A1")
    .add_path(origin=O1, path=(N1, L1, N2, L2, N3), destination=D1)
    .add_origin(O2, N2)
)

metanet.engines.use("casadi", sym_type="SX")
net.is_valid(raises=True)
net.step(T=T, tau=tau, eta=eta, kappa=kappa, delta=delta)

########################################################################
# Main: Simulation in Open Loop
########################################################################
if RUN_NO_CTRL:
    print("**************************** OPEN-LOOP ****************************")
    F = metanet.engine.to_function(
        net=net, more_out=True, compact=1, T=T
    )
    # F: (rho[6], v[6], w[2], v_ctrl, r, d[2]) -> (rho+[6], v+[6], w+[2], q[6], q_o[2])

    # initial conditions (Highway initially empty)
    rho = cs.DM([0, 0, 0, 0, 0, 0])
    v = cs.DM([v_free, v_free, v_free, v_free, v_free, v_free]) # cs.DM([80, 80, 78, 72.5, 66, 62])
    w = cs.DM([0, 0]) # queue lengths at origin nodes O1 and O2

    # open loop (i.e. without VSL and RM)
    v_ctrl = cs.DM.inf(1, 1) # control speed at O1
    r = cs.DM.ones(1, 1) # ramp metering rate at O2

    density, speed, queue_lengths, flow, origin_flow = [], [], [], [], []
    for d in demands:
        rho, v, w, q, q_o = F(rho, v, w, v_ctrl, r, d)
        density.append(rho)
        speed.append(v)
        queue_lengths.append(w)
        flow.append(q)
        origin_flow.append(q_o)
    density, speed, queue_lengths, flow, origin_flow = (np.squeeze(o) for o in (density, speed, queue_lengths, flow, origin_flow))

    # compute VKT and VHT metrics
    vkt, vht = compute_metrics(density, flow, queue_lengths, num_lanes, L, T)
    print(f"VKT = {vkt:.4f} veh.km \nVHT or TTS = {vht:.4f} veh.h \n Avg Speed = {vkt/vht:.4f} km/h")


    # plotting
    _, axs = plt.subplots(3, 2, constrained_layout=True, sharex=True)
    axs[0, 0].plot(time, speed)
    axs[0, 0].set_ylabel("Speed (km/h)")
    axs[0, 0].set_xlabel("Time (h)")
    axs[0, 0].set_xlim(0, T_sim+T_warmup+T_cooldown)

    axs[0, 1].plot(time, flow)
    axs[0, 1].set_ylabel("Flow (veh/h)")

    axs[1, 0].plot(time, density)
    axs[1, 0].set_ylabel("Density (veh/km/lane)")

    axs[1, 1].plot(time, demands)
    axs[1, 1].set_ylabel("Origin Demands (veh/h)")

    axs[2, 0].plot(time, origin_flow)
    axs[2, 0].set_ylabel("Origin Flow (veh/h)")

    axs[2, 1].plot(time, queue_lengths)
    axs[2, 1].set_ylabel("Queue Lengths (veh)")

    for ax in axs.flat:
        ax.set_ylim(0, ax.get_ylim()[1])

    n_seg = 6
    X, Y = np.meshgrid(time * 3600, (np.arange(n_seg) + 1) * L)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    sc = axs[0].pcolormesh(X, Y, speed.T, shading='auto', cmap='jet_r')
    plt.colorbar(sc, label='Speed (km/h)', ax=axs[0])
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Position (km)')

    sc = axs[1].pcolormesh(X, Y, density.T, shading='auto', cmap='jet')
    plt.colorbar(sc, label='Density (veh/km/lane)', ax=axs[1])
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Position (km)')

    sc = axs[2].pcolormesh(X, Y, flow.T, shading='auto', cmap='jet_r')
    plt.colorbar(sc, label='Flow (veh/h)', ax=axs[2])
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Position (km)')

    fig.suptitle('Time-Space Diagrams for No Control')
    fig.tight_layout()

########################################################################
# Main: Simulation in Closed Loop For Ramp Metering and VSL
########################################################################
if RUN_MPC_CTRL:
    print("**************************** MPC: RAMP METERING + VSL ****************************")
    L1 = metanet.LinkWithVsl(4, num_lanes, L, rho_max, rho_crit, v_free, a, name="L1",
                            segments_with_vsl={2, 3}, alpha=0.1) # alpha = Non-compliance factor to the indicated speed limit.
    net = (
        metanet.Network(name="A1")
        .add_path(origin=O1, path=(N1, L1, N2, L2, N3), destination=D1)
        .add_origin(O2, N2)
    )
    metanet.engines.use("casadi", sym_type="SX")
    net.is_valid(raises=True)
    net.step(T=T, tau=tau, eta=eta, kappa=kappa, delta=delta, init_conditions={O1: {"v_ctrl": v_free*2}})
    F = metanet.engine.to_function(
        net=net, more_out=True, compact=2, T=T
    )
    # F: (x[14], u[3], d[2]) -> (x+[14], q[8])

    # create MPC controller
    # T_c (sampling interval of MPC) = 300 s = M * T_s; M = 30, T_s = 10 s
    # Np = 2 for 600 s prediction horizon
    Np, Nc, M = 2, 2, 30
    mpc = Mpc[cs.SX](
        nlp=Nlp[cs.SX](sym_type="SX"),
        prediction_horizon=Np*M,
        control_horizon=Nc*M,
        input_spacing=M
    )
    # create states, actions and disturbance for MPC
    n_seg, n_orig = sum(link.N for _, _, link in net.links), len(net.origins)
    rho, _ = mpc.state("rho", n_seg, lb=0)
    v, _ = mpc.state("v", n_seg, lb=0)
    w, _ = mpc.state("w", n_orig, lb=0, ub=[[200], [100]]) # O2 queue is constrained
    v_ctrl, _ = mpc.action("v_ctrl", len(L1.vsl), lb=20, ub=v_free)
    r, _ = mpc.action("r", lb=0, ub=1)
    d = mpc.disturbance("d", n_orig)
    # set dynamics constraints
    mpc.set_dynamics(F)
    # set objective function
    v_ctrl_last = mpc.parameter("v_ctrl_last", (v_ctrl.size1(), 1))
    r_last = mpc.parameter("r_last", (r.size1(), 1))
    mpc.minimize(
        T * cs.sum2(cs.sum1(rho * L * num_lanes) + cs.sum1(w)) 
        + 0.4*cs.sumsqr(cs.diff(cs.horzcat(r_last, r)))
        + 0.4*cs.sumsqr(cs.diff(cs.horzcat(v_ctrl_last, v_ctrl), 1, 1) / v_free)
    )
    # set solver for MPC NLP
    opts = {
        "expand": True,
        "print_time": False,
        "ipopt": {"max_iter": 500, "sb": "yes", "print_level": 0},
    }
    mpc.init_solver(solver="ipopt", opts=opts)

    # initial conditions (Highway initially empty)
    rho = cs.DM([0, 0, 0, 0, 0, 0])
    v = cs.DM([v_free, v_free, v_free, v_free, v_free, v_free]) # cs.DM([80, 80, 78, 72.5, 66, 62])
    w = cs.DM([0, 0]) # queue lengths at origin nodes O1 and O2
    v_ctrl_last = v[:L1.N][L1.vsl]
    r_last = cs.DM.ones(r.size1(), 1)
    sol_prev = None

    density, speed, queue_lengths, flow, origin_flow, r_ctrl, vsl_ctrl = [], [], [], [], [], [], []
    for k in range(demands.shape[0]):
        # get demand forecast
        d_hat = demands_forecast[k:k+Np*M, :]
        if d_hat.shape[0] < Np*M:
            d_hat = np.pad(d_hat, ((0, Np*M-d_hat.shape[0]), (0, 0)), "edge")
        # solve MPC problem every M steps
        if k >= int(T_warmup//T) and k <= int((T_warmup+T_sim)//T) and k % M == 0:
            sol = mpc.solve(
                pars={"rho_0": rho, "v_0": v, "w_0": w, "d": d_hat.T, "v_ctrl_last": v_ctrl_last, "r_last": r_last},
                vals0=sol_prev,
            )
            v_ctrl_last = sol.vals["v_ctrl"][:, 0]
            r_last = sol.vals["r"][0]
            # Handle NaN mpc outputs by retaining previous solution
            if np.any(np.isnan(v_ctrl_last)):
                v_ctrl_last[np.isnan(v_ctrl_last)] = sol_prev["v_ctrl"][np.isnan(v_ctrl_last), 0]
            if np.isnan(r_last):
                r_last = sol_prev["r"][0]
            sol_prev = sol.vals
        if k > int((T_warmup+T_sim)//T):
            v_ctrl_last = v_free*cs.DM.ones(len(L1.vsl), 1)
            r_last = cs.DM.ones(r.size1(), 1)
        # step dynamics
        x_next, q_all = F(cs.vertcat(rho, v, w), cs.vertcat(v_ctrl_last, r_last), demands[k, :])
        rho, v, w = cs.vertsplit(x_next, (0, n_seg, 2*n_seg, 2*n_seg+n_orig))
        q, q_o = cs.vertsplit(q_all, (0, n_seg, n_seg+n_orig))
        density.append(rho)
        speed.append(v)
        queue_lengths.append(w)
        flow.append(q)
        origin_flow.append(q_o)
        r_ctrl.append(r_last)
        vsl_ctrl.append(v_ctrl_last)
        if k % 100 == 0:
            print(f"Step {k} of {demands.shape[0]}.")
    density, speed, queue_lengths, flow, origin_flow, r_ctrl, vsl_ctrl = (np.squeeze(o) for o in (density, speed, queue_lengths, flow, origin_flow, r_ctrl, vsl_ctrl))

    # compute VKT and VHT metrics
    vkt, vht = compute_metrics(density, flow, queue_lengths, num_lanes, L, T)
    print(f"VKT = {vkt:.4f} veh.km \nVHT or TTS = {vht:.4f} veh.h \n Avg Speed = {vkt/vht:.4f} km/h")

    # plotting
    _, axs = plt.subplots(4, 2, constrained_layout=True, sharex=True)
    axs[0, 0].plot(time, speed)
    axs[0, 0].set_ylabel("Speed (km/h)")
    axs[0, 0].set_xlabel("Time (h)")
    axs[0, 0].set_xlim(0, T_sim+T_warmup+T_cooldown)

    axs[0, 1].plot(time, flow)
    axs[0, 1].set_ylabel("Flow (veh/h)")

    axs[1, 0].plot(time, density)
    axs[1, 0].set_ylabel("Density (veh/km/lane)")

    axs[1, 1].plot(time, demands)
    axs[1, 1].set_ylabel("Origin Demands (veh/h)")

    axs[2, 0].plot(time, origin_flow)
    axs[2, 0].set_ylabel("Origin Flow (veh/h)")

    axs[2, 1].plot(time, queue_lengths)
    axs[2, 1].set_ylabel("Queue Lengths (veh)")

    axs[3, 0].plot(time, r_ctrl)
    axs[3, 0].set_ylabel("Ramp metering rate (-)")

    axs[3, 1].plot(time, vsl_ctrl)
    axs[3, 1].set_ylabel("VSL (km/h)")

    for ax in axs.flat:
        ax.set_ylim(0, ax.get_ylim()[1])

    n_seg = 6
    X, Y = np.meshgrid(time * 3600, (np.arange(n_seg) + 1) * L)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    sc = axs[0].pcolormesh(X, Y, speed.T, shading='auto', cmap='jet_r')
    plt.colorbar(sc, label='Speed (km/h)', ax=axs[0])
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Position (km)')

    sc = axs[1].pcolormesh(X, Y, density.T, shading='auto', cmap='jet')
    plt.colorbar(sc, label='Density (veh/km/lane)', ax=axs[1])
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Position (km)')

    sc = axs[2].pcolormesh(X, Y, flow.T, shading='auto', cmap='jet_r')
    plt.colorbar(sc, label='Flow (veh/h)', ax=axs[2])
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Position (km)')

    fig.suptitle('Time-Space Diagrams for RM+VSL MPC Control')
    fig.tight_layout()

########################################################################
# Main: Simulation in Closed Loop For Ramp Metering and VSL with DRL
########################################################################
# 1) find the directory this script lives in
here = os.path.dirname(os.path.realpath(__file__))
# 2) climb up into your project root and then down into the folder with your zip
zip_path = os.path.normpath(
    os.path.join(here, "..",           # up from new_codes/
                        "updated",
                        "updated_logs",
                        "low",
                        "ddpg_low_final.zip")
)

if RUN_MPC_DRL_CTRL:
    print("**************************** MPC + DRL: RAMP METERING + VSL ****************************")
    # Load trained policy
    if DEMAND_NOISE == "MEDIUM":
        model = DDPG.load("ddpg_medium.zip")
    elif DEMAND_NOISE == "HIGH":
        model = DDPG.load("ddpg_high.zip")
    else:
        model = DDPG.load(zip_path)

    L1 = metanet.LinkWithVsl(4, num_lanes, L, rho_max, rho_crit, v_free, a, name="L1",
                            segments_with_vsl={2, 3}, alpha=0.1) # alpha = Non-compliance factor to the indicated speed limit.
    net = (
        metanet.Network(name="A1")
        .add_path(origin=O1, path=(N1, L1, N2, L2, N3), destination=D1)
        .add_origin(O2, N2)
    )
    metanet.engines.use("casadi", sym_type="SX")
    net.is_valid(raises=True)
    net.step(T=T, tau=tau, eta=eta, kappa=kappa, delta=delta, init_conditions={O1: {"v_ctrl": v_free*2}})
    F = metanet.engine.to_function(
        net=net, more_out=True, compact=2, T=T
    )
    # F: (x[14], u[3], d[2]) -> (x+[14], q[8])

    # create MPC controller
    # T_c (sampling interval of MPC) = 300 s = M * T_s; M = 30, T_s = 10 s
    # Np = 2 for 600 s prediction horizon
    Np, Nc, M_mpc, M_drl = 2, 2, 30, 6
    mpc = Mpc[cs.SX](
        nlp=Nlp[cs.SX](sym_type="SX"),
        prediction_horizon=Np*M_mpc,
        control_horizon=Nc*M_mpc,
        input_spacing=M_mpc
    )
    # create states, actions and disturbance for MPC
    n_seg, n_orig = sum(link.N for _, _, link in net.links), len(net.origins)
    rho, _ = mpc.state("rho", n_seg, lb=0)
    v, _ = mpc.state("v", n_seg, lb=0)
    w, _ = mpc.state("w", n_orig, lb=0, ub=[[200], [100]]) # O2 queue is constrained
    v_min = 20
    v_ctrl, _ = mpc.action("v_ctrl", len(L1.vsl), lb=v_min, ub=v_free)
    r, _ = mpc.action("r", lb=0, ub=1)
    d = mpc.disturbance("d", n_orig)
    # set dynamics constraints
    mpc.set_dynamics(F)
    # set objective function
    v_ctrl_last = mpc.parameter("v_ctrl_last", (v_ctrl.size1(), 1))
    r_last = mpc.parameter("r_last", (r.size1(), 1))
    mpc.minimize(
        T * cs.sum2(cs.sum1(rho * L * num_lanes) + cs.sum1(w)) 
        + 0.4*cs.sumsqr(cs.diff(cs.horzcat(r_last, r)))
        + 0.4*cs.sumsqr(cs.diff(cs.horzcat(v_ctrl_last, v_ctrl), 1, 1) / v_free)
    )
    # set solver for MPC NLP
    opts = {
        "expand": True,
        "print_time": False,
        "ipopt": {"max_iter": 500, "sb": "yes", "print_level": 0},
    }
    mpc.init_solver(solver="ipopt", opts=opts)

    # initial conditions (Highway initially empty)
    rho = cs.DM([0, 0, 0, 0, 0, 0])
    v = cs.DM([v_free, v_free, v_free, v_free, v_free, v_free]) # cs.DM([80, 80, 78, 72.5, 66, 62])
    w = cs.DM([0, 0]) # queue lengths at origin nodes O1 and O2
    v_ctrl_last = v[:L1.N][L1.vsl]
    r_last = cs.DM.ones(r.size1(), 1)
    v_ctrl_last_combined = v[:L1.N][L1.vsl]
    r_last_combined = cs.DM.ones(r.size1(), 1)
    sol_prev = None

    density, speed, queue_lengths, flow, origin_flow, r_mpc, vsl_mpc, r_ctrl, vsl_ctrl = [], [], [], [], [], [], [], [], []
    for k in range(demands.shape[0]):
        # get demand forecast
        d_hat = demands_forecast[k:k+Np*M_mpc, :]
        if d_hat.shape[0] < Np*M_mpc:
            d_hat = np.pad(d_hat, ((0, Np*M_mpc-d_hat.shape[0]), (0, 0)), "edge")
        # solve MPC problem every M steps
        if k >= int(T_warmup//T) and k <= int((T_warmup+T_sim)//T) and k % M_mpc == 0:
            sol = mpc.solve(
                pars={"rho_0": rho, "v_0": v, "w_0": w, "d": d_hat.T, 
                      "v_ctrl_last": v_ctrl_last_combined, "r_last": r_last_combined},
                vals0=sol_prev,
            )
            v_ctrl_last = sol.vals["v_ctrl"][:, 0]
            r_last = sol.vals["r"][0]
            # Handle NaN mpc outputs by retaining previous solution
            if np.any(np.isnan(v_ctrl_last)):
                v_ctrl_last[np.isnan(v_ctrl_last)] = sol_prev["v_ctrl"][np.isnan(v_ctrl_last), 0]
            if np.isnan(r_last):
                r_last = sol_prev["r"][0]
            sol_prev = sol.vals
        
        if k >= int(T_warmup//T) and k <= int((T_warmup+T_sim)//T) and k % M_drl == 0:
            u_mpc = np.concatenate([r_last, v_ctrl_last]).flatten()
            obs = normalize_observations(rho, v, w, u_mpc, 
                                         u_prev=np.concatenate([r_last_combined, v_ctrl_last_combined]).flatten(),
                                         demands=demands[k, :])
            u_drl, _ = model.predict(obs, deterministic=True)
            v_ctrl_last_combined = np.clip(v_ctrl_last + u_drl[1:], a_min=v_min, a_max=v_free)
            r_last_combined = np.clip(r_last + u_drl[0], a_min=0, a_max=1)
        
        if k > int((T_warmup+T_sim)//T):
            v_ctrl_last_combined = v_free*cs.DM.ones(len(L1.vsl), 1)
            r_last_combined = cs.DM.ones(r.size1(), 1)
        
        # step dynamics
        x_next, q_all = F(cs.vertcat(rho, v, w), cs.vertcat(v_ctrl_last_combined, r_last_combined), demands[k, :])
        rho, v, w = cs.vertsplit(x_next, (0, n_seg, 2*n_seg, 2*n_seg+n_orig))
        q, q_o = cs.vertsplit(q_all, (0, n_seg, n_seg+n_orig))
        density.append(rho)
        speed.append(v)
        queue_lengths.append(w)
        flow.append(q)
        origin_flow.append(q_o)
        r_mpc.append(r_last)
        vsl_mpc.append(v_ctrl_last)
        r_ctrl.append(r_last_combined)
        vsl_ctrl.append(v_ctrl_last_combined)
        if k % 100 == 0:
            print(f"Step {k} of {demands.shape[0]}.")
    density, speed, queue_lengths, flow, origin_flow, r_mpc, vsl_mpc, r_ctrl, vsl_ctrl = (np.squeeze(o) for o in (density, speed, queue_lengths, flow, origin_flow, r_mpc, vsl_mpc, r_ctrl, vsl_ctrl))

    # compute VKT and VHT metrics
    vkt, vht = compute_metrics(density, flow, queue_lengths, num_lanes, L, T)
    print(f"VKT = {vkt:.4f} veh.km \nVHT or TTS = {vht:.4f} veh.h \n Avg Speed = {vkt/vht:.4f} km/h")

    # plotting
    _, axs = plt.subplots(4, 2, constrained_layout=True, sharex=True)
    axs[0, 0].plot(time, speed)
    axs[0, 0].set_ylabel("Speed (km/h)")
    axs[0, 0].set_xlabel("Time (h)")
    axs[0, 0].set_xlim(0, T_sim+T_warmup+T_cooldown)

    axs[0, 1].plot(time, flow)
    axs[0, 1].set_ylabel("Flow (veh/h)")

    axs[1, 0].plot(time, density)
    axs[1, 0].set_ylabel("Density (veh/km/lane)")

    axs[1, 1].plot(time, demands)
    axs[1, 1].set_ylabel("Origin Demands (veh/h)")

    axs[2, 0].plot(time, origin_flow)
    axs[2, 0].set_ylabel("Origin Flow (veh/h)")

    axs[2, 1].plot(time, queue_lengths)
    axs[2, 1].set_ylabel("Queue Lengths (veh)")

    axs[3, 0].plot(time, r_mpc, label="MPC")
    axs[3, 0].plot(time, r_ctrl, label="MPC+DRL")
    axs[3, 0].set_ylabel("Ramp metering rate (-)")
    axs[3, 0].legend()

    axs[3, 1].plot(time, vsl_mpc, label="MPC")
    axs[3, 1].plot(time, vsl_ctrl, label="MPC+DRL")
    axs[3, 1].set_ylabel("VSL (km/h)")
    axs[3, 1].legend()
    
    for ax in axs.flat:
        ax.set_ylim(0, ax.get_ylim()[1])

    n_seg = 6
    X, Y = np.meshgrid(time * 3600, (np.arange(n_seg) + 1) * L)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    sc = axs[0].pcolormesh(X, Y, speed.T, shading='auto', cmap='jet_r')
    plt.colorbar(sc, label='Speed (km/h)', ax=axs[0])
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Position (km)')

    sc = axs[1].pcolormesh(X, Y, density.T, shading='auto', cmap='jet')
    plt.colorbar(sc, label='Density (veh/km/lane)', ax=axs[1])
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Position (km)')

    sc = axs[2].pcolormesh(X, Y, flow.T, shading='auto', cmap='jet_r')
    plt.colorbar(sc, label='Flow (veh/h)', ax=axs[2])
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Position (km)')

    fig.suptitle('Time-Space Diagrams for RM+VSL MPC+DRL Control')
    fig.tight_layout()

plt.show()