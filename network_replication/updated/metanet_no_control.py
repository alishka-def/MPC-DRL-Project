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
from typing import Tuple

########################################################################
# Methods
########################################################################
def create_demands(time: np.ndarray) -> np.ndarray:
    return np.stack(
        (
            np.interp(time, (2.0, 2.25), (3500, 1000)),
            np.interp(time, (0.0, 0.15, 0.35, 0.5), (500, 1500, 1500, 500))
        )
    )


def compute_metrics(density: np.ndarray, flow: np.ndarray, queue_lengths, num_lanes: int, L: float, T: float) -> Tuple[float, float]:
    vkt = T * sum((q * L * num_lanes).sum() for q in flow)
    vht = T * sum(w.sum() + (rho * L * num_lanes).sum() for rho, w in zip(density, queue_lengths))
    return vkt, vht


########################################################################
# Main: Parameters
########################################################################
T = 10.0 / 3600 # temporal discretization (hours)
T_final = 2.5
time = np.arange(0, T_final, T)
demands = create_demands(time).T # size = (900, 2)

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
print("**************************** OPEN-LOOP ****************************")
F = metanet.engine.to_function(
    net=net, more_out=True, compact=1, T=T
)
# F: (rho[6], v[6], w[2], v_ctrl, r, d[2]) -> (rho+[6], v+[6], w+[2], q[6], q_o[2])

# initial conditions
rho = cs.DM([22, 22, 22.5, 24, 30, 32])
v = cs.DM([80, 80, 78, 72.5, 66, 62])
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
print(f"VKT = {vkt:.4f} veh.km \nVHT or TTS = {vht:.4f} veh.h")

# plotting
_, axs = plt.subplots(3, 2, constrained_layout=True, sharex=True)
axs[0, 0].plot(time, speed)
axs[0, 0].set_ylabel("Speed (km/h)")
axs[0, 0].set_xlabel("Time (h)")
axs[0, 0].set_xlim(0, T_final)

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
plt.savefig("no_control_subplots.png", dpi=300, bbox_inches="tight")
plt.show()

# plotting (TSD diagrams)
n_seg = 6
X, Y = np.meshgrid(time * 3600, (np.arange(n_seg) + 1) * L)

fig, axs = plt.subplots(1, 3, figsize=(12, 4))
c0 = axs[0].pcolormesh(X, Y, speed.T, shading='auto', cmap='jet_r')
plt.colorbar(c0, label='Speed (km/h)', ax=axs[0])
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Position (km)')

c1 = axs[1].pcolormesh(X, Y, density.T, shading='auto', cmap='jet_r')
plt.colorbar(c1, label='Density (veh/km/lane)', ax=axs[1])
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Position (km)')

c2 = axs[2].pcolormesh(X, Y, flow.T, shading='auto', cmap='jet_r')
plt.colorbar(c2, label='Flow (veh/h)', ax=axs[2])
axs[2].set_xlabel('Time (s)')
axs[2].set_ylabel('Position (km)')

fig.suptitle('Time-Space Diagrams for No Control')
fig.tight_layout()
fig.savefig("no_control_time_space.png", dpi=300, bbox_inches="tight")
plt.show()