"""
Reproduces the results in Figure 6.7 of [1], where ramp metering control and variable
speed limits are controlled in a coordinated fashion. The control is achieved via a
Model Predictive Control (MPC) scheme, which is implemented with the CasADi-based `csnlp`
library.

To install it, run
```bash
pip install csnlp
```

References
----------
[1] Hegyi, A., 2004. Model predictive control for integrating traffic control measures.
    Netherlands TRAIL Research School.
"""

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
from csnlp import Nlp
from csnlp.wrappers import Mpc

import sym_metanet as metanet
from sym_metanet import (
    Destination,
    Link,
    LinkWithVsl,
    MainstreamOrigin,
    MeteredOnRamp,
    Network,
    Node,
    engines,
)


def create_demands(time: np.ndarray) -> np.ndarray:
    return np.stack(
        (
            np.interp(time, (2.0, 2.25), (3500, 1000)),
            np.interp(time, (0.0, 0.15, 0.35, 0.5), (500, 1500, 1500, 500)),
        )
    )


# parameters
T = 10 / 3600
Tfin = 2.5
time = np.arange(0, Tfin, T)
L = 1
lanes = 2
C = (4000, 2000)
tau = 18 / 3600
kappa = 40
eta = 60
rho_max = 180
delta = 0.0122
a = 1.867
rho_crit = 33.5
v_free = 102
args = (lanes, L, rho_max, rho_crit, v_free, a)

# build network
N1 = Node(name="N1")
N2 = Node(name="N2")
N3 = Node(name="N3")
O1 = MainstreamOrigin[cs.SX](name="O1")
O2 = MeteredOnRamp[cs.SX](C[1], name="O2")
D1 = Destination[cs.SX](name="D1")
L1 = LinkWithVsl[cs.SX](4, *args, segments_with_vsl={2, 3}, alpha=0.1, name="L1")
L2 = Link[cs.SX](2, *args, name="L2")
net = (
    Network(name="A1")
    .add_path(origin=O1, path=(N1, L1, N2, L2, N3), destination=D1)
    .add_origin(O2, N2)
)

# make a casadi function out of the network
engines.use("casadi", sym_type="SX")
net.is_valid(raises=True)
net.step(
    T=T,
    tau=tau,
    eta=eta,
    kappa=kappa,
    delta=delta,
    init_conditions={O1: {"v_ctrl": v_free * 2}},
)
F: cs.Function = metanet.engine.to_function(
    net=net,
    more_out=True,
    compact=2,
    T=T,
)
# F: (x[14], u[3], d[2]) -> (x+[14], q[8])

# create demands
demands = create_demands(time).T

# create the MPC controller
Np, Nc, M = 7, 5, 6
mpc = Mpc[cs.SX](
    nlp=Nlp[cs.SX](sym_type="SX"),
    prediction_horizon=Np * M,
    control_horizon=Nc * M,
    input_spacing=M,
)

# create states, action, and disturbaces
n_seg, n_orig = sum(link.N for _, _, link in net.links), len(net.origins)
rho, _ = mpc.state("rho", n_seg, lb=0)
v, _ = mpc.state("v", n_seg, lb=0)
w, _ = mpc.state("w", n_orig, lb=0, ub=[[np.inf], [100]])  # O2 queue is constrained
v_ctrl, _ = mpc.action("v_ctrl", len(L1.vsl), lb=20, ub=v_free)
r, _ = mpc.action("r", lb=0, ub=1)
d = mpc.disturbance("d", n_orig)

# add dynamics constraints
mpc.set_dynamics(F)

# set the optimization objective
v_ctrl_last = mpc.parameter("v_ctrl_last", (v_ctrl.size1(), 1))
r_last = mpc.parameter("r_last", (r.size1(), 1))
mpc.minimize(
    T * cs.sum2(cs.sum1(rho * L * lanes) + cs.sum1(w))
    + 0.4 * cs.sumsqr(cs.diff(cs.horzcat(v_ctrl_last, v_ctrl), 1, 1) / v_free)
    + 0.4 * cs.sumsqr(cs.diff(cs.horzcat(r_last, r)))
)

# set optimization solver for the MPC's NLP
opts = {
    "expand": True,
    "print_time": False,
    "ipopt": {"max_iter": 500, "sb": "yes", "print_level": 0},
}
mpc.init_solver(solver="ipopt", opts=opts)

# create initial conditions
rho = cs.DM([22, 22, 22.5, 24, 30, 32])
v = cs.DM([80, 80, 78, 72.5, 66, 62])
w = cs.DM([0, 0])

# simulate
v_ctrl_last = v[: L1.N][L1.vsl]
r_last = cs.DM.ones(r.size1(), 1)
sol_prev = None
RHO, V, W, Q, Q_o, V_CTRL, R = [], [], [], [], [], [], []
for k in range(demands.shape[0]):
    # get the demand forecast - pad if at the end of the simulation
    d_hat = demands[k : k + Np * M, :]
    if d_hat.shape[0] < Np * M:
        d_hat = np.pad(d_hat, ((0, Np * M - d_hat.shape[0]), (0, 0)), "edge")

    # solve the mpc problem every M steps
    if k % M == 0:
        sol = mpc.solve(
            pars={
                "rho_0": rho,
                "v_0": v,
                "w_0": w,
                "d": d_hat.T,
                "v_ctrl_last": v_ctrl_last,
                "r_last": r_last,
            },
            vals0=sol_prev,
        )
        sol_prev = sol.vals
        v_ctrl_last = sol.vals["v_ctrl"][:, 0]
        r_last = sol.vals["r"][0]

    # step the dynamics
    x_next, q_all = F(
        cs.vertcat(rho, v, w), cs.vertcat(v_ctrl_last, r_last), demands[k, :]
    )
    rho, v, w = cs.vertsplit(x_next, (0, n_seg, 2 * n_seg, 2 * n_seg + n_orig))
    q, q_o = cs.vertsplit(q_all, (0, n_seg, n_seg + n_orig))
    RHO.append(rho)
    V.append(v)
    W.append(w)
    Q.append(q)
    Q_o.append(q_o)
    V_CTRL.append(v_ctrl_last)
    R.append(r_last)
    if k % 100 == 0:
        print(f"step {k} of {demands.shape[0]}")
RHO, V, W, Q, Q_o, V_CTRL, R = (  # type: ignore[assignment]
    np.squeeze(o) for o in (RHO, V, W, Q, Q_o, V_CTRL, R)
)

# compute TTS metric (Total-Time-Spent)
tts = T * sum((rho * L * lanes).sum() + w.sum() for rho, w in zip(RHO, W))
print(f"TTS = {tts:.3f} veh.h")

# plot
_, axs = plt.subplots(4, 2, constrained_layout=True, sharex=True)
axs[0, 0].plot(time, V)
axs[0, 0].set_ylabel("speed")
axs[0, 1].plot(time, Q)
axs[0, 1].set_ylabel("flow")
axs[1, 0].plot(time, RHO)
axs[1, 0].set_ylabel("density")
axs[1, 1].plot(time, demands)
axs[1, 1].set_ylabel("origin demands")
axs[2, 0].plot(time, Q_o)
axs[2, 0].set_ylabel("origin flow")
axs[2, 1].plot(time, W)
axs[2, 1].set_ylabel("queue")
axs[3, 0].step(time, R, where="post")
axs[3, 0].set_ylabel("metering rate")
axs[3, 1].step(time, V_CTRL, where="post")
axs[3, 1].set_ylabel("VSL")
axs[0, 0].set_xlim(0, Tfin)
for ax in axs.flat:
    ax.set_ylim(0, ax.get_ylim()[1])
# Save the entire figure as a PNG file.
plt.savefig("mpc_control.png", dpi=300, bbox_inches="tight")
plt.show()

# Number of segments (e.g., for your network with L1 having 4 segments and L2 having 2 segments)
n_seg = 6

# Create a mesh grid for time-space coordinates.
# Here, time is converted to seconds (assuming the original time vector is in hours)
X, Y = np.meshgrid(time * 3600, (np.arange(n_seg) + 1) * L)

# Create a figure with three subplots for the three variables: Speed, Density, and Flow.
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

# Time-space diagram for Speed
c0 = axs[0].pcolormesh(X, Y, V.T, shading='auto', cmap='jet_r')
plt.colorbar(c0, label="Speed (km/h)", ax=axs[0])
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("Position (km)")

# Time-space diagram for Density
c1 = axs[1].pcolormesh(X, Y, RHO.T, shading='auto', cmap='jet_r')
plt.colorbar(c1, label="Density (veh/km/lane)", ax=axs[1])
axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel("Position (km)")

# Time-space diagram for Flow
c2 = axs[2].pcolormesh(X, Y, Q.T, shading='auto', cmap='jet_r')
plt.colorbar(c2, label="Flow (veh/h)", ax=axs[2])
axs[2].set_xlabel("Time (s)")
axs[2].set_ylabel("Position (km)")

fig.suptitle("Time-Space Diagrams for MPC Control")
fig.tight_layout()

# Save the time-space diagrams to a PNG file
fig.savefig("mpc_control_time_space.png", dpi=300, bbox_inches="tight")
plt.show()
