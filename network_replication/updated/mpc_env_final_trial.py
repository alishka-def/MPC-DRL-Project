########################################################################
# Imports
########################################################################
import os
import sys

import numpy as np
import gymnasium as gym
import casadi as cs
import sym_metanet as metanet

from csnlp import Nlp
from csnlp.wrappers import Mpc

from mpc_env_final import MetanetEnv

########################################################################
# Main: Parameters
########################################################################
T = 10.0 / 3600 # temporal discretization (hours)
T_final = 2.5 # simulation time of 2.5 hours
time = np.arange(0, T_final, T)

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

L2 = metanet.Link[cs.SX](2, num_lanes, L, rho_max, rho_crit, v_free, a, name="L2") # num_segments = 2
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

########################################################################
# Main: Environment
########################################################################
env = MetanetEnv(dynamics=F, time=time, mpc=mpc)
env.reset()
for _ in range(len(time)):
    a = np.zeros(shape=(3,))
    env.step(action=a)
