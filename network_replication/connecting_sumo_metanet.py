# metanet part

"""
Reproduces the results in Figure 6.7 of [1], where ramp metering control and variable
speed limits are controlled in a coordinated fashion. The control is achieved via a
Model Predictive Control (MPC) scheme, which is implemeted with the CasADi-based `csnlp`
library.

Essentially, metanet is a macroscopic model that is translated into a function that
is then used by casadi. Metanet model has to be very similar to SUMO network model and have similar demands.
"""

import casadi as cs
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from csnlp import Nlp
from csnlp.wrappers import Mpc
import traci

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


# --------------------------
# defining demands for metanet model (the same as for SUMO model)
def create_demands(time: np.ndarray) -> np.ndarray: # input should be a NumPy array, as well as the output
    return np.stack( # outputs from the two interpolations will be combines into a single NumPy array
        (
            np.interp(time, (2.0, 2.25), (3500, 1000)),
            np.interp(time, (0.0, 0.15, 0.35, 0.5), (500, 1500, 1500, 500)),
        )
    )

# --------------------------
# parameters for metanet model
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
a = 1.867 # maximum acceleration rate of a vehicle (m/s2) -> how quickly vehicles can speed up in this simulation
rho_crit = 33.5
v_free = 102 # km/h
args = (lanes, L, rho_max, rho_crit, v_free, a) # tuple with several parameters

# --------------------------
""" 
Building a metanet model. Metanet assumes that all vehicles flow equally distributed on the edge,
it does not take separate lanes. 
"""
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

# --------------------------
# making casadi function out of the network
engines.use("casadi", sym_type="SX") # using casadi as the backend engine for symbolic computation
net.is_valid(raises=True) # checking whether metanet model has been properly set up and configured
net.step( # incorporating the defined parameters and initial conditions to generate casadi function
    T=T,
    tau=tau,
    eta=eta,
    kappa=kappa,
    delta=delta,
    init_conditions={O1: {"v_ctrl": v_free * 2}}, # sets the initial condition for a specific component
)

# --------------------------
F: cs.Function = metanet.engine.to_function( # casadi is used for optimization, simulation, analysis
    net=net, # passes already defined metanet network
    more_out=True, # the generated function should return additional outputs
    compact=2,
    T=T, # specifies time duration, 2.5 hours in our case
)
# F: (x[14], u[3], d[2]) -> (x+[14], q[8])

# --------------------------
# creating demands for metanet model
demands = create_demands(time).T # .T transposes the array -> e.g. if original shape is (2,n), it changes the array to (n,2)

# --------------------------
# creating the MPC controller
Np, Nc, M = 7, 5, 6 # 7-> base number of prediction steps for mpc; 5-> base number of control steps; 6-> multiplier
mpc = Mpc[cs.SX]( # creating mpc controller
    nlp=Nlp[cs.SX](sym_type="SX"), # setting up instance of nonlinear programming problem -> underlying optimization problem that mpc will solve
    prediction_horizon=Np * M, # over 42 steps
    control_horizon=Nc * M, # over 30 steps
    input_spacing=M, # control inputs are adjusted every 6 steps -> mpc solves the optimization problem every 60 seconds
)

# --------------------------
# creating states, action, and disturbances for metanet model
# defining the number of segments and origins
n_seg, n_orig = sum(link.N for _, _, link in net.links), len(net.origins) # iterating over all the links, extracting its attributes. We sum all these values to get the total number of segments in the network
rho, _ = mpc.state("rho", n_seg, lb=0) # function returns rho; ensures that density is not negative
v, _ = mpc.state("v", n_seg, lb=0) # creating velocity as state variable
w, _ = mpc.state("w", n_orig, lb=0, ub=[[np.inf], [100]])  # O2 queue is constrained
v_ctrl, _ = mpc.action("v_ctrl", len(L1.vsl), lb=20, ub=v_free) # creating a control action for VSL. Control speed cannot be lower than 20 km/h and more than free speed
r, _ = mpc.action("r", lb=0, ub=1) # creating control action for on-ramp -> in a form of a normalized ratio
d = mpc.disturbance("d", n_orig) # creating disturbance variable. Disturbances usually capture external influences or uncertainties e.g. fluctuating demands

# --------------------------
# adding dynamic constraints for metanet model
mpc.set_dynamics(F) # ensures that any control actions produce feasible system trajectories

# --------------------------
# setting the optimization objective
# before defining the objective, we set up parameters to hold the "last" (previous) control inputs
v_ctrl_last = mpc.parameter("v_ctrl_last", (v_ctrl.size1(),1)) # parameter used to compute the difference between the previous and current control values
r_last = mpc.parameter("r_last", r.size1(),1) # holds previous ramp metering values to enable penalization of abrupt changes

# --------------------------
mpc.minimize(
    T * cs.sum2(cs.sum1(rho * L * lanes) + cs.sum1(w))  # penalizes high traffic density and long queues, thereby incentivizing control actions that reduce congestion
    + 0.4 * cs.sumsqr(cs.diff(cs.horzcat(v_ctrl_last, v_ctrl), 1, 1) / v_free) # penalizes large/ abrupt changes in the variable speed control, encouraging smoother transitions
    + 0.4 * cs.sumsqr(cs.diff(cs.horzcat(r_last, r))) # discourages rapid changes in ramp metering actions, promoting gradual adjustments
)

# --------------------------
# setting optimization solver for MPC's NLP
opts = {
    "expand": True,
    "print_time": False, # disables the printing of timing information during the optimization process, keeping the output clear
    "ipopt": {"max_iter": 500, "sb": "yes", "print_level": 0}, # 500-> max. number of iterations allowed for the solver
}
mpc.init_solver(solver="ipopt", opts=opts)

# --------------------------
# create initial conditions for metanet model --> these will be overridden by SUMO simulation
rho = cs.DM([22, 22, 22.5, 24, 30, 32]) # defines the starting density for each road segment
v = cs.DM([80, 80, 78, 72.5, 66, 62]) # setting initial speed for each road segment
w = cs.DM([0, 0]) # initial queue length at the network's origins -> no vehicles are queued at the start of the simulation

# --------------------------
# Setting up SUMO environment
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

# --------------------------
# Defining Sumo configuration
Sumo_config = [
    'sumo-gui',
    '-c', 'network_example.sumocfg',
    '--lateral-resolution', '0.1'
]

# --------------------------
# Starting SUMO
traci.start(Sumo_config)

# --------------------------
# Defining edge IDs
mainline_edges = ["O1", "L1a", "L1b", "L1c", "L2", "D1"]
onramp_edges = ["O2"]
# TODO: add queue actual ID
queue_edge_main = "E0" # new edge added in netedit to measure queue length (find details in network file)
default_speed = 28.33 # m/s --> 102 km/h

# Initializing data containers for logging aggregated metric for the mainline and the on-ramp
time_steps = []
traffic_data = {
    "mainline": {
        "flow": [],
        "speed": [],
        "density": []
    },
    "onramp": {
        "flow": [],
        "speed": [],
        "density": []
    }
}

# this function filters the list of all lane IDs and returns those that
# start with the given edge_id
def get_lanes_of_edges(edge_id):
    all_lanes = traci.lane.getIDList()
    lanes = [lane for lane in all_lanes if lane.startwith(edge_id)]
    return lanes

# --------------------------
# I couldn't find function in edge class to calculate edge_length,
# so I will just write them separately since they are known.

edge_lengths = { # in m
    "O1": 1000,
    "L1a": 1000,
    "L1b": 1000,
    "L1c": 1000,
    "L2": 1000,
    "D1": 1000,
    "O2": 750
}

# creating helper functions

# aggregate density over multiple edges
# for each edge in edge_ids, it retrieves density using traci function,
# then we sum them to get a total density for mainline_edge --> would make more sense for metanet
# returning the aggregate density as a casadi dm
def get_edge_density(edge_ids):

    total_vehicles = 0
    total_length = 0
    for edge in edge_ids:
        veh_count = traci.edge.getLastStepVehicleNumber(edge)
        length = edge_lengths[edge]
        total_vehicles += veh_count
        total_length += length
    density = total_vehicles/ (total_length/ 1000.0) # veh/km
    return density

# computes the weighted average speed in km/h over multiple edges
# the weights are given by the vehicle count on each edge
# returns the average speed as a casadi DM
def get_edge_speed(edge_ids):
    total_vehicles = 0
    weighted_speed_sum = 0
    for edge in edge_ids:
        veh_count = traci.edge.getLastStepVehicleNumber(edge)
        speed = traci.edge.getLastStepMeanSpeed(edge) # in m/s
        total_vehicles += veh_count
        weighted_speed_sum += veh_count * speed
    if total_vehicles > 0:
        avg_speed = weighted_speed_sum/ total_vehicles
    else:
        avg_speed = 0
    return avg_speed * 3.6 # covert m/s to km/h

# aggregating flow (in veh/h) over multiple edges
# calculating flow by (aggregated density in veh/km) * (weighted average speed in km/h)
# returning the total flow as casadi DM
def get_edge_flow(edge_ids):
    density = get_edge_density(edge_ids)
    avg_speed = get_edge_speed(edge_ids)
    flow = density * avg_speed
    return flow

# setting the maximum speed (in ms/s) for every lane on the given edges
# since direct edge-level speed-setting is not available, this functon iterates over each edge's lanes
def set_vsl(edge_id, speed):
    lanes = get_lanes_of_edges(edge_id)
    for lane in lanes:
        traci.lane.setMaxSpeed(lane, float(speed))

# setting the ramp meter signal. For a 60-seconds cycle, constructs a state string where the first
# green time seconds are green and the remaining seconds are red
def custom_light_state(tl_id, state_string):
    traci.trafficlight.setRedYellowGreenState(tl_id, state_string)

# aggregated queue length function
def get_edge_queues(mainlane_q, onramp_edge):
    mainline_q_n = traci.edge.getLastStepVehicleNumber(mainlane_q)
    onramp_q_n = traci.edge.getLastStepVehicleNumber(onramp_edge[0])
    return mainline_q_n, onramp_q_n






























v_ctrl_last = v[: L1.N][L1.vsl] # setting initial "last" variable speed control input
r_last = cs.DM.ones(r.size1(), 1) # initializes the previous ramp metering control values
sol_prev = None # prepares a variable to store the previous solution from the MPC colver
RHO, V, W, Q, Q_o, V_CTRL, R = [], [], [], [], [], [], [] # initializes empty lists to record simulation outputs over time

# the actual simulation loop of metanet model
for k in range(demands.shape[0]): # iterates over each simulation time step, where the total number of steps is determined by the number of rows in the demands array
    # get the demand forecast - pad if at the end of the simulation
    d_hat = demands[k : k + Np * M, :] # extracts a forecast of future demand over the predicted horizon
    if d_hat.shape[0] < Np * M: # ensures that demand forecast has the required length
        d_hat = np.pad(d_hat, ((0, Np * M - d_hat.shape[0]), (0, 0)), "edge")

    # solve the mpc problem every M steps -> every 60 seconds
    if k % M == 0: # mpc optimization is performed every M time steps. k is a simulation time step
        sol = mpc.solve( # solving optimization problem using the current state and forecast (TAKING THESE FROM SUMO)
            pars={
                "rho_0": rho, # change to SUMo states
                "v_0": v, # change to SUMo states
                "w_0": w, # change to SUMo states
                "d": d_hat.T,
                "v_ctrl_last": v_ctrl_last,
                "r_last": r_last,
            },
            vals0=sol_prev, # uses the previous solution as a warm start to speed up convergence
        )
        sol_prev = sol.vals # updates sol_prev with the current solution, to be used as an initial guess in the next optimization
        v_ctrl_last = sol.vals["v_ctrl"][:, 0] # extracts the first control action from the MPC solution -> these values will be applied in the next simulation step
        r_last = sol.vals["r"][0]


    # step the dynamics
    x_next, q_all = F( # computes the next state and additional outputs by applying the system dynamics
        cs.vertcat(rho, v, w), cs.vertcat(v_ctrl_last, r_last), demands[k, :]
    ) # returns the next state vector and additional outputs
    rho, v, w = cs.vertsplit(x_next, (0, n_seg, 2 * n_seg, 2 * n_seg + n_orig)) # splits the updated state vector x_next back into individual state components
    q, q_o = cs.vertsplit(q_all, (0, n_seg, n_seg + n_orig))
    # appending the current states, outputs, and control actions to the respective lists so they can be used for analysis
    RHO.append(rho) # density
    V.append(v) # speed
    W.append(w) # queue length
    Q.append(q) # flow (segment-based)
    Q_o.append(q_o) # flow (origin-based)
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







