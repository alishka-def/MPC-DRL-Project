"""
Integration of SUMO with metanet to control on-ramp meter and variable speed limits.

This script integrates SUMO with a metanet‐based MPC controller:
  - The high-level predictive model (metanet dynamic function F and demand forecast) remains.
  - Actual state measurements (density, speed, and queues) are extracted from SUMO.
  - Control actions (VSL and ramp metering rate) computed by MPC are applied back to SUMO.
  - The queue state is a 2-element vector: [mainline queue from edge "E0", on‑ramp queue from edge "O2"].
  - The simulation loop runs while vehicles are expected and simulation time is ≤ 9000 s.

"""
# Importing all necessary libraries and packages
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

# Defining demands for metanet model (the same as for SUMO model)
def create_demands(time: np.ndarray) -> np.ndarray: # input should be a NumPy array, as well as the output
    return np.stack( # outputs from the two interpolations will be combines into a single NumPy array
        (
            np.interp(time, (2.0, 2.25), (3500, 1000)),
            np.interp(time, (0.0, 0.15, 0.35, 0.5), (500, 1500, 1500, 500)),
        )
    )

# Parameters for metanet model
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


# Building metanet model
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


# Making a casadi function out of the metanet network
engines.use("casadi", sym_type="SX") # using casadi as the backend engine for symbolic computation
net.is_valid(raises=True) # checking whether metanet model has been properly set up and configured
net.step( # incorporating the defined parameters and initial conditions to generate casadi function
    T=T,
    tau=tau,
    eta=eta,
    kappa=kappa,
    delta=delta,
    init_conditions={O1: {"v_ctrl": v_free * 2}}, # sets the initial condition for Origin 1 - VSL is set to be 204 km/h
)
F: cs.Function = metanet.engine.to_function( # casadi is used for optimization, simulation, analysis
    net=net, # passes already defined metanet network
    more_out=True,  # the generated function should return additional outputs
    compact=2,
    T=T, # specifies time duration, 2.5 hours in our case
)

# F: (x[14], u[3], d[2]) -> (x+[14], q[8])

# Here, x is a state vector with 14 elements, u is a control vector with 3 elements, and d is a disturbance vector with 2 elements.
# The function F takes these inputs and returns a new state vector (x+), a queue vector (q) with 8 elements.
# Hence, for x[14] we take 6 speeds (mainline segment), 6 densities (mainline segment), 2 queues (mainline and on-ramp) --> in the form of NumPy array
# For u[3] we take 2 VSLs (mainline segment) and 1 ramp metering rate (on-ramp) --> in the form of NumPy array
# For d[2] we take 2 demands (mainline and on-ramp) --> in the form of NumPy array --> from create_demands function
# The dynamic function returns 14 new state variables as well as 8 flows --> 6 exit flows and 2 origin flows


# Creating demands for metanet model
demands = create_demands(time).T # .T transposes the array -> e.g. if original shape is (2,n), it changes the array to (n,2)

# Creating the MPC controller
Np, Nc, M = 7, 5, 6 # 7-> base number of prediction steps for mpc; 5-> base number of control steps; 6-> multiplier
mpc = Mpc[cs.SX]( # creating mpc controller
    nlp=Nlp[cs.SX](sym_type="SX"), # setting up instance of nonlinear programming problem -> underlying optimization problem that mpc will solve
    prediction_horizon=Np * M, # over 42 steps
    control_horizon=Nc * M, # over 30 steps
    input_spacing=M,  # control inputs are adjusted every 6 steps -> mpc solves the optimization problem every 60 seconds
)

# Creating states, action, and disturbances for metanet model
# This code simply defines the variables for metanet model
n_seg, n_orig = sum(link.N for _, _, link in net.links), len(net.origins) # iterating over all the links, extracting its attributes. We sum all these values to get the total number of segments in the network
rho, _ = mpc.state("rho", n_seg, lb=0) # function returns rho; ensures that density is not negative
v, _ = mpc.state("v", n_seg, lb=0) # creating velocity as state variable
w, _ = mpc.state("w", n_orig, lb=0, ub=[[np.inf], [100]])  # O2 queue is constrained
v_ctrl, _ = mpc.action("v_ctrl", len(L1.vsl), lb=20, ub=v_free) # creating a control action for VSL. Control speed cannot be lower than 20 km/h and more than free speed
r, _ = mpc.action("r", lb=0, ub=1) # creating control action for on-ramp -> in a form of a normalized ratio
d = mpc.disturbance("d", n_orig) # creating disturbance variable

# Adding dynamics constraints
mpc.set_dynamics(F) # ensures that any control actions produce feasible system trajectories

# Setting the optimization objective
# Before defining the objective, we set up parameters to hold the "last" (previous) control inputs
v_ctrl_last = mpc.parameter("v_ctrl_last", (v_ctrl.size1(), 1)) # parameter used to compute the difference between the previous and current control values
r_last = mpc.parameter("r_last", (r.size1(), 1)) # holds previous ramp metering values to enable penalization of abrupt changes
mpc.minimize(
    T * cs.sum2(cs.sum1(rho * L * lanes) + cs.sum1(w)) # penalizes high traffic density and long queues, thereby incentivizing control actions that reduce congestion
    + 0.4 * cs.sumsqr(cs.diff(cs.horzcat(v_ctrl_last, v_ctrl), 1, 1) / v_free) # penalizes large/ abrupt changes in the variable speed control, encouraging smoother transitions
    + 0.4 * cs.sumsqr(cs.diff(cs.horzcat(r_last, r))) # discourages rapid changes in ramp metering actions, promoting gradual adjustments
)

# Setting optimization solver for the MPC's NLP
opts = {
    "expand": True,
    "print_time": False, # disables the printing of timing information during the optimization process, keeping the output clear
    "ipopt": {"max_iter": 500, "sb": "yes", "print_level": 0}, # 500-> max. number of iterations allowed for the solver
}
mpc.init_solver(solver="ipopt", opts=opts)
# In case of integrating with SUMO, we don't need to create initial conditions, since these values will be taken from SUMO

# Defining edge IDs
mainline_edges = ["O1", "L1a", "L1b", "L1c", "L2", "D1"]
onramp_edges = ["O2"]
queue_edge_main = "E0"
traffic_light = "N2"

# Defining edge lengths
edge_lengths = { # in m
    "E0": 750,
    "O1": 1000,
    "L1a": 1000,
    "L1b": 1000,
    "L1c": 1000,
    "L2": 1000,
    "D1": 1000,
    "O2": 750
}

# Defining helper functions --> NumPy arrays need to be returned for speed and density on mainline, and for queue length on on-ramp and mainline
"""
Function 1: get_lanes_of_edges(edge_id)
Function for internal use.
This function filters the list of all lane IDs and returns those
that start with the given edge ID.
"""
def get_lanes_of_edges(edge_id):
    all_lanes = traci.lane.getIDList()
    lanes = [lane for lane in all_lanes if lane.startswith(edge_id)]
    return lanes

"""
Function 2: get_edge_density(edge_id)
This function computes the density (veh/km) for each edge in the given list (corresponds to segments from metanet model).
This function calculates the density for each individual edge/segment and returns a numpy array for these values.
"""
def get_edge_density(edge_ids):
    densities = [] # initializing an empty list to hold density values for each edge
    for edge in edge_ids:
        veh_count = traci.edge.getLastStepVehicleNumber(edge) # getting the number of vehicles on the edge
        length = edge_lengths[edge] # returning the length of the edge in meters
        if veh_count > 0:
            density = veh_count / (length / 1000.0)  # computing density in veh/km
        else:
            density = 0
        densities.append(density)
    return np.array(densities) # converting list to NumPy array e.g. [a,b,c,d,e,f]

"""
Function 3: get_edge_speed(edge_ids)
This function computes the weighted average (in km/h) for each edge in the provided list,
returning a NumPy array with one value per edge.
"""
def get_edge_speed(edge_ids):
    speeds = [] # list to store speed for each edge
    for edge in edge_ids:
        veh_count = traci.edge.getLastStepVehicleNumber(edge)
        speed_mps = traci.edge.getLastStepMeanSpeed(edge) # retrieving the average speed (m/s) for the edge from SUMO
        if veh_count > 0:
            speed_kms = speed_mps * 3.6
        else:
            speed_kms = 0
        speeds.append(speed_kms) # converting speed to km/h
    return np.array(speeds) # converting list to NumPy array e.g. [a,b,c,d,e,f]

"""
Function 4: get_edge_queues(mainline_edge, onramp_edge)
This function retrieves the queue lengths for two specified edges and returns them as a NumPy array
with two values [mainline_queue, onramp_queue].
"""
def get_edge_queues(mainline_edge, onramp_edge):
    mainline_q_n = traci.edge.getLastStepVehicleNumber(mainline_edge)
    onramp_q_n = traci.edge.getLastStepVehicleNumber(onramp_edge)
    return np.array([mainline_q_n, onramp_q_n])

"""
Function 5: set_vsl(edge_id,speed)
This function setting the maximum speed (in ms/s) for every lane on the given edges.
Since direct edge-level speed-setting is not available, this functon iterates over each edge's lanes.
"""
def set_vsl(edge_id, speed):
    lanes = get_lanes_of_edges(edge_id)
    for lane in lanes:
        traci.lane.setMaxSpeed(lane, float(speed))

"""
Function 6: update_ramp_signal_control_logic (metering_rate, cycle_duration, ramp_id)
In intersection N2, traffic light is always set to green for the mainline lanes and only on-ramp lane is set to red.
The function takes the metering rate(number between 0 and 1). multiplies it by the cycle duration to compute the green
time for the on-ramp lane, and then calculates the remaining (red) time for the on-ramp. The traffic light program is
assumed to consist of two phases:
Phase 0: The on-ramp has a green signal (with the mainline already green).
Phase 1: The on-ramp signal is red (again, the mainline remains green). 
Thus, the function adjusts the traffic light program for intersection 'ramp_id' (e.g. N2) to control the on-ramp signal
based on the metering rate. 
"""
def update_ramp_signal_control_logic(metering_rate, cycle_duration, ramp):
    # calculating green and red times (in seconds)
    green_time = int(metering_rate * cycle_duration)
    red_time = cycle_duration - green_time

    # retrieving the current traffic light program logic for the given ramp_id
    program_logic = traci.trafficlight.getAllProgramLogics(ramp)[0]

    # iterating over the phases in the program logic
    for ph_id in range(0,len(program_logic.phases)):
        if ph_id == 0: # green phase
            # Set phase 0 to have exactly green_time for the on-ramp
            program_logic.phases[ph_id].minDur = green_time
            program_logic.phases[ph_id].maxDur = green_time
            program_logic.phases[ph_id].duration = green_time
        elif ph_id == 1: # red phase
            # Set phase 1 to have exactly red_time for the on-ramp.
            program_logic.phases[ph_id].minDur = red_time
            program_logic.phases[ph_id].maxDur = red_time
            program_logic.phases[ph_id].duration = red_time

    # applying updated program logic to the traffic light
    traci.trafficlight.setProgramLogic(ramp, program_logic)
    # reset the phase to 0 so that the cycle starts with the on-ramp green phase
    traci.trafficlight.setPhase(ramp, 0)

# Initializing data containers as empty lists
time_steps = []
mainline_density_log = []
mainline_speed_log = []
queue_log = []
vsl_log = []
ramp_log = []
q_log = []
q_o_log = []

# Setting initial last conditions
v_ctrl_last = v[: L1.N][L1.vsl] # setting initial "last" variable speed control input
r_last = cs.DM.ones(r.size1(), 1) # initializes the previous ramp metering control values
sol_prev = None # prepares a variable to store the previous solution from the MPC colver


# Setting up SUMO environment
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

# Defining SUMO configurations
Sumo_config = [
    'sumo-gui',
    '-c', 'network_example.sumocfg',
    '--lateral-resolution', '0.1'
]

# Starting SUMO
traci.start(Sumo_config)

# Starting SUMO simulation
for k in range(demands.shape[0]): # iterates over each simulation time step, where the total number of steps is determined by the number of rows in the demands array
    if traci.simulation.getMinExpectedNumber() > 0 and traci.simulation.getTime() < 9000: # the loop runs the simulation as long as there are vehicles expected to be inserted or still on the road
        traci.simulationStep()

        # getting current simulation time
        sim_time = traci.simulation.getTime()
        time_steps.append(sim_time)

        # using helper functions to define the current states
        mainline_speed = get_edge_speed(mainline_edges)
        mainline_density = get_edge_density(mainline_edges)
        queues = get_edge_queues(queue_edge_main, onramp_edges[0])

        # getting the current demand values
        d_hat = demands[k: k + Np * M, :]
        if d_hat.shape[0] < Np * M:
            d_hat = np.pad(d_hat, ((0, Np * M - d_hat.shape[0]), (0, 0)), "edge")
        # Transpose d_hat as the mpc.solve() function expects this format
        d_hat = d_hat.T

        # solving MPC problem every M steps
        if k % M == 0:
            sol = mpc.solve(
                pars={
                    "rho_0": mainline_density,
                    "v_0": mainline_speed,
                    "w_0": queues,
                    "d": d_hat,
                    "v_ctrl_last": v_ctrl_last,
                    "r_last": r_last,
                },
                vals0=sol_prev, # uses the previous solution as a warm start to speed up convergence
            )
            sol_prev=sol.vals # updates sol_prev with the current solution, to be used as an initial guess in the next optimization
            v_ctrl_last=sol.vals["v_ctrl"][:, 0] # extracts the first control action from the MPC solution -> these values will be applied in the next simulation step
            r_last = sol.vals["r"][0]

            # using dynamic function F to get flows --> q, q_o
            state_prediction = np.concatenate([mainline_density, mainline_speed, queues])
            state_dm = cs.DM(state_prediction)
            x_next, q_all = F(state_dm, cs.vertcat(v_ctrl_last, r_last), demands[k, :])
            # splitting q_all into segment-based flows
            q, q_o = cs.vertsplit(q_all, (n_seg, n_seg + n_orig))
            # converting flows to NumPy arrays
            q_np = np.array(q.full()).flatten()
            q_o_np = np.array(q_o.full()).flatten()


            # applying control actions in SUMO via helper functions
            set_vsl("L1a", v_ctrl_last)
            set_vsl("L1b", v_ctrl_last)
            update_ramp_signal_control_logic(r_last, 60, traffic_light)

        # appending the current states, outputs, and control actions to the respective lists so they can be used for analysis
        mainline_density_log.append(mainline_density)
        mainline_speed_log.append(mainline_speed)
        queue_log.append(queues)
        vsl_log.append(v_ctrl_last)
        ramp_log.append(r_last)
        q_log.append(q_np) # TODO: Might get issues if q_np is not initialized yet (due to M steps not occuring yet)
        q_o_log.append(q_o_np)

        if k % 100 == 0:
            print(f"step {k} of {demands.shape[0]}")
    else:
        break

traci.close()

# Computing Total Time Spent (TTS) --> vehicle-hours
"""
T is the time step in hours --> 10/3600 h
For each simulation step, we sum:
- The total number of vehicles on the mainline
- Plus the total queue count (from the two queue measurements)
"""
tts = T * sum(
    (np.sum( mainline_density * L * lanes) + np.sum(queues))
    for mainline_density, queues in zip(mainline_density_log, queue_log)
)
print(f"TTS = {tts:.3f} veh.h")