# importing all required packages
import os
import sys
import numpy as np
import casadi as cs
import matplotlib.pyplot as plt

from csnlp import Nlp
from csnlp.wrappers import Mpc
import sym_metanet as metanet
from sym_metanet import (
    Destination,
    Link,
    MeteredOnRamp,
    MainstreamOrigin,
    Network,
    Node,
    engines,
)

from example import rho_max
from network_replication.network_traffic_control_traci import Sumo_config

# Setting up SUMO
if 'SUMO_HOME' in os.environ:
    tools= os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME")

import traci

# Defining SUMO configurations
Sumo_config = [
    'sumo-gui',
    '-c', 'network_example.sumocfg'
]

# Starting SUMO
traci.start(Sumo_config)

# Defining lane IDs and traffic light for ramp metering
mainline_lanes = ["L1c_0", "L1c_1"]
onramp_lane = "O2_0"
all_lanes = mainline_lanes + [onramp_lane]
ramp_tl_id = "N2,"
default_speed = 28.33 # m/s --> 102 km/h

# Data containers for recording simulation results
time_steps = []
traffic_data = {
    lane: {
        "flow": [],
        "speed": [],
        "density": []
    } for lane in all_lanes
}

# Getting traffic states - these measurements are later used as the observed state to feed into MPC controller
def get_traffic_state(lane_id):
    vehicle_count = traci.lane.getLastStepVehicleNumber(lane_id)
    speed_mps = traci.lane.getLastStepMeanSpeed(lane_id)
    speed_kmh = speed_mps * 3.6
    lane_length_m = traci.lane.getLength(lane_id)
    density = vehicle_count/ (lane_length_m/ 1000.0) # veh/km
    flow = speed_mps * density * 3.6
    return flow, speed_kmh, density

# MPC and Network Model Setup
# Setting parameters (CHECK THOSE!!!) -> used to define the metanet model.
# My SUMO network: one main road of 6 km with 2 lanes.
#    We represent the main road as two links:
#      - L1: from node N1 to N2, length = 3000 m (divided into 3 segments, each 1000 m)
#      - L2: from node N2 to N3, length = 3000 m (divided into 3 segments, each 1000 m)
#    The on-ramp (O2) is attached at node N2.
T = 10/3600 # simulation time step in hours (10 s)
L = 2 # segment length
lanes = 2 # number of lanes on the mainline
C = (4000,2000) # capacities for mainline and on ramp
tau = 18/3600
kappa = 40
eta = 60
rho_max = 180 # veh/km/lane
a = 1.867
rho_crit = 33.5 # veh/km/lane
v_free = 102 # free-flow speed (km/h ) --> perhaps change to m/s

# Build a simplified metanet network model.
# Here we create a mainline link (divided into several segments) and a metered onramp.
N1 = Node("N1")
N2 = Node("N2")
N3 = Node("N3")
O1 = MainstreamOrigin("O1")   # mainline origin
O2 = MeteredOnRamp(C[1], "O2")  # onramp with metering control
D1 = Destination("D1")
# Create a mainline link (without VSL for now)
# Upstream link L1: 3000 m total, 3 segments → each segment = 1000 m.
L1 = Link(3, lanes, 3000/3, rho_max, rho_crit, v_free, a, "L1")
# Downstream link L2: 3000 m total, 3 segments.
L2 = Link(3, lanes, 3000 / 3, rho_max, rho_crit, v_free, a, "L2")
net = (
    Network("MyNetwork")
    .add_path(origin=O1, path=(N1, L1, N2, L2, N3), destination=D1)
    .add_origin(O2, N2)
)

# initializing metanet engine with CasAdi
# Telling metanet to use the CasAdi backend
engines.use("casadi", sym_type = "SX")
# Checking that my network definition is consistent
net.is_valid(raises = True)
# Provide an initial condition for v_ctrl (not used in our ramp-only control here)
net.step(T=T, tau=tau, eta=eta, kappa=kappa, init_conditions={O1: {"v_ctrl": v_free * 2}})

# Creating the CasADi function representing the dynamics: F(x,u,d)
# F becomes a CasADi function that represents the network dynamics. This function takes the current state, control and disturbances, and returns the next state and flows.
F = metanet.engine.to_function(net=net, more_out=True, compact=2, T=T)

# Setup MPC using csnlp
# Define MPC horizons and spacing
# 60 seconds per control update
# the MPC predicts 42 steps (7 x 6) ahead and controls over 30 steps (5 x 6 = 30 steps).
Np, Nc, M_spacing = 7, 5, 6 # For a 60-second cycle: 6 steps * 10 s = 60 s
mpc = Mpc(
    nlp=Nlp(sym_type="SX"),
    prediction_horizon=Np * M_spacing,
    control_horizon=Nc * M_spacing,
    input_spacing=M_spacing,
)

# State variables:
# - rho: mainline density in each segment (n_seg segments)
# - w: onramp queue (modeled as one state variable)
n_seg = L1.N + L2.N
n_orig = 1    # one onramp
rho, _ = mpc.state("rho", n_seg, lb=0) # density in each segment of the mainline link
w, _ = mpc.state("w", n_orig, lb=0, ub=[[np.inf]]) # the queue length at the onramp

# Control variables: ramp metering rate r (0 = strict metering, 1 = free-flow)
r, _ = mpc.action("r", lb=0, ub=1)

# Disturbance: onramp demand forecast (if needed)
d = mpc.disturbance("d", n_orig)

# Parameter: previous control action (for smoothing). Used to penalize large changes in the metering rate between successive control intervals.
r_last = mpc.parameter("r_last", (r.size1(), 1))

# Add the dynamics (from the metanet function F)
mpc.set_dynamics(F)

# Define the optimization objective.
# We aim to minimize total time spent (approximated by mainline occupancy and ramp queue)
# The CasADi function F is set as the dynamic constraint for the MPC
# plus a smoothing term on the metering rate.
# For the mainline occupancy, we compute an average segment length:
avg_segment_length = (6000) / n_seg  # here 6000 m total / 6 segments = 1000 m per segment.
mpc.minimize(
    T * cs.sum2(cs.sum1(rho * avg_segment_length * lanes) + cs.sum1(w))
    + 0.4 * cs.sumsqr(cs.diff(cs.horzcat(r_last, r)))
) # gives estimation in veh/h --> maybe change that?

# Initialize the solver (IPOPT)
opts = {"expand": True, "print_time": False, "ipopt": {"max_iter": 500, "sb": "yes", "print_level": 0}}
mpc.init_solver(solver="ipopt", opts=opts)

# Start with an initial metering rate of 1 (i.e. initially no metering applies)
r_last_val = cs.DM.ones(r.size1(), 1)

# --- Control Loop Settings ---
# We now use a 60-second control interval.
# Every 60 seconds, the MPC problem is solved and a new ramp metering rate is computed.
# The computed rate (between 0 and 1) determines the fraction of a 60s cycle during which the ramp's traffic light is green.
control_interval = 60  # seconds
# We will store the current computed metering rate (from MPC) and use it to update the ramp traffic light.
current_metering_rate = 1.0  # initial value (1 means full green)

# --- Simulation Loop ---
while traci.simulation.getMinExpectedNumber() > 0 and traci.simulation.getTime() < 9000:
    traci.simulationStep()
    sim_time = traci.simulation.getTime()
    time_steps.append(sim_time)

    # Record traffic states for each lane
    for lane in all_lanes:
        flow, speed, density = get_traffic_state(lane)
        traffic_data[lane]["flow"].append(flow)
        traffic_data[lane]["speed"].append(speed)
        traffic_data[lane]["density"].append(density)

    # Every control_interval seconds, update the MPC solution.
    if sim_time % control_interval == 0:
        # --- 1. Extract Measurements ---
        # For the mainline, take the average density over the two lanes.
        mainline_densities = []
        for lane in mainline_lanes:
            _, _, d_val = get_traffic_state(lane)
            mainline_densities.append(d_val)
        measured_rho = np.mean(mainline_densities)

        # For the onramp, use the measured density as a proxy for queue length.
        _, _, ramp_density = get_traffic_state(onramp_lane)
        measured_w = ramp_density

        # --- 2. Map Measurements to MPC State Vectors ---
        # Here we assume that the density is uniform across all segments.
        rho0 = cs.DM.ones(n_seg, 1) * measured_rho
        w0 = cs.DM([measured_w])

        # --- 3. Create a Simple Demand Forecast ---
        # For simplicity, assume constant forecast using the measured value.
        d_forecast = cs.DM.ones(n_orig, 1) * measured_w
        d_forecast_mat = cs.repmat(d_forecast, 1, Np * M_spacing)

        # --- 4. Solve the MPC Problem ---
        sol = mpc.solve(
            pars={
                "rho_0": rho0,
                "w_0": w0,
                "d": d_forecast_mat,
                "r_last": r_last_val,
            },
            vals0=None,  # You may warm-start with a previous solution if available.
        )
        # The MPC problem is solved with these parameters, yielding an optimal ramp metering rate (r_opt).
        r_opt = sol.vals["r"][:, 0]  # get the control action for the next interval
        r_last_val = r_opt  # update for the next MPC solve
        current_metering_rate = float(r_opt)  # update the metering rate (0 to 1)
        print(f"[{sim_time:.1f}s] MPC computed metering rate: {current_metering_rate:.2f}")

        # --- 5. Update the Ramp Traffic Light ---
        # We emulate ramp metering by switching the traffic light "rampTL" on the onramp.
        # The cycle time is 60 seconds. The MPC metering rate determines the green time.
        cycle_time = 60.0  # seconds
        # Determine current time within the 60-second cycle:
        cycle_time_pos = sim_time % cycle_time
        # Compute green time based on the metering rate.
        green_time = current_metering_rate * cycle_time
        if cycle_time_pos < green_time:
            # Allow vehicles to enter: set the onramp traffic light to green.
            # Here, "G" denotes green; adjust the state string to match your network's definitions.
            traci.trafficlight.setRedYellowGreenState(ramp_tl_id, "G")
        else:
            # Disallow entry: set the traffic light to red.
            traci.trafficlight.setRedYellowGreenState(ramp_tl_id, "r")

traci.close()

# --- Plotting the Results ---
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Plot Flow for each lane.
for lane in all_lanes:
    axes[0].plot(time_steps, traffic_data[lane]["flow"], label=lane)
axes[0].set_ylabel("Flow [veh/h]")
axes[0].set_title("Traffic Flow over Time")
axes[0].legend()
axes[0].grid(True)

# Plot Speed for each lane.
for lane in all_lanes:
    axes[1].plot(time_steps, traffic_data[lane]["speed"], label=lane)
axes[1].set_ylabel("Speed [km/h]")
axes[1].set_title("Traffic Speed over Time")
axes[1].legend()
axes[1].grid(True)

# Plot Density for each lane.
for lane in all_lanes:
    axes[2].plot(time_steps, traffic_data[lane]["density"], label=lane)
axes[2].set_ylabel("Density [veh/km]")
axes[2].set_xlabel("Time [s]")
axes[2].set_title("Traffic Density over Time")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.show()





