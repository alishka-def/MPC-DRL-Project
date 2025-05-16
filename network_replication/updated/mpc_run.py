import os
import sys
import traci

import numpy as np
import casadi as cs
import sym_metanet as metanet
import pandas_read_xml as pdx
import matplotlib.pyplot as plt
import seaborn as sns

from csnlp import Nlp
from csnlp.wrappers import Mpc
from sym_metanet import (
    Destination, Link, LinkWithVsl, MainstreamOrigin, MeteredOnRamp, Network, Node, engines,
)

# Setting up SUMO environment
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    #libsumo = os.path.join(os.environ['SUMO_HOME'], 'libsumo')
    sys.path.append(tools)
    #sys.path.append(libsumo)

else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

# Defining SUMO configurations
Sumo_config = [
    'sumo-gui',
    '-c', 'network_example.sumocfg',
    '--lateral-resolution', '0.1'
]

# Defining edge IDs
mainline_edges = ["O1", "L1a", "L1b", "L1c", "L2", "D1"]
onramp_edges = ["O2"]
queue_edge_main = "E0"
traffic_light = "J1"

# Defining edge lengths
edge_lengths = { # in m
    "E0": 750,
    "O1": 1000,
    "L1a": 1000,
    "L1b": 1000,
    "L1c": 1000,
    "L2": 1000,
    "D1": 1000,
    "O2": 700
}

edge_lanes = {
    "E0": 2,
    "O1": 1,
    "L1a": 2,
    "L1b": 2,
    "L1c": 2,
    "L2": 2,
    "D1": 2,
    "O2": 1
}
FREE_FLOW_SPEED = 102 # km/h

##################################################################################
# FUNCTIONS
##################################################################################
def create_demands(time: np.ndarray) -> np.ndarray: # input should be a NumPy array, as well as the output
    return np.stack( # outputs from the two interpolations will be combines into a single NumPy array
        (
            np.interp(time, (2.0, 2.25), (3500, 1000)),
            np.interp(time, (0.0, 0.15, 0.35, 0.5), (500, 1500, 1500, 500)),
        )
    )


def get_lanes_of_edges(edge_id):
    """
        Function 1: get_lanes_of_edges(edge_id)
        Function for internal use.
        This function filters the list of all lane IDs and returns those
        that start with the given edge ID.
    """
    all_lanes = traci.lane.getIDList()
    lanes = [lane for lane in all_lanes if lane.startswith(edge_id)]
    return lanes


def get_edge_density(edge_ids):
    """
        Function 2: get_edge_density(edge_id)
        This function computes the density (veh/km) for each edge in the given list (corresponds to segments from metanet model).
        This function calculates the density for each individual edge/segment and returns a numpy array for these values.
    """
    densities = [] # initializing an empty list to hold density values for each edge
    lane_densities = [] # initializing an empty list to hold density values for each edge
    for edge in edge_ids:
        veh_count = traci.edge.getLastStepVehicleNumber(edge) # getting the number of vehicles on the edge
        length = edge_lengths[edge] # returning the length of the edge in meters
        density = veh_count / (length / 1000.0) / edge_lanes[edge]  # computing density in veh/km
        densities.append(density)
        lane_densities.append(density / edge_lanes[edge])  # computing density in veh/km/lane
    return np.array(densities), np.array(lane_densities) # converting list to NumPy array e.g. [a,b,c,d,e,f]


def get_edge_speed(edge_ids):
    """
        Function 3: get_edge_speed(edge_ids)
        This function computes the weighted average (in km/h) for each edge in the provided list,
        returning a NumPy array with one value per edge.
    """
    speeds = [] # list to store speed for each edge
    for edge in edge_ids:
        veh_count = traci.edge.getLastStepVehicleNumber(edge)
        speed_mps = traci.edge.getLastStepMeanSpeed(edge) # retrieving the average speed (m/s) for the edge from SUMO
        if veh_count > 0:
            speed_kmh = speed_mps * 3.6
        else:
            speed_kmh = FREE_FLOW_SPEED
        speeds.append(speed_kmh) # converting speed to km/h
    return np.array(speeds) # converting list to NumPy array e.g. [a,b,c,d,e,f]


def get_edge_queues(mainline_edge, onramp_edge):
    """
        Function 4: get_edge_queues(mainline_edge, onramp_edge)
        This function retrieves the queue lengths for two specified edges and returns them as a NumPy array
        with two values [mainline_queue, onramp_queue].
    """
    mainline_q_n = traci.edge.getLastStepVehicleNumber(mainline_edge)
    onramp_q_n = traci.edge.getLastStepVehicleNumber(onramp_edge)
    return np.array([mainline_q_n, onramp_q_n])


def set_vsl(edge_id, speed):
    """
        Function 5: set_vsl(edge_id,speed)
        This function setting the maximum speed (in m/s) for every lane on the given edges.
        Since direct edge-level speed-setting is not available, this functon iterates over each edge's lanes.
    """
    lanes = get_lanes_of_edges(edge_id)
    for lane in lanes:
        traci.lane.setMaxSpeed(lane, float(speed))


def update_ramp_signal_control_logic(metering_rate, cycle_duration, ramp):
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
    # calculating green and red times (in seconds)
    green_time = int(metering_rate * cycle_duration)
    red_time = cycle_duration - green_time

    # retrieving the current traffic light program logic for the given ramp_id
    program_logic = traci.trafficlight.getAllProgramLogics(ramp)[0]

    # iterating over the phases in the program logic
    for ph_id in range(0, len(program_logic.phases)):
        print(program_logic.phases[ph_id])
        #sys.exit(1)
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

##################################################################################
# CREATE METANET MODEL AND MPC OBJECT FOR NETWORK
##################################################################################
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
engines.use("casadi", sym_type="SX") 
net.is_valid(raises=True)
net.step( 
    T=T, tau=tau, eta=eta, kappa=kappa, delta=delta,
    init_conditions={O1: {"v_ctrl": v_free * 2}}, # sets the initial condition for Origin 1 - VSL is set to be 204 km/h
)
F: cs.Function = metanet.engine.to_function(
    net=net, more_out=True,  compact=2, T=T
)
# F: (x[14], u[3], d[2]) -> (x+[14], q[8])

# Creating demands for metanet model
demands = create_demands(time).T

# Creating the MPC controller
Np, Nc, M = 7, 5, 6 # 7-> base number of prediction steps for mpc; 5-> base number of control steps; 6-> multiplier
mpc = Mpc[cs.SX]( # creating mpc controller
    nlp=Nlp[cs.SX](sym_type="SX"), # setting up instance of nonlinear programming problem -> underlying optimization problem that mpc will solve
    prediction_horizon=Np * M, # over 42 steps
    control_horizon=Nc * M, # over 30 steps
    input_spacing=M,  # control inputs are adjusted every 6 steps -> mpc solves the optimization problem every 60 seconds
)

# Creating states, action, and disturbances for metanet model
n_seg, n_orig = sum(link.N for _, _, link in net.links), len(net.origins) 
rho, _ = mpc.state("rho", n_seg, lb=0) 
v, _ = mpc.state("v", n_seg, lb=0) 
w, _ = mpc.state("w", n_orig, lb=0, ub=[[np.inf], [100]])  
v_ctrl, _ = mpc.action("v_ctrl", len(L1.vsl), lb=20, ub=v_free) 
r, _ = mpc.action("r", lb=0, ub=1) 
d = mpc.disturbance("d", n_orig) 

# Adding dynamics constraints
mpc.set_dynamics(F)

# Setting the optimization objective
v_ctrl_last = mpc.parameter("v_ctrl_last", (v_ctrl.size1(), 1)) 
r_last = mpc.parameter("r_last", (r.size1(), 1)) 
mpc.minimize(
    T * cs.sum2(cs.sum1(rho * L * lanes) + cs.sum1(w)) 
    + 0.4 * cs.sumsqr(cs.diff(cs.horzcat(v_ctrl_last, v_ctrl), 1, 1) / v_free) 
    + 0.4 * cs.sumsqr(cs.diff(cs.horzcat(r_last, r))) 
)

# Setting optimization solver for the MPC's NLP
opts = {
    "expand": True,
    "print_time": False, 
    "ipopt": {"max_iter": 500, "sb": "yes", "print_level": 0}, 
}
mpc.init_solver(solver="ipopt", opts=opts)

##################################################################################
# Start Simulation
##################################################################################
# Different sampling time intervals
sumo_step, metanet_step = 1.0, T * 3600
control_step = M * metanet_step

# Saving results into a dictionary
times = np.arange(0, 9000, sumo_step)
results_sumo ={
    'Time': times,
    'Density': np.zeros(shape=(len(mainline_edges), len(times))),
    'Density_perLane': np.zeros(shape=(len(mainline_edges), len(times))),
    'Flow': np.zeros(shape=(len(mainline_edges), len(times))),
    'Speed': np.zeros(shape=(len(mainline_edges), len(times))),
    'Queue_Lengths': np.zeros(shape=(2, len(times))),
    'Ramp_Metering_Rate': np.zeros(shape=(1, len(times))),
    'VSL': np.zeros(shape=(2, len(times))),
}

# Setting initial last conditions
v_ctrl_last = v_free*cs.DM.ones(len(L1.vsl), 1)
r_last = cs.DM.ones(r.size1(), 1)
sol_prev = None

# Start Traci
traci.start(Sumo_config)
# Starting simulation with ramp metering rate to be 1 (60 seconds of green light)
update_ramp_signal_control_logic(metering_rate=1.0, cycle_duration=60, ramp=traffic_light)
for k_sumo in range(len(times)):
    # getting current simulation time
    sim_time = traci.simulation.getTime()
    results_sumo['Time'][k_sumo] = sim_time

    # using helper functions to define the current states
    results_sumo['Speed'][:, k_sumo] = get_edge_speed(mainline_edges)
    results_sumo['Density'][:, k_sumo], results_sumo['Density_perLane'][:, k_sumo] = get_edge_density(mainline_edges)
    results_sumo['Flow'][:, k_sumo] = results_sumo['Density'][:, k_sumo] * results_sumo['Speed'][:, k_sumo]
    results_sumo['Queue_Lengths'][:, k_sumo] = get_edge_queues(queue_edge_main, onramp_edges[0])

    
    # getting the current demand values
    k = int(sim_time // metanet_step)
    d_hat = demands[k: k + Np * M, :]
    if d_hat.shape[0] < Np * M:
        d_hat = np.pad(d_hat, ((0, Np * M - d_hat.shape[0]), (0, 0)), "edge")

    # at the beginning of the simulation I have added a warm-up period of 10 minutes (600 seconds) so that network has some logical density and speed measurements.
    if k_sumo > 600 and sim_time % control_step == 0:
        # Here, we handle the effect of the different time steps used for SUMO and METANET by averaging the last 10 measurements (i.e. spanning 10 seconds)
        mainline_density_perLane = np.mean(results_sumo['Density_perLane'][:, k_sumo-int(metanet_step//sumo_step)+1:k_sumo+1], axis=1)
        mainline_density = np.mean(results_sumo['Density'][:, k_sumo-int(metanet_step//sumo_step)+1:k_sumo+1], axis=1)
        mainline_flow = np.mean(results_sumo['Flow'][:, k_sumo-int(metanet_step//sumo_step)+1:k_sumo+1], axis=1)
        mainline_speed = np.minimum(FREE_FLOW_SPEED, mainline_flow / mainline_density + 1e-06) # avoid division by zero
        queues = np.mean(results_sumo['Queue_Lengths'][:, k_sumo-int(metanet_step//sumo_step)+1:k_sumo+1], axis=1)
        
        sol = mpc.solve(
            pars={"rho_0": mainline_density_perLane, "v_0": mainline_speed, "w_0": queues, "d": d_hat.T, "v_ctrl_last": v_ctrl_last, "r_last": r_last},
            vals0=sol_prev,
        ) 

        sol_prev = sol.vals
        v_ctrl_last = sol.vals["v_ctrl"][:, 0]
        r_last = sol.vals["r"][0]

        if np.isnan(v_ctrl_last[0]) or np.isnan(v_ctrl_last[1]) or np.isnan(r_last):
            print(v_ctrl_last[0], v_ctrl_last[1], r_last)
            traci.close()
            sys.exit(1)

        # applying control actions in SUMO via helper functions
        set_vsl("L1a", v_ctrl_last[0]*3.6) # convert VSL from km/h to m/s
        set_vsl("L1b", v_ctrl_last[1]*3.6)
        print("v_ctrl_last: ", v_ctrl_last)
        print("r_last: ", r_last)
        update_ramp_signal_control_logic(r_last.__float__(), control_step, traffic_light)

    results_sumo['Ramp_Metering_Rate'][:, k_sumo] = np.asarray(r_last).flatten()
    results_sumo['VSL'][:, k_sumo] = np.asarray(v_ctrl_last).flatten()
    traci.simulationStep()

traci.close()

# Total Time Spent metrics
tts = T * np.sum(np.sum(results_sumo['Density'], axis=0) * L * lanes + np.sum(results_sumo['Queue_Lengths'], axis=0))
print(f"TTS = {tts:.3f} veh.h")

##################################################################################
# Plotting
##################################################################################
plt.figure()
plt.plot(results_sumo['Time'], results_sumo['Ramp_Metering_Rate'].T)

plt.figure()
plt.plot(results_sumo['Time'], results_sumo['VSL'].T)

plt.show()
# save plot to ./plots
plt.savefig('./plots/ramp_metering_rate_no_training.png')
#sys.exit(1)

df = pdx.read_xml('./Logs/log_edges.xml', ['meandata'])
df = pdx.flatten(df)
df = df.pipe(pdx.flatten)
df = df.pipe(pdx.flatten)
df = df.pipe(pdx.flatten)
df = df.rename(
    columns={
        'interval|@begin': 'begin', 'interval|@end': 'end', 'interval|edge|@sampledSeconds': 'sampledSeconds',
        'interval|edge|@density': 'density', 'interval|edge|@laneDensity': 'laneDensity', 'interval|edge|@speed': 'speed', 'interval|edge|@occupancy': 'occupancy',
        'interval|edge|@id': 'edge_id'
    }
)
df['begin'] = df['begin'].astype(float)
df['end'] = df['end'].astype(float)
df["sampledSeconds"] = df["sampledSeconds"].astype(float)
df["density"] = df["density"].astype(float)
df["laneDensity"] = df["laneDensity"].astype(float)
df["speed"] = df["speed"].astype(float)
df = df.replace(np.nan, 0)
df['begin'] = df['begin'].astype(int)
df['flow'] = df['density'] * df['speed']

print(df.columns)

plt.figure()
sns.lineplot(data=df, x='begin', y='density', hue='edge_id')

plt.figure()
sns.lineplot(data=df, x='begin', y='flow', hue='edge_id')

plt.figure()
sns.lineplot(data=df, x='begin', y='speed', hue='edge_id')
plt.show()
# save plot to ./plots
#plt.savefig('./plots/speed.png')