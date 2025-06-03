##################################################################################
# IMPORTS
##################################################################################
import os
import sys
import traci

import numpy as np
import casadi as cs
import sym_metanet as metanet
import matplotlib.pyplot as plt

from csnlp import Nlp
from csnlp.wrappers import Mpc
from stable_baselines3 import DDPG
from sym_metanet import (
    Destination, Link, LinkWithVsl, MainstreamOrigin, MeteredOnRamp, Network, Node, engines,
)
from matplotlib.colors import Normalize

########################################################################
# Global: Parameters
########################################################################
RUN_MODE = "NO_CTRL" # options: "NO_CTRL" or "MPC" or "MPC_DRL"

##################################################################################
# Setting up SUMO Environment
##################################################################################
# Setting up SUMO environment
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    libsumo = os.path.join(os.environ['SUMO_HOME'], 'libsumo')
    sys.path.append(tools)
    sys.path.append(libsumo)

else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

# Defining SUMO configurations
Sumo_config = [
    "sumo-gui",
    '-c', 'toy_highway_network.sumocfg',
    '--lateral-resolution', '0.1', 
    '--seed', '1'
]


##################################################################################
# Setting up network parameters
##################################################################################
mainline_edges = ["O1", "L1a", "L1b", "L1c", "L2", "D1"]
onramp_edges = ["O2"]
queue_edge_main = "E0"
traffic_light = "J1"
FREE_FLOW_SPEED = 102 # km/h
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
    "O1": 2,
    "L1a": 2,
    "L1b": 2,
    "L1c": 2,
    "L2": 2,
    "D1": 2,
    "O2": 1
}

##################################################################################
# FUNCTIONS
##################################################################################
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
        density = veh_count / (length / 1000.0)  # computing density in veh/km
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


def create_demands(time: np.ndarray) -> np.ndarray:
    return np.stack(
        (
            #np.interp(time, (0, 15/60, 2.50, 2.75, 3.00, 3.25), (0, 3500, 3500, 1000, 1000, 0)), 
            #np.interp(time, (0, 15/60, 30/60, 0.60, 0.85, 1.0, 3.00, 3.25), (0, 500, 500, 1500, 1500, 500, 500, 0))

            np.interp(time, (0, 15/60, 2.50, 2.75, 3.00, 3.25), (0, 2000, 2000, 800, 800, 0)), 
            np.interp(time, (0, 15/60, 30/60, 0.60, 0.85, 1.0, 3.00, 3.25), (0, 500, 500, 1000, 1000, 500, 500, 0))
        )
    )


def create_MPC(metanet_interval, mpc_interval):
    T = metanet_interval # hr
    L = 1 # km
    lanes = 2
    C = (4000, 2000) # veh/h
    tau = 18 / 3600 # hr
    kappa, rho_max, rho_crit = 40, 180, 33.5 # veh/km/lane
    delta, a = 0.0122, 1.867 
    eta = 60
    v_free = 102 # km/h
    args = (lanes, L, rho_max, rho_crit, v_free, a)

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

    # Creating the MPC controller
    Np, Nc, M = 2, 2, int(np.round(mpc_interval/T, decimals=0))
    print("MPC control spacing (M_mpc) = ", M)
    mpc = Mpc[cs.SX]( # creating mpc controller
        nlp=Nlp[cs.SX](sym_type="SX"), 
        prediction_horizon=Np*M, 
        control_horizon=Nc*M, 
        input_spacing=M, 
    )

    # Creating states, action, and disturbances for metanet model
    n_seg, n_orig = sum(link.N for _, _, link in net.links), len(net.origins) 
    rho, _ = mpc.state("rho", n_seg, lb=0) 
    v, _ = mpc.state("v", n_seg, lb=0) 
    w, _ = mpc.state("w", n_orig, lb=0, ub=[[np.inf], [30]])  
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

    # Setting initial last conditions
    v_ctrl_last = v_free*cs.DM.ones(len(L1.vsl), 1)
    r_last = cs.DM.ones(r.size1(), 1)

    return mpc, Np, M, v_ctrl_last, r_last


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

        d_norm = demands / np.array([2000.0, 1000.0]) #np.array([3500.0, 1500.0])
        return np.concatenate([rho_norm, v_norm, w_norm, u_mpc_norm, d_norm, u_prev_norm], dtype=np.float32)


##################################################################################
# MAIN: Parameters
##################################################################################
cycle_duration = 60.0
T_warmup, T_sim, T_cooldown = 30.0/60, 2.5, 180.0/60
sumo_step, metanet_step, drl_step, mpc_step = 1.0, 10.0, 60.0, 300.0
mpc_active = RUN_MODE.startswith("MPC")

# Saving results into a dictionary
times = np.arange(0, (T_warmup+T_sim+T_cooldown)*3600+sumo_step, sumo_step)

if mpc_active:
    demands_forecast = create_demands(times/3600).T
    mpc, Np, M_mpc, v_ctrl_last, r_last = create_MPC(metanet_step/3600, mpc_step/3600)
    v_ctrl_mpc, r_mpc = v_ctrl_last, r_last
    sol_prev = None
    print("created MPC object")

results_sumo = {
    'Time': times,
    'Density': np.zeros(shape=(len(mainline_edges), len(times))),
    'Density_perLane': np.zeros(shape=(len(mainline_edges), len(times))),
    'Flow': np.zeros(shape=(len(mainline_edges), len(times))),
    'Speed': np.zeros(shape=(len(mainline_edges), len(times))),
    'Queue_Lengths': np.zeros(shape=(2, len(times))),
    'Ramp_Metering_Rate_MPC': np.zeros(shape=(1, len(times))),
    'VSL_MPC': np.zeros(shape=(2, len(times))),
    'Ramp_Metering_Rate': np.zeros(shape=(1, len(times))),
    'VSL': np.zeros(shape=(2, len(times))),
}

# 1) find the directory this script lives in
here = os.path.dirname(os.path.realpath(__file__))
# 2) climb up into your project root and then down into the folder with your zip
zip_path = os.path.normpath(
    os.path.join(here, "..",           # up from new_codes/
                        "logs",
                        "low",
                        "ddpg_low_final.zip")
)
if RUN_MODE == "MPC_DRL":
    drl_agent = DDPG.load(zip_path)

##################################################################################
# MAIN: Start Simulation
##################################################################################
stop_ctrl = False
traci.start(Sumo_config)
update_ramp_signal_control_logic(metering_rate=1.0, cycle_duration=cycle_duration, ramp=traffic_light)
for k_sumo in range(len(times)):
    # getting current simulation time
    sim_time = traci.simulation.getTime()
    results_sumo['Time'][k_sumo] = sim_time

    # using helper functions to define the current states
    results_sumo['Speed'][:, k_sumo] = np.clip(get_edge_speed(mainline_edges), a_min=0, a_max=FREE_FLOW_SPEED)
    results_sumo['Density'][:, k_sumo], results_sumo['Density_perLane'][:, k_sumo] = get_edge_density(mainline_edges)
    results_sumo['Flow'][:, k_sumo] = results_sumo['Density'][:, k_sumo] * results_sumo['Speed'][:, k_sumo]
    results_sumo['Queue_Lengths'][:, k_sumo] = get_edge_queues(queue_edge_main, onramp_edges[0])

    if mpc_active:
        if sim_time > (T_warmup+T_sim)*3600:
            if not stop_ctrl:   
                # disable control algorithm and set everything back to normal
                set_vsl("L1b", FREE_FLOW_SPEED/3.6) # convert VSL from km/h to m/s
                set_vsl("L1c", FREE_FLOW_SPEED/3.6)
                update_ramp_signal_control_logic(1.0, cycle_duration, traffic_light)             
                stop_ctrl = True
            results_sumo['Ramp_Metering_Rate'][:, k_sumo] = np.asarray(1.0).flatten()
            results_sumo['VSL'][:, k_sumo] = np.asarray([FREE_FLOW_SPEED, FREE_FLOW_SPEED]).flatten()
            results_sumo['Ramp_Metering_Rate_MPC'][:, k_sumo] = np.asarray(1.0).flatten()
            results_sumo['VSL_MPC'][:, k_sumo] = np.asarray([FREE_FLOW_SPEED, FREE_FLOW_SPEED]).flatten()
            traci.simulationStep()
            continue

        # getting the current demand values
        k = int(sim_time // metanet_step)
        d_hat = demands_forecast[k: k + Np * M_mpc, :]
        if d_hat.shape[0] < Np * M_mpc:
            d_hat = np.pad(d_hat, ((0, Np * M_mpc - d_hat.shape[0]), (0, 0)), "edge")
        
        if sim_time >= T_warmup*3600 and sim_time % mpc_step == 0:
            # Here, we handle the effect of the different time steps used for SUMO and METANET 
            # by averaging the last 10 measurements (i.e. spanning 10 seconds)
            mainline_density_perLane = np.mean(results_sumo['Density_perLane'][:, k_sumo-int(metanet_step//sumo_step)+1:k_sumo+1], axis=1)
            mainline_density = np.mean(results_sumo['Density'][:, k_sumo-int(metanet_step//sumo_step)+1:k_sumo+1], axis=1)
            mainline_flow = np.mean(results_sumo['Flow'][:, k_sumo-int(metanet_step//sumo_step)+1:k_sumo+1], axis=1)
            mainline_speed = np.minimum(FREE_FLOW_SPEED, mainline_flow / (mainline_density + 1e-06)) # avoid division by zero
            queues = np.mean(results_sumo['Queue_Lengths'][:, k_sumo-int(metanet_step//sumo_step)+1:k_sumo+1], axis=1)

            sol = mpc.solve(
                pars={"rho_0": mainline_density_perLane, "v_0": mainline_speed, "w_0": queues, "d": d_hat.T, "v_ctrl_last": v_ctrl_last, "r_last": r_last},
                vals0=sol_prev,
            )

            sol_prev = sol.vals
            v_ctrl_mpc = sol.vals["v_ctrl"][:, 0] # np.round(sol.vals["v_ctrl"][:, 0]/5, decimals=0)*5
            r_mpc = sol.vals["r"][0]

            if np.isnan(v_ctrl_mpc[0]) or np.isnan(v_ctrl_mpc[1]) or np.isnan(r_mpc):
                print(v_ctrl_mpc[0], v_ctrl_mpc[1], r_mpc)
                traci.close()
                sys.exit(1)

            if RUN_MODE == "MPC":
                v_ctrl_last = v_ctrl_mpc
                r_last = r_mpc
                # applying control actions in SUMO via helper functions
                set_vsl("L1b", v_ctrl_last[0]/3.6) # convert VSL from km/h to m/s
                set_vsl("L1c", v_ctrl_last[1]/3.6)
                update_ramp_signal_control_logic(r_last.__float__(), cycle_duration, traffic_light)

        if RUN_MODE == "MPC_DRL":
            results_sumo['Ramp_Metering_Rate_MPC'][:, k_sumo] = np.asarray(r_mpc).flatten()
            results_sumo['VSL_MPC'][:, k_sumo] = np.asarray(v_ctrl_mpc).flatten()

            if sim_time >= T_warmup*3600 and sim_time % drl_step == 0:
                mainline_density_perLane = np.mean(results_sumo['Density_perLane'][:, k_sumo-int(metanet_step//sumo_step)+1:k_sumo+1], axis=1)
                mainline_speed = np.minimum(FREE_FLOW_SPEED, mainline_flow / (mainline_density + 1e-06)) # avoid division by zero
                queues = np.mean(results_sumo['Queue_Lengths'][:, k_sumo-int(metanet_step//sumo_step)+1:k_sumo+1], axis=1)
                
                u_mpc = np.concatenate([r_mpc, v_ctrl_mpc]).flatten()
                obs = normalize_observations(mainline_density_perLane, mainline_speed, queues, u_mpc, 
                                             u_prev=np.concatenate([r_last, v_ctrl_last]).flatten(),
                                             demands=demands_forecast[k, :])
                u_drl, _ = drl_agent.predict(obs, deterministic=True)
                v_ctrl_last = np.clip(v_ctrl_mpc + u_drl[1:], a_min=20, a_max=FREE_FLOW_SPEED)
                r_last = np.clip(r_mpc + u_drl[0], a_min=0, a_max=1)

                # applying control actions in SUMO via helper functions
                set_vsl("L1b", v_ctrl_last[0]/3.6) # convert VSL from km/h to m/s
                set_vsl("L1c", v_ctrl_last[1]/3.6)
                update_ramp_signal_control_logic(r_last.__float__(), cycle_duration, traffic_light)
        
        results_sumo['Ramp_Metering_Rate'][:, k_sumo] = np.asarray(r_last).flatten()
        results_sumo['VSL'][:, k_sumo] = np.asarray(v_ctrl_last).flatten()
            
    traci.simulationStep()

traci.close()

# Total Time Spent metrics
T = 10.0 / 3600
L = 1
lanes = 2
tts = T * np.sum(np.sum(results_sumo['Density_perLane'], axis=0) * L * lanes + np.sum(results_sumo['Queue_Lengths'], axis=0))
vkt = T * np.sum(np.sum(results_sumo['Flow'], axis=0) * L)
print(f"TTS = {tts:.3f} veh.h, VKT = {vkt:.3f} veh.km, Avg Speed = {vkt/tts:.3f} km/h")

##################################################################################
# Plotting
##################################################################################
plt.figure()
plt.plot(results_sumo['Time'], results_sumo['Density_perLane'].T)
plt.legend(mainline_edges)
plt.xlabel('Time (s)')
plt.ylabel('Density (veh/km/lane)')

plt.figure()
plt.plot(results_sumo['Time'], results_sumo['Speed'].T)
plt.legend(mainline_edges)
plt.xlabel('Time (s)')
plt.ylabel('Speed (km/h)')

plt.figure()
plt.plot(results_sumo['Time'], results_sumo['Queue_Lengths'].T)
plt.legend([queue_edge_main, onramp_edges[0]])
plt.xlabel('Time (s)')
plt.ylabel('Queue Length (veh)')

X, Y = np.meshgrid(results_sumo['Time'], (np.arange(len(mainline_edges)) + 1) * 1000)
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
norm = Normalize(vmin=0, vmax=FREE_FLOW_SPEED)
sc = axs[0].pcolormesh(X, Y, results_sumo['Speed'], shading='auto', cmap='jet_r', norm=norm)
plt.colorbar(sc, label='Speed (km/h)', ax=axs[0])
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Position (km)')

sc = axs[1].pcolormesh(X, Y, results_sumo['Density_perLane'], shading='auto', cmap='jet')
plt.colorbar(sc, label='Density (veh/km/lane)', ax=axs[1])
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Position (km)')

sc = axs[2].pcolormesh(X, Y, results_sumo['Flow'], shading='auto', cmap='jet_r')
plt.colorbar(sc, label='Flow (veh/h)', ax=axs[2])
axs[2].set_xlabel('Time (s)')
axs[2].set_ylabel('Position (km)')

fig.suptitle('Time-Space Diagrams')
fig.tight_layout()

if RUN_MODE == "MPC":
    plt.figure()
    plt.plot(results_sumo['Time'], results_sumo['Ramp_Metering_Rate'].T)
    plt.xlabel('Time (s)')
    plt.ylabel('Ramp Metering Rate')

    plt.figure()
    plt.plot(results_sumo['Time'], results_sumo['VSL'].T)
    plt.xlabel('Time (s)')
    plt.ylabel('VSL (m/s)')

if RUN_MODE == "MPC_DRL":
    plt.figure()
    plt.plot(results_sumo['Time'], results_sumo['Ramp_Metering_Rate_MPC'].T, label="MPC")
    plt.plot(results_sumo['Time'], results_sumo['Ramp_Metering_Rate'].T, label="MPC+DRL")
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Ramp Metering Rate')

    plt.figure()
    plt.plot(results_sumo['Time'], results_sumo['VSL_MPC'].T, label="MPC")
    plt.plot(results_sumo['Time'], results_sumo['VSL'].T, label="MPC+DRL")
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('VSL (m/s)')

plt.show()