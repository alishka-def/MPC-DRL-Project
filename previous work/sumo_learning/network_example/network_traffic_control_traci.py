import os
import sys
import matplotlib.pyplot as plt
import casadi as cs
import numpy as np
from csnlp import Nlp
from csnlp.wrappers import Mpc

# Setting up SUMO environment
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci

# Defining Sumo configuration
Sumo_config = [
    'sumo-gui',
    '-c', 'network_example.sumocfg',
    '--lateral-resolution', '0.1'
]

# Starting SUMO
traci.start(Sumo_config)

# Defining lane IDs
mainline_lanes = ["L1c_0", "L1c_1"]
onramp_lane = "O2_0"
all_lanes = mainline_lanes + [onramp_lane]
default_speed = 28.33 # m/s --> 102 km/h

# Initializing data containers
time_steps = []
traffic_data = {
    lane: {
        "flow": [],
        "speed": [],
        "density": []
    } for lane in all_lanes
}

# Defining function for changing traffic lights
# Setting the traffic light to a specific state string e.g. "GrG"
def custom_light_state(tl_id, state_string):
    traci.trafficlight.setRedYellowGreenState(tl_id, state_string)

# Setting max speed on a specific lane
def set_lane_speed(lane_id, speed):
    traci.lane.setMaxSpeed(lane_id,speed)

# Getting traffic states --> maybe get for edge. density has to be per lane.
def get_traffic_state(lane_id):
    vehicle_count = traci.lane.getLastStepVehicleNumber(lane_id)
    speed_mps = traci.lane.getLastStepMeanSpeed(lane_id)
    speed_kmh = speed_mps * 3.6
    lane_length_m = traci.lane.getLength(lane_id)
    density = vehicle_count/ (lane_length_m/ 1000.0) # veh/km
    flow = speed_mps * density * 3.6
    return flow, speed_kmh, density


# Taking simulation steps until there are no more vehicles expected

# the loop runs the simulation as long as there are vehicles expected to be inserted or still on the road
while traci.simulation.getMinExpectedNumber() > 0 and traci.simulation.getTime() < 9000:
    traci.simulationStep() # simulation step is 0.1 seconds

    # Getting the current simulation time
    sim_time = traci.simulation.getTime()
    time_steps.append(sim_time)

    # Changing traffic light N2 states over time
    if 10 <= sim_time < 20:
        custom_light_state("N2", "GrG")
    elif 20 <= sim_time < 30:
        custom_light_state("N2","yry")
    elif 30 <= sim_time < 40:
        custom_light_state("N2","rGr")
    elif 40 <= sim_time < 50:
        custom_light_state("N2","ryr")
    else:
        pass

    # Variable speed limit (VSL) control

    if 0 <= sim_time < 900:
        set_lane_speed("L1c_0",33.33) # Lane 0 -> 120 km/h
        set_lane_speed("L1c_1",33.33)

    elif 900 <= sim_time < 1800:
        set_lane_speed("L1c_0",25.0) # Lane 0 is slower
        set_lane_speed("L1c_1",33.33) # Lane 1 is the same
    elif 1800 <= sim_time < 2700:
        set_lane_speed("L1c_0",20.0)
        set_lane_speed("L1c_1",25.0)
    else:
        set_lane_speed("L1c_0",default_speed)
        set_lane_speed("L1c_1",default_speed)

    # Traffic state extraction
    print(f"[{sim_time:.1f}s] Traffic States per Lane:")

    for lane in all_lanes:
        flow, speed, density = get_traffic_state(lane)
        traffic_data[lane]["flow"].append(flow)
        traffic_data[lane]["speed"].append(speed)
        traffic_data[lane]["density"].append(density)

        print(f"  {lane}: Flow = {flow:.2f} veh/h | Speed = {speed:.2f} km/h | Density = {density:.2f} veh/km")


traci.close()

# Plotting results
fig, axes = plt.subplots(3,1,figsize=(12,10), sharex=True)

# Plotting Flow
for lane in all_lanes:
    axes[0].plot(time_steps, traffic_data[lane]["flow"], label=lane)
axes[0].set_ylabel("Flow [veh/h]")
axes[0].set_title("Traffic Flow over Time")
axes[0].legend()
axes[0].grid(True)

# Speed plot
for lane in all_lanes:
    axes[1].plot(time_steps, traffic_data[lane]["speed"], label=lane)
axes[1].set_ylabel("Speed [km/h]")
axes[1].set_title("Traffic Speed over Time")
axes[1].legend()
axes[1].grid(True)

# Density plot
for lane in all_lanes:
    axes[2].plot(time_steps, traffic_data[lane]["density"], label=lane)
axes[2].set_ylabel("Density [veh/km]")
axes[2].set_xlabel("Time [s]")
axes[2].set_title("Traffic Density over Time")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.show()