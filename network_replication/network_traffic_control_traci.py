import os
import sys

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
    '--step-length', '0.05',
    '--delay', '1000',
    '--lateral-resolution', '0.1'
]

# Starting SUMO
traci.start(Sumo_config)

# Defining variables for VSL
mainline_lanes = ["L1c_0", "L1c_1"]
default_speed = 28.33 # m/s --> 102 km/h

# Defining function for changing traffic lights
# Setting the traffic light to a specific state string e.g. "GGGGrrrrGGGGrrrr"

def custom_light_state(tl_id, state_string):
    traci.trafficlight.setRedYellowGreenState(tl_id, state_string)

# Setting max speed on a specific lane
def set_lane_speed(lane_id, speed):
    traci.lane.setMaxSpeed(lane_id,speed)

# Getting the current state of the traffic light
print(traci.trafficlight.getRedYellowGreenState("N2"))
# Taking simulation steps until there are no more vehicles expected

# the loop runs the simulation as long as there are vehicles expected to be inserted or still on the road
while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep() # simulation step is 0.1 seconds

    # Getting the current simulation time
    sim_time = traci.simulation.getTime()

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

        # Checking whether the VSL applied
        print(f"[{sim_time:.1f}s] Speed limits applied:")
        print(f"  L1c_0 → {traci.lane.getMaxSpeed('L1c_0'):.2f} m/s")
        print(f"  L1c_1 → {traci.lane.getMaxSpeed('L1c_1'):.2f} m/s")

    elif 900 <= sim_time < 1800:
        set_lane_speed("L1c_0",25.0) # Lane 0 is slower
        set_lane_speed("L1c_1",33.33) # Lane 1 is the same
    elif 1800 <= sim_time < 2700:
        set_lane_speed("L1c_0",20.0)
        set_lane_speed("L1c_1",25.0)
    else:
        set_lane_speed("L1c_0",default_speed)
        set_lane_speed("L1c_1",default_speed)



traci.close()
