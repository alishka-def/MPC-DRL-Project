# Step 1: Add modules to provide access to specific libraries and functions
import os  # Module provides functions to handle file paths, directories, environment variables
import sys  # Module provides access to Python-specific system parameters and functions

# Step 2: Establish path to SUMO (SUMO_HOME)
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# Step 3: Add Traci module to provide access to specific libraries and functions
import traci  # Module for controlling SUMO simulations via TraCI

# Step 4: Define SUMO configuration
Sumo_config = [
    'sumo-gui',
    '-c', 'Traci.sumocfg',
    '--step-length', '0.05',
    '--delay', '1000',
    '--lateral-resolution', '0.1'
]

# Step 5: Open connection between SUMO and Traci
traci.start(Sumo_config)

# Step 6: Define Variables
vehicle_speed = 0
total_speed = 0


# Step 7: Define Functions
def process_vehicles():
    # vehicles_list = traci.vehicle.getIDList()
    if 'veh1' in traci.vehicle.getIDList():
        traci.vehicle.moveToXY('veh1', 'E0.62', '0', 13, -10, 0, keepRoute=2)
        position = traci.vehicle.getPosition('veh1')  # Get the (x, y) position of the vehicle
        angle = traci.vehicle.getAngle('veh1')
        print(f"Vehicle ID: {'veh1'}, Position: {position}, Angle: {angle}")


# Step 8: Take simulation steps until there are no more vehicles in the network
while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()  # Move simulation forward 1 step
    # Here you can decide what to do with simulation data at each step
    process_vehicles()

# Step 9: Close connection between SUMO and Traci
traci.close()
