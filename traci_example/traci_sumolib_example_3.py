#Step 1: Add modules to provide access to specific libraries and functions
import os  # Provides functions to handle file paths, directories, environment variables
import sys  # Provides access to Python-specific system parameters and functions

#Step 2: establish path to SUMO (SUMO_HOME)
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

#Step 3: Add Traci or Sumolib module to provide access to specific libraries and functions
import sumolib

#Step 4: Define Sumo configuration
#IT DOES NOT HAVE

#Step 5:  Open connection between SUMO and SUMOLib
net = sumolib.net.readNet('Traci.net.xml')  # Load network file for analysis

#Step 6: Define Variables
speedsum = 0
edgecount = 0
avgSpeed = 0

#Step 7: Define Functions
for edge in net.getEdges():
    speedsum = edge.getSpeed()  # Sum up the speeds of all edges
    edgecount += 1  # Count the number of edges

if edgecount > 0:
    avgSpeed = speedsum / edgecount  # Calculate average speed
    print(f"Average speed in all edges: {avgSpeed}")
    print(speedsum)
    print(edgecount)

#Step 8: Take simulation steps until there are no more vehicles in the network
#IT DOES NOT HAVE

#Step 9: close connection between sumo and traci
#IT DOES NOT HAVE