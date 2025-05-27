import os
import sys
import traci

import numpy as np
import pandas_read_xml as pdx
import matplotlib.pyplot as plt
import seaborn as sns


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
    'sumo-gui',
    '-c', 'network_example.sumocfg',
    '--lateral-resolution', '0.1'
]

traffic_light = "J1"

##################################################################################
# FUNCTIONS
##################################################################################
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

##################################################################################
# Start Simulation
##################################################################################
#"""
traci.start(Sumo_config)
update_ramp_signal_control_logic(metering_rate=1.0, cycle_duration=60, ramp=traffic_light)
step = 0
while step <= 9000:
    traci.simulationStep()
    step += 1

traci.close()
#"""

##################################################################################
# Plotting
##################################################################################
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