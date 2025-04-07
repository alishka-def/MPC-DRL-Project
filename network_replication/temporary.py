# --------------------------
# create initial conditions for metanet model --> these will be overridden by SUMO simulation
rho = cs.DM([22, 22, 22.5, 24, 30, 32]) # defines the starting density for each road segment
v = cs.DM([80, 80, 78, 72.5, 66, 62]) # setting initial speed for each road segment
w = cs.DM([0, 0]) # initial queue length at the network's origins -> no vehicles are queued at the start of the simulation


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