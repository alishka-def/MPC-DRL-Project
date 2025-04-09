# Changes you should make to the network:
Traffic light "N2" should not be placed at the junction connecting the on-ramp and the mainline branches. This will cause the mainline to have a red light at some points, and we do not want that.
Instead, add a junction 50 meters before this junction and put a traffic light on the on-ramp edge "O2" only (see example image below). This ensures the traffic light only controls the on-ramp leg without affecting the mainline and you can have only 2 phases "G" and "r". 
You will need to change the variable traffic_light to properly ID this new traffic light.

![image](https://github.com/user-attachments/assets/bfe846a0-1d36-4f8f-85d6-0c14677ce797)


# Main Changes already done to the code:
1. simple_run.py is the open-loop (i.e. no control case). Just make sure that your ramp metering rate is 1.0 and is properly running.
2. mpc_run.py is the MPC controlled case.
3. get_edge_density(edge_ids) now returns both densities in veh/km and lane_densities in veh/km/lane
4. get_edge_speed(edge_ids) now returns the free-flow speed (km/h) when the vehicle count on edge is zero.
5. Note the different time steps used by SUMO (1 second), METANET (10 seconds), and MPC (60 seconds). We handle that effect in the MPC feedback measurements by averaging the last 10-second window of measurements.
6. set_vsl("L1a", v_ctrl_last[0]*3.6), factor 3.6 was added to convert VSL from km/h to m/s
