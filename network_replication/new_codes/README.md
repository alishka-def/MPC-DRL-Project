# metanet_run.py
This file runs the METANET model for the 3 cases of no control, MPC control, and MPC+DRL control. The model uploaded here for DRL is maybe not fully trained. Please check that.

# sumo_run.py
This file runs the SUMO network for the 3 cases of no control, MPC control, and MPC+DRL control. The MPC+DRL control case is now implemented as well.

## SUMO visualization tools (commands)
```
python C:/Users/selbaklish/Desktop/SUMO/sumo-1.19.0/tools/visualization/plotXMLAttributes.py Logs/log_summary.xml -x time -y meanTravelTime,meanWaitingTime --legend
```
```
python C:/Users/selbaklish/Desktop/SUMO/sumo-1.19.0/tools/visualization/plotXMLAttributes.py Logs/log_summary.xml -x time -y meanSpeed --legend
```
```
python C:/Users/selbaklish/Desktop/SUMO/sumo-1.19.0/tools/visualization/plotXMLAttributes.py Logs/log_edges.xml -x begin -y occupancy --legend
```
