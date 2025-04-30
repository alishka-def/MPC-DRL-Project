import gymnasium as gym
from gymnasium import spaces
import numpy as np
import casadi as cs

"""
Creating a custom Gymnasium environment for metanet/mpc control example for toy network.
Here, the mpc control action for the ramp metering and vsl is tweaked by high-frequency output action
of DRL agent (in this example, DDPG agent is used). 

Observations: normalized values of 1) states (density, speed, queue), 2) mpc output, 3) previous combined control
action (mpc+drl), 4) demands. 

Actions: ramp metering rate and vsl.

Rewards: negative total time spent per step + include penalties function. 
Adding penalty if combined action of mpc and drl exceeds the limit. 
"""

# Saturation function (equation 3)
# Purpose: making sure the combined control never exceeds the physical bounds of ramp-meter or VSL
def saturate(control_signal: np.ndarray,
             min_bounds: np.ndarray,
             max_bounds: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(control_signal, min_bounds), max_bounds)

class MetanetEnv(gym.Env):
    def __init__(
            self,
            dynamics: cs.Function,
            demands: np.array,
            horizon_steps: int,
            v_free: float = 102.0,
            v_min: float = 20.0,
            l: float = 1.0,
            lanes: int = 2,







    ):