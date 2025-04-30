import gymnasium as gym
from gymnasium import spaces
import numpy as np
import casadi as cs
"""
    Creating gymnasium environment for coordinated ramp metering
    and VSL control.

    Observations: states --> densities, speeds, on-ramp queues, mpc output,
    prev control actions, demands

    Actions: ramp metering rate [0,1] and
    variable speed limit [v_min, v_free] —> v_free is 102 km/h.

    Terminations: truncated when simulation time exceeds horizon_step (2.5 hours)

    Reset function: state variables as initial fixed conditions (for easier replication)

    Reward: negative total time spent per step + include penalties function
    Mpc ramp-metering and drl ramp-metering —> add penalty if the sum >1 —> same for vsl
    
    This environment wraps:
    - A high-level MPC controller as a baseline
    A low-level DRL tweak (higher frequency) based on DDPG
"""

# saturation function eq.3 --> making sure the combined control never exceeds
# the physical bounds of ramp-meter or VSL --> making sure that each control element
# is between low and high. u is an array of raw control values.

# equation 3
def saturate(control_signal: np.ndarray,
             min_bounds: np.ndarray,
             max_bounds: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(control_signal, min_bounds), max_bounds)

class MetanetEnv(gym.Env):
    """
    Custom Gymnasium environment wrapping hierarchical MPC-DRL
    for coordinated ramp metering (RM) and variable speed limits (VSL).

    - High-level MPC baseline at low frequency (Tc)
    - Low-level DRL tweak at high frequency (Td)

    Observations x_rl = [rho, v, w, u_s, d, u_c_prev]  (Eq. 7)
    Actions    u_rl: tweak to u_s, bounded by w_u * (u_max - u_min) (Eq. 8)
    Reward     negative total time spent + queue violation penalty (Eq. 9)
    Termination when current_step >= horizon_steps
    """
    metadata = {"render_modes": ["human"]}
    def __init__(
            self,
            dynamics: cs.Function,  # CasADi function F(x, u, d) -> x_next
            demands: np.ndarray,    # demand time series shape (Tsim, n_orig)
            horizon_steps: int,     # max DRL control steps before truncation
            n_seg: int,             # number of freeway segments
            n_orig: int,            # number of origins (on-ramps)
            n_ramp: int,            # number of ramp controllers
            n_vsl: int,             # number of VSL segments
            v_free: float,          # free-flow speed (km/h)
            v_min: float = 20.0,    # minimum VSL (km/h)
            L: float = 1.0,         # segment length (km)
            lanes: int = 2,         # lanes per segment
            mpc=None,               # high-level MPC controller instance
            drl_ratio: int = 6,     # m2: DRL steps per MPC step (Eq. 1)
            w_u: float = 0.4,       # DRL tweak scale (Eq. 8)
            w_p: float = 10.0,      # queue penalty weight in reward (Eq. 9)
    ):
        super().__init__() # initializing base class
        # saving parameters for use in reset/step functions
        self.dynamics = dynamics
        self.demands = demands
        self.horizon_steps = horizon_steps
        self.n_seg = n_seg
        self.n_orig = n_orig
        self.n_ramp = n_ramp
        self.n_vsl = n_vsl
        self.v_free = v_free
        self.v_min = v_min
        self.L = L
        self.lanes = lanes
        self.mpc = mpc
        self.drl_ratio = drl_ratio
        self.w_u = w_u
        self.w_p = w_p

        # action space - equation 8
        # delta_U = [1]*n_ramp + (v_free - v_min)*[1]*n_vsl
        # action bounds = ± w_u * delta_U
        full_range = np.concatenate([
            np.ones(self.n_ramp),  # ramp delta = 1
            (self.v_free - self.v_min) * np.ones(self.n_vsl)  # VSL delta
        ])
        low = -self.w_u * full_range
        high = +self.w_u * full_range
        self.action_space = spaces.Box(
            low=low.astype(np.float32),
            high=high.astype(np.float32),
            dtype=np.float32
        )

        # observation space - equation 7
        # list of six blocks concatenated into one vector
        # rho(density), v(speed), w(queue length), u_s(MPC output), u_c_prev(previous control action), d(demands)
        obs_dim = (
                2 * self.n_seg
                + self.n_orig
                + (self.n_ramp + self.n_vsl)
                + self.n_orig
                + (self.n_ramp + self.n_vsl)
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32
        )

        # initializing the state
        self.current_step = 0
        self.rho = np.zeros(self.n_seg, dtype=np.float32)
        self.v = np.full(self.n_seg, self.v_free, dtype=np.float32)
        self.w = np.zeros(self.n_orig, dtype=np.float32)
        self.u_s = np.zeros(self.n_ramp + self.n_vsl, dtype=np.float32)
        self.u_c_prev = np.zeros(self.n_ramp + self.n_vsl, dtype=np.float32)

    # reset function -> resets the environment to initial conditions, computes initial MPC baseline, and returns initial observations
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step=0

        # Empty network
        self.rho=np.zeros(self.n_seg, dtype=np.float32)
        self.v = np.full(self.n_seg, self.v_free, dtype=np.float32)
        self.w = np.zeros(self.n_orig, dtype=np.float32)

        # Initial MPC solve to get the baseline control at t=0
        sol = self.mpc.solve({
            "rho_0": cs.DM(self.rho),
            "v_0": cs.DM(self.v),
            "w_0": cs.DM(self.w),
            "d": cs.DM(self.demands[:self.drl_ratio].T),
            "v_ctrl_last": cs.DM.zeros(self.n_vsl,1),
            "r_last": cs.DM.ones(self.n_ramp,1),

        })

        r = np.array(sol.vals["r"].flatten())
        v_ctrl = np.array(sol.vals["v_ctrl"].flatten())
        self.u_s = np.hstack([r, v_ctrl]).flatten()
        self.u_c_prev = self.u_s.copy() # TODO: 1 for ramp metering rate + v_free for vsl

        # building initial observations
        d0 = self.demands[0].astype(np.float32)
        obs = np.concatenate([
            self.rho,
            self.v,
            self.w,
            self.u_s,
            d0,
            self.u_c_prev,
        ])

        return obs, {}

    def step(self, action):
        """
        1. Combining MPC baseline with DRL tweak action
        2. Saturating to physical bounds
        3. Simulating drl_ratio steps via Casadi function
        4. Accumulate reward
        5. update MPC baseline every drl_ratio steps
        6. Return next obs, rewards, done, truncated, info
        """
        u_tweak = action.astype(np.float32)
        u_min = np.hstack([np.zeros(self.n_ramp), self.v_min * np.ones(self.n_vsl)])
        u_max = np.hstack([np.ones(self.n_ramp), self.v_free * np.ones(self.n_vsl)])
        u_combined = saturate(self.u_s + u_tweak, u_min, u_max)

        total_reward = 0.0
        for j in range(self.drl_ratio):
            idx = self.current_step * self.drl_ratio + j
            d_k = self.demands[min(idx, len(self.demands) - 1)]
            x = cs.vertcat(cs.DM(self.rho), cs.DM(self.v), cs.DM(self.w))
            u = cs.vertcat(cs.DM(u_combined[self.n_ramp:]), cs.DM(u_combined[:self.n_ramp]))
            x_next, _ = self.dynamics(x, u, cs.DM(d_k))
            arr = np.array(x_next).flatten()
            self.rho = arr[:self.n_seg]
            self.v = arr[self.n_seg:2 * self.n_seg]
            self.w = arr[2 * self.n_seg:2 * self.n_seg + self.n_orig]
            J = np.sum(self.rho * self.L * self.lanes) + np.sum(self.w)
            Ps = max(0.0, np.max(self.w) - 100.0)
            total_reward -= (J + self.w_p * Ps)

        self.current_step += 1
        if self.current_step % self.drl_ratio == 0:
            start = (self.current_step - 1) * self.drl_ratio
            end = start + self.drl_ratio
            d_slice = (self.demands[start:end].T if end <= len(self.demands) else self.demands[start:].T)
            sol = self.mpc.solve({
                "rho_0": cs.DM(self.rho),
                "v_0": cs.DM(self.v),
                "w_0": cs.DM(self.w),
                "d": cs.DM(d_slice),
                "v_ctrl_last": cs.DM(self.u_c_prev[self.n_ramp:]),
                "r_last": cs.DM(self.u_c_prev[:self.n_ramp])
            })
            r = np.array(sol.vals["r"]).flatten()
            v_ctrl = np.array(sol.vals["v_ctrl"]).flatten()
            self.u_s = np.hstack([r, v_ctrl]).astype(np.float32)

        self.u_c_prev = u_combined.copy()
        d_next = self.demands[min(self.current_step * self.drl_ratio, len(self.demands) - 1)].astype(np.float32)
        obs = np.concatenate([
            self.rho, self.v, self.w,
            self.u_s, d_next, self.u_c_prev
        ])
        done = False
        truncated = (self.current_step >= self.horizon_steps)
        return obs, float(total_reward), done, truncated, {}

    def close(self):
        pass




