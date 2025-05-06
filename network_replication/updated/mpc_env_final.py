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

# Generating demands (taken from metanet_control.py)
def create_demands(time: np.ndarray) -> np.ndarray:
    return np.stack(
        (
            np.interp(time, (2.0, 2.25), (3500, 1000)),
            np.interp(time, (0.0, 0.15, 0.35, 0.5), (500, 1500, 1500, 500)),
        )
    )

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
            time: np.ndarray,
            n_seg: int = 6,
            n_orig: int = 2,
            n_ramp: int = 1,
            n_vsl: int = 2,
            v_free: float = 102.0,
            v_min: float = 20.0,
            rho_max: float = 180.0,
            L: float = 1.0,
            lanes: int = 2,
            mpc = None,
            drl_ratio: int = 6,
            w_u: float = 0.4,
            w_p: float = 10.0,
            horizon_hours: float = 2.5,
    ):
        super().__init__()
        self.dynamics = dynamics
        self.time = time
        self.n_seg = n_seg
        self.n_orig = n_orig
        self.n_ramp = n_ramp
        self.n_vsl = n_vsl
        self.v_free = v_free
        self.v_min = v_min
        self.rho_max = rho_max
        self.L = L
        self.lanes = lanes
        self.mpc = mpc
        self.drl_ratio = drl_ratio
        self.w_u = w_u
        self.w_p = w_p
        # normalization constants
        self.max_queue = np.array([200.0, 100.0], dtype = np.float32)
        self.max_demands = np.array([3500.0, 1500.0], dtype = np.float32)
        # generating raw demands and set horizon
        self.demands_raw = create_demands(self.time).T.astype(np.float32)
        # computing number of drl steps
        T = (self.time[1] - self.time[0]) * 3600.0  # small step in seconds (10 s)
        self.horizon_steps = int(horizon_hours * 3600.0 / (self.drl_ratio * T))
        self.current_step = 0

        # Action space (equation 8)
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

        # Observation space (equation 7)
        # list of six blocks concentrated into one vector
        # rho(density), v(speed), w(queue length), u_s(MPC output), u_c_prev(previous control action), d(demands)
        # all the values are normalized [0,1]
        obs_dim = (
            2 * self.n_seg # densities + speeds
            + self.n_orig # queues
            + (self.n_ramp + self.n_vsl) # MPC output
            + self.n_orig # demands
            + (self.n_ramp + self.n_vsl) # previous control action
        )

        self.observation_space = spaces.Box(
            low = 0.0,
            high = 1.0,
            shape = (obs_dim,),
            dtype = np.float32,
        )

        # Initializing the state at t=0
        rho0 = np.array([22, 22, 22.5, 24, 30, 32], dtype=np.float32)
        v0 = np.array([80, 80, 78, 72.5, 66, 62], dtype = np.float32)
        w0 = np.zeros(self.n_orig, dtype = np.float32)

        # MPC outputs: ramp=1, VSL=v_free
        u_s0 = np.concatenate([
            np.ones(self.n_ramp, dtype= np.float32),
            np.full(self.n_vsl, self.v_free, dtype = np.float32)
        ])

        # Previous combined DRL+MPC control: start equal to MPC
        u_prev0 = u_s0.copy()

        # Storing non-normalized versions for simulation
        self.rho_raw = rho0
        self.v_raw = v0
        self.w_raw = w0
        self.u_s = u_s0
        self.u_prev = u_prev0

        # Computing initial normalized observations
        self.state = self.normalize_observations(
            rho0,
            v0,
            w0,
            u_s0,
            u_prev0,
            self.demands_raw[0],
        )

    def normalize_observations(
            self,
            rho: np.ndarray,
            v: np.ndarray,
            w: np.ndarray,
            u_mpc: np.ndarray,
            u_prev: np.ndarray,
            demands: np.ndarray,
    ) -> np.ndarray:
        """
        Normalize all components into [0,1]:
        rho_norm = rho/ rho_max
        v_norm = v/ v_free
        w_norm = w/ max_queue
        u_mpc_norm = [ramp_rate, vsl/ v_free]
        u_prev_norm = [prev_ramo, prev_vsl/ v_free]
        d_norm = demands/ max_demands
        Returns concentrated observation vector.
        """
        rho_norm = rho / self.rho_max
        v_norm = v/ self.v_free
        w_norm = w/ self.max_queue

        ramp_u, vsl_u = np.split(u_mpc, [self.n_ramp])
        ramp_norm = ramp_u
        # vsl normalization puts the minimum vsl at about 0.2 (v_min/v_free) rather than 0
        vsl_norm  = vsl_u / self.v_free
        u_mpc_norm = np.concatenate([ramp_norm, vsl_norm])

        ramp_prev, vsl_prev = np.split(u_prev, [self.n_ramp])
        ramp_prev_norm = ramp_prev
        vsl_prev_norm = vsl_prev/ self.v_free
        u_prev_norm = np.concatenate([ramp_prev_norm, vsl_prev_norm])

        d_norm = demands/ self.max_demands

        return np.concatenate([
            rho_norm,
            v_norm,
            w_norm,
            u_mpc_norm,
            d_norm,
            u_prev_norm,
        ])

    # reset function -> resets the environment to initial conditions, computes initial MPC baseline, and returns initial observations
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # Initializing conditions (same as in metanet)
        rho0 = np.array([22, 22, 22.5, 24, 30, 32], dtype=np.float32)
        v0 = np.array([80, 80, 78, 72.5, 66, 62], dtype=np.float32)
        w0 = np.zeros(self.n_orig, dtype=np.float32)

        # copying the states - to prevent possible bugs
        self.rho_raw = rho0.copy()
        self.v_raw = v0.copy()
        self.w_raw = w0.copy()

        # Previous control: ramp = 1, vsl = v_free
        u_prev0 = np.concatenate([
            np.ones(self.n_ramp, dtype=np.float32),
            np.full(self.n_vsl, self.v_free, dtype=np.float32),
        ])
        self.u_prev = u_prev0.copy()

        # MPC solver
        pars = {
            "rho_0": cs.DM(self.rho_raw),
            "v_0": cs.DM(self.v_raw),
            "w_0": cs.DM(self.w_raw),
            "d": cs.DM(self.demands_raw[: self.drl_ratio].T),
            "r_last": cs.DM(self.u_prev[: self.n_ramp].reshape(-1, 1)),
            "v_ctrl_last": cs.DM(self.u_prev[self.n_ramp:].reshape(-1, 1)),
        }
        sol = self.mpc.solve(pars=pars)
        r_act = np.array(sol.vals["r"].full()).flatten()
        v_act = np.array(sol.vals["v_ctrl"].full()).flatten()
        self.u_s = np.concatenate([r_act, v_act]).astype(np.float32)

        # Building and returning normalized observations
        obs = self.normalize_observations(
            self.rho_raw,
            self.v_raw,
            self.w_raw,
            self.u_s,
            self.u_prev,
            self.demands_raw[0],
        )
        self.state = obs
        return obs, {}

    """
    Step function -> T=10 sDRL = 6 steps (6*10=60s). MPC covers 30 steps
    (30*10=300s).
    1. Denormalize current MPC baseline u_s
    2. Combine with DRL tweak. saturate to [0≤r≤1, v_min≤vsl≤v_free]
    3. Roll out self.drl_steps through the dynamics function
    4. Accumulate reward
    5. Return normalized observation, reward, done, truncated, info
    """

    def step(self,action):
        # multiplying by [1, v_free] to denormalize the MPC baseline (u_s)
        denorm = np.concatenate([
            np.ones(self.n_ramp, dtype=np.float32),
            np.full(self.n_vsl, self.v_free, dtype=np.float32),
        ])
        u_s_phys = self.u_s * denorm

        # saturation function - equations 2,3
        u_tweak = action.astype(np.float32)
        # adding physical bounds for ramp and vsl
        u_min = np.concatenate([
            np.zeros(self.n_ramp, dtype=np.float32),
            np.full(self.n_vsl, self.v_min, dtype=np.float32),
        ])
        u_max = np.concatenate([
            np.ones(self.n_ramp, dtype=np.float32),
            np.full(self.n_vsl, self.v_free, dtype=np.float32),
        ])
        # ensures that combined control action never violates physical constraints
        u_combined = saturate(u_s_phys + u_tweak, u_min, u_max)

        # simulating drl steps via casadi function
        total_reward = 0.0
        # this loop runs 6 times
        for j in range(self.drl_ratio):
            # self.current_step is the count of drl actions I took already
            idx = self.current_step*self.drl_ratio + j
            d_k = self.demands_raw[min(idx, len(self.demands_raw)-1)]
            # building the state vector x=[p;v;w] for dynamics function
            x = cs.vertcat(
                cs.DM(self.rho_raw),
                cs.DM(self.v_raw),
                cs.DM(self.w_raw),
            )
            # reordering u as expected by F
            u_dyn = np.concatenate([
                u_combined[self.n_ramp:],
                u_combined[:self.n_ramp],
            ])
            # step one T = 10 s
            x_next, _ = self.dynamics(x, cs.DM(u_dyn), cs.DM(d_k))
            arr = np.array(x_next.full()).flatten()
            # unpack next raw state
            self.rho_raw = arr[: self.n_seg]
            self.v_raw = arr[self.n_seg: 2 * self.n_seg]
            self.w_raw = arr[2 * self.n_seg: 2 * self.n_seg + self.n_orig]
            # computing TTS+queue penalty
            J  = (self.rho_raw * self.L * self.lanes).sum() + self.w_raw.sum()
            Ps = max(0.0, np.max(self.w_raw) - self.max_queue[1])
            total_reward -= (J + self.w_p * Ps)

        # advance DRL step counter - recording 1 drl action done
        self.current_step +=1
        # stores the last combined control so the next mpc solve can use it as "previous"
        self.u_prev = u_combined.copy()

        # MPC updates every 5 drl actions (5*60 s = 300 s)
        if self.current_step % (30 // self.drl_ratio) == 0:
            start = (self.current_step - 1) * self.drl_ratio
            end = start + self.drl_ratio
            d_slice = (self.demands_raw[start:end].T
                       if end <= len(self.demands_raw)
                        else self.demands_raw[start:].T)

            sol = self.mpc.solve({
                "rho_0": cs.DM(self.rho_raw),
                "v_0": cs.DM(self.v_raw),
                "w_0": cs.DM(self.w_raw),
                "d": cs.DM(d_slice),
                "r_last": cs.DM(self.u_prev[:self.n_ramp].reshape(-1, 1)),
                "v_ctrl_last": cs.DM(self.u_prev[self.n_ramp:].reshape(-1, 1)),
            })

            r_act = np.array(sol.vals["r"].full()).flatten()
            v_act = np.array(sol.vals["v_ctrl"].full()).flatten()
            # creating a new baseline that the drl will tweak next time
            self.u_s = np.concatenate([r_act, v_act]).astype(np.float32)

            # returning next normalized observations
            next_idx = min(self.current_step * self.drl_ratio, len(self.demands_raw) - 1)
            next_d = self.demands_raw[next_idx]
            obs = self.normalize_observations(
                self.rho_raw,
                self.v_raw,
                self.w_raw,
                self.u_s,
                self.u_prev,
                next_d
            )

            self.state = obs
            done = False
            # once we hit 150 steps, the simulation will end)
            truncated = (self.current_step >= self.horizon_steps)
            return obs, float(total_reward), done, truncated, {}















