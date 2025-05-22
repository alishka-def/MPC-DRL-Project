########################################################################
# Imports
########################################################################
import os
import sys

import numpy as np
import gymnasium as gym
import casadi as cs
import sym_metanet as metanet

from csnlp import Nlp
from csnlp.wrappers import Mpc
from gymnasium import spaces


########################################################################
# Methods
########################################################################
# Generate time-varying demands for two origins (mainline and on-ramp)
def create_demands(time: np.ndarray) -> np.ndarray:
    return np.stack(
        (
            np.interp(time, (2.0, 2.25), (3500, 1000)),
            np.interp(time, (0.0, 0.15, 0.35, 0.5), (500, 1500, 1500, 500)),
        )
    )

########################################################################
# Class: MetanetMPCEnv - custom gymnasium environment for traffic control
########################################################################
class MetanetMPCEnv(gym.Env):
    def __init__(self, M_drl: int = 6, w_u: float = 0.4, w_p: float = 10.0):
        super(MetanetMPCEnv, self).__init__()
        self._init_network() # build METANET network model
        self._init_mpc() # set up MPC controller
        self.T_sim = 2.5 # simulation time of 2.5 hours
        self.M_drl = M_drl # number of DRL steps (6*10=60s)
        self.w_u = w_u # weight for the control action
        self.w_p = w_p # penalty weight for queue violations

        # variance for low-level noise [mainline, on-ramp]
        self.noise_var = np.array([75.0, 30.0], dtype=np.float32)
        # standard deviation for low-level noise [mainline, on-ramp]
        self.noise_std = np.sqrt(self.noise_var)

        # normalization constants
        self._max_queue = np.array([200.0, 100.0], dtype = np.float32)
        self._max_demands = np.array([3500.0, 1500.0], dtype = np.float32)
        # lower and upper bounds for the control action
        self._u_lb = np.array([0] + [self._vsl_min for _ in range(self._n_vsl)])
        self._u_ub = np.array([1] + [self._v_free for _ in range(self._n_vsl)])
        # Action space (equation 8)
        full_range = np.concatenate([
            np.ones(self._n_ramp),  # ramp delta = 1
            (self._v_free - self._vsl_min) * np.ones(self._n_vsl)  # range for VSL
        ])
        self.action_space = spaces.Box(
            low=-self.w_u*full_range.astype(np.float32), # minimum action values
            high=+self.w_u*full_range.astype(np.float32), # maximum action values
            dtype=np.float32
        )
        # Observation space (equation 7)
        obs_dim = (
            2 * self._n_seg # densities + speeds
            + self._n_orig # queues
            + (self._n_ramp + self._n_vsl) # MPC output
            + self._n_orig # demands
            + (self._n_ramp + self._n_vsl) # previous control action
        )
        self.observation_space = spaces.Box(
            low = 0.0,
            high = 1.0,
            shape = (obs_dim,),
            dtype = np.float32,
        )
        self.reset()

    def _init_network(self):
        # METANET parameters
        self._T = 10.0 / 3600 # temporal discretization (hours)
        self._L = 1.0 # spatial discretization (km)
        self._num_lanes = 2 # number of lanes
        self._tau = 18.0 / 3600 # reaction time (hours)
        self._kappa = 40.0 # smoothing parameter
        self._eta = 60.0 # prediction horizon parameter
        self._delta = 0.0122 # congestion effect parameter
        self._a = 1.867 # acceleration exponent
        self._rho_max = 180.0 # jam density (veh/km/lane)
        self._rho_crit = 33.5 # critical density (veh/km/lane)
        self._capacity_lane = 2000.0 # capacity per lane (veh/h/lane)
        self._v_free = 102.0 # free flow speed (km/h)
        self._vsl_min = 20.0 # minimum speed limit (km/h)

        # METANET network
        N1 = metanet.Node(name="N1")
        N2 = metanet.Node(name="N2")
        N3 = metanet.Node(name="N3")
        O1 = metanet.MainstreamOrigin[cs.SX](name="O1")
        O2 = metanet.MeteredOnRamp[cs.SX](self._capacity_lane, name="O2")
        D1 = metanet.Destination[cs.SX](name="D1")
        L1 = metanet.LinkWithVsl(
            4, self._num_lanes, self._L, self._rho_max, self._rho_crit, self._v_free, self._a, name="L1",
            segments_with_vsl={2, 3}, alpha=0.1
        ) # num_segments = 4, alpha = Non-compliance factor to the indicated speed limit.
        L2 = metanet.Link[cs.SX](
            2, self._num_lanes, self._L, self._rho_max, self._rho_crit, self._v_free, self._a, name="L2"
        ) # num_segments = 2
        self._net = (
            metanet.Network(name="A1")
            .add_path(origin=O1, path=(N1, L1, N2, L2, N3), destination=D1)
            .add_origin(O2, N2)
        )
        # configurate engine and validate
        metanet.engines.use("casadi", sym_type="SX")
        self._net.is_valid(raises=True)
        # initialize states and compile dynamics function
        self._net.step(T=self._T, tau=self._tau, eta=self._eta, kappa=self._kappa, delta=self._delta, init_conditions={O1: {"v_ctrl": self._v_free*2}})
        self._dynamics = metanet.engine.to_function(
            net=self._net, more_out=True, compact=2, T=self._T
        )
        # number of segments and origins
        self._n_seg, self._n_orig = sum(link.N for _, _, link in self._net.links), len(self._net.origins)
        # number of ramps and VSL control points
        self._n_ramp, self._n_vsl = 1, 2

    def _init_mpc(self):
        # MPC controller
        self._Np, self._Nc, self._M_mpc = 2, 2, 30
        # instantiate MPC wrapper
        self._mpc = Mpc[cs.SX](
            nlp=Nlp[cs.SX](sym_type="SX"),
            prediction_horizon=self._Np*self._M_mpc,
            control_horizon=self._Nc*self._M_mpc,
            input_spacing=self._M_mpc
        )
        # define states: densities, speeds, queue lengths
        rho, _ = self._mpc.state("rho", self._n_seg, lb=0)
        v, _ = self._mpc.state("v", self._n_seg, lb=0)
        w, _ = self._mpc.state("w", self._n_orig, lb=0, ub=[[200], [100]]) # O2 queue is constrained
        # define actions: VSL controls and ramp metering rate
        v_ctrl, _ = self._mpc.action("v_ctrl", self._n_vsl, lb=self._vsl_min, ub=self._v_free)
        r, _ = self._mpc.action("r", lb=0, ub=1)
        # define disturbance (demands)
        d = self._mpc.disturbance("d", self._n_orig)
        # set dynamics constraints
        self._mpc.set_dynamics(self._dynamics)
        # set objective function
        v_ctrl_last = self._mpc.parameter("v_ctrl_last", (self._n_vsl, 1))
        r_last = self._mpc.parameter("r_last", (self._n_ramp, 1))
        self._mpc.minimize(
            self._T * cs.sum2(cs.sum1(rho * self._L * self._num_lanes) + cs.sum1(w)) 
            + 0.4*cs.sumsqr(cs.diff(cs.horzcat(r_last, r)))
            + 0.4*cs.sumsqr(cs.diff(cs.horzcat(v_ctrl_last, v_ctrl), 1, 1) / self._v_free)
        )
        # set solver for MPC NLP
        opts = {
            "expand": True,
            "print_time": False,
            "ipopt": {"max_iter": 500, "sb": "yes", "print_level": 0},
        }
        self._mpc.init_solver(solver="ipopt", opts=opts)
    
    def normalize_observations(self, rho: np.ndarray, v: np.ndarray, w: np.ndarray, u_mpc: np.ndarray, u_prev: np.ndarray, demands: np.ndarray) -> np.ndarray:
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
        rho_norm = rho / self._rho_max
        v_norm = v / self._v_free
        w_norm = w / self._max_queue

        u_mpc_norm = u_mpc / self._u_ub
        u_prev_norm = u_prev / self._u_ub

        d_norm = demands / self._max_demands
        return np.concatenate([rho_norm, v_norm, w_norm, u_mpc_norm, d_norm, u_prev_norm])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # generating raw demands and set horizon
        self.time = np.arange(0, self.T_sim+self._T, self._T)
        # forecast demands (without noise) for mpc
        self.demands_forecast = create_demands(self.time).T.astype(np.float32)
        # actual demands (with noise) = nominal (forecast) + noise
        noise = np.random.normal(loc=0.0, scale=self.noise_std[np.newaxis, :], size=self.demands_forecast.shape).astype(np.float32)
        # clipping the results, so that I wouldn't get a negative demand values
        self.demands_actual = np.clip(self.demands_forecast + noise, a_min=0.0, a_max=None)

        self.current_timestep = 0
        # Initializing conditions (same as in METANET)
        self.rho_raw = np.array([22, 22, 22.5, 24, 30, 32], dtype=np.float32)
        self.v_raw = np.array([80, 80, 78, 72.5, 66, 62], dtype=np.float32)
        self.w_raw = np.zeros(self._n_orig, dtype=np.float32)
        # Previous control: ramp = 1, vsl = v_free
        self.u_prev_raw = np.concatenate([
            np.ones(self._n_ramp, dtype=np.float32),
            np.full(self._n_vsl, self._v_free, dtype=np.float32),
        ])
        self._sol_mpc_prev = None # clear previous MPC solution

        # WARM-UP: 10 minutes without mpc or drl
        u_reordered = cs.vertcat(self.u_prev_raw[self._n_ramp:], self.u_prev_raw[:self._n_ramp])

        # warm-up period of 10 minutes (600 seconds)
        warm_up_steps = int(600/ (self._T*3600)) # self._T is in hours, so _T*3600 = 10 s
        for _ in range(warm_up_steps):
            x_next, _ = self._dynamics(
                cs.vertcat(self.rho_raw, self.v_raw, self.w_raw),
                u_reordered,
                self.demands_actual[self.current_timestep, :]
            )
            self.rho_raw, self.v_raw, self.w_raw = cs.vertsplit(x_next, (
            0, self._n_seg, 2 * self._n_seg, 2 * self._n_seg + self._n_orig))
            self.rho_raw, self.v_raw, self.w_raw = np.array(self.rho_raw).flatten(), np.array(
                self.v_raw).flatten(), np.array(self.w_raw).flatten()


        # TODO: If no issue remove asserts
        assert not np.isnan(self.rho_raw).any(), "NaN in rho_raw"
        assert not np.isnan(self.v_raw).any(), "NaN in v_raw"
        assert not np.isnan(self.w_raw).any(), "NaN in w_raw"
        assert not np.isnan(self.demands_forecast).any(), "NaN in demands_forecast"
        sol = self._mpc.solve(
            pars={"rho_0": self.rho_raw, "v_0": self.v_raw, "w_0": self.w_raw, 
                  "d": self.demands_forecast[:self._Np*self._M_mpc, :].T,
                  "r_last": self.u_prev_raw[: self._n_ramp].reshape(-1, 1), 
                  "v_ctrl_last": self.u_prev_raw[self._n_ramp:].reshape(-1, 1)},
            vals0=self._sol_mpc_prev
        )
        self._sol_mpc_prev = sol.vals # store for next solve
        v_ctrl_last = sol.vals["v_ctrl"][:, 0] # extract VSL solution
        r_last = sol.vals["r"][0] # extract VSL solution
        self.u_mpc_raw = np.concatenate([r_last, v_ctrl_last]).astype(np.float32).flatten()
        # Building and returning normalized observations
        self.state_norm = self.normalize_observations(self.rho_raw, self.v_raw, self.w_raw, self.u_mpc_raw, self.u_prev_raw, self.demands_actual[0])
        # Creating results dict for plotting
        self.sim_results = {
            "Density": [], "Flow": [], "Speed": [], "Queue_Length": [], "Origin_Flow": [], 
            "Ramp_Metering_Rate": [], "VSL": [],
            "u_MPC": [], "u_DRL": [],
        }
        return self.state_norm, {}

    def step(self, action):
        """
        Step function -> T=10 s, DRL = 6 steps (6*10=60s). MPC covers 30 steps (30*10=300s).
            1. Combine MPC baseline with DRL tweak. saturate to [0≤r≤1, v_min≤vsl≤v_free].
            2. Roll out self.M_drl steps through the dynamics function
            3. Accumulate reward
            4. Return normalized observation, reward, done, truncated, info
        """
        # combine baseline MPC control with DRL action and saturate
        u_combined = self.saturate(self.u_mpc_raw + action)
        # reorder to match dynamics input
        u_reordered = cs.vertcat(u_combined[self._n_ramp:], u_combined[:self._n_ramp])
        reward = 0.0
        for _ in range(self.M_drl):
            x_next, q_all = self._dynamics(
                cs.vertcat(self.rho_raw, self.v_raw, self.w_raw),
                u_reordered,
                self.demands_actual[self.current_timestep, :]
            )
            # step dynamics
            self.rho_raw, self.v_raw, self.w_raw = cs.vertsplit(x_next, (0, self._n_seg, 2*self._n_seg, 2*self._n_seg+self._n_orig))
            self.rho_raw, self.v_raw, self.w_raw = np.array(self.rho_raw).flatten(), np.array(self.v_raw).flatten(), np.array(self.w_raw).flatten()

            q, q_o = cs.vertsplit(q_all, (0, self._n_seg, self._n_seg+self._n_orig))
            self.sim_results["Density"].append(self.rho_raw)
            self.sim_results["Speed"].append(self.v_raw)
            self.sim_results["Queue_Length"].append(self.w_raw)
            self.sim_results["Flow"].append(np.array(q).flatten())
            self.sim_results["Origin_Flow"].append(np.array(q_o).flatten())
            self.sim_results["VSL"].append(u_combined[self._n_ramp:])
            self.sim_results["Ramp_Metering_Rate"].append(u_combined[:self._n_ramp])
            self.sim_results["u_MPC"].append(self.u_mpc_raw)
            self.sim_results["u_DRL"].append(action)
            # computing TTS + queue penalty
            mpc_cost = (self.rho_raw * self._L * self._num_lanes).sum() + self.w_raw.sum()
            mpc_cost += 0.4*np.sum(np.square((self.u_prev_raw - u_combined) / self._u_ub))
            Ps = np.maximum(0.0, self.w_raw - self._max_queue)
            queue_penalty = self.w_p * Ps.sum()

            # subtract both as a scalar
            reward -= (mpc_cost + queue_penalty)
            self.current_timestep += 1
            print(f"[ENV] current_timestep = {self.current_timestep}")

        # update previous control action for next step
        self.u_prev_raw = u_combined.copy()
        # T_sim = 9000 s (length of the simulation)
        # _T = 10 s (length of the time step)
        # T_sim//_T = 9000/10 = 900 steps
        # Therefore, the simulation is truncated when the current timestep reaches 900
        truncated = (self.current_timestep >= int(self.T_sim//self._T))
        # recalculate MPC output every 300 seconds (every 30 steps)
        if not truncated and self.current_timestep % self._M_mpc == 0:
            # get demand forecast
            d_hat = self.demands_forecast[self.current_timestep:self.current_timestep+self._Np*self._M_mpc, :]
            # pad if forecast horizon exceeds remaining time
            if d_hat.shape[0] < self._Np*self._M_mpc:
                d_hat = np.pad(d_hat, ((0, self._Np*self._M_mpc-d_hat.shape[0]), (0, 0)), "edge")
            sol = self._mpc.solve(
                pars={"rho_0": self.rho_raw, "v_0": self.v_raw, "w_0": self.w_raw, "d": d_hat.T, 
                      "r_last": self.u_prev_raw[: self._n_ramp].reshape(-1, 1), 
                      "v_ctrl_last": self.u_prev_raw[self._n_ramp:].reshape(-1, 1)},
                vals0=self._sol_mpc_prev,
            )
            # store new solution
            self._sol_mpc_prev = sol.vals
            v_ctrl_last = sol.vals["v_ctrl"][:, 0]
            r_last = sol.vals["r"][0]
            self.u_mpc_raw = np.concatenate([r_last, v_ctrl_last]).astype(np.float32).flatten()

        self.state_norm = self.normalize_observations(self.rho_raw, self.v_raw, self.w_raw, self.u_mpc_raw, self.u_prev_raw, self.demands_actual[self.current_timestep])
        done = False
        reward = float(reward) # convert to float
        return self.state_norm, reward, done, truncated, {}

    # saturate control signal to [0,1] for ramp and [v_min, v_free] for VSL
    def saturate(self, control_signal: np.ndarray) -> np.ndarray:
        return np.minimum(np.maximum(control_signal, self._u_lb), self._u_ub)


########################################################################
# Main: Trial
########################################################################
if __name__ == "__main__":
    env = MetanetMPCEnv() # instantiate environment
    env.reset() # reset to initial state
    Num_Steps = 150
    for _ in range(Num_Steps):
        # Here just a random input to test the environment
        a = np.random.uniform(low=env.action_space.low, high=env.action_space.high).astype(np.float32)
        obs, reward, done, truncated, _ = env.step(action=a)
        if truncated:
            break
    
    env.sim_results["Density"] = np.stack(env.sim_results["Density"], axis=-1)
    env.sim_results["Speed"] = np.stack(env.sim_results["Speed"], axis=-1)
    env.sim_results["Queue_Length"] = np.stack(env.sim_results["Queue_Length"], axis=-1)
    env.sim_results["Flow"] = np.stack(env.sim_results["Flow"], axis=-1)
    env.sim_results["Origin_Flow"] = np.stack(env.sim_results["Origin_Flow"], axis=-1)
    env.sim_results["VSL"] = np.stack(env.sim_results["VSL"], axis=-1)
    env.sim_results["Ramp_Metering_Rate"] = np.stack(env.sim_results["Ramp_Metering_Rate"], axis=-1)
    env.sim_results["u_MPC"] = np.stack(env.sim_results["u_MPC"], axis=-1)
    env.sim_results["u_DRL"] = np.stack(env.sim_results["u_DRL"], axis=-1)
    
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(env.time[:env.current_timestep], env.sim_results["Density"].T)
    plt.xlabel("Time [h]")
    plt.ylabel("Density [veh/km/lane]")
    plt.savefig("plots/density.png")

    plt.figure()
    plt.plot(env.time[:env.current_timestep], env.sim_results["u_MPC"][0, :], label="MPC Baseline")
    plt.plot(env.time[:env.current_timestep], env.sim_results["Ramp_Metering_Rate"].T, label="Combined Input")
    plt.legend()
    plt.xlabel("Time [h]")
    plt.ylabel("Ramp Metering Rate [-]")
    plt.savefig("plots/ramp_metering_rate.png")
    plt.show()