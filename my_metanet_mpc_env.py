########################################################################
# Imports
########################################################################
import logging

import os
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
# Generate time-varying demands for two origins (mainline and on-ramp).
# Warm-up period of half an hour is added to demand profile.
def create_demands(time: np.ndarray) -> np.ndarray:
    return np.stack(
        (
            np.interp(time, (0, 15/60, 2.50, 2.75, 3.00), (0, 3500, 3500, 1000, 1000)),
            np.interp(time, (0, 15/60, 30/60, 0.60, 0.85, 1.0, 3.00), (0, 500, 500, 1500, 1500, 500, 500))
        )
    )

########################################################################
# Class: MetanetMPCEnv - custom gymnasium environment for traffic control
########################################################################
class MetanetMPCEnv(gym.Env):
    def __init__(self, M_drl: int = 6, w_u: float = 0.4, w_p: float = 10.0):
        super(MetanetMPCEnv, self).__init__()

        # adding debugging flags
        self.debug_mode = True
        self.step_count = 0
        self.nan_count = 0
        self.mpc_fail_count = 0
        self.episode_count = 0
        # set up logging
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)


        self._init_network() # build METANET network model
        self._init_mpc() # set up MPC controller
        self.T_warmup = 30.0/60 # warm-up time of 30 minutes (0.5 hours)
        self.T_sim = 2.5 # simulation time of 2.5 hours (main simulation)
        self.M_drl = M_drl # number of DRL steps (6*10=60s)
        self.w_u = w_u # weight for the control action
        self.w_p = w_p # penalty weight for queue violations

        # initializing state variables
        self.demands_actual = []
        self.time = []
        self.demands_forecast = []
        self.current_timestep = 0
        self.u_prev_raw = []
        self.rho_raw = []
        self.v_raw = []
        self.w_raw = []
        self._sol_mpc_prev = None
        self.u_mpc_raw = []
        self.state_norm = []
        self.sim_results = {}

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
        # no default values - we will start with an empty network


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
        self._net.step(T=self._T, tau=self._tau, eta=self._eta, kappa=self._kappa, delta=self._delta, init_conditions={O1: {"v_ctrl": self._v_free * 2}})
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
        w, _ = self._mpc.state("w", self._n_orig, lb=0, ub=[[200], [100]])
        # define actions: VSL controls and ramp metering rate
        v_ctrl, _ = self._mpc.action("v_ctrl", self._n_vsl, lb=self._vsl_min, ub=self._v_free)
        r, _ = self._mpc.action("r", lb=0, ub=1)
        # define disturbance (demands)
        d  = self._mpc.disturbance("d", self._n_orig)
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
            "ipopt": {"max_iter": 500,
                      "sb": "yes",
                      "print_level": 0,
                      "tol": 1e-6,
                      "acceptable_tol": 1e-4,
                      "mu_strategy": "adaptive",
                      "max_cpu_time": 10.0,},
        }
        self._mpc.init_solver(solver="ipopt", opts=opts)

    def _check_state_validity(self, rho, v, w):
        issues = []
        # check densities
        if np.any(rho<0):
            issues.append(f"Negative density: min={np.min(rho)}")
        if np.any(rho> self._rho_max*1.1):
            issues.append(f"Excessive density: max={np.max(rho)}")

        # check speeds
        if np.any(v<0):
            issues.append(f"Negative speed: min={np.min(v)}")
        if np.any(v > self._v_free * 1.2):
            issues.append(f"Excessive speed: max={np.max(v)}")

        # check queues
        if np.any(w<0):
            issues.append(f"Negative queue: min={np.min(w)}")

        queue_violations = w - self._max_queue
        if np.any(queue_violations>0):
            violating_queues = np.where(queue_violations>0)[0]
            for i in violating_queues:
                queue_type = "mainline" if i == 0 else "on-ramp"
                issues.append(f"Queue overflow ({queue_type}): {w[i]:.1f} > {self._max_queue[i]:.1f}")

        # check for NaN or inf values
        for name, arr in [("rho", rho), ("v", v), ("w", w)]:
            if np.any(np.isnan(arr)):
                issues.append(f"NaN in {name}")
            if np.any(np.isinf(arr)):
                issues.append(f"Inf in {name}")
        return issues

    def _safe_mpc_solve(self, pars, vals0=None):
        try:
            # Validate parameters before solving
            for key, value in pars.items():
                if np.any(np.isnan(value)) or np.any(np.isinf(value)):
                    self.logger.error(f"Invalid MPC parameter {key}: {value}")
                    return None
            # check state bounds
            state_issues = self._check_state_validity(pars["rho_0"], pars["v_0"], pars["w_0"])
            if state_issues:
                self.logger.warning(f"State issues before MPC solve: {state_issues}")

            # Solve MPC
            sol = self._mpc.solve(pars=pars, vals0=vals0)

            # check solution status
            if hasattr(sol, 'stats'):
                status = sol.stats.get("return_status", "unknown")
                if status not in ["Solve_Succeeded", "Solved_To_Acceptable_Level"]:
                    self.logger.warning(f"MPC solve status: {status}")
                    self.mpc_fail_count += 1
                    return None
            return sol
        except Exception as e:
            self.logger.error(f"MPC solve exception: {e}")
            self.nan_count +=1
            return None

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
        try:
            rho_norm = np.clip(np.asarray(rho).flatten() / self._rho_max, 0, 1)
            v_norm = np.clip(np.asarray(v).flatten() / self._v_free, 0, 1)
            w_norm = np.clip(np.asarray(w).flatten()/ self._max_queue, 0, 2)

            u_mpc_norm = np.clip(np.array(u_mpc).flatten() / self._u_ub, 0, 1)
            u_prev_norm = np.clip(np.array(u_prev).flatten() / self._u_ub, 0, 1)
            d_norm = np.clip(demands / self._max_demands, 0, 1)

        # DEBUG:
        #print("SHAPES in normalize:")
        #print("  rho_norm:", rho_norm.shape)
        #print("  v_norm:  ", v_norm.shape)
        #print("  w_norm:  ", w_norm.shape)
        #print("  u_mpc_norm:", u_mpc_norm.shape)
        #print("  d_norm:  ", d_norm.shape)
        #print("  u_prev_norm:", u_prev_norm.shape)
            normalized_obs = np.concatenate([rho_norm, v_norm, w_norm, u_mpc_norm, d_norm, u_prev_norm])
            # check for NaN in normalized observations
            if np.any(np.isnan(normalized_obs)):
                self.logger.error("NaN in normalized observations!")
                return np.zeros_like(normalized_obs)

            return normalized_obs.astype(np.float32)
        except Exception as e:
            self.logger.error(f"Normalization error: {e}")
            # Return safe fallback observation
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)


    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.episode_count +=1
        self.step_count = 0

        if self.debug_mode:
            self.logger.info(f"=== RESET Episode {self.episode_count} ===")
            if self.episode_count > 1:
                self.logger.info(f"Previous episode stats: "
                                 f"NaN count: {self.nan_count}, MPC failures: {self.mpc_fail_count}")

        try:
            # 1) build your time and demands
            self.time = np.arange(0, self.T_warmup + self.T_sim + self._T, self._T )
            # 2) creating demands with low-noise only during the main simulation period
            self.demands_forecast = create_demands(self.time).T.astype(np.float32)
            noise = np.random.normal(
                loc=0.0,
                scale=self.noise_std[np.newaxis, :],
                size=self.demands_forecast.shape
            ).astype(np.float32)

            warm_steps = int(self.T_warmup/ self._T)
            # zero noise during warm-up period
            noise[:warm_steps, :] = 0.0
            self.demands_actual = np.clip(self.demands_forecast + noise, 0.0, None)
            self.current_timestep = 0

            # 3) force empty initial states (zero densities & queues, freeflow speeds)
            self.rho_raw = np.zeros(self._n_seg, dtype=np.float32)
            self.v_raw = np.full(self._n_seg, self._v_free, dtype=np.float32)
            self.w_raw = np.zeros(self._n_orig, dtype=np.float32)


            # reset previous control & clear MPC warm‐start
            self.u_prev_raw = np.concatenate([
                np.ones(self._n_ramp, dtype=np.float32),
                np.full(self._n_vsl, self._v_free, dtype=np.float32)
            ])
            self._sol_mpc_prev = None

            # 4) warm-up for 30 minutes with forecast demands (no noise)
            u_reordered = cs.vertcat(
                self.u_prev_raw[self._n_ramp:],
                self.u_prev_raw[:self._n_ramp]
            )
            warm_up_steps = int(self.T_warmup/ self._T)
            self.logger.info(f"Starting warm-up: {warm_up_steps} steps ({self.T_warmup * 60:.0f} minutes)")

            # initializing simulation results storage
            self.sim_results = {
                "Density": [], "Flow": [], "Speed": [], "Queue_Length": [],
                "Origin_Flow": [], "Ramp_Metering_Rate": [], "VSL": [],
                "u_MPC": [], "u_DRL": []
            }

            for _ in range(warm_up_steps):
                try:
                    x_next, q_all = self._dynamics(
                        cs.vertcat(self.rho_raw, self.v_raw, self.w_raw),
                        u_reordered,
                        self.demands_forecast[self.current_timestep, :] # no noise during warm-up
                    )

                    # DEBUG: Looking for queue length
                    x_next_np = np.array(x_next).flatten()
                    q_all_np = np.array(q_all).flatten()
                    print(f"x_next shape: {x_next_np.shape}, content: {x_next_np}")
                    print(f"q_all shape: {q_all_np.shape}, content: {q_all_np}")

                    if len(q_all_np) > self._n_seg + self._n_orig:
                        potential_queues = q_all_np[-(self._n_orig):]
                        print(f"Potential queues from q_all: {potential_queues}")

                    demands = self.demands_forecast[self.current_timestep, :]
                    q_all_np = np.array(q_all).flatten()
                    origin_flow = q_all_np[6:8]
                    queue_change = demands - origin_flow

                    if np.any(np.abs(queue_change)>0.1):
                        print(f"QUEUE FORMATION DETECTED at step {_}:")
                        print(f"  Demands: {demands}")
                        print(f"  Origin flows: {origin_flow}")
                        print(f"  Queue change: {queue_change}")
                        print(f"  Current queues: {np.array(x_next)[12:14]}")


                    # DEBUG: Print what x_next contains
                    x_next_np = np.array(x_next).flatten()
                    print(f"Step {_}: x_next shape = {x_next_np.shape}")
                    print(f"  Full state: {x_next_np}")
                    print(f"  rho (first 6): {x_next_np[:6]}")
                    print(f"  v (next 6): {x_next_np[6:12]}")
                    print(f"  w (last 2): {x_next_np[12:14]}")



                    r1, v1, w1 = cs.vertsplit(
                        x_next,
                        (0, self._n_seg, 2 * self._n_seg, 2 * self._n_seg + self._n_orig)
                    )

                    print(f"  After vertsplit:")
                    print(f"    rho: {np.array(r1).flatten()}")
                    print(f"    v: {np.array(v1).flatten()}")
                    print(f"    w: {np.array(w1).flatten()}")

                    self.rho_raw = np.maximum(np.array(r1).flatten(), 0.0)
                    self.v_raw = np.maximum(np.array(v1).flatten(), 1.0)
                    self.w_raw = np.maximum(np.array(w1).flatten(), 0.0)
                    self.current_timestep += 1

                    q, q_o = cs.vertsplit(q_all, (0, self._n_seg, self._n_seg + self._n_orig))
                    self.sim_results["Density"].append(self.rho_raw)
                    self.sim_results["Speed"].append(self.v_raw)
                    self.sim_results["Queue_Length"].append(self.w_raw)
                    self.sim_results["Flow"].append(np.array(q).flatten())
                    self.sim_results["Origin_Flow"].append(np.array(q_o).flatten())
                    self.sim_results["VSL"].append(self.u_prev_raw[self._n_ramp:])
                    self.sim_results["Ramp_Metering_Rate"].append(self.u_prev_raw[:self._n_ramp])
                    self.sim_results["u_MPC"].append(self.u_mpc_raw)
                    self.sim_results["u_DRL"].append(np.zeros_like(self.u_prev_raw))

                    # Log warm-up progress
                    if self.debug_mode and _ % 50 == 0:
                        self.logger.info(f"Warm-up step {_}/{warm_up_steps}: "
                                        f"avg_rho={np.mean(self.rho_raw):.1f}, "
                                        f"avg_v={np.mean(self.v_raw):.1f}, "
                                        f"total_queue={np.sum(self.w_raw):.1f}")


                except Exception as e:
                    self.logger.error(f"Error in warm-up step {_}: {e}")
                    # For warm-up, just continue with current state
                    self.current_timestep += 1
            self.logger.info(f"Warm-up completed. Final state: "
                            f"avg_rho={np.mean(self.rho_raw):.1f}, "
                            f"avg_v={np.mean(self.v_raw):.1f}, "
                            f"total_queue={np.sum(self.w_raw):.1f}")
            # initial mpc solve with safe handling
            mpc_pars = {
                "rho_0": self.rho_raw,
                "v_0": self.v_raw,
                "w_0": self.w_raw,
                "d": self.demands_forecast[:self._Np * self._M_mpc, :].T,
                "r_last": self.u_prev_raw[:self._n_ramp].reshape(-1,1),
                "v_ctrl_last": self.u_prev_raw[self._n_ramp:].reshape(-1,1)
            }
            sol = self._safe_mpc_solve(mpc_pars, self._sol_mpc_prev)

            if sol is not None:
                self._sol_mpc_prev = sol.vals
                v_ctrl_last = sol.vals["v_ctrl"][:,0]
                r_last = sol.vals["r"][0]
                self.u_mpc_raw = np.concatenate([r_last, v_ctrl_last]).astype(np.float32).flatten()
            else:
                # fallback mpc output
                self.logger.warning("Using fallback MPC output in reset")
                self.u_mpc_raw = self.u_prev_raw.copy()

            # generating initial observation (for main simulation)
            self.state_norm = self.normalize_observations(
                self.rho_raw, self.v_raw, self.w_raw,
                self.u_mpc_raw, self.u_prev_raw,
                self.demands_actual[0] #TODO: maybe change to current_timestep to be more consistent
            )

            if self.debug_mode:
                self.logger.info(f"Reset completed. Post warm-up state checks:")
                initial_issues = self._check_state_validity(self.rho_raw, self.v_raw, self.w_raw)
                if initial_issues:
                    self.logger.warning(f"Post-warm-up state issues: {initial_issues}")
                else:
                    self.logger.info("Post-warm-up state looks good")

            return self.state_norm, {}

        except Exception as e:
            self.logger.error(f"Critical error in reset: {e}")
            # return emergency fallback state:
            return np.zeros(self.observation_space[0], dtype=np.float32), {}



    def step(self, action):
        self.step_count +=1
        """
        Step function -> T=10 s, DRL = 6 steps (6*10=60s). MPC covers 30 steps (30*10=300s).
            1. Combine MPC baseline with DRL tweak. saturate to [0≤r≤1, v_min≤vsl≤v_free].
            2. Roll out self.M_drl steps through the dynamics function
            3. Accumulate reward
            4. Return normalized observation, reward, done, truncated, info
        """
        try:
            # validate input action
            if np.any(np.isnan(action)) or np.any(np.isinf(action)):
                self.logger.warning(f"Invalid action received: {action}")
                action = np.zeros_like(action)
            # check state validity before step
            state_issues = self._check_state_validity(self.rho_raw, self.v_raw, self.w_raw)
            if state_issues and self.debug_mode and self.step_count % 50 == 0:
                self.logger.warning(f"State issues before step {self.step_count}: {state_issues}")

            # combine baseline MPC control with DRL action and saturate
            u_combined = self.saturate(self.u_mpc_raw + action)
            u_reordered = cs.vertcat(u_combined[self._n_ramp:], u_combined[:self._n_ramp])
            reward = 0.0

            # execute M_drl dynamics steps
            for _ in range(self.M_drl):
                try:
                    x_next, q_all = self._dynamics(
                        cs.vertcat(self.rho_raw, self.v_raw, self.w_raw),
                        u_reordered,
                        self.demands_actual[self.current_timestep, :]
                    )

                    demands_drl = self.demands_actual[self.current_timestep, :]
                    q_all_np = np.array(q_all).flatten()
                    origin_flow = q_all_np[6:8]
                    queue_change = demands_drl - origin_flow

                    if np.any(np.abs(queue_change) > 0.1):
                        print(f"QUEUE FORMATION DETECTED at step {_}:")
                        print(f"  Demands: {demands_drl}")
                        print(f"  Origin flows: {origin_flow}")
                        print(f"  Queue change: {queue_change}")
                        print(f"  Current queues: {np.array(x_next)[12:14]}")


                    # update states with bounds checking
                    self.rho_raw, self.v_raw, self.w_raw = cs.vertsplit(x_next, (0, self._n_seg, 2 * self._n_seg, 2 * self._n_seg + self._n_orig))
                    # apply bounds to prevent numerical issues
                    self.rho_raw = np.clip(np.array(self.rho_raw).flatten(), 0.1, self._rho_max)
                    self.v_raw = np.clip(np.array(self.v_raw).flatten(), 1.0, self._v_free)
                    self.w_raw = np.clip(np.array(self.w_raw).flatten(), 0.0, self._max_queue * 2) # allowing for some overflow

                    # storing results
                    q, q_o = cs.vertsplit(q_all, (0, self._n_seg, self._n_seg + self._n_orig))
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
                    mpc_cost += 0.4 * np.sum(np.square((self.u_prev_raw - u_combined) / self._u_ub))
                    Ps = np.maximum(0.0, self.w_raw - self._max_queue)
                    queue_penalty = self.w_p * Ps.sum()

                    # subtract both as a scalar
                    reward -= (mpc_cost + queue_penalty)
                    self.current_timestep += 1

                except Exception as e:
                    self.logger.error(f"Dynamics error at step {self.current_timestep}, drl_step {_}: {e}")
                    # fallback: small perturbation of current state
                    self.rho_raw *= 1.001
                    self.v_raw *= 0.999
                    self.current_timestep += 1
                    reward -= 100 # penalty for dynamics failure
            # update previous control action for next step
            self.u_prev_raw = u_combined.copy()
            # Truncated after warm-up + sim
            sim_start = int(self.T_warmup/self._T)
            sim_end = sim_start + int(self.T_sim/self._T)
            truncated = (self.current_timestep >= sim_end)

            # mpc re-solve during main simulation
            # only re-solve mpc during the main simulation time
            if not truncated:
                sim_start = int(self.T_warmup/self._T)
                sim_end = sim_start + int(self.T_sim/self._T)

                if (sim_start <= self.current_timestep < sim_end and
                    (self.current_timestep - sim_start) % self._M_mpc == 0):

                    # getting demand forecast
                    d_hat = self.demands_forecast[self.current_timestep:self.current_timestep + self._Np * self._M_mpc,
                            :]
                    # pad if forecast horizon exceeds remaining time
                    if d_hat.shape[0] < self._Np * self._M_mpc:
                        d_hat = np.pad(d_hat, ((0, self._Np * self._M_mpc - d_hat.shape[0]), (0, 0)), "edge")

                    # safe mpc solve
                    mpc_pars = {
                        "rho_0": self.rho_raw,
                        "v_0": self.v_raw,
                        "w_0": self.w_raw,
                        "d": d_hat.T,
                        "r_last": self.u_prev_raw[:self._n_ramp].reshape(-1,1),
                        "v_ctrl_last": self.u_prev_raw[self._n_ramp].reshape(-1,1)

                    }
                    sol = self._safe_mpc_solve(mpc_pars, self._sol_mpc_prev)

                    if sol is not None:
                        self._sol_mpc_prev = sol.vals
                        v_ctrl_last = sol.vals["v_ctrl"][:,0]
                        r_last = sol.vals["r"][0]
                        self.u_mpc_raw = np.concatenate([r_last, v_ctrl_last]).astype(np.float32).flatten()

                    else:
                        # Fallback: keep previous MPC output or use safe default
                        self.logger.warning(f"MPC failed at step {self.current_timestep}, using fallback")
                        # Simple proportional controller as fallback
                        queue_ratio = self.w_raw / (self._max_queue + 1e-6)
                        # Reduce ramp rate if queues are getting full
                        fallback_ramp = np.clip(0.9 - 0.5 * np.max(queue_ratio), 0.3, 1.0)
                        # Reduce VSL if density is high
                        avg_density_ratio = np.mean(self.rho_raw) / self._rho_max
                        fallback_vsl = np.clip(self._v_free - 30 * avg_density_ratio, self._vsl_min, self._v_free)
                        self.u_mpc_raw = np.array([fallback_ramp, fallback_vsl, fallback_vsl], dtype=np.float32)

            # generating observations
            current_demand_idx = min(self.current_timestep, len(self.demands_actual)-1)
            self.state_norm = self.normalize_observations(
                self.rho_raw, self.v_raw, self.w_raw,
                self.u_mpc_raw, self.u_prev_raw,
                self.demands_actual[current_demand_idx]
            )
            done = truncated
            reward = float(reward)

            # final state validation and logging
            if self.debug_mode and (truncated or self.step_count % 100 ==0):
                final_issues = self._check_state_validity(self.rho_raw, self.v_raw, self.w_raw)
                if final_issues:
                    self.logger.warning(f"Final state issues at step {self.step_count}: {final_issues}")

                if truncated:
                    self.logger.info(f"Episode {self.episode_count} completed after {self.step_count} steps")
                    self.logger.info(f"Episode stats - NaN: {self.nan_count}, MPC fails: {self.mpc_fail_count}")

            return self.state_norm, reward, done, truncated, {}
        except Exception as e:
            self.logger.error(f"Critical error in step {self.step_count}: {e}")
            # return safe fallback
            return self.state_norm, -1000.0, True, True, {"error": str(e)}


    # saturate control signal to [0,1] for ramp and [v_min, v_free] for VSL
    def saturate(self, control_signal: np.ndarray) -> np.ndarray:
        return np.minimum(np.maximum(control_signal, self._u_lb), self._u_ub)

    def calculate_expected_episode_length(self):
        main_sim_time_hours = self.T_sim
        main_sim_env_steps = int(main_sim_time_hours/ self._T) # 900 steps
        total_drl_steps = main_sim_env_steps // self.M_drl # 150 drl steps
        return total_drl_steps

    def get_debug_info(self):
        # getting comprehensive debug information
        info = {
            "episode_count": self.episode_count,
            "step_count": self.step_count,
            "current_timestep": self.current_timestep,
            "nan_count": self.nan_count,
            "mpc_fail_count": self.mpc_fail_count,
            "expected_episode_length": self.calculate_expected_episode_length(),
            "current_state": {
                "rho": self.rho_raw.tolist() if hasattr(self, 'rho_raw') else None,
                "v": self.v_raw.tolist() if hasattr(self, 'v_raw') else None,
                "w": self.w_raw.tolist() if hasattr(self, 'w_raw') else None,
            },
            "queue_utilization": (self.w_raw/ self._max_queue).tolist() if hasattr(self, 'w_raw') else None,
            "state_issues": self._check_state_validity(self.rho_raw, self.v_raw, self.w_raw) if hasattr(self, 'rho_raw') else []
        }
        return info


########################################################################
# Main: Trial
########################################################################
if __name__ == "__main__":
    print("=== TESTING ENHANCED METANET MPC ENVIRONMENT ===")
    env = MetanetMPCEnv() # instantiate environment
    env.debug_mode = True

    print(f"Expected episode length: {env.calculate_expected_episode_length()} steps")
    print(f"Queue constraints: {env._max_queue}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space shape: {env.observation_space.shape}")

    # Test reset
    print("\n=== TESTING RESET ===")
    obs, info = env.reset()
    print(f"Reset successful. Observation shape: {obs.shape}")
    initial_debug = env.get_debug_info()
    print(f"Initial debug info: {initial_debug}")

    # Test steps
    print("\n=== TESTING STEPS ===")
    Num_Steps = 50  # Reduced for debugging

    for step_i in range(Num_Steps):
        # Random action within bounds
        action = np.zeros(shape=(3,))
            #low=env.action_space.low,
            ##).astype(np.float32)

        obs, reward, done, truncated, step_info = env.step(action)

        if step_i % 10 == 0:
            print(f"Step {step_i}: reward={reward:.2f}, done={done}, truncated={truncated}")
            debug_info = env.get_debug_info()
            if debug_info["state_issues"]:
                print(f"  State issues: {debug_info['state_issues']}")

        if done or truncated:
            print(f"Episode ended at step {step_i}")
            break

    # Final debug information
    print("\n=== FINAL DEBUG INFO ===")
    final_debug = env.get_debug_info()
    for key, value in final_debug.items():
        if key != "current_state":  # Skip detailed state info
            print(f"{key}: {value}")

    print(f"\nTotal NaN errors: {env.nan_count}")
    print(f"Total MPC failures: {env.mpc_fail_count}")
    print(f"Final step count: {env.step_count}")

    # Test plotting if we have results
    if env.sim_results["Density"]:
        try:
            import matplotlib.pyplot as plt

            # Stack results for plotting
            env.sim_results["Density"] = np.stack(env.sim_results["Density"], axis=-1)
            env.sim_results["Queue_Length"] = np.stack(env.sim_results["Queue_Length"], axis=-1)

            plt.figure(figsize=(12, 8))

            # Plot densities
            plt.subplot(2, 2, 1)
            plt.plot(env.sim_results["Density"].T)
            plt.title("Density Evolution")
            plt.ylabel("Density [veh/km/lane]")
            plt.grid(True)

            # Plot queue lengths
            plt.subplot(2, 2, 2)
            plt.plot(env.sim_results["Queue_Length"].T, label=['Mainline', 'On-ramp'])
            plt.axhline(y=env._max_queue[0], color='r', linestyle='--', label='Mainline limit')
            plt.axhline(y=env._max_queue[1], color='orange', linestyle='--', label='On-ramp limit')
            plt.title("Queue Length Evolution")
            plt.ylabel("Queue Length [veh]")
            plt.legend()
            plt.grid(True)

            # Plot control actions
            if env.sim_results["u_MPC"] and env.sim_results["Ramp_Metering_Rate"]:
                env.sim_results["u_MPC"] = np.stack(env.sim_results["u_MPC"], axis=-1)
                env.sim_results["Ramp_Metering_Rate"] = np.stack(env.sim_results["Ramp_Metering_Rate"], axis=-1)

                plt.subplot(2, 2, 3)
                plt.plot(env.sim_results["u_MPC"][0, :], label="MPC Baseline", linestyle='--')
                plt.plot(env.sim_results["Ramp_Metering_Rate"].T, label="Combined Input")
                plt.title("Ramp Metering Rate")
                plt.ylabel("Ramp Rate [-]")
                plt.legend()
                plt.grid(True)

                plt.subplot(2, 2, 4)
                if len(env.sim_results["u_MPC"]) > 1:
                    plt.plot(env.sim_results["u_MPC"][1:, :].T, label=['VSL 1', 'VSL 2'])
                    plt.title("Variable Speed Limits")
                    plt.ylabel("Speed Limit [km/h]")
                    plt.legend()
                    plt.grid(True)

            plt.tight_layout()

            # Create plots directory if it doesn't exist
            os.makedirs("plots", exist_ok=True)
            plt.savefig("plots/debug_results.png", dpi=150, bbox_inches='tight')
            print("Debug plots saved to plots/debug_results.png")
            plt.show()

        except ImportError:
            print("Matplotlib not available for plotting")
        except Exception as e:
            print(f"Plotting error: {e}")

    # env.sim_results["Density"] = np.stack(env.sim_results["Density"], axis=-1)
    # env.sim_results["Speed"] = np.stack(env.sim_results["Speed"], axis=-1)
    # env.sim_results["Queue_Length"] = np.stack(env.sim_results["Queue_Length"], axis=-1)
    # env.sim_results["Flow"] = np.stack(env.sim_results["Flow"], axis=-1)
    # env.sim_results["Origin_Flow"] = np.stack(env.sim_results["Origin_Flow"], axis=-1)
    # env.sim_results["VSL"] = np.stack(env.sim_results["VSL"], axis=-1)
    # env.sim_results["Ramp_Metering_Rate"] = np.stack(env.sim_results["Ramp_Metering_Rate"], axis=-1)
    # env.sim_results["u_MPC"] = np.stack(env.sim_results["u_MPC"], axis=-1)
    # env.sim_results["u_DRL"] = np.stack(env.sim_results["u_DRL"], axis=-1)
    #
    # import matplotlib.pyplot as plt
    #
    # plt.figure()
    # plt.plot(env.time[:env.current_timestep], env.sim_results["Density"].T)
    # plt.xlabel("Time [h]")
    # plt.ylabel("Density [veh/km/lane]")
    # plt.savefig("plots/density.png")
    #
    # plt.figure()
    # plt.plot(env.time[:env.current_timestep], env.sim_results["u_MPC"][0, :], label="MPC Baseline")
    # plt.plot(env.time[:env.current_timestep], env.sim_results["Ramp_Metering_Rate"].T, label="Combined Input")
    # plt.legend()
    # plt.xlabel("Time [h]")
    # plt.ylabel("Ramp Metering Rate [-]")
    # plt.savefig("plots/ramp_metering_rate.png")
    # plt.show()