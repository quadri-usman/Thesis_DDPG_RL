"""Stage-3B event-terminal peak-aware randomized GFM PI-gain adaptation environment.

Uses the validated corrected GFM benchmark with explicit TRAIN / VALIDATION /
TEST load splits. A new split-consistent RandomLoadScenario is generated at
every reset.

Observation (12):
    [df, RoCoF, evd, Evd, evq, eid, Eid, eiq, logKpv, logKiv, logKpi, logKii]

Action (4):
    normalized [dKpv, dKiv, dKpi, dKii] in [-1,1]

Timing:
    plant/controller Ts = 10 us
    RL decision time    = 1 ms by default
    warm-up             = 1.5 s by default

During warm-up:
    * incoming RL actions are ignored;
    * gains remain exactly [2,14,2,20];
    * raw RL integrals Evd/Eid are not accumulated.

At the warm-up boundary:
    * Evd = Eid = 0;
    * RoCoF history is re-anchored;
    * RL gain adaptation becomes active.

Stage 3 keeps the Stage-2 plant, 12-state observation, refined gain bounds,
and alpha=0.01 unchanged, but strengthens the reward around frequency quality.

Peak-aware terms are incremental within each disturbance window:
    J_peak_frequency = (p_f,new^2 - p_f,old^2) / frequency_scale^2
    J_peak_rocof     = (p_R,new^2 - p_R,old^2) / rocof_scale^2
where p_f and p_R are running absolute peaks since the latest disturbance.
The peak memories reset at every disturbance event. This avoids repeatedly
charging the full historical peak at every time step.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:
    raise ImportError("Install Gymnasium with: pip install gymnasium") from exc

from benchmark_scenarios_rl import MatlabScenarioBenchmark
from controller_rl_gains_refined import GainBounds, RLGainAdapter
from controller_rl_gains_target import (
    PIGains, RLGainAdaptiveGFMController
)
from training_scenarios_split import (
    RandomLoadScenario,
    RandomLoadScenarioConfig,
    ScenarioSplit,
    validate_breaker_map,
    validate_experimental_split,
)


@dataclass(frozen=True)
class RewardConfig:
    # Stage-3 primary-objective weights.
    w_frequency: float = 8.0
    w_rocof: float = 4.0
    w_settling: float = 3.0
    # Stage-3B: event-terminal objective + optimized-gain prior.
    # Incremental peak penalties are disabled; the complete event excursion is
    # charged once at the next disturbance/end of episode.
    w_peak_frequency: float = 0.0
    w_peak_rocof: float = 0.0
    w_event_frequency: float = 4.0
    w_event_rocof: float = 2.0
    w_event_settling: float = 4.0
    w_gain_prior: float = 0.25
    event_settling_scale_s: float = 1.0

    # Secondary regulation / smoothness objectives remain present but weaker.
    w_voltage: float = 0.25
    w_current: float = 0.10
    w_gain_motion: float = 0.005

    frequency_scale_hz: float = 0.1
    rocof_scale_hz_s: float = 15.0
    voltage_error_scale_v: float = 380.0
    current_error_scale_a: float = 200.0

    settling_band_hz: float = 0.01
    settling_dwell_s: float = 0.10


@dataclass(frozen=True)
class ObservationConfig:
    df_scale_hz: float = 0.5
    rocof_scale_hz_s: float = 15.0
    voltage_error_scale_v: float = 380.0
    voltage_integral_scale_vs: float = 100.0
    current_error_scale_a: float = 200.0
    current_integral_scale_as: float = 100.0
    clip: float = 10.0


@dataclass(frozen=True)
class SafetyConfig:
    min_frequency_hz: float = 55.0
    max_frequency_hz: float = 65.0
    max_voltage_dq_v: float = 900.0
    max_current_dq_a: float = 550.0
    max_modulation_abs: float = 2000.0
    terminal_penalty: float = 1.0e4


class RandomizedGFMGainTuningEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario_config=RandomLoadScenarioConfig(),
        split=ScenarioSplit.TRAIN,
        rl_dt=1.0e-3,
        reward_config=RewardConfig(),
        observation_config=ObservationConfig(),
        gain_bounds=GainBounds(),
        safety_config=SafetyConfig(),
    ):
        super().__init__()
        validate_breaker_map()
        validate_experimental_split()

        self.scenario_config = scenario_config
        self.split = ScenarioSplit(split)
        self.episode_duration = float(scenario_config.episode_duration)
        self.warmup_time = float(scenario_config.warmup_time)
        self.rl_dt = float(rl_dt)
        self.reward_cfg = reward_config
        self.obs_cfg = observation_config
        self.gain_bounds = gain_bounds
        self.safety_cfg = safety_config

        self.sim = MatlabScenarioBenchmark()
        self.base_dt = float(self.sim.params.dt)
        n = int(round(self.rl_dt / self.base_dt))
        if n < 1 or not np.isclose(n*self.base_dt, self.rl_dt, atol=1e-15, rtol=0):
            raise ValueError("rl_dt must be an integer multiple of base dt")
        self.base_steps_per_action = n

        self.action_space = spaces.Box(
            -1.0, 1.0, shape=(4,), dtype=np.float32
        )
        c = float(self.obs_cfg.clip)
        self.observation_space = spaces.Box(
            -c, c, shape=(12,), dtype=np.float32
        )

        self.gain_adapter = RLGainAdapter(self.gain_bounds)
        self.scenario = None

        self._Evd = 0.0
        self._Eid = 0.0
        self._previous_frequency = 60.0
        self._last_sample = None
        self._rl_active = False

        self._event_index = 0
        self._settling_active = False
        self._settling_elapsed = 0.0
        self._inside_band_elapsed = 0.0
        self._last_settling_time = math.nan

        # Stage-3 running absolute peaks for the ACTIVE disturbance window.
        # Peak penalties are zero before the first disturbance.
        self._peak_window_active = False
        self._peak_abs_df = 0.0
        self._peak_abs_rocof = 0.0
        self._event_settled_time = math.nan
        self._pending_event_terminal_cost = 0.0
        self._last_event_terminal_cost = 0.0

    def _attach_controller(self):
        self.gain_adapter.reset()
        self.sim.controller = RLGainAdaptiveGFMController(
            self.sim.params, self.gain_adapter.gains
        )

    def _patch_random_scenario(self):
        """Redirect this benchmark instance to the episode's breaker schedule.

        benchmark_scenarios_rl._input() resolves breaker_group_commands in its
        defining module. Replacing that module-global function makes every
        plant input use this episode's randomized topology while leaving the
        validated electrical integration code unchanged.
        """
        import benchmark_scenarios_rl as bench

        scenario = self.scenario
        bench.breaker_group_commands = lambda t: scenario.breaker_groups(t)
        bench.nominal_operating_condition = (
            lambda t: (scenario.load_power(t), True)
        )

    def _obs(self, sample, rocof):
        c = self.obs_cfg
        o = sample.controller
        x = np.array([
            (o.frequency-60.0)/c.df_scale_hz,
            rocof/c.rocof_scale_hz_s,
            o.vd_error/c.voltage_error_scale_v,
            self._Evd/c.voltage_integral_scale_vs,
            o.vq_error/c.voltage_error_scale_v,
            o.id_error/c.current_error_scale_a,
            self._Eid/c.current_integral_scale_as,
            o.iq_error/c.current_error_scale_a,
            *self.gain_adapter.normalized_gain_state(),
        ], dtype=float)
        return np.clip(x, -c.clip, c.clip).astype(np.float32)

    def _start_events_crossed(self, t0, t1):
        times = self.scenario.event_times
        while self._event_index < len(times) and times[self._event_index] <= t1+1e-12:
            te = float(times[self._event_index])
            if te > t0+1e-12:
                self._settling_active = True
                self._settling_elapsed = 0.0
                self._inside_band_elapsed = 0.0
                self._last_settling_time = math.nan

                # Finalize the PREVIOUS disturbance window exactly once.
                if self._peak_window_active:
                    self._pending_event_terminal_cost += self._event_terminal_cost()

                # A new disturbance starts a fresh event-metric window.
                self._peak_window_active = True
                self._peak_abs_df = 0.0
                self._peak_abs_rocof = 0.0
                self._event_settled_time = math.nan
            self._event_index += 1

    def _settling_cost(self, df):
        if not self._settling_active:
            return 0.0
        cfg = self.reward_cfg
        self._settling_elapsed += self.rl_dt
        if abs(df) <= cfg.settling_band_hz:
            self._inside_band_elapsed += self.rl_dt
        else:
            self._inside_band_elapsed = 0.0
        cost = 1.0
        if self._inside_band_elapsed >= cfg.settling_dwell_s:
            self._settling_active = False
            self._last_settling_time = max(
                0.0, self._settling_elapsed-cfg.settling_dwell_s
            )
            self._event_settled_time = self._last_settling_time
            cost = 0.0
        return cost

    def _event_terminal_cost(self):
        """One-shot objective cost for the completed disturbance window.

        Uses absolute physical normalizers rather than hidden validation/test
        reference outcomes, avoiding leakage from held-out scenarios.
        """
        c = self.reward_cfg
        jf = (self._peak_abs_df / c.frequency_scale_hz) ** 2
        jr = (self._peak_abs_rocof / c.rocof_scale_hz_s) ** 2
        # If not settled before the next event, charge the full observed window.
        ts = self._event_settled_time
        if not np.isfinite(ts):
            ts = self._settling_elapsed
        jt = (float(ts) / c.event_settling_scale_s) ** 2
        cost = c.w_event_frequency*jf + c.w_event_rocof*jr + c.w_event_settling*jt
        self._last_event_terminal_cost = float(cost)
        return float(cost)

    def _gain_prior_cost(self):
        """Quadratic distance from experimentally refined [2,1.4,.7,2]."""
        z = self.gain_adapter.normalized_gain_state().astype(float)
        zstar = self.gain_adapter.target_to_action(
            np.array([2.0, 1.4, 0.7, 2.0], dtype=float)
        ).astype(float)
        return float(np.mean((z-zstar)**2))

    def _reward(self, sample, rocof, gain_motion, settling_cost):
        c = self.reward_cfg
        o = sample.controller
        df = o.frequency - 60.0

        # Instantaneous frequency-quality costs.
        Jf = (df / c.frequency_scale_hz) ** 2
        Jr = (rocof / c.rocof_scale_hz_s) ** 2
        Js = float(settling_cost)

        # Track complete event peaks; charge them only at event termination.
        Jpf = Jpr = 0.0
        if self._peak_window_active:
            self._peak_abs_df = max(self._peak_abs_df, abs(float(df)))
            self._peak_abs_rocof = max(self._peak_abs_rocof, abs(float(rocof)))

        Jv = (o.vd_error**2 + o.vq_error**2) / (c.voltage_error_scale_v**2)
        Ji = (o.id_error**2 + o.iq_error**2) / (c.current_error_scale_a**2)

        # Penalize actual normalized gain movement, not target magnitude.
        Jk = float(np.dot(gain_motion, gain_motion))
        Jprior = self._gain_prior_cost()
        Jevent = float(self._pending_event_terminal_cost)
        self._pending_event_terminal_cost = 0.0

        cost = (
            c.w_frequency * Jf
            + c.w_rocof * Jr
            + c.w_settling * Js
            + c.w_peak_frequency * Jpf
            + c.w_peak_rocof * Jpr
            + c.w_voltage * Jv
            + c.w_current * Ji
            + c.w_gain_motion * Jk
            + c.w_gain_prior * Jprior
            + Jevent
        )
        return -float(cost), dict(
            J_frequency=float(Jf),
            J_rocof=float(Jr),
            J_settling=float(Js),
            J_peak_frequency=float(Jpf),
            J_peak_rocof=float(Jpr),
            J_voltage=float(Jv),
            J_current=float(Ji),
            J_gain_motion=float(Jk),
            J_gain_prior=float(Jprior),
            J_event_terminal=float(Jevent),
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Gymnasium's seeded RNG guarantees repeatable episodes.
        self.scenario = RandomLoadScenario.generate(
            rng=self.np_random,
            config=self.scenario_config,
            seed=seed,
            split=self.split,
        )

        self.sim = MatlabScenarioBenchmark()
        self._patch_random_scenario()
        self._attach_controller()

        self._Evd = self._Eid = 0.0
        self._previous_frequency = 60.0
        self._last_sample = None
        self._rl_active = False

        self._event_index = 0
        self._settling_active = False
        self._settling_elapsed = 0.0
        self._inside_band_elapsed = 0.0
        self._last_settling_time = math.nan
        self._peak_window_active = False
        self._peak_abs_df = 0.0
        self._peak_abs_rocof = 0.0

        # One validated base sample creates the initial observation.
        sample = self.sim.step()
        self._last_sample = sample
        self._previous_frequency = sample.controller.frequency

        return self._obs(sample, 0.0), self._info(sample, 0.0)

    def step(self, action):
        requested_action = np.clip(
            np.asarray(action, dtype=float).reshape(4), -1.0, 1.0
        )
        if not np.all(np.isfinite(requested_action)):
            raise ValueError("RL target-gain action contains NaN/Inf")

        t0 = float(self.sim.state.time)
        normalized_before = self.gain_adapter.normalized_gain_state().copy()

        # Activate RL only after the validated startup/warm-up.
        if t0 >= self.warmup_time-0.5*self.base_dt:
            if not self._rl_active:
                self._rl_active = True
                self._Evd = 0.0
                self._Eid = 0.0
                self._previous_frequency = self._last_sample.controller.frequency
            effective_action = requested_action
            gains = self.gain_adapter.apply_action(effective_action)
            self.sim.controller.set_gains(gains)
        else:
            effective_action = np.zeros(4, dtype=float)
            # Explicitly enforce nominal gains throughout warm-up.
            self.gain_adapter.reset()
            self.sim.controller.set_gains(self.gain_adapter.gains)

        normalized_after = self.gain_adapter.normalized_gain_state().copy()
        gain_motion = normalized_after - normalized_before

        sample = None
        for _ in range(self.base_steps_per_action):
            sample = self.sim.step()

            # Raw RL integrals start only when RL becomes active.
            if self._rl_active:
                self._Evd += sample.controller.vd_error*self.base_dt
                self._Eid += sample.controller.id_error*self.base_dt

            if self.sim.state.time >= self.episode_duration-0.5*self.base_dt:
                break

        assert sample is not None
        self._last_sample = sample
        t1 = float(self.sim.state.time)

        f = sample.controller.frequency
        elapsed = max(t1-t0, self.base_dt)
        rocof = (f-self._previous_frequency)/elapsed
        self._previous_frequency = f

        self._start_events_crossed(t0, t1)
        settle = self._settling_cost(f-60.0)

        # Final episode boundary also terminates the last disturbance window.
        ending = t1 >= self.episode_duration-0.5*self.base_dt
        if ending and self._peak_window_active:
            # Include this final sample before computing the one-shot cost.
            self._peak_abs_df = max(self._peak_abs_df, abs(float(f-60.0)))
            self._peak_abs_rocof = max(self._peak_abs_rocof, abs(float(rocof)))
            self._pending_event_terminal_cost += self._event_terminal_cost()
            self._peak_window_active = False

        # Do not train the agent on startup reward before it has authority.
        if self._rl_active:
            reward, components = self._reward(
                sample, rocof, gain_motion, settle
            )
        else:
            reward = 0.0
            components = dict(
                J_frequency=0.0, J_rocof=0.0, J_settling=0.0,
                J_peak_frequency=0.0, J_peak_rocof=0.0,
                J_voltage=0.0, J_current=0.0, J_gain_motion=0.0,
                J_gain_prior=0.0, J_event_terminal=0.0
            )

        safety = np.array([
            f, sample.measurements.vod, sample.measurements.voq,
            sample.measurements.iod, sample.measurements.ioq,
            sample.measurements.ifd, sample.measurements.ifq,
            sample.controller.md, sample.controller.mq
        ], dtype=float)
        sc = self.safety_cfg
        voltage_mag = float(np.hypot(sample.measurements.vod, sample.measurements.voq))
        output_current_mag = float(np.hypot(sample.measurements.iod, sample.measurements.ioq))
        filter_current_mag = float(np.hypot(sample.measurements.ifd, sample.measurements.ifq))
        modulation_peak = float(max(abs(sample.controller.md), abs(sample.controller.mq)))
        safety_reason = ""
        if not np.all(np.isfinite(safety)):
            safety_reason = "nonfinite"
        elif not (sc.min_frequency_hz <= f <= sc.max_frequency_hz):
            safety_reason = "frequency"
        elif voltage_mag > sc.max_voltage_dq_v:
            safety_reason = "voltage"
        elif max(output_current_mag, filter_current_mag) > sc.max_current_dq_a:
            safety_reason = "current"
        elif modulation_peak > sc.max_modulation_abs:
            safety_reason = "modulation"
        terminated = bool(safety_reason)
        if terminated:
            reward -= sc.terminal_penalty

        truncated = t1 >= self.episode_duration-0.5*self.base_dt
        obs = self._obs(sample, rocof if self._rl_active else 0.0)
        info = self._info(sample, rocof)
        info.update(components)
        info["requested_action"] = requested_action.copy()
        info["effective_action"] = effective_action.copy()
        info["safety_terminated"] = bool(terminated)
        info["safety_reason"] = safety_reason
        info["voltage_dq_magnitude"] = voltage_mag
        info["output_current_dq_magnitude"] = output_current_mag
        info["filter_current_dq_magnitude"] = filter_current_mag
        info["modulation_peak"] = modulation_peak

        return obs, reward, terminated, truncated, info

    def _info(self, sample, rocof):
        g = self.gain_adapter.gains
        return dict(
            time=float(sample.time),
            frequency=float(sample.controller.frequency),
            frequency_deviation=float(sample.controller.frequency-60.0),
            rocof=float(rocof),
            evd=float(sample.controller.vd_error), Evd=float(self._Evd),
            evq=float(sample.controller.vq_error),
            eid=float(sample.controller.id_error), Eid=float(self._Eid),
            eiq=float(sample.controller.iq_error),
            Kpv=float(g.voltage_kp), Kiv=float(g.voltage_ki),
            Kpi=float(g.current_kp), Kii=float(g.current_ki),
            rl_active=bool(self._rl_active),
            settling_active=bool(self._settling_active),
            last_settling_time=float(self._last_settling_time),
            peak_window_active=bool(self._peak_window_active),
            running_peak_abs_df_hz=float(self._peak_abs_df),
            running_peak_abs_rocof_hz_s=float(self._peak_abs_rocof),
            last_event_terminal_cost=float(self._last_event_terminal_cost),
            nominal_load_power=float(sample.nominal_load_power),
            grid_connected=True,
            scenario_split=self.split.value,
            normalized_gains=self.gain_adapter.normalized_gain_state().copy(),
            allowed_loads_w=self.scenario.allowed_loads_w.copy(),
            scenario_events=[
                (float(e.time), float(e.load_power_w))
                for e in self.scenario.events
            ],
        )


if __name__ == "__main__":
    for split in (ScenarioSplit.TRAIN, ScenarioSplit.VALIDATION, ScenarioSplit.TEST):
        env = RandomizedGFMGainTuningEnv(split=split)
        obs, info = env.reset(seed=42)
        print(env.scenario.describe())
        print("split            :", info["scenario_split"])
        print("obs shape        :", obs.shape)
        print("action shape     :", env.action_space.shape)
        print("base steps/action:", env.base_steps_per_action)
        print()

    # Warm-up authority check on TRAIN split.
    env = RandomizedGFMGainTuningEnv(split=ScenarioSplit.TRAIN)
    obs, info = env.reset(seed=42)
    action = np.ones(4, dtype=np.float32)
    for k in range(5):
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"k={k} t={info['time']:.6f} rl={info['rl_active']} "
            f"K={[info['Kpv'],info['Kiv'],info['Kpi'],info['Kii']]} "
            f"reward={reward:.6f}"
        )
