"""Randomized Stage-1 Gymnasium environment for DDPG PI-gain adaptation.

Uses the validated corrected GFM benchmark, but replaces the fixed evaluation
breaker schedule with a new RandomLoadScenario at every reset.

Observation (8):
    [df, RoCoF, evd, Evd, evq, eid, Eid, eiq]

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

Stage 1 is load-only: corrected grid breaker G2 remains closed.
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
from controller_rl_gains import (
    PIGains, GainBounds, RLGainAdapter, RLGainAdaptiveGFMController
)
from training_scenarios import (
    RandomLoadScenario, RandomLoadScenarioConfig, validate_breaker_map
)


@dataclass(frozen=True)
class RewardConfig:
    w_frequency: float = 5.0
    w_rocof: float = 2.0
    w_settling: float = 1.0
    w_voltage: float = 0.5
    w_current: float = 0.25
    w_gain_motion: float = 0.01

    frequency_scale_hz: float = 0.5
    rocof_scale_hz_s: float = 1.0
    voltage_error_scale_v: float = 380.0
    current_error_scale_a: float = 200.0

    settling_band_hz: float = 0.01
    settling_dwell_s: float = 0.10


@dataclass(frozen=True)
class ObservationConfig:
    df_scale_hz: float = 0.5
    rocof_scale_hz_s: float = 1.0
    voltage_error_scale_v: float = 380.0
    voltage_integral_scale_vs: float = 100.0
    current_error_scale_a: float = 200.0
    current_integral_scale_as: float = 100.0
    clip: float = 10.0


class RandomizedGFMGainTuningEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario_config=RandomLoadScenarioConfig(),
        rl_dt=1.0e-3,
        reward_config=RewardConfig(),
        observation_config=ObservationConfig(),
        gain_bounds=GainBounds(),
    ):
        super().__init__()
        validate_breaker_map()

        self.scenario_config = scenario_config
        self.episode_duration = float(scenario_config.episode_duration)
        self.warmup_time = float(scenario_config.warmup_time)
        self.rl_dt = float(rl_dt)
        self.reward_cfg = reward_config
        self.obs_cfg = observation_config
        self.gain_bounds = gain_bounds

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
            -c, c, shape=(8,), dtype=np.float32
        )

        self.gain_adapter = RLGainAdapter(PIGains(), self.gain_bounds)
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

    def _attach_controller(self):
        self.gain_adapter.reset(PIGains())
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
            cost = 0.0
        return cost

    def _reward(self, sample, rocof, effective_action, settling_cost):
        c = self.reward_cfg
        o = sample.controller
        df = o.frequency-60.0
        Jf = (df/c.frequency_scale_hz)**2
        Jr = (rocof/c.rocof_scale_hz_s)**2
        Js = float(settling_cost)
        Jv = (o.vd_error**2+o.vq_error**2)/(c.voltage_error_scale_v**2)
        Ji = (o.id_error**2+o.iq_error**2)/(c.current_error_scale_a**2)
        Jk = float(np.dot(effective_action, effective_action))
        cost = (
            c.w_frequency*Jf + c.w_rocof*Jr + c.w_settling*Js
            + c.w_voltage*Jv + c.w_current*Ji + c.w_gain_motion*Jk
        )
        return -float(cost), dict(
            J_frequency=float(Jf), J_rocof=float(Jr),
            J_settling=float(Js), J_voltage=float(Jv),
            J_current=float(Ji), J_gain_motion=float(Jk)
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Gymnasium's seeded RNG guarantees repeatable episodes.
        self.scenario = RandomLoadScenario.generate(
            rng=self.np_random, config=self.scenario_config, seed=seed
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
            raise ValueError("DDPG action contains NaN/Inf")

        t0 = float(self.sim.state.time)

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
            self.gain_adapter.reset(PIGains())
            self.sim.controller.set_gains(self.gain_adapter.gains)

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

        # Do not train the agent on startup reward before it has authority.
        if self._rl_active:
            reward, components = self._reward(
                sample, rocof, effective_action, settle
            )
        else:
            reward = 0.0
            components = dict(
                J_frequency=0.0, J_rocof=0.0, J_settling=0.0,
                J_voltage=0.0, J_current=0.0, J_gain_motion=0.0
            )

        safety = np.array([
            f, sample.measurements.vod, sample.measurements.voq,
            sample.measurements.iod, sample.measurements.ioq,
            sample.measurements.ifd, sample.measurements.ifq,
            sample.controller.md, sample.controller.mq
        ])
        terminated = not np.all(np.isfinite(safety))
        if terminated:
            reward -= 1e4

        truncated = t1 >= self.episode_duration-0.5*self.base_dt
        obs = self._obs(sample, rocof if self._rl_active else 0.0)
        info = self._info(sample, rocof)
        info.update(components)
        info["requested_action"] = requested_action.copy()
        info["effective_action"] = effective_action.copy()

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
            nominal_load_power=float(sample.nominal_load_power),
            grid_connected=True,
            scenario_events=[
                (float(e.time), float(e.load_power_w))
                for e in self.scenario.events
            ],
        )


if __name__ == "__main__":
    env = RandomizedGFMGainTuningEnv()
    obs, info = env.reset(seed=42)

    print(env.scenario.describe())
    print("\nEnvironment:")
    print("obs shape        :", obs.shape)
    print("action shape     :", env.action_space.shape)
    print("base steps/action:", env.base_steps_per_action)

    # Demonstrate that even a nonzero requested action is ignored in warm-up.
    action = np.ones(4, dtype=np.float32)
    for k in range(5):
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"k={k} t={info['time']:.6f} rl={info['rl_active']} "
            f"K={[info['Kpv'],info['Kiv'],info['Kpi'],info['Kii']]} "
            f"reward={reward:.6f}"
        )
