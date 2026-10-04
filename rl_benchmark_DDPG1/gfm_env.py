"""Gymnasium environment for Stage-1 DDPG PI-gain adaptation.

Plant/controller base sample:  Ts = 10 us (100 kHz)
Default RL decision period:    T_RL = 1 ms = 100 base samples

Observation (8):
    [df, RoCoF, evd, Evd, evq, eid, Eid, eiq]

Action (4), normalized to [-1,1]:
    [dKpv, dKiv, dKpi, dKii]

The action is passed through RLGainAdapter, which applies bounded incremental
gain changes.  The four gains are held constant during the following RL
decision interval.

Reward objectives:
    frequency deviation + RoCoF + settling-time proxy
    + smaller voltage/current regulation and gain-motion penalties.

This first environment deliberately wraps the validated deterministic scenario.
Randomized training scenarios should be added only after this environment passes
the zero-action regression and reward/observation sanity tests.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:
    raise ImportError(
        "gfm_env.py requires Gymnasium. Install with: pip install gymnasium"
    ) from exc

from benchmark_scenarios_rl import MatlabScenarioBenchmark
from controller_rl_gains import (
    PIGains,
    GainBounds,
    RLGainAdapter,
    RLGainAdaptiveGFMController,
)


@dataclass(frozen=True)
class RewardConfig:
    # Primary objectives
    w_frequency: float = 5.0
    w_rocof: float = 2.0
    w_settling: float = 1.0

    # Secondary controller-quality objectives
    w_voltage: float = 0.5
    w_current: float = 0.25
    w_gain_motion: float = 0.01

    # Normalization scales
    frequency_scale_hz: float = 0.5
    rocof_scale_hz_s: float = 15.0
    voltage_error_scale_v: float = 380.0
    current_error_scale_a: float = 200.0

    # Settling-time definition
    settling_band_hz: float = 0.01
    settling_dwell_s: float = 0.10


@dataclass(frozen=True)
class ObservationConfig:
    # Scales map the physical observations to roughly order-one values.
    df_scale_hz: float = 0.5
    rocof_scale_hz_s: float = 15.0
    voltage_error_scale_v: float = 380.0
    voltage_integral_scale_vs: float = 100.0
    current_error_scale_a: float = 200.0
    current_integral_scale_as: float = 100.0
    clip: float = 10.0


class GFMGainTuningEnv(gym.Env):
    """Stage-1 DDPG environment for adapting four cascaded PI gains."""

    metadata = {"render_modes": []}

    # Known disturbance boundaries of the current validated benchmark.
    EVENT_TIMES = np.array([2.0, 3.5, 5.0, 6.5, 9.0, 11.0, 12.0])

    def __init__(
        self,
        episode_duration=15.0,
        rl_dt=1.0e-3,
        reward_config=RewardConfig(),
        observation_config=ObservationConfig(),
        gain_bounds=GainBounds(),
    ):
        super().__init__()

        self.episode_duration = float(episode_duration)
        self.rl_dt = float(rl_dt)
        self.reward_cfg = reward_config
        self.obs_cfg = observation_config
        self.gain_bounds = gain_bounds

        # Construct once to obtain the validated 10-us base step.
        self.sim = MatlabScenarioBenchmark()
        self.base_dt = float(self.sim.params.dt)

        ratio = self.rl_dt / self.base_dt
        self.base_steps_per_action = int(round(ratio))
        if self.base_steps_per_action < 1:
            raise ValueError("rl_dt must be >= the 10-us plant/controller step")
        if not np.isclose(
            self.base_steps_per_action * self.base_dt,
            self.rl_dt,
            rtol=0.0,
            atol=1e-15,
        ):
            raise ValueError("rl_dt must be an integer multiple of the base dt")

        # DDPG action is always normalized.
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # Observations are normalized but allowed beyond +/-1 during large
        # disturbances; final clipping is controlled by ObservationConfig.
        c = float(self.obs_cfg.clip)
        self.observation_space = spaces.Box(
            low=-c, high=c, shape=(8,), dtype=np.float32
        )

        self.gain_adapter = RLGainAdapter(
            initial_gains=PIGains(),
            bounds=self.gain_bounds,
        )

        self._Evd = 0.0
        self._Eid = 0.0
        self._previous_frequency = 60.0
        self._last_sample = None

        # Settling-time bookkeeping.
        self._event_index = 0
        self._settling_active = False
        self._settling_elapsed = 0.0
        self._inside_band_elapsed = 0.0
        self._last_settling_time = math.nan

    def _attach_nominal_rl_controller(self):
        """Guarantee the benchmark uses the RL-ready controller at nominal gains."""
        self.gain_adapter.reset(PIGains())
        self.sim.controller = RLGainAdaptiveGFMController(
            params=self.sim.params,
            gains=self.gain_adapter.gains,
        )

    def _normalized_observation(self, sample, rocof):
        c = self.obs_cfg
        out = sample.controller

        df = out.frequency - 60.0

        obs = np.array(
            [
                df / c.df_scale_hz,
                rocof / c.rocof_scale_hz_s,
                out.vd_error / c.voltage_error_scale_v,
                self._Evd / c.voltage_integral_scale_vs,
                out.vq_error / c.voltage_error_scale_v,
                out.id_error / c.current_error_scale_a,
                self._Eid / c.current_integral_scale_as,
                out.iq_error / c.current_error_scale_a,
            ],
            dtype=np.float64,
        )

        obs = np.clip(obs, -c.clip, c.clip)
        return obs.astype(np.float32)

    def _start_events_crossed(self, t0, t1):
        """Start settling tracking for any event crossed in (t0,t1]."""
        while (
            self._event_index < len(self.EVENT_TIMES)
            and self.EVENT_TIMES[self._event_index] <= t1 + 1e-12
        ):
            te = float(self.EVENT_TIMES[self._event_index])
            if te > t0 + 1e-12:
                self._settling_active = True
                self._settling_elapsed = 0.0
                self._inside_band_elapsed = 0.0
                self._last_settling_time = math.nan
            self._event_index += 1

    def _update_settling_tracker(self, df):
        if not self._settling_active:
            return 0.0

        cfg = self.reward_cfg
        self._settling_elapsed += self.rl_dt

        if abs(df) <= cfg.settling_band_hz:
            self._inside_band_elapsed += self.rl_dt
        else:
            self._inside_band_elapsed = 0.0

        # Per-step settling-time proxy: every RL interval spent in an unsettled
        # transient incurs a cost. This gives DDPG dense feedback.
        settling_cost = 1.0

        if self._inside_band_elapsed >= cfg.settling_dwell_s:
            self._settling_active = False
            self._last_settling_time = max(
                0.0,
                self._settling_elapsed - cfg.settling_dwell_s,
            )
            settling_cost = 0.0

        return settling_cost

    def _reward(self, sample, rocof, action, settling_cost):
        cfg = self.reward_cfg
        out = sample.controller

        df = out.frequency - 60.0

        Jf = (df / cfg.frequency_scale_hz) ** 2
        Jr = (rocof / cfg.rocof_scale_hz_s) ** 2
        Js = float(settling_cost)

        Jv = (
            out.vd_error**2 + out.vq_error**2
        ) / (cfg.voltage_error_scale_v**2)

        Ji = (
            out.id_error**2 + out.iq_error**2
        ) / (cfg.current_error_scale_a**2)

        # Penalize requested action movement. The adapter itself rate-limits the
        # physical gain increments.
        Jk = float(np.dot(action, action))

        cost = (
            cfg.w_frequency * Jf
            + cfg.w_rocof * Jr
            + cfg.w_settling * Js
            + cfg.w_voltage * Jv
            + cfg.w_current * Ji
            + cfg.w_gain_motion * Jk
        )

        return -float(cost), {
            "J_frequency": float(Jf),
            "J_rocof": float(Jr),
            "J_settling": float(Js),
            "J_voltage": float(Jv),
            "J_current": float(Ji),
            "J_gain_motion": float(Jk),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.sim = MatlabScenarioBenchmark()
        self._attach_nominal_rl_controller()

        self._Evd = 0.0
        self._Eid = 0.0
        self._previous_frequency = 60.0
        self._last_sample = None

        self._event_index = 0
        self._settling_active = False
        self._settling_elapsed = 0.0
        self._inside_band_elapsed = 0.0
        self._last_settling_time = math.nan

        # Obtain a genuine current observation without advancing time by using
        # one base sample only after reset. This mirrors the benchmark's normal
        # execution path and makes reset deterministic.
        sample = self.sim.step()
        self._last_sample = sample

        # Integrate raw (unweighted) errors for the RL observation.
        self._Evd += sample.controller.vd_error * self.base_dt
        self._Eid += sample.controller.id_error * self.base_dt

        f = sample.controller.frequency
        self._previous_frequency = f
        obs = self._normalized_observation(sample, rocof=0.0)

        info = self._info(sample, rocof=0.0)
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(4)
        if not np.all(np.isfinite(action)):
            raise ValueError("DDPG action contains NaN/Inf")
        action = np.clip(action, -1.0, 1.0)

        # One DDPG decision -> one bounded gain increment.
        gains = self.gain_adapter.apply_action(action)
        self.sim.controller.set_gains(gains)

        t0 = float(self.sim.state.time)
        sample = None

        # Hold gains for the complete 1-ms RL interval while the validated
        # electrical/controller/PWM model continues at 10 us.
        for _ in range(self.base_steps_per_action):
            sample = self.sim.step()

            # Raw error integrals for the RL state (not Ki-weighted PI memory).
            self._Evd += sample.controller.vd_error * self.base_dt
            self._Eid += sample.controller.id_error * self.base_dt

            if self.sim.state.time >= self.episode_duration - 0.5*self.base_dt:
                break

        assert sample is not None
        self._last_sample = sample
        t1 = float(self.sim.state.time)

        # RoCoF is deliberately evaluated over the RL interval, not 10 us.
        frequency = sample.controller.frequency
        elapsed = max(t1 - t0, self.base_dt)
        rocof = (frequency - self._previous_frequency) / elapsed
        self._previous_frequency = frequency

        self._start_events_crossed(t0, t1)
        df = frequency - 60.0
        settling_cost = self._update_settling_tracker(df)

        reward, components = self._reward(
            sample, rocof, action, settling_cost
        )

        terminated = False
        truncated = t1 >= self.episode_duration - 0.5*self.base_dt

        # Basic numerical safety termination.
        safety_values = np.array(
            [
                frequency,
                sample.measurements.vod,
                sample.measurements.voq,
                sample.measurements.iod,
                sample.measurements.ioq,
                sample.measurements.ifd,
                sample.measurements.ifq,
                sample.controller.md,
                sample.controller.mq,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(safety_values)):
            terminated = True
            reward -= 1.0e4

        obs = self._normalized_observation(sample, rocof)
        info = self._info(sample, rocof)
        info.update(components)

        return obs, reward, terminated, truncated, info

    def _info(self, sample, rocof):
        g = self.gain_adapter.gains
        return {
            "time": float(sample.time),
            "frequency": float(sample.controller.frequency),
            "frequency_deviation": float(sample.controller.frequency - 60.0),
            "rocof": float(rocof),
            "evd": float(sample.controller.vd_error),
            "Evd": float(self._Evd),
            "evq": float(sample.controller.vq_error),
            "eid": float(sample.controller.id_error),
            "Eid": float(self._Eid),
            "eiq": float(sample.controller.iq_error),
            "Kpv": float(g.voltage_kp),
            "Kiv": float(g.voltage_ki),
            "Kpi": float(g.current_kp),
            "Kii": float(g.current_ki),
            "settling_active": bool(self._settling_active),
            "last_settling_time": float(self._last_settling_time),
            "nominal_load_power": float(sample.nominal_load_power),
            "grid_connected": bool(sample.nominal_grid_connected),
        }


if __name__ == "__main__":
    env = GFMGainTuningEnv(episode_duration=0.02, rl_dt=1e-3)
    obs, info = env.reset()

    print("Stage-1 GFM DDPG environment smoke test")
    print("---------------------------------------")
    print("observation shape:", obs.shape)
    print("action shape     :", env.action_space.shape)
    print("base dt          :", env.base_dt)
    print("RL dt            :", env.rl_dt)
    print("base steps/action:", env.base_steps_per_action)
    print("initial obs      :", obs)
    print("initial gains    :", [info[k] for k in ("Kpv","Kiv","Kpi","Kii")])

    total_reward = 0.0
    for k in range(5):
        obs, reward, terminated, truncated, info = env.step(
            np.zeros(4, dtype=np.float32)
        )
        total_reward += reward
        print(
            f"k={k:2d} t={info['time']:.6f} "
            f"f={info['frequency']:.6f} "
            f"RoCoF={info['rocof']:.6f} "
            f"reward={reward:.6f}"
        )
        if terminated or truncated:
            break

    print("total reward:", total_reward)
