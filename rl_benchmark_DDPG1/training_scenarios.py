"""Randomized load-disturbance scenarios for Stage-1 DDPG training.

The corrected generated SPS model has discrete physically-realized load
configurations.  This module randomizes ONLY among those configurations:

    50.0, 52.5, 55.0, 60.0, 70.0 kW

The grid breaker (G2) remains CLOSED throughout Stage-1 load-only training.

Every episode:
  * starts at the validated 50-kW, grid-connected topology;
  * preserves that topology through a startup/warm-up interval;
  * randomizes load-event times after warm-up;
  * prevents consecutive events from selecting the same load;
  * can optionally restore 50 kW at the final event.

This module does not modify plant parameters or synthesize arbitrary loads.
It only selects among breaker configurations present in the generated model.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# ---------------------------------------------------------------------------
# Corrected generated breaker mapping
# ---------------------------------------------------------------------------
# Group order: [G0,G1,G2,G3,G4,G5,G6]
#
# G2 = grid breaker and MUST remain 1 for Stage-1 load-only training.
#
# The load selections follow the corrected generated scenario topology:
# 50 kW   -> G0 + G6, with grid G2 also closed
# 52.5 kW -> G1,      with grid G2 closed
# 55 kW   -> G3,      with grid G2 closed
# 60 kW   -> G4,      with grid G2 closed
# 70 kW   -> G5,      with grid G2 closed

LOAD_LEVELS_W = np.array(
    [50_000.0, 52_500.0, 55_000.0, 60_000.0, 70_000.0],
    dtype=np.float64,
)

LOAD_TO_BREAKERS = {
    50_000: np.array([1, 0, 1, 0, 0, 0, 1], dtype=np.int8),
    52_500: np.array([0, 1, 1, 0, 0, 0, 0], dtype=np.int8),
    55_000: np.array([0, 0, 1, 1, 0, 0, 0], dtype=np.int8),
    60_000: np.array([0, 0, 1, 0, 1, 0, 0], dtype=np.int8),
    70_000: np.array([0, 0, 1, 0, 0, 1, 0], dtype=np.int8),
}


@dataclass(frozen=True)
class LoadEvent:
    time: float
    load_power_w: float

    def __post_init__(self):
        if self.time < 0.0:
            raise ValueError("event time must be nonnegative")
        if int(round(self.load_power_w)) not in LOAD_TO_BREAKERS:
            raise ValueError(
                f"unsupported load {self.load_power_w}; "
                f"allowed={LOAD_LEVELS_W.tolist()}"
            )


@dataclass(frozen=True)
class RandomLoadScenarioConfig:
    episode_duration: float = 8.0

    # DDPG remains inactive during startup. First randomized event occurs after
    # this point plus a random hold interval.
    warmup_time: float = 1.5

    # Random duration between successive load changes.
    min_hold_time: float = 1.0
    max_hold_time: float = 2.0

    # Number of randomized load changes after startup.
    min_events: int = 3
    max_events: int = 5

    # Keep final event away from the exact episode boundary.
    terminal_margin: float = 0.25

    # Useful for episodes that should end at the nominal operating point.
    restore_50kw_at_end: bool = False

    def validate(self):
        if self.episode_duration <= 0:
            raise ValueError("episode_duration must be positive")
        if self.warmup_time < 0:
            raise ValueError("warmup_time must be nonnegative")
        if not (0 < self.min_hold_time <= self.max_hold_time):
            raise ValueError("invalid hold-time range")
        if self.min_events < 1 or self.max_events < self.min_events:
            raise ValueError("invalid event-count range")
        if self.warmup_time >= self.episode_duration:
            raise ValueError("warmup_time must be less than episode_duration")
        if self.terminal_margin < 0:
            raise ValueError("terminal_margin must be nonnegative")


class RandomLoadScenario:
    """One immutable randomized load schedule."""

    def __init__(self, events, config, seed=None):
        self.config = config
        self.seed = seed
        self.events = tuple(events)

        if not self.events:
            raise ValueError("scenario must contain at least the t=0 event")
        if abs(self.events[0].time) > 1e-15:
            raise ValueError("first event must be at t=0")
        if self.events[0].load_power_w != 50_000.0:
            raise ValueError("scenario must start at 50 kW")

        times = np.array([e.time for e in self.events])
        if np.any(np.diff(times) <= 0):
            raise ValueError("event times must be strictly increasing")

    @classmethod
    def generate(cls, rng=None, config=RandomLoadScenarioConfig(), seed=None):
        config.validate()

        if rng is None:
            rng = np.random.default_rng(seed)

        n_requested = int(
            rng.integers(config.min_events, config.max_events + 1)
        )

        events = [LoadEvent(0.0, 50_000.0)]
        t = float(config.warmup_time)
        current_load = 50_000.0

        for _ in range(n_requested):
            t += float(
                rng.uniform(config.min_hold_time, config.max_hold_time)
            )

            if t > config.episode_duration - config.terminal_margin:
                break

            choices = LOAD_LEVELS_W[
                LOAD_LEVELS_W != current_load
            ]
            new_load = float(rng.choice(choices))

            events.append(LoadEvent(t, new_load))
            current_load = new_load

        # If random timing produced too few usable events, do not force an event
        # at an invalid/too-close time. The generated schedule remains valid.
        if config.restore_50kw_at_end and current_load != 50_000.0:
            latest = config.episode_duration - config.terminal_margin
            candidate = min(
                latest,
                events[-1].time + config.min_hold_time,
            )
            if candidate > events[-1].time + 1e-12:
                events.append(LoadEvent(candidate, 50_000.0))

        return cls(events=events, config=config, seed=seed)

    @property
    def event_times(self):
        # t=0 is initialization, not a disturbance event.
        return np.array(
            [e.time for e in self.events[1:]],
            dtype=np.float64,
        )

    @property
    def disturbance_events(self):
        return self.events[1:]

    @property
    def warmup_time(self):
        return self.config.warmup_time

    @property
    def episode_duration(self):
        return self.config.episode_duration

    def event_index_at(self, t):
        times = np.array([e.time for e in self.events])
        return int(np.searchsorted(times, t, side="right") - 1)

    def load_power(self, t):
        i = self.event_index_at(float(t))
        return float(self.events[i].load_power_w)

    def breaker_groups(self, t):
        load = int(round(self.load_power(t)))
        return LOAD_TO_BREAKERS[load].copy()

    def grid_connected(self, t):
        # Stage 1 randomized training is deliberately load-only.
        return True

    def describe(self):
        lines = [
            "Random Stage-1 load scenario",
            "----------------------------",
            f"duration : {self.episode_duration:.3f} s",
            f"warm-up  : {self.warmup_time:.3f} s",
            "grid     : connected throughout",
            "",
            "events:",
        ]
        for e in self.events:
            g = self.breaker_groups(e.time)
            lines.append(
                f"  t={e.time:8.4f} s  "
                f"load={e.load_power_w/1000:5.1f} kW  "
                f"groups={g.tolist()}"
            )
        return "\n".join(lines)


def validate_breaker_map():
    """Fail loudly if the training topology violates Stage-1 assumptions."""
    expected = set(int(x) for x in LOAD_LEVELS_W)
    actual = set(LOAD_TO_BREAKERS.keys())
    if expected != actual:
        raise AssertionError("load/breaker mapping is incomplete")

    for load, groups in LOAD_TO_BREAKERS.items():
        groups = np.asarray(groups)
        if groups.shape != (7,):
            raise AssertionError(f"{load}: breaker vector must have 7 groups")
        if not np.all(np.isin(groups, [0, 1])):
            raise AssertionError(f"{load}: breaker commands must be binary")
        if groups[2] != 1:
            raise AssertionError(
                f"{load}: G2 grid breaker must stay CLOSED in Stage 1"
            )

    if not np.array_equal(
        LOAD_TO_BREAKERS[50_000],
        np.array([1, 0, 1, 0, 0, 0, 1], dtype=np.int8),
    ):
        raise AssertionError("50-kW initial topology is not the validated one")

    return True


if __name__ == "__main__":
    validate_breaker_map()

    print("Breaker-map validation: PASS\n")

    # Reproducibility demonstration.
    for seed in (1, 2, 3):
        scenario = RandomLoadScenario.generate(seed=seed)
        print(scenario.describe())
        print()

    # Same seed must generate the same schedule.
    a = RandomLoadScenario.generate(seed=123)
    b = RandomLoadScenario.generate(seed=123)

    assert [
        (e.time, e.load_power_w) for e in a.events
    ] == [
        (e.time, e.load_power_w) for e in b.events
    ]

    print("Seed reproducibility: PASS")
