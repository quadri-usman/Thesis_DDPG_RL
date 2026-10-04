"""Stage-1 load scenarios with explicit TRAIN / VALIDATION / TEST splits.

Physical SPS load configurations available:
    50.0, 52.5, 55.0, 60.0, 70.0 kW

Experimental split
------------------
TRAIN:
    50, 55, 70 kW
    Used for actor/critic learning.

VALIDATION:
    50, 55, 70 kW
    Same load magnitudes as training, but unseen seeds/timings/sequences.
    Used for checkpoint/model selection.

TEST (unseen disturbance magnitudes):
    50, 52.5, 60 kW
    The 52.5- and 60-kW disturbance magnitudes are withheld from training.
    50 kW is retained as the validated initial/restoration operating point.

All Stage-1 scenarios:
    * start at 50 kW;
    * remain grid-connected (G2 = 1);
    * use only physically generated SPS breaker configurations;
    * randomize event times after warm-up;
    * prohibit consecutive identical loads.

The untouched 15-s MATLAB benchmark remains a separate final benchmark and is
NOT used to train or select the DDPG policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import numpy as np


# ---------------------------------------------------------------------------
# Physical load configurations in the corrected generated SPS model
# ---------------------------------------------------------------------------

ALL_LOADS_W = np.array(
    [50_000.0, 52_500.0, 55_000.0, 60_000.0, 70_000.0],
    dtype=np.float64,
)

# Backward-compatible alias.
LOAD_LEVELS_W = ALL_LOADS_W

# Group order: [G0,G1,G2,G3,G4,G5,G6].
# G2 is the grid breaker and remains CLOSED (=1) for Stage-1 load experiments.
LOAD_TO_BREAKERS = {
    50_000: np.array([1, 0, 1, 0, 0, 0, 1], dtype=np.int8),
    52_500: np.array([0, 1, 1, 0, 0, 0, 0], dtype=np.int8),
    55_000: np.array([0, 0, 1, 1, 0, 0, 0], dtype=np.int8),
    60_000: np.array([0, 0, 1, 0, 1, 0, 0], dtype=np.int8),
    70_000: np.array([0, 0, 1, 0, 0, 1, 0], dtype=np.int8),
}


# ---------------------------------------------------------------------------
# Experimental load split
# ---------------------------------------------------------------------------

TRAIN_LOADS_W = np.array(
    [50_000.0, 55_000.0, 70_000.0],
    dtype=np.float64,
)

VALIDATION_LOADS_W = TRAIN_LOADS_W.copy()

TEST_LOADS_W = np.array(
    [50_000.0, 52_500.0, 60_000.0],
    dtype=np.float64,
)

UNSEEN_TEST_LOADS_W = np.array(
    [52_500.0, 60_000.0],
    dtype=np.float64,
)


class ScenarioSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


def loads_for_split(split):
    split = ScenarioSplit(split)

    if split is ScenarioSplit.TRAIN:
        return TRAIN_LOADS_W.copy()
    if split is ScenarioSplit.VALIDATION:
        return VALIDATION_LOADS_W.copy()
    if split is ScenarioSplit.TEST:
        return TEST_LOADS_W.copy()

    raise ValueError(f"unsupported scenario split: {split}")


@dataclass(frozen=True)
class LoadEvent:
    time: float
    load_power_w: float

    def __post_init__(self):
        if self.time < 0.0:
            raise ValueError("event time must be nonnegative")

        if int(round(self.load_power_w)) not in LOAD_TO_BREAKERS:
            raise ValueError(
                f"unsupported physical load {self.load_power_w}; "
                f"allowed={ALL_LOADS_W.tolist()}"
            )


@dataclass(frozen=True)
class RandomLoadScenarioConfig:
    episode_duration: float = 8.0
    warmup_time: float = 1.5

    min_hold_time: float = 1.0
    max_hold_time: float = 2.0

    min_events: int = 3
    max_events: int = 5

    terminal_margin: float = 0.25
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
    """One immutable randomized schedule belonging to a named data split."""

    def __init__(
        self,
        events,
        config,
        split=ScenarioSplit.TRAIN,
        seed=None,
    ):
        self.config = config
        self.split = ScenarioSplit(split)
        self.seed = seed
        self.events = tuple(events)

        if not self.events:
            raise ValueError("scenario must contain at least the t=0 event")
        if abs(self.events[0].time) > 1e-15:
            raise ValueError("first event must be at t=0")
        if self.events[0].load_power_w != 50_000.0:
            raise ValueError("scenario must start at 50 kW")

        times = np.asarray([e.time for e in self.events], dtype=float)
        if np.any(np.diff(times) <= 0):
            raise ValueError("event times must be strictly increasing")

        allowed = set(int(x) for x in loads_for_split(self.split))
        for e in self.events:
            if int(round(e.load_power_w)) not in allowed:
                raise ValueError(
                    f"{e.load_power_w} W is not allowed in split "
                    f"{self.split.value}; allowed={sorted(allowed)}"
                )

    @classmethod
    def generate(
        cls,
        rng=None,
        config=RandomLoadScenarioConfig(),
        seed=None,
        split=ScenarioSplit.TRAIN,
    ):
        config.validate()
        split = ScenarioSplit(split)
        allowed_loads = loads_for_split(split)

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

            choices = allowed_loads[
                allowed_loads != current_load
            ]

            if choices.size == 0:
                raise RuntimeError(
                    f"split {split.value} has no alternative load "
                    f"to current load {current_load}"
                )

            new_load = float(rng.choice(choices))
            events.append(LoadEvent(t, new_load))
            current_load = new_load

        if config.restore_50kw_at_end and current_load != 50_000.0:
            latest = config.episode_duration - config.terminal_margin
            candidate = min(
                latest,
                events[-1].time + config.min_hold_time,
            )

            if candidate > events[-1].time + 1e-12:
                events.append(LoadEvent(candidate, 50_000.0))

        return cls(
            events=events,
            config=config,
            split=split,
            seed=seed,
        )

    @property
    def event_times(self):
        # t=0 is initialization, not a disturbance event.
        return np.asarray(
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

    @property
    def allowed_loads_w(self):
        return loads_for_split(self.split)

    def event_index_at(self, t):
        times = np.asarray(
            [e.time for e in self.events],
            dtype=np.float64,
        )
        return int(np.searchsorted(times, t, side="right") - 1)

    def load_power(self, t):
        i = self.event_index_at(float(t))
        return float(self.events[i].load_power_w)

    def breaker_groups(self, t):
        load = int(round(self.load_power(t)))
        return LOAD_TO_BREAKERS[load].copy()

    def grid_connected(self, t):
        return True

    def describe(self):
        lines = [
            f"Random Stage-1 {self.split.value.upper()} scenario",
            "-" * 45,
            f"duration       : {self.episode_duration:.3f} s",
            f"warm-up        : {self.warmup_time:.3f} s",
            "grid           : connected throughout",
            "allowed loads  : "
            + ", ".join(
                f"{x/1000:g}" for x in self.allowed_loads_w
            )
            + " kW",
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


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def generate_training_scenario(
    rng=None,
    config=RandomLoadScenarioConfig(),
    seed=None,
):
    return RandomLoadScenario.generate(
        rng=rng,
        config=config,
        seed=seed,
        split=ScenarioSplit.TRAIN,
    )


def generate_validation_scenario(
    rng=None,
    config=RandomLoadScenarioConfig(),
    seed=None,
):
    return RandomLoadScenario.generate(
        rng=rng,
        config=config,
        seed=seed,
        split=ScenarioSplit.VALIDATION,
    )


def generate_test_scenario(
    rng=None,
    config=RandomLoadScenarioConfig(),
    seed=None,
):
    return RandomLoadScenario.generate(
        rng=rng,
        config=config,
        seed=seed,
        split=ScenarioSplit.TEST,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_breaker_map():
    expected = set(int(x) for x in ALL_LOADS_W)
    actual = set(LOAD_TO_BREAKERS.keys())

    if expected != actual:
        raise AssertionError("physical load/breaker mapping is incomplete")

    for load, groups in LOAD_TO_BREAKERS.items():
        groups = np.asarray(groups)

        if groups.shape != (7,):
            raise AssertionError(
                f"{load}: breaker vector must have 7 groups"
            )

        if not np.all(np.isin(groups, [0, 1])):
            raise AssertionError(
                f"{load}: breaker commands must be binary"
            )

        if groups[2] != 1:
            raise AssertionError(
                f"{load}: G2 grid breaker must remain CLOSED "
                "for Stage-1 load scenarios"
            )

    expected_initial = np.array(
        [1, 0, 1, 0, 0, 0, 1],
        dtype=np.int8,
    )

    if not np.array_equal(
        LOAD_TO_BREAKERS[50_000],
        expected_initial,
    ):
        raise AssertionError(
            "50-kW initial topology is not the validated one"
        )

    return True


def validate_experimental_split():
    train_disturbances = set(
        int(x) for x in TRAIN_LOADS_W if x != 50_000
    )
    unseen_test = set(int(x) for x in UNSEEN_TEST_LOADS_W)

    if train_disturbances & unseen_test:
        raise AssertionError(
            "unseen test loads leaked into the training set"
        )

    if 50_000 not in set(int(x) for x in TRAIN_LOADS_W):
        raise AssertionError("training must include the 50-kW initial point")

    if 50_000 not in set(int(x) for x in TEST_LOADS_W):
        raise AssertionError("test must include the 50-kW initial point")

    return True


if __name__ == "__main__":
    validate_breaker_map()
    validate_experimental_split()

    print("Breaker-map validation   : PASS")
    print("Experimental split check : PASS\n")

    print("TRAIN loads      :", TRAIN_LOADS_W / 1000, "kW")
    print("VALIDATION loads :", VALIDATION_LOADS_W / 1000, "kW")
    print("TEST loads       :", TEST_LOADS_W / 1000, "kW")
    print("UNSEEN test only :", UNSEEN_TEST_LOADS_W / 1000, "kW")
    print()

    for split in (
        ScenarioSplit.TRAIN,
        ScenarioSplit.VALIDATION,
        ScenarioSplit.TEST,
    ):
        scenario = RandomLoadScenario.generate(
            seed=42,
            split=split,
        )
        print(scenario.describe())
        print()

    # Reproducibility within each split.
    for split in ScenarioSplit:
        a = RandomLoadScenario.generate(seed=123, split=split)
        b = RandomLoadScenario.generate(seed=123, split=split)

        assert [
            (e.time, e.load_power_w) for e in a.events
        ] == [
            (e.time, e.load_power_w) for e in b.events
        ]

    # Explicit leakage test: unseen magnitudes must never occur in many
    # generated TRAIN episodes.
    unseen = set(int(x) for x in UNSEEN_TEST_LOADS_W)

    for seed in range(100):
        s = generate_training_scenario(seed=seed)
        used = set(int(round(e.load_power_w)) for e in s.events)

        if used & unseen:
            raise AssertionError(
                f"test-load leakage in training seed {seed}: {used & unseen}"
            )

    print("Seed reproducibility     : PASS")
    print("100-seed leakage check   : PASS")
