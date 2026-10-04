"""Measurements for the NEW fixed-topology GFM_scenarios benchmark.

Uses ScenarioPlantOutput from scenario_plant_exact_new.py.

Generated StateSpace_o1 mapping:
    y[0:27]   : ideal-switch port outputs
    y[27:30]  : B3 three-phase voltage
    y[30:33]  : auxiliary three-phase quantity
    y[33:36]  : B3 output current
    y[36]     : DC/battery current
    y[37:40]  : B1/inverter-side current

The controller-frame transform reproduces the generated amplitude-invariant
Clarke transform followed by the "90 degrees behind phase A axis" Park
alignment used in the validated MATLAB benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from scenario_plant_exact_new import ScenarioPlantOutput

SQRT3_OVER_2 = math.sqrt(3.0) / 2.0
TWO_THIRDS = 2.0 / 3.0


@dataclass(frozen=True)
class DQ:
    d: float
    q: float
    zero: float = 0.0


@dataclass(frozen=True)
class ScenarioMeasurements:
    vb3_abc: np.ndarray
    aux_abc: np.ndarray
    ib3_abc: np.ndarray
    ib1_abc: np.ndarray

    vod: float
    voq: float
    iod: float
    ioq: float
    ifd: float
    ifq: float

    aux_d: float
    aux_q: float

    P: float
    Q: float
    S: float

    battery_current: float

    breaker_group_states: np.ndarray
    breaker_group_gates: np.ndarray

    def __post_init__(self):
        for name in ("vb3_abc", "aux_abc", "ib3_abc", "ib1_abc"):
            object.__setattr__(
                self,
                name,
                np.asarray(
                    getattr(self, name),
                    dtype=np.float64,
                ).reshape(3),
            )

        object.__setattr__(
            self,
            "breaker_group_states",
            np.asarray(
                self.breaker_group_states,
                dtype=np.int32,
            ).reshape(7, 3),
        )

        object.__setattr__(
            self,
            "breaker_group_gates",
            np.asarray(
                self.breaker_group_gates,
                dtype=np.int32,
            ).reshape(7, 3),
        )

    @property
    def voltage_magnitude_dq(self):
        return math.hypot(self.vod, self.voq)

    @property
    def output_current_magnitude_dq(self):
        return math.hypot(self.iod, self.ioq)

    @property
    def inverter_current_magnitude_dq(self):
        return math.hypot(self.ifd, self.ifq)


def abc_to_alpha_beta_zero(a, b, c):
    alpha = TWO_THIRDS * (
        a - 0.5 * b - 0.5 * c
    )

    beta = TWO_THIRDS * (
        SQRT3_OVER_2 * b
        - SQRT3_OVER_2 * c
    )

    zero = (a + b + c) / 3.0

    return (
        float(alpha),
        float(beta),
        float(zero),
    )


def alpha_beta_to_dq_90_behind(
    alpha,
    beta,
    theta,
):
    s = math.sin(theta)
    c = math.cos(theta)

    return DQ(
        d=float(alpha * s - beta * c),
        q=float(alpha * c + beta * s),
        zero=0.0,
    )


def abc_to_dq(a, b, c, theta):
    alpha, beta, zero = abc_to_alpha_beta_zero(
        a, b, c
    )

    dq = alpha_beta_to_dq_90_behind(
        alpha,
        beta,
        theta,
    )

    return DQ(
        d=dq.d,
        q=dq.q,
        zero=zero,
    )


def decode_state_space_outputs(y):
    y = np.asarray(
        y,
        dtype=np.float64,
    ).reshape(40)

    return {
        "switch_outputs": y[0:27].copy(),
        "vb3_abc": y[27:30].copy(),
        "aux_abc": y[30:33].copy(),
        "ib3_abc": y[33:36].copy(),
        "battery_current": float(y[36]),
        "ib1_abc": y[37:40].copy(),
    }


def measurement_step(
    plant_output: ScenarioPlantOutput,
    theta: float,
) -> ScenarioMeasurements:

    raw = decode_state_space_outputs(
        plant_output.y
    )

    voltage = abc_to_dq(
        *raw["vb3_abc"],
        theta,
    )

    current_b3 = abc_to_dq(
        *raw["ib3_abc"],
        theta,
    )

    current_b1 = abc_to_dq(
        *raw["ib1_abc"],
        theta,
    )

    auxiliary = abc_to_dq(
        *raw["aux_abc"],
        theta,
    )

    P = 1.5 * (
        voltage.d * current_b3.d
        + voltage.q * current_b3.q
    )

    Q = 1.5 * (
        voltage.q * current_b3.d
        - voltage.d * current_b3.q
    )

    S = math.hypot(P, Q)

    return ScenarioMeasurements(
        vb3_abc=raw["vb3_abc"],
        aux_abc=raw["aux_abc"],
        ib3_abc=raw["ib3_abc"],
        ib1_abc=raw["ib1_abc"],

        vod=voltage.d,
        voq=voltage.q,
        iod=current_b3.d,
        ioq=current_b3.q,
        ifd=current_b1.d,
        ifq=current_b1.q,

        aux_d=auxiliary.d,
        aux_q=auxiliary.q,

        P=float(P),
        Q=float(Q),
        S=float(S),

        battery_current=raw[
            "battery_current"
        ],

        breaker_group_states=(
            plant_output.breaker_group_states
        ),

        breaker_group_gates=(
            plant_output.gate_states[:21]
            .reshape(7, 3)
        ),
    )


def print_measurement_diagnostics(m):
    print("NEW scenario measurement diagnostics")
    print("------------------------------------")
    print("Vabc B3 :", m.vb3_abc)
    print("Iabc B3 :", m.ib3_abc)
    print("Iabc B1 :", m.ib1_abc)
    print("Aux abc :", m.aux_abc)
    print()
    print("Vod, Voq:", m.vod, m.voq)
    print("Iod, Ioq:", m.iod, m.ioq)
    print("Ifd, Ifq:", m.ifd, m.ifq)
    print("P, Q    :", m.P, m.Q)
    print("S       :", m.S)
    print("Idc     :", m.battery_current)
    print(
        "breaker group gates:",
        m.breaker_group_gates[:, 0],
    )
    print("physical breaker states:")
    print(m.breaker_group_states)


if __name__ == "__main__":
    # Mapping/transform smoke test.
    y = np.zeros(40, dtype=np.float64)
    y[27:30] = [
        1.0,
        -0.5,
        -0.5,
    ]

    out = ScenarioPlantOutput(
        y=y,
        switch_states=np.zeros(
            27,
            dtype=np.int32,
        ),
        gate_states_used=np.zeros(
            27,
            dtype=np.int32,
        ),
        loops_remaining=20,
    )

    m = measurement_step(
        out,
        theta=math.pi / 2.0,
    )

    print_measurement_diagnostics(m)
