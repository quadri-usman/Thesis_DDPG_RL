"""Initialization diagnostic for the NEW corrected GFM_scenarios SPS model.

Uses:
    scenario_parameters_new.py
    scenario_plant_exact_new.py
    scenario_measurements_new.py

The purpose is to test whether the regenerated MATLAB/Simulink SPS model,
including the corrected initially-closed grid breaker, now reproduces the
MATLAB electrical operating point at t=0 before controller integration.

MATLAB reference at t=0:
    Vod = 188.8 V
    Voq = -155.3 V
    Iod = -3.252 A
    Ioq = -3.402 A
    Ifd = -0.2735 A
    Ifq = 0.2249 A
    P = -128.9 W
    Q = 1721 var
"""

from __future__ import annotations

import math
import numpy as np

from scenario_parameters_new import (
    C, D, X0, N_INPUT
)
from scenario_plant_exact_new import (
    GeneratedScenarioElectricalPlantExactNew,
    ScenarioPlantInput,
    INITIAL_BREAKER_GROUPS,
    INITIAL_BREAKER_GATES,
)
from scenario_measurements_new import (
    abc_to_dq,
    decode_state_space_outputs,
)


GRID_LL_RMS = 13_800.0
GRID_FREQUENCY = 60.0
GRID_PHASE_DEG = 0.0
VDC_T0 = 0.0


MATLAB_TARGET = {
    "vod": 188.8,
    "voq": -155.3,
    "iod": -3.252,
    "ioq": -3.402,
    "ifd": -0.2735,
    "ifq": 0.2249,
    "P": -128.9,
    "Q": 1721.0,
}


def source_abc(t=0.0):
    peak = (
        GRID_LL_RMS
        / math.sqrt(3.0)
        * math.sqrt(2.0)
    )

    theta = (
        2.0 * math.pi
        * GRID_FREQUENCY
        * t
        + math.radians(GRID_PHASE_DEG)
    )

    return np.array(
        [
            peak * math.sin(theta),
            peak * math.sin(
                theta - 2.0 * math.pi / 3.0
            ),
            peak * math.sin(
                theta + 2.0 * math.pi / 3.0
            ),
        ],
        dtype=np.float64,
    )


def make_u(
    t=0.0,
    vdc=VDC_T0,
):
    u = np.zeros(
        N_INPUT,
        dtype=np.float64,
    )

    u[27] = float(vdc)
    u[28:31] = source_abc(t)

    return u


def calculate_metrics(
    y,
    theta=0.0,
):
    raw = decode_state_space_outputs(y)

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

    P = 1.5 * (
        voltage.d * current_b3.d
        + voltage.q * current_b3.q
    )

    Q = 1.5 * (
        voltage.q * current_b3.d
        - voltage.d * current_b3.q
    )

    return raw, {
        "vod": voltage.d,
        "voq": voltage.q,
        "iod": current_b3.d,
        "ioq": current_b3.q,
        "ifd": current_b1.d,
        "ifq": current_b1.q,
        "P": float(P),
        "Q": float(Q),
    }


def target_error(metrics):
    return {
        name: (
            metrics[name]
            - MATLAB_TARGET[name]
        )
        for name in MATLAB_TARGET
    }


def print_case(
    name,
    y,
    theta=0.0,
):
    raw, metrics = calculate_metrics(
        y,
        theta=theta,
    )

    err = target_error(metrics)

    print("\n" + "=" * 88)
    print(name)
    print("=" * 88)

    print("y[27:40]:")
    print(
        np.array2string(
            np.asarray(y[27:40]),
            precision=12,
            suppress_small=False,
        )
    )

    print("\nVabc B3:", raw["vb3_abc"])
    print("Aux abc :", raw["aux_abc"])
    print("Iabc B3:", raw["ib3_abc"])
    print(
        "Idc     :",
        raw["battery_current"],
    )
    print("Iabc B1:", raw["ib1_abc"])

    print("\nController-frame values:")
    for key in (
        "vod", "voq",
        "iod", "ioq",
        "ifd", "ifq",
        "P", "Q",
    ):
        print(
            f"{key:<5} = "
            f"{metrics[key]: .12f}"
            f"    target="
            f"{MATLAB_TARGET[key]: .12f}"
            f"    error="
            f"{err[key]: .12f}"
        )

    rms_error = math.sqrt(
        sum(
            float(v) ** 2
            for v in err.values()
        )
        / len(err)
    )

    print(
        "\nUnscaled target RMS error:",
        rms_error,
    )

    return metrics


def main():
    np.set_printoptions(
        precision=12,
        suppress=False,
    )

    x0 = np.asarray(
        X0,
        dtype=np.float64,
    ).copy()

    u0 = make_u()

    print("=" * 88)
    print(
        "NEW GFM_SCENARIOS INITIALIZATION "
        "DIAGNOSTIC"
    )
    print("=" * 88)

    print("\nNEW X0:")
    print(x0)

    print("\nGrid source abc at t=0:")
    print(source_abc(0.0))

    print("\nState-space u[27:31]:")
    print(u0[27:31])

    print("\nCorrected initial breaker groups:")
    print(INITIAL_BREAKER_GROUPS)

    print("\nCorrected initial 21 breaker gates:")
    print(
        INITIAL_BREAKER_GATES.reshape(7, 3)
    )

    # --------------------------------------------------------------
    # CASE A: regenerated raw matrices before any switch update.
    # --------------------------------------------------------------
    y_base = (
        np.asarray(C)
        @ x0
        + np.asarray(D)
        @ u0
    )

    print_case(
        "CASE A — NEW RAW MATRICES: "
        "y = C*X0 + D*u",
        y_base,
    )

    # --------------------------------------------------------------
    # CASE B: apply corrected initially commanded breaker poles only.
    # --------------------------------------------------------------
    breaker_only = (
        GeneratedScenarioElectricalPlantExactNew()
    )

    breaker_only.SwitchChange.fill(0)
    breaker_only.SwitchChange[
        :21
    ] = INITIAL_BREAKER_GATES

    breaker_only._apply_switch_changes()

    breaker_only.switch_status[
        :21
    ] = INITIAL_BREAKER_GATES

    y_breaker = (
        breaker_only.Cs
        @ x0
        + breaker_only.Ds
        @ u0
    )

    print_case(
        "CASE B — NEW MATRICES WITH CORRECTED "
        "INITIAL BREAKERS APPLIED",
        y_breaker,
    )

    print("\nCASE B switch status:")
    print(
        breaker_only.switch_status
    )

    # --------------------------------------------------------------
    # CASE C: exact first generated switch solve.
    # --------------------------------------------------------------
    plant = (
        GeneratedScenarioElectricalPlantExactNew()
    )

    state = plant.initial_state()

    source = source_abc(0.0)

    inp = ScenarioPlantInput(
        dc_voltage=VDC_T0,
        source_a=source[0],
        source_b=source[1],
        source_c=source[2],
        breaker_gates=(
            INITIAL_BREAKER_GROUPS
        ),
        inverter_gates=np.zeros(
            6,
            dtype=np.int32,
        ),
    )

    out = plant.solve_switches(
        state,
        inp,
    )

    print_case(
        "CASE C — NEW EXACT FIRST "
        "SWITCH-RESOLUTION OUTPUT",
        out.y,
    )

    print("\nCASE C physical breaker states:")
    print(
        out.breaker_group_states
    )

    print(
        "CASE C inverter switch states:",
        out.inverter_switch_states,
    )

    print(
        "CASE C gates used:",
        out.gate_states,
    )

    print(
        "CASE C loops remaining:",
        out.loops_remaining,
    )

    # --------------------------------------------------------------
    # CASE D: recalc y with final resolved Cs/Ds.
    # --------------------------------------------------------------
    y_recalc = (
        plant.Cs
        @ x0
        + plant.Ds
        @ u0
    )

    print_case(
        "CASE D — NEW RECALCULATED OUTPUT "
        "WITH FINAL Cs/Ds",
        y_recalc,
    )

    print(
        "\nmax |Case D - Case C| =",
        np.max(
            np.abs(
                y_recalc - out.y
            )
        ),
    )

    print(
        "(D-C)[27:40] =",
        y_recalc[27:40]
        - out.y[27:40],
    )

    # --------------------------------------------------------------
    # CASE E: one complete plant step.
    # --------------------------------------------------------------
    plant2 = (
        GeneratedScenarioElectricalPlantExactNew()
    )

    state2 = plant2.initial_state()

    next_state, out_step = plant2.step(
        state2,
        inp,
    )

    print_case(
        "CASE E — NEW ONE COMPLETE "
        "plant.step() OUTPUT",
        out_step.y,
    )

    print("\nNEW x[1]:")
    print(next_state.x)

    print("\n" + "=" * 88)
    print("MATLAB REFERENCE AT t=0")
    print("=" * 88)

    for key, value in MATLAB_TARGET.items():
        print(
            f"{key:<5} = {value}"
        )

    print("=" * 88)

    print(
        "\nThe most important result is Case C. "
        "If its Vod/Voq/Iod/Ioq now approach "
        "the MATLAB reference, the corrected "
        "grid-breaker-generated model has "
        "resolved the initialization mismatch."
    )


if __name__ == "__main__":
    main()
