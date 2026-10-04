"""Exact 27-switch SPS plant for the NEW GFM_scenarios generated model.

Uses scenario_parameters_new.py extracted from GFM_scenariosNew_ert_rtw.zip.

Important corrected generated initialization:
    breaker groups = [1, 0, 1, 0, 0, 0, 1]

Thus group G2 is CLOSED initially, consistent with the corrected
grid-connected startup model.

This preserves the literal switch-resolution structure used in the previous
exact plant, but uses only the newly generated A/B/C/D/X0 parameter set.
"""

from dataclasses import dataclass
import numpy as np

from states import ElectricalNetworkState
from scenario_parameters_new import (
    A, B, C, D, X0,
    N_STATE, N_INPUT, N_OUTPUT,
    SWITCH_CONDUCTANCE,
)

N_BREAKER_SWITCH = 21
N_INVERTER_SWITCH = 6
N_SWITCH = 27
N_BREAKER_GROUPS = 7

SWITCH_TYPES = np.array([2] * 21 + [7] * 6, dtype=np.int32)

# Corrected NEW generated initial topology:
# G0 closed, G1 open, G2 closed, G3-G5 open, G6 closed.
INITIAL_BREAKER_GROUPS = np.array(
    [1, 0, 1, 0, 0, 0, 1],
    dtype=np.int32,
)
INITIAL_BREAKER_GATES = np.repeat(
    INITIAL_BREAKER_GROUPS, 3
).astype(np.int32)

INITIAL_INVERTER_GATES = np.zeros(6, dtype=np.int32)


def expand_breaker_groups(groups):
    g = np.asarray(groups, dtype=np.int32).reshape(N_BREAKER_GROUPS)
    return np.repeat(g, 3)


@dataclass
class ScenarioPlantInput:
    dc_voltage: float
    source_a: float
    source_b: float
    source_c: float
    breaker_gates: np.ndarray
    inverter_gates: np.ndarray

    def __post_init__(self):
        bg = np.asarray(self.breaker_gates, dtype=np.int32).reshape(-1)
        if bg.size == N_BREAKER_GROUPS:
            bg = expand_breaker_groups(bg)
        if bg.size != N_BREAKER_SWITCH:
            raise ValueError(
                "breaker_gates must contain 7 group commands or 21 pole commands"
            )
        self.breaker_gates = bg

        self.inverter_gates = np.asarray(
            self.inverter_gates, dtype=np.int32
        ).reshape(N_INVERTER_SWITCH)

    @property
    def gates(self):
        return np.concatenate(
            (self.breaker_gates, self.inverter_gates)
        ).astype(np.int32, copy=False)

    def state_space_input(self):
        u = np.zeros(N_INPUT, dtype=np.float64)

        # Generated StateSpace_u mapping.
        # 0:21  = breaker SwitchCurrents constants
        # 21:27 = inverter switch-current inputs (zero)
        # 27    = battery/DC terminal voltage
        # 28:31 = three-phase grid source
        u[27] = float(self.dc_voltage)
        u[28] = float(self.source_a)
        u[29] = float(self.source_b)
        u[30] = float(self.source_c)

        return u


@dataclass
class ScenarioPlantOutput:
    y: np.ndarray
    switch_states: np.ndarray
    gate_states_used: np.ndarray
    loops_remaining: int

    @property
    def breaker_states(self):
        return self.switch_states[:N_BREAKER_SWITCH].copy()

    @property
    def breaker_group_states(self):
        return self.breaker_states.reshape(N_BREAKER_GROUPS, 3)

    @property
    def inverter_switch_states(self):
        return self.switch_states[N_BREAKER_SWITCH:].copy()

    @property
    def gate_states(self):
        # Compatibility with scenario_measurements.
        return self.gate_states_used


class GeneratedScenarioElectricalPlantExactNew:
    """Literal SPS ideal-switch solver using the NEW generated matrices."""

    def __init__(self):
        self.As = np.array(A, dtype=np.float64, copy=True)
        self.Bs = np.array(B, dtype=np.float64, copy=True)
        self.Cs = np.array(C, dtype=np.float64, copy=True)
        self.Ds = np.array(D, dtype=np.float64, copy=True)

        self.switch_status = np.zeros(N_SWITCH, dtype=np.int32)
        self.switch_status_init = np.zeros(N_SWITCH, dtype=np.int32)
        self.SwitchChange = np.zeros(N_SWITCH, dtype=np.int32)

        self.gState = np.concatenate(
            (INITIAL_BREAKER_GATES, INITIAL_INVERTER_GATES)
        ).astype(np.int32)

        self.DxCol = np.zeros(N_OUTPUT, dtype=np.float64)
        self.tmp2 = np.zeros(N_INPUT, dtype=np.float64)
        self.BDcol = np.zeros(N_STATE, dtype=np.float64)
        self.tmp1 = np.zeros(N_STATE, dtype=np.float64)

        self.uswlast = np.zeros(N_SWITCH, dtype=np.float64)
        self.StateSpace_o1 = np.zeros(N_OUTPUT, dtype=np.float64)

    @staticmethod
    def initial_state():
        return ElectricalNetworkState(
            x=np.array(X0, dtype=np.float64, copy=True)
        )

    def reset(self):
        self.As[:] = A
        self.Bs[:] = B
        self.Cs[:] = C
        self.Ds[:] = D

        self.switch_status.fill(0)
        self.switch_status_init.fill(0)
        self.SwitchChange.fill(0)

        self.gState[:] = np.concatenate(
            (INITIAL_BREAKER_GATES, INITIAL_INVERTER_GATES)
        )

        self.DxCol.fill(0.0)
        self.tmp2.fill(0.0)
        self.BDcol.fill(0.0)
        self.tmp1.fill(0.0)
        self.uswlast.fill(0.0)
        self.StateSpace_o1.fill(0.0)

    def _compute_outputs(self, x, u):
        self.StateSpace_o1[:] = self.Cs @ x + self.Ds @ u

    def _apply_switch_changes(self):
        for i in range(N_SWITCH):
            change = int(self.SwitchChange[i])
            if change == 0:
                continue

            a1 = SWITCH_CONDUCTANCE * float(change)
            denominator = 1.0 - self.Ds[i, i] * a1

            if abs(denominator) < 1.0e-15:
                raise FloatingPointError(
                    f"Singular switch update at switch {i}"
                )

            temp = 1.0 / denominator

            self.DxCol[:] = self.Ds[:, i] * temp * a1
            self.DxCol[i] = temp

            self.BDcol[:] = self.Bs[:, i] * a1

            self.tmp1[:] = self.Cs[i, :]
            self.Cs[i, :] = 0.0

            self.tmp2[:] = self.Ds[i, :]
            self.Ds[i, :] = 0.0

            self.Cs += np.outer(self.DxCol, self.tmp1)
            self.Ds += np.outer(self.DxCol, self.tmp2)

            self.As += np.outer(self.BDcol, self.Cs[i, :])
            self.Bs += np.outer(self.BDcol, self.Ds[i, :])

    def solve_switches(self, state, inputs):
        x = np.asarray(state.x, dtype=np.float64)
        u = inputs.state_space_input()

        self.switch_status_init[:] = self.switch_status

        # Previous StateSpace_o1 switch-port outputs.
        self.uswlast[:] = self.StateSpace_o1[:N_SWITCH]

        swChanged = 0
        loopsToDo = 20

        while True:
            if loopsToDo == 1:
                swChanged = 0

                for i in range(N_SWITCH):
                    change = (
                        int(self.switch_status_init[i])
                        - int(self.switch_status[i])
                    )
                    self.SwitchChange[i] = change

                    if change != 0:
                        swChanged = 1

                    self.switch_status[i] = self.switch_status_init[i]

            else:
                self._compute_outputs(x, u)
                swChanged = 0

                for i in range(N_SWITCH):
                    y0 = float(self.StateSpace_o1[i])

                    if SWITCH_TYPES[i] == 2:
                        if self.gState[i] > 0:
                            newState = 1
                        elif y0 * self.uswlast[i] < 0.0:
                            newState = 0
                        else:
                            newState = int(self.switch_status[i])

                    else:
                        if (
                            ((y0 > 0.0) and (self.gState[i] > 0))
                            or (y0 < 0.0)
                        ):
                            newState = 1
                        elif (y0 > 0.0) and (self.gState[i] == 0):
                            newState = 0
                        else:
                            newState = int(self.switch_status[i])

                    change = newState - int(self.switch_status[i])
                    self.SwitchChange[i] = change

                    if change != 0:
                        swChanged = 1

                    self.switch_status[i] = newState

            if swChanged:
                self._apply_switch_changes()

            if not (swChanged > 0):
                break

            loopsToDo -= 1
            if not (loopsToDo > 0):
                break

        if loopsToDo == 0:
            self._compute_outputs(x, u)

        return ScenarioPlantOutput(
            y=self.StateSpace_o1.copy(),
            switch_states=self.switch_status.copy(),
            gate_states_used=self.gState.copy(),
            loops_remaining=int(loopsToDo),
        )

    def advance_state(self, state, inputs):
        u = inputs.state_space_input()
        x_next = self.As @ state.x + self.Bs @ u
        return ElectricalNetworkState(x=x_next)

    def store_gates_for_next_step(self, inputs):
        self.gState[:] = inputs.gates

    def step(self, state, inputs):
        out = self.solve_switches(state, inputs)
        next_state = self.advance_state(state, inputs)
        self.store_gates_for_next_step(inputs)
        return next_state, out


# Convenient benchmark-compatible alias.
GeneratedScenarioElectricalPlant = GeneratedScenarioElectricalPlantExactNew


if __name__ == "__main__":
    plant = GeneratedScenarioElectricalPlantExactNew()
    state = plant.initial_state()

    # At generated startup the battery Unit Delay initially supplies 0 V.
    peak = 13_800.0 / np.sqrt(3.0) * np.sqrt(2.0)
    source = np.array(
        [
            0.0,
            peak * np.sin(-2.0 * np.pi / 3.0),
            peak * np.sin(2.0 * np.pi / 3.0),
        ]
    )

    inp = ScenarioPlantInput(
        dc_voltage=0.0,
        source_a=source[0],
        source_b=source[1],
        source_c=source[2],
        breaker_gates=INITIAL_BREAKER_GROUPS,
        inverter_gates=np.zeros(6, dtype=np.int32),
    )

    next_state, out = plant.step(state, inp)

    print("NEW exact scenario plant smoke test")
    print("-----------------------------------")
    print("X0:")
    print(state.x)
    print("initial breaker groups:", INITIAL_BREAKER_GROUPS)
    print("physical breaker states:")
    print(out.breaker_group_states)
    print("inverter switch states:", out.inverter_switch_states)
    print("loops remaining:", out.loops_remaining)
    print("y[27:40]:")
    print(out.y[27:40])
    print("finite y:", np.isfinite(out.y).all())
    print("finite x_next:", np.isfinite(next_state.x).all())
