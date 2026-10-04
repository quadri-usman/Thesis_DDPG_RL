"""plant.py

Generated Specialized Power Systems plant core translated from the Simulink
Coder S-function ``<S330>/State-Space`` in ``SubsystemGFM.cpp``.

This is a benchmark/reference implementation.  It preserves the generated
15-state, 10-input, 19-output discrete state-space network and the six-switch
matrix-update algorithm used by ``sfun_spssw_discc``.

The first six state-space input columns are switch ports.  As in the generated
C++ they are supplied as zeros during the C/D and A/B calculations; switch
conduction is represented by modifying A/B/C/D when a switch changes state.
The remaining four inputs are::

    u[6] = DC/battery terminal voltage (generated ``Voltage (V)`` signal)
    u[7] = three-phase source A
    u[8] = three-phase source B
    u[9] = three-phase source C

The six PWM gate commands are *not* entries of u.  They are passed separately
and stored as the generated ``gState`` values for the switch solver.
"""

from dataclasses import dataclass
import numpy as np

from parameters import A, B, C, D, N_INPUT, N_OUTPUT, N_STATE
from load_models import get_load_model
from states import ElectricalNetworkState


N_SWITCH = 6
SWITCH_CONDUCTANCE = 1000.0  # generated 1 / 0.001
MAX_SWITCH_ITERATIONS = 20


@dataclass
class PlantInput:
    """Inputs required by the generated electrical-network block."""

    dc_voltage: float
    source_a: float
    source_b: float
    source_c: float
    gates: np.ndarray

    def __post_init__(self):
        self.gates = np.asarray(self.gates, dtype=np.int32).reshape(N_SWITCH)

    def state_space_input(self) -> np.ndarray:
        """Return the exact 10-element input vector used by generated code."""
        u = np.zeros(N_INPUT, dtype=np.float64)
        u[6] = float(self.dc_voltage)
        u[7] = float(self.source_a)
        u[8] = float(self.source_b)
        u[9] = float(self.source_c)
        return u


@dataclass
class PlantOutput:
    """Raw generated State-Space outputs.

    ``y`` corresponds exactly to ``StateSpace_o1[0:19]`` in the C++.
    Physical signal names (B1/B3 voltage/current, battery current, etc.) will
    be mapped in the measurement layer after their generated indices are
    verified from ``SubsystemGFM.cpp``.
    """

    y: np.ndarray
    switch_states: np.ndarray

    def __post_init__(self):
        self.y = np.asarray(self.y, dtype=np.float64).reshape(N_OUTPUT)
        self.switch_states = np.asarray(
            self.switch_states, dtype=np.int32
        ).reshape(N_SWITCH)


class GeneratedElectricalPlant:
    """Runtime implementation of the generated SPS state-space network."""

    def __init__(self, load_power_w: int = 50_000):
        # Keep an explicit *base* generated network.  As/Bs/Cs/Ds are mutable
        # copies because the generated ideal-switch solver modifies them.
        model = get_load_model(load_power_w)
        self.load_power_w = int(model.power_w)

        self.base_A = np.array(model.A, dtype=np.float64, copy=True)
        self.base_B = np.array(model.B, dtype=np.float64, copy=True)
        self.base_C = np.array(model.C, dtype=np.float64, copy=True)
        self.base_D = np.array(model.D, dtype=np.float64, copy=True)

        self.As = self.base_A.copy()
        self.Bs = self.base_B.copy()
        self.Cs = self.base_C.copy()
        self.Ds = self.base_D.copy()

        # Generated switch work vectors are zero initialized.
        self.switch_status = np.zeros(N_SWITCH, dtype=np.int32)
        self.g_state = np.zeros(N_SWITCH, dtype=np.int32)

    def reset(self):
        """Reset mutable matrices to the currently selected load model."""
        self.As[...] = self.base_A
        self.Bs[...] = self.base_B
        self.Cs[...] = self.base_C
        self.Ds[...] = self.base_D
        self.switch_status.fill(0)
        self.g_state.fill(0)

    def _rebuild_for_switch_status(self, switch_status):
        """Rebuild mutable matrices from the base network and replay switches."""
        status = np.asarray(switch_status, dtype=np.int32).reshape(N_SWITCH).copy()

        self.As[...] = self.base_A
        self.Bs[...] = self.base_B
        self.Cs[...] = self.base_C
        self.Ds[...] = self.base_D
        self.switch_status.fill(0)

        for i, value in enumerate(status):
            if value:
                self._apply_one_switch_change(i, 1)
                self.switch_status[i] = 1

    def set_load_model(self, load_power_w: int):
        """Switch to an exact MATLAB-generated load model without resetting x.

        The caller owns the 15 electrical state vector, so this method changes
        only the plant realization.  Current switch conduction and stored PWM
        gate commands are preserved.
        """
        model = get_load_model(load_power_w)

        old_switch_status = self.switch_status.copy()
        old_g_state = self.g_state.copy()

        self.load_power_w = int(model.power_w)
        self.base_A = np.array(model.A, dtype=np.float64, copy=True)
        self.base_B = np.array(model.B, dtype=np.float64, copy=True)
        self.base_C = np.array(model.C, dtype=np.float64, copy=True)
        self.base_D = np.array(model.D, dtype=np.float64, copy=True)

        self._rebuild_for_switch_status(old_switch_status)
        self.g_state[:] = old_g_state

    @staticmethod
    def _output(Cs, Ds, x, u):
        return Cs @ x + Ds @ u

    def _apply_one_switch_change(self, switch_index: int, change: int):
        """Translate the generated rank-one A/B/C/D switch update exactly."""
        i = int(switch_index)
        a1 = SWITCH_CONDUCTANCE * float(change)

        denominator = 1.0 - self.Ds[i, i] * a1
        if abs(denominator) < 1.0e-14:
            raise FloatingPointError(
                f"Generated switch update is singular for switch {i}."
            )

        temp = 1.0 / denominator

        # Generated C++:
        #   DxCol = Ds[:,i] * temp * a1
        #   DxCol[i] = temp
        dx_col = self.Ds[:, i].copy() * temp * a1
        dx_col[i] = temp

        # Generated C++:
        #   BDcol = Bs[:,i] * a1
        bd_col = self.Bs[:, i].copy() * a1

        # Save switch row, then zero it.
        tmp1 = self.Cs[i, :].copy()
        tmp2 = self.Ds[i, :].copy()
        self.Cs[i, :] = 0.0
        self.Ds[i, :] = 0.0

        # Cs += DxCol * tmp1
        # Ds += DxCol * tmp2
        self.Cs += np.outer(dx_col, tmp1)
        self.Ds += np.outer(dx_col, tmp2)

        # As += BDcol * Cs[switch row]
        # Bs += BDcol * Ds[switch row]
        self.As += np.outer(bd_col, self.Cs[i, :])
        self.Bs += np.outer(bd_col, self.Ds[i, :])

    def solve_switches(self, state: ElectricalNetworkState, inputs: PlantInput):
        """Resolve the six ideal switches as the generated S-function does.

        Returns the final 19 network outputs after switch convergence.
        """
        x = state.x
        u = inputs.state_space_input()

        switch_status_initial = self.switch_status.copy()
        loops_left = MAX_SWITCH_ITERATIONS

        while True:
            # Compute outputs for the current network matrices.
            y = self._output(self.Cs, self.Ds, x, u)

            new_status = self.switch_status.copy()
            for i in range(N_SWITCH):
                yi = y[i]
                gate = self.g_state[i]

                # Exact logical expression emitted by Simulink Coder.
                if ((yi > 0.0) and (gate > 0)) or (yi < 0.0):
                    new_status[i] = 1
                elif (yi > 0.0) and (gate == 0):
                    new_status[i] = 0
                else:
                    new_status[i] = self.switch_status[i]

            changes = new_status - self.switch_status
            changed = bool(np.any(changes != 0))
            self.switch_status[:] = new_status

            if not changed:
                return y

            for i in range(N_SWITCH):
                if changes[i] != 0:
                    self._apply_one_switch_change(i, int(changes[i]))

            loops_left -= 1
            if loops_left <= 0:
                # Generated code falls back to the original switch status and
                # evaluates outputs once more.  Reconstruct the base matrices
                # and replay the original status to make the fallback explicit.
                old_g_state = self.g_state.copy()
                self._rebuild_for_switch_status(switch_status_initial)
                self.g_state[:] = old_g_state
                return self._output(self.Cs, self.Ds, x, u)

    def output(self, state: ElectricalNetworkState, inputs: PlantInput) -> PlantOutput:
        """Evaluate network outputs without advancing the 15 electrical states."""
        y = self.solve_switches(state, inputs)
        return PlantOutput(y=y.copy(), switch_states=self.switch_status.copy())

    def step(
        self,
        state: ElectricalNetworkState,
        inputs: PlantInput,
    ) -> tuple[ElectricalNetworkState, PlantOutput]:
        """Execute one generated electrical-network sample.

        Ordering follows ``SubsystemGFM.cpp``:
          1. resolve switches and compute StateSpace_o1,
          2. compute x[k+1] = As*x[k] + Bs*u[k],
          3. store the six PWM gate commands for the *next* sample.
        """
        output = self.output(state, inputs)
        u = inputs.state_space_input()

        x_next = self.As @ state.x + self.Bs @ u
        next_state = ElectricalNetworkState(x=x_next)

        # Generated update section stores DataTypeConversion[0:6] into gState.
        self.g_state[:] = inputs.gates

        return next_state, output


# =============================================================================
# Minimal sanity check
# =============================================================================

if __name__ == "__main__":
    from states import create_initial_state

    benchmark_state = create_initial_state()
    plant = GeneratedElectricalPlant()

    test_input = PlantInput(
        dc_voltage=800.0,
        source_a=0.0,
        source_b=0.0,
        source_c=0.0,
        gates=np.zeros(6, dtype=np.int32),
    )

    next_network, out = plant.step(benchmark_state.network, test_input)

    print("Generated electrical plant sanity check")
    print("x shape       :", benchmark_state.network.x.shape)
    print("y shape       :", out.y.shape)
    print("switch states :", out.switch_states)
    print("finite y      :", np.isfinite(out.y).all())
    print("finite x_next :", np.isfinite(next_network.x).all())
