"""Exact battery/DC-side dynamics translated from GFM_scenarios generated C++.

Source blocks:
    GFM_scenarios/Battery/Model/Discrete
    GFM_scenarios/Battery/Current Measurement/Model

The SPS plant current used by the battery model is StateSpace_o1[36].

Important generated behavior
----------------------------
* The state-space plant does NOT receive a fixed 800-V DC source.
* It receives the saturated Battery Unit Delay output ``VoltageV``.
* At generated initialization the Unit Delay is zero-initialized, so the first
  model output is 0 V.  The first update computes the battery terminal voltage
  for the following 10-us sample.
* The model includes the generated current filter, coulomb counter, exponential
  zone state, charge/discharge voltage-loss equations, and 0..1636.4447-V
  terminal-voltage saturation.
"""

from dataclasses import dataclass
import math


DT = 1.0e-5

# Generated constants.
VOLTAGE_MAX = 1636.444691978842
VOLTAGE_MIN = 0.0

E0 = 818.222345989421
R_INTERNAL = 0.008

Q_AH = 1156.2499999999886
Q_CLAMP_AH = 1156.1343749999885
Q_DISCHARGE_LOWER_AH = -115.61343749999885
Q_OFFSET_AH = 115.62499999999886

K = 0.023825738183501019

# Exponential-zone state equation coefficient.
EXP_GAIN = 54.430285589526278
ABS_CURRENT_GAIN = 0.00025000000000000006

# Current-filter generated discrete transfer function.
CURRENT_FILTER_OUTPUT_GAIN = 9.9999949998430537e-7
CURRENT_FILTER_FEEDBACK = -0.9999990000005

# Coulomb-counter limits/initialization.
COULOMB_MAX = 4.162499999999959e6
INITIAL_IT_MEMORY = 374999.99999999994
INITIAL_EXP_STATE = 3.4839433927591004e-44


@dataclass
class BatteryDCState:
    """Generated battery discrete states."""

    # <S41>/Unit Delay. Static generated storage is initially zero.
    voltage_delay: float = 0.0

    # <S41>/Current Filter. Static generated storage is initially zero.
    current_filter_state: float = 0.0

    # <S44>/Discrete-Time Integrator.
    exponential_state: float = INITIAL_EXP_STATE

    # <S41>/Initial it.
    initial_it_memory: float = INITIAL_IT_MEMORY

    # <S41>/Coulomb Counter.
    coulomb_counter: float = 0.0
    coulomb_prev_reset_state: int = 2
    coulomb_ic_loading: bool = True


@dataclass(frozen=True)
class BatteryDCOutput:
    voltage: float
    current: float
    soc_percent: float

    # Useful internal diagnostics.
    filtered_current: float
    charge_ah: float
    exponential_state: float
    voltage_loss: float


class GeneratedBatteryDC:
    """Readable one-sample translation of the generated lead-acid battery."""

    def __init__(self):
        self.state = BatteryDCState()

    def reset(self):
        self.state = BatteryDCState()
        return self.state

    @staticmethod
    def _saturate_voltage(v):
        return min(max(float(v), VOLTAGE_MIN), VOLTAGE_MAX)

    def output_voltage(self):
        """Voltage supplied to the SPS plant during the current sample."""
        return self._saturate_voltage(self.state.voltage_delay)

    def evaluate(self, battery_current):
        """Evaluate generated battery algebra using current discrete states.

        ``battery_current`` is exactly StateSpace_o1[36].
        """
        s = self.state
        i = float(battery_current)

        # Generated Current Filter output before its state update.
        i_filtered = CURRENT_FILTER_OUTPUT_GAIN * s.current_filter_state

        is_charging = i < 0.0
        is_discharging = i > 0.0

        # Coulomb Counter IC-loading/reset logic is executed before Gain1.
        q_counter = s.coulomb_counter

        if s.coulomb_ic_loading:
            q_counter = s.initial_it_memory
            q_counter = min(q_counter, COULOMB_MAX)

        if is_discharging and s.coulomb_prev_reset_state <= 0:
            q_counter = s.initial_it_memory
            q_counter = min(q_counter, COULOMB_MAX)

        charge_ah = (1.0 / 3600.0) * q_counter

        # MATLAB Function / lead-acid voltage-loss model.
        q_clamped = min(max(charge_ah, 0.0), Q_CLAMP_AH)

        if is_charging:
            q_branch = min(
                max(charge_ah, Q_DISCHARGE_LOWER_AH),
                Q_CLAMP_AH,
            )
            voltage_loss = (
                Q_AH / (q_branch + Q_OFFSET_AH)
                * (-K) * i_filtered
                - Q_AH / (Q_AH - q_branch)
                * K * q_branch
                + s.exponential_state
            )
        else:
            voltage_loss = (
                Q_AH / (Q_AH - q_clamped)
                * (-K) * (q_clamped + i_filtered)
                + s.exponential_state
            )

        # This is the value written to UnitDelay_DSTATE at the end of the step.
        voltage_next_unsat = (
            voltage_loss
            - R_INTERNAL * i
            + E0
        )

        # SOC output uses q_clamped, as in generated rtb_Gain_idx_0.
        soc = (Q_AH - q_clamped) / Q_AH * 100.0

        return BatteryDCOutput(
            voltage=self.output_voltage(),
            current=i,
            soc_percent=float(soc),
            filtered_current=float(i_filtered),
            charge_ah=float(q_clamped),
            exponential_state=float(s.exponential_state),
            voltage_loss=float(voltage_loss),
        ), float(voltage_next_unsat), float(q_counter)

    def step(self, battery_current):
        """Advance the battery by one 10-us generated sample."""
        s = self.state
        i = float(battery_current)

        out, voltage_next_unsat, q_counter = self.evaluate(i)

        is_charging = i < 0.0
        is_discharging = i > 0.0

        # <S44>/Gain: 0.00025*abs(I)
        exp_drive = ABS_CURRENT_GAIN * abs(i)

        # Update for <S44>/Discrete-Time Integrator:
        # state += (54.430... * charging * exp_drive
        #           - state * exp_drive) * Ts
        exp_next = s.exponential_state + (
            EXP_GAIN * (1.0 if is_charging else 0.0) * exp_drive
            - s.exponential_state * exp_drive
        ) * DT

        # Update for Current Filter:
        # state = I - (-0.9999990000005)*state
        current_filter_next = (
            i - CURRENT_FILTER_FEEDBACK * s.current_filter_state
        )

        # Generated Initial-it memory is updated from 3600*q_clamped.
        initial_it_next = 3600.0 * out.charge_ah

        # Coulomb counter integrates the current-filter *output*.
        i_filtered = CURRENT_FILTER_OUTPUT_GAIN * s.current_filter_state
        coulomb_next = q_counter + DT * i_filtered
        coulomb_next = min(coulomb_next, COULOMB_MAX)

        self.state = BatteryDCState(
            voltage_delay=float(voltage_next_unsat),
            current_filter_state=float(current_filter_next),
            exponential_state=float(exp_next),
            initial_it_memory=float(initial_it_next),
            coulomb_counter=float(coulomb_next),
            coulomb_prev_reset_state=1 if is_discharging else 0,
            coulomb_ic_loading=False,
        )

        return out

    def diagnostics(self, battery_current):
        out, voltage_next, _ = self.evaluate(battery_current)
        return {
            "voltage_to_plant": out.voltage,
            "battery_current": out.current,
            "soc_percent": out.soc_percent,
            "filtered_current": out.filtered_current,
            "charge_ah": out.charge_ah,
            "exponential_state": out.exponential_state,
            "voltage_loss": out.voltage_loss,
            "next_voltage_unsaturated": voltage_next,
        }


if __name__ == "__main__":
    b = GeneratedBatteryDC()

    print("Generated battery/DC smoke test")
    print("--------------------------------")
    print("initial voltage to SPS:", b.output_voltage())

    for k, current in enumerate([0.0, 0.0, 10.0, 10.0]):
        out = b.step(current)
        print(
            f"k={k:2d}  I={current:8.3f} A  "
            f"V_used={out.voltage:12.6f} V  "
            f"V_next={b.output_voltage():12.6f} V  "
            f"SOC={out.soc_percent:10.6f}%"
        )
