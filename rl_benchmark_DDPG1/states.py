"""states.py

State containers for the full MATLAB/Simulink generated-code benchmark.

This module is intentionally a *reference-model* state definition.  It keeps
state names close to the names emitted by Simulink Coder in Subsystem.h so
that later translations of SubsystemGFM.cpp can be checked line-by-line.

The Specialized Power Systems electrical network is represented by the
15-state vector used with the generated A/B/C/D matrices in parameters.py.
The remaining scalar states correspond to generated Unit Delay, discrete
integrators, transfer-function states, battery states, and source/PWM helper
states.
"""

from dataclasses import dataclass, field
import numpy as np

from parameters import N_STATE, X0


# =============================================================================
# Generated Specialized Power Systems plant state
# =============================================================================

@dataclass
class ElectricalNetworkState:
    """15 generated Specialized Power Systems electrical states."""

    x: np.ndarray = field(
        default_factory=lambda: np.array(X0, dtype=np.float64, copy=True)
    )

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=np.float64).reshape(N_STATE)

    def copy(self):
        return ElectricalNetworkState(self.x.copy())


# =============================================================================
# GFM controller states
# =============================================================================

@dataclass
class GFMControllerState:
    """Persistent states belonging to the generated GFM controller.

    Mapping to generated C++ names
    ------------------------------
    theta                    -> DiscreteTimeIntegrator_DSTATE (<S60>)
    q_filter_state           -> DiscreteTransferFcn1_states
    vd_integral              -> Integrator_DSTATE (<S288>)
    p_filter_state           -> DiscreteTransferFcn_states
    id_integral              -> Integrator_DSTATE_f (<S99>)
    vq_integral              -> Integrator_DSTATE_d (<S236>)
    iq_integral              -> Integrator_DSTATE_l (<S151>)

    The generated PI integrator states already contain the integral *output*;
    e.g. vd_integral is updated by 14*e_vd*Ts and id_integral by 20*e_id*Ts.
    """

    theta: float = 0.0

    p_filter_state: float = 0.0
    q_filter_state: float = 0.0

    vd_integral: float = 0.0
    vq_integral: float = 0.0

    id_integral: float = 0.0
    iq_integral: float = 0.0


# =============================================================================
# Independent source / transformation angle state
# =============================================================================

@dataclass
class SourceAngleState:
    """Additional generated angle integrator (<S43>).

    Kept separate from the GFM oscillator angle because the generated model
    contains both DiscreteTimeIntegrator_DSTATE and
    DiscreteTimeIntegrator_DSTATE_n.
    """

    theta: float = 0.0


# =============================================================================
# Battery generated states
# =============================================================================

@dataclass
class BatteryGeneratedState:
    """Persistent battery-model states emitted by Simulink Coder."""

    # <S34>/Unit Delay.  Static storage is zero-initialized by generated C++.
    unit_delay_voltage: float = 0.0

    # <S37>/Discrete-Time Integrator.  Explicit generated initialization.
    dynamic_integrator: float = 3.4839433927591004e-44

    # <S34>/Current Filter
    current_filter_state: float = 0.0

    # <S34>/Coulomb Counter.  Loaded from Initialit_PreviousInput on first use.
    coulomb_counter: float = 374999.99999999994

    # Generated initialization for '<S34>/Initial it'.
    initial_it_previous_input: float = 374999.99999999994

    # Generated reset bookkeeping for Coulomb Counter.
    coulomb_counter_prev_reset_state: int = 2
    coulomb_counter_ic_loading: bool = True


# =============================================================================
# Generated sine-source recurrence states
# =============================================================================

@dataclass
class ThreePhaseSourceState:
    """Persistent recurrence values used by generated three-phase sine blocks.

    Simulink Coder maintains lastSin/lastCos values for phases A/B/C.  Their
    exact enable-time initialization is performed by the generated step code,
    so these are kept explicitly for the later full translation.
    """

    last_sin_a: float = 0.0
    last_cos_a: float = 1.0

    last_sin_b: float = 0.0
    last_cos_b: float = 1.0

    last_sin_c: float = 0.0
    last_cos_c: float = 1.0

    enable_a: bool = True
    enable_b: bool = True
    enable_c: bool = True


# =============================================================================
# Complete benchmark state
# =============================================================================

@dataclass
class BenchmarkState:
    """Complete dynamic/persistent state of the translated benchmark."""

    network: ElectricalNetworkState = field(
        default_factory=ElectricalNetworkState
    )

    controller: GFMControllerState = field(
        default_factory=GFMControllerState
    )

    source_angle: SourceAngleState = field(
        default_factory=SourceAngleState
    )

    battery: BatteryGeneratedState = field(
        default_factory=BatteryGeneratedState
    )

    source: ThreePhaseSourceState = field(
        default_factory=ThreePhaseSourceState
    )

    # Generated model execution time / tick count.
    time: float = 0.0
    step_count: int = 0

    def copy(self):
        """Return an independent state copy."""
        return BenchmarkState(
            network=self.network.copy(),
            controller=GFMControllerState(**vars(self.controller)),
            source_angle=SourceAngleState(**vars(self.source_angle)),
            battery=BatteryGeneratedState(**vars(self.battery)),
            source=ThreePhaseSourceState(**vars(self.source)),
            time=float(self.time),
            step_count=int(self.step_count),
        )


# =============================================================================
# Factory
# =============================================================================

def create_initial_state() -> BenchmarkState:
    """Create the generated-code benchmark initial state.

    The 15 electrical states come directly from X0 in parameters.py.  Scalar
    generated states use the explicit initial values found in
    SubsystemGFM.cpp; generated static states without an explicit assignment
    are initialized to zero, matching C/C++ static-storage initialization.
    """

    return BenchmarkState()


# =============================================================================
# Sanity check
# =============================================================================

if __name__ == "__main__":
    state = create_initial_state()

    print("MATLAB generated benchmark state initialized")
    print("Electrical states :", state.network.x.shape)
    print("First x state     :", state.network.x[0])
    print("GFM theta         :", state.controller.theta)
    print("P LPF state       :", state.controller.p_filter_state)
    print("Q LPF state       :", state.controller.q_filter_state)
    print("Battery charge IC :", state.battery.coulomb_counter)
    print("Time              :", state.time)
