"""controller.py

GFM controller for the MATLAB/Simulink generated-code benchmark.

Equations are traced from SubsystemGFM.cpp.  The public controller operates on
BenchmarkMeasurements (Vod/Voq, Iod/Ioq, Ifd/Ifq) and updates the persistent
GFMControllerState defined in states.py.

The generated C++ confirms:
    omega = 2*pi*60 - 5e-5 * (P_filtered - 15000)
    Vod_ref = 380 - 3e-5 * Q_filtered
    voltage PI: Kp=2, Ki=14
    current PI: Kp=2, Ki=20
    Cf=50e-6, Lf=3.5e-3
    md,mq = (2/380) * Vdq_command

No artificial command or modulation saturation is applied here because the
controller path in generated C++ passes md/mq to the PWM comparison logic.
PWM/gate generation belongs in pwm.py, not in this module.
"""

from dataclasses import dataclass
import math

from parameters import PARAMS, BenchmarkParameters
from states import GFMControllerState

@dataclass(frozen=True)
class ControllerOutput:
    # Power measurement / filtering
    P: float
    Q: float
    P_filtered: float
    Q_filtered: float

    # Droop
    omega: float
    frequency: float
    theta: float
    vod_ref: float
    voq_ref: float

    # Voltage-loop quantities
    vd_error: float
    vq_error: float
    vd_pi: float
    vq_pi: float
    ifd_ref: float
    ifq_ref: float

    # Current-loop quantities
    id_error: float
    iq_error: float
    id_pi: float
    iq_pi: float
    vd_command: float
    vq_command: float

    # Generated controller-to-PWM normalized dq commands
    md: float
    mq: float

    # dq -> alpha-beta commands used before the generated 3-phase PWM mapping
    modulation_alpha: float
    modulation_beta: float


class GeneratedGFMController:
    """Readable Python realization of the generated GFM control path."""

    def __init__(self, params: BenchmarkParameters = PARAMS):
        self.params = params

    def evaluate(
        self,
        state: GFMControllerState,
        measurements,
    ) -> ControllerOutput:
        """Evaluate controller outputs at the current state, without updating it."""

        p = self.params

        # The generated transfer-function output is omega_c * internal state.
        # With the continuous block G(s)=omega_c/(s+omega_c), the stored Python
        # reference state is the normalized state whose output is omega_c*x.
        P_filtered = p.lpf_omega_c * state.p_filter_state
        Q_filtered = p.lpf_omega_c * state.q_filter_state

        # Primary P-f and Q-V droop (SubsystemGFM.cpp lines ~749-761).
        omega = p.omega_nom - p.kp_droop * (P_filtered - p.P_ref)
        frequency = omega / (2.0 * math.pi)

        vod_ref = p.voltage_ref - p.kq_droop * (Q_filtered - p.Q_ref)
        voq_ref = 0.0

        # Voltage controller.  The generated C++ uses B3 output current
        # feed-forward and omega*Cf decoupling.
        vd_error = vod_ref - measurements.vod
        vq_error = voq_ref - measurements.voq

        vd_pi = p.voltage_kp * vd_error + state.vd_integral
        vq_pi = p.voltage_kp * vq_error + state.vq_integral

        ifd_ref = (
            vd_pi
            + measurements.iod
            - omega * p.filter_C * measurements.voq
        )

        ifq_ref = (
            vq_pi
            + omega * p.filter_C * measurements.vod
            + measurements.ioq
        )

        # Current controller.  B1 currents are Ifd/Ifq.
        id_error = ifd_ref - measurements.ifd
        iq_error = ifq_ref - measurements.ifq

        id_pi = p.current_kp * id_error + state.id_integral
        iq_pi = p.current_kp * iq_error + state.iq_integral

        vd_command = (
            id_pi
            + measurements.vod
            - omega * p.inverter_L * measurements.ifq
        )

        vq_command = (
            iq_pi
            + measurements.voq
            + omega * p.inverter_L * measurements.ifd
        )

        # Exact generated Gain blocks: 0.005263157894736842 = 2/380.
        md = p.modulation_gain * vd_command
        mq = p.modulation_gain * vq_command

        # Exact generated inverse dq rotation immediately before 3-phase PWM:
        # alpha = md*sin(theta) + mq*cos(theta)
        # beta  = -md*cos(theta) + mq*sin(theta)
        s = math.sin(state.theta)
        c = math.cos(state.theta)
        modulation_alpha = md * s + mq * c
        modulation_beta = -md * c + mq * s

        return ControllerOutput(
            P=float(measurements.P),
            Q=float(measurements.Q),
            P_filtered=float(P_filtered),
            Q_filtered=float(Q_filtered),
            omega=float(omega),
            frequency=float(frequency),
            theta=float(state.theta),
            vod_ref=float(vod_ref),
            voq_ref=float(voq_ref),
            vd_error=float(vd_error),
            vq_error=float(vq_error),
            vd_pi=float(vd_pi),
            vq_pi=float(vq_pi),
            ifd_ref=float(ifd_ref),
            ifq_ref=float(ifq_ref),
            id_error=float(id_error),
            iq_error=float(iq_error),
            id_pi=float(id_pi),
            iq_pi=float(iq_pi),
            vd_command=float(vd_command),
            vq_command=float(vq_command),
            md=float(md),
            mq=float(mq),
            modulation_alpha=float(modulation_alpha),
            modulation_beta=float(modulation_beta),
        )

    def update(
        self,
        state: GFMControllerState,
        measurements,
        output: ControllerOutput | None = None,
    ) -> GFMControllerState:
        """Advance the generated controller states by one base sample.

        The PI and angle updates follow the generated C++ exactly at Ts=1e-5.
        The P/Q filters implement the Simulink transfer functions
        omega_c/(s+omega_c) with forward-Euler state integration at the same
        generated fixed step.  This is the stable readable realization of the
        generated continuous LPF semantics.
        """

        if output is None:
            output = self.evaluate(state, measurements)

        p = self.params
        dt = p.dt

        # LPF normalized state: xdot = input - omega_c*x, y=omega_c*x.
        p_filter_state = state.p_filter_state + dt * (
            measurements.P - p.lpf_omega_c * state.p_filter_state
        )
        q_filter_state = state.q_filter_state + dt * (
            measurements.Q - p.lpf_omega_c * state.q_filter_state
        )

        # Generated discrete angle integrator.
        theta = state.theta + dt * output.omega

        # Generated PI integral-state updates.
        vd_integral = state.vd_integral + dt * p.voltage_ki * output.vd_error
        vq_integral = state.vq_integral + dt * p.voltage_ki * output.vq_error
        id_integral = state.id_integral + dt * p.current_ki * output.id_error
        iq_integral = state.iq_integral + dt * p.current_ki * output.iq_error

        return GFMControllerState(
            theta=float(theta),
            p_filter_state=float(p_filter_state),
            q_filter_state=float(q_filter_state),
            vd_integral=float(vd_integral),
            vq_integral=float(vq_integral),
            id_integral=float(id_integral),
            iq_integral=float(iq_integral),
        )

    def step(
        self,
        state: GFMControllerState,
        measurements,
    ) -> tuple[GFMControllerState, ControllerOutput]:
        """Evaluate outputs, then advance controller state by one sample."""

        output = self.evaluate(state, measurements)
        next_state = self.update(state, measurements, output)
        return next_state, output


if __name__ == "__main__":
    import numpy as np
    from measurements import BenchmarkMeasurements

    controller = GeneratedGFMController()
    state = GFMControllerState()

    # Simple synthetic controller-only smoke test.
    m = BenchmarkMeasurements(
        vabc_b3=np.zeros(3),
        iabc_b3=np.zeros(3),
        iabc_b1=np.zeros(3),
        aux_abc=np.zeros(3),
        battery_current=0.0,
        vod=380.0,
        voq=0.0,
        v0=0.0,
        iod=0.0,
        ioq=0.0,
        i0_b3=0.0,
        ifd=0.0,
        ifq=0.0,
        i0_b1=0.0,
        P=0.0,
        Q=0.0,
        S=0.0,
        switch_outputs=np.zeros(6),
    )

    state, out = controller.step(state, m)

    print("Generated GFM controller sanity check")
    print("frequency     :", out.frequency)
    print("Vod_ref       :", out.vod_ref)
    print("Ifd_ref       :", out.ifd_ref)
    print("Ifq_ref       :", out.ifq_ref)
    print("Vd command    :", out.vd_command)
    print("Vq command    :", out.vq_command)
    print("md, mq        :", out.md, out.mq)
    print("theta next    :", state.theta)
