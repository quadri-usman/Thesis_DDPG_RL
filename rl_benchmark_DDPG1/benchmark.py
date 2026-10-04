"""benchmark.py

Top-level runner for the MATLAB/Simulink generated-code benchmark translation.

Connects the translated modules in generated execution order:

    generated SPS plant output
        -> B1/B3 measurements and abc->dq
        -> GFM controller
        -> 5-kHz PWM
        -> six switch gates
        -> generated 15-state SPS plant update

The electrical grid/transformer/filter/load/switching network is represented by
A/B/C/D and the six-switch solver in plant.py, rather than by hand-written
component equations.

Current scope
-------------
The generated electrical network, GFM controller, PWM and measurements are
connected.  The DC terminal is held at the confirmed MATLAB nominal 800 V and
the upstream three-phase source is generated as an ideal 13.8-kV LL RMS,
60-Hz source.  The separate generated battery dynamic states retained in
states.py are not yet connected here; that will be the next refinement if an
exact battery transient is required.
"""

from dataclasses import dataclass, field
import math
import numpy as np

from parameters import PARAMS, BenchmarkParameters
from states import BenchmarkState, create_initial_state
from plant import GeneratedElectricalPlant, PlantInput, PlantOutput
from measurements import BenchmarkMeasurements, measurement_step
from controller import GeneratedGFMController, ControllerOutput
from pwm import GeneratedPWM, PWMOutput


@dataclass(frozen=True)
class BenchmarkConfig:
    """External operating conditions not compiled into controller constants."""

    # MATLAB Three-Phase Source block
    grid_voltage_ll_rms: float = 13_800.0
    grid_frequency: float = 60.0
    grid_phase_deg: float = 0.0

    # Confirmed MATLAB DC source/battery nominal voltage
    dc_voltage: float = 800.0


@dataclass(frozen=True)
class BenchmarkScenario:
    """Episode/disturbance definition for the generated benchmark.

    Loads must currently be exact generated models available in load_models.py
    (40, 50, or 60 kW).  A load step changes A/B/C/D but preserves the live
    15-state electrical vector and all controller states.
    """

    load_initial: int = 50_000
    load_final: int | None = None
    load_step_time: float | None = None

    def __post_init__(self):
        from load_models import available_loads

        valid = set(available_loads())

        if int(self.load_initial) not in valid:
            raise ValueError(
                f"load_initial={self.load_initial} W is unavailable; "
                f"choose one of {sorted(valid)}."
            )

        if self.load_final is not None and int(self.load_final) not in valid:
            raise ValueError(
                f"load_final={self.load_final} W is unavailable; "
                f"choose one of {sorted(valid)}."
            )

        if self.load_step_time is not None and self.load_step_time < 0.0:
            raise ValueError("load_step_time must be >= 0.")

        if (self.load_final is None) != (self.load_step_time is None):
            raise ValueError(
                "load_final and load_step_time must either both be set or both be None."
            )


@dataclass(frozen=True)
class BenchmarkSample:
    """One complete benchmark sample for logging/diagnostics."""

    time: float
    plant: PlantOutput
    measurements: BenchmarkMeasurements
    controller: ControllerOutput
    pwm: PWMOutput
    load_power: float


@dataclass
class BenchmarkHistory:
    time: list = field(default_factory=list)

    frequency: list = field(default_factory=list)
    omega: list = field(default_factory=list)
    theta: list = field(default_factory=list)

    P: list = field(default_factory=list)
    Q: list = field(default_factory=list)
    P_filtered: list = field(default_factory=list)
    Q_filtered: list = field(default_factory=list)

    vod: list = field(default_factory=list)
    voq: list = field(default_factory=list)
    vod_ref: list = field(default_factory=list)

    iod: list = field(default_factory=list)
    ioq: list = field(default_factory=list)
    ifd: list = field(default_factory=list)
    ifq: list = field(default_factory=list)
    ifd_ref: list = field(default_factory=list)
    ifq_ref: list = field(default_factory=list)

    vd_command: list = field(default_factory=list)
    vq_command: list = field(default_factory=list)
    md: list = field(default_factory=list)
    mq: list = field(default_factory=list)

    carrier: list = field(default_factory=list)
    ma: list = field(default_factory=list)
    mb: list = field(default_factory=list)
    mc: list = field(default_factory=list)

    battery_current: list = field(default_factory=list)
    load_power: list = field(default_factory=list)

    def append(self, sample: BenchmarkSample):
        m = sample.measurements
        c = sample.controller
        p = sample.pwm

        self.time.append(sample.time)
        self.frequency.append(c.frequency)
        self.omega.append(c.omega)
        self.theta.append(c.theta)

        self.P.append(m.P)
        self.Q.append(m.Q)
        self.P_filtered.append(c.P_filtered)
        self.Q_filtered.append(c.Q_filtered)

        self.vod.append(m.vod)
        self.voq.append(m.voq)
        self.vod_ref.append(c.vod_ref)

        self.iod.append(m.iod)
        self.ioq.append(m.ioq)
        self.ifd.append(m.ifd)
        self.ifq.append(m.ifq)
        self.ifd_ref.append(c.ifd_ref)
        self.ifq_ref.append(c.ifq_ref)

        self.vd_command.append(c.vd_command)
        self.vq_command.append(c.vq_command)
        self.md.append(c.md)
        self.mq.append(c.mq)

        self.carrier.append(p.carrier)
        self.ma.append(p.ma)
        self.mb.append(p.mb)
        self.mc.append(p.mc)

        self.battery_current.append(m.battery_current)
        self.load_power.append(sample.load_power)

    def arrays(self):
        """Return history as a dictionary of NumPy arrays."""
        return {
            name: np.asarray(value, dtype=np.float64)
            for name, value in vars(self).items()
        }


class MatlabFullBenchmark:
    """Runnable top-level translated MATLAB generated-code benchmark."""

    def __init__(
        self,
        params: BenchmarkParameters = PARAMS,
        config: BenchmarkConfig | None = None,
        scenario: BenchmarkScenario | None = None,
    ):
        self.params = params
        self.config = config or BenchmarkConfig()
        self.scenario = scenario or BenchmarkScenario()

        self.plant = GeneratedElectricalPlant(
            load_power_w=self.scenario.load_initial
        )
        self.controller = GeneratedGFMController(params)
        self.pwm = GeneratedPWM(params)

        self.state = create_initial_state()
        self.history = BenchmarkHistory()

        # Gate values applied/stored at startup.  The generated plant itself
        # contains one-sample gate storage (gState), so these are simply the
        # command passed during the first top-level evaluation.
        self.gates = np.zeros(6, dtype=np.int8)
        self.current_load_power = int(self.scenario.load_initial)
        self._load_step_applied = False
        self.reset()

    def reset(self):
        from load_models import get_load_model

        initial = get_load_model(self.scenario.load_initial)

        # Select the initial generated network, then initialize x from that
        # model's own generated X0.  This is only done at episode reset.
        self.plant.set_load_model(self.scenario.load_initial)
        self.plant.reset()

        self.state = create_initial_state()
        self.state.network.x[:] = initial.X0

        self.history = BenchmarkHistory()
        self.gates = np.zeros(6, dtype=np.int8)
        self.current_load_power = int(self.scenario.load_initial)
        self._load_step_applied = False
        return self.state

    def _apply_scheduled_disturbances(self):
        """Apply one-shot scenario events before evaluating the current sample."""
        s = self.scenario

        if (
            not self._load_step_applied
            and s.load_step_time is not None
            and self.state.time >= s.load_step_time
        ):
            # Crucially, set_load_model() does NOT reset state.network.x.
            self.plant.set_load_model(s.load_final)
            self.current_load_power = int(s.load_final)
            self._load_step_applied = True

    def source_abc(self, time: float) -> tuple[float, float, float]:
        """Ideal MATLAB upstream source supplied to generated SPS network."""
        cfg = self.config
        phase_peak = (
            cfg.grid_voltage_ll_rms
            / math.sqrt(3.0)
            * math.sqrt(2.0)
        )
        theta = (
            2.0 * math.pi * cfg.grid_frequency * time
            + math.radians(cfg.grid_phase_deg)
        )
        va = phase_peak * math.sin(theta)
        vb = phase_peak * math.sin(theta - 2.0 * math.pi / 3.0)
        vc = phase_peak * math.sin(theta + 2.0 * math.pi / 3.0)
        return va, vb, vc

    def _plant_input(self, gates=None) -> PlantInput:
        va, vb, vc = self.source_abc(self.state.time)
        return PlantInput(
            dc_voltage=self.config.dc_voltage,
            source_a=va,
            source_b=vb,
            source_c=vc,
            gates=self.gates if gates is None else gates,
        )

    def observe(self) -> tuple[PlantOutput, BenchmarkMeasurements]:
        """Evaluate current generated plant and decode MATLAB measurements."""
        plant_input = self._plant_input()
        plant_output = self.plant.output(self.state.network, plant_input)
        measurements = measurement_step(
            plant_output,
            theta=self.state.controller.theta,
        )
        return plant_output, measurements

    def step(self) -> BenchmarkSample:
        """Execute one 10-us generated benchmark sample."""
        self._apply_scheduled_disturbances()
        t = self.state.time

        # 1) Generated electrical-network outputs using stored switch state.
        plant_input = self._plant_input()
        plant_output = self.plant.output(self.state.network, plant_input)

        # 2) B1/B3 measurements in the GFM rotating frame.
        measurements = measurement_step(
            plant_output,
            theta=self.state.controller.theta,
        )

        # 3) GFM controller outputs from current persistent states.
        controller_output = self.controller.evaluate(
            self.state.controller,
            measurements,
        )

        # 4) Generated 5-kHz PWM and six gate commands.
        pwm_output = self.pwm.step(t, controller_output)

        # 5) Advance controller persistent states.
        controller_next = self.controller.update(
            self.state.controller,
            measurements,
            controller_output,
        )

        # 6) Advance generated 15-state SPS network.  plant.step stores the
        #    newly generated gates into gState for the next sample, matching
        #    the generated update ordering.
        next_network, _ = self.plant.step(
            self.state.network,
            PlantInput(
                dc_voltage=self.config.dc_voltage,
                source_a=plant_input.source_a,
                source_b=plant_input.source_b,
                source_c=plant_input.source_c,
                gates=pwm_output.gates,
            ),
        )

        sample = BenchmarkSample(
            time=t,
            plant=plant_output,
            measurements=measurements,
            controller=controller_output,
            pwm=pwm_output,
            load_power=float(self.current_load_power),
        )
        self.history.append(sample)

        # 7) Commit top-level state/time.
        self.state.network = next_network
        self.state.controller = controller_next
        self.state.time = t + self.params.dt
        self.state.step_count += 1
        self.gates = pwm_output.gates.copy()

        return sample

    def run(self, duration: float, progress_every: int | None = None):
        """Run the benchmark for ``duration`` seconds and return history."""
        if duration <= 0.0:
            raise ValueError("duration must be positive")

        n_steps = int(round(duration / self.params.dt))
        for k in range(n_steps):
            self.step()
            if progress_every and (k + 1) % progress_every == 0:
                print(
                    f"step {k + 1}/{n_steps}  "
                    f"t={self.state.time:.6f} s"
                )
        return self.history


if __name__ == "__main__":
    sim = MatlabFullBenchmark()
    history = sim.run(1.0e-4)

    a = history.arrays()
    print("MATLAB full benchmark top-level sanity check")
    print("steps     :", sim.state.step_count)
    print("time      :", sim.state.time)
    print("finite P  :", np.isfinite(a["P"]).all())
    print("finite Q  :", np.isfinite(a["Q"]).all())
    print("finite f  :", np.isfinite(a["frequency"]).all())
    print("last f    :", a["frequency"][-1])
    print("last P    :", a["P"][-1])
    print("last Q    :", a["Q"][-1])
