"""Corrected final fixed-topology MATLAB scenario benchmark.

Source of truth: GFM_scenariosNew_ert_rtw.zip. This version uses the
regenerated SPS matrices/X0 and the corrected initially CLOSED grid breaker
(breaker group G2).

This runner reproduces the breaker command schedules embedded in
GFM_scenarios_ert_rtw rather than reconstructing the events from the frequency
plot.  The seven 3-phase breaker groups use the exact generated lookup-table
transition times and values.

Generated lookup schedules (group commands)
-------------------------------------------
G0: 1 [0,2) -> 0 [2,9) -> 1 [9,11) -> 0 [11,12) -> 1 thereafter
G1: 0 [0,2) -> 1 [2,3.5) -> 0 thereafter
G2: 1 [0,11) -> 0 [11,12) -> 1 thereafter
G3: 0 [0,3.5) -> 1 [3.5,5) -> 0 thereafter
G4: 0 [0,5) -> 1 [5,6.5) -> 0 thereafter
G5: 0 [0,6.5) -> 1 [6.5,9) -> 0 thereafter
G6: 1 [0,2) -> 0 [2,9) -> 1 thereafter

These group labels are intentionally neutral.  The generated C++ proves the
command trajectories, but not every physical breaker name can be inferred from
the lookup arrays alone.  The complete combination is what reproduces MATLAB.

The user-described operating conditions are logged separately:
  0-2 s      : 50 kW, grid connected
  2-3.5 s    : 52.5 kW, grid connected
  3.5-5 s    : 55 kW, grid connected
  5-6.5 s    : 60 kW, grid connected
  6.5-9 s    : 70 kW, grid connected
  9-11 s     : 50 kW, grid connected
  11-12 s    : 10 kW, grid disconnected
  >=12 s     : 50 kW, grid connected
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import numpy as np

from parameters import PARAMS, BenchmarkParameters
from states import create_initial_state
from controller_rl_gains import (
    GeneratedGFMController,
    ControllerOutput,
    PIGains,
)
from pwm import GeneratedPWM, PWMOutput
from battery_dc import GeneratedBatteryDC

from scenario_plant_exact_new import (
    GeneratedScenarioElectricalPlant,
    ScenarioPlantInput,
    ScenarioPlantOutput,
)
from scenario_measurements_new import ScenarioMeasurements, measurement_step


@dataclass(frozen=True)
class ScenarioConfig:
    grid_voltage_ll_rms: float = 13_800.0
    grid_frequency: float = 60.0
    grid_phase_deg: float = 0.0
    dc_voltage: float = 800.0


@dataclass(frozen=True)
class ScenarioSample:
    time: float
    plant: ScenarioPlantOutput
    measurements: ScenarioMeasurements
    controller: ControllerOutput
    pwm: PWMOutput
    breaker_groups: np.ndarray
    nominal_load_power: float
    nominal_grid_connected: bool
    vdc: float
    battery_soc: float
    battery_filtered_current: float
    battery_charge_ah: float
    battery_voltage_loss: float


@dataclass
class ScenarioHistory:
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
    voa: list = field(default_factory=list)
    vob: list = field(default_factory=list)
    voc: list = field(default_factory=list)

    ioa: list = field(default_factory=list)
    iob: list = field(default_factory=list)
    ioc: list = field(default_factory=list)

    ifa: list = field(default_factory=list)
    ifb: list = field(default_factory=list)
    ifc: list = field(default_factory=list)

    vd_command: list = field(default_factory=list)
    vq_command: list = field(default_factory=list)
    md: list = field(default_factory=list)
    mq: list = field(default_factory=list)

    battery_current: list = field(default_factory=list)
    vdc: list = field(default_factory=list)
    battery_soc: list = field(default_factory=list)
    battery_filtered_current: list = field(default_factory=list)
    battery_charge_ah: list = field(default_factory=list)
    battery_voltage_loss: list = field(default_factory=list)
    aux_d: list = field(default_factory=list)
    aux_q: list = field(default_factory=list)

    nominal_load_power: list = field(default_factory=list)
    nominal_grid_connected: list = field(default_factory=list)

    breaker_g0: list = field(default_factory=list)
    breaker_g1: list = field(default_factory=list)
    breaker_g2: list = field(default_factory=list)
    breaker_g3: list = field(default_factory=list)
    breaker_g4: list = field(default_factory=list)
    breaker_g5: list = field(default_factory=list)
    breaker_g6: list = field(default_factory=list)

    def append(self, s: ScenarioSample):
        m=s.measurements; c=s.controller
        self.time.append(s.time)
        self.frequency.append(c.frequency); self.omega.append(c.omega); self.theta.append(c.theta)
        self.P.append(m.P); self.Q.append(m.Q)
        self.P_filtered.append(c.P_filtered); self.Q_filtered.append(c.Q_filtered)
        self.vod.append(m.vod); self.voq.append(m.voq); self.vod_ref.append(c.vod_ref)
        self.iod.append(m.iod); self.ioq.append(m.ioq)
        self.ifd.append(m.ifd); self.ifq.append(m.ifq)
        self.ifd_ref.append(c.ifd_ref); self.ifq_ref.append(c.ifq_ref)
        self.voa.append(m.vb3_abc[0])
        self.vob.append(m.vb3_abc[1])
        self.voc.append(m.vb3_abc[2])
        self.ioa.append(m.ib3_abc[0])
        self.iob.append(m.ib3_abc[1])
        self.ioc.append(m.ib3_abc[2])
        self.ifa.append(m.ib1_abc[0])
        self.ifb.append(m.ib1_abc[1])
        self.ifc.append(m.ib1_abc[2])
        self.vd_command.append(c.vd_command); self.vq_command.append(c.vq_command)
        self.md.append(c.md); self.mq.append(c.mq)
        self.battery_current.append(m.battery_current)
        self.vdc.append(s.vdc)
        self.battery_soc.append(s.battery_soc)
        self.battery_filtered_current.append(s.battery_filtered_current)
        self.battery_charge_ah.append(s.battery_charge_ah)
        self.battery_voltage_loss.append(s.battery_voltage_loss)
        self.aux_d.append(m.aux_d); self.aux_q.append(m.aux_q)
        self.nominal_load_power.append(s.nominal_load_power)
        self.nominal_grid_connected.append(1.0 if s.nominal_grid_connected else 0.0)
        for i in range(7):
            getattr(self,f"breaker_g{i}").append(float(s.breaker_groups[i]))

    def arrays(self):
        return {k:np.asarray(v,dtype=np.float64) for k,v in vars(self).items()}


def breaker_group_commands(t: float) -> np.ndarray:
    """Exact seven generated lookup-table command trajectories."""
    # Use event boundaries as represented by the Simulink scenario.
    g0 = 1 if (t < 2.0 or 9.0 <= t < 11.0 or t >= 12.0) else 0
    g1 = 1 if 2.0 <= t < 3.5 else 0
    g2 = 0 if 11.0 <= t < 12.0 else 1
    g3 = 1 if 3.5 <= t < 5.0 else 0
    g4 = 1 if 5.0 <= t < 6.5 else 0
    g5 = 1 if 6.5 <= t < 9.0 else 0
    g6 = 1 if (t < 2.0 or t >= 9.0) else 0
    return np.array([g0,g1,g2,g3,g4,g5,g6],dtype=np.int8)


def nominal_operating_condition(t: float):
    """Human-readable scenario interpretation supplied with the MATLAB model."""
    if t < 2.0:   return 50_000.0, True
    if t < 3.5:   return 52_500.0, True
    if t < 5.0:   return 55_000.0, True
    if t < 6.5:   return 60_000.0, True
    if t < 9.0:   return 70_000.0, True
    if t < 11.0:  return 50_000.0, True
    if t < 12.0:  return 10_000.0, False
    return 50_000.0, True


class MatlabScenarioBenchmark:
    def __init__(
        self,
        params: BenchmarkParameters=PARAMS,
        config: ScenarioConfig|None=None,
    ):
        self.params=params
        self.config=config or ScenarioConfig()
        self.plant=GeneratedScenarioElectricalPlant()
        self.controller=GeneratedGFMController(params)
        self.pwm=GeneratedPWM(params)
        self.battery=GeneratedBatteryDC()
        self.reset()

    def reset(self):
        self.plant.reset()
        self.battery.reset()
        self.state=create_initial_state()
        # The final scenario plant owns its exact generated X0.
        self.state.network=self.plant.initial_state()
        self.history=ScenarioHistory()
        self.inverter_gates=np.zeros(6,dtype=np.int8)
        return self.state

    def source_abc(self,t):
        peak=self.config.grid_voltage_ll_rms/math.sqrt(3.0)*math.sqrt(2.0)
        th=2*math.pi*self.config.grid_frequency*t+math.radians(self.config.grid_phase_deg)
        return (
            peak*math.sin(th),
            peak*math.sin(th-2*math.pi/3),
            peak*math.sin(th+2*math.pi/3),
        )

    def _input(self,t,inverter_gates=None,dc_voltage=None):
        va,vb,vc=self.source_abc(t)
        return ScenarioPlantInput(
            dc_voltage=(self.battery.output_voltage() if dc_voltage is None else dc_voltage),
            source_a=va,source_b=vb,source_c=vc,
            breaker_gates=breaker_group_commands(t),
            inverter_gates=self.inverter_gates if inverter_gates is None else inverter_gates,
        )

    def step(self):
        t=self.state.time

        # Generated battery Unit Delay supplies Vdc[k] to the SPS plant.
        vdc_k = self.battery.output_voltage()
        inp=self._input(t,dc_voltage=vdc_k)

        # Exact generated ordering: resolve the 27-switch SPS network ONCE.
        # The returned StateSpace_o1 is used by measurements/controller for
        # this same sample.
        plant_out=self.plant.solve_switches(self.state.network,inp)
        m=measurement_step(plant_out,theta=self.state.controller.theta)
        c=self.controller.evaluate(self.state.controller,m)
        p=self.pwm.step(t,c)
        controller_next=self.controller.update(self.state.controller,m,c)

        # Advance x[k+1] with the already-resolved As/Bs. Do not invoke the
        # switch solver a second time.
        next_network=self.plant.advance_state(self.state.network,inp)

        # Generated update stores current breaker and PWM commands only after
        # the electrical state update, for use by the next 10-us sample.
        gate_input=self._input(t,inverter_gates=p.gates,dc_voltage=vdc_k)
        self.plant.store_gates_for_next_step(gate_input)

        # StateSpace_o1[36] is the generated battery/DC current.  Update the
        # battery only after the current sample has used Vdc[k], producing
        # Vdc[k+1] through the generated Unit Delay.
        battery_out = self.battery.step(m.battery_current)

        load,grid=nominal_operating_condition(t)
        groups=breaker_group_commands(t)

        sample=ScenarioSample(
            time=t,plant=plant_out,measurements=m,controller=c,pwm=p,
            breaker_groups=groups,
            nominal_load_power=load,
            nominal_grid_connected=grid,
            vdc=float(vdc_k),
            battery_soc=float(battery_out.soc_percent),
            battery_filtered_current=float(battery_out.filtered_current),
            battery_charge_ah=float(battery_out.charge_ah),
            battery_voltage_loss=float(battery_out.voltage_loss),
        )
        self.history.append(sample)

        self.state.network=next_network
        self.state.controller=controller_next
        self.state.time=t+self.params.dt
        self.state.step_count+=1
        self.inverter_gates=p.gates.copy()
        return sample

    def run(self,duration,progress_every=None):
        if duration<=0: raise ValueError("duration must be positive")
        n=int(round(duration/self.params.dt))
        for k in range(n):
            self.step()
            if progress_every and (k+1)%progress_every==0:
                print(f"step {k+1}/{n}  t={self.state.time:.6f} s")
        return self.history


if __name__=="__main__":
    print("Exact generated breaker schedule:")
    for t in [0,2,3.5,5,6.5,9,11,12,13]:
        print(f"t={t:4.1f}: groups={breaker_group_commands(t).tolist()}, "
              f"condition={nominal_operating_condition(t)}")

    sim=MatlabScenarioBenchmark()
    h=sim.run(1e-4)
    a=h.arrays()
    print("sanity finite:",np.isfinite(a["frequency"]).all(),np.isfinite(a["P"]).all())
    print("last frequency:",a["frequency"][-1])
