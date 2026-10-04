"""Stage-1 DDPG gain-adaptive GFM controller.

RL adapts only Kpv, Kiv, Kpi, Kii. Droop, feed-forward, decoupling,
modulation and controller state equations remain those of the validated
generated-code benchmark.
"""
from dataclasses import dataclass
import math
import numpy as np
from parameters import PARAMS, BenchmarkParameters
from states import GFMControllerState

@dataclass(frozen=True)
class PIGains:
    voltage_kp: float = 2.0
    voltage_ki: float = 14.0
    current_kp: float = 2.0
    current_ki: float = 20.0
    def as_array(self):
        return np.array([self.voltage_kp,self.voltage_ki,
                         self.current_kp,self.current_ki],dtype=float)

@dataclass(frozen=True)
class GainBounds:
    minimum: tuple = (1.0,7.0,1.0,10.0)
    maximum: tuple = (3.0,21.0,3.0,30.0)
    max_increment: tuple = (0.002,0.014,0.002,0.020)
    def arrays(self):
        return tuple(np.asarray(x,dtype=float) for x in
                     (self.minimum,self.maximum,self.max_increment))

@dataclass(frozen=True)
class ControllerOutput:
    P: float; Q: float; P_filtered: float; Q_filtered: float
    omega: float; frequency: float; theta: float
    vod_ref: float; voq_ref: float
    vd_error: float; vq_error: float; vd_pi: float; vq_pi: float
    ifd_ref: float; ifq_ref: float
    id_error: float; iq_error: float; id_pi: float; iq_pi: float
    vd_command: float; vq_command: float
    md: float; mq: float; modulation_alpha: float; modulation_beta: float
    voltage_kp: float; voltage_ki: float; current_kp: float; current_ki: float

class RLGainAdapter:
    """Normalized DDPG action -> bounded incremental gain changes."""
    def __init__(self,initial_gains=PIGains(),bounds=GainBounds()):
        self.bounds=bounds
        self._gains=initial_gains
        self._previous_action=np.zeros(4,dtype=float)
    @property
    def gains(self): return self._gains
    @property
    def previous_action(self): return self._previous_action.copy()
    def reset(self,gains=PIGains()):
        self._gains=gains; self._previous_action.fill(0.0); return self._gains
    def apply_action(self,action):
        a=np.asarray(action,dtype=float).reshape(4)
        if not np.all(np.isfinite(a)): raise ValueError("RL action contains NaN/Inf")
        a=np.clip(a,-1.0,1.0)
        lo,hi,inc=self.bounds.arrays()
        v=np.clip(self._gains.as_array()+a*inc,lo,hi)
        self._gains=PIGains(*map(float,v))
        self._previous_action[:]=a
        return self._gains

class RLGainAdaptiveGFMController:
    def __init__(self,params: BenchmarkParameters=PARAMS,gains:PIGains=PIGains()):
        self.params=params; self.gains=gains
    def set_gains(self,gains):
        v=gains.as_array()
        if not np.all(np.isfinite(v)) or np.any(v<0): raise ValueError("Invalid PI gains")
        self.gains=gains
    def evaluate(self,state:GFMControllerState,measurements):
        p=self.params; g=self.gains
        Pf=p.lpf_omega_c*state.p_filter_state
        Qf=p.lpf_omega_c*state.q_filter_state
        omega=p.omega_nom-p.kp_droop*(Pf-p.P_ref)
        frequency=omega/(2*math.pi)
        vod_ref=p.voltage_ref-p.kq_droop*(Qf-p.Q_ref); voq_ref=0.0
        evd=vod_ref-measurements.vod; evq=voq_ref-measurements.voq
        vd_pi=g.voltage_kp*evd+state.vd_integral
        vq_pi=g.voltage_kp*evq+state.vq_integral
        ifd_ref=vd_pi+measurements.iod-omega*p.filter_C*measurements.voq
        ifq_ref=vq_pi+omega*p.filter_C*measurements.vod+measurements.ioq
        eid=ifd_ref-measurements.ifd; eiq=ifq_ref-measurements.ifq
        id_pi=g.current_kp*eid+state.id_integral
        iq_pi=g.current_kp*eiq+state.iq_integral
        vd_cmd=id_pi+measurements.vod-omega*p.inverter_L*measurements.ifq
        vq_cmd=iq_pi+measurements.voq+omega*p.inverter_L*measurements.ifd
        md=p.modulation_gain*vd_cmd; mq=p.modulation_gain*vq_cmd
        s=math.sin(state.theta); c=math.cos(state.theta)
        alpha=md*s+mq*c; beta=-md*c+mq*s
        return ControllerOutput(
            float(measurements.P),float(measurements.Q),float(Pf),float(Qf),
            float(omega),float(frequency),float(state.theta),float(vod_ref),0.0,
            float(evd),float(evq),float(vd_pi),float(vq_pi),
            float(ifd_ref),float(ifq_ref),float(eid),float(eiq),
            float(id_pi),float(iq_pi),float(vd_cmd),float(vq_cmd),
            float(md),float(mq),float(alpha),float(beta),
            g.voltage_kp,g.voltage_ki,g.current_kp,g.current_ki)
    def update(self,state,measurements,output=None):
        if output is None: output=self.evaluate(state,measurements)
        p=self.params; g=self.gains; dt=p.dt
        Pf=state.p_filter_state+dt*(measurements.P-p.lpf_omega_c*state.p_filter_state)
        Qf=state.q_filter_state+dt*(measurements.Q-p.lpf_omega_c*state.q_filter_state)
        theta=state.theta+dt*output.omega
        # State stores Ki*integral(e dt), matching existing validated controller.
        vd=state.vd_integral+dt*g.voltage_ki*output.vd_error
        vq=state.vq_integral+dt*g.voltage_ki*output.vq_error
        iid=state.id_integral+dt*g.current_ki*output.id_error
        iiq=state.iq_integral+dt*g.current_ki*output.iq_error
        return GFMControllerState(theta=float(theta),p_filter_state=float(Pf),
            q_filter_state=float(Qf),vd_integral=float(vd),vq_integral=float(vq),
            id_integral=float(iid),iq_integral=float(iiq))
    def step(self,state,measurements):
        out=self.evaluate(state,measurements)
        return self.update(state,measurements,out),out

GeneratedGFMController=RLGainAdaptiveGFMController

if __name__=="__main__":
    a=RLGainAdapter()
    print("Nominal:",a.gains)
    print("Zero action:",a.apply_action([0,0,0,0]))
    print("Test action:",a.apply_action([1,-1,.5,-.5]))
    print("Stage-1 observation: [df, RoCoF, evd, Evd, evq, eid, Eid, eiq]")
