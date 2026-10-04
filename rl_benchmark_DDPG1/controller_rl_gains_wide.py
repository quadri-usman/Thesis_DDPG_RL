"""Wide-range Stage-1 PI gain adapter: 0.1x to 10x nominal, log-space actions."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from parameters import PARAMS, BenchmarkParameters
from states import GFMControllerState

@dataclass(frozen=True)
class PIGains:
    voltage_kp: float=2.0
    voltage_ki: float=14.0
    current_kp: float=2.0
    current_ki: float=20.0
    def as_array(self):
        return np.array([self.voltage_kp,self.voltage_ki,self.current_kp,self.current_ki],float)

NOMINAL_GAINS=PIGains()

@dataclass(frozen=True)
class GainBounds:
    lower_factor: float=0.1
    upper_factor: float=10.0
    log_step: float=0.001
    def arrays(self):
        k=NOMINAL_GAINS.as_array()
        return self.lower_factor*k,self.upper_factor*k
    @property
    def minimum(self): return tuple(self.arrays()[0])
    @property
    def maximum(self): return tuple(self.arrays()[1])
    def normalized_log_gains(self,gains):
        return np.log(gains.as_array()/NOMINAL_GAINS.as_array())/math.log(10.0)

@dataclass(frozen=True)
class ControllerOutput:
    P:float; Q:float; P_filtered:float; Q_filtered:float
    omega:float; frequency:float; theta:float; vod_ref:float; voq_ref:float
    vd_error:float; vq_error:float; vd_pi:float; vq_pi:float
    ifd_ref:float; ifq_ref:float; id_error:float; iq_error:float
    id_pi:float; iq_pi:float; vd_command:float; vq_command:float
    md:float; mq:float; modulation_alpha:float; modulation_beta:float
    voltage_kp:float; voltage_ki:float; current_kp:float; current_ki:float

class RLGainAdapter:
    def __init__(self,initial_gains=NOMINAL_GAINS,bounds=GainBounds()):
        self.bounds=bounds; self._gains=initial_gains
        self._previous_action=np.zeros(4,float)
    @property
    def gains(self): return self._gains
    @property
    def previous_action(self): return self._previous_action.copy()
    def reset(self,gains=NOMINAL_GAINS):
        self._gains=gains; self._previous_action.fill(0); return self._gains
    def normalized_gains(self): return self.bounds.normalized_log_gains(self._gains)
    def apply_action(self,action):
        a=np.asarray(action,float).reshape(4)
        if not np.all(np.isfinite(a)): raise ValueError("RL gain action contains NaN/Inf")
        a=np.clip(a,-1,1); lo,hi=self.bounds.arrays()
        k=np.clip(self._gains.as_array()*np.exp(self.bounds.log_step*a),lo,hi)
        self._gains=PIGains(*map(float,k)); self._previous_action[:]=a
        return self._gains

class RLGainAdaptiveGFMController:
    def __init__(self,params:BenchmarkParameters=PARAMS,gains:PIGains=NOMINAL_GAINS):
        self.params=params; self.gains=gains
    def set_gains(self,gains):
        v=gains.as_array()
        if not np.all(np.isfinite(v)) or np.any(v<=0): raise ValueError("Invalid PI gains")
        self.gains=gains
    def evaluate(self,state:GFMControllerState,measurements):
        p=self.params; g=self.gains
        Pf=p.lpf_omega_c*state.p_filter_state; Qf=p.lpf_omega_c*state.q_filter_state
        omega=p.omega_nom-p.kp_droop*(Pf-p.P_ref); frequency=omega/(2*math.pi)
        vod_ref=p.voltage_ref-p.kq_droop*(Qf-p.Q_ref); voq_ref=0.0
        evd=vod_ref-measurements.vod; evq=-measurements.voq
        vd_pi=g.voltage_kp*evd+state.vd_integral; vq_pi=g.voltage_kp*evq+state.vq_integral
        ifd_ref=vd_pi+measurements.iod-omega*p.filter_C*measurements.voq
        ifq_ref=vq_pi+omega*p.filter_C*measurements.vod+measurements.ioq
        eid=ifd_ref-measurements.ifd; eiq=ifq_ref-measurements.ifq
        id_pi=g.current_kp*eid+state.id_integral; iq_pi=g.current_kp*eiq+state.iq_integral
        vd=id_pi+measurements.vod-omega*p.inverter_L*measurements.ifq
        vq=iq_pi+measurements.voq+omega*p.inverter_L*measurements.ifd
        md=p.modulation_gain*vd; mq=p.modulation_gain*vq
        s=math.sin(state.theta); c=math.cos(state.theta)
        alpha=md*s+mq*c; beta=-md*c+mq*s
        return ControllerOutput(float(measurements.P),float(measurements.Q),float(Pf),float(Qf),
          float(omega),float(frequency),float(state.theta),float(vod_ref),0.0,
          float(evd),float(evq),float(vd_pi),float(vq_pi),float(ifd_ref),float(ifq_ref),
          float(eid),float(eiq),float(id_pi),float(iq_pi),float(vd),float(vq),float(md),float(mq),
          float(alpha),float(beta),g.voltage_kp,g.voltage_ki,g.current_kp,g.current_ki)
    def update(self,state,measurements,output=None):
        if output is None: output=self.evaluate(state,measurements)
        p=self.params; g=self.gains; dt=p.dt
        pf=state.p_filter_state+dt*(measurements.P-p.lpf_omega_c*state.p_filter_state)
        qf=state.q_filter_state+dt*(measurements.Q-p.lpf_omega_c*state.q_filter_state)
        return GFMControllerState(theta=float(state.theta+dt*output.omega),
          p_filter_state=float(pf),q_filter_state=float(qf),
          vd_integral=float(state.vd_integral+dt*g.voltage_ki*output.vd_error),
          vq_integral=float(state.vq_integral+dt*g.voltage_ki*output.vq_error),
          id_integral=float(state.id_integral+dt*g.current_ki*output.id_error),
          iq_integral=float(state.iq_integral+dt*g.current_ki*output.iq_error))
    def step(self,state,measurements):
        o=self.evaluate(state,measurements); return self.update(state,measurements,o),o

GeneratedGFMController=RLGainAdaptiveGFMController

if __name__=="__main__":
    b=GainBounds(); print("Nominal:",NOMINAL_GAINS.as_array())
    print("Minimum:",b.arrays()[0]); print("Maximum:",b.arrays()[1])
    a=RLGainAdapter(); print("+1:",a.apply_action([1]*4).as_array())
    a.reset(); print("-1:",a.apply_action([-1]*4).as_array())
    print("normalized nominal:",a.normalized_gains())
