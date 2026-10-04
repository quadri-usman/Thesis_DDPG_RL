"""Stage-1 target-gain PI adapter: 0.1x to 10x nominal.

The RL action is an ABSOLUTE normalized log-gain target, not an increment:
    a_i = -1 -> 0.1 x nominal
    a_i =  0 -> 1.0 x nominal
    a_i = +1 -> 10  x nominal

The internal normalized gain follows the target with a first-order update:
    z[k+1] = z[k] + alpha * (a[k] - z[k])
    K[k+1] = K_nominal * 10**z[k+1]

This removes cumulative gain drift: a constant action converges to a finite
physical gain and then stops changing. The validated GFM PI equations are
unchanged.
"""
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
    target_tracking_alpha: float=0.002
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
    """Smooth absolute log-gain target adapter.

    The internal state z is the normalized base-10 logarithm of the gains:
        z = log10(K / K_nominal)
    and therefore lies in [-1, +1] for the configured 0.1x--10x bounds.
    """
    def __init__(self,initial_gains=NOMINAL_GAINS,bounds=GainBounds()):
        self.bounds=bounds
        self._gains=initial_gains
        self._normalized=self.bounds.normalized_log_gains(initial_gains).astype(float)
        self._previous_action=np.zeros(4,float)

    @property
    def gains(self): return self._gains

    @property
    def previous_action(self): return self._previous_action.copy()

    @property
    def target_tracking_alpha(self): return float(self.bounds.target_tracking_alpha)

    def reset(self,gains=NOMINAL_GAINS):
        self._gains=gains
        self._normalized=self.bounds.normalized_log_gains(gains).astype(float)
        self._previous_action.fill(0.0)
        return self._gains

    def normalized_gains(self):
        return self._normalized.copy()

    def target_gains(self,action):
        """Return the physical gains requested by an action without changing state."""
        a=np.asarray(action,float).reshape(4)
        if not np.all(np.isfinite(a)):
            raise ValueError("RL gain action contains NaN/Inf")
        a=np.clip(a,-1.0,1.0)
        k=NOMINAL_GAINS.as_array()*np.power(10.0,a)
        lo,hi=self.bounds.arrays()
        return PIGains(*map(float,np.clip(k,lo,hi)))

    def apply_action(self,action):
        a=np.asarray(action,float).reshape(4)
        if not np.all(np.isfinite(a)):
            raise ValueError("RL gain action contains NaN/Inf")
        a=np.clip(a,-1.0,1.0)

        alpha=self.target_tracking_alpha
        if not (0.0 < alpha <= 1.0):
            raise ValueError("target_tracking_alpha must be in (0, 1]")

        # First-order tracking in normalized log10 gain space.
        self._normalized += alpha*(a-self._normalized)
        self._normalized=np.clip(self._normalized,-1.0,1.0)

        k=NOMINAL_GAINS.as_array()*np.power(10.0,self._normalized)
        lo,hi=self.bounds.arrays()
        k=np.clip(k,lo,hi)

        self._gains=PIGains(*map(float,k))
        self._previous_action[:]=a
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
    b=GainBounds()
    print("Nominal:",NOMINAL_GAINS.as_array())
    print("Minimum:",b.arrays()[0])
    print("Maximum:",b.arrays()[1])
    print("tracking alpha:",b.target_tracking_alpha)

    for u in (0.0,0.5,-0.5,1.0,-1.0):
        a=RLGainAdapter()
        target=a.target_gains([u]*4).as_array()
        for _ in range(2000):
            a.apply_action([u]*4)
        print(f"action {u:+.1f} target   :",target)
        print(f"action {u:+.1f} achieved :",a.gains.as_array())
        before=a.gains.as_array().copy()
        for _ in range(1000):
            a.apply_action([u]*4)
        print("additional drift max:",float(np.max(np.abs(a.gains.as_array()-before))))
