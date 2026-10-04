"""Refined absolute target-gain adapter for Stage-2 RL."""
from dataclasses import dataclass
import numpy as np

NOMINAL_GAINS=np.array([2.,14.,2.,20.],dtype=float)
REFINED_TARGET_GAINS=np.array([2.,1.4,.7,2.],dtype=float)

@dataclass
class GainBounds:
    kpv_min:float=1.; kpv_max:float=4.
    kiv_min:float=1.4; kiv_max:float=14.
    kpi_min:float=.4; kpi_max:float=2.
    kii_min:float=2.; kii_max:float=20.
    target_tracking_alpha:float=.01
    def __post_init__(self):
        if np.any(self.lower<=0) or np.any(self.upper<=self.lower): raise ValueError("Invalid gain bounds")
        if not 0<float(self.target_tracking_alpha)<=1: raise ValueError("target_tracking_alpha must be in (0,1]")
        if np.any(NOMINAL_GAINS<self.lower) or np.any(NOMINAL_GAINS>self.upper): raise ValueError("Nominal gains outside bounds")
    @property
    def lower(self): return np.array([self.kpv_min,self.kiv_min,self.kpi_min,self.kii_min],float)
    @property
    def upper(self): return np.array([self.kpv_max,self.kiv_max,self.kpi_max,self.kii_max],float)

@dataclass
class PhysicalGains:
    voltage_kp:float; voltage_ki:float; current_kp:float; current_ki:float
    def as_array(self): return np.array([self.voltage_kp,self.voltage_ki,self.current_kp,self.current_ki],float)

class RLGainAdapter:
    def __init__(self,bounds=None):
        self.bounds=bounds or GainBounds(); self._lo=self.bounds.lower; self._hi=self.bounds.upper
        self._llo=np.log(self._lo); self._lhi=np.log(self._hi); self._alpha=float(self.bounds.target_tracking_alpha)
        self._gains=NOMINAL_GAINS.copy(); self._target=NOMINAL_GAINS.copy()
    @property
    def gains(self): return PhysicalGains(*map(float,self._gains))
    @property
    def target_gains(self): return PhysicalGains(*map(float,self._target))
    @property
    def gains_array(self): return self._gains.copy()
    @property
    def target_array(self): return self._target.copy()
    def reset(self):
        self._gains=NOMINAL_GAINS.copy(); self._target=NOMINAL_GAINS.copy(); return self.gains
    def action_to_target(self,action):
        a=np.asarray(action,float).reshape(-1)
        if a.size!=4 or np.any(~np.isfinite(a)): raise ValueError("Expected four finite actions")
        u=.5*(np.clip(a,-1,1)+1)
        return np.exp(self._llo+u*(self._lhi-self._llo))
    def target_to_action(self,target,clip=True):
        k=np.asarray(target,float).reshape(-1)
        if k.size!=4 or np.any(~np.isfinite(k)) or np.any(k<=0): raise ValueError("Expected four positive finite gains")
        if not clip and (np.any(k<self._lo) or np.any(k>self._hi)): raise ValueError("Target outside refined bounds")
        k=np.clip(k,self._lo,self._hi); u=(np.log(k)-self._llo)/(self._lhi-self._llo)
        return np.clip(2*u-1,-1,1).astype(np.float32)
    def apply_action(self,action):
        self._target=self.action_to_target(action)
        self._gains += self._alpha*(self._target-self._gains)
        self._gains=np.clip(self._gains,self._lo,self._hi)
        return self.gains
    def normalized_gain_state(self):
        u=(np.log(self._gains)-self._llo)/(self._lhi-self._llo)
        return np.clip(2*u-1,-1,1).astype(np.float32)
    def normalized_target_state(self):
        u=(np.log(self._target)-self._llo)/(self._lhi-self._llo)
        return np.clip(2*u-1,-1,1).astype(np.float32)
    def set_physical_gains(self,gains):
        k=np.asarray(gains,float).reshape(-1)
        if k.size!=4 or np.any(k<self._lo) or np.any(k>self._hi): raise ValueError("Physical gains outside refined bounds")
        self._gains=k.copy(); self._target=k.copy(); return self.gains

def refined_target_action(bounds=None): return RLGainAdapter(bounds).target_to_action(REFINED_TARGET_GAINS)
def nominal_target_action(bounds=None): return RLGainAdapter(bounds).target_to_action(NOMINAL_GAINS)

if __name__=="__main__":
    b=GainBounds(); a=RLGainAdapter(b)
    an=nominal_target_action(b); ar=refined_target_action(b)
    print("lower:",b.lower); print("upper:",b.upper)
    print("nominal action:",an); print("refined action:",ar)
    print("decode nominal:",a.action_to_target(an)); print("decode refined:",a.action_to_target(ar))
    assert np.allclose(a.action_to_target(an),NOMINAL_GAINS,rtol=1e-6)
    assert np.allclose(a.action_to_target(ar),REFINED_TARGET_GAINS,rtol=1e-6)
    for _ in range(2000): a.apply_action(ar)
    print("final:",a.gains_array); assert np.allclose(a.gains_array,REFINED_TARGET_GAINS,rtol=1e-4)
    print("PASS")
