"""Stage-4 residual PI-gain adapter centered on optimized Fixed PI."""
from dataclasses import dataclass
import numpy as np

NOMINAL_GAINS=np.array([2.,14.,2.,20.],float)
OPTIMIZED_GAINS=np.array([2.,1.4,.7,2.],float)

@dataclass
class GainBounds:
    kpv_min:float=1.5; kpv_max:float=2.5
    kiv_min:float=1.4; kiv_max:float=3.0
    kpi_min:float=.5; kpi_max:float=1.0
    kii_min:float=2.0; kii_max:float=5.0
    target_tracking_alpha:float=.01
    @property
    def lower(self): return np.array([self.kpv_min,self.kiv_min,self.kpi_min,self.kii_min],float)
    @property
    def upper(self): return np.array([self.kpv_max,self.kiv_max,self.kpi_max,self.kii_max],float)
    @property
    def center(self): return OPTIMIZED_GAINS.copy()
    def __post_init__(self):
        if np.any(self.lower<=0) or np.any(self.upper<=self.lower): raise ValueError("Invalid residual gain bounds")
        if np.any(OPTIMIZED_GAINS<self.lower) or np.any(OPTIMIZED_GAINS>self.upper): raise ValueError("Optimized gains outside bounds")
        if not 0<float(self.target_tracking_alpha)<=1: raise ValueError("target_tracking_alpha must be in (0,1]")

@dataclass
class PhysicalGains:
    voltage_kp:float; voltage_ki:float; current_kp:float; current_ki:float
    def as_array(self): return np.array([self.voltage_kp,self.voltage_ki,self.current_kp,self.current_ki],float)

class RLGainAdapter:
    def __init__(self,bounds=None):
        self.bounds=bounds or GainBounds(); self._lo=self.bounds.lower; self._hi=self.bounds.upper
        self._c=self.bounds.center; self._alpha=float(self.bounds.target_tracking_alpha)
        self._gains=self._c.copy(); self._target=self._c.copy(); self._previous_action=np.zeros(4)
    @property
    def gains(self): return PhysicalGains(*map(float,self._gains))
    @property
    def target_gains(self): return PhysicalGains(*map(float,self._target))
    @property
    def gains_array(self): return self._gains.copy()
    @property
    def target_array(self): return self._target.copy()
    @property
    def previous_action(self): return self._previous_action.copy()
    @property
    def target_tracking_alpha(self): return self._alpha
    def reset(self):
        self._gains=self._c.copy(); self._target=self._c.copy(); self._previous_action.fill(0); return self.gains
    def action_to_target(self,action):
        a=np.asarray(action,float).reshape(-1)
        if a.size!=4 or np.any(~np.isfinite(a)): raise ValueError("Expected four finite residual actions")
        a=np.clip(a,-1,1); lc=np.log(self._c); ll=np.log(self._lo); lh=np.log(self._hi)
        return np.exp(np.where(a<=0,lc+(-a)*(ll-lc),lc+a*(lh-lc)))
    def target_to_action(self,target,clip=True):
        k=np.asarray(target,float).reshape(-1)
        if k.size!=4 or np.any(~np.isfinite(k)) or np.any(k<=0): raise ValueError("Expected four positive finite gains")
        if not clip and (np.any(k<self._lo) or np.any(k>self._hi)): raise ValueError("Target outside residual bounds")
        k=np.clip(k,self._lo,self._hi); lk=np.log(k); lc=np.log(self._c); ll=np.log(self._lo); lh=np.log(self._hi)
        a=np.zeros(4); below=k<self._c; above=k>self._c
        dlo=lc-ll; dhi=lh-lc
        m=below&(dlo>1e-15); a[m]=-(lc[m]-lk[m])/dlo[m]
        m=above&(dhi>1e-15); a[m]=(lk[m]-lc[m])/dhi[m]
        return np.clip(a,-1,1).astype(np.float32)
    def apply_action(self,action):
        a=np.asarray(action,float).reshape(-1)
        if a.size!=4 or np.any(~np.isfinite(a)): raise ValueError("Expected four finite residual actions")
        a=np.clip(a,-1,1); self._target=self.action_to_target(a)
        self._gains += self._alpha*(self._target-self._gains); self._gains=np.clip(self._gains,self._lo,self._hi)
        self._previous_action[:]=a; return self.gains
    def normalized_gain_state(self): return self.target_to_action(self._gains)
    def normalized_target_state(self): return self.target_to_action(self._target)
    def residual_from_optimized(self): return ((self._gains-self._c)/self._c).astype(np.float32)
    def set_physical_gains(self,gains):
        k=np.asarray(gains,float).reshape(-1)
        if k.size!=4 or np.any(k<self._lo) or np.any(k>self._hi): raise ValueError("Physical gains outside residual bounds")
        self._gains=k.copy(); self._target=k.copy(); self._previous_action=self.target_to_action(k).astype(float); return self.gains

def optimized_target_action(bounds=None): return RLGainAdapter(bounds).target_to_action(OPTIMIZED_GAINS)
def zero_residual_action(): return np.zeros(4,dtype=np.float32)

if __name__=="__main__":
    b=GainBounds(); a=RLGainAdapter(b); z=zero_residual_action()
    print("lower:",b.lower); print("optimized:",b.center); print("upper:",b.upper)
    print("zero target:",a.action_to_target(z)); print("zero state:",a.normalized_gain_state())
    assert np.allclose(a.action_to_target(z),OPTIMIZED_GAINS)
    assert np.allclose(a.action_to_target(-np.ones(4)),b.lower)
    assert np.allclose(a.action_to_target(np.ones(4)),b.upper)
    for _ in range(10000): a.apply_action(z)
    assert np.allclose(a.gains_array,OPTIMIZED_GAINS)
    print("PASS")
