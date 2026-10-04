"""27-switch fixed-topology SPS plant translated from GFM_scenarios C++."""
from dataclasses import dataclass
import numpy as np
from states import ElectricalNetworkState
from scenario_parameters import A,B,C,D,X0,N_STATE,N_INPUT,N_OUTPUT,SWITCH_CONDUCTANCE,MAX_SWITCH_ITERATIONS

N_BREAKER_SWITCH=21
N_INVERTER_SWITCH=6
N_SWITCH=27
N_BREAKER_GROUPS=7
SWITCH_TYPES=np.array([2]*21+[7]*6,dtype=np.int32)

# Exact generated initial gate memory: breaker groups 0 and 6 closed.
INITIAL_BREAKER_GATES=np.array(
    [1,1,1, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 1,1,1],
    dtype=np.int8
)
INITIAL_INVERTER_GATES=np.zeros(6,dtype=np.int8)

def expand_breaker_groups(groups):
    g=np.asarray(groups,dtype=np.int8).reshape(N_BREAKER_GROUPS)
    return np.repeat(g,3)

@dataclass
class ScenarioPlantInput:
    dc_voltage: float
    source_a: float
    source_b: float
    source_c: float
    breaker_gates: np.ndarray
    inverter_gates: np.ndarray

    def __post_init__(self):
        b=np.asarray(self.breaker_gates,dtype=np.int8).reshape(-1)
        if b.size==7:
            b=expand_breaker_groups(b)
        if b.size!=21:
            raise ValueError("breaker_gates must have 7 group or 21 pole commands")
        self.breaker_gates=b
        self.inverter_gates=np.asarray(self.inverter_gates,dtype=np.int8).reshape(6)

    @property
    def gates(self):
        return np.concatenate((self.breaker_gates,self.inverter_gates))

    def state_space_input(self):
        # Generated C++: 21 SwitchCurrents (zero), 6 literal zeros,
        # then Vdc, source A, source B, source C.
        u=np.zeros(N_INPUT,dtype=np.float64)
        u[27]=self.dc_voltage
        u[28]=self.source_a
        u[29]=self.source_b
        u[30]=self.source_c
        return u

@dataclass
class ScenarioPlantOutput:
    y: np.ndarray
    switch_states: np.ndarray
    gate_states: np.ndarray

    @property
    def breaker_states(self):
        return self.switch_states[:21].copy()

    @property
    def breaker_group_states(self):
        return self.breaker_states.reshape(7,3)

    @property
    def inverter_switch_states(self):
        return self.switch_states[21:].copy()

class GeneratedScenarioElectricalPlant:
    def __init__(self):
        self.base_A=np.array(A,dtype=np.float64,copy=True)
        self.base_B=np.array(B,dtype=np.float64,copy=True)
        self.base_C=np.array(C,dtype=np.float64,copy=True)
        self.base_D=np.array(D,dtype=np.float64,copy=True)
        self.As=self.base_A.copy(); self.Bs=self.base_B.copy()
        self.Cs=self.base_C.copy(); self.Ds=self.base_D.copy()
        self.switch_status=np.zeros(27,dtype=np.int32)
        self.g_state=np.concatenate((INITIAL_BREAKER_GATES,INITIAL_INVERTER_GATES)).astype(np.int32)
        self.last_switch_outputs=np.zeros(27,dtype=np.float64)

    @staticmethod
    def initial_state():
        return ElectricalNetworkState(x=np.array(X0,dtype=np.float64,copy=True))

    def reset(self):
        self.As[:]=self.base_A; self.Bs[:]=self.base_B
        self.Cs[:]=self.base_C; self.Ds[:]=self.base_D
        self.switch_status.fill(0)
        self.g_state[:]=np.concatenate((INITIAL_BREAKER_GATES,INITIAL_INVERTER_GATES))
        self.last_switch_outputs.fill(0)

    @staticmethod
    def _output(Cm,Dm,x,u):
        return Cm@x+Dm@u

    def _apply_one_switch_change(self,i,change):
        a1=SWITCH_CONDUCTANCE*float(change)
        temp=1.0/(1.0-self.Ds[i,i]*a1)
        dx=self.Ds[:,i].copy()*temp*a1
        dx[i]=temp
        bd=self.Bs[:,i].copy()*a1
        rC=self.Cs[i,:].copy(); rD=self.Ds[i,:].copy()
        self.Cs[i,:]=0.0; self.Ds[i,:]=0.0
        self.Cs += np.outer(dx,rC)
        self.Ds += np.outer(dx,rD)
        self.As += np.outer(bd,self.Cs[i,:])
        self.Bs += np.outer(bd,self.Ds[i,:])

    def _rebuild(self,status):
        status=np.asarray(status,dtype=np.int32).reshape(27).copy()
        self.As[:]=self.base_A; self.Bs[:]=self.base_B
        self.Cs[:]=self.base_C; self.Ds[:]=self.base_D
        self.switch_status.fill(0)
        for i,v in enumerate(status):
            if v:
                self._apply_one_switch_change(i,1)
                self.switch_status[i]=1

    def solve_switches(self,state,inputs):
        x=state.x; u=inputs.state_space_input()
        initial=self.switch_status.copy()
        uswlast=self.last_switch_outputs.copy()
        loops=MAX_SWITCH_ITERATIONS
        while True:
            y=self._output(self.Cs,self.Ds,x,u)
            new=self.switch_status.copy()
            for i in range(27):
                yi=float(y[i]); gate=int(self.g_state[i])
                if SWITCH_TYPES[i]==2:
                    new[i]=1 if gate>0 else (0 if yi*uswlast[i]<0 else self.switch_status[i])
                else:
                    if ((yi>0) and (gate>0)) or yi<0: new[i]=1
                    elif yi>0 and gate==0: new[i]=0
                    else: new[i]=self.switch_status[i]
            changes=new-self.switch_status
            self.switch_status[:]=new
            if not np.any(changes):
                return y
            for i,ch in enumerate(changes):
                if ch: self._apply_one_switch_change(i,int(ch))
            loops-=1
            if loops<=0:
                g=self.g_state.copy(); last=self.last_switch_outputs.copy()
                self._rebuild(initial); self.g_state[:]=g; self.last_switch_outputs[:]=last
                return self._output(self.Cs,self.Ds,x,u)

    def output(self,state,inputs):
        y=self.solve_switches(state,inputs)
        return ScenarioPlantOutput(y.copy(),self.switch_status.copy(),self.g_state.copy())

    def step(self,state,inputs):
        out=self.output(state,inputs)
        u=inputs.state_space_input()
        nxt=ElectricalNetworkState(x=self.As@state.x+self.Bs@u)
        self.last_switch_outputs[:]=out.y[:27]
        self.g_state[:]=inputs.gates
        return nxt,out

if __name__=="__main__":
    p=GeneratedScenarioElectricalPlant()
    s=p.initial_state()
    inp=ScenarioPlantInput(800.0,0.0,0.0,0.0,INITIAL_BREAKER_GATES,np.zeros(6,dtype=np.int8))
    sn,o=p.step(s,inp)
    print("A/B/C/D topology: 15 states, 31 inputs, 40 outputs")
    print("switches:",o.switch_states.shape,"(21 breaker + 6 inverter)")
    print("breaker groups:",o.breaker_group_states.shape)
    print("initial group commands:",INITIAL_BREAKER_GATES.reshape(7,3)[:,0])
    print("finite:",np.isfinite(o.y).all(),np.isfinite(sn.x).all())
