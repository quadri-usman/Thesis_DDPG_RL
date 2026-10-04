"""Initialization-only diagnostic for the exact GFM_scenarios SPS plant."""
import math
import numpy as np
from scenario_parameters import C,D,X0,N_INPUT
from scenario_plant_exact import GeneratedScenarioElectricalPlantExact,ScenarioPlantInput,INITIAL_BREAKER_GATES
from scenario_measurements import abc_to_dq,decode_state_space_outputs

def source_abc(t=0.0):
    peak=13800.0/math.sqrt(3.0)*math.sqrt(2.0)
    th=2*math.pi*60.0*t
    return np.array([peak*math.sin(th),peak*math.sin(th-2*math.pi/3),peak*math.sin(th+2*math.pi/3)])

def make_u():
    u=np.zeros(N_INPUT); u[27]=0.0; u[28:31]=source_abc(); return u

def report(name,y):
    r=decode_state_space_outputs(y)
    v=abc_to_dq(*r["vb3_abc"],0.0); io=abc_to_dq(*r["ib3_abc"],0.0); ii=abc_to_dq(*r["ib1_abc"],0.0)
    P=1.5*(v.d*io.d+v.q*io.q); Q=1.5*(v.q*io.d-v.d*io.q)
    print("\n"+"="*80); print(name); print("="*80)
    print("y[27:40] =",np.asarray(y[27:40]))
    print("Vabc =",r["vb3_abc"]); print("Iabc B3 =",r["ib3_abc"]); print("Iabc B1 =",r["ib1_abc"])
    print("Vod, Voq =",v.d,v.q); print("Iod, Ioq =",io.d,io.q); print("Ifd, Ifq =",ii.d,ii.q)
    print("P, Q =",P,Q)

def main():
    x=np.asarray(X0,float).copy(); u=make_u()
    print("X0 =",x); print("source abc =",source_abc())
    report("CASE A: raw C*X0 + D*u",np.asarray(C)@x+np.asarray(D)@u)

    pb=GeneratedScenarioElectricalPlantExact()
    pb.SwitchChange.fill(0); pb.SwitchChange[:21]=INITIAL_BREAKER_GATES
    pb._apply_switch_changes(); pb.switch_status[:21]=INITIAL_BREAKER_GATES
    report("CASE B: commanded breaker poles applied only",pb.Cs@x+pb.Ds@u)
    print("switch_status =",pb.switch_status)

    p=GeneratedScenarioElectricalPlantExact(); st=p.initial_state()
    vg=source_abc()
    inp=ScenarioPlantInput(0.0,vg[0],vg[1],vg[2],INITIAL_BREAKER_GATES,np.zeros(6,dtype=np.int32))
    out=p.solve_switches(st,inp)
    report("CASE C: first exact switch-resolution return",out.y)
    print("switch_status =",out.switch_states); print("gates used =",out.gate_states); print("loops remaining =",out.loops_remaining)

    yr=p.Cs@x+p.Ds@u
    report("CASE D: recalc with final Cs/Ds",yr)
    print("max |D-C| =",np.max(np.abs(yr-out.y)))
    print("(D-C)[27:40] =",yr[27:40]-out.y[27:40])

    p2=GeneratedScenarioElectricalPlantExact(); st2=p2.initial_state()
    nxt,out2=p2.step(st2,inp)
    report("CASE E: one complete plant.step output",out2.y)
    print("x[1] =",nxt.x)

    print("\nMATLAB t=0 TARGETS")
    print("Vod=188.8, Voq=-155.3, Iod=-3.252, Ioq=-3.402")
    print("Ifd=-0.2735, Ifq=0.2249, P=-128.9, Q=1721")

if __name__=="__main__":
    main()
