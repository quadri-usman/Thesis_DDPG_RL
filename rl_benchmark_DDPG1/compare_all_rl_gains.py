"""Compare DDPG, TD3, SAC and PPO adaptive PI gain trajectories."""
import argparse, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AGENTS=("DDPG","TD3","SAC","PPO")
NOMINAL={"Kpv":2.0,"Kiv":14.0,"Kpi":2.0,"Kii":20.0}
EVENTS=(2.0,3.5,5.0,6.5,9.0,11.0,12.0)

def load(path):
    with open(path,newline="") as f: rows=list(csv.DictReader(f))
    if not rows: raise ValueError(f"{path} is empty")
    need=("time_s","Kpv","Kiv","Kpi","Kii")
    miss=[k for k in need if k not in rows[0]]
    if miss: raise ValueError(f"{path}: missing {miss}; columns={list(rows[0])}")
    d={k:np.array([float(r[k]) for r in rows]) for k in need}
    order=np.argsort(d["time_s"])
    return {k:v[order] for k,v in d.items()}

def savecsv(path,rows):
    with open(path,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser()
    for a in ("ddpg","td3","sac","ppo"): p.add_argument("--"+a,required=True)
    p.add_argument("--output_dir",default="all_rl_gain_comparison")
    p.add_argument("--report_start",type=float,default=1.5)
    p.add_argument("--report_end",type=float,default=15.0)
    a=p.parse_args()
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    paths={"DDPG":a.ddpg,"TD3":a.td3,"SAC":a.sac,"PPO":a.ppo}
    data={name:load(path) for name,path in paths.items()}
    for name,d in data.items():
        print(f"{name}: {len(d['time_s'])} samples, t={d['time_s'][0]:.3f}-{d['time_s'][-1]:.3f}s")
    for gain,k0 in NOMINAL.items():
        fig,ax=plt.subplots(figsize=(13,6))
        ax.axhline(k0,ls="--",lw=1.5,label=f"Fixed PI nominal ({k0:g})")
        for name in AGENTS:
            d=data[name];m=(d["time_s"]>=a.report_start)&(d["time_s"]<=a.report_end)
            if not np.any(m): raise RuntimeError(f"No {name} samples in report interval")
            ax.plot(d["time_s"][m],d[gain][m],lw=1.7,label=name)
        for te in EVENTS: ax.axvline(te,ls=":",lw=.8)
        ax.set_xlim(a.report_start,a.report_end);ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"{gain} gain");ax.set_title(f"{gain}: adaptive-gain comparison")
        ax.grid(True,alpha=.3);ax.legend(ncol=3);fig.tight_layout()
        fig.savefig(out/f"all_agents_{gain}.png",dpi=300);plt.close(fig)
    rows=[]
    for gain,k0 in NOMINAL.items():
        rows.append(dict(controller="Fixed PI",gain=gain,nominal_gain=k0,min_gain=k0,
                         max_gain=k0,mean_gain=k0,median_gain=k0,std_gain=0.0,
                         final_gain=k0,mean_ratio_to_nominal=1.0,final_ratio_to_nominal=1.0))
    for name in AGENTS:
        d=data[name];m=(d["time_s"]>=a.report_start)&(d["time_s"]<=a.report_end)
        for gain,k0 in NOMINAL.items():
            v=d[gain][m]
            rows.append(dict(controller=name,gain=gain,nominal_gain=k0,
                min_gain=float(v.min()),max_gain=float(v.max()),mean_gain=float(v.mean()),
                median_gain=float(np.median(v)),std_gain=float(v.std()),final_gain=float(v[-1]),
                mean_ratio_to_nominal=float(v.mean()/k0),final_ratio_to_nominal=float(v[-1]/k0)))
    savecsv(out/"gain_summary.csv",rows)
    print("Created:")
    for n in ("all_agents_Kpv.png","all_agents_Kiv.png","all_agents_Kpi.png","all_agents_Kii.png","gain_summary.csv"):
        print(out/n)
if __name__=="__main__": main()
