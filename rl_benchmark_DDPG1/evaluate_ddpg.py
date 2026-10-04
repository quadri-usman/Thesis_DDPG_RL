"""Frozen DDPG vs fixed-PI evaluation on validation or unseen TEST loads."""
import argparse, csv, json, math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from gfm_env_random_split import RandomizedGFMGainTuningEnv
from training_scenarios_split import ScenarioSplit
from ddpg_agent import DDPGAgent, DDPGConfig

def run(agent,seed,split,adaptive):
    env=RandomizedGFMGainTuningEnv(split=ScenarioSplit(split))
    s,info=env.reset(seed=seed)
    R=[]; rec={k:[] for k in ["time","f","df","rocof","Kpv","Kiv","Kpi","Kii"]}
    sts=[]; last=np.nan; term=trunc=False
    while not(term or trunc):
        a=agent.select_action(s,False) if adaptive and info["rl_active"] else np.zeros(4,np.float32)
        ns,r,term,trunc,ni=env.step(a)
        if ni["rl_active"]:
            R.append(r)
            for k,v in [("time",ni["time"]),("f",ni["frequency"]),("df",ni["frequency_deviation"]),
                        ("rocof",ni["rocof"]),("Kpv",ni["Kpv"]),("Kiv",ni["Kiv"]),
                        ("Kpi",ni["Kpi"]),("Kii",ni["Kii"])]: rec[k].append(v)
            st=ni["last_settling_time"]
            if np.isfinite(st) and (not np.isfinite(last) or not np.isclose(st,last)):
                sts.append(st); last=st
        s,info=ns,ni
    rec={k:np.asarray(v,float) for k,v in rec.items()}
    m={"seed":seed,"controller":"DDPG" if adaptive else "Fixed PI",
       "max_abs_df_hz":float(np.max(np.abs(rec["df"]))),
       "rms_df_hz":float(np.sqrt(np.mean(rec["df"]**2))),
       "max_abs_rocof_hz_s":float(np.max(np.abs(rec["rocof"]))),
       "mean_settling_time_s":float(np.mean(sts)) if sts else math.nan,
       "episode_reward":float(np.sum(R))}
    return m,rec,env.scenario

def improve(b,d):
    return 100*(b-d)/b if np.isfinite(b) and np.isfinite(d) and b!=0 else math.nan

def write_csv(path,rows):
    if rows:
        with open(path,"w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--output_dir",default="ddpg_unseen_test")
    ap.add_argument("--seeds",nargs="+",type=int,default=[2001,2002,2003,2004,2005])
    ap.add_argument("--split",choices=["validation","test"],default="test")
    ap.add_argument("--device",default=None)
    a=ap.parse_args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    agent=DDPGAgent(DDPGConfig(),device=a.device); agent.load(a.checkpoint,load_optimizers=False)
    agent.actor.eval(); agent.critic.eval()
    print("="*88); print("FROZEN DDPG EVALUATION")
    print("split:",a.split," seeds:",a.seeds," exploration: OFF, learning: OFF")
    if a.split=="test": print("WITHHELD disturbances: 52.5 and 60 kW")
    rows=[]; comps=[]
    for seed in a.seeds:
        bm,br,sc=run(agent,seed,a.split,False); dm,dr,sc2=run(agent,seed,a.split,True)
        e1=[(e.time,e.load_power_w) for e in sc.events]; e2=[(e.time,e.load_power_w) for e in sc2.events]
        if e1!=e2: raise RuntimeError("Scenario mismatch between controllers")
        rows += [bm,dm]
        c={"seed":seed,"fixed_df":bm["max_abs_df_hz"],"ddpg_df":dm["max_abs_df_hz"],
           "df_improvement_pct":improve(bm["max_abs_df_hz"],dm["max_abs_df_hz"]),
           "fixed_rocof":bm["max_abs_rocof_hz_s"],"ddpg_rocof":dm["max_abs_rocof_hz_s"],
           "rocof_improvement_pct":improve(bm["max_abs_rocof_hz_s"],dm["max_abs_rocof_hz_s"]),
           "fixed_Ts":bm["mean_settling_time_s"],"ddpg_Ts":dm["mean_settling_time_s"],
           "settling_improvement_pct":improve(bm["mean_settling_time_s"],dm["mean_settling_time_s"])}
        comps.append(c)
        print(f"\nseed {seed} events:",[(round(t,4),p/1000) for t,p in e1])
        print(f"|df|max  {bm['max_abs_df_hz']:.5f} -> {dm['max_abs_df_hz']:.5f} Hz ({c['df_improvement_pct']:+.2f}%)")
        print(f"|R|max   {bm['max_abs_rocof_hz_s']:.4f} -> {dm['max_abs_rocof_hz_s']:.4f} Hz/s ({c['rocof_improvement_pct']:+.2f}%)")
        print(f"Ts       {bm['mean_settling_time_s']:.4f} -> {dm['mean_settling_time_s']:.4f} s ({c['settling_improvement_pct']:+.2f}%)")
        fig,ax=plt.subplots(figsize=(11,5)); ax.plot(br["time"],br["f"],label="Fixed PI"); ax.plot(dr["time"],dr["f"],label="DDPG")
        for e in sc.disturbance_events: ax.axvline(e.time,ls=":",lw=.8)
        ax.set(xlabel="Time (s)",ylabel="Frequency (Hz)",title=f"{a.split} seed {seed}: frequency"); ax.grid(alpha=.3); ax.legend(); fig.tight_layout()
        fig.savefig(out/f"seed_{seed}_frequency.png",dpi=200); plt.close(fig)
        fig,ax=plt.subplots(figsize=(11,5))
        for k in ["Kpv","Kiv","Kpi","Kii"]: ax.plot(dr["time"],dr[k],label=k)
        for e in sc.disturbance_events: ax.axvline(e.time,ls=":",lw=.8)
        ax.set(xlabel="Time (s)",ylabel="Gain",title=f"{a.split} seed {seed}: DDPG gains"); ax.grid(alpha=.3); ax.legend(); fig.tight_layout()
        fig.savefig(out/f"seed_{seed}_gains.png",dpi=200); plt.close(fig)
    write_csv(out/"controller_metrics.csv",rows); write_csv(out/"fixed_vs_ddpg.csv",comps)
    mean=lambda k: float(np.nanmean([x[k] for x in comps]))
    summary={"split":a.split,"checkpoint":a.checkpoint,"seeds":a.seeds,
             "mean_df_improvement_pct":mean("df_improvement_pct"),
             "mean_rocof_improvement_pct":mean("rocof_improvement_pct"),
             "mean_settling_improvement_pct":mean("settling_improvement_pct")}
    (out/"summary.json").write_text(json.dumps(summary,indent=2))
    print("\n"+"="*88); print("MEAN IMPROVEMENT")
    print(f"frequency deviation: {summary['mean_df_improvement_pct']:+.2f}%")
    print(f"RoCoF              : {summary['mean_rocof_improvement_pct']:+.2f}%")
    print(f"settling time      : {summary['mean_settling_improvement_pct']:+.2f}%")
    print("results:",out); print("="*88)
if __name__=="__main__": main()
