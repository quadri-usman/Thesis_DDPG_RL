"""Train wide-range Stage-1 DDPG for adaptive GFM PI gains.

TRAIN loads:      50, 55, 70 kW
VALIDATION loads: 50, 55, 70 kW with held-out seeds/timings
TEST loads:       52.5 and 60 kW remain completely withheld

Observation (12):
[df, RoCoF, evd, Evd, evq, eid, Eid, eiq,
 logKpv, logKiv, logKpi, logKii]

Action (4):
normalized multiplicative/log-space gain commands.

Gain range:
0.1x to 10x nominal.

This trainer logs:
* reward components
* frequency deviation / RoCoF / settling
* final AND episode min/max gains
* safety terminations and reasons
* deterministic validation checkpoints

TEST split is never instantiated here.
"""

from __future__ import annotations
import argparse, csv, json, math, time
from collections import Counter
from pathlib import Path
import numpy as np

from gfm_env_random_refined import RandomizedGFMGainTuningEnv
from controller_rl_gains_refined import GainBounds, nominal_target_action, refined_target_action
from ddpg_agent_wide import DDPGAgent, DDPGConfig

REWARD_KEYS=("J_frequency","J_rocof","J_settling",
             "J_voltage","J_current","J_gain_motion")
GAIN_KEYS=("Kpv","Kiv","Kpi","Kii")

def empty_envelope():
    return {"max_voltage_dq_v":0.0,"max_output_current_dq_a":0.0,
            "max_filter_current_dq_a":0.0,"max_modulation":0.0}

def update_envelope(e,info):
    e["max_voltage_dq_v"]=max(e["max_voltage_dq_v"],float(info["voltage_dq_magnitude"]))
    e["max_output_current_dq_a"]=max(e["max_output_current_dq_a"],float(info["output_current_dq_magnitude"]))
    e["max_filter_current_dq_a"]=max(e["max_filter_current_dq_a"],float(info["filter_current_dq_magnitude"]))
    e["max_modulation"]=max(e["max_modulation"],float(info["modulation_peak"]))

def make_env(split):
    if split not in ("train","validation"):
        raise ValueError("Trainer may use only train or validation split")
    return RandomizedGFMGainTuningEnv(split=split, gain_bounds=GainBounds(target_tracking_alpha=0.01))

def write_csv(path,rows):
    if not rows:return
    with open(path,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

def run_evaluation(agent,seeds):
    rows=[]
    safety_counts=Counter()
    for seed in seeds:
        env=make_env("validation")
        s,info=env.reset(seed=int(seed))
        total=0.; maxdf=0.; maxr=0.; sts=[]; last=np.nan
        gains={k:[] for k in GAIN_KEYS}; envelope=empty_envelope()
        term=trunc=False
        while not(term or trunc):
            a=agent.select_action(s,explore=False) if info["rl_active"] else np.zeros(4,np.float32)
            ns,r,term,trunc,ni=env.step(a)
            update_envelope(envelope,ni)
            if ni["rl_active"]:
                total+=r
                maxdf=max(maxdf,abs(ni["frequency_deviation"]))
                maxr=max(maxr,abs(ni["rocof"]))
                for k in GAIN_KEYS:gains[k].append(ni[k])
                st=ni["last_settling_time"]
                if np.isfinite(st) and (not np.isfinite(last) or not np.isclose(st,last)):
                    sts.append(float(st)); last=float(st)
            s,info=ns,ni
        if info.get("safety_terminated",False):
            safety_counts[info.get("safety_reason","unknown")]+=1
        row={"seed":int(seed),"reward":float(total),
             "max_abs_df":float(maxdf),"max_abs_rocof":float(maxr),
             "mean_settling_time":float(np.mean(sts)) if sts else math.nan,
             "safety_terminated":bool(info.get("safety_terminated",False)),
             "safety_reason":info.get("safety_reason",""),**envelope}
        for k in GAIN_KEYS:
            x=np.asarray(gains[k],float)
            row[k+"_final"]=float(info[k])
            row[k+"_min"]=float(np.min(x)) if len(x) else float(info[k])
            row[k+"_max"]=float(np.max(x)) if len(x) else float(info[k])
        rows.append(row)

    def mf(k):
        x=np.asarray([r[k] for r in rows],float); x=x[np.isfinite(x)]
        return float(np.mean(x)) if len(x) else math.nan
    return {"mean_reward":mf("reward"),"mean_max_abs_df":mf("max_abs_df"),
            "mean_max_abs_rocof":mf("max_abs_rocof"),
            "mean_settling_time":mf("mean_settling_time"),
            "mean_max_voltage_dq_v":float(np.mean([r["max_voltage_dq_v"] for r in rows])),
            "mean_max_output_current_dq_a":float(np.mean([r["max_output_current_dq_a"] for r in rows])),
            "mean_max_filter_current_dq_a":float(np.mean([r["max_filter_current_dq_a"] for r in rows])),
            "mean_max_modulation":float(np.mean([r["max_modulation"] for r in rows])),
            "safety_terminations":int(sum(safety_counts.values())),
            "safety_reasons":dict(safety_counts),"rows":rows}

def run_scripted_baseline(seeds, optimized=False):
    rows=[]
    for seed in seeds:
        env=make_env("validation"); s,info=env.reset(seed=int(seed))
        action=(refined_target_action(env.gain_bounds) if optimized else nominal_target_action(env.gain_bounds))
        total=0.; maxdf=maxr=0.; sts=[]; last=np.nan; envelope=empty_envelope(); term=trunc=False
        while not(term or trunc):
            a=action if info["rl_active"] else np.zeros(4,np.float32)
            ns,r,term,trunc,ni=env.step(a); update_envelope(envelope,ni)
            if ni["rl_active"]:
                total+=r; maxdf=max(maxdf,abs(ni["frequency_deviation"])); maxr=max(maxr,abs(ni["rocof"]))
                st=ni["last_settling_time"]
                if np.isfinite(st) and (not np.isfinite(last) or not np.isclose(st,last)): sts.append(float(st)); last=float(st)
            s,info=ns,ni
        rows.append({"seed":int(seed),"reward":float(total),"max_abs_df":float(maxdf),"max_abs_rocof":float(maxr),
                     "mean_settling_time":float(np.mean(sts)) if sts else math.nan,
                     "safety_terminated":bool(info.get("safety_terminated",False)),"safety_reason":info.get("safety_reason",""),**envelope})
    def mf(k):
        x=np.asarray([r[k] for r in rows],float); x=x[np.isfinite(x)]; return float(np.mean(x)) if len(x) else math.nan
    return {"mean_max_abs_df":mf("max_abs_df"),"mean_max_abs_rocof":mf("max_abs_rocof"),
            "mean_settling_time":mf("mean_settling_time"),"safety_terminations":sum(r["safety_terminated"] for r in rows),"rows":rows}

def score_vs(candidate, baseline):
    ratios={"df":candidate["mean_max_abs_df"]/baseline["mean_max_abs_df"],
            "rocof":candidate["mean_max_abs_rocof"]/baseline["mean_max_abs_rocof"],
            "Ts":candidate["mean_settling_time"]/baseline["mean_settling_time"]}
    S=0.4*ratios["df"]+0.2*ratios["rocof"]+0.4*ratios["Ts"] if all(np.isfinite(list(ratios.values()))) else math.inf
    eligible=bool(candidate["safety_terminations"]==0 and np.isfinite(S) and S<1 and max(ratios.values())<=1.10)
    return ratios,float(S),eligible

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--episodes",type=int,default=100)
    ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--output_dir",default="ddpg_target_results")
    ap.add_argument("--eval_every",type=int,default=10)
    ap.add_argument("--checkpoint_every",type=int,default=10)
    ap.add_argument("--updates_per_step",type=int,default=1)
    ap.add_argument("--eval_seeds",type=int,nargs="+",default=[1001,1002,1003])
    ap.add_argument("--device",default=None)
    args=ap.parse_args()

    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    cfg=DDPGConfig(seed=args.seed)
    agent=DDPGAgent(cfg,device=args.device)
    nominal_base=run_scripted_baseline(args.eval_seeds,optimized=False)
    optimized_base=run_scripted_baseline(args.eval_seeds,optimized=True)
    write_csv(out/"nominal_fixed_pi_validation.csv",nominal_base["rows"])
    write_csv(out/"optimized_fixed_pi_validation.csv",optimized_base["rows"])

    config={"episodes":args.episodes,"training_seed":args.seed,
            "training_loads_kw":[50,55,70],
            "validation_loads_kw":[50,55,70],
            "withheld_test_loads_kw":[52.5,60],
            "state_dim":12,"action_dim":4,
            "gain_range":"gain-specific: Kpv[1,4], Kiv[1.4,14], Kpi[0.4,2], Kii[2,20]",
            "target_tracking_alpha":0.01,
            "optimized_fixed_pi":[2,1.4,0.7,2],
            "gain_action":"absolute gain-specific normalized logarithmic target",
            "eval_seeds":args.eval_seeds,
            "ddpg":{k:(list(v) if isinstance(v,tuple) else v)
                    for k,v in vars(cfg).items()}}
    (out/"training_config.json").write_text(json.dumps(config,indent=2))

    print("="*100)
    print("STAGE-2 DDPG PHYSICALLY-INFORMED TARGET-GAIN TRAINING")
    print("="*100)
    print("device              :",agent.device)
    print("episodes            :",args.episodes)
    print("state/action dims   :",cfg.state_dim,cfg.action_dim)
    print("gain range          : gain-specific refined bounds")
    print("nominal gains       : [2, 14, 2, 20]")
    print("physical bounds     : Kpv[1,4], Kiv[1.4,14], Kpi[0.4,2], Kii[2,20]")
    print("tracking alpha      : 0.01")
    print("optimized Fixed PI  : [2, 1.4, 0.7, 2]")
    print("training loads      : 50, 55, 70 kW")
    print("validation loads    : 50, 55, 70 kW (held-out scenarios)")
    print("TEST loads withheld : 52.5, 60 kW")
    print("learning starts     :",cfg.learning_starts)
    print("evaluation seeds    :",args.eval_seeds)
    print("="*100)

    hist=[]; evalhist=[]; best_nom=math.inf; best_opt=math.inf
    global_steps=0; total_safety=Counter(); wall=time.time()

    for ep in range(1,args.episodes+1):
        env=make_env("train"); seed=args.seed+ep-1
        s,info=env.reset(seed=seed)
        R=0.; comps={k:0. for k in REWARD_KEYS}; nrl=0
        maxdf=maxroc=0.; sts=[]; lastst=np.nan
        aloss=[]; closs=[]; gains={k:[] for k in GAIN_KEYS}; envelope=empty_envelope()
        term=trunc=False

        while not(term or trunc):
            a=agent.select_action(s,explore=True) if info["rl_active"] else np.zeros(4,np.float32)
            ns,r,term,trunc,ni=env.step(a)
            update_envelope(envelope,ni)

            if ni["rl_active"]:
                # Store the transition even if safety termination occurred.
                # Terminal transitions do not bootstrap.
                agent.remember(s,a,r,ns,bool(term))
                global_steps+=1; nrl+=1; R+=r
                maxdf=max(maxdf,abs(ni["frequency_deviation"]))
                maxroc=max(maxroc,abs(ni["rocof"]))
                for k in REWARD_KEYS: comps[k]+=float(ni[k])
                for k in GAIN_KEYS:gains[k].append(float(ni[k]))
                st=ni["last_settling_time"]
                if np.isfinite(st) and (not np.isfinite(lastst) or not np.isclose(st,lastst)):
                    sts.append(float(st)); lastst=float(st)
                for _ in range(args.updates_per_step):
                    m=agent.update()
                    if m:
                        aloss.append(m["actor_loss"]); closs.append(m["critic_loss"])
            s,info=ns,ni

        safety=bool(info.get("safety_terminated",False))
        reason=info.get("safety_reason","")
        if safety: total_safety[reason or "unknown"]+=1

        row={"episode":ep,"seed":seed,"rl_steps":nrl,"reward":float(R),
             "max_abs_df":float(maxdf),"max_abs_rocof":float(maxroc),
             "mean_settling_time":float(np.mean(sts)) if sts else math.nan,
             "safety_terminated":safety,"safety_reason":reason,
             "exploration_std":float(agent.exploration_std),
             "actor_loss":float(np.mean(aloss)) if aloss else math.nan,
             "critic_loss":float(np.mean(closs)) if closs else math.nan,**envelope}
        for k in GAIN_KEYS:
            x=np.asarray(gains[k],float)
            row[k+"_final"]=float(info[k])
            row[k+"_min"]=float(np.min(x)) if len(x) else float(info[k])
            row[k+"_max"]=float(np.max(x)) if len(x) else float(info[k])
        for k in REWARD_KEYS:row[k+"_sum"]=comps[k]
        hist.append(row); write_csv(out/"training_history.csv",hist)
        agent.save(out/"latest.pt")

        flag=(f" SAFETY={reason} V={envelope['max_voltage_dq_v']:.1f}"
              f" Io={envelope['max_output_current_dq_a']:.1f}"
              f" If={envelope['max_filter_current_dq_a']:.1f}"
              f" mod={envelope['max_modulation']:.1f}") if safety else ""
        print(f"ep={ep:4d}/{args.episodes} seed={seed:5d} R={R:11.2f} "
              f"|df|max={maxdf:7.4f} Hz |RoCoF|max={maxroc:8.3f} Hz/s "
              f"K=[{info['Kpv']:.3f},{info['Kiv']:.3f},{info['Kpi']:.3f},{info['Kii']:.3f}] "
              f"sigma={agent.exploration_std:.3f}{flag}")

        if ep%args.checkpoint_every==0:
            agent.save(out/f"checkpoint_ep{ep:04d}.pt")

        if ep%args.eval_every==0:
            ev=run_evaluation(agent,args.eval_seeds)
            er={"episode":ep,"mean_reward":ev["mean_reward"],
                "mean_max_abs_df":ev["mean_max_abs_df"],
                "mean_max_abs_rocof":ev["mean_max_abs_rocof"],
                "mean_settling_time":ev["mean_settling_time"],
                "mean_max_voltage_dq_v":ev["mean_max_voltage_dq_v"],
                "mean_max_output_current_dq_a":ev["mean_max_output_current_dq_a"],
                "mean_max_filter_current_dq_a":ev["mean_max_filter_current_dq_a"],
                "mean_max_modulation":ev["mean_max_modulation"],
                "safety_terminations":ev["safety_terminations"],
                "safety_reasons":json.dumps(ev["safety_reasons"],sort_keys=True)}
            evalhist.append(er); write_csv(out/"evaluation_history.csv",evalhist)
            print(f"  EVAL R={ev['mean_reward']:.2f} |df|max={ev['mean_max_abs_df']:.4f} Hz "
                  f"|RoCoF|max={ev['mean_max_abs_rocof']:.3f} Hz/s "
                  f"Ts={ev['mean_settling_time']:.4f} s safety={ev['safety_terminations']}")

            rn,Sn,en=score_vs(ev,nominal_base); ro,So,eo=score_vs(ev,optimized_base)
            print(f"  EVAL vs nominal   S={Sn:.4f} eligible={en}")
            print(f"       |df|max : {nominal_base['mean_max_abs_df']:.5f} -> {ev['mean_max_abs_df']:.5f} Hz ({100*(1-rn['df']):+.2f}%)")
            print(f"       |RoCoF| : {nominal_base['mean_max_abs_rocof']:.4f} -> {ev['mean_max_abs_rocof']:.4f} Hz/s ({100*(1-rn['rocof']):+.2f}%)")
            print(f"       Ts      : {nominal_base['mean_settling_time']:.4f} -> {ev['mean_settling_time']:.4f} s ({100*(1-rn['Ts']):+.2f}%)")
            print(f"  EVAL vs optimized S={So:.4f} eligible={eo}")
            print(f"       |df|max : {optimized_base['mean_max_abs_df']:.5f} -> {ev['mean_max_abs_df']:.5f} Hz ({100*(1-ro['df']):+.2f}%)")
            print(f"       |RoCoF| : {optimized_base['mean_max_abs_rocof']:.4f} -> {ev['mean_max_abs_rocof']:.4f} Hz/s ({100*(1-ro['rocof']):+.2f}%)")
            print(f"       Ts      : {optimized_base['mean_settling_time']:.4f} -> {ev['mean_settling_time']:.4f} s ({100*(1-ro['Ts']):+.2f}%)")
            if en and Sn<best_nom:
                best_nom=Sn; agent.save(out/"best.pt"); write_csv(out/"best_evaluation_seeds.csv",ev["rows"]); print("  -> new best policy BEATING nominal Fixed PI saved as best.pt")
            if eo and So<best_opt:
                best_opt=So; agent.save(out/"best_vs_optimized.pt"); write_csv(out/"best_vs_optimized_seeds.csv",ev["rows"]); print("  -> new best policy BEATING optimized Fixed PI saved")

    print("\\n"+"="*100)
    print("STAGE-2 DDPG REFINED TRAINING COMPLETE")
    print("RL-active transitions:",global_steps)
    print("gradient updates      :",agent.total_updates)
    print("training safety stops :",sum(total_safety.values()),dict(total_safety))
    print("best S vs nominal     :",best_nom if np.isfinite(best_nom) else "NONE")
    print("best S vs optimized   :",best_opt if np.isfinite(best_opt) else "NONE")
    print("wall time             :",f"{(time.time()-wall)/60:.2f} min")
    print("results               :",out)
    print("="*100)

if __name__=="__main__":
    main()
