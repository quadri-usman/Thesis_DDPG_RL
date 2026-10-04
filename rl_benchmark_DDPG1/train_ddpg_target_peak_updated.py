"""Train refined Stage-1 DDPG with peak-aware target-gain adaptation.

TRAIN loads:      50, 55, 70 kW
VALIDATION loads: 50, 55, 70 kW with held-out seeds/timings
TEST loads:       52.5 and 60 kW remain completely withheld

Observation (12):
[df, RoCoF, evd, Evd, evq, eid, Eid, eiq,
 logKpv, logKiv, logKpi, logKii]

Action (4):
absolute normalized log-gain targets in [-1,1]^4.

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

from gfm_env_random_target_peak import RandomizedGFMGainTuningEnv
from controller_rl_gains_target import GainBounds
from ddpg_agent_wide import DDPGAgent, DDPGConfig

REWARD_KEYS=("J_frequency","J_rocof","J_settling",
    "J_frequency_peak",
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
    return RandomizedGFMGainTuningEnv(split=split, gain_bounds=GainBounds(target_tracking_alpha=0.002))

def write_csv(path,rows):
    if not rows:return
    with open(path,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

def run_validation_episode(agent,seed,adaptive):
    env=make_env("validation"); state,info=env.reset(seed=int(seed))
    total=maxdf=maxr=0.0; sts=[]; last=np.nan; gains={k:[] for k in GAIN_KEYS}; term=trunc=False
    while not(term or trunc):
        action=agent.select_action(state,explore=False) if info["rl_active"] and adaptive else np.zeros(4,np.float32)
        ns,r,term,trunc,ni=env.step(action)
        if ni["rl_active"]:
            total+=float(r); maxdf=max(maxdf,abs(float(ni["frequency_deviation"]))); maxr=max(maxr,abs(float(ni["rocof"])))
            for k in GAIN_KEYS:gains[k].append(float(ni[k]))
            st=float(ni["last_settling_time"])
            if np.isfinite(st) and (not np.isfinite(last) or not np.isclose(st,last)): sts.append(st); last=st
        state,info=ns,ni
    row={"seed":int(seed),"controller":"DDPG" if adaptive else "Fixed PI","reward":float(total),"max_abs_df":float(maxdf),"max_abs_rocof":float(maxr),"mean_settling_time":float(np.mean(sts)) if sts else math.nan,"completed_settling_events":len(sts),"safety_terminated":bool(info.get("safety_terminated",False)),"safety_reason":info.get("safety_reason","")}
    for k in GAIN_KEYS:
        x=np.asarray(gains[k],float); row[k+"_final"]=float(info[k]); row[k+"_min"]=float(np.min(x)) if len(x) else float(info[k]); row[k+"_max"]=float(np.max(x)) if len(x) else float(info[k])
    return row

def mf(rows,key):
    x=np.asarray([r[key] for r in rows],float); x=x[np.isfinite(x)]; return float(np.mean(x)) if len(x) else math.nan

def summarize(rows):
    return {"mean_reward":mf(rows,"reward"),"mean_max_abs_df":mf(rows,"max_abs_df"),"mean_max_abs_rocof":mf(rows,"max_abs_rocof"),"mean_settling_time":mf(rows,"mean_settling_time"),"safety_terminations":int(sum(bool(r["safety_terminated"]) for r in rows))}

def validate_against_fixed(agent,seeds,fixed_rows,fixed):
    rl_rows=[run_validation_episode(agent,x,True) for x in seeds]
    rl=summarize(rl_rows)
    ratios={
        "df":rl["mean_max_abs_df"]/fixed["mean_max_abs_df"],
        "rocof":rl["mean_max_abs_rocof"]/fixed["mean_max_abs_rocof"],
        "Ts":rl["mean_settling_time"]/fixed["mean_settling_time"],
    }
    diagnostic_score=(0.4*ratios["df"]+0.2*ratios["rocof"]+0.4*ratios["Ts"]
                      if all(np.isfinite(list(ratios.values()))) else math.inf)

    # Strict checkpoint selection: an unsafe or incomplete validation seed must
    # never look artificially good because mean_finite() dropped its missing Ts.
    incomplete=any(
        r["safety_terminated"] or not np.isfinite(r["mean_settling_time"])
        for r in rl_rows
    )
    selection_score=math.inf if incomplete else float(diagnostic_score)
    eligible=bool(
        not incomplete and np.isfinite(selection_score) and selection_score<1.0
        and max(ratios.values())<=1.10
    )
    return rl_rows,rl,ratios,float(diagnostic_score),float(selection_score),eligible,incomplete

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
    fixed_rows=[run_validation_episode(agent,x,False) for x in args.eval_seeds]
    fixed=summarize(fixed_rows)
    if fixed["safety_terminations"] or not all(np.isfinite([fixed["mean_max_abs_df"],fixed["mean_max_abs_rocof"],fixed["mean_settling_time"]])): raise RuntimeError("Invalid Fixed PI validation baseline")
    write_csv(out/"fixed_pi_validation_baseline.csv",fixed_rows)
    (out/"fixed_pi_validation_baseline.json").write_text(json.dumps(fixed,indent=2))

    config={"episodes":args.episodes,"training_seed":args.seed,
            "training_loads_kw":[50,55,70],
            "validation_loads_kw":[50,55,70],
            "withheld_test_loads_kw":[52.5,60],
            "state_dim":12,"action_dim":4,
            "gain_range":"0.1x to 10x nominal",
            "gain_action":"absolute normalized log-gain target","target_tracking_alpha":0.002,
            "eval_seeds":args.eval_seeds,
            "ddpg":{k:(list(v) if isinstance(v,tuple) else v)
                    for k,v in vars(cfg).items()}}
    (out/"training_config.json").write_text(json.dumps(config,indent=2))

    print("="*100)
    print("REFINED STAGE-1 DDPG TARGET-GAIN PEAK-AWARE TRAINING")
    print("="*100)
    print("device              :",agent.device)
    print("episodes            :",args.episodes)
    print("state/action dims   :",cfg.state_dim,cfg.action_dim)
    print("gain range          : 0.1x to 10x nominal")
    print("nominal gains       : [2, 14, 2, 20]")
    print("physical bounds     : Kpv[0.2,20], Kiv[1.4,140], Kpi[0.2,20], Kii[2,200]")
    print("training loads      : 50, 55, 70 kW")
    print("validation loads    : 50, 55, 70 kW (held-out scenarios)")
    print("TEST loads withheld : 52.5, 60 kW")
    print("learning starts     :",cfg.learning_starts)
    print("evaluation seeds    :",args.eval_seeds)
    print("="*100)

    hist=[]; evalhist=[]; best=math.inf
    global_steps=0; total_safety=Counter(); wall=time.time()

    for ep in range(1,args.episodes+1):
        env=make_env("train"); seed=args.seed+ep-1
        s,info=env.reset(seed=seed)
        R=0.; comps={k:0. for k in REWARD_KEYS}; nrl=0
        maxdf=maxroc=0.; sts=[]; lastst=np.nan
        aloss=[]; closs=[]
        gains={k:[] for k in GAIN_KEYS}; envelope=empty_envelope()
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
             "critic_loss":float(np.mean(closs)) if closs else math.nan,
             "gradient_updates_total":int(agent.total_updates),**envelope}
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
              f"sigma={agent.exploration_std:.4f}{flag}")

        if ep%args.checkpoint_every==0:
            agent.save(out/f"checkpoint_ep{ep:04d}.pt")

        if ep%args.eval_every==0:
            rl_rows,rl,ratios,score,selection_score,eligible,incomplete=validate_against_fixed(agent,args.eval_seeds,fixed_rows,fixed)
            er={"episode":ep,"score_S_diagnostic":score,"score_S_selection":selection_score,"eligible":eligible,"incomplete_validation":incomplete,"safety_terminations":rl["safety_terminations"],"fixed_df":fixed["mean_max_abs_df"],"ddpg_df":rl["mean_max_abs_df"],"ratio_df":ratios["df"],"df_improvement_pct":100*(1-ratios["df"]),"fixed_rocof":fixed["mean_max_abs_rocof"],"ddpg_rocof":rl["mean_max_abs_rocof"],"ratio_rocof":ratios["rocof"],"rocof_improvement_pct":100*(1-ratios["rocof"]),"fixed_Ts":fixed["mean_settling_time"],"ddpg_Ts":rl["mean_settling_time"],"ratio_Ts":ratios["Ts"],"settling_improvement_pct":100*(1-ratios["Ts"])}
            evalhist.append(er); write_csv(out/"evaluation_history.csv",evalhist)
            fixed_ep=[dict(r,episode=ep) for r in fixed_rows]
            ddpg_ep=[dict(r,episode=ep) for r in rl_rows]
            write_csv(out/f"validation_fixed_ep{ep:04d}.csv",fixed_ep)
            write_csv(out/f"validation_ddpg_ep{ep:04d}.csv",ddpg_ep)
            for vr in rl_rows:
                if vr["safety_terminated"] or not np.isfinite(vr["mean_settling_time"]):
                    print(f"       INVALID seed={vr['seed']} safety={vr['safety_terminated']} "
                          f"reason={vr['safety_reason'] or '-'} Ts={vr['mean_settling_time']}")
            print(f"  EVAL Sdiag={score:.4f} Sselect={selection_score:.4f} eligible={eligible} safety={rl['safety_terminations']}")
            print(f"       |df|max : {fixed['mean_max_abs_df']:.5f} -> {rl['mean_max_abs_df']:.5f} Hz ({100*(1-ratios['df']):+.2f}%)")
            print(f"       |RoCoF| : {fixed['mean_max_abs_rocof']:.4f} -> {rl['mean_max_abs_rocof']:.4f} Hz/s ({100*(1-ratios['rocof']):+.2f}%)")
            print(f"       Ts      : {fixed['mean_settling_time']:.4f} -> {rl['mean_settling_time']:.4f} s ({100*(1-ratios['Ts']):+.2f}%)")
            if eligible and selection_score<best:
                best=selection_score; agent.save(out/"best.pt"); write_csv(out/"best_evaluation_fixed.csv",fixed_rows); write_csv(out/"best_evaluation_ddpg.csv",rl_rows)
                (out/"best_evaluation.json").write_text(json.dumps({"episode":ep,"score_S_diagnostic":score,"score_S_selection":selection_score,"fixed_pi_score":1.0,"fixed":fixed,"ddpg":rl,"ratios":ratios},indent=2))
                print("  -> new best policy BEATING Fixed PI saved")

    print("\\n"+"="*100)
    print("REFINED TARGET-GAIN PEAK-AWARE TRAINING COMPLETE")
    print("RL-active transitions:",global_steps)
    print("gradient updates       :",agent.total_updates)
    print("training safety stops :",sum(total_safety.values()),dict(total_safety))
    print("Fixed PI score        : 1.0")
    print("best relative score S :",best if np.isfinite(best) else "NONE")
    print("best policy           :",out/"best.pt" if np.isfinite(best) else "NONE -- no checkpoint beat Fixed PI")
    print("wall time             :",f"{(time.time()-wall)/60:.2f} min")
    print("results               :",out)
    print("="*100)

if __name__=="__main__":
    main()
