"""Main-study quantitative comparison: DDPG vs TD3 vs SAC vs PPO only.

Conventional settling time Ts60 is evaluated for all benchmark events EXCEPT
the 11--12 s grid-disconnection event.  Frequency-deviation and RoCoF metrics
still include all seven events.

For DDPG/TD3/SAC/PPO this script reports event-level values and min/mean/max
summary statistics. MATLAB and Fixed PI are intentionally excluded.

Input: each algorithm's benchmark_event_metrics.csv from its report-ready
evaluator.
"""

from __future__ import annotations
import argparse, csv, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AGENTS = {"DDPG":"ddpg", "TD3":"td3", "SAC":"sac", "PPO":"ppo"}
NAMES = tuple(AGENTS)
DISCONNECT_EVENT_TIME = 11.0


def read_csv(path):
    path=Path(path)
    if not path.exists(): raise FileNotFoundError(path)
    with path.open(newline="") as f: rows=list(csv.DictReader(f))
    if not rows: raise ValueError(f"Empty CSV: {path}")
    return rows


def write_csv(path,rows):
    if not rows:return
    with Path(path).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        w.writeheader();w.writerows(rows)


def num(r,key):
    v=str(r.get(key,"")).strip()
    return math.nan if v.lower() in ("","nan","none") else float(v)


def stats(vals):
    x=np.asarray(vals,float);x=x[np.isfinite(x)]
    if not len(x):return math.nan,math.nan,math.nan,0
    return float(np.min(x)),float(np.mean(x)),float(np.max(x)),int(len(x))


def first_available_key(row, candidates, metric, controller):
    for key in candidates:
        if key in row:
            return key
    raise KeyError(
        f"{controller}: could not find {metric}. Tried {candidates}. "
        f"Available columns: {list(row.keys())}"
    )


def parse(path,name):
    rows=read_csv(path); p=AGENTS[name]; first=rows[0]

    # Older TD3 report-ready exports may retain legacy ddpg_* metric names.
    # Detect the real adaptive columns instead of assuming an algorithm prefix.
    df_key=first_available_key(first,[
        f"{p}_max_abs_df_hz","ddpg_max_abs_df_hz","td3_max_abs_df_hz",
        "sac_max_abs_df_hz","ppo_max_abs_df_hz","rl_max_abs_df_hz",
        "max_abs_df_hz"],"maximum frequency deviation",name)
    rocof_key=first_available_key(first,[
        f"{p}_max_abs_rocof_hz_s","ddpg_max_abs_rocof_hz_s",
        "td3_max_abs_rocof_hz_s","sac_max_abs_rocof_hz_s",
        "ppo_max_abs_rocof_hz_s","rl_max_abs_rocof_hz_s",
        "max_abs_rocof_hz_s"],"maximum RoCoF",name)
    ts_key=first_available_key(first,[
        f"{p}_settling_time_s","ddpg_settling_time_s","td3_settling_time_s",
        "sac_settling_time_s","ppo_settling_time_s","rl_settling_time_s",
        "settling_time_s"],"settling time",name)

    print(f"{name}: metric columns -> df={df_key}, RoCoF={rocof_key}, Ts={ts_key}")

    out=[]
    for i,r in enumerate(rows):
        out.append({
            "controller":name,
            "event_index":int(float(r.get("event_index",i+1))),
            "event_time_s":num(r,"event_time_s"),
            "event":r.get("event",f"Event {i+1}"),
            "max_abs_df_hz":num(r,df_key),
            "max_abs_rocof_hz_s":num(r,rocof_key),
            "settling_time_s":num(r,ts_key),
        })
    return out

def grouped(path,labels,data,key,ylabel,title,indices=None):
    if indices is None:indices=range(len(labels))
    indices=list(indices);x=np.arange(len(indices));width=.19
    fig,ax=plt.subplots(figsize=(14,6))
    for j,name in enumerate(NAMES):
        vals=[data[name][i][key] for i in indices]
        ax.bar(x+(j-1.5)*width,vals,width,label=name)
    ax.set_xticks(x);ax.set_xticklabels([labels[i] for i in indices],rotation=25,ha="right")
    ax.set_ylabel(ylabel);ax.set_title(title);ax.grid(True,axis="y",alpha=.3)
    ax.legend(ncol=4);fig.tight_layout();fig.savefig(path,dpi=300);plt.close(fig)


def mean_plot(path,summary,key,ylabel,title):
    fig,ax=plt.subplots(figsize=(8.5,5.5))
    ax.bar([r["controller"] for r in summary],[r[key] for r in summary])
    ax.set_ylabel(ylabel);ax.set_title(title);ax.grid(True,axis="y",alpha=.3)
    fig.tight_layout();fig.savefig(path,dpi=300);plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    for n in NAMES:ap.add_argument(f"--{n.lower()}",required=True)
    ap.add_argument("--output_dir",default="rl_types_quantitative_no_disconnect")
    a=ap.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)

    data={n:parse(getattr(a,n.lower()),n) for n in NAMES}
    ref=data["DDPG"]
    for n in NAMES[1:]:
        if len(data[n])!=len(ref):raise RuntimeError(f"{n} event count differs from DDPG")
        for x,y in zip(ref,data[n]):
            if x["event_index"]!=y["event_index"]:raise RuntimeError(f"{n} event indexing differs")
    labels=[r["event"] for r in ref]
    ts_idx=[i for i,r in enumerate(ref) if not np.isclose(r["event_time_s"],DISCONNECT_EVENT_TIME)]

    grouped(out/"event_frequency_deviation.png",labels,data,"max_abs_df_hz",
            "Maximum |frequency deviation| (Hz)","RL algorithms: event frequency deviation")
    grouped(out/"event_rocof.png",labels,data,"max_abs_rocof_hz_s",
            "Maximum |RoCoF| (Hz/s)","RL algorithms: event RoCoF")
    grouped(out/"event_settling_time_excluding_disconnect.png",labels,data,"settling_time_s",
            "Settling time to 60 Hz band (s)",
            "RL algorithms: nominal-frequency settling (11-12 s disconnect excluded)",ts_idx)

    event_rows=[r for n in NAMES for r in data[n]]
    write_csv(out/"rl_event_metrics.csv",event_rows)

    summary=[]
    for n in NAMES:
        d=data[n]
        dfmin,dfmean,dfmax,dfn=stats([r["max_abs_df_hz"] for r in d])
        rmin,rmean,rmax,rn=stats([r["max_abs_rocof_hz_s"] for r in d])
        tsvals=[d[i]["settling_time_s"] for i in ts_idx]
        tsmin,tsmean,tsmax,tsn=stats(tsvals)
        summary.append({
            "controller":n,
            "df_min":dfmin,"df_mean":dfmean,"df_max":dfmax,
            "rocof_min":rmin,"rocof_mean":rmean,"rocof_max":rmax,
            "Ts60_min_excl_disconnect":tsmin,
            "Ts60_mean_excl_disconnect":tsmean,
            "Ts60_max_excl_disconnect":tsmax,
            "Ts60_settled_events_excl_disconnect":tsn,
            "Ts60_total_events_excl_disconnect":len(ts_idx),
        })
    write_csv(out/"rl_quantitative_summary_min_mean_max.csv",summary)

    mean_plot(out/"mean_frequency_deviation.png",summary,"df_mean",
              "Mean event maximum |frequency deviation| (Hz)","RL algorithms: mean frequency deviation")
    mean_plot(out/"mean_rocof.png",summary,"rocof_mean",
              "Mean event maximum |RoCoF| (Hz/s)","RL algorithms: mean RoCoF")
    mean_plot(out/"mean_settling_time_excluding_disconnect.png",summary,"Ts60_mean_excl_disconnect",
              "Mean settling time (s)","RL algorithms: mean nominal-frequency settling")

    print("="*135)
    print("RL-TYPE COMPARISON -- MIN / MEAN / MAX (NO MATLAB, NO FIXED PI)")
    print("="*135)
    print(f"{'Agent':7s} {'df min':>9s} {'df mean':>9s} {'df max':>9s} | "
          f"{'R min':>9s} {'R mean':>9s} {'R max':>9s} | "
          f"{'Ts min':>9s} {'Ts mean':>9s} {'Ts max':>9s} {'settled':>8s}")
    for r in summary:
        print(f"{r['controller']:7s} {r['df_min']:9.6f} {r['df_mean']:9.6f} {r['df_max']:9.6f} | "
              f"{r['rocof_min']:9.4f} {r['rocof_mean']:9.4f} {r['rocof_max']:9.4f} | "
              f"{r['Ts60_min_excl_disconnect']:9.4f} {r['Ts60_mean_excl_disconnect']:9.4f} "
              f"{r['Ts60_max_excl_disconnect']:9.4f} "
              f"{r['Ts60_settled_events_excl_disconnect']:2d}/{r['Ts60_total_events_excl_disconnect']}")
    print("="*135)


if __name__=="__main__":main()
