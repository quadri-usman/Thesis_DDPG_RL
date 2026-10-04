"""Combine standardized Fixed-PI / DDPG / TD3 / SAC / PPO benchmark exports.

Inputs are the matlab_benchmark directories produced by the report-ready evaluators.
The script does NOT rerun EMT simulations.

Produces
--------
combined_benchmark_frequency.png
combined_benchmark_frequency_rl_only.png
all_algorithms_summary.csv
all_algorithms_event_metrics.csv
event_improvement_table.csv

Optional:
  --matlab_frequency_csv Matlab_frequency_scenario.csv
"""

from __future__ import annotations
import argparse, csv, math
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

F_NOM = 60.0
REPORT_START_S = 1.5
EVENTS = [
    (2.0, "50->52.5 kW"),
    (3.5, "52.5->55 kW"),
    (5.0, "55->60 kW"),
    (6.5, "60->70 kW"),
    (9.0, "70->50 kW"),
    (11.0, "50->10 kW + grid disconnect"),
    (12.0, "10->50 kW + grid reconnect"),
]
ALGOS = {
    "DDPG": "ddpg_frequency_hz",
    "TD3": "td3_frequency_hz",
    "SAC": "sac_frequency_hz",
    "PPO": "ppo_frequency_hz",
}


def read_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def numeric_column(rows, key):
    try:
        return np.asarray([float(r[key]) for r in rows], dtype=float)
    except KeyError:
        raise KeyError(f"Column {key!r} not found. Available: {list(rows[0])}")


def load_frequency(path, algo):
    rows = read_csv(path)
    t = numeric_column(rows, "time_s")
    fixed = numeric_column(rows, "fixed_pi_frequency_hz")
    rl = numeric_column(rows, ALGOS[algo])
    order = np.argsort(t)
    return t[order], fixed[order], rl[order]


def load_matlab(path):
    if path is None:
        return None
    data = np.genfromtxt(path, delimiter=",", names=True)
    names = list(data.dtype.names or [])
    if len(names) < 2:
        raise ValueError("MATLAB CSV must contain time and frequency columns")
    tn = next((n for n in names if "time" in n.lower()), names[0])
    fn = next((n for n in names if "freq" in n.lower()), names[1])
    return np.asarray(data[tn], float), np.asarray(data[fn], float)


def settling_time(t, f, te, tend, band=0.01, dwell=0.10):
    mask = (t >= te) & (t < tend)
    tt, ff = t[mask], f[mask]
    if len(tt) < 2:
        return math.nan
    dt = float(np.median(np.diff(tt)))
    need = max(1, int(math.ceil(dwell/dt)))
    inside = np.abs(ff-F_NOM) <= band
    run = 0
    for i, ok in enumerate(inside):
        run = run+1 if ok else 0
        if run >= need:
            first = i-need+1
            return max(0.0, float(tt[first]-te))
    return math.nan


def metrics(t, f, start=REPORT_START_S):
    m = t >= start
    ff = f[m]
    # RoCoF from aligned 1-ms-ish exported samples.
    rr = np.gradient(f, t)
    return {
        "f_min_hz": float(np.min(ff)),
        "f_max_hz": float(np.max(ff)),
        "max_abs_df_hz": float(np.max(np.abs(ff-F_NOM))),
        "max_abs_rocof_hz_s": float(np.max(np.abs(rr[m]))),
    }


def event_metrics(t, f, controller):
    r = np.gradient(f, t)
    out = []
    for i,(te,label) in enumerate(EVENTS):
        tend = EVENTS[i+1][0] if i+1 < len(EVENTS) else float(t[-1])+1e-12
        m = (t >= te) & (t < tend)
        if not np.any(m):
            continue
        ff = f[m]
        out.append({
            "controller": controller,
            "event_index": i+1,
            "event_time_s": te,
            "event": label,
            "window_end_s": tend,
            "f_min_hz": float(np.min(ff)),
            "f_max_hz": float(np.max(ff)),
            "max_abs_df_hz": float(np.max(np.abs(ff-F_NOM))),
            "max_abs_rocof_hz_s": float(np.max(np.abs(r[m]))),
            "settling_time_s": settling_time(t,f,te,tend),
        })
    return out


def improvement(fixed, candidate):
    if not (np.isfinite(fixed) and np.isfinite(candidate)) or fixed == 0:
        return math.nan
    return 100.0*(fixed-candidate)/fixed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddpg", required=True, help="DDPG benchmark_frequency.csv")
    ap.add_argument("--td3", required=True, help="TD3 benchmark_frequency.csv")
    ap.add_argument("--sac", required=True, help="SAC benchmark_frequency.csv")
    ap.add_argument("--ppo", required=True, help="PPO benchmark_frequency.csv")
    ap.add_argument("--matlab_frequency_csv", default=None)
    ap.add_argument("--output_dir", default="all_rl_benchmark_comparison")
    ap.add_argument("--report_start", type=float, default=REPORT_START_S)
    args = ap.parse_args()

    report_start = float(args.report_start)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = {"DDPG":args.ddpg, "TD3":args.td3, "SAC":args.sac, "PPO":args.ppo}
    data = {}
    for algo,path in paths.items():
        data[algo] = load_frequency(path,algo)

    # Use DDPG fixed PI as master reference, then verify every export agrees.
    t0, fixed0, _ = data["DDPG"]
    for algo,(t,fixed,rl) in data.items():
        if len(t) != len(t0) or not np.allclose(t,t0,atol=1e-10,rtol=0):
            raise RuntimeError(f"{algo} time vector does not match DDPG")
        if not np.allclose(fixed,fixed0,atol=1e-8,rtol=0):
            d = float(np.max(np.abs(fixed-fixed0)))
            raise RuntimeError(f"{algo} Fixed PI differs from DDPG (max {d:g} Hz)")

    trajectories = {"Fixed PI": (t0,fixed0)}
    for algo,(t,_,rl) in data.items():
        trajectories[algo] = (t,rl)

    matlab = load_matlab(args.matlab_frequency_csv)

    # Plot 1: MATLAB + Fixed PI + all RL.
    fig,ax=plt.subplots(figsize=(13,6))
    if matlab is not None:
        mm=(matlab[0] >= report_start)
        ax.plot(matlab[0][mm],matlab[1][mm],label="MATLAB")
    for name,(t,f) in trajectories.items():
        m=t>=report_start
        ax.plot(t[m],f[m],label=name)
    for te,_ in EVENTS:
        ax.axvline(te,linestyle=":",linewidth=.8)
    ax.axhline(F_NOM,linestyle="--",linewidth=.8)
    ax.set_xlim(report_start,15.0)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Untouched MATLAB benchmark: Fixed PI vs adaptive RL controllers")
    ax.grid(True,alpha=.3); ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(out/"combined_benchmark_frequency.png",dpi=300)
    plt.close(fig)

    # Plot 2: controller comparison without MATLAB.
    fig,ax=plt.subplots(figsize=(13,6))
    for name,(t,f) in trajectories.items():
        m=t>=report_start
        ax.plot(t[m],f[m],label=name)
    for te,_ in EVENTS:
        ax.axvline(te,linestyle=":",linewidth=.8)
    ax.axhline(F_NOM,linestyle="--",linewidth=.8)
    ax.set_xlim(report_start,15.0)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Fixed PI vs DDPG vs TD3 vs SAC vs PPO")
    ax.grid(True,alpha=.3); ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(out/"combined_benchmark_frequency_rl_only.png",dpi=300)
    plt.close(fig)

    # Overall controlled-period summary.
    base = metrics(t0,fixed0,start=report_start)
    summary=[]
    for name,(t,f) in trajectories.items():
        m=metrics(t,f,start=report_start)
        summary.append({
            "controller":name,
            "report_start_s":report_start,
            **m,
            "df_improvement_pct_vs_fixed":0.0 if name=="Fixed PI" else improvement(base["max_abs_df_hz"],m["max_abs_df_hz"]),
            "rocof_improvement_pct_vs_fixed":0.0 if name=="Fixed PI" else improvement(base["max_abs_rocof_hz_s"],m["max_abs_rocof_hz_s"]),
        })
    write_csv(out/"all_algorithms_summary.csv",summary)

    # Event-level metrics and Fixed-PI-relative improvements.
    event_rows=[]
    by_controller={}
    for name,(t,f) in trajectories.items():
        rows=event_metrics(t,f,name)
        by_controller[name]=rows
        event_rows.extend(rows)
    write_csv(out/"all_algorithms_event_metrics.csv",event_rows)

    fixed_events={r["event_index"]:r for r in by_controller["Fixed PI"]}
    imp=[]
    for algo in ("DDPG","TD3","SAC","PPO"):
        for r in by_controller[algo]:
            b=fixed_events[r["event_index"]]
            imp.append({
                "controller":algo,
                "event_index":r["event_index"],
                "event_time_s":r["event_time_s"],
                "event":r["event"],
                "fixed_f_min_hz":b["f_min_hz"],
                "rl_f_min_hz":r["f_min_hz"],
                "fixed_f_max_hz":b["f_max_hz"],
                "rl_f_max_hz":r["f_max_hz"],
                "fixed_max_abs_df_hz":b["max_abs_df_hz"],
                "rl_max_abs_df_hz":r["max_abs_df_hz"],
                "df_improvement_pct":improvement(b["max_abs_df_hz"],r["max_abs_df_hz"]),
                "fixed_max_abs_rocof_hz_s":b["max_abs_rocof_hz_s"],
                "rl_max_abs_rocof_hz_s":r["max_abs_rocof_hz_s"],
                "rocof_improvement_pct":improvement(b["max_abs_rocof_hz_s"],r["max_abs_rocof_hz_s"]),
                "fixed_settling_time_s":b["settling_time_s"],
                "rl_settling_time_s":r["settling_time_s"],
                "settling_improvement_pct":improvement(b["settling_time_s"],r["settling_time_s"]),
            })
    write_csv(out/"event_improvement_table.csv",imp)

    print("="*110)
    print("ALL-RL BENCHMARK COMPARISON")
    print("="*110)
    print(f"report interval: {report_start:.3f}--15.000 s")
    for r in summary:
        print(f"{r['controller']:8s} fmin={r['f_min_hz']:.6f} fmax={r['f_max_hz']:.6f} "
              f"|df|max={r['max_abs_df_hz']:.6f} RoCoF={r['max_abs_rocof_hz_s']:.4f} "
              f"dfImp={r['df_improvement_pct_vs_fixed']:+.2f}% "
              f"RImp={r['rocof_improvement_pct_vs_fixed']:+.2f}%")
    print("-"*110)
    print("files:")
    for p in (
        "combined_benchmark_frequency.png",
        "combined_benchmark_frequency_rl_only.png",
        "all_algorithms_summary.csv",
        "all_algorithms_event_metrics.csv",
        "event_improvement_table.csv",
    ):
        print(" ",out/p)
    print("="*110)


if __name__=="__main__":
    main()
