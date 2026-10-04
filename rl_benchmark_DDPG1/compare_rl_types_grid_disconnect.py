"""11--12 s grid-disconnection comparison: DDPG vs TD3 vs SAC vs PPO only.

Uses benchmark_frequency.csv trajectories. MATLAB and Fixed PI are excluded.

Stable-state settling Tsss is the first 0.10-s window after 11 s whose linear
frequency slope has magnitude <= 0.02 Hz/s and whose detrended peak-to-peak
residual is <= 0.003 Hz. The search is restricted to 11 <= t < 12 s.

Reports Tsss, fss, |fss-60|, fmin/fmax, max |df| and max |RoCoF|.
"""

from __future__ import annotations
import argparse,csv,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

F0=60.0;T0=11.0;T1=12.0;DWELL=.10;SLOPE_MAX=.02;RESID_PP_MAX=.003
AGENTS={"DDPG":"ddpg_frequency_hz","TD3":"td3_frequency_hz",
        "SAC":"sac_frequency_hz","PPO":"ppo_frequency_hz"}
NAMES=tuple(AGENTS)


def read(path):
    with Path(path).open(newline="") as f:r=list(csv.DictReader(f))
    if not r:raise ValueError(f"Empty CSV: {path}")
    return r


def load(path,name):
    r=read(path);cols=list(r[0]);fc=AGENTS[name]
    if fc not in cols:
        # Allow generic DDPG-style sensitivity export naming if renamed externally.
        candidates=[c for c in cols if c.lower().endswith("frequency_hz") and "fixed" not in c.lower()]
        if len(candidates)!=1:raise KeyError(f"{fc} not found in {path}; available={cols}")
        fc=candidates[0]
    t=np.asarray([float(x["time_s"]) for x in r])
    f=np.asarray([float(x[fc]) for x in r])
    o=np.argsort(t);return t[o],f[o]


def stable(t,f):
    m=(t>=T0)&(t<T1);tt=t[m];ff=f[m]
    if len(tt)<3:return math.nan,math.nan,math.nan,math.nan
    dt=float(np.median(np.diff(tt)));need=max(3,int(math.ceil(DWELL/dt)))
    for i in range(len(tt)-need+1):
        tw=tt[i:i+need];fw=ff[i:i+need]
        slope,intercept=np.polyfit(tw,fw,1)
        residual=fw-(slope*tw+intercept)
        pp=float(np.max(residual)-np.min(residual))
        if abs(slope)<=SLOPE_MAX and pp<=RESID_PP_MAX:
            return float(tw[0]-T0),float(np.mean(fw)),float(slope),pp
    return math.nan,math.nan,math.nan,math.nan


def write(path,rows):
    with Path(path).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    ap=argparse.ArgumentParser()
    for n in NAMES:ap.add_argument(f"--{n.lower()}",required=True)
    ap.add_argument("--output_dir",default="rl_types_grid_disconnect")
    a=ap.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    data={n:load(getattr(a,n.lower()),n) for n in NAMES}

    rows=[]
    for n,(t,f) in data.items():
        m=(t>=T0)&(t<T1);tt=t[m];ff=f[m];rr=np.gradient(f,t)[m]
        ts,fss,slope,pp=stable(t,f)
        rows.append({
            "controller":n,"Tsss_s":ts,
            "status":"SETTLED" if np.isfinite(ts) else "NS",
            "fss_hz":fss,
            "steady_state_error_hz":abs(fss-F0) if np.isfinite(fss) else math.nan,
            "f_min_hz":float(np.min(ff)),"f_max_hz":float(np.max(ff)),
            "max_abs_df_hz":float(np.max(np.abs(ff-F0))),
            "max_abs_rocof_hz_s":float(np.max(np.abs(rr))),
            "stable_window_slope_hz_s":slope,
            "stable_window_detrended_pp_hz":pp,
        })
    write(out/"grid_disconnect_metrics.csv",rows)

    # Frequency response, 11--12 s only.
    fig,ax=plt.subplots(figsize=(11,5.5))
    for n,(t,f) in data.items():
        m=(t>=T0)&(t<T1);ax.plot(t[m],f[m],label=n)
    ax.axhline(F0,ls="--",lw=.8);ax.set_xlim(T0,T1)
    ax.set_xlabel("Time (s)");ax.set_ylabel("Frequency (Hz)")
    ax.set_title("RL algorithms: grid-disconnection frequency response (11-12 s)")
    ax.grid(True,alpha=.3);ax.legend(ncol=4);fig.tight_layout()
    fig.savefig(out/"grid_disconnect_frequency.png",dpi=300);plt.close(fig)

    # Dedicated Tsss figure.
    vals=np.asarray([r["Tsss_s"] for r in rows],float)
    fig,ax=plt.subplots(figsize=(8.5,5.5));x=np.arange(len(NAMES));good=np.isfinite(vals)
    ax.bar(x[good],vals[good])
    top=max(vals[good])*1.10 if np.any(good) else 1.0
    for xx in x[~good]:ax.text(xx,top*.92,"NS",ha="center",va="bottom")
    ax.set_xticks(x);ax.set_xticklabels(NAMES);ax.set_ylabel("Steady-state settling time Tsss (s)")
    ax.set_title("RL algorithms: grid-disconnection settling to stable frequency")
    ax.grid(True,axis="y",alpha=.3);fig.tight_layout()
    fig.savefig(out/"grid_disconnect_Tsss.png",dpi=300);plt.close(fig)

    # fss and error figure.
    fss=np.asarray([r["fss_hz"] for r in rows],float)
    fig,ax=plt.subplots(figsize=(8.5,5.5));good=np.isfinite(fss)
    ax.bar(x[good],fss[good]);ax.axhline(F0,ls="--",lw=.8)
    ax.set_xticks(x);ax.set_xticklabels(NAMES);ax.set_ylabel("Stable frequency fss (Hz)")
    ax.set_title("RL algorithms: grid-disconnection stable frequency")
    ax.grid(True,axis="y",alpha=.3);fig.tight_layout()
    fig.savefig(out/"grid_disconnect_fss.png",dpi=300);plt.close(fig)

    print("="*105);print("RL-TYPE GRID-DISCONNECTION COMPARISON (11-12 s)")
    print("="*105)
    print(f"{'Agent':7s} {'Tsss(s)':>10s} {'fss(Hz)':>10s} {'|fss-60|':>11s} "
          f"{'fmin':>10s} {'fmax':>10s} {'max|df|':>10s} {'max|R|':>10s}")
    for r in rows:
        def fm(v,w=10,p=5):return f"{v:{w}.{p}f}" if np.isfinite(v) else f"{'NS':>{w}s}"
        print(f"{r['controller']:7s} {fm(r['Tsss_s'])} {fm(r['fss_hz'])} "
              f"{fm(r['steady_state_error_hz'],11)} {fm(r['f_min_hz'])} {fm(r['f_max_hz'])} "
              f"{fm(r['max_abs_df_hz'])} {fm(r['max_abs_rocof_hz_s'])}")
    print("="*105)


if __name__=="__main__":main()
