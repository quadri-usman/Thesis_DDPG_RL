"""Wide-range Stage-1 DDPG final evaluation (12-state actor).

Modes
-----
1. split:
   Fixed PI vs frozen DDPG on randomized VALIDATION or withheld TEST loads.

2. benchmark:
   Fixed PI vs frozen DDPG on the untouched corrected 15-s MATLAB benchmark:
       50 kW
       52.5 kW @ 2.0 s
       55 kW   @ 3.5 s
       60 kW   @ 5.0 s
       70 kW   @ 6.5 s
       50 kW   @ 9.0 s
       10 kW + grid open @ 11.0 s
       50 kW + grid reclosed @ 12.0 s

For both modes the script reports episode-level AND event-by-event:
    max |frequency deviation|
    max |RoCoF|
    settling time
    percentage improvement of DDPG over Fixed PI

It also saves the four DDPG gain trajectories Kpv/Kiv/Kpi/Kii.

Optional:
    --matlab_frequency_csv Matlab_frequency_scenario.csv
adds the actual MATLAB frequency trace to the benchmark frequency plot and
reports Fixed-PI/Python and DDPG errors relative to MATLAB.

The policy is frozen: exploration OFF, learning OFF.

This version is for checkpoints trained with:
    ddpg_agent_wide.py
    controller_rl_gains_wide.py
    gfm_env_random_wide.py

The benchmark observation includes the four normalized log-gain states, so the
actor input is exactly 12-dimensional during both split and MATLAB-benchmark
evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ddpg_agent_wide import DDPGAgent, DDPGConfig
from controller_rl_gains_wide import (
    PIGains,
    RLGainAdapter,
    RLGainAdaptiveGFMController,
)
from gfm_env_random_wide import RandomizedGFMGainTuningEnv
from training_scenarios_split import ScenarioSplit
from benchmark_scenarios_new import MatlabScenarioBenchmark


F_NOM = 60.0
RL_DT = 1.0e-3
SETTLING_BAND_HZ = 0.01
SETTLING_DWELL_S = 0.10
WARMUP_S = 1.5

BENCHMARK_EVENTS = [
    (2.0,  "50->52.5 kW"),
    (3.5,  "52.5->55 kW"),
    (5.0,  "55->60 kW"),
    (6.5,  "60->70 kW"),
    (9.0,  "70->50 kW"),
    (11.0, "50->10 kW + grid disconnect"),
    (12.0, "10->50 kW + grid reconnect"),
]


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def improvement_pct(fixed, ddpg):
    if not (np.isfinite(fixed) and np.isfinite(ddpg)) or fixed == 0:
        return math.nan
    return 100.0 * (fixed - ddpg) / fixed


def settling_time(t, f, event_time, end_time):
    """First entry into +/- band that remains there for dwell time."""
    mask = (t >= event_time) & (t < end_time)
    tt = t[mask]
    ff = f[mask]

    if len(tt) < 2:
        return math.nan

    inside = np.abs(ff - F_NOM) <= SETTLING_BAND_HZ
    dt = float(np.median(np.diff(tt)))
    needed = max(1, int(math.ceil(SETTLING_DWELL_S / dt)))

    if len(inside) < needed:
        return math.nan

    # Efficient consecutive-True search.
    run = 0
    for i, ok in enumerate(inside):
        run = run + 1 if ok else 0
        if run >= needed:
            first = i - needed + 1
            return max(0.0, float(tt[first] - event_time))

    return math.nan


def event_metrics(record, events, episode_end):
    rows = []
    t = record["time"]
    f = record["frequency"]
    r = record["rocof"]

    for i, (te, label) in enumerate(events):
        tend = events[i + 1][0] if i + 1 < len(events) else episode_end
        mask = (t >= te) & (t < tend)

        if not np.any(mask):
            continue

        rows.append({
            "event_index": i + 1,
            "event_time_s": float(te),
            "event": label,
            "window_end_s": float(tend),
            "max_abs_df_hz": float(np.max(np.abs(f[mask] - F_NOM))),
            "max_abs_rocof_hz_s": float(np.max(np.abs(r[mask]))),
            "settling_time_s": settling_time(t, f, te, tend),
        })

    return rows


def compare_event_rows(fixed_rows, ddpg_rows, context):
    out = []
    for b, d in zip(fixed_rows, ddpg_rows):
        if b["event"] != d["event"]:
            raise RuntimeError("event mismatch between Fixed PI and DDPG")

        out.append({
            "context": context,
            "event_index": b["event_index"],
            "event_time_s": b["event_time_s"],
            "event": b["event"],
            "window_end_s": b["window_end_s"],

            "fixed_max_abs_df_hz": b["max_abs_df_hz"],
            "ddpg_max_abs_df_hz": d["max_abs_df_hz"],
            "df_improvement_pct": improvement_pct(
                b["max_abs_df_hz"], d["max_abs_df_hz"]
            ),

            "fixed_max_abs_rocof_hz_s": b["max_abs_rocof_hz_s"],
            "ddpg_max_abs_rocof_hz_s": d["max_abs_rocof_hz_s"],
            "rocof_improvement_pct": improvement_pct(
                b["max_abs_rocof_hz_s"], d["max_abs_rocof_hz_s"]
            ),

            "fixed_settling_time_s": b["settling_time_s"],
            "ddpg_settling_time_s": d["settling_time_s"],
            "settling_improvement_pct": improvement_pct(
                b["settling_time_s"], d["settling_time_s"]
            ),
        })
    return out


def episode_metrics(record):
    df = record["frequency"] - F_NOM
    return {
        "max_abs_df_hz": float(np.max(np.abs(df))),
        "rms_df_hz": float(np.sqrt(np.mean(df**2))),
        "max_abs_rocof_hz_s": float(np.max(np.abs(record["rocof"]))),
    }


# ---------------------------------------------------------------------------
# Randomized validation / withheld-test split evaluation
# ---------------------------------------------------------------------------

def run_split_episode(agent, seed, split, adaptive):
    env = RandomizedGFMGainTuningEnv(split=ScenarioSplit(split))
    state, info = env.reset(seed=int(seed))

    rec = {k: [] for k in (
        "time", "frequency", "rocof",
        "Kpv", "Kiv", "Kpi", "Kii", "load",
    )}

    terminated = truncated = False

    while not (terminated or truncated):
        if adaptive and info["rl_active"]:
            action = agent.select_action(state, explore=False)
        else:
            action = np.zeros(4, dtype=np.float32)

        state, reward, terminated, truncated, info = env.step(action)

        # Record full trajectory; RL authority remains disabled during warm-up.
        rec["time"].append(info["time"])
        rec["frequency"].append(info["frequency"])
        rec["rocof"].append(info["rocof"] if info["rl_active"] else 0.0)
        rec["Kpv"].append(info["Kpv"])
        rec["Kiv"].append(info["Kiv"])
        rec["Kpi"].append(info["Kpi"])
        rec["Kii"].append(info["Kii"])
        rec["load"].append(info["nominal_load_power"])

    rec = {k: np.asarray(v, dtype=float) for k, v in rec.items()}

    events = []
    previous = 50_000.0
    for e in env.scenario.disturbance_events:
        current = float(e.load_power_w)
        events.append(
            (float(e.time), f"{previous/1000:g}->{current/1000:g} kW")
        )
        previous = current

    return rec, events, env.scenario


# ---------------------------------------------------------------------------
# Untouched 15-s MATLAB benchmark
# ---------------------------------------------------------------------------

def _sample_to_obs(sample, Evd, Eid, previous_f, elapsed, adapter):
    """Build the exact 12-state observation used by the wide-range policy."""
    o = sample.controller
    rocof = (o.frequency - previous_f) / elapsed

    obs = np.array([
        (o.frequency - F_NOM) / 0.5,
        rocof / 15.0,
        o.vd_error / 380.0,
        Evd / 100.0,
        o.vq_error / 380.0,
        o.id_error / 200.0,
        Eid / 100.0,
        o.iq_error / 200.0,
        *adapter.normalized_gains(),
    ], dtype=np.float32)

    return np.clip(obs, -10.0, 10.0), rocof


def run_benchmark_episode(agent, adaptive):
    """Run untouched corrected benchmark; only controller gains may adapt."""
    sim = MatlabScenarioBenchmark()

    adapter = RLGainAdapter(initial_gains=PIGains())
    if adaptive:
        sim.controller = RLGainAdaptiveGFMController(
            params=sim.params,
            gains=adapter.gains,
        )

    base_dt = float(sim.params.dt)
    base_per_rl = int(round(RL_DT / base_dt))

    Evd = 0.0
    Eid = 0.0
    rl_active = False
    previous_rl_frequency = None

    rec = {k: [] for k in (
        "time", "frequency", "rocof",
        "Kpv", "Kiv", "Kpi", "Kii",
        "load", "grid_connected",
    )}

    # Execute at the validated 10-us base rate, but record/control every 1 ms.
    total_steps = int(round(15.0 / base_dt))

    last_sample = None

    for k in range(total_steps):
        t0 = float(sim.state.time)

        # At the first base step of each RL interval, update adaptive gains.
        at_rl_boundary = (k % base_per_rl == 0)

        if at_rl_boundary and t0 >= WARMUP_S - 0.5 * base_dt:
            if not rl_active:
                rl_active = True
                Evd = 0.0
                Eid = 0.0
                if last_sample is not None:
                    previous_rl_frequency = last_sample.controller.frequency

            if adaptive and last_sample is not None:
                # Observation uses the response at the previous RL boundary.
                prev_f = (
                    previous_rl_frequency
                    if previous_rl_frequency is not None
                    else last_sample.controller.frequency
                )
                obs, _ = _sample_to_obs(
                    last_sample, Evd, Eid, prev_f, RL_DT, adapter
                )
                action = agent.select_action(obs, explore=False)
                gains = adapter.apply_action(action)
                sim.controller.set_gains(gains)

        sample = sim.step()
        last_sample = sample

        if rl_active:
            Evd += sample.controller.vd_error * base_dt
            Eid += sample.controller.id_error * base_dt

        # Record complete benchmark from the first 1-ms sample.
        if (k + 1) % base_per_rl == 0:
            f = sample.controller.frequency
            rocof_record = 0.0 if not rec["frequency"] else (f - rec["frequency"][-1]) / RL_DT
            if rl_active:
                previous_rl_frequency = f

            if adaptive:
                g = adapter.gains
                gains = (g.voltage_kp, g.voltage_ki, g.current_kp, g.current_ki)
            else:
                gains = (2.0, 14.0, 2.0, 20.0)

            rec["time"].append(sample.time)
            rec["frequency"].append(f)
            rec["rocof"].append(rocof_record)
            rec["Kpv"].append(gains[0])
            rec["Kiv"].append(gains[1])
            rec["Kpi"].append(gains[2])
            rec["Kii"].append(gains[3])
            rec["load"].append(sample.nominal_load_power)
            rec["grid_connected"].append(1.0 if sample.nominal_grid_connected else 0.0)

    return {k: np.asarray(v, dtype=float) for k, v in rec.items()}


# ---------------------------------------------------------------------------
# Plotting / MATLAB CSV
# ---------------------------------------------------------------------------

def load_matlab_frequency(path):
    if path is None:
        return None

    data = np.genfromtxt(path, delimiter=",", names=True)
    names = list(data.dtype.names or [])

    if len(names) < 2:
        raise ValueError("MATLAB CSV must contain time and frequency columns")

    time_name = next(
        (n for n in names if "time" in n.lower()),
        names[0],
    )
    freq_name = next(
        (n for n in names if "freq" in n.lower()),
        names[1],
    )

    return (
        np.asarray(data[time_name], dtype=float),
        np.asarray(data[freq_name], dtype=float),
    )


def matlab_error_metrics(record, matlab):
    if matlab is None:
        return None

    tm, fm = matlab
    mask = (
        np.isfinite(tm) & np.isfinite(fm)
        & (tm >= record["time"][0])
        & (tm <= record["time"][-1])
    )
    tm = tm[mask]
    fm = fm[mask]

    if len(tm) == 0:
        return None

    pred = np.interp(tm, record["time"], record["frequency"])
    e = pred - fm

    return {
        "rmse_hz": float(np.sqrt(np.mean(e**2))),
        "mae_hz": float(np.mean(np.abs(e))),
        "max_abs_error_hz": float(np.max(np.abs(e))),
        "mean_error_hz": float(np.mean(e)),
    }


def plot_frequency(path, fixed, ddpg, events, title, matlab=None):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    if matlab is not None:
        ax.plot(matlab[0], matlab[1], label="MATLAB")
    ax.plot(fixed["time"], fixed["frequency"], label="Fixed PI")
    ax.plot(ddpg["time"], ddpg["frequency"], label="DDPG adaptive PI")
    for te, _ in events:
        ax.axvline(te, linestyle=":", linewidth=0.8)
    ax.axhline(F_NOM, linestyle="--", linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=250)
    plt.close(fig)


def plot_rocof(path, fixed, ddpg, events, title):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(fixed["time"], fixed["rocof"], label="Fixed PI")
    ax.plot(ddpg["time"], ddpg["rocof"], label="DDPG adaptive PI")
    for te, _ in events:
        ax.axvline(te, linestyle=":", linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RoCoF (Hz/s)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=250)
    plt.close(fig)


def plot_gains(path, ddpg, events, title):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for key in ("Kpv", "Kiv", "Kpi", "Kii"):
        ax.plot(ddpg["time"], ddpg[key], label=key)
    for te, _ in events:
        ax.axvline(te, linestyle=":", linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("PI gain")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=250)
    plt.close(fig)


def evaluate_pair(fixed, ddpg, events, episode_end, context):
    fm = episode_metrics(fixed)
    dm = episode_metrics(ddpg)

    episode_row = {
        "context": context,
        "fixed_max_abs_df_hz": fm["max_abs_df_hz"],
        "ddpg_max_abs_df_hz": dm["max_abs_df_hz"],
        "df_improvement_pct": improvement_pct(
            fm["max_abs_df_hz"], dm["max_abs_df_hz"]
        ),
        "fixed_rms_df_hz": fm["rms_df_hz"],
        "ddpg_rms_df_hz": dm["rms_df_hz"],
        "rms_df_improvement_pct": improvement_pct(
            fm["rms_df_hz"], dm["rms_df_hz"]
        ),
        "fixed_max_abs_rocof_hz_s": fm["max_abs_rocof_hz_s"],
        "ddpg_max_abs_rocof_hz_s": dm["max_abs_rocof_hz_s"],
        "rocof_improvement_pct": improvement_pct(
            fm["max_abs_rocof_hz_s"], dm["max_abs_rocof_hz_s"]
        ),
    }

    fe = event_metrics(fixed, events, episode_end)
    de = event_metrics(ddpg, events, episode_end)
    ce = compare_event_rows(fe, de, context)

    return episode_row, ce


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument(
        "--mode",
        choices=["split", "benchmark", "all"],
        default="all",
    )
    ap.add_argument(
        "--split",
        choices=["validation", "test"],
        default="test",
    )
    ap.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[2001, 2002, 2003, 2004, 2005],
    )
    ap.add_argument("--output_dir", default="ddpg_final_evaluation")
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--matlab_frequency_csv",
        default=None,
        help="Optional actual MATLAB frequency CSV for benchmark overlay.",
    )
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = DDPGConfig()
    if cfg.state_dim != 12 or cfg.action_dim != 4:
        raise RuntimeError(
            f"Wide evaluator requires state_dim=12, action_dim=4; "
            f"got {cfg.state_dim}, {cfg.action_dim}"
        )
    agent = DDPGAgent(cfg, device=args.device)
    agent.load(args.checkpoint, load_optimizers=False)
    agent.actor.eval()
    agent.critic.eval()

    print("=" * 100)
    print("FROZEN WIDE-RANGE STAGE-1 DDPG FINAL EVALUATION")
    print("=" * 100)
    print("checkpoint :", args.checkpoint)
    print("mode       :", args.mode)
    print("exploration: OFF")
    print("learning   : OFF")
    print("=" * 100)

    episode_rows = []
    event_rows = []

    if args.mode in ("split", "all"):
        split_dir = out / f"{args.split}_split"
        split_dir.mkdir(exist_ok=True)

        print(f"\n{args.split.upper()} SPLIT")
        if args.split == "test":
            print("Withheld disturbance magnitudes: 52.5 and 60 kW")

        for seed in args.seeds:
            fixed, events, sc1 = run_split_episode(
                agent, seed, args.split, adaptive=False
            )
            ddpg, events2, sc2 = run_split_episode(
                agent, seed, args.split, adaptive=True
            )

            s1 = [(e.time, e.load_power_w) for e in sc1.events]
            s2 = [(e.time, e.load_power_w) for e in sc2.events]
            if s1 != s2 or events != events2:
                raise RuntimeError("Fixed PI/DDPG scenario mismatch")

            context = f"{args.split}_seed_{seed}"
            ep, ev = evaluate_pair(
                fixed, ddpg, events, sc1.episode_duration, context
            )
            ep["seed"] = seed
            ep["split"] = args.split
            episode_rows.append(ep)
            event_rows.extend(ev)

            plot_frequency(
                split_dir / f"seed_{seed}_frequency.png",
                fixed, ddpg, events,
                f"{args.split.title()} seed {seed}: Fixed PI vs DDPG",
            )
            plot_rocof(
                split_dir / f"seed_{seed}_rocof.png",
                fixed, ddpg, events,
                f"{args.split.title()} seed {seed}: RoCoF",
            )
            plot_gains(
                split_dir / f"seed_{seed}_gains.png",
                ddpg, events,
                f"{args.split.title()} seed {seed}: adaptive gains",
            )

            print(
                f"seed {seed}: "
                f"|df|max {ep['fixed_max_abs_df_hz']:.5f} -> "
                f"{ep['ddpg_max_abs_df_hz']:.5f} Hz "
                f"({ep['df_improvement_pct']:+.2f}%), "
                f"RoCoF {ep['fixed_max_abs_rocof_hz_s']:.3f} -> "
                f"{ep['ddpg_max_abs_rocof_hz_s']:.3f} Hz/s "
                f"({ep['rocof_improvement_pct']:+.2f}%)"
            )

    if args.mode in ("benchmark", "all"):
        bench_dir = out / "matlab_benchmark"
        bench_dir.mkdir(exist_ok=True)

        print("\nUNTOUCHED 15-s MATLAB BENCHMARK")
        fixed = run_benchmark_episode(agent, adaptive=False)
        ddpg = run_benchmark_episode(agent, adaptive=True)

        ep, ev = evaluate_pair(
            fixed, ddpg, BENCHMARK_EVENTS, 15.0, "matlab_benchmark"
        )
        ep["seed"] = ""
        ep["split"] = "untouched_matlab_benchmark"
        episode_rows.append(ep)
        event_rows.extend(ev)

        matlab = load_matlab_frequency(args.matlab_frequency_csv)

        plot_frequency(
            bench_dir / "benchmark_frequency.png",
            fixed, ddpg, BENCHMARK_EVENTS,
            "Untouched MATLAB benchmark: Fixed PI vs DDPG",
            matlab=matlab,
        )
        plot_rocof(
            bench_dir / "benchmark_rocof.png",
            fixed, ddpg, BENCHMARK_EVENTS,
            "Untouched MATLAB benchmark: RoCoF",
        )
        plot_gains(
            bench_dir / "benchmark_gains.png",
            ddpg, BENCHMARK_EVENTS,
            "Untouched MATLAB benchmark: DDPG gain trajectories",
        )

        np.savez_compressed(
            bench_dir / "benchmark_trajectories.npz",
            fixed_time=fixed["time"],
            fixed_frequency=fixed["frequency"],
            fixed_rocof=fixed["rocof"],
            ddpg_time=ddpg["time"],
            ddpg_frequency=ddpg["frequency"],
            ddpg_rocof=ddpg["rocof"],
            Kpv=ddpg["Kpv"],
            Kiv=ddpg["Kiv"],
            Kpi=ddpg["Kpi"],
            Kii=ddpg["Kii"],
        )

        print(
            f"benchmark: |df|max "
            f"{ep['fixed_max_abs_df_hz']:.5f} -> "
            f"{ep['ddpg_max_abs_df_hz']:.5f} Hz "
            f"({ep['df_improvement_pct']:+.2f}%)"
        )
        print(
            f"benchmark: RoCoF "
            f"{ep['fixed_max_abs_rocof_hz_s']:.3f} -> "
            f"{ep['ddpg_max_abs_rocof_hz_s']:.3f} Hz/s "
            f"({ep['rocof_improvement_pct']:+.2f}%)"
        )

        if matlab is not None:
            fixed_mat = matlab_error_metrics(fixed, matlab)
            ddpg_mat = matlab_error_metrics(ddpg, matlab)

            matlab_rows = [
                {"controller": "Fixed PI", **fixed_mat},
                {"controller": "DDPG", **ddpg_mat},
            ]
            write_csv(
                bench_dir / "matlab_frequency_error.csv",
                matlab_rows,
            )
            print(
                "MATLAB frequency RMSE: "
                f"Fixed PI={fixed_mat['rmse_hz']:.6f} Hz, "
                f"DDPG={ddpg_mat['rmse_hz']:.6f} Hz"
            )

    write_csv(out / "episode_comparison.csv", episode_rows)
    write_csv(out / "event_by_event_comparison.csv", event_rows)

    # Save concise summary.
    summary = {
        "checkpoint": args.checkpoint,
        "mode": args.mode,
        "settling_band_hz": SETTLING_BAND_HZ,
        "settling_dwell_s": SETTLING_DWELL_S,
        "rl_dt_s": RL_DT,
        "episode_rows": episode_rows,
    }
    with (out / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 100)
    print("FILES")
    print("=" * 100)
    print("episode metrics :", out / "episode_comparison.csv")
    print("event metrics   :", out / "event_by_event_comparison.csv")
    print("summary         :", out / "summary.json")
    print("=" * 100)


if __name__ == "__main__":
    main()
