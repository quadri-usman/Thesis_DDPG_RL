"""Joined quantitative benchmark comparison: Fixed PI vs DDPG/TD3/SAC/PPO."""

from __future__ import annotations
import argparse
import csv
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AGENTS = {"DDPG": "ddpg", "TD3": "td3", "SAC": "sac", "PPO": "ppo"}


def read_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def number(row, key):
    value = row.get(key, "")
    if str(value).strip().lower() in ("", "nan", "none"):
        return math.nan
    return float(value)


def finite_mean(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.mean(a)) if len(a) else math.nan


def improvement(base, candidate):
    if not (np.isfinite(base) and np.isfinite(candidate)) or base == 0:
        return math.nan
    return 100.0 * (base - candidate) / base


def parse_file(path, agent):
    rows = read_csv(path)
    prefix = AGENTS[agent]
    parsed = []
    for i, r in enumerate(rows):
        parsed.append({
            "event_index": int(float(r.get("event_index", i + 1))),
            "event_time_s": number(r, "event_time_s"),
            "event": r.get("event", f"Event {i+1}"),
            "fixed_df": number(r, "fixed_max_abs_df_hz"),
            "agent_df": number(r, f"{prefix}_max_abs_df_hz"),
            "fixed_rocof": number(r, "fixed_max_abs_rocof_hz_s"),
            "agent_rocof": number(r, f"{prefix}_max_abs_rocof_hz_s"),
            "fixed_ts": number(r, "fixed_settling_time_s"),
            "agent_ts": number(r, f"{prefix}_settling_time_s"),
        })
    return parsed


def grouped_plot(path, labels, series, ylabel, title):
    names = ["Fixed PI", "DDPG", "TD3", "SAC", "PPO"]
    x = np.arange(len(labels))
    width = 0.16

    fig, ax = plt.subplots(figsize=(14, 6))
    for j, name in enumerate(names):
        ax.bar(x + (j - 2) * width, series[name], width, label=name)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(ncol=5)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def mean_plot(path, summary, key, ylabel, title):
    names = [r["controller"] for r in summary]
    values = [r[key] for r in summary]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(names, values)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddpg", required=True)
    ap.add_argument("--td3", required=True)
    ap.add_argument("--sac", required=True)
    ap.add_argument("--ppo", required=True)
    ap.add_argument("--output_dir", default="quantitative_rl_comparison")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = {"DDPG": args.ddpg, "TD3": args.td3, "SAC": args.sac, "PPO": args.ppo}
    data = {name: parse_file(path, name) for name, path in paths.items()}

    ref = data["DDPG"]
    for name in ("TD3", "SAC", "PPO"):
        if len(data[name]) != len(ref):
            raise RuntimeError(f"{name} event count differs from DDPG")
        for a, b in zip(ref, data[name]):
            if a["event_index"] != b["event_index"]:
                raise RuntimeError(f"{name} event indexing differs from DDPG")

    labels = [r["event"] for r in ref]

    df_series = {"Fixed PI": [r["fixed_df"] for r in ref]}
    rocof_series = {"Fixed PI": [r["fixed_rocof"] for r in ref]}
    ts_series = {"Fixed PI": [r["fixed_ts"] for r in ref]}

    for name in AGENTS:
        df_series[name] = [r["agent_df"] for r in data[name]]
        rocof_series[name] = [r["agent_rocof"] for r in data[name]]
        ts_series[name] = [r["agent_ts"] for r in data[name]]

    grouped_plot(
        out / "event_frequency_deviation.png", labels, df_series,
        "Maximum |frequency deviation| (Hz)",
        "Event-by-event frequency-deviation comparison",
    )
    grouped_plot(
        out / "event_rocof.png", labels, rocof_series,
        "Maximum |RoCoF| (Hz/s)",
        "Event-by-event RoCoF comparison",
    )
    grouped_plot(
        out / "event_settling_time.png", labels, ts_series,
        "Settling time (s)",
        "Event-by-event settling-time comparison",
    )

    # Long-form event table.
    event_rows = []
    for i, base in enumerate(ref):
        event_rows.append({
            "controller": "Fixed PI",
            "event_index": base["event_index"],
            "event_time_s": base["event_time_s"],
            "event": base["event"],
            "max_abs_df_hz": base["fixed_df"],
            "max_abs_rocof_hz_s": base["fixed_rocof"],
            "settling_time_s": base["fixed_ts"],
            "df_improvement_pct_vs_fixed": 0.0,
            "rocof_improvement_pct_vs_fixed": 0.0,
            "settling_improvement_pct_vs_fixed": 0.0,
        })
        for name in AGENTS:
            r = data[name][i]
            event_rows.append({
                "controller": name,
                "event_index": base["event_index"],
                "event_time_s": base["event_time_s"],
                "event": base["event"],
                "max_abs_df_hz": r["agent_df"],
                "max_abs_rocof_hz_s": r["agent_rocof"],
                "settling_time_s": r["agent_ts"],
                "df_improvement_pct_vs_fixed": improvement(base["fixed_df"], r["agent_df"]),
                "rocof_improvement_pct_vs_fixed": improvement(base["fixed_rocof"], r["agent_rocof"]),
                "settling_improvement_pct_vs_fixed": improvement(base["fixed_ts"], r["agent_ts"]),
            })

    write_csv(out / "quantitative_event_comparison.csv", event_rows)

    base_df = finite_mean(df_series["Fixed PI"])
    base_rocof = finite_mean(rocof_series["Fixed PI"])
    base_ts = finite_mean(ts_series["Fixed PI"])

    summary = []
    for name in ["Fixed PI", "DDPG", "TD3", "SAC", "PPO"]:
        md = finite_mean(df_series[name])
        mr = finite_mean(rocof_series[name])
        mt = finite_mean(ts_series[name])
        summary.append({
            "controller": name,
            "mean_max_abs_df_hz": md,
            "mean_max_abs_rocof_hz_s": mr,
            "mean_settling_time_s_finite_events": mt,
            "finite_settling_events": int(np.sum(np.isfinite(np.asarray(ts_series[name], float)))),
            "df_improvement_pct_vs_fixed": 0.0 if name == "Fixed PI" else improvement(base_df, md),
            "rocof_improvement_pct_vs_fixed": 0.0 if name == "Fixed PI" else improvement(base_rocof, mr),
            "settling_improvement_pct_vs_fixed": 0.0 if name == "Fixed PI" else improvement(base_ts, mt),
        })

    write_csv(out / "quantitative_summary.csv", summary)

    mean_plot(
        out / "mean_frequency_deviation.png", summary, "mean_max_abs_df_hz",
        "Mean event maximum |frequency deviation| (Hz)",
        "Overall frequency-deviation comparison",
    )
    mean_plot(
        out / "mean_rocof.png", summary, "mean_max_abs_rocof_hz_s",
        "Mean event maximum |RoCoF| (Hz/s)",
        "Overall RoCoF comparison",
    )
    mean_plot(
        out / "mean_settling_time.png", summary, "mean_settling_time_s_finite_events",
        "Mean settling time (s)",
        "Overall settling-time comparison",
    )

    print("=" * 112)
    print("QUANTITATIVE FIXED-PI / RL BENCHMARK COMPARISON")
    print("=" * 112)
    print(
        f"{'Controller':10s} {'mean|df|':>12s} {'mean|RoCoF|':>14s} "
        f"{'mean Ts':>12s} {'df imp':>10s} {'R imp':>10s} {'Ts imp':>10s}"
    )
    for r in summary:
        print(
            f"{r['controller']:10s} "
            f"{r['mean_max_abs_df_hz']:12.6f} "
            f"{r['mean_max_abs_rocof_hz_s']:14.5f} "
            f"{r['mean_settling_time_s_finite_events']:12.5f} "
            f"{r['df_improvement_pct_vs_fixed']:+9.2f}% "
            f"{r['rocof_improvement_pct_vs_fixed']:+9.2f}% "
            f"{r['settling_improvement_pct_vs_fixed']:+9.2f}%"
        )

    print("\nOutputs:")
    for name in (
        "event_frequency_deviation.png",
        "event_rocof.png",
        "event_settling_time.png",
        "mean_frequency_deviation.png",
        "mean_rocof.png",
        "mean_settling_time.png",
        "quantitative_event_comparison.csv",
        "quantitative_summary.csv",
    ):
        print(" ", out / name)
    print("=" * 112)


if __name__ == "__main__":
    main()
