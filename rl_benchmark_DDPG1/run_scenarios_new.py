"""Command-line runner for the final fixed-topology MATLAB scenario benchmark."""

from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmark_scenarios_new import MatlabScenarioBenchmark
from parameters import PARAMS

EVENT_TIMES = (2.0, 3.5, 5.0, 6.5, 9.0, 11.0, 12.0)


def save_plot(path, t, series, ylabel, title):
    plt.figure(figsize=(11, 4.8))
    for y, label in series:
        plt.plot(t, y, label=label)
    for te in EVENT_TIMES:
        plt.axvline(te, linestyle=":", linewidth=0.9)
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    if any(label for _, label in series):
        plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def finite_report(a):
    names = [
        "frequency", "P", "Q", "P_filtered", "Q_filtered",
        "vod", "voq", "iod", "ioq", "ifd", "ifq",
        "ifd_ref", "ifq_ref", "vd_command", "vq_command",
        "md", "mq", "battery_current", "aux_d", "aux_q",
    ]
    print("\n" + "="*82)
    print("NaN / INF AND RANGE CHECK")
    print("="*82)
    ok = True
    for name in names:
        x = a[name]
        finite = np.isfinite(x).all()
        ok &= finite
        print(
            f"{name:<18} finite={str(finite):<5} "
            f"min={np.nanmin(x):>14.6f} "
            f"max={np.nanmax(x):>14.6f} "
            f"final={x[-1]:>14.6f}"
        )
    print("="*82)
    print("PASS: all selected signals are finite." if ok else "FAIL: NaN/Inf detected.")
    return ok


def event_report(a):
    t = a["time"]
    print("\n" + "="*82)
    print("SCENARIO EVENT CHECK")
    print("="*82)
    for te in (0.0,) + EVENT_TIMES:
        idx = int(np.argmin(np.abs(t-te)))
        groups = [int(a[f"breaker_g{i}"][idx]) for i in range(7)]
        print(
            f"t={t[idx]:7.3f} s | "
            f"load={a['nominal_load_power'][idx]/1000:5.1f} kW | "
            f"grid={'ON ' if a['nominal_grid_connected'][idx] > 0.5 else 'OFF'} | "
            f"groups={groups} | f={a['frequency'][idx]:.6f} Hz"
        )
    print("="*82)


def main():
    parser = argparse.ArgumentParser(
        description="Run the final fixed-topology MATLAB GFM disturbance benchmark."
    )
    parser.add_argument("--time", type=float, default=15.0)
    parser.add_argument("--output_dir", type=str, default="scenario_results_new")
    parser.add_argument("--progress", type=float, default=0.5)
    args = parser.parse_args()

    if args.time <= 0:
        raise ValueError("--time must be positive")

    os.makedirs(args.output_dir, exist_ok=True)

    sim = MatlabScenarioBenchmark()

    progress_every = None
    if args.progress > 0:
        progress_every = max(1, int(round(args.progress / PARAMS.dt)))

    n_steps = int(round(args.time / PARAMS.dt))

    print("="*82)
    print("CORRECTED MATLAB-GENERATED GFM SCENARIO BENCHMARK")
    print("="*82)
    print(f"dt              : {PARAMS.dt:.9g} s")
    print(f"duration        : {args.time:.6f} s")
    print(f"steps           : {n_steps}")
    print("events:")
    print("  0.0-2.0 s   : 50.0 kW, grid connected")
    print("  2.0-3.5 s   : 52.5 kW, grid connected")
    print("  3.5-5.0 s   : 55.0 kW, grid connected")
    print("  5.0-6.5 s   : 60.0 kW, grid connected")
    print("  6.5-9.0 s   : 70.0 kW, grid connected")
    print("  9.0-11.0 s  : 50.0 kW, grid connected")
    print(" 11.0-12.0 s  : 10.0 kW, grid disconnected")
    print(" >=12.0 s     : 50.0 kW, grid connected")
    print("="*82)

    history = sim.run(args.time, progress_every=progress_every)
    a = history.arrays()

    finite_report(a)
    event_report(a)

    print("\n" + "="*82)
    print("FINAL VALUES")
    print("="*82)
    for name, unit in [
        ("frequency","Hz"), ("P","W"), ("P_filtered","W"),
        ("Q","var"), ("Q_filtered","var"),
        ("vod","V"), ("voq","V"),
        ("iod","A"), ("ioq","A"), ("ifd","A"), ("ifq","A"),
        ("md",""), ("mq",""), ("battery_current","A"),
    ]:
        print(f"{name:<18}: {a[name][-1]:14.6f} {unit}")
    print("="*82)

    # Complete numerical history.
    npz_path = os.path.join(args.output_dir, "scenario_history.npz")
    np.savez_compressed(npz_path, **a)

    # Clean CSV specifically for MATLAB/Python frequency validation.
    freq_csv = os.path.join(args.output_dir, "python_frequency_scenario.csv")
    pd.DataFrame({
        "Time_s": a["time"],
        "Frequency_Hz": a["frequency"],
    }).to_csv(freq_csv, index=False)

    # Useful compact event/operating-condition CSV.
    condition_csv = os.path.join(args.output_dir, "scenario_conditions.csv")
    pd.DataFrame({
        "Time_s": a["time"],
        "Load_kW": a["nominal_load_power"]/1000.0,
        "Grid_connected": a["nominal_grid_connected"],
        **{f"Breaker_G{i}": a[f"breaker_g{i}"] for i in range(7)},
    }).to_csv(condition_csv, index=False)

    t = a["time"]
    specs = [
        ("frequency.png", [(a["frequency"],"Frequency")],
         "Frequency (Hz)", "Frequency Response"),
        ("load_profile.png", [(a["nominal_load_power"]/1000.0,"Nominal load")],
         "Load (kW)", "Scenario Load Profile"),
        ("grid_status.png", [(a["nominal_grid_connected"],"Grid connected")],
         "Connected = 1", "Nominal Grid Status"),
        ("breaker_groups.png",
         [(a[f"breaker_g{i}"],f"G{i}") for i in range(7)],
         "Command", "Generated Three-Phase Breaker Group Commands"),
        ("active_power.png", [(a["P"],"P_B3"),(a["P_filtered"],"P_filtered")],
         "Power (W)", "Active Power"),
        ("reactive_power.png", [(a["Q"],"Q_B3"),(a["Q_filtered"],"Q_filtered")],
         "Reactive power (var)", "Reactive Power"),
        ("b3_voltage_dq.png",
         [(a["vod"],"Vod"),(a["voq"],"Voq"),(a["vod_ref"],"Vod_ref")],
         "Voltage (V)", "B3 Voltage and Reference"),
        ("b3_current_dq.png", [(a["iod"],"Iod"),(a["ioq"],"Ioq")],
         "Current (A)", "B3 Current"),
        ("b1_current_dq.png",
         [(a["ifd"],"Ifd"),(a["ifq"],"Ifq"),
          (a["ifd_ref"],"Ifd_ref"),(a["ifq_ref"],"Ifq_ref")],
         "Current (A)", "B1 Current and References"),
        ("voltage_commands.png",
         [(a["vd_command"],"Vd_command"),(a["vq_command"],"Vq_command")],
         "Voltage command", "Current-Controller Voltage Commands"),
        ("modulation_dq.png", [(a["md"],"md"),(a["mq"],"mq")],
         "Modulation", "dq Modulation Commands"),
        ("aux_dq.png", [(a["aux_d"],"aux_d"),(a["aux_q"],"aux_q")],
         "Auxiliary value", "Unidentified Generated Auxiliary dq Group"),
    ]

    for filename, series, ylabel, title in specs:
        save_plot(
            os.path.join(args.output_dir, filename),
            t, series, ylabel, title
        )

    print("\nSaved results:")
    print(" ", npz_path)
    print(" ", freq_csv)
    print(" ", condition_csv)
    for filename, *_ in specs:
        print(" ", os.path.join(args.output_dir, filename))

    return history


if __name__ == "__main__":
    main()
