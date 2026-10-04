"""Short-horizon sample-by-sample diagnostic for the exact scenario benchmark.

Runs the fixed-topology scenario for a short interval (default 20 ms) and
exports enough internal data to locate the first MATLAB/Python divergence.

Required files:
    benchmark_scenarios.py
    scenario_plant_exact.py
    scenario_measurements.py
    scenario_parameters.py
    controller.py
    pwm.py
    battery_dc.py
    parameters.py
    states.py

The script deliberately uses MatlabScenarioBenchmark.step(), so the diagnostic
tests the exact same integration path as the benchmark under investigation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from benchmark_scenarios import MatlabScenarioBenchmark


def make_row(sim, sample, k):
    m = sample.measurements
    c = sample.controller
    p = sample.pwm
    plant = sim.plant

    row = {
        "step": k,
        "time_s": sample.time,

        "Vdc_V": sample.vdc,
        "Idc_A": m.battery_current,
        "SOC_pct": sample.battery_soc,

        "frequency_Hz": c.frequency,
        "omega_rad_s": c.omega,
        "theta_rad": c.theta,

        "P_W": m.P,
        "Q_var": m.Q,
        "P_filtered_W": c.P_filtered,
        "Q_filtered_var": c.Q_filtered,

        "Vod_V": m.vod,
        "Voq_V": m.voq,
        "Iod_A": m.iod,
        "Ioq_A": m.ioq,
        "Ifd_A": m.ifd,
        "Ifq_A": m.ifq,

        "Voa_V": m.vb3_abc[0],
        "Vob_V": m.vb3_abc[1],
        "Voc_V": m.vb3_abc[2],

        "Ioa_A": m.ib3_abc[0],
        "Iob_A": m.ib3_abc[1],
        "Ioc_A": m.ib3_abc[2],

        "Ifa_A": m.ib1_abc[0],
        "Ifb_A": m.ib1_abc[1],
        "Ifc_A": m.ib1_abc[2],

        "aux_a": m.aux_abc[0],
        "aux_b": m.aux_abc[1],
        "aux_c": m.aux_abc[2],

        "ifd_ref_A": c.ifd_ref,
        "ifq_ref_A": c.ifq_ref,
        "vd_command_V": c.vd_command,
        "vq_command_V": c.vq_command,

        "md": c.md,
        "mq": c.mq,
        "mod_alpha": c.modulation_alpha,
        "mod_beta": c.modulation_beta,

        "carrier": p.carrier,
        "ma": p.ma,
        "mb": p.mb,
        "mc": p.mc,
    }

    # PWM gates generated THIS sample.
    for i, value in enumerate(p.gates):
        row[f"pwm_gate_{i}"] = int(value)

    # Physical switch states used for this sample.
    for i, value in enumerate(sample.plant.switch_states):
        row[f"switch_{i}"] = int(value)

    # Gate memory actually used by the SPS switch solver this sample.
    gate_used = sample.plant.gate_states
    for i, value in enumerate(gate_used):
        row[f"gate_used_{i}"] = int(value)

    # Electrical state x[k] that produced this sample.  benchmark.step() has
    # already committed x[k+1], so reconstruct x[k] from a snapshot supplied
    # by the caller separately; placeholders are overwritten below.
    for i in range(15):
        row[f"x_{i}"] = np.nan

    # Exact-solver diagnostics.
    if hasattr(sample.plant, "loops_remaining"):
        row["switch_loops_remaining"] = int(sample.plant.loops_remaining)

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--time",
        type=float,
        default=0.020,
        help="Diagnostic duration in seconds (default: 20 ms).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="exact_diagnostic_20ms.csv",
    )
    parser.add_argument(
        "--npz",
        type=str,
        default="exact_diagnostic_20ms.npz",
    )
    parser.add_argument(
        "--print_first",
        type=int,
        default=20,
        help="Number of initial samples to print.",
    )
    args = parser.parse_args()

    sim = MatlabScenarioBenchmark()
    dt = sim.params.dt
    n_steps = int(round(args.time / dt))

    rows = []
    x_before = []
    y_samples = []

    for k in range(n_steps):
        # x[k] before benchmark.step() commits x[k+1].
        xk = np.asarray(sim.state.network.x, dtype=np.float64).copy()
        x_before.append(xk)

        sample = sim.step()
        y_samples.append(np.asarray(sample.plant.y, dtype=np.float64).copy())
        row = make_row(sim, sample, k)

        for i in range(15):
            row[f"x_{i}"] = xk[i]

        rows.append(row)

    df = pd.DataFrame(rows)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    npz = Path(args.npz)
    npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz,
        time_s=df["time_s"].to_numpy(),
        x=np.asarray(x_before),
        y=np.asarray(y_samples, dtype=np.float64),
    )

    print("=" * 88)
    print("EXACT SPS SHORT-HORIZON DIAGNOSTIC")
    print("=" * 88)
    print(f"dt       : {dt:.9g} s")
    print(f"duration : {args.time:.9g} s")
    print(f"steps    : {n_steps}")
    print(f"CSV      : {out}")
    print(f"NPZ      : {npz}")
    print("=" * 88)

    cols = [
        "step", "time_s", "Vdc_V",
        "frequency_Hz", "P_W", "P_filtered_W",
        "Vod_V", "Voq_V", "Iod_A", "Ioq_A",
        "md", "mq", "carrier",
        "pwm_gate_0", "pwm_gate_1",
        "pwm_gate_2", "pwm_gate_3",
        "pwm_gate_4", "pwm_gate_5",
    ]

    print("\nFIRST SAMPLES")
    print(df[cols].head(args.print_first).to_string(index=False))

    print("\nINITIAL SPS STATE x[0]")
    print(df[[f"x_{i}" for i in range(15)]].iloc[0].to_numpy())

    print("\nINITIAL SWITCH STATES")
    print(df[[f"switch_{i}" for i in range(27)]].iloc[0].to_numpy(dtype=int))

    print("\nINITIAL GATES USED BY SPS")
    print(df[[f"gate_used_{i}" for i in range(27)]].iloc[0].to_numpy(dtype=int))

    # Check first changes in physical switch states.
    sw = df[[f"switch_{i}" for i in range(27)]].to_numpy(dtype=int)
    changed = np.any(sw[1:] != sw[:-1], axis=1)
    change_rows = np.where(changed)[0] + 1

    print("\nFIRST PHYSICAL SWITCH-STATE CHANGES")
    if change_rows.size:
        for idx in change_rows[:20]:
            changed_idx = np.where(sw[idx] != sw[idx-1])[0]
            print(
                f"step={idx:6d} t={df.time_s.iloc[idx]:.9f} "
                f"switches={changed_idx.tolist()} "
                f"new={sw[idx, changed_idx].tolist()}"
            )
    else:
        print("No physical switch-state changes after the first recorded sample.")

    # Check first changes in generated PWM gates.
    pg = df[[f"pwm_gate_{i}" for i in range(6)]].to_numpy(dtype=int)
    pchanged = np.any(pg[1:] != pg[:-1], axis=1)
    prows = np.where(pchanged)[0] + 1

    print("\nFIRST PWM-GATE CHANGES")
    if prows.size:
        for idx in prows[:20]:
            changed_idx = np.where(pg[idx] != pg[idx-1])[0]
            print(
                f"step={idx:6d} t={df.time_s.iloc[idx]:.9f} "
                f"gates={changed_idx.tolist()} "
                f"new={pg[idx, changed_idx].tolist()}"
            )
    else:
        print("No PWM gate changes in diagnostic interval.")

    print("\nFINAL DIAGNOSTIC SAMPLE")
    last = df.iloc[-1]
    for name in [
        "time_s", "Vdc_V", "frequency_Hz",
        "P_W", "P_filtered_W", "Q_var", "Q_filtered_var",
        "Vod_V", "Voq_V", "Iod_A", "Ioq_A",
        "Ifd_A", "Ifq_A", "md", "mq",
    ]:
        print(f"{name:<18}: {last[name]: .12g}")


if __name__ == "__main__":
    main()
