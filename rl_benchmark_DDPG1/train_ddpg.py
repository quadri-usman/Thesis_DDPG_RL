"""Train Stage-1 DDPG for adaptive GFM PI gains.

Environment:
    RandomizedGFMGainTuningEnv
Observation:
    [df, RoCoF, evd, Evd, evq, eid, Eid, eiq]
Action:
    [dKpv, dKiv, dKpi, dKii]

The script:
  * trains only after the environment warm-up becomes RL-active;
  * stores only RL-active transitions in replay;
  * performs one gradient update per stored transition by default;
  * logs reward components, frequency metrics, RoCoF, settling and gains;
  * periodically evaluates with exploration OFF on fixed validation seeds;
  * saves latest and best checkpoints plus CSV histories.

Before training, use rocof_scale_hz_s=15 in both RewardConfig and
ObservationConfig of gfm_env_random.py, based on the fixed-PI 10-seed baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from gfm_env_random import (
    RandomizedGFMGainTuningEnv,
    RewardConfig,
    ObservationConfig,
)
from ddpg_agent import DDPGAgent, DDPGConfig


REWARD_KEYS = (
    "J_frequency",
    "J_rocof",
    "J_settling",
    "J_voltage",
    "J_current",
    "J_gain_motion",
)


def make_env():
    # Explicit here so training does not silently depend on old 1-Hz/s defaults.
    reward_cfg = RewardConfig(rocof_scale_hz_s=15.0)
    obs_cfg = ObservationConfig(rocof_scale_hz_s=15.0)
    return RandomizedGFMGainTuningEnv(
        reward_config=reward_cfg,
        observation_config=obs_cfg,
    )


def run_evaluation(agent, seeds):
    """Deterministic-policy evaluation; no replay writes and no learning."""
    rows = []

    for seed in seeds:
        env = make_env()
        state, info = env.reset(seed=int(seed))

        ep_reward = 0.0
        max_abs_df = 0.0
        max_abs_rocof = 0.0
        settling_times = []

        terminated = truncated = False

        while not (terminated or truncated):
            if info["rl_active"]:
                action = agent.select_action(state, explore=False)
            else:
                action = np.zeros(4, dtype=np.float32)

            next_state, reward, terminated, truncated, next_info = env.step(action)

            if next_info["rl_active"]:
                ep_reward += reward
                max_abs_df = max(
                    max_abs_df, abs(next_info["frequency_deviation"])
                )
                max_abs_rocof = max(
                    max_abs_rocof, abs(next_info["rocof"])
                )

                st = next_info["last_settling_time"]
                if np.isfinite(st):
                    if not settling_times or not np.isclose(
                        st, settling_times[-1]
                    ):
                        settling_times.append(float(st))

            state, info = next_state, next_info

        rows.append({
            "seed": int(seed),
            "reward": float(ep_reward),
            "max_abs_df": float(max_abs_df),
            "max_abs_rocof": float(max_abs_rocof),
            "mean_settling_time": (
                float(np.mean(settling_times))
                if settling_times else math.nan
            ),
            "Kpv": float(info["Kpv"]),
            "Kiv": float(info["Kiv"]),
            "Kpi": float(info["Kpi"]),
            "Kii": float(info["Kii"]),
        })

    def mean_finite(key):
        x = np.asarray([r[key] for r in rows], dtype=float)
        x = x[np.isfinite(x)]
        return float(np.mean(x)) if len(x) else math.nan

    return {
        "mean_reward": mean_finite("reward"),
        "mean_max_abs_df": mean_finite("max_abs_df"),
        "mean_max_abs_rocof": mean_finite("max_abs_rocof"),
        "mean_settling_time": mean_finite("mean_settling_time"),
        "rows": rows,
    }


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output_dir", type=str, default="ddpg_results")
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--checkpoint_every", type=int, default=10)
    ap.add_argument("--updates_per_step", type=int, default=1)
    ap.add_argument(
        "--eval_seeds", type=int, nargs="+",
        default=[1001, 1002, 1003]
    )
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = DDPGConfig(seed=args.seed)
    agent = DDPGAgent(cfg, device=args.device)

    with (outdir / "training_config.json").open("w") as f:
        json.dump({
            "episodes": args.episodes,
            "training_seed": args.seed,
            "eval_every": args.eval_every,
            "checkpoint_every": args.checkpoint_every,
            "updates_per_step": args.updates_per_step,
            "eval_seeds": args.eval_seeds,
            "ddpg": {
                k: (list(v) if isinstance(v, tuple) else v)
                for k, v in vars(cfg).items()
            },
            "reward_rocof_scale_hz_s": 15.0,
            "observation_rocof_scale_hz_s": 15.0,
        }, f, indent=2)

    print("=" * 92)
    print("STAGE-1 DDPG TRAINING")
    print("=" * 92)
    print("device             :", agent.device)
    print("episodes           :", args.episodes)
    print("state/action dims  :", cfg.state_dim, cfg.action_dim)
    print("learning starts    :", cfg.learning_starts, "RL-active transitions")
    print("replay capacity    :", cfg.buffer_size)
    print("batch size         :", cfg.batch_size)
    print("RoCoF normalization: 15 Hz/s")
    print("evaluation seeds   :", args.eval_seeds)
    print("=" * 92)

    episode_rows = []
    eval_rows = []
    best_eval_reward = -np.inf
    global_rl_steps = 0
    start_wall = time.time()

    for episode in range(1, args.episodes + 1):
        env = make_env()
        episode_seed = args.seed + episode - 1
        state, info = env.reset(seed=episode_seed)

        reward_sum = 0.0
        reward_component_sums = {k: 0.0 for k in REWARD_KEYS}
        rl_steps = 0
        max_abs_df = 0.0
        max_abs_rocof = 0.0
        settling_times = []
        actor_losses = []
        critic_losses = []

        terminated = truncated = False

        while not (terminated or truncated):
            # During warm-up the environment has no RL authority. Do not add
            # those transitions to replay because reward is intentionally zero.
            if info["rl_active"]:
                action = agent.select_action(state, explore=True)
            else:
                action = np.zeros(4, dtype=np.float32)

            next_state, reward, terminated, truncated, next_info = env.step(action)

            # A transition is trainable if RL was active for the interval that
            # just executed. This also captures the first action at activation.
            if next_info["rl_active"]:
                done_for_bootstrap = bool(terminated)
                agent.remember(
                    state,
                    action,
                    reward,
                    next_state,
                    done_for_bootstrap,
                )
                global_rl_steps += 1
                rl_steps += 1
                reward_sum += reward

                for key in REWARD_KEYS:
                    reward_component_sums[key] += float(next_info[key])

                max_abs_df = max(
                    max_abs_df,
                    abs(next_info["frequency_deviation"]),
                )
                max_abs_rocof = max(
                    max_abs_rocof,
                    abs(next_info["rocof"]),
                )

                st = next_info["last_settling_time"]
                if np.isfinite(st):
                    if not settling_times or not np.isclose(
                        st, settling_times[-1]
                    ):
                        settling_times.append(float(st))

                for _ in range(args.updates_per_step):
                    metrics = agent.update()
                    if metrics is not None:
                        actor_losses.append(metrics["actor_loss"])
                        critic_losses.append(metrics["critic_loss"])

            state, info = next_state, next_info

        row = {
            "episode": episode,
            "seed": episode_seed,
            "rl_steps": rl_steps,
            "reward": float(reward_sum),
            "max_abs_df": float(max_abs_df),
            "max_abs_rocof": float(max_abs_rocof),
            "mean_settling_time": (
                float(np.mean(settling_times))
                if settling_times else math.nan
            ),
            "Kpv_final": float(info["Kpv"]),
            "Kiv_final": float(info["Kiv"]),
            "Kpi_final": float(info["Kpi"]),
            "Kii_final": float(info["Kii"]),
            "exploration_std": float(agent.exploration_std),
            "actor_loss": (
                float(np.mean(actor_losses)) if actor_losses else math.nan
            ),
            "critic_loss": (
                float(np.mean(critic_losses)) if critic_losses else math.nan
            ),
        }
        for key in REWARD_KEYS:
            row[key + "_sum"] = reward_component_sums[key]

        episode_rows.append(row)

        print(
            f"ep={episode:4d}/{args.episodes} "
            f"seed={episode_seed:5d} "
            f"R={reward_sum:11.2f} "
            f"|df|max={max_abs_df:7.4f} Hz "
            f"|RoCoF|max={max_abs_rocof:8.3f} Hz/s "
            f"K=[{info['Kpv']:.3f},{info['Kiv']:.3f},"
            f"{info['Kpi']:.3f},{info['Kii']:.3f}] "
            f"sigma={agent.exploration_std:.3f}"
        )

        write_csv(outdir / "training_history.csv", episode_rows)
        agent.save(outdir / "latest.pt")

        if episode % args.checkpoint_every == 0:
            agent.save(outdir / f"checkpoint_ep{episode:04d}.pt")

        if episode % args.eval_every == 0:
            ev = run_evaluation(agent, args.eval_seeds)
            erow = {
                "episode": episode,
                "mean_reward": ev["mean_reward"],
                "mean_max_abs_df": ev["mean_max_abs_df"],
                "mean_max_abs_rocof": ev["mean_max_abs_rocof"],
                "mean_settling_time": ev["mean_settling_time"],
            }
            eval_rows.append(erow)
            write_csv(outdir / "evaluation_history.csv", eval_rows)

            print(
                "  EVAL "
                f"R={ev['mean_reward']:.2f} "
                f"|df|max={ev['mean_max_abs_df']:.4f} Hz "
                f"|RoCoF|max={ev['mean_max_abs_rocof']:.3f} Hz/s "
                f"Ts={ev['mean_settling_time']:.4f} s"
            )

            if ev["mean_reward"] > best_eval_reward:
                best_eval_reward = ev["mean_reward"]
                agent.save(outdir / "best.pt")
                with (outdir / "best_evaluation.json").open("w") as f:
                    json.dump(
                        {
                            k: v for k, v in ev.items()
                            if k != "rows"
                        },
                        f,
                        indent=2,
                    )
                write_csv(
                    outdir / "best_evaluation_seeds.csv",
                    ev["rows"],
                )
                print("  -> new best deterministic policy saved")

    elapsed = time.time() - start_wall

    print("\n" + "=" * 92)
    print("TRAINING COMPLETE")
    print("=" * 92)
    print("RL-active transitions:", global_rl_steps)
    print("gradient updates      :", agent.total_updates)
    print("best eval reward      :", best_eval_reward)
    print("wall time             :", f"{elapsed/60:.2f} min")
    print("results               :", outdir)
    print("best policy           :", outdir / "best.pt")
    print("=" * 92)


if __name__ == "__main__":
    main()
