import numpy as np
from gfm_env_random_target import RandomizedGFMGainTuningEnv

nominal = np.array([2.0, 14.0, 2.0, 20.0])

print("=" * 100)
print("10-SEED TARGET-ACTION ZERO-ACTION REGRESSION")
print("=" * 100)

all_finite = True
all_nominal = True
all_safe = True

for seed in range(10):

    env = RandomizedGFMGainTuningEnv(split="train")
    obs, info = env.reset(seed=seed)

    fmin = np.inf
    fmax = -np.inf
    max_rocof = 0.0
    max_gain_error = 0.0

    terminated = truncated = False

    while not (terminated or truncated):

        action = np.zeros(4, dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)

        gains = np.array([
            info["Kpv"],
            info["Kiv"],
            info["Kpi"],
            info["Kii"],
        ])

        max_gain_error = max(
            max_gain_error,
            np.max(np.abs(gains - nominal))
        )

        fmin = min(fmin, info["frequency"])
        fmax = max(fmax, info["frequency"])

        max_rocof = max(
            max_rocof,
            abs(info["rocof"])
        )

        if not np.all(np.isfinite(obs)):
            all_finite = False

    gains_ok = max_gain_error < 1e-10
    safe = not info["safety_terminated"]

    all_nominal &= gains_ok
    all_safe &= safe

    print(f"\nSeed {seed}")
    print("-" * 80)
    print("gains nominal :", gains_ok)
    print("max gain error:", max_gain_error)
    print("f_min         :", fmin)
    print("f_max         :", fmax)
    print("max |RoCoF|   :", max_rocof)
    print("safe          :", safe)

    if not safe:
        print("reason        :", info["safety_reason"])


print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)

print("All finite       :", all_finite)
print("All gains nominal:", all_nominal)
print("All episodes safe:", all_safe)

if all_finite and all_nominal and all_safe:
    print("\nPASS: target-action zero-action regression")
else:
    print("\nFAIL: investigate before RL training")