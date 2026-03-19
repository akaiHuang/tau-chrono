#!/usr/bin/env python3
"""
Experiment 1: Non-Uniform Noise Surface Code Scaling (IMPROVED)
================================================================

Fixes over the original exp1_qec_baseline.py:
1. ERROR BARS: Every data point is the mean of N_TRIALS independent runs
   with different random seeds. Reports mean +/- std.
2. LARGER CODE DISTANCES: d=3,5,7,9,11 (was only d=3,5,7).
3. MORE REALISTIC NOISE: Adds a realistic noise profile where 2Q gates
   are 10x noisier than 1Q gates (non-uniform noise).
4. MEASUREMENT ERRORS: Includes measurement error model.
5. SAVES JSON: All results with error bars saved to JSON.

Uses Google Stim for circuit-level simulation, PyMatching for MWPM decoding.
"""

from __future__ import annotations

import json
import os
import sys
import time
import numpy as np
import stim
import pymatching

# ============================================================
# Configuration
# ============================================================
N_TRIALS = 5           # independent runs per data point
DISTANCES = [3, 5, 7, 9, 11]
NOISE_RATES = np.logspace(-3, -1.5, 8)  # 8 points from 1e-3 to ~0.032
SHOTS_PER_TRIAL = 50_000   # per trial (x5 trials = 250k effective)
BATCH_SIZE = 50_000
RESULTS_DIR = "/Users/akaihuangm1/Desktop/github/tau-chrono/results"


def run_single_trial(distance: int, noise_rate: float, num_shots: int,
                     noise_model: str = "uniform", seed: int = 0) -> dict:
    """
    Run a single trial for one (distance, noise_rate) point.

    noise_model: "uniform" or "realistic"
      - uniform: same depolarizing rate everywhere
      - realistic: 2Q gates 10x noisier, measurement errors included
    """
    rounds = distance

    if noise_model == "uniform":
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            rounds=rounds,
            distance=distance,
            after_clifford_depolarization=noise_rate,
            after_reset_flip_probability=noise_rate,
            before_measure_flip_probability=noise_rate,
            before_round_data_depolarization=noise_rate,
        )
    else:
        # Realistic: 2Q gates 10x noisier, measurement errors separate
        p_1q = noise_rate
        p_2q = min(noise_rate * 10, 0.5)  # cap at 50%
        p_meas = noise_rate * 2  # measurement errors
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            rounds=rounds,
            distance=distance,
            after_clifford_depolarization=p_2q,  # dominated by 2Q gates
            after_reset_flip_probability=p_1q,
            before_measure_flip_probability=p_meas,
            before_round_data_depolarization=p_1q,
        )

    # Build decoder
    dem = circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)

    # Sample with explicit seed for reproducibility
    sampler = circuit.compile_detector_sampler(seed=seed)
    detection_events, observable_flips = sampler.sample(
        num_shots, separate_observables=True
    )

    # Decode
    predictions = matcher.decode_batch(detection_events)
    logical_errors = (predictions != observable_flips).flatten()

    total_logical_errors = int(logical_errors.sum())
    standard_rate = total_logical_errors / num_shots

    # Post-selection: zero-defect shots
    zero_defect_mask = ~detection_events.any(axis=1)
    zero_defect_count = int(zero_defect_mask.sum())
    survival = zero_defect_count / num_shots

    if zero_defect_count > 0:
        ps_logical_errors = int(logical_errors[zero_defect_mask].sum())
        ps_rate = ps_logical_errors / zero_defect_count
    else:
        ps_logical_errors = 0
        ps_rate = float('nan')

    return {
        "standard_rate": standard_rate,
        "ps_rate": ps_rate,
        "survival": survival,
        "total_logical_errors": total_logical_errors,
        "ps_logical_errors": ps_logical_errors,
        "zero_defect_count": zero_defect_count,
    }


def run_experiment(noise_model: str = "uniform"):
    """Run full experiment with error bars across multiple trials."""

    print("=" * 70)
    print(f"  Experiment 1: Surface Code Scaling ({noise_model} noise)")
    print(f"  Distances: {DISTANCES}")
    print(f"  Noise rates: {len(NOISE_RATES)} points")
    print(f"  Trials per point: {N_TRIALS}")
    print(f"  Shots per trial: {SHOTS_PER_TRIAL:,}")
    print("=" * 70)

    all_results = {}
    t0 = time.time()

    for d in DISTANCES:
        all_results[str(d)] = []
        for p_idx, p in enumerate(NOISE_RATES):
            trial_std_rates = []
            trial_ps_rates = []
            trial_survivals = []

            for trial in range(N_TRIALS):
                seed = trial * 1000 + d * 100 + p_idx
                try:
                    res = run_single_trial(d, p, SHOTS_PER_TRIAL, noise_model, seed)
                    trial_std_rates.append(res["standard_rate"])
                    trial_ps_rates.append(res["ps_rate"])
                    trial_survivals.append(res["survival"])
                except Exception as e:
                    print(f"  WARNING: d={d}, p={p:.2e}, trial {trial} failed: {e}")
                    continue

            if len(trial_std_rates) == 0:
                continue

            std_rates = np.array(trial_std_rates)
            ps_rates_arr = np.array(trial_ps_rates)
            surv_arr = np.array(trial_survivals)

            # Compute mean +/- std (handle NaN in ps_rates)
            std_mean = float(np.mean(std_rates))
            std_std = float(np.std(std_rates))

            valid_ps = ps_rates_arr[~np.isnan(ps_rates_arr)]
            if len(valid_ps) > 0:
                ps_mean = float(np.mean(valid_ps))
                ps_std = float(np.std(valid_ps))
            else:
                ps_mean = float('nan')
                ps_std = float('nan')

            surv_mean = float(np.mean(surv_arr))
            surv_std = float(np.std(surv_arr))

            point_result = {
                "distance": d,
                "noise_rate": float(p),
                "noise_model": noise_model,
                "n_trials": len(trial_std_rates),
                "shots_per_trial": SHOTS_PER_TRIAL,
                "standard_rate_mean": std_mean,
                "standard_rate_std": std_std,
                "ps_rate_mean": ps_mean,
                "ps_rate_std": ps_std,
                "survival_mean": surv_mean,
                "survival_std": surv_std,
                "raw_standard_rates": [float(x) for x in trial_std_rates],
                "raw_ps_rates": [float(x) if not np.isnan(x) else None for x in trial_ps_rates],
            }
            all_results[str(d)].append(point_result)

            ps_str = f"{ps_mean:.2e}+/-{ps_std:.2e}" if not np.isnan(ps_mean) else "N/A"
            print(f"  d={d:2d}, p={p:.2e} | std={std_mean:.2e}+/-{std_std:.2e}, "
                  f"ps={ps_str}, surv={surv_mean:.4f}")

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s ({elapsed/60:.1f}min)")

    # Compute scaling exponents
    scaling = {}
    print("\nScaling exponents (log-log fits):")
    print("-" * 60)
    for d_str in sorted(all_results.keys(), key=int):
        d = int(d_str)
        points = all_results[d_str]
        if len(points) < 2:
            continue

        p_arr = np.array([pt["noise_rate"] for pt in points])
        std_arr = np.array([pt["standard_rate_mean"] for pt in points])
        ps_arr = np.array([pt["ps_rate_mean"] for pt in points])

        # Standard scaling
        valid = std_arr > 0
        if valid.sum() >= 2:
            coeffs = np.polyfit(np.log10(p_arr[valid]), np.log10(std_arr[valid]), 1)
            std_slope = coeffs[0]
        else:
            std_slope = float('nan')

        # Post-selected scaling
        valid_ps = (~np.isnan(ps_arr)) & (ps_arr > 0)
        if valid_ps.sum() >= 2:
            coeffs_ps = np.polyfit(np.log10(p_arr[valid_ps]), np.log10(ps_arr[valid_ps]), 1)
            ps_slope = coeffs_ps[0]
        else:
            ps_slope = float('nan')

        expected_std = (d + 1) / 2
        expected_ps = d
        print(f"  d={d}: standard slope={std_slope:.2f} (expected {expected_std:.1f}), "
              f"ps slope={ps_slope:.2f} (expected {expected_ps:.1f})")

        scaling[d_str] = {
            "standard_slope": float(std_slope) if not np.isnan(std_slope) else None,
            "expected_standard": expected_std,
            "ps_slope": float(ps_slope) if not np.isnan(ps_slope) else None,
            "expected_ps": expected_ps,
        }

    return all_results, scaling


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Run both noise models
    results_uniform, scaling_uniform = run_experiment("uniform")
    results_realistic, scaling_realistic = run_experiment("realistic")

    # Save everything
    output = {
        "experiment": "exp1_nonuniform_noise_scaling",
        "description": "Surface code scaling with error bars, d=3-11, uniform and realistic noise",
        "config": {
            "n_trials": N_TRIALS,
            "shots_per_trial": SHOTS_PER_TRIAL,
            "distances": DISTANCES,
            "noise_rates": [float(x) for x in NOISE_RATES],
        },
        "uniform_noise": {
            "results": results_uniform,
            "scaling": scaling_uniform,
        },
        "realistic_noise": {
            "description": "2Q gates 10x noisier than 1Q, measurement errors 2x base rate",
            "results": results_realistic,
            "scaling": scaling_realistic,
        },
    }

    out_path = os.path.join(RESULTS_DIR, "exp1_nonuniform_noise_scaling.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
