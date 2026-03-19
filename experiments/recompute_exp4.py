#!/usr/bin/env python3
"""Recompute Exp 4 using REAL gate characterization from saved T-9 data."""

import json
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tau_chrono import depolarizing, bayesian_compose

# Load T-9 results
with open('results/validation_tuna9_20260319_185441.json') as f:
    data = json.load(f)

# Extract gate error rates from REAL tomography
gate_error_rates = {}
for gc in data['exp3_gate_characterization']:
    gate_name = gc['gate']
    tomo = gc['tomography']
    # Extract error by comparing measured output to ideal output
    # Use the measurement basis that directly tests the gate action
    if gate_name == 'id':
        # id|0>=|0> (Z), id|1>=|1> (Z)
        p_err = 1.0 - (tomo['z0_z']['p0'] + (1 - tomo['z1_z']['p0'])) / 2
    elif gate_name == 'x':
        # X|0>=|1> (Z), X|1>=|0> (Z)
        p_err = 1.0 - ((1 - tomo['z0_z']['p0']) + tomo['z1_z']['p0']) / 2
    elif gate_name == 'h':
        # H|0>=|+> → measure X, expect P(0)=1.0
        # H|1>=|-> → measure X, expect P(0)=0.0
        p_err = (1.0 - tomo['z0_x']['p0'] + tomo['z1_x']['p0']) / 2
    elif gate_name == 'sx':
        # SX|+>=|+> (up to phase) → measure X, expect P(0)=1.0
        p_err = 1.0 - tomo['x0_x']['p0']
    elif gate_name == 's':
        # S|0>=|0> (Z), S|1>=i|1> (Z)
        p_err = 1.0 - (tomo['z0_z']['p0'] + (1 - tomo['z1_z']['p0'])) / 2
    else:
        p_err = 0.05
    gate_error_rates[gate_name] = max(p_err, 0.001)

print("=== REAL Gate Error Rates (from T-9 tomography) ===")
for g, p in gate_error_rates.items():
    print(f"  {g}: p_error = {p:.4f}")

print("\n=== EXP 4: Prediction Validation (REAL data) ===")
print(f"{'Depth':>5} | {'Actual F':>8} | {'Naive F':>8} | {'Bayes F':>8} | {'Naive err':>9} | {'Bayes err':>9} | {'Winner':>10} | {'Improv':>7}")
print("-" * 85)

results = []
for dr in data['exp1_depth_scaling']:
    depth = dr['depth']
    actual_f = dr['p_correct']
    forward_gates = dr['forward_gates']

    # Build channels from REAL gate error rates
    all_gate_names = forward_gates + list(reversed(forward_gates))
    all_channels = [depolarizing(gate_error_rates.get(g, 0.05)) for g in all_gate_names]

    rho = np.array([[1, 0], [0, 0]], dtype=complex)
    sigma = np.eye(2, dtype=complex) / 2

    result = bayesian_compose(all_channels, sigma_0=sigma, rho=rho)

    tau_naive = result.tau_multiplicative_total
    tau_bayes = result.tau_bayesian_total
    f_naive = 1 - tau_naive
    f_bayes = 1 - tau_bayes

    error_naive = abs(actual_f - f_naive)
    error_bayes = abs(actual_f - f_bayes)
    bayes_better = error_bayes < error_naive
    improvement = (error_naive - error_bayes) / error_naive * 100 if error_naive > 0 else 0

    winner = "BAYES" if bayes_better else "NAIVE"
    print(f"{depth:5d} | {actual_f:8.4f} | {f_naive:8.4f} | {f_bayes:8.4f} | {error_naive:9.4f} | {error_bayes:9.4f} | {winner:>10} | {improvement:6.1f}%")

    results.append({
        'depth': depth,
        'actual_fidelity': actual_f,
        'tau_naive': tau_naive, 'tau_bayes': tau_bayes,
        'f_naive': f_naive, 'f_bayes': f_bayes,
        'error_naive': error_naive, 'error_bayes': error_bayes,
        'bayes_wins': bayes_better, 'improvement_pct': improvement,
    })

# Summary
n_wins = sum(1 for r in results if r['bayes_wins'])
avg_imp = np.mean([r['improvement_pct'] for r in results if r['bayes_wins']]) if n_wins > 0 else 0
print(f"\nBayesian wins: {n_wins}/{len(results)} depths")
print(f"Avg improvement when Bayesian wins: {avg_imp:.1f}%")

# Save updated results
data['exp4_prediction_validation_real'] = results
with open('results/validation_tuna9_20260319_185441.json', 'w') as f:
    json.dump(data, f, indent=2, default=str)
print(f"\nUpdated results saved.")
