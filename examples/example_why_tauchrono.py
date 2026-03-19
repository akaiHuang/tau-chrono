#!/usr/bin/env python3
"""
tau-chrono: Why you need it — a practical example.

This script demonstrates two things:
  1. OVERHEAD: tau-chrono adds almost zero cost (no extra circuits needed)
  2. BENEFIT: tau-chrono lets you run deeper circuits that naive model rejects

Run:
    pip install tau-chrono
    python examples/example_why_tauchrono.py
"""

import numpy as np
from tau_chrono import depolarizing, tau_chrono_compose

# =====================================================================
# YOUR QUANTUM HARDWARE
# =====================================================================
# These are real error rates from QuTech Tuna-9 process tomography.
# On your hardware, replace with your own calibration data.

GATE_ERRORS = {
    'h':  0.0214,   # Hadamard gate error rate
    'x':  0.0269,   # X gate
    'sx': 0.0110,   # sqrt(X) gate
    's':  0.0170,   # S gate
    'id': 0.0226,   # idle error
    'cx': 0.0400,   # CNOT (estimated from single-qubit data)
}

rho = np.array([[1, 0], [0, 0]], dtype=complex)    # |0>
sigma = np.eye(2, dtype=complex) / 2                # maximally mixed


# =====================================================================
# EXAMPLE 1: Overhead comparison
# =====================================================================
print("=" * 65)
print("EXAMPLE 1: What does tau-chrono cost?")
print("=" * 65)
print()
print("Standard approach (without tau-chrono):")
print("  - You already run gate characterization during calibration")
print("  - You use error rates to estimate: F = prod(1 - p_i)")
print("  - Cost: 0 extra circuits")
print()
print("With tau-chrono:")
print("  - Use the SAME calibration data (no extra circuits)")
print("  - Run tau_chrono_compose() instead of multiplying error rates")
print("  - Cost: 0 extra circuits, ~1ms computation time")
print()

import time
gates_20 = [depolarizing(GATE_ERRORS['h']) for _ in range(20)]
t0 = time.perf_counter()
for _ in range(100):
    result = tau_chrono_compose(gates_20, sigma_0=sigma, rho=rho)
elapsed = (time.perf_counter() - t0) / 100 * 1000

print(f"  Benchmark: tau_chrono_compose() for 20 gates = {elapsed:.2f} ms")
print(f"  Overhead: ZERO extra hardware time. Just {elapsed:.2f} ms of CPU.")
print()


# =====================================================================
# EXAMPLE 2: Benefit — circuit depth ceiling
# =====================================================================
print("=" * 65)
print("EXAMPLE 2: How much deeper can you go?")
print("=" * 65)
print()
print("Scenario: You're running a variational algorithm (VQE, QAOA, etc.)")
print("and want to know the maximum useful circuit depth.")
print()
print("The 'stop rule': if predicted fidelity F < 0.5, the circuit")
print("output is more noise than signal. Don't waste QPU time.")
print()

THRESHOLD = 0.5
max_depth_naive = 0
max_depth_tauchrono = 0

print(f"{'Depth':>5} | {'Naive F':>8} | {'τ-chrono F':>10} | {'Naive':>8} | {'τ-chrono':>8}")
print("-" * 55)

for depth in [2, 5, 10, 15, 20, 30, 40, 50, 60, 80, 100]:
    # Build circuit: mix of gates
    gate_sequence = []
    for i in range(depth):
        gate = ['h', 'sx', 's', 'x'][i % 4]
        gate_sequence.append(depolarizing(GATE_ERRORS[gate]))

    result = tau_chrono_compose(gate_sequence, sigma_0=sigma, rho=rho)

    f_naive = 1 - result.tau_multiplicative_total
    f_chrono = 1 - result.tau_bayesian_total

    naive_go = "GO" if f_naive >= THRESHOLD else "STOP"
    chrono_go = "GO" if f_chrono >= THRESHOLD else "STOP"

    if f_naive >= THRESHOLD:
        max_depth_naive = depth
    if f_chrono >= THRESHOLD:
        max_depth_tauchrono = depth

    print(f"{depth:5d} | {f_naive:8.4f} | {f_chrono:10.4f} | {naive_go:>8} | {chrono_go:>8}")

print()
print(f"  Naive model max usable depth:      {max_depth_naive}")
print(f"  tau-chrono max usable depth:        {max_depth_tauchrono}")
if max_depth_tauchrono > max_depth_naive:
    ratio = max_depth_tauchrono / max_depth_naive if max_depth_naive > 0 else float('inf')
    print(f"  tau-chrono extends depth by:        {ratio:.1f}x")
print()


# =====================================================================
# EXAMPLE 3: Real-world scenario — Bernstein-Vazirani
# =====================================================================
print("=" * 65)
print("EXAMPLE 3: Bernstein-Vazirani algorithm decision")
print("=" * 65)
print()
print("You want to run BV with hidden string s=1011 (4 qubits).")
print("Each oracle call adds ~6 gates. How many repetitions can you do?")
print()

for n_rep in [1, 3, 5, 8, 12, 20]:
    # BV circuit: H + n_rep*(oracle) + H + measure
    # Oracle for s=1011: 3 CNOTs per repetition
    n_gates = 4 + n_rep * 6 + 4  # H_init + oracle*n_rep + H_final

    gates = [depolarizing(GATE_ERRORS['cx']) for _ in range(n_gates)]
    result = tau_chrono_compose(gates, sigma_0=sigma, rho=rho)

    f_naive = 1 - result.tau_multiplicative_total
    f_chrono = 1 - result.tau_bayesian_total

    naive_verdict = "RUN" if f_naive >= THRESHOLD else "SKIP"
    chrono_verdict = "RUN" if f_chrono >= THRESHOLD else "SKIP"

    print(f"  n_rep={n_rep:2d} ({n_gates:3d} gates): "
          f"Naive F={f_naive:.3f} [{naive_verdict:>4}]  "
          f"τ-chrono F={f_chrono:.3f} [{chrono_verdict:>4}]")

print()
print("Without tau-chrono: your compiler skips circuits that would")
print("actually produce correct results. You waste nothing on hardware")
print("but miss valid computation.")
print()
print("With tau-chrono: you know the circuit is still usable, so you")
print("run it and get the answer.")
print()

# =====================================================================
# SUMMARY
# =====================================================================
print("=" * 65)
print("SUMMARY")
print("=" * 65)
print()
print("  Cost of tau-chrono:    0 extra circuits, <1ms CPU per circuit")
print("  Benefit:               Run deeper circuits with confidence")
print(f"  Depth extension:       {max_depth_naive} → {max_depth_tauchrono} gates ({max_depth_tauchrono/max_depth_naive:.1f}x)" if max_depth_naive > 0 else "")
print()
print("  Install:  pip install tau-chrono")
print("  GitHub:   https://github.com/akaiHuang/tau-chrono")
print("  Website:  https://tau-chrono.pages.dev")
