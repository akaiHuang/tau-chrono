#!/usr/bin/env python3
"""
tau-chrono showcase: two experiments that REQUIRE real quantum hardware.

Experiment A: COST SAVINGS — Same task, fewer shots needed
  BV algorithm: tau-chrono says "circuit is reliable" → no need for
  expensive error mitigation or extra shots. Without tau-chrono,
  naive model says "unreliable" → forces 3x shots for majority voting.

Experiment B: DEPTH CEILING — Same budget, better results
  GHZ entanglement: with tau-chrono, you can trust deeper circuits
  and prepare larger entangled states. Without tau-chrono, you stop
  too early and miss valid results.

Usage:
    python experiments/run_showcase.py --backend "Tuna-9" --shots 4096
    python experiments/run_showcase.py --backend "QX emulator" --shots 1024
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
from qiskit import QuantumCircuit
from qiskit.compiler import transpile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tau_chrono import depolarizing, tau_chrono_compose

# Real T-9 gate errors from process tomography
GATE_ERRORS = {
    'h': 0.0214, 'x': 0.0269, 'sx': 0.0110,
    's': 0.0170, 'id': 0.0226, 'cx': 0.0400,
}
RHO = np.array([[1, 0], [0, 0]], dtype=complex)
SIGMA = np.eye(2, dtype=complex) / 2
THRESHOLD = 0.5


def get_backend(name):
    from qiskit_quantuminspire.qi_provider import QIProvider
    return QIProvider().get_backend(name)


def run_circuit(backend, qc, shots):
    transpiled = transpile(qc, backend, optimization_level=0)
    job = backend.run(transpiled, shots=shots)
    job.wait_for_final_state(timeout=1800)
    return job.result().get_counts(0)


def predict(n_gates_by_type):
    """Predict fidelity from gate counts by type."""
    channels = []
    for gate_type, count in n_gates_by_type.items():
        p = GATE_ERRORS.get(gate_type, 0.03)
        channels.extend([depolarizing(p)] * count)
    if not channels:
        return {'f_naive': 1.0, 'f_tauchrono': 1.0}
    result = tau_chrono_compose(channels, sigma_0=SIGMA, rho=RHO)
    return {
        'f_naive': 1 - result.tau_multiplicative_total,
        'f_tauchrono': 1 - result.tau_bayesian_total,
    }


# =====================================================================
# EXPERIMENT A: Cost to complete the same task
# =====================================================================
def experiment_a_cost(backend, shots):
    """
    Bernstein-Vazirani: find hidden string s=101.

    Without tau-chrono:
      Naive model says "unreliable" at n_rep>=3 → user adds majority
      voting (run 3x, take majority) → 3x QPU cost for same answer.

    With tau-chrono:
      tau-chrono says "still reliable" → run once → correct answer.
      Saves 2/3 of QPU time.
    """
    print("\n" + "=" * 65)
    print("EXPERIMENT A: Cost savings (same task, less QPU time)")
    print("=" * 65)
    print()
    print("Task: Find hidden string s=101 using Bernstein-Vazirani")
    print(f"Shots per run: {shots}")
    print()

    hidden = '101'
    n_q = len(hidden)
    results = []

    for n_rep in [1, 2, 3, 5, 8]:
        # Build BV circuit
        qc = QuantumCircuit(n_q + 1, n_q)
        for i in range(n_q):
            qc.h(i)
        qc.x(n_q); qc.h(n_q)

        for _ in range(n_rep):
            for i, bit in enumerate(reversed(hidden)):
                if bit == '1':
                    qc.cx(i, n_q)

        for i in range(n_q):
            qc.h(i)
        for i in range(n_q):
            qc.measure(i, i)

        expected = hidden[::-1] if n_rep % 2 == 1 else '0' * n_q

        # Count gates by type
        gate_counts = {'h': 2 * n_q, 'x': 1, 'cx': n_rep * hidden.count('1')}

        # Predict
        pred = predict(gate_counts)
        naive_reliable = pred['f_naive'] >= THRESHOLD
        chrono_reliable = pred['f_tauchrono'] >= THRESHOLD

        # Run on hardware
        t0 = time.time()
        counts = run_circuit(backend, qc, shots)
        elapsed = time.time() - t0

        # Check success
        p_correct = max(counts.get(expected, 0), counts.get(expected[::-1], 0)) / shots
        top_result = max(counts, key=counts.get)
        got_answer = (top_result == expected or top_result == expected[::-1])

        # Cost analysis
        if naive_reliable:
            naive_cost = shots  # run once
        else:
            naive_cost = shots * 3  # majority voting: 3 runs

        chrono_cost = shots  # tau-chrono always says run once (if reliable)
        savings = (1 - chrono_cost / naive_cost) * 100 if naive_cost > 0 else 0

        status_naive = "RUN 1x" if naive_reliable else "RUN 3x (majority)"
        status_chrono = "RUN 1x" if chrono_reliable else "SKIP"

        print(f"  n_rep={n_rep}: P(correct)={p_correct:.3f}  "
              f"answer={'CORRECT' if got_answer else 'WRONG':>7}  "
              f"naive:[{status_naive}]  chrono:[{status_chrono}]  "
              f"savings:{savings:.0f}%")

        results.append({
            'n_rep': n_rep,
            'n_gates': qc.size(),
            'gate_counts': gate_counts,
            'expected': expected,
            'p_correct': p_correct,
            'got_answer': got_answer,
            'top_result': top_result,
            'counts': counts,
            'f_naive': pred['f_naive'],
            'f_tauchrono': pred['f_tauchrono'],
            'naive_reliable': naive_reliable,
            'chrono_reliable': chrono_reliable,
            'naive_cost_shots': naive_cost,
            'chrono_cost_shots': chrono_cost,
            'savings_pct': savings,
            'elapsed_s': round(elapsed, 1),
        })

    # Summary
    total_naive = sum(r['naive_cost_shots'] for r in results)
    total_chrono = sum(r['chrono_cost_shots'] for r in results)
    print(f"\n  Total QPU shots without tau-chrono: {total_naive:,}")
    print(f"  Total QPU shots with tau-chrono:    {total_chrono:,}")
    print(f"  Savings: {(1 - total_chrono/total_naive)*100:.0f}%")

    return results


# =====================================================================
# EXPERIMENT B: Performance ceiling
# =====================================================================
def experiment_b_ceiling(backend, shots):
    """
    GHZ state: prepare |00...0> + |11...1> on 2-5 qubits.

    This REQUIRES a quantum computer (real entanglement).

    Without tau-chrono:
      Naive model says "too noisy" at N qubits → stop.
      You never know if larger GHZ states are achievable.

    With tau-chrono:
      tau-chrono says "still usable" → keep going → bigger entanglement.
    """
    print("\n" + "=" * 65)
    print("EXPERIMENT B: Depth ceiling (same budget, better results)")
    print("=" * 65)
    print()
    print("Task: Prepare GHZ entangled state on increasing qubits")
    print(f"Budget: {shots} shots per circuit (fixed)")
    print("Success: GHZ fidelity = P(|00..0>) + P(|11..1>) > 0.5")
    print()

    results = []

    for n_qubits in [2, 3, 4, 5]:
        # Build GHZ circuit: H + CNOT chain
        qc = QuantumCircuit(n_qubits, n_qubits)
        qc.h(0)
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
        qc.measure(range(n_qubits), range(n_qubits))

        # Gate counts: 1 H + (n-1) CNOTs
        gate_counts = {'h': 1, 'cx': n_qubits - 1}
        pred = predict(gate_counts)

        naive_go = pred['f_naive'] >= THRESHOLD
        chrono_go = pred['f_tauchrono'] >= THRESHOLD

        # Run on hardware
        t0 = time.time()
        counts = run_circuit(backend, qc, shots)
        elapsed = time.time() - t0

        # GHZ fidelity
        zeros = '0' * n_qubits
        ones = '1' * n_qubits
        p0 = max(counts.get(zeros, 0), counts.get(zeros[::-1], 0)) / shots
        p1 = max(counts.get(ones, 0), counts.get(ones[::-1], 0)) / shots
        ghz_f = p0 + p1
        ghz_success = ghz_f > THRESHOLD

        print(f"  {n_qubits} qubits: GHZ_F={ghz_f:.4f} ({'OK' if ghz_success else 'FAIL'})  "
              f"naive:{'GO' if naive_go else 'STOP':>4}  "
              f"chrono:{'GO' if chrono_go else 'STOP':>4}  "
              f"({elapsed:.1f}s)")

        # Key question: did naive wrongly say STOP for a circuit that actually works?
        if not naive_go and ghz_success:
            print(f"    → Naive WRONG: said STOP but GHZ actually works!")
        if naive_go and not ghz_success:
            print(f"    → Naive WRONG: said GO but GHZ actually fails!")

        results.append({
            'n_qubits': n_qubits,
            'n_gates': qc.size(),
            'gate_counts': gate_counts,
            'ghz_fidelity': ghz_f,
            'ghz_success': ghz_success,
            'p_all_zeros': p0,
            'p_all_ones': p1,
            'counts': counts,
            'f_naive': pred['f_naive'],
            'f_tauchrono': pred['f_tauchrono'],
            'naive_go': naive_go,
            'chrono_go': chrono_go,
            'elapsed_s': round(elapsed, 1),
        })

    # Summary
    max_naive = max((r['n_qubits'] for r in results if r['naive_go']), default=0)
    max_chrono = max((r['n_qubits'] for r in results if r['chrono_go']), default=0)
    max_actual = max((r['n_qubits'] for r in results if r['ghz_success']), default=0)

    print(f"\n  Max GHZ qubits (naive model allows):     {max_naive}")
    print(f"  Max GHZ qubits (tau-chrono allows):       {max_chrono}")
    print(f"  Max GHZ qubits (actually works on T-9):   {max_actual}")

    return results


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', default='QX emulator')
    parser.add_argument('--shots', type=int, default=4096)
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    is_hw = 'emulator' not in args.backend.lower()

    print(f"tau-chrono Showcase Experiments")
    print(f"Backend: {args.backend} | Shots: {args.shots} | Hardware: {is_hw}")
    print(f"Timestamp: {timestamp}")

    backend = get_backend(args.backend)

    results = {
        'metadata': {
            'backend': args.backend, 'is_hardware': is_hw,
            'shots': args.shots, 'timestamp': timestamp,
            'datetime': datetime.now().isoformat(),
        },
        'experiment_a_cost': experiment_a_cost(backend, args.shots),
        'experiment_b_ceiling': experiment_b_ceiling(backend, args.shots),
    }

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    suffix = 'tuna9' if is_hw else 'emulator'
    out_path = os.path.join(out_dir, f'showcase_{suffix}_{timestamp}.json')

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*65}")
    print(f"Results saved: {out_path}")
    print(f"{'='*65}")


if __name__ == '__main__':
    main()
