#!/usr/bin/env python3
"""
Experiment B (redesigned): Does tau-chrono push the usable depth ceiling?

Circuit: Multi-qubit mirror circuit (entangling gates + their inverse).
  - Apply N layers of H + CNOT (creates real entanglement)
  - Then undo everything
  - Correct answer: always |000>
  - Deeper = more layers = more noise

This REQUIRES a quantum computer: the intermediate states are entangled.

The question: at what depth does each model say STOP?
  - Naive model: stops early (pessimistic)
  - tau-chrono: stops later (accurate)
  - Actual: the circuit might still work beyond where naive says STOP

Usage:
    python experiments/run_deep_ceiling.py --backend "Tuna-9" --shots 4096
    python experiments/run_deep_ceiling.py --backend "QX emulator" --shots 1024
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


def predict_fidelity(n_h, n_cx, n_s=0):
    """Predict fidelity using tau-chrono and naive model."""
    channels = []
    channels.extend([depolarizing(GATE_ERRORS['h'])] * n_h)
    channels.extend([depolarizing(GATE_ERRORS['cx'])] * n_cx)
    channels.extend([depolarizing(GATE_ERRORS['s'])] * n_s)
    if not channels:
        return {'f_naive': 1.0, 'f_tauchrono': 1.0, 'n_total': 0}
    result = tau_chrono_compose(channels, sigma_0=SIGMA, rho=RHO)
    return {
        'f_naive': 1 - result.tau_multiplicative_total,
        'f_tauchrono': 1 - result.tau_bayesian_total,
        'n_total': len(channels),
    }


def build_mirror_circuit(n_qubits, n_layers):
    """Build a mirror circuit: forward entangling layers + inverse.

    Each forward layer:
      - H on qubit 0
      - CNOT chain: q0→q1, q1→q2, ...
      - S on last qubit

    Then all layers undone in reverse order.
    Correct output: |000...0>

    Total gates per layer: 1 H + (n_qubits-1) CNOT + 1 S = n_qubits + 1
    Total circuit: 2 * n_layers * (n_qubits + 1) gates
    """
    qc = QuantumCircuit(n_qubits, n_qubits)

    # Forward pass: n_layers of entangling
    forward_ops = []
    for layer in range(n_layers):
        qc.h(0)
        forward_ops.append(('h', 0))

        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
            forward_ops.append(('cx', q, q + 1))

        qc.s(n_qubits - 1)
        forward_ops.append(('s', n_qubits - 1))

    # Inverse pass: undo in reverse
    for op in reversed(forward_ops):
        if op[0] == 'h':
            qc.h(op[1])
        elif op[0] == 'cx':
            qc.cx(op[1], op[2])
        elif op[0] == 's':
            # S† = S·S·S
            qc.s(op[1])
            qc.s(op[1])
            qc.s(op[1])

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', default='QX emulator')
    parser.add_argument('--shots', type=int, default=4096)
    parser.add_argument('--qubits', type=int, default=3)
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    is_hw = 'emulator' not in args.backend.lower()
    n_q = args.qubits

    print(f"Experiment B: Deep Circuit Ceiling Test")
    print(f"Backend: {args.backend} | Qubits: {n_q} | Shots: {args.shots}")
    print(f"Hardware: {is_hw} | Timestamp: {timestamp}")
    print()

    backend = get_backend(args.backend)

    # Test increasing depths
    layer_counts = [1, 2, 3, 5, 8, 12, 16, 20, 25, 30]
    results = []

    print(f"{'Layers':>6} | {'Gates':>5} | {'Actual F':>8} | {'Naive F':>8} | {'τ-chrono F':>10} | {'Naive':>6} | {'τ-chrono':>8} | {'Actual':>6}")
    print("-" * 80)

    for n_layers in layer_counts:
        qc = build_mirror_circuit(n_q, n_layers)

        # Count gates
        # Forward: n_layers * (1 H + (n_q-1) CX + 1 S)
        # Inverse: n_layers * (1 H + (n_q-1) CX + 3 S)  (S† = SSS)
        n_h = n_layers * 2  # forward H + inverse H
        n_cx = n_layers * (n_q - 1) * 2  # forward CX + inverse CX
        n_s = n_layers * (1 + 3)  # forward S + inverse S†=SSS
        n_total = n_h + n_cx + n_s

        # Predict
        pred = predict_fidelity(n_h, n_cx, n_s)

        naive_go = pred['f_naive'] >= THRESHOLD
        chrono_go = pred['f_tauchrono'] >= THRESHOLD

        # Run on hardware
        t0 = time.time()
        counts = run_circuit(backend, qc, args.shots)
        elapsed = time.time() - t0

        # Check: P(|000...0>)
        expected = '0' * n_q
        p0 = max(counts.get(expected, 0), counts.get(expected[::-1], 0)) / args.shots
        actual_ok = p0 > THRESHOLD

        naive_str = "GO" if naive_go else "STOP"
        chrono_str = "GO" if chrono_go else "STOP"
        actual_str = "OK" if actual_ok else "FAIL"

        # Flag the interesting cases
        flag = ""
        if not naive_go and actual_ok:
            flag = " ← NAIVE WRONG (said STOP, actually works!)"
        if not naive_go and chrono_go and actual_ok:
            flag = " ← τ-CHRONO SAVES THIS CIRCUIT"

        print(f"{n_layers:6d} | {n_total:5d} | {p0:8.4f} | {pred['f_naive']:8.4f} | {pred['f_tauchrono']:10.4f} | {naive_str:>6} | {chrono_str:>8} | {actual_str:>6}{flag}")

        results.append({
            'n_layers': n_layers,
            'n_qubits': n_q,
            'n_gates_total': n_total,
            'n_h': n_h, 'n_cx': n_cx, 'n_s': n_s,
            'p_correct': p0,
            'actual_ok': actual_ok,
            'f_naive': pred['f_naive'],
            'f_tauchrono': pred['f_tauchrono'],
            'naive_go': naive_go,
            'chrono_go': chrono_go,
            'counts': counts,
            'elapsed_s': round(elapsed, 1),
        })

    # Summary
    max_naive = max((r['n_layers'] for r in results if r['naive_go']), default=0)
    max_chrono = max((r['n_layers'] for r in results if r['chrono_go']), default=0)
    max_actual = max((r['n_layers'] for r in results if r['actual_ok']), default=0)

    saved_circuits = sum(1 for r in results if not r['naive_go'] and r['chrono_go'] and r['actual_ok'])

    print()
    print(f"  Max layers naive allows:      {max_naive} ({max_naive * (n_q + 3)} gates)")
    print(f"  Max layers tau-chrono allows:  {max_chrono} ({max_chrono * (n_q + 3)} gates)")
    print(f"  Max layers actually work:      {max_actual} ({max_actual * (n_q + 3)} gates)")
    print(f"  Circuits saved by tau-chrono:  {saved_circuits}")

    if max_chrono > max_naive:
        print(f"\n  τ-chrono extends usable depth by {max_chrono/max_naive:.1f}x!")

    # Save
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    suffix = 'tuna9' if is_hw else 'emulator'
    out_path = os.path.join(out_dir, f'deep_ceiling_{suffix}_{timestamp}.json')

    with open(out_path, 'w') as f:
        json.dump({
            'metadata': {
                'backend': args.backend, 'is_hardware': is_hw,
                'n_qubits': n_q, 'shots': args.shots, 'timestamp': timestamp,
            },
            'results': results,
            'summary': {
                'max_layers_naive': max_naive,
                'max_layers_tauchrono': max_chrono,
                'max_layers_actual': max_actual,
                'circuits_saved': saved_circuits,
            }
        }, f, indent=2, default=str)

    print(f"\n  Results saved: {out_path}")


if __name__ == '__main__':
    main()
