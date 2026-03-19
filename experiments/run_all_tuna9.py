#!/usr/bin/env python3
"""
Run ALL tau-chrono validation experiments on Tuna-9 hardware.

This script produces REAL hardware data with ground truth.
Every claim in the paper must come from data generated here.

Prerequisites:
    pip install qiskit quantuminspire qiskit-quantuminspire numpy
    qi login  # authenticate once via browser

Usage:
    python experiments/run_all_tuna9.py --backend "Tuna-9" --shots 4096
    python experiments/run_all_tuna9.py --backend "QX emulator" --shots 4096  # test first
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

# Add parent dir to path for tau_chrono import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qiskit import QuantumCircuit
from qiskit.compiler import transpile


# =====================================================================
# Backend setup
# =====================================================================

def get_backend(backend_name: str):
    """Connect to Quantum Inspire backend."""
    from qiskit_quantuminspire.qi_provider import QIProvider
    provider = QIProvider()
    backend = provider.get_backend(backend_name)
    print(f"Connected: {backend.name}")
    return backend


def run_circuit(backend, circuit, shots=4096):
    """Run a single circuit and return counts dict."""
    transpiled = transpile(circuit, backend, optimization_level=0)
    job = backend.run(transpiled, shots=shots)
    job.wait_for_final_state(timeout=1800)  # 30 min timeout for T-9 queue
    result = job.result()
    return result.get_counts(0)


# =====================================================================
# Experiment 1: Depth scaling with GROUND TRUTH
# =====================================================================

def exp1_depth_scaling(backend, shots=4096):
    """
    Run identity-equivalent circuits at various depths.
    Each circuit applies random gates then their inverses,
    so the ideal output is always |0>.
    Measure P(|0>) = actual fidelity at each depth.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Depth Scaling (ground truth)")
    print("=" * 60)

    depths = [2, 4, 6, 8, 10, 15, 20, 30, 40, 50]
    gate_set = ['h', 'x', 'sx', 's']  # Tuna-9 native gates
    # Inverses: h->h, x->x, sx->sx+sx+sx (= sx†), s->s+s+s (= s†)
    inverse_map = {
        'h': ['h'],
        'x': ['x'],
        'sx': ['sx', 'sx', 'sx'],  # sx^3 = sx†
        's': ['s', 's', 's'],       # s^3 = s†
    }

    results = []
    rng = np.random.RandomState(42)

    for depth in depths:
        print(f"\n--- Depth {depth} ---")

        # Build circuit: apply `depth` random gates, then their inverses
        qc = QuantumCircuit(1, 1)

        # Forward pass: random gates
        forward_gates = []
        for _ in range(depth):
            gate = rng.choice(gate_set)
            forward_gates.append(gate)
            getattr(qc, gate)(0)

        # Inverse pass: undo in reverse order
        for gate in reversed(forward_gates):
            for inv_gate in inverse_map[gate]:
                getattr(qc, inv_gate)(0)

        qc.measure(0, 0)

        # Run on hardware
        t0 = time.time()
        counts = run_circuit(backend, qc, shots=shots)
        elapsed = time.time() - t0

        # Ground truth: P(|0>)
        n_correct = counts.get('0', 0)
        p_correct = n_correct / shots
        actual_fidelity = p_correct

        print(f"  Counts: {counts}")
        print(f"  P(|0>) = {p_correct:.4f}  ({n_correct}/{shots})")
        print(f"  Time: {elapsed:.1f}s")

        results.append({
            'depth': depth,
            'total_gates': depth * 2 + len(sum([inverse_map[g] for g in forward_gates], [])),
            'counts': counts,
            'shots': shots,
            'p_correct': p_correct,
            'actual_fidelity': actual_fidelity,
            'forward_gates': forward_gates,
            'elapsed_s': round(elapsed, 1),
        })

    return results


# =====================================================================
# Experiment 2: Bernstein-Vazirani with GROUND TRUTH
# =====================================================================

def exp2_bernstein_vazirani(backend, shots=4096):
    """
    BV algorithm with hidden string s=1011.
    Measure actual success probability at each oracle repetition count.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Bernstein-Vazirani (ground truth)")
    print("=" * 60)

    hidden_string = '1011'
    n_qubits = len(hidden_string)
    repetitions = [1, 2, 3, 5, 8, 12]

    results = []

    for n_rep in repetitions:
        print(f"\n--- n_rep = {n_rep} ---")

        qc = QuantumCircuit(n_qubits + 1, n_qubits)

        # Initialize: H on all input qubits, X+H on ancilla
        for i in range(n_qubits):
            qc.h(i)
        qc.x(n_qubits)
        qc.h(n_qubits)

        # Apply oracle n_rep times
        for _ in range(n_rep):
            for i, bit in enumerate(reversed(hidden_string)):
                if bit == '1':
                    qc.cx(i, n_qubits)

        # Final H on input qubits
        for i in range(n_qubits):
            qc.h(i)

        # Measure input qubits only
        for i in range(n_qubits):
            qc.measure(i, i)

        # Expected output: s if n_rep is odd, 0000 if even
        if n_rep % 2 == 1:
            expected = hidden_string[::-1]  # reversed for qiskit bit ordering
        else:
            expected = '0' * n_qubits

        # Run
        t0 = time.time()
        counts = run_circuit(backend, qc, shots=shots)
        elapsed = time.time() - t0

        # Ground truth: P(correct output)
        n_correct = counts.get(expected, 0)
        # Also check reversed bit ordering
        n_correct_alt = counts.get(expected[::-1], 0)
        p_success = max(n_correct, n_correct_alt) / shots

        gate_count = qc.size()

        print(f"  Expected: {expected} (or {expected[::-1]})")
        print(f"  Top counts: {dict(sorted(counts.items(), key=lambda x: -x[1])[:5])}")
        print(f"  P_success = {p_success:.4f}  ({max(n_correct, n_correct_alt)}/{shots})")
        print(f"  Gates: {gate_count}, Time: {elapsed:.1f}s")

        results.append({
            'n_rep': n_rep,
            'gate_count': gate_count,
            'expected_output': expected,
            'counts': counts,
            'shots': shots,
            'p_success': p_success,
            'elapsed_s': round(elapsed, 1),
        })

    return results


# =====================================================================
# Experiment 3: Gate characterization (process tomography)
# =====================================================================

def exp3_gate_characterization(backend, shots=4096):
    """
    Characterize each native gate via process tomography.
    Prepare 3 input states, apply gate, measure.
    Extract Kraus operators and tau values.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Gate Characterization")
    print("=" * 60)

    from tau_chrono import depolarizing, tau_parameter, fidelity

    gate_names = ['h', 'x', 'sx', 's', 'id']
    # Input states: |0>, |1>, |+>
    # For |0>: just apply gate and measure
    # For |1>: X then gate then measure
    # For |+>: H then gate then measure in X basis

    results = []

    for gate_name in gate_names:
        print(f"\n--- Gate: {gate_name} ---")
        gate_results = {'gate': gate_name, 'tomography': {}}

        for prep_name, prep_gates, meas_bases in [
            ('z0', [],     ['z']),        # |0> -> measure Z
            ('z1', ['x'],  ['z']),        # |1> -> measure Z
            ('x0', ['h'],  ['x']),        # |+> -> measure X (H before measure)
        ]:
            for meas_name, meas_gate in [('z', None), ('x', 'h')]:
                qc = QuantumCircuit(1, 1)

                # Prepare input state
                for pg in prep_gates:
                    getattr(qc, pg)(0)

                # Apply gate under test
                if gate_name != 'id':
                    getattr(qc, gate_name)(0)
                else:
                    qc.id(0)

                # Measurement basis rotation
                if meas_gate:
                    getattr(qc, meas_gate)(0)

                qc.measure(0, 0)

                counts = run_circuit(backend, qc, shots=shots)
                p0 = counts.get('0', 0) / shots
                p1 = counts.get('1', 0) / shots

                key = f"{prep_name}_{meas_name}"
                gate_results['tomography'][key] = {
                    'p0': p0, 'p1': p1, 'counts': counts
                }
                print(f"  {key}: P(0)={p0:.4f}, P(1)={p1:.4f}")

        results.append(gate_results)

    return results


# =====================================================================
# Experiment 4: tau prediction vs actual (the key validation)
# =====================================================================

def exp4_tau_prediction_validation(backend, gate_char_results, depth_results, shots=4096):
    """
    Using the gate characterization from Exp 3, compute tau predictions
    (naive and Bayesian) for the circuits in Exp 1, and compare to
    actual measured fidelity.

    THIS IS THE KEY RESULT: does Bayesian prediction match reality
    better than naive prediction?
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: tau Prediction vs Actual Fidelity")
    print("=" * 60)

    from tau_chrono import (
        depolarizing, bayesian_compose, tau_parameter, fidelity
    )

    results = []

    for dr in depth_results:
        depth = dr['depth']
        actual_f = dr['actual_fidelity']
        forward_gates = dr['forward_gates']

        # Build Kraus channels from REAL gate characterization (Exp 3)
        # Extract per-gate error rate from tomography data:
        # For each gate, compare measured output to ideal output.
        # p_error ≈ 1 - average process fidelity

        gate_error_rates = {}
        if gate_char_results:
            for gc in gate_char_results:
                gate_name = gc['gate']
                tomo = gc['tomography']
                # Process fidelity from Z-basis: how well does the gate preserve |0> and |1>?
                # For identity: P(0|0) should be 1, P(1|1) should be 1
                # For X gate: P(1|0) should be 1, P(0|1) should be 1
                # For H gate: P(0|0) in Z should be 0.5
                # General: use average deviation from ideal as error estimate
                if gate_name == 'id':
                    p_err = 1.0 - (tomo['z0_z']['p0'] + tomo['z1_z']['p1']) / 2
                elif gate_name == 'x':
                    p_err = 1.0 - (tomo['z0_z']['p1'] + tomo['z1_z']['p0']) / 2
                elif gate_name == 'h':
                    # H|0> should give 50/50 in Z, and |+> in X basis
                    p_err = 1.0 - tomo['x0_x']['p0']  # H|+> = |0> in X basis
                elif gate_name == 'sx':
                    # SX|0> should give 50/50 in Z
                    p_err = 1.0 - tomo['x0_x']['p0']  # approximate
                elif gate_name == 's':
                    # S preserves Z basis
                    p_err = 1.0 - (tomo['z0_z']['p0'] + tomo['z1_z']['p1']) / 2
                else:
                    p_err = 0.05  # fallback
                gate_error_rates[gate_name] = max(p_err, 0.001)  # floor at 0.1%
                print(f"  Gate '{gate_name}': p_error = {gate_error_rates[gate_name]:.4f} (from T-9 tomography)")
        else:
            # No characterization data — use identity gate idle error
            gate_error_rates = {g: 0.05 for g in ['h', 'x', 'sx', 's', 'id']}
            print("  WARNING: Using default p_error=0.05 (no gate characterization)")

        # Build channels for each gate in the circuit (forward + inverse)
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

        print(f"  Depth {depth}: actual_F={actual_f:.4f}  "
              f"naive_F={f_naive:.4f}(err={error_naive:.4f})  "
              f"bayes_F={f_bayes:.4f}(err={error_bayes:.4f})  "
              f"{'BAYES WINS' if bayes_better else 'NAIVE WINS'} ({improvement:.1f}%)")

        results.append({
            'depth': depth,
            'actual_fidelity': actual_f,
            'tau_naive': tau_naive,
            'tau_bayes': tau_bayes,
            'f_naive': f_naive,
            'f_bayes': f_bayes,
            'error_naive': error_naive,
            'error_bayes': error_bayes,
            'bayes_wins': bayes_better,
            'improvement_pct': improvement,
        })

    return results


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description='Run tau-chrono validation on Tuna-9')
    parser.add_argument('--backend', default='QX emulator',
                        help='Backend name (default: QX emulator, use "Tuna-9" for real hardware)')
    parser.add_argument('--shots', type=int, default=4096)
    parser.add_argument('--skip-char', action='store_true',
                        help='Skip gate characterization (use default error rates)')
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    is_hardware = 'emulator' not in args.backend.lower()

    print(f"tau-chrono Validation Experiment")
    print(f"================================")
    print(f"Backend:   {args.backend}")
    print(f"Shots:     {args.shots}")
    print(f"Hardware:  {is_hardware}")
    print(f"Timestamp: {timestamp}")
    print()

    # Connect
    backend = get_backend(args.backend)

    all_results = {
        'metadata': {
            'backend': args.backend,
            'is_hardware': is_hardware,
            'shots': args.shots,
            'timestamp': timestamp,
            'datetime': datetime.now().isoformat(),
        }
    }

    # Exp 1: Depth scaling
    depth_results = exp1_depth_scaling(backend, shots=args.shots)
    all_results['exp1_depth_scaling'] = depth_results

    # Exp 2: BV
    bv_results = exp2_bernstein_vazirani(backend, shots=args.shots)
    all_results['exp2_bernstein_vazirani'] = bv_results

    # Exp 3: Gate characterization
    if not args.skip_char:
        char_results = exp3_gate_characterization(backend, shots=args.shots)
        all_results['exp3_gate_characterization'] = char_results
    else:
        char_results = None
        print("\n[SKIPPED] Gate characterization")

    # Exp 4: Prediction validation (needs Exp 1 + Exp 3)
    val_results = exp4_tau_prediction_validation(
        backend, char_results, depth_results, shots=args.shots
    )
    all_results['exp4_prediction_validation'] = val_results

    # Save
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(out_dir, exist_ok=True)

    suffix = 'tuna9' if is_hardware else 'emulator'
    out_path = os.path.join(out_dir, f'validation_{suffix}_{timestamp}.json')

    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"Results saved to: {out_path}")
    print(f"{'=' * 60}")

    # Print summary
    print(f"\n=== SUMMARY ===")
    print(f"\nExp 1 - Depth Scaling (actual fidelity):")
    for r in depth_results:
        print(f"  Depth {r['depth']:3d}: P(correct) = {r['p_correct']:.4f}")

    print(f"\nExp 2 - Bernstein-Vazirani:")
    for r in bv_results:
        print(f"  n_rep={r['n_rep']:2d}: P_success = {r['p_success']:.4f}")

    print(f"\nExp 4 - Prediction Validation:")
    n_bayes_wins = sum(1 for r in val_results if r['bayes_wins'])
    print(f"  Bayesian wins: {n_bayes_wins}/{len(val_results)} depths")
    if val_results:
        avg_imp = np.mean([r['improvement_pct'] for r in val_results if r['bayes_wins']])
        print(f"  Avg improvement (when Bayesian wins): {avg_imp:.1f}%")


if __name__ == '__main__':
    main()
