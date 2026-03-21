#!/usr/bin/env python3
"""
Run the trained variational quantum circuit on real T-9 hardware.

This script:
1. Re-trains the quantum model (deterministic seed=42) to obtain trained parameters
2. Builds Qiskit QuantumCircuits using those trained parameters
3. Submits each test case to T-9 via Quantum Inspire (4096 shots)
4. Compares: random vs classical (80p) vs quantum (simulator) vs quantum (T-9)

Architecture (from quantum_variational.py):
  - 6 qubits
  - Input encoding: R_y rotations (data-dependent)
  - 6 variational layers: R_y + R_z per qubit + CNOT ring
  - Final R_y layer
  - Measure qubits 0,1,2 -> 8 outcomes -> token prediction

T-9 native gates: ry, rz, cx (CNOT), plus standard single-qubit gates.
Qiskit transpiler handles qubit routing automatically.
"""

import sys
import os
import json
import time
import numpy as np
from datetime import datetime

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..', '..')
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', 'quantum_diffusion_lm'))

from data import SENTENCES, VOCAB, VOCAB_SIZE
from classical_tiny import encode_input_simple, make_dataset, TinyMLP, AdamOptimizer
from quantum_variational import (
    VariationalQuantumCircuit, train_quantum, evaluate_model,
    N_QUBITS
)


def get_trained_models():
    """
    Re-run training with deterministic seed to reproduce trained parameters.
    Returns (quantum_model, classical_model, train_data, test_data).
    """
    print("=" * 70)
    print("STEP 1: Reproducing trained models (deterministic seed=42)")
    print("=" * 70)

    # --- Train quantum model (exactly as in compare.py) ---
    print("\nTraining quantum model (1000 steps, seed=42)...")
    q_model, q_history, q_train, q_test = train_quantum(
        n_steps=1000, lr=0.02, seed=42, verbose=True, eval_every=200
    )
    q_final_acc = q_history['test_acc'][-1]
    print(f"Quantum simulator final test accuracy: {q_final_acc:.3f}")

    # --- Train classical model (exactly as in compare.py) ---
    print("\nTraining classical model (1000 steps, seed=42)...")
    rng = np.random.default_rng(42)
    c_model = TinyMLP(rng=rng)

    all_examples = make_dataset(SENTENCES)
    n_total = len(all_examples)
    indices = rng.permutation(n_total)
    n_train = int(0.8 * n_total)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    c_train = [all_examples[i] for i in train_idx]
    c_test = [all_examples[i] for i in test_idx]

    optimizer = AdamOptimizer(
        [c_model.W1.shape, c_model.b1.shape, c_model.b2.shape], lr=0.01
    )
    for step in range(1000):
        idx = rng.integers(len(c_train))
        x, y = c_train[idx]
        loss, dW1, db1, db2 = c_model.compute_gradients(x, y)
        optimizer.step([c_model.W1, c_model.b1, c_model.b2], [dW1, db1, db2])

    c_loss, c_acc = evaluate_model_classical(c_model, c_test)
    print(f"Classical final test accuracy: {c_acc:.3f}")

    return q_model, c_model, q_test


def evaluate_model_classical(model, data):
    """Evaluate classical model on dataset."""
    total_loss = 0.0
    correct = 0
    for x, y in data:
        probs = model.predict_probs(x)
        total_loss += -np.log(probs[y] + 1e-10)
        if np.argmax(probs) == y:
            correct += 1
    return total_loss / len(data), correct / len(data)


def build_qiskit_circuit(q_model, input_x):
    """
    Build a Qiskit QuantumCircuit that implements the trained VQC
    for a given input vector.

    Architecture:
      1. Input encoding: R_y(angle) on each of 6 qubits
      2. 6 variational layers:
         - R_y(theta) + R_z(phi) on each qubit
         - CNOT ring: 0->1, 1->2, 2->3, 3->4, 4->5, 5->0
      3. Final R_y layer on each qubit
      4. Measure qubits 0, 1, 2
    """
    from qiskit import QuantumCircuit

    n_qubits = q_model.n_qubits  # 6
    qc = QuantumCircuit(n_qubits, 3)  # 6 qubits, 3 classical bits

    # --- Input encoding ---
    input_angles = q_model.encode_input(input_x)
    for q in range(n_qubits):
        qc.ry(float(input_angles[q]), q)

    qc.barrier()

    # --- Variational layers ---
    for layer in range(q_model.n_layers):
        # R_y and R_z on each qubit
        for q in range(n_qubits):
            ry_angle = float(q_model.layer_params[layer, q, 0])
            rz_angle = float(q_model.layer_params[layer, q, 1])
            qc.ry(ry_angle, q)
            qc.rz(rz_angle, q)

        # CNOT ring: 0->1, 1->2, 2->3, 3->4, 4->5, 5->0
        for q in range(n_qubits):
            control = q
            target = (q + 1) % n_qubits
            qc.cx(control, target)

        qc.barrier()

    # --- Final R_y layer ---
    for q in range(n_qubits):
        qc.ry(float(q_model.final_ry[q]), q)

    # --- Measure output qubits 0, 1, 2 ---
    qc.measure(0, 0)
    qc.measure(1, 1)
    qc.measure(2, 2)

    return qc


def run_on_t9(q_model, c_model, test_data, shots=4096):
    """
    Run the trained quantum circuit on T-9 for all test cases.
    Also run classical model and simulator for comparison.
    """
    from qiskit.compiler import transpile
    from qiskit_quantuminspire.qi_provider import QIProvider

    print("\n" + "=" * 70)
    print("STEP 2: Running on T-9 hardware")
    print("=" * 70)

    provider = QIProvider()
    backend = provider.get_backend("Tuna-9")
    print(f"Backend: {backend.name}, {backend.num_qubits} qubits")

    n_test = len(test_data)
    print(f"Test cases: {n_test}")
    print(f"Shots per circuit: {shots}")
    print(f"Estimated time: {n_test * 30}-{n_test * 90} seconds")
    print()

    # Track results
    results_list = []
    correct_random = 0
    correct_classical = 0
    correct_sim = 0
    correct_t9 = 0
    total_t9_time = 0.0

    rng = np.random.default_rng(12345)  # for random baseline

    for i, (x, y_true) in enumerate(test_data):
        print(f"--- Test {i+1}/{n_test} (target token: {VOCAB[y_true]}) ---")

        # 1. Random baseline
        random_pred = rng.integers(VOCAB_SIZE)
        random_correct = (random_pred == y_true)
        correct_random += random_correct

        # 2. Classical prediction
        c_probs = c_model.predict_probs(x)
        c_pred = int(np.argmax(c_probs))
        c_correct = (c_pred == y_true)
        correct_classical += c_correct

        # 3. Quantum simulator prediction
        sim_probs = q_model.predict_probs(x)
        sim_pred = int(np.argmax(sim_probs))
        sim_correct = (sim_pred == y_true)
        correct_sim += sim_correct

        # 4. Quantum T-9 prediction
        qc = build_qiskit_circuit(q_model, x)

        t0 = time.time()
        # Transpile for T-9 (optimization_level=1 for reasonable depth)
        transpiled = transpile(qc, backend, optimization_level=1)
        depth = transpiled.depth()

        # Submit to T-9
        job = backend.run(transpiled, shots=shots)
        job.wait_for_final_state(timeout=1800)
        result = job.result()
        counts = result.get_counts(0)
        elapsed = time.time() - t0
        total_t9_time += elapsed

        # Parse measurement results: 3-bit string -> token ID
        # Qiskit returns bitstrings in reverse order (q2 q1 q0)
        # but we need to map measured bits to token IDs
        token_counts = {}
        total_shots_received = 0
        for bitstring, count in counts.items():
            # Qiskit bitstring is in little-endian: rightmost bit = q0
            # Take only the last 3 characters (measured bits)
            bits = bitstring.strip()
            # Handle possible extra bits from unused qubits
            # We measured qubits 0,1,2 into classical bits 0,1,2
            # Qiskit returns: c[2] c[1] c[0] (MSB first)
            if len(bits) >= 3:
                bits = bits[-3:]  # take last 3 bits
            token_id = int(bits, 2)
            token_id = token_id % VOCAB_SIZE
            token_counts[token_id] = token_counts.get(token_id, 0) + count
            total_shots_received += count

        # Most frequent token = T-9 prediction
        t9_pred = max(token_counts, key=token_counts.get)
        t9_correct = (t9_pred == y_true)
        correct_t9 += t9_correct

        # Token distribution from T-9
        t9_dist = {VOCAB[k]: round(v / total_shots_received, 4)
                   for k, v in sorted(token_counts.items())}

        # Simulator distribution for comparison
        sim_dist = {VOCAB[k]: round(float(sim_probs[k]), 4)
                    for k in range(VOCAB_SIZE)}

        status_c = "OK" if c_correct else "MISS"
        status_s = "OK" if sim_correct else "MISS"
        status_t = "OK" if t9_correct else "MISS"

        print(f"  Classical: {VOCAB[c_pred]:>4s} [{status_c}]  "
              f"Sim: {VOCAB[sim_pred]:>4s} [{status_s}]  "
              f"T-9: {VOCAB[t9_pred]:>4s} [{status_t}]  "
              f"(depth={depth}, {elapsed:.1f}s)")
        print(f"  T-9 distribution: {t9_dist}")

        results_list.append({
            'test_index': i,
            'target_token': VOCAB[y_true],
            'target_id': int(y_true),
            'classical_pred': VOCAB[c_pred],
            'classical_correct': c_correct,
            'simulator_pred': VOCAB[sim_pred],
            'simulator_correct': sim_correct,
            'simulator_probs': sim_dist,
            't9_pred': VOCAB[t9_pred],
            't9_correct': t9_correct,
            't9_distribution': t9_dist,
            'raw_counts': {str(k): int(v) for k, v in counts.items()},
            'circuit_depth': depth,
            'elapsed_s': round(elapsed, 1),
        })

    # --- Summary ---
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'':>22s} {'Correct':>8s} {'Accuracy':>10s}")
    print("-" * 45)
    print(f"{'Random baseline':>22s} {correct_random:>5d}/{n_test:<3d} "
          f"{100*correct_random/n_test:>8.1f}%")
    print(f"{'Classical (80p)':>22s} {correct_classical:>5d}/{n_test:<3d} "
          f"{100*correct_classical/n_test:>8.1f}%")
    print(f"{'Quantum (simulator)':>22s} {correct_sim:>5d}/{n_test:<3d} "
          f"{100*correct_sim/n_test:>8.1f}%")
    print(f"{'Quantum (T-9)':>22s} {correct_t9:>5d}/{n_test:<3d} "
          f"{100*correct_t9/n_test:>8.1f}%   <-- KEY NUMBER")
    print("-" * 45)

    # Interpret
    print()
    t9_acc = correct_t9 / n_test
    sim_acc = correct_sim / n_test
    c_acc = correct_classical / n_test
    random_acc = 1.0 / VOCAB_SIZE

    if t9_acc > c_acc:
        verdict = "QUANTUM ADVANTAGE SURVIVES NOISE: T-9 quantum > classical"
        print(f"VERDICT: {verdict}")
        print(f"  The quantum expressiveness advantage persists on real hardware!")
    elif t9_acc > random_acc + 0.01:
        verdict = "PARTIAL SURVIVAL: T-9 quantum > random but < classical"
        print(f"VERDICT: {verdict}")
        print(f"  Noise hurts but quantum model still learned something.")
    else:
        verdict = "NOISE DESTROYS ADVANTAGE: T-9 quantum ~ random"
        print(f"VERDICT: {verdict}")
        print(f"  Hardware noise washes out the quantum advantage.")

    print(f"\nTotal T-9 execution time: {total_t9_time:.1f}s "
          f"(avg {total_t9_time/n_test:.1f}s per circuit)")

    return {
        'random': {'correct': int(correct_random), 'total': n_test,
                   'accuracy': correct_random / n_test},
        'classical': {'correct': int(correct_classical), 'total': n_test,
                      'accuracy': correct_classical / n_test},
        'quantum_simulator': {'correct': int(correct_sim), 'total': n_test,
                              'accuracy': correct_sim / n_test},
        'quantum_t9': {'correct': int(correct_t9), 'total': n_test,
                       'accuracy': correct_t9 / n_test},
        'verdict': verdict,
        'test_cases': results_list,
        'total_t9_time_s': round(total_t9_time, 1),
    }


def main():
    """Main entry point."""
    print("=" * 70)
    print("QUANTUM EXPRESSIVENESS ON T-9 HARDWARE")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)
    print()

    # Step 1: Get trained models
    q_model, c_model, test_data = get_trained_models()

    # Verify simulator accuracy matches previous results
    sim_loss, sim_acc = evaluate_model(q_model, test_data)
    print(f"\nVerification - simulator test accuracy: {sim_acc:.3f}")
    print(f"  (expected ~0.231 from results.json)")
    print(f"  Test cases: {len(test_data)}")

    # Show trained parameter summary
    params = q_model.get_params_flat()
    print(f"\nTrained parameters: {len(params)} values")
    print(f"  Range: [{params.min():.4f}, {params.max():.4f}]")
    print(f"  Mean:  {params.mean():.4f}")
    print(f"  Std:   {params.std():.4f}")

    # Build one example circuit to show its structure
    from qiskit import QuantumCircuit
    example_x, example_y = test_data[0]
    example_qc = build_qiskit_circuit(q_model, example_x)
    print(f"\nExample circuit:")
    print(f"  Qubits: {example_qc.num_qubits}")
    print(f"  Depth: {example_qc.depth()}")
    print(f"  Gate count: {dict(example_qc.count_ops())}")

    # Step 2: Run on T-9
    results = run_on_t9(q_model, c_model, test_data, shots=4096)

    # Step 3: Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(PROJECT_ROOT, 'results',
                            f'quantum_expressiveness_t9_{timestamp}.json')

    output = {
        'experiment': 'quantum_expressiveness_t9',
        'description': ('Variational quantum circuit (80 params, 6 qubits, '
                        'trained via simulator) run on T-9 real hardware '
                        'for masked token prediction'),
        'backend': 'Tuna-9',
        'shots': 4096,
        'timestamp': datetime.now().isoformat(),
        'n_qubits': N_QUBITS,
        'n_variational_layers': 6,
        'n_parameters': 80,
        'n_test_cases': len(test_data),
        'training_config': {
            'n_steps': 1000,
            'lr_quantum': 0.02,
            'lr_classical': 0.01,
            'seed': 42,
            'optimizer': 'Adam',
            'gradient_method': 'parameter_shift_rule',
        },
        'summary': {
            'random_baseline_acc': results['random']['accuracy'],
            'classical_80p_acc': results['classical']['accuracy'],
            'quantum_simulator_acc': results['quantum_simulator']['accuracy'],
            'quantum_t9_acc': results['quantum_t9']['accuracy'],
        },
        'verdict': results['verdict'],
        'test_cases': results['test_cases'],
        'total_t9_time_s': results['total_t9_time_s'],
    }

    # Also save a compact table
    print("\n" + "=" * 70)
    print("FINAL TABLE")
    print("=" * 70)
    print(f"{'':>22s} {'Test Accuracy':>14s}")
    print("-" * 40)
    print(f"{'Random':>22s} {100*results['random']['accuracy']:>12.1f}%")
    print(f"{'Classical (80p)':>22s} {100*results['classical']['accuracy']:>12.1f}%")
    print(f"{'Quantum (simulator)':>22s} {100*results['quantum_simulator']['accuracy']:>12.1f}%")
    print(f"{'Quantum (T-9)':>22s} {100*results['quantum_t9']['accuracy']:>12.1f}%")
    print("-" * 40)
    print(f"Verdict: {results['verdict']}")

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")

    # Also save to the experiment directory
    local_path = os.path.join(SCRIPT_DIR, 'results_t9.json')
    with open(local_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Also saved: {local_path}")

    return output


if __name__ == '__main__':
    main()
