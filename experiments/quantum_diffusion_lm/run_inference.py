#!/usr/bin/env python3
"""
Run inference using all three methods and compare results.

Methods:
  A) Classical MLP (trained model, direct numpy forward pass)
  B) Quantum simulation (Qiskit Aer simulator, local)
  C) Random baseline (uniform random over 8 tokens)

Also includes tau-chrono fidelity prediction for T-9 hardware.

Usage: python run_inference.py
"""

import sys
import os
import time
import numpy as np

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import (SENTENCES, VOCAB, VOCAB_SIZE, NUM_BITS, INPUT_DIM,
                  build_input_vector, decode_sentence, bits_to_token,
                  make_masked_examples)
from model import MLPDenoiser

try:
    from qiskit import QuantumCircuit
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False

try:
    from qiskit_aer import AerSimulator
    HAS_AER = True
except ImportError:
    HAS_AER = False

from convert_to_quantum import (convert_model, build_circuit_for_token,
                                extract_quantum_params, build_circuit_for_distribution,
                                _simulate_3qubit_circuit)


# ─────────────────────────────────────────────
# Test cases: (sentence, mask_position) pairs
# ─────────────────────────────────────────────
# Mix of unambiguous cases (clear answer from context)
# and ambiguous cases (multiple valid answers).
TEST_CASES = [
    # UNAMBIGUOUS: strong contextual signal
    ([0, 1, 2, 3, 0, 5], 0, "___ cat sat on the mat"),    # must be 'the'
    ([0, 1, 2, 3, 0, 5], 1, "the ___ sat on the mat"),    # cat (sat selects cat)
    ([0, 6, 7, 3, 0, 5], 1, "the ___ ran on the mat"),    # dog (ran selects dog)
    ([0, 1, 2, 3, 0, 5], 3, "the cat sat ___ the mat"),   # on
    ([0, 1, 2, 3, 0, 5], 5, "the cat sat on the ___"),    # mat
    ([0, 6, 7, 3, 0, 5], 2, "the dog ___ on the mat"),    # ran
    ([0, 6, 7, 3, 0, 5], 3, "the dog ran ___ the mat"),   # on
    ([1, 2, 3, 4, 5],     1, "cat ___ on a mat"),          # sat
    ([1, 2, 3, 4, 5],     2, "cat sat ___ a mat"),         # on
    ([6, 7, 3, 0, 5],     0, "___ ran on the mat"),        # dog
    ([6, 7, 3, 0, 5],     1, "dog ___ on the mat"),        # ran
    ([0, 1, 7],            0, "___ cat ran"),               # the
    ([0, 6, 2],            0, "___ dog sat"),               # the
    # AMBIGUOUS: equally valid answers (cat/dog, sat/ran, the/a)
    ([0, 1, 2],            1, "the ___ sat"),               # cat (ambig w/ dog)
    ([0, 6, 7],            2, "the dog ___"),               # ran (ambig w/ sat)
    ([1, 7],               0, "___ ran"),                   # cat or dog
]


def run_classical(model, test_cases):
    """Run classical MLP inference on test cases."""
    results = []
    total_time = 0

    for sent, mask_pos, desc in test_cases:
        inp = build_input_vector(sent, mask_pos)
        t0 = time.perf_counter()
        probs = model.forward(inp)
        t1 = time.perf_counter()
        pred = int(np.argmax(probs))
        actual = sent[mask_pos]
        total_time += (t1 - t0)
        results.append({
            'desc': desc,
            'pred': pred,
            'actual': actual,
            'correct': pred == actual,
            'probs': probs.copy(),
            'confidence': float(probs[pred]),
        })

    return results, total_time


def build_quantum_circuits(model, test_cases, verbose=True):
    """
    Build quantum circuits for each test case (expensive, do once).

    Returns list of (QuantumCircuit, theta, actual_token) tuples.
    """
    if not HAS_QISKIT:
        return None

    circuits = []
    if verbose:
        print("  Building per-test-case quantum circuits...")

    for idx, (sent, mask_pos, desc) in enumerate(test_cases):
        inp = build_input_vector(sent, mask_pos)
        classical_probs = model.forward(inp).astype(np.float64)
        qc, theta = build_circuit_for_distribution(classical_probs, n_restarts=15)

        # Verify circuit quality
        sim_probs = _simulate_3qubit_circuit(theta)
        ideal_pred = int(np.argmax(sim_probs))
        classical_pred = int(np.argmax(classical_probs))
        match = "OK" if ideal_pred == classical_pred else "XX"

        if verbose and idx < 4:
            print(f"    [{match}] {desc}: classical={VOCAB[classical_pred]}, "
                  f"q-ideal={VOCAB[ideal_pred]}")

        circuits.append((qc, theta, sent[mask_pos]))

    return circuits


def run_quantum_sim_from_circuits(circuits, test_cases, shots=4096, verbose=True):
    """
    Run pre-built quantum circuits on the Aer simulator.

    This is fast since circuits are already built.
    """
    if not HAS_AER or circuits is None:
        if verbose:
            print("  [SKIP] Qiskit/Aer not available")
        return None, 0

    simulator = AerSimulator()
    results = []
    total_time = 0

    for idx, ((qc, theta, actual), (sent, mask_pos, desc)) in enumerate(
            zip(circuits, test_cases)):

        t0 = time.perf_counter()
        job = simulator.run(qc, shots=shots)
        counts = job.result().get_counts()
        t1 = time.perf_counter()
        total_time += (t1 - t0)

        # Find most frequent measurement outcome
        best_bitstr = max(counts, key=counts.get)
        bits = [int(b) for b in best_bitstr][::-1]
        pred = bits_to_token(bits)

        # Compute probability distribution from counts
        probs = np.zeros(VOCAB_SIZE)
        for bitstr, count in counts.items():
            b = [int(x) for x in bitstr][::-1]
            tok = bits_to_token(b)
            probs[tok] += count
        probs /= shots

        results.append({
            'desc': desc,
            'pred': pred,
            'actual': actual,
            'correct': pred == actual,
            'probs': probs,
            'confidence': float(probs[pred]),
            'counts': counts,
        })

    return results, total_time


def run_random(test_cases, seed=123):
    """Random baseline."""
    rng = np.random.default_rng(seed)
    results = []
    for sent, mask_pos, desc in test_cases:
        pred = rng.integers(0, VOCAB_SIZE)
        actual = sent[mask_pos]
        results.append({
            'desc': desc,
            'pred': pred,
            'actual': actual,
            'correct': pred == actual,
            'probs': np.ones(VOCAB_SIZE) / VOCAB_SIZE,
            'confidence': 1.0 / VOCAB_SIZE,
        })
    return results, 0.0


def predict_t9_accuracy(model, test_cases):
    """
    Use tau-chrono framework to predict T-9 hardware accuracy.

    Key parameters for T-9 (Starmon-5 class):
    - Single-qubit gate error: ~0.1% (1e-3)
    - Two-qubit gate error: ~1% (1e-2)
    - Readout error: ~2% (2e-2)
    - T1 ~ 15 us, T2 ~ 20 us
    - Gate time: ~20 ns (single), ~40 ns (two-qubit)

    tau-chrono prediction:
      tau = 1 - F (process infidelity)
      F = prod(1 - e_gate) for all gates in circuit
    """
    # T-9 estimated error rates
    e_1q = 1e-3   # single-qubit gate error
    e_2q = 1e-2   # two-qubit gate error
    e_ro = 2e-2   # readout error per qubit

    # Count gates from a sample circuit (all circuits have same structure)
    sample_probs = np.ones(VOCAB_SIZE) / VOCAB_SIZE
    sample_qc, _ = build_circuit_for_distribution(sample_probs, n_restarts=1)
    ops = sample_qc.count_ops()

    n_1q = sum(v for k, v in ops.items()
               if k in ['ry', 'rz', 'x', 'h', 's', 't', 'rx'])
    n_2q = sum(v for k, v in ops.items() if k in ['cx', 'cnot', 'cz'])
    n_measure = ops.get('measure', 0)

    print(f"  Circuit gate count: {n_1q} single-qubit, {n_2q} two-qubit, "
          f"{n_measure} measurements")

    # Process fidelity
    F_gates = (1 - e_1q) ** n_1q * (1 - e_2q) ** n_2q
    F_readout = (1 - e_ro) ** n_measure
    F_total = F_gates * F_readout

    tau = 1 - F_total

    # Predicted accuracy:
    # With fidelity F, the correct outcome has probability
    # p_correct = F * p_ideal + (1-F) * (1/8)
    # where p_ideal is the ideal circuit's success probability

    # Estimate ideal circuit accuracy: the quantum circuit encodes the
    # classical model's distribution, so its ideal accuracy should match
    # the classical model's accuracy
    n_correct_ideal = 0
    for sent, mask_pos, desc in test_cases:
        inp = build_input_vector(sent, mask_pos)
        classical_probs = model.forward(inp).astype(np.float64)
        pred_ideal = int(np.argmax(classical_probs))
        actual = sent[mask_pos]
        if pred_ideal == actual:
            n_correct_ideal += 1

    p_ideal_acc = n_correct_ideal / len(test_cases)
    print(f"  Ideal quantum circuit accuracy: {p_ideal_acc:.1%} "
          f"({n_correct_ideal}/{len(test_cases)})")
    print(f"  (matches classical accuracy since circuit encodes MLP distribution)")

    # With hardware noise, correct outcome probability degrades as:
    # p_correct = F * p_ideal + (1-F) * (1/8)
    p_t9 = F_total * p_ideal_acc + (1 - F_total) * (1.0 / VOCAB_SIZE)

    return {
        'F_gates': F_gates,
        'F_readout': F_readout,
        'F_total': F_total,
        'tau': tau,
        'n_1q_gates': n_1q,
        'n_2q_gates': n_2q,
        'predicted_accuracy': p_t9,
        'e_1q': e_1q,
        'e_2q': e_2q,
        'e_ro': e_ro,
    }


def print_results_table(classical, quantum_sim, random_res,
                        t_classical, t_quantum, tau_pred):
    """Print a nice comparison table."""
    n = len(TEST_CASES)

    acc_classical = sum(1 for r in classical if r['correct']) / n
    acc_random = sum(1 for r in random_res if r['correct']) / n
    acc_quantum = (sum(1 for r in quantum_sim if r['correct']) / n
                   if quantum_sim else None)

    print()
    print("=" * 72)
    print("RESULTS COMPARISON")
    print("=" * 72)
    print()

    # Detailed results
    header = f"{'Test Case':<30s} | {'Actual':>6s} | {'Class.':>6s} | "
    if quantum_sim:
        header += f"{'Q-Sim':>6s} | "
    header += f"{'Random':>6s}"
    print(header)
    print("-" * len(header))

    for i in range(n):
        actual = VOCAB[classical[i]['actual']]
        c_pred = VOCAB[classical[i]['pred']]
        r_pred = VOCAB[random_res[i]['pred']]
        c_mark = " OK" if classical[i]['correct'] else " XX"
        r_mark = " OK" if random_res[i]['correct'] else " XX"

        row = f"{TEST_CASES[i][2]:<30s} | {actual:>6s} | {c_pred:>3s}{c_mark} | "
        if quantum_sim:
            q_pred = VOCAB[quantum_sim[i]['pred']]
            q_mark = " OK" if quantum_sim[i]['correct'] else " XX"
            row += f"{q_pred:>3s}{q_mark} | "
        row += f"{r_pred:>3s}{r_mark}"
        print(row)

    print()

    # Summary table
    print("=" * 52)
    print(f"{'Method':<20s} | {'Accuracy':>10s} | {'Time':>10s}")
    print("-" * 52)
    print(f"{'Random guess':<20s} | {acc_random:>9.1%} | {'~0s':>10s}")
    print(f"{'Classical MLP':<20s} | {acc_classical:>9.1%} | {t_classical*1000:>7.1f} ms")
    if quantum_sim:
        print(f"{'Quantum (sim)':<20s} | {acc_quantum:>9.1%} | {t_quantum*1000:>7.1f} ms")
    if tau_pred:
        print(f"{'Quantum (T-9 pred.)':<20s} | "
              f"{tau_pred['predicted_accuracy']:>9.1%} | {'~30 s':>10s}")
    print("=" * 52)

    # tau-chrono predictions
    if tau_pred:
        print()
        print("tau-chrono PREDICTION for T-9 hardware:")
        print(f"  Process fidelity (gates):  F_gates   = {tau_pred['F_gates']:.4f}")
        print(f"  Readout fidelity:          F_readout = {tau_pred['F_readout']:.4f}")
        print(f"  Total fidelity:            F_total   = {tau_pred['F_total']:.4f}")
        print(f"  Process infidelity:        tau       = {tau_pred['tau']:.4f}")
        print(f"  Predicted T-9 accuracy:              = {tau_pred['predicted_accuracy']:.1%}")
        print()
        print(f"  Circuit: {tau_pred['n_1q_gates']} single-qubit gates, "
              f"{tau_pred['n_2q_gates']} two-qubit gates")
        print(f"  Error rates: e_1q={tau_pred['e_1q']:.1e}, "
              f"e_2q={tau_pred['e_2q']:.1e}, e_ro={tau_pred['e_ro']:.1e}")


def print_quantum_distribution(quantum_sim):
    """Print the quantum measurement distribution for each test case."""
    if not quantum_sim:
        return

    print()
    print("=" * 60)
    print("QUANTUM MEASUREMENT DISTRIBUTIONS")
    print("=" * 60)
    for i, r in enumerate(quantum_sim):
        desc = TEST_CASES[i][2]
        print(f"\n  {desc}")
        print(f"  Target: {VOCAB[r['actual']]}")
        # Sort by probability
        sorted_toks = np.argsort(-r['probs'])
        for tok in sorted_toks[:4]:
            bar = '#' * int(r['probs'][tok] * 40)
            print(f"    {VOCAB[tok]:>3s}: {r['probs'][tok]:.3f} {bar}")


def main():
    print("=" * 60)
    print("Quantum Diffusion LM - Inference Comparison")
    print("=" * 60)

    # Load trained model
    model_path = os.path.join(os.path.dirname(__file__), 'model_weights.npz')
    if not os.path.exists(model_path):
        print(f"\nERROR: Trained model not found at {model_path}")
        print("Run train_classical.py first!")
        sys.exit(1)

    model = MLPDenoiser()
    model.load(model_path)
    print(f"\nLoaded model from {model_path}")
    print(f"  Parameters: {model.num_params}")

    # Evaluate full training set accuracy
    all_examples = make_masked_examples(SENTENCES)
    all_inputs = np.array([e[0] for e in all_examples])
    all_targets = np.array([e[1] for e in all_examples])
    all_preds = np.argmax(model.forward(all_inputs), axis=-1)
    train_acc = np.mean(all_preds == all_targets)
    print(f"  Training set accuracy: {train_acc:.1%}")

    n = len(TEST_CASES)
    print(f"\nRunning {n} test cases...\n")

    # A) Classical
    print("--- Classical MLP ---")
    classical_results, t_classical = run_classical(model, TEST_CASES)
    n_correct = sum(1 for r in classical_results if r['correct'])
    print(f"  Accuracy: {n_correct}/{n} = {n_correct/n:.1%}")

    # B) Quantum simulation
    print("\n--- Quantum Simulation ---")
    q_circuits = build_quantum_circuits(model, TEST_CASES)
    quantum_results, t_quantum = run_quantum_sim_from_circuits(
        q_circuits, TEST_CASES, shots=4096)
    if quantum_results:
        n_correct_q = sum(1 for r in quantum_results if r['correct'])
        print(f"  Accuracy: {n_correct_q}/{n} = {n_correct_q/n:.1%}")

    # C) Random baseline
    print("\n--- Random Baseline ---")
    random_results, t_random = run_random(TEST_CASES)
    n_correct_r = sum(1 for r in random_results if r['correct'])
    print(f"  Accuracy: {n_correct_r}/{n} = {n_correct_r/n:.1%}")

    # D) tau-chrono prediction
    print("\n--- tau-chrono T-9 Prediction ---")
    tau_pred = predict_t9_accuracy(model, TEST_CASES) if HAS_QISKIT else None

    # Print comparison
    print_results_table(classical_results, quantum_results, random_results,
                        t_classical, t_quantum, tau_pred)

    # Print quantum distributions
    print_quantum_distribution(quantum_results)

    # Shot count scaling analysis
    if HAS_QISKIT and HAS_AER and q_circuits is not None:
        print()
        print("=" * 60)
        print("SHOT COUNT SCALING ANALYSIS")
        print("=" * 60)
        print("  (Higher shot count reduces sampling noise)")
        print("  (Averaged over 10 runs per shot count)")
        print()
        for shots in [128, 512, 2048, 8192, 32768]:
            accs = []
            for trial in range(10):
                q_res, _ = run_quantum_sim_from_circuits(
                    q_circuits, TEST_CASES, shots=shots, verbose=False)
                if q_res:
                    acc = sum(1 for r in q_res if r['correct']) / n
                    accs.append(acc)
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            bar = '#' * int(mean_acc * 50)
            print(f"  {shots:>6d} shots: {mean_acc:>5.1%} +/- {std_acc:>4.1%}  {bar}")
        c_acc = sum(1 for r in classical_results if r['correct']) / n
        print(f"  Classical:    {c_acc:>5.1%}          "
              f"{'#' * int(c_acc * 50)}")
        print()

    print("Done.")


if __name__ == '__main__':
    main()
