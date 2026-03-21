#!/usr/bin/env python3
"""Run the quantum diffusion LM on real T-9 hardware."""

import sys
import os
import json
import time
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from data import VOCAB, VOCAB_SIZE, TOKEN2ID, build_input_vector

ID_TO_TOKEN = VOCAB
from model import MLPDenoiser
from convert_to_quantum import build_circuit_for_distribution

# Load model
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), 'model_weights.npz')
model = MLPDenoiser()
model.load(WEIGHTS_PATH)
print(f"Loaded model ({model._count_params()} params)")

# Test cases
TEST_CASES = [
    ([0, 1, -1], 2, "the cat ___"),   # the cat [sat]
    ([0, 6, -1], 7, "the dog ___"),   # the dog [ran]
    ([-1, 1, 2], 0, "___ cat sat"),   # [the] cat sat
    ([0, -1, 2, 3, 4, 5], 1, "the ___ sat on a mat"),  # [cat]
    ([1, -1], 7, "cat ___"),          # cat [ran]
    ([6, -1], 2, "dog ___"),          # dog [sat]
    ([-1, 5], 0, "___ mat"),          # [the]
    ([0, 1, 2, 3, -1], 5, "the cat sat on ___"),  # [mat]
]


def run_on_t9(shots=4096):
    """Run quantum circuits on real T-9 hardware."""
    from qiskit import QuantumCircuit
    from qiskit.compiler import transpile
    from qiskit_quantuminspire.qi_provider import QIProvider

    provider = QIProvider()
    backend = provider.get_backend("Tuna-9")
    print(f"Backend: {backend.name}, status: {backend.status}")

    results = {
        'backend': 'Tuna-9',
        'shots': shots,
        'timestamp': datetime.now().isoformat(),
        'test_cases': [],
    }

    correct_classical = 0
    correct_quantum = 0
    correct_random = 0

    for i, (tokens, expected, desc) in enumerate(TEST_CASES):
        print(f"\n--- Test {i+1}: {desc} → expected: {ID_TO_TOKEN[expected]} ---")

        # Classical prediction
        mask_pos = tokens.index(-1)
        x = build_input_vector(tokens, mask_pos)
        logits = model.forward(x)
        probs = MLPDenoiser.softmax(logits)
        classical_pred = int(np.argmax(probs))
        classical_correct = (classical_pred == expected)
        correct_classical += classical_correct
        print(f"  Classical: {ID_TO_TOKEN[classical_pred]} ({'✅' if classical_correct else '❌'})")

        # Build quantum circuit
        qc, _ = build_circuit_for_distribution(probs, label=desc)

        # Run on T-9
        t0 = time.time()
        transpiled = transpile(qc, backend, optimization_level=0)
        job = backend.run(transpiled, shots=shots)
        job.wait_for_final_state(timeout=1800)
        counts = job.result().get_counts(0)
        elapsed = time.time() - t0

        # Parse results: 3-qubit measurement → token ID
        token_counts = {}
        for bitstring, count in counts.items():
            # Convert bitstring to token ID
            token_id = int(bitstring[::-1], 2) if len(bitstring) == 3 else int(bitstring, 2)
            token_id = token_id % VOCAB_SIZE
            token_counts[token_id] = token_counts.get(token_id, 0) + count

        # Most frequent token
        quantum_pred = max(token_counts, key=token_counts.get)
        quantum_correct = (quantum_pred == expected)
        correct_quantum += quantum_correct

        # Token distribution
        dist = {ID_TO_TOKEN.get(k, f'?{k}'): v/shots for k, v in sorted(token_counts.items())}
        print(f"  Quantum:   {ID_TO_TOKEN.get(quantum_pred, '?')} ({'✅' if quantum_correct else '❌'}) "
              f"({elapsed:.1f}s)")
        print(f"  Distribution: {dist}")

        # Random baseline
        random_pred = np.random.randint(VOCAB_SIZE)
        correct_random += (random_pred == expected)

        results['test_cases'].append({
            'description': desc,
            'expected': ID_TO_TOKEN[expected],
            'classical_pred': ID_TO_TOKEN[classical_pred],
            'quantum_pred': ID_TO_TOKEN.get(quantum_pred, f'?{quantum_pred}'),
            'classical_correct': classical_correct,
            'quantum_correct': quantum_correct,
            'quantum_distribution': dist,
            'raw_counts': {str(k): v for k, v in counts.items()},
            'elapsed_s': round(elapsed, 1),
        })

    n = len(TEST_CASES)
    print(f"\n{'='*50}")
    print(f"RESULTS (T-9 Real Hardware)")
    print(f"{'='*50}")
    print(f"  Random:    {correct_random}/{n} = {100*correct_random/n:.1f}%")
    print(f"  Quantum:   {correct_quantum}/{n} = {100*correct_quantum/n:.1f}%")
    print(f"  Classical: {correct_classical}/{n} = {100*correct_classical/n:.1f}%")

    results['summary'] = {
        'random_accuracy': correct_random / n,
        'quantum_accuracy': correct_quantum / n,
        'classical_accuracy': correct_classical / n,
    }

    # Save
    out_path = os.path.join(os.path.dirname(__file__), '..', '..', 'results',
                            f'quantum_lm_t9_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")

    return results


if __name__ == '__main__':
    run_on_t9(shots=4096)
