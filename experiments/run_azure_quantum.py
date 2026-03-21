#!/usr/bin/env python3
"""
Run tau-chrono validation on Azure Quantum (Quantinuum / Rigetti).
Uses the free Basic tier — no cost.

Usage:
    python experiments/run_azure_quantum.py
"""

import os
import sys
import json
from datetime import datetime

# Azure Quantum connection
RESOURCE_ID = "/subscriptions/af96894e-e908-4478-9c67-81bfbc846abb/resourceGroups/TauQuantum/providers/Microsoft.Quantum/Workspaces/tauQuantum"
LOCATION = "westeurope"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tau_chrono.api import predict_gates, should_enable_qec

def get_workspace():
    from azure.quantum import Workspace
    return Workspace(resource_id=RESOURCE_ID, location=LOCATION)

def list_targets(workspace):
    """List available quantum targets."""
    print("Available targets:")
    for target in workspace.get_targets():
        print(f"  {target.name} — {target.provider_id} — {target.current_availability}")
    return workspace.get_targets()

def run_on_quantinuum_emulator(workspace):
    """Run tau-chrono validation on Quantinuum H1 Emulator (FREE)."""
    from qiskit import QuantumCircuit
    from azure.quantum.qiskit import AzureQuantumProvider

    provider = AzureQuantumProvider(workspace=workspace)

    # Use Quantinuum emulator (free)
    # List available backends
    print("Available Qiskit backends:")
    for b in provider.backends():
        print(f"  {b.name}")

    # Use Quantinuum H2 Emulator (FREE, has realistic noise model)
    # NOT the syntax checker (h2-1sc) which has no noise
    backend = provider.get_backend("quantinuum.sim.h2-1e")
    print(f"\nBackend: {backend.name}")

    results = {
        'backend': backend.name,
        'timestamp': datetime.now().isoformat(),
        'experiments': []
    }

    # =========================================================
    # Experiment 1: Depth scaling (mirror circuits)
    # =========================================================
    print("\n=== Experiment 1: Depth Scaling ===")

    for n_layers in [1, 2, 3, 5, 8]:
        # Mirror circuit: H + CNOTs forward, then inverse
        n_qubits = 2
        qc = QuantumCircuit(n_qubits, n_qubits)

        # Forward
        for _ in range(n_layers):
            qc.h(0)
            qc.cx(0, 1)

        # Inverse (mirror)
        for _ in range(n_layers):
            qc.cx(0, 1)
            qc.h(0)

        qc.measure([0, 1], [0, 1])

        # tau-chrono prediction
        gate_list = ['h', 'cx'] * n_layers + ['cx', 'h'] * n_layers
        pred = predict_gates(gate_list, gate_errors={
            'h': 0.0003, 'cx': 0.003,  # Quantinuum error rates
        })

        # Run on emulator
        job = backend.run(qc, shots=100)
        result = job.result()
        counts = result.get_counts()

        # Expected output: |00>
        p_correct = counts.get('00', 0) / 100

        n_gates = n_layers * 4  # 2 forward + 2 inverse per layer

        print(f"  Layers={n_layers} ({n_gates} gates): "
              f"P(correct)={p_correct:.3f}  "
              f"tau_chrono_F={pred.f_tauchrono:.4f}  "
              f"naive_F={pred.f_naive:.4f}")

        results['experiments'].append({
            'type': 'depth_scaling',
            'n_layers': n_layers,
            'n_gates': n_gates,
            'p_correct': p_correct,
            'f_tauchrono': pred.f_tauchrono,
            'f_naive': pred.f_naive,
            'counts': counts,
        })

    # =========================================================
    # Experiment 2: QEC prediction
    # =========================================================
    print("\n=== Experiment 2: QEC Prediction ===")

    # Quantinuum error rates
    qec_result = should_enable_qec({
        'cx': 0.003, 'h': 0.0003, 'x': 0.0003,
    })
    print(f"  should_enable_qec: {qec_result.enable}")
    print(f"  Reason: {qec_result.reason}")

    results['qec_prediction'] = {
        'enable': qec_result.enable,
        'reason': qec_result.reason,
        'predicted_ler_without': qec_result.predicted_ler_without_qec,
        'predicted_ler_with': qec_result.predicted_ler_with_qec,
    }

    # =========================================================
    # Experiment 3: Bernstein-Vazirani
    # =========================================================
    print("\n=== Experiment 3: Bernstein-Vazirani (s=101) ===")

    hidden = '101'
    n_q = len(hidden)

    for n_rep in [1, 3, 5]:
        qc = QuantumCircuit(n_q + 1, n_q)

        for i in range(n_q):
            qc.h(i)
        qc.x(n_q)
        qc.h(n_q)

        for _ in range(n_rep):
            for i, bit in enumerate(reversed(hidden)):
                if bit == '1':
                    qc.cx(i, n_q)

        for i in range(n_q):
            qc.h(i)
        for i in range(n_q):
            qc.measure(i, i)

        expected = hidden[::-1] if n_rep % 2 == 1 else '0' * n_q

        job = backend.run(qc, shots=100)
        result = job.result()
        counts = result.get_counts()

        p_correct = max(
            counts.get(expected, 0),
            counts.get(expected[::-1], 0)
        ) / 100

        print(f"  n_rep={n_rep}: P(correct)={p_correct:.3f}  expected={expected}")

        results['experiments'].append({
            'type': 'bernstein_vazirani',
            'n_rep': n_rep,
            'expected': expected,
            'p_correct': p_correct,
            'counts': counts,
        })

    # Save results
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(out_dir, f'azure_quantinuum_{timestamp}.json')

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved: {out_path}")
    return results


def main():
    print("tau-chrono Azure Quantum Validation")
    print("=" * 50)

    workspace = get_workspace()
    targets = list_targets(workspace)

    print("\n--- Running on Quantinuum Emulator (FREE) ---")
    results = run_on_quantinuum_emulator(workspace)

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    depth_results = [e for e in results['experiments'] if e['type'] == 'depth_scaling']
    for r in depth_results:
        print(f"  Depth {r['n_gates']:3d}: actual={r['p_correct']:.3f}  "
              f"tau_chrono={r['f_tauchrono']:.4f}  naive={r['f_naive']:.4f}")

    print(f"\n  QEC: {results['qec_prediction']['enable']} — {results['qec_prediction']['reason']}")


if __name__ == '__main__':
    main()
