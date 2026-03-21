#!/usr/bin/env python3
"""
Run tau-chrono validation on IonQ simulator via Azure Quantum (FREE).

Experiments:
  1. Depth scaling with mirror circuits (depths 2,4,6,8,10,15,20)
  2. Bernstein-Vazirani (hidden string 101, n_rep 1,3,5)
  3. QEC recommendation with IonQ error rates

Usage:
    python experiments/run_ionq_simulator.py
"""

import os
import sys
import json
from datetime import datetime

# Azure Quantum connection
RESOURCE_ID = (
    "/subscriptions/af96894e-e908-4478-9c67-81bfbc846abb"
    "/resourceGroups/TauQuantum"
    "/providers/Microsoft.Quantum/Workspaces/tauQuantum"
)
LOCATION = "westeurope"

# IonQ error rates (typical published values)
IONQ_ERRORS = {
    "h": 0.0003,
    "x": 0.0003,
    "cx": 0.005,
    "measure": 0.003,
}

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tau_chrono.api import predict_gates, should_enable_qec


def get_workspace():
    from azure.quantum import Workspace
    return Workspace(resource_id=RESOURCE_ID, location=LOCATION)


def list_targets(workspace):
    """List available quantum targets."""
    print("Available targets:")
    for target in workspace.get_targets():
        print(f"  {target.name} -- {target.provider_id} -- {target.current_availability}")
    return workspace.get_targets()


def run_on_ionq_simulator(workspace):
    """Run tau-chrono validation on IonQ simulator (FREE)."""
    from qiskit import QuantumCircuit
    from azure.quantum.qiskit import AzureQuantumProvider

    provider = AzureQuantumProvider(workspace=workspace)

    # List available backends
    print("Available Qiskit backends:")
    for b in provider.backends():
        print(f"  {b.name}")

    backend = provider.get_backend("ionq.simulator")
    print(f"\nBackend: {backend.name}")

    results = {
        "backend": backend.name,
        "timestamp": datetime.now().isoformat(),
        "ionq_error_rates": IONQ_ERRORS,
        "experiments": [],
    }

    # =================================================================
    # Experiment 1: Depth Scaling (mirror circuits)
    # =================================================================
    print("\n" + "=" * 60)
    print("Experiment 1: Depth Scaling (mirror circuits)")
    print("=" * 60)

    depth_list = [2, 4, 6, 8, 10, 15, 20]
    shots = 100

    for depth in depth_list:
        n_layers = depth // 2  # each layer = H + CX forward, CX + H inverse
        n_qubits = 2
        qc = QuantumCircuit(n_qubits, n_qubits)

        # Forward: H then CX, repeated n_layers times
        for _ in range(n_layers):
            qc.h(0)
            qc.cx(0, 1)

        # Mirror (inverse): CX then H, repeated n_layers times
        for _ in range(n_layers):
            qc.cx(0, 1)
            qc.h(0)

        qc.measure([0, 1], [0, 1])

        total_gates = n_layers * 4  # 2 forward + 2 inverse per layer

        # tau-chrono prediction
        gate_list = ["h", "cx"] * n_layers + ["cx", "h"] * n_layers
        pred = predict_gates(gate_list, gate_errors=IONQ_ERRORS)

        # Run on IonQ simulator
        print(f"  Submitting depth={depth} ({total_gates} gates) ...")
        job = backend.run(qc, shots=shots)
        result = job.result()
        counts = result.get_counts()

        # Mirror circuit should return |00>
        p_correct = counts.get("00", 0) / shots

        print(
            f"  Depth={depth:3d} ({total_gates:3d} gates): "
            f"P(correct)={p_correct:.3f}  "
            f"tau_chrono_F={pred.f_tauchrono:.4f}  "
            f"naive_F={pred.f_naive:.4f}"
        )

        results["experiments"].append({
            "type": "depth_scaling",
            "depth": depth,
            "n_layers": n_layers,
            "n_gates": total_gates,
            "p_correct": p_correct,
            "f_tauchrono": pred.f_tauchrono,
            "f_naive": pred.f_naive,
            "improvement_pct": pred.improvement_pct,
            "should_run": pred.should_run,
            "counts": counts,
        })

    # =================================================================
    # Experiment 2: Bernstein-Vazirani (hidden string = 101)
    # =================================================================
    print("\n" + "=" * 60)
    print("Experiment 2: Bernstein-Vazirani (s=101)")
    print("=" * 60)

    hidden = "101"
    n_q = len(hidden)

    for n_rep in [1, 3, 5]:
        qc = QuantumCircuit(n_q + 1, n_q)

        # Hadamard on query qubits
        for i in range(n_q):
            qc.h(i)

        # Ancilla in |-> state
        qc.x(n_q)
        qc.h(n_q)

        # Oracle repeated n_rep times
        for _ in range(n_rep):
            for i, bit in enumerate(reversed(hidden)):
                if bit == "1":
                    qc.cx(i, n_q)

        # Final Hadamard on query qubits
        for i in range(n_q):
            qc.h(i)

        # Measure query qubits only
        for i in range(n_q):
            qc.measure(i, i)

        # Expected: for odd n_rep -> hidden string; for even n_rep -> 000
        expected = hidden[::-1] if n_rep % 2 == 1 else "0" * n_q

        # tau-chrono prediction for the circuit
        bv_gates = (
            ["h"] * n_q          # initial H
            + ["x", "h"]         # ancilla prep
            + ["cx"] * (hidden.count("1") * n_rep)  # oracle CX gates
            + ["h"] * n_q        # final H
        )
        pred = predict_gates(bv_gates, gate_errors=IONQ_ERRORS)

        print(f"  Submitting BV n_rep={n_rep} ...")
        job = backend.run(qc, shots=shots)
        result = job.result()
        counts = result.get_counts()

        # Check both bit orderings
        p_correct = max(
            counts.get(expected, 0),
            counts.get(expected[::-1], 0),
        ) / shots

        print(
            f"  n_rep={n_rep}: P(correct)={p_correct:.3f}  "
            f"expected={expected}  "
            f"counts={counts}"
        )

        results["experiments"].append({
            "type": "bernstein_vazirani",
            "hidden_string": hidden,
            "n_rep": n_rep,
            "expected": expected,
            "p_correct": p_correct,
            "f_tauchrono": pred.f_tauchrono,
            "f_naive": pred.f_naive,
            "counts": counts,
        })

    # =================================================================
    # Experiment 3: QEC Recommendation (IonQ error rates)
    # =================================================================
    print("\n" + "=" * 60)
    print("Experiment 3: QEC Recommendation (IonQ error rates)")
    print("=" * 60)

    qec_result = should_enable_qec(IONQ_ERRORS)
    print(f"  should_enable_qec: {qec_result.enable}")
    print(f"  Reason: {qec_result.reason}")
    print(f"  Predicted LER without QEC: {qec_result.predicted_ler_without_qec:.6f}")
    print(f"  Predicted LER with QEC:    {qec_result.predicted_ler_with_qec:.6f}")
    print(f"  Threshold error rate:      {qec_result.threshold_error_rate:.4f}")

    results["qec_prediction"] = {
        "enable": qec_result.enable,
        "reason": qec_result.reason,
        "predicted_ler_without": qec_result.predicted_ler_without_qec,
        "predicted_ler_with": qec_result.predicted_ler_with_qec,
        "threshold_error_rate": qec_result.threshold_error_rate,
    }

    # Save results
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"azure_ionq_simulator_{timestamp}.json")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved: {out_path}")
    return results, out_path


def print_summary(results):
    """Print a summary of all experiments."""
    print("\n" + "=" * 60)
    print("SUMMARY: IonQ Simulator Validation")
    print("=" * 60)

    print(f"\nBackend: {results['backend']}")
    print(f"Timestamp: {results['timestamp']}")
    print(f"IonQ error rates: cx={IONQ_ERRORS['cx']}, h={IONQ_ERRORS['h']}")

    # Depth scaling
    print("\n--- Depth Scaling (mirror circuits) ---")
    print(f"{'Depth':>6} {'Gates':>6} {'P(00)':>8} {'tau-chrono F':>14} {'naive F':>10} {'Improv%':>8}")
    print("-" * 56)
    depth_results = [e for e in results["experiments"] if e["type"] == "depth_scaling"]
    for r in depth_results:
        print(
            f"{r['depth']:6d} {r['n_gates']:6d} {r['p_correct']:8.3f} "
            f"{r['f_tauchrono']:14.4f} {r['f_naive']:10.4f} {r['improvement_pct']:8.1f}"
        )

    # BV
    print("\n--- Bernstein-Vazirani (s=101) ---")
    bv_results = [e for e in results["experiments"] if e["type"] == "bernstein_vazirani"]
    for r in bv_results:
        print(
            f"  n_rep={r['n_rep']}: P(correct)={r['p_correct']:.3f}  "
            f"expected={r['expected']}  counts={r['counts']}"
        )

    # QEC
    print(f"\n--- QEC Recommendation ---")
    qec = results["qec_prediction"]
    print(f"  Enable QEC: {qec['enable']}")
    print(f"  Reason: {qec['reason']}")
    print(f"  LER without QEC: {qec['predicted_ler_without']:.6f}")
    print(f"  LER with QEC:    {qec['predicted_ler_with']:.6f}")


def main():
    print("=" * 60)
    print("tau-chrono IonQ Simulator Validation (Azure Quantum)")
    print("=" * 60)

    workspace = get_workspace()
    targets = list_targets(workspace)

    print("\n--- Running on IonQ Simulator (FREE) ---")
    results, out_path = run_on_ionq_simulator(workspace)

    print_summary(results)


if __name__ == "__main__":
    main()
