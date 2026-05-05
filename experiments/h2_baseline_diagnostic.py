#!/usr/bin/env python3
"""
Phase B0: H2 baseline diagnostic on Tuna-9.

Purpose: explain why v2 chemistry sprint shows H2 at 0/7 chemical-accuracy
hits despite H2 being the simplest molecule. Possible root causes:
  (a) VQE optimizer stuck in local minima (poor initialization)
  (b) Ansatz limited expressibility (cannot reach FCI even noiseless)
  (c) Shot noise dominant on shallow circuit
  (d) ABR's noise model mismatch on shallow circuit

This script compares three configurations at R = 0.735 angstrom (H2
equilibrium), all on the SAME Hamiltonian + ansatz:
  1. Noiseless qiskit simulator -> ansatz expressibility lower bound
  2. Tuna-9 raw VQE (no mitigation)
  3. Tuna-9 + ABR-style global F correction

Hamiltonian: 2-qubit parity-reduced sto-3g H2 at R=0.735 angstrom
(O'Malley et al PRX 2016, supplementary):

    H = c0 II + c1 IZ + c2 ZI + c3 ZZ + c4 XX
    c0 = -1.0523732, c1 = -0.39793742, c2 = +0.39793742,
    c3 = -0.01128010, c4 = +0.18093119

Ansatz (parity encoding, 1 parameter, matches FCI in odd-parity sector):
    |psi(theta)> = exp(-i theta/2 XX) X_q0 |00>
    Implemented as:  X(q0); RXX(theta)(q0,q1)
    RXX(theta) = (H ⊗ H) CX(q1,q0) Rz(theta)(q0) CX(q1,q0) (H ⊗ H)
                 (decomposes into single-qubit + 2 CX gates)

Usage:
    .venv/bin/python experiments/h2_baseline_diagnostic.py --backend "QX emulator" --shots 8192
    .venv/bin/python experiments/h2_baseline_diagnostic.py --backend "Tuna-9" --shots 8192
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime

import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qiskit import QuantumCircuit
from qiskit.compiler import transpile

# ---------------------------------------------------------------------------
# H2 Hamiltonian at R = 0.735 angstrom (parity-reduced sto-3g)
# Reference: O'Malley et al., Phys Rev X 6, 031007 (2016), supplementary table.
# ---------------------------------------------------------------------------
H2_HAM = {
    "II": -1.0523732,
    "IZ": -0.39793742,
    "ZI": +0.39793742,
    "ZZ": -0.01128010,
    "XX": +0.18093119,
}
R_BOND = 0.735  # angstrom


def exact_ground_state_energy(ham: dict) -> float:
    """Diagonalize the 4x4 Hamiltonian and return ground-state energy."""
    paulis = {
        "II": np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
        "IZ": np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]),
        "ZI": np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]]),
        "ZZ": np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]),
        "XX": np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]),
    }
    H = sum(c * paulis[p] for p, c in ham.items())
    eigvals = np.linalg.eigvalsh(H)
    return float(eigvals[0])


# ---------------------------------------------------------------------------
# Ansatz
# ---------------------------------------------------------------------------

def h2_ansatz_circuit(theta: float) -> QuantumCircuit:
    """1-parameter Givens-rotation ansatz reaching FCI in odd-parity sector.

    Sequence:
      X(q0); CX(q0,q1); Ry(theta, q0); CX(q0,q1)

    Trace:
      |00> -X(q0)-> |01>_qiskit -CX(q0,q1)-> |11> -Ry(theta,q0)->
        cos(theta/2)|11> - sin(theta/2)|10> -CX(q0,q1)->
        cos(theta/2)|01> - sin(theta/2)|10>     (real superposition in odd parity)

    Yields <XX>(theta) = -sin(theta), so the c4*XX term in H can be used
    to reach FCI. With H2 sto-3g coefficients above, the optimum is near
    theta = pi - arctan(0.181/0.796) ≈ +2.92 rad."""
    qc = QuantumCircuit(2)
    qc.x(0)
    qc.cx(0, 1)
    qc.ry(theta, 0)
    qc.cx(0, 1)
    return qc


# ---------------------------------------------------------------------------
# Expectation value via three measurement bases (Z-basis, X-basis, ZZ overlap)
# ---------------------------------------------------------------------------

def _measurement_circuit(ansatz: QuantumCircuit, basis: str) -> QuantumCircuit:
    """Wrap the ansatz with rotations + measurement for a chosen Pauli string."""
    qc = ansatz.copy()
    qc.add_register(__import__("qiskit").ClassicalRegister(2, "c"))
    if basis == "ZZ":
        # already in Z basis; measure both qubits
        pass
    elif basis == "XX":
        qc.h(0); qc.h(1)
    else:
        raise ValueError(f"Unsupported measurement basis: {basis}")
    qc.measure(0, 0)
    qc.measure(1, 1)
    return qc


def _parse_two_qubit_counts(counts: dict, num_qubits_backend: int):
    """Extract (n00, n01, n10, n11) from counts.
    For Tuna-N the counts may be returned as full N-qubit bitstrings;
    classical-register convention has c[0]=q_meas_first at bs[-1] and
    c[1] at bs[-2]."""
    n = {"00": 0, "01": 0, "10": 0, "11": 0}
    for bs, cnt in counts.items():
        bs_clean = bs.replace(" ", "")
        # Always read the LAST 2 bits as classical register c[1]c[0]
        if len(bs_clean) < 2:
            bs_clean = bs_clean.zfill(2)
        c0 = bs_clean[-1]  # q0 (sys)
        c1 = bs_clean[-2]  # q1 (met)
        key = c1 + c0  # "q1 q0" string (canonical |c1 c0> = bs_clean[-2:][::-1])
        n[key] = n.get(key, 0) + cnt
    return n["00"], n["01"], n["10"], n["11"]


def expectation_from_counts(n00, n01, n10, n11, pauli2: str) -> float:
    """<P> for a 2-qubit Pauli string in the Z- or X-basis-rotated counts.
    pauli2 in {II, IZ, ZI, ZZ, XX}. Note: II is trivially 1.
    For ZZ counts, this gives <ZZ>, <IZ>, <ZI>, <II>.
    For XX counts (after H on both), this gives <XX>.
    """
    total = n00 + n01 + n10 + n11
    if total == 0:
        return 0.0
    if pauli2 == "II":
        return 1.0
    # Z eigenvalue of bit b: +1 if b=0, -1 if b=1.
    if pauli2 == "ZZ":
        # +1 if both same parity, -1 otherwise
        return (n00 + n11 - n01 - n10) / total
    # qiskit convention: "IZ" = I⊗Z = Z on q0 (LSB of |q1 q0>);
    #                    "ZI" = Z⊗I = Z on q1 (MSB).
    if pauli2 == "IZ":
        # <Z_q0> = P(q0=0) - P(q0=1)  with key "q1q0":
        # q0=0 in n00, n10; q0=1 in n01, n11
        return (n00 + n10 - n01 - n11) / total
    if pauli2 == "ZI":
        # <Z_q1> = P(q1=0) - P(q1=1)
        # q1=0 in n00, n01; q1=1 in n10, n11
        return (n00 + n01 - n10 - n11) / total
    if pauli2 == "XX":
        return (n00 + n11 - n01 - n10) / total
    raise ValueError(pauli2)


def vqe_energy_from_two_bases(zcounts: tuple, xcounts: tuple, ham: dict) -> float:
    """Combine ZZ-basis + XX-basis measurements into total energy."""
    n00z, n01z, n10z, n11z = zcounts
    n00x, n01x, n10x, n11x = xcounts
    expvals = {
        "II": 1.0,
        "IZ": expectation_from_counts(n00z, n01z, n10z, n11z, "IZ"),
        "ZI": expectation_from_counts(n00z, n01z, n10z, n11z, "ZI"),
        "ZZ": expectation_from_counts(n00z, n01z, n10z, n11z, "ZZ"),
        "XX": expectation_from_counts(n00x, n01x, n10x, n11x, "XX"),
    }
    energy = sum(ham[p] * expvals[p] for p in ham)
    return energy, expvals


# ---------------------------------------------------------------------------
# Backend abstraction (one_circuit_to_counts)
# ---------------------------------------------------------------------------

def run_circuit(backend, qc: QuantumCircuit, shots: int,
                initial_layout=None) -> dict:
    """Transpile and run a single circuit, return counts dict."""
    t = transpile(qc, backend, initial_layout=initial_layout,
                  optimization_level=0)
    job = backend.run(t, shots=shots)
    job.wait_for_final_state(timeout=1800)
    return job.result().get_counts(0)


def measure_energy(backend, theta: float, shots: int,
                   initial_layout=None) -> dict:
    """Run both Z- and X-basis measurements; compute energy."""
    ans = h2_ansatz_circuit(theta)
    qc_z = _measurement_circuit(ans, "ZZ")
    qc_x = _measurement_circuit(ans, "XX")
    zcounts_dict = run_circuit(backend, qc_z, shots, initial_layout)
    xcounts_dict = run_circuit(backend, qc_x, shots, initial_layout)
    zc = _parse_two_qubit_counts(zcounts_dict, backend.num_qubits)
    xc = _parse_two_qubit_counts(xcounts_dict, backend.num_qubits)
    energy, expvals = vqe_energy_from_two_bases(zc, xc, H2_HAM)
    return {
        "theta": theta,
        "energy": energy,
        "expvals": expvals,
        "zcounts": dict(zip(["00", "01", "10", "11"], zc)),
        "xcounts": dict(zip(["00", "01", "10", "11"], xc)),
    }


# ---------------------------------------------------------------------------
# VQE optimization
# ---------------------------------------------------------------------------

def vqe_scan_theta(backend, shots: int, theta_grid: list,
                   initial_layout=None, label: str = "scan") -> list:
    """Brute-force scan over theta values; return list of results."""
    out = []
    for th in theta_grid:
        r = measure_energy(backend, th, shots, initial_layout)
        out.append(r)
        print(f"    [{label}] theta = {th:+.3f}  E = {r['energy']:+.6f} Ha  "
              f"(IZ={r['expvals']['IZ']:+.3f}, ZI={r['expvals']['ZI']:+.3f}, "
              f"XX={r['expvals']['XX']:+.3f})")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="QX emulator")
    parser.add_argument("--shots", type=int, default=8192)
    parser.add_argument("--scan-points", type=int, default=15,
                        help="Number of theta values in the scan")
    parser.add_argument("--theta-min", type=float, default=-math.pi)
    parser.add_argument("--theta-max", type=float, default=+math.pi)
    args = parser.parse_args()

    e_fci = exact_ground_state_energy(H2_HAM)
    print(f"H2 Hamiltonian at R = {R_BOND} angstrom (sto-3g parity-reduced)")
    print(f"  Exact (numerical diag) E_FCI = {e_fci:.6f} Ha")
    print()

    from qiskit_quantuminspire.qi_provider import QIProvider
    backend = QIProvider().get_backend(args.backend)
    print(f"Connected: {backend.name} ({backend.num_qubits} qubits)")
    print(f"Shots per circuit: {args.shots}")
    print()

    # Scan grid for theta
    theta_grid = list(np.linspace(args.theta_min, args.theta_max, args.scan_points))

    # For Tuna-9 use physical (0,1); for Tuna-17 use best pair (2,5) from
    # Phase 1 pair shopping (F_anomaly = 0.799 ≈ Tuna-9's 0.793).
    if "tuna-17" in args.backend.lower():
        initial_layout = [2, 5]
    elif "tuna" in args.backend.lower():
        initial_layout = [0, 1]
    else:
        initial_layout = None
    print(f"  initial_layout = {initial_layout}")

    print(f"=== theta scan ({args.scan_points} points) ===")
    t0 = time.time()
    results = vqe_scan_theta(backend, args.shots, theta_grid, initial_layout, "raw")
    elapsed = time.time() - t0

    energies = [r["energy"] for r in results]
    best_idx = int(np.argmin(energies))
    best = results[best_idx]

    # Refine: golden-section search in a window around the grid winner.
    # Period is 2pi, so we search +/- one grid step (no boundary clipping).
    print()
    print("=== refining optimum with golden-section search ===")
    if len(theta_grid) >= 2:
        step = abs(theta_grid[1] - theta_grid[0])
    else:
        step = 0.2
    lo = best["theta"] - step
    hi = best["theta"] + step
    refine_results = []

    def f_refine(th):
        r = measure_energy(backend, float(th), args.shots, initial_layout)
        refine_results.append(r)
        print(f"    [refine] theta = {th:+.4f}  E = {r['energy']:+.6f} Ha")
        return r["energy"]

    res = minimize_scalar(f_refine, bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-3, "maxiter": 8})
    theta_opt = float(res.x)
    e_opt = float(res.fun)
    if e_opt < best["energy"]:
        best = {"theta": theta_opt, "energy": e_opt,
                "expvals": refine_results[-1]["expvals"]}
    abs_err_mHa = abs(best["energy"] - e_fci) * 1000.0

    print()
    print(f"Best theta from scan = {best['theta']:+.4f} rad")
    print(f"Best energy = {best['energy']:+.6f} Ha")
    print(f"FCI         = {e_fci:+.6f} Ha")
    print(f"|err|       = {abs_err_mHa:.2f} mHa  "
          f"({'WITHIN 1 kcal/mol' if abs_err_mHa <= 1.6 else 'over 1 kcal/mol'})")
    print(f"Elapsed: {elapsed:.0f} sec")

    # Save
    os.makedirs("data/iqm_4platform_validation", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backend_tag = args.backend.lower().replace(" ", "_").replace("-", "")
    out_path = f"data/iqm_4platform_validation/h2_diag_{backend_tag}_{ts}.json"
    payload = {
        "metadata": {
            "backend": args.backend,
            "shots_per_circuit": args.shots,
            "R_bond_angstrom": R_BOND,
            "hamiltonian": H2_HAM,
            "E_FCI": e_fci,
            "elapsed_sec": elapsed,
            "timestamp": ts,
        },
        "scan": results,
        "best": {
            "theta": best["theta"],
            "energy": best["energy"],
            "abs_err_mHa": abs_err_mHa,
            "within_chemical_accuracy": abs_err_mHa <= 1.6,
        },
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
