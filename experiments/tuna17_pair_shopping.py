#!/usr/bin/env python3
"""
Tuna-17 pair shopping: anomalous-weak-value F_anomaly per coupler-connected pair.

Phase 1 of the chemistry sprint (TUNA17_CHEMISTRY_PLAN.md).
This identifies the cluster of qubits with high mean F_anomaly and low
non-uniformity, which is then used by Phase 2 (per-pair calibration) and
Phase 3+ (chemistry VQE).

Usage:
    # Simulator validation first (free, fast)
    .venv/bin/python experiments/tuna17_pair_shopping.py --backend "QX emulator" --shots 4096

    # Tuna-9 sanity check (we know F_anomaly = 0.793 +/- 0.01 from v2)
    .venv/bin/python experiments/tuna17_pair_shopping.py --backend "Tuna-9" --shots 4096

    # Real run on Tuna-17
    .venv/bin/python experiments/tuna17_pair_shopping.py --backend "Tuna-17" --shots 4096

The script auto-detects the coupling map and runs one circuit per connected
pair. Single g=0.30 (the v2 4-platform validation g). All output JSON.
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qiskit import QuantumCircuit
from qiskit.compiler import transpile

from tau_chrono_v2.anomaly import (
    extract_F_anomaly,
    theory_pointer,
)

# v2 reference angles (used in 4-platform validation)
THETA_PSI_DEG = 250.5
THETA_PHI_DEG = 109.5


def build_anomaly_circuit(theta_psi: float, theta_phi: float, g: float,
                          qa_logical: int = 0, qb_logical: int = 1) -> QuantumCircuit:
    """Standard Aharonov-Vaidman 1988 weak-value circuit on 2 qubits.

    Encodes the protocol matching tau_chrono_v2.anomaly.theory_pointer:
      1. R_y(theta_psi) on system  ->  |psi> = R_y(theta_psi)|0>
      2. controlled-R_y(2g) with sys as control, met as target
         (decomposed as Ry(g) - CX - Ry(-g) - CX, 2 CX gates total)
      3. R_y(-theta_phi) on system (rotates the post-select state to |0>)
      4. H on meter (rotates to X basis for pointer readout)
      5. Measure both qubits in Z basis
         - Post-select on bit_sys = 0  (i.e. system landed in |phi>)
         - Pointer = <sigma_x>_meter = (n_met=0 - n_met=1) / n_postselected
    """
    qc = QuantumCircuit(2, 2)
    sys, met = qa_logical, qb_logical

    # Step 1: prep |psi> on system
    qc.ry(theta_psi, sys)

    # Step 2: controlled-R_y(2g)
    qc.ry(g,  met)
    qc.cx(sys, met)
    qc.ry(-g, met)
    qc.cx(sys, met)

    # Step 3: post-select rotation on system
    qc.ry(-theta_phi, sys)

    # Step 4: pointer rotation on meter (X-basis readout)
    qc.h(met)

    # Step 5: measure
    qc.barrier()
    qc.measure(sys, 0)
    qc.measure(met, 1)
    return qc


def _parse_bits_for_backend(bs: str, qa_phys: int, qb_phys: int,
                            backend_name: str):
    """Return (bit_sys, bit_met) for one bitstring.

    Backend conventions differ:
    - Quantum Inspire QX emulator: bs is in physical-qubit order
      (bs[-1]=q0, bs[-2]=q1, ...). Index by physical qubit.
    - QuTech Tuna-9 / Tuna-17 (and any QuTech `Tuna-N` chip): bs is in
      classical-register order — bs[-1] = c[0] = sys, bs[-2] = c[1] = met
      regardless of which physical qubits we mapped to. Empirically
      verified on Tuna-9 (v2 raw_counts) and Tuna-17 (Bell test on q4-q7
      gives 94% |00>+|11> correlation when read as bs[-1]=control,
      bs[-2]=target)."""
    bs = bs.replace(" ", "")
    name = backend_name.lower()
    if name.startswith("tuna"):
        # Classical-register convention
        bs_clean = bs.zfill(2)
        return bs_clean[-1], bs_clean[-2]
    # Default (QX emulator and similar): physical-qubit ordered
    pad = bs.zfill(max(qa_phys, qb_phys) + 1)
    return pad[-(qa_phys + 1)], pad[-(qb_phys + 1)]


def run_pair(backend, qa_phys: int, qb_phys: int, shots: int,
             theta_psi: float, theta_phi: float, g: float,
             max_attempts: int = 3) -> dict:
    """Build, transpile (mapping logical 0,1 -> physical qa,qb), run.
    Returns post-selected pointer + F_anomaly estimate."""
    qc = build_anomaly_circuit(theta_psi, theta_phi, g, 0, 1)

    # Map logical 0 -> qa_phys, logical 1 -> qb_phys
    transpiled = transpile(qc, backend, initial_layout=[qa_phys, qb_phys],
                           optimization_level=0)
    n_cz = sum(1 for instr in transpiled.data if instr.operation.name == "cz")
    n_cx = sum(1 for instr in transpiled.data if instr.operation.name == "cx")

    # Submit job
    last_err = None
    counts = None
    for attempt in range(max_attempts):
        try:
            job = backend.run(transpiled, shots=shots)
            job.wait_for_final_state(timeout=1800)
            counts = job.result().get_counts(0)
            break
        except Exception as e:
            last_err = str(e)
            print(f"      attempt {attempt+1} failed: {e}")
            time.sleep(5)
    if counts is None:
        return {"status": "fail", "error": last_err,
                "qa": qa_phys, "qb": qb_phys, "g": g}

    n_post = 0
    n_post_met0 = 0
    n_post_met1 = 0
    total_shots_seen = 0
    for bs, n in counts.items():
        bit_sys, bit_met = _parse_bits_for_backend(
            bs, qa_phys, qb_phys, backend.name)
        total_shots_seen += n
        if bit_sys == "0":
            n_post += n
            if bit_met == "0":
                n_post_met0 += n
            else:
                n_post_met1 += n

    if n_post < 50:
        # Too few post-selected events — F_anomaly very noisy
        return {
            "status": "low_postselect",
            "qa": qa_phys, "qb": qb_phys,
            "g": g, "shots": shots, "n_post": n_post,
            "n_cz": n_cz, "n_cx": n_cx,
            "p_post": n_post / max(total_shots_seen, 1),
        }

    # Pointer = <sigma_x>_meter for post-selected events
    pointer = (n_post_met0 - n_post_met1) / n_post
    pointer_sigma = math.sqrt((1 - pointer**2) / max(n_post - 1, 1))

    # F_anomaly = pointer_obs / pointer_theory
    th = theory_pointer(math.radians(theta_psi), math.radians(theta_phi), g)
    F = pointer / th["pointer"] if th["pointer"] != 0 else float("nan")
    F_sigma = pointer_sigma / abs(th["pointer"]) if th["pointer"] != 0 else float("nan")

    # weak values
    sin2g = math.sin(2 * g)
    wv_pi1 = pointer / sin2g
    wv_pi0 = 1.0 - wv_pi1

    return {
        "status": "ok",
        "qa": qa_phys, "qb": qb_phys,
        "g": g, "shots": shots,
        "n_cz": n_cz, "n_cx": n_cx,
        "n_post": n_post, "p_post": n_post / total_shots_seen,
        "pointer": pointer, "pointer_sigma": pointer_sigma,
        "pointer_theory": th["pointer"],
        "F_anomaly": F, "F_anomaly_sigma": F_sigma,
        "wv_pi0": wv_pi0, "wv_pi1": wv_pi1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="QX emulator",
                        help="Quantum Inspire backend name")
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--g", type=float, default=0.30)
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="Limit number of pairs (for quick tests)")
    parser.add_argument("--out", default=None,
                        help="Output JSON path")
    args = parser.parse_args()

    from qiskit_quantuminspire.qi_provider import QIProvider
    provider = QIProvider()
    backend = provider.get_backend(args.backend)
    print(f"Connected: {backend.name} ({backend.num_qubits} qubits)")

    # Get coupling map
    cm = backend.coupling_map
    if cm is None:
        n = backend.num_qubits
        edges = [(i, i + 1) for i in range(min(n - 1, 3))]
        print(f"No coupling map (all-to-all); using {len(edges)} adjacent pairs for validation")
    else:
        edges = sorted(set(tuple(sorted([a, b])) for a, b in cm.get_edges()))
        print(f"Found {len(edges)} undirected coupler pairs")

    if args.max_pairs:
        edges = edges[: args.max_pairs]
        print(f"  -> limited to first {len(edges)} pairs")

    theta_psi = math.radians(THETA_PSI_DEG)
    theta_phi = math.radians(THETA_PHI_DEG)

    # Theory baseline
    th = theory_pointer(theta_psi, theta_phi, args.g)
    print(f"\nTheory at g={args.g}, theta_psi={THETA_PSI_DEG}deg, theta_phi={THETA_PHI_DEG}deg:")
    print(f"  pointer    = {th['pointer']:.4f}")
    print(f"  p_post     = {th['p_post']:.4f}")
    print(f"  <Pi_1>_w   = {th['pi1_observed']:.4f}")
    print(f"  <Pi_0>_w   = {th['pi0_observed']:.4f}  (negative = anomalous)\n")

    results = {}
    t0 = time.time()
    for i, (qa, qb) in enumerate(edges):
        key = f"q{qa}_q{qb}"
        print(f"[{i+1}/{len(edges)}] pair {key}  ", end="", flush=True)
        r = run_pair(backend, qa, qb, args.shots, THETA_PSI_DEG, THETA_PHI_DEG, args.g)
        results[key] = r
        if r["status"] == "ok":
            print(f"  F={r['F_anomaly']:+.3f}±{r['F_anomaly_sigma']:.3f}  "
                  f"<Pi_0>_w={r['wv_pi0']:+.3f}  "
                  f"n_post={r['n_post']}/{args.shots}")
        else:
            print(f"  STATUS={r['status']}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s.")

    # Summary
    ok_pairs = [r for r in results.values() if r["status"] == "ok"]
    if ok_pairs:
        Fs = np.array([r["F_anomaly"] for r in ok_pairs])
        print(f"\nSummary across {len(ok_pairs)} OK pairs:")
        print(f"  F_anomaly  mean = {Fs.mean():.3f}")
        print(f"  F_anomaly  std  = {Fs.std():.3f}")
        print(f"  F_anomaly  range= [{Fs.min():.3f}, {Fs.max():.3f}]   spread={Fs.max()-Fs.min():.3f}")
        print(f"  Best pair (highest F_anomaly):")
        best = max(ok_pairs, key=lambda r: r["F_anomaly"])
        print(f"    q{best['qa']}-q{best['qb']}: F={best['F_anomaly']:.3f}")

    # Save
    os.makedirs("data/iqm_4platform_validation", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out is None:
        backend_tag = args.backend.lower().replace(" ", "_").replace("-", "")
        out_path = f"data/iqm_4platform_validation/awv_t17_pairsweep_{backend_tag}_{ts}.json"
    else:
        out_path = args.out
    payload = {
        "metadata": {
            "backend": args.backend,
            "shots_per_pair": args.shots,
            "g": args.g,
            "theta_psi_deg": THETA_PSI_DEG,
            "theta_phi_deg": THETA_PHI_DEG,
            "n_pairs": len(edges),
            "timestamp": ts,
            "elapsed_sec": elapsed,
        },
        "theory": {"g": args.g, **th},
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
