#!/usr/bin/env python3
"""
Phase B0c: Zero-Noise Extrapolation (ZNE) on H2 at best theta on Tuna-17 q2-q5.

Pipeline:
  1. Take best theta from previous H2 scan (theta = +2.918, near analytical optimum)
  2. Run circuit with CNOT folding scales 1, 3, 5 (each: ZZ + XX basis)
  3. Apply symmetry post-selection on ZZ basis counts
  4. Apply per-pair F_XX correction on XX basis counts
  5. For each scale, compute corrected energy
  6. Linear extrapolate energies vs scale to scale=0
  7. Compare to FCI

CNOT folding: replace each CX(c,t) with CX(c,t) CX(c,t) CX(c,t) for scale=3,
i.e. 3× the gates → 3× the noise.

Usage:
    .venv/bin/python experiments/h2_zne_at_best_theta.py \
        --backend "Tuna-17" --shots 8192 --theta 2.918
"""

import argparse
import glob
import json
import math
import os
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.compiler import transpile

from experiments.h2_baseline_diagnostic import (
    H2_HAM, R_BOND, exact_ground_state_energy,
    expectation_from_counts, _parse_two_qubit_counts,
)


def h2_ansatz_with_folding(theta: float, fold: int) -> QuantumCircuit:
    """H2 ansatz with each CX gate folded `fold` times.
    fold = 1: original (X; CX; Ry; CX) -> 2 CX gates total
    fold = 3: each CX -> CX CX CX -> 6 CX gates total
    fold = 5: each CX -> CX CX CX CX CX -> 10 CX gates total
    Folding is identity (CX^2 = I) in noiseless, but adds noise on hardware."""
    if fold % 2 != 1 or fold < 1:
        raise ValueError(f"fold must be odd positive integer, got {fold}")
    qc = QuantumCircuit(2)
    qc.x(0)
    for _ in range(fold):
        qc.cx(0, 1)
    qc.ry(theta, 0)
    for _ in range(fold):
        qc.cx(0, 1)
    return qc


def build_measure_circuit(ansatz: QuantumCircuit, basis: str) -> QuantumCircuit:
    qc = ansatz.copy()
    qc.add_register(ClassicalRegister(2, "c"))
    if basis == "XX":
        qc.h(0); qc.h(1)
    qc.measure(0, 0)
    qc.measure(1, 1)
    return qc


def run_one(backend, qc: QuantumCircuit, shots: int, layout: list,
            label: str = "") -> dict:
    t = transpile(qc, backend, initial_layout=layout, optimization_level=0)
    n_cx = sum(1 for i in t.data if i.operation.name == "cx")
    n_cz = sum(1 for i in t.data if i.operation.name == "cz")
    job = backend.run(t, shots=shots)
    job.wait_for_final_state(timeout=1800)
    counts = job.result().get_counts(0)
    print(f"  [{label}] CX={n_cx} CZ={n_cz} depth={t.depth()} -> counts top: "
          f"{dict(sorted(counts.items(), key=lambda x:-x[1])[:3])}")
    return {"counts": counts, "n_cx": n_cx, "n_cz": n_cz, "depth": t.depth()}


def compute_corrected_energy(zcounts: dict, xcounts: dict, calibration: dict,
                              ham: dict) -> dict:
    """PS on Z-basis (drop |00>, |11>); F_XX correction on X-basis."""
    n00z = zcounts.get("00", 0)
    n01z = zcounts.get("01", 0)
    n10z = zcounts.get("10", 0)
    n11z = zcounts.get("11", 0)
    n_ps = n01z + n10z
    n_total_z = n00z + n01z + n10z + n11z
    p_keep = n_ps / max(n_total_z, 1)

    if n_ps > 0:
        # In odd-parity subset: |10> has q0=0 (IZ=+1), q1=1 (ZI=-1)
        #                       |01> has q0=1 (IZ=-1), q1=0 (ZI=+1)
        iz_ps = (n10z - n01z) / n_ps
        zi_ps = (n01z - n10z) / n_ps
        zz_ps = -1.0
    else:
        iz_ps = zi_ps = zz_ps = 0.0

    n00x = xcounts.get("00", 0)
    n01x = xcounts.get("01", 0)
    n10x = xcounts.get("10", 0)
    n11x = xcounts.get("11", 0)
    total_x = n00x + n01x + n10x + n11x
    if total_x > 0:
        xx_raw = (n00x + n11x - n01x - n10x) / total_x
    else:
        xx_raw = 0.0
    F_xx, b_xx = calibration["XX"]
    xx_corr = (xx_raw - b_xx) / F_xx

    e_ps_only = (ham["II"] + ham["IZ"] * iz_ps + ham["ZI"] * zi_ps
                 + ham["ZZ"] * zz_ps + ham["XX"] * xx_raw)
    e_ps_fxx = (ham["II"] + ham["IZ"] * iz_ps + ham["ZI"] * zi_ps
                + ham["ZZ"] * zz_ps + ham["XX"] * xx_corr)
    return {
        "iz_ps": iz_ps, "zi_ps": zi_ps, "zz_ps": zz_ps,
        "xx_raw": xx_raw, "xx_corr": xx_corr,
        "e_ps_only": e_ps_only, "e_ps_fxx": e_ps_fxx,
        "p_keep_PS": p_keep, "n_ps": n_ps,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="Tuna-17")
    parser.add_argument("--shots", type=int, default=8192)
    parser.add_argument("--theta", type=float, default=2.918,
                        help="theta value for H2 ansatz (analytical optimum)")
    parser.add_argument("--fold-scales", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--cal-json", default=None,
                        help="Path to ABR calibration JSON (defaults to latest)")
    args = parser.parse_args()

    # Load calibration
    if args.cal_json is None:
        backend_tag = args.backend.lower().replace(" ", "_").replace("-", "")
        cands = sorted(glob.glob(
            f"data/iqm_4platform_validation/h2_abr_calib_{backend_tag}_*.json"))
        if not cands:
            print(f"ERROR: no calibration JSON found for {args.backend}")
            sys.exit(1)
        args.cal_json = cands[-1]
    print(f"Loading calibration from: {args.cal_json}")
    cal_data = json.load(open(args.cal_json))
    calibration = {
        p: (cal_data["calibration_params"][p]["F"],
            cal_data["calibration_params"][p]["bias"])
        for p in ["IZ", "ZI", "ZZ", "XX"]
    }
    print(f"  XX calibration: F = {calibration['XX'][0]:.4f}, "
          f"bias = {calibration['XX'][1]:.4f}")

    e_fci = exact_ground_state_energy(H2_HAM)
    print(f"  E_FCI = {e_fci:+.6f} Ha")
    print()

    from qiskit_quantuminspire.qi_provider import QIProvider
    backend = QIProvider().get_backend(args.backend)
    layout = [2, 5] if "tuna-17" in args.backend.lower() else [0, 1]
    print(f"Backend: {backend.name}, layout = {layout}")
    print(f"Theta: {args.theta:+.4f} rad")
    print()

    # Run for each fold
    results_per_fold = {}
    t0 = time.time()
    for fold in args.fold_scales:
        print(f"=== Fold scale = {fold}× ===")
        ans = h2_ansatz_with_folding(args.theta, fold)
        z_qc = build_measure_circuit(ans, "ZZ")
        x_qc = build_measure_circuit(ans, "XX")
        z_run = run_one(backend, z_qc, args.shots, layout,
                        label=f"fold={fold} ZZ")
        x_run = run_one(backend, x_qc, args.shots, layout,
                        label=f"fold={fold} XX")
        zc = dict(zip(["00", "01", "10", "11"],
                      _parse_two_qubit_counts(z_run["counts"], backend.num_qubits)))
        xc = dict(zip(["00", "01", "10", "11"],
                      _parse_two_qubit_counts(x_run["counts"], backend.num_qubits)))
        corr = compute_corrected_energy(zc, xc, calibration, H2_HAM)
        results_per_fold[fold] = {
            "fold": fold,
            "n_cx_total": z_run["n_cx"],
            "depth": z_run["depth"],
            "zcounts": zc, "xcounts": xc,
            **corr,
        }
        print(f"  fold={fold}: e_PS = {corr['e_ps_only']:+.6f}, "
              f"e_PS+F_XX = {corr['e_ps_fxx']:+.6f}, "
              f"p_keep_PS = {corr['p_keep_PS']:.3f}")
        print()
    elapsed = time.time() - t0

    # Linear extrapolation in (1/F effective ~ scale) → scale=0
    # Standard ZNE: fit E(scale) linearly, extrapolate to scale=0.
    folds = sorted(args.fold_scales)
    energies_psf = [results_per_fold[f]["e_ps_fxx"] for f in folds]
    # Linear fit
    scales = np.array(folds, dtype=float)
    energies = np.array(energies_psf)
    if len(folds) >= 2:
        # Least-squares linear: E = a + b*scale
        A = np.vstack([np.ones_like(scales), scales]).T
        coeffs, *_ = np.linalg.lstsq(A, energies, rcond=None)
        e_zne = float(coeffs[0])  # extrapolated to scale = 0
    else:
        e_zne = energies[0]

    abs_err_zne_mHa = abs(e_zne - e_fci) * 1000
    abs_err_psf_1x_mHa = abs(results_per_fold[1]["e_ps_fxx"] - e_fci) * 1000

    print("=== Summary ===")
    print(f"{'fold':>5} {'E_PS+F_XX (Ha)':>16} {'|err| (mHa)':>14}")
    for f in folds:
        e = results_per_fold[f]["e_ps_fxx"]
        err = abs(e - e_fci) * 1000
        print(f"{f:>5} {e:+15.6f}    {err:>11.2f}")
    print(f"{'ZNE→0':>5} {e_zne:+15.6f}    {abs_err_zne_mHa:>11.2f}")
    print(f"{'FCI':>5} {e_fci:+15.6f}    {0:>11.2f}")
    print()
    print(f"Reduction PS+F_XX -> ZNE: "
          f"{abs_err_psf_1x_mHa / max(abs_err_zne_mHa, 1e-6):.2f}×")
    print(f"Within 1 kcal/mol (1.6 mHa)? "
          f"{'YES ✓' if abs_err_zne_mHa <= 1.6 else 'NO'}")
    print(f"Elapsed: {elapsed:.0f} sec")

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backend_tag = args.backend.lower().replace(" ", "_").replace("-", "")
    out_path = f"data/iqm_4platform_validation/h2_zne_{backend_tag}_{ts}.json"
    payload = {
        "metadata": {
            "backend": args.backend,
            "layout": layout,
            "theta": args.theta,
            "shots_per_circuit": args.shots,
            "fold_scales": args.fold_scales,
            "calibration_json": args.cal_json,
            "E_FCI": e_fci,
            "elapsed_sec": elapsed,
            "timestamp": ts,
        },
        "calibration": {p: {"F": F, "bias": b}
                        for p, (F, b) in calibration.items()},
        "per_fold": results_per_fold,
        "ZNE_extrapolated_energy": e_zne,
        "ZNE_abs_err_mHa": abs_err_zne_mHa,
        "PS_FXX_abs_err_mHa_at_fold1": abs_err_psf_1x_mHa,
        "ZNE_reduction_factor": float(abs_err_psf_1x_mHa / max(abs_err_zne_mHa, 1e-6)),
        "within_chemical_accuracy": bool(abs_err_zne_mHa <= 1.6),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
