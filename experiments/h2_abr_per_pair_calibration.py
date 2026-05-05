#!/usr/bin/env python3
"""
Phase B0b: Per-pair (F, bias) ABR calibration on Tuna-17 q2-q5,
applied OFFLINE to the saved H2 diagnostic JSON.

Steps:
  1. Run 4 calibration circuits (|00>, |11>, |++>, |+->) to extract
     (F_P, bias_P) for each Pauli P in {IZ, ZI, ZZ, XX}.
  2. Load the most recent H2 diagnostic JSON for Tuna-17.
  3. Apply <P>_corrected = (<P>_obs - bias_P) / F_P to every theta point.
  4. Recompute energy at each theta.
  5. Re-find best theta + report calibrated |err| vs FCI.

Usage:
    .venv/bin/python experiments/h2_abr_per_pair_calibration.py \
        --backend "Tuna-17" --shots 8192 \
        --h2-json data/iqm_4platform_validation/h2_diag_tuna17_20260428_191538.json
"""

import argparse
import glob
import json
import math
import os
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.compiler import transpile

from experiments.h2_baseline_diagnostic import (
    H2_HAM, R_BOND, exact_ground_state_energy,
    expectation_from_counts, _parse_two_qubit_counts,
)


def build_calibration_circuit(state: str, basis: str) -> QuantumCircuit:
    """Prepare a 2-qubit reference state, optionally rotate to X-basis,
    measure both qubits.

    state: '00', '11', '++', '+-'
    basis: 'ZZ' (no rotation) or 'XX' (apply H on both before measuring)
    """
    qc = QuantumCircuit(2)
    if state == "00":
        pass
    elif state == "11":
        qc.x(0); qc.x(1)
    elif state == "++":
        qc.h(0); qc.h(1)
    elif state == "+-":
        qc.h(0); qc.x(1); qc.h(1)
    else:
        raise ValueError(f"Unknown state {state}")
    if basis == "XX":
        qc.h(0); qc.h(1)
    qc.add_register(ClassicalRegister(2, "c"))
    qc.measure(0, 0)
    qc.measure(1, 1)
    return qc


def calibration_truth(state: str):
    """Return ideal expectation values for a calibration state."""
    if state == "00":
        return {"IZ": +1, "ZI": +1, "ZZ": +1, "XX": 0}
    if state == "11":
        return {"IZ": -1, "ZI": -1, "ZZ": +1, "XX": 0}
    if state == "++":
        return {"IZ": 0, "ZI": 0, "ZZ": 0, "XX": +1}
    if state == "+-":
        return {"IZ": 0, "ZI": 0, "ZZ": 0, "XX": -1}
    raise ValueError(state)


def run_calibration(backend, initial_layout, shots: int):
    """Run 4 calibration circuits (2 in ZZ basis, 2 in XX basis), return
    observed expectation values + ideal truth for each Pauli."""
    # Plan: <IZ>, <ZI>, <ZZ> from {|00>, |11>} in ZZ basis (2 circuits)
    #       <XX>             from {|++>, |+->} in XX basis (2 circuits)
    obs_data = {}
    for state, basis in [("00", "ZZ"), ("11", "ZZ"),
                          ("++", "XX"), ("+-", "XX")]:
        qc = build_calibration_circuit(state, basis)
        t = transpile(qc, backend, initial_layout=initial_layout,
                      optimization_level=0)
        job = backend.run(t, shots=shots)
        job.wait_for_final_state(timeout=1800)
        counts = job.result().get_counts(0)
        n00, n01, n10, n11 = _parse_two_qubit_counts(counts, backend.num_qubits)
        truth = calibration_truth(state)
        obs = {}
        for pauli in ["IZ", "ZI", "ZZ", "XX"]:
            obs[pauli] = expectation_from_counts(n00, n01, n10, n11, pauli)
        obs_data[(state, basis)] = {"obs": obs, "truth": truth,
                                     "counts": (n00, n01, n10, n11)}
        print(f"  prep |{state}> measure {basis}: counts=({n00},{n01},{n10},{n11})  "
              f"obs={ {p: round(v,3) for p,v in obs.items()} }")
    return obs_data


def fit_F_bias(obs_data, pauli: str, ref_states: list):
    """Linear fit <P>_obs = F * <P>_true + bias from two reference states."""
    if len(ref_states) != 2:
        raise ValueError("need exactly 2 reference states for linear fit")
    (s1, b1), (s2, b2) = ref_states
    x1 = obs_data[(s1, b1)]["truth"][pauli]
    y1 = obs_data[(s1, b1)]["obs"][pauli]
    x2 = obs_data[(s2, b2)]["truth"][pauli]
    y2 = obs_data[(s2, b2)]["obs"][pauli]
    if abs(x1 - x2) < 1e-9:
        raise ValueError(f"Reference states give same truth for {pauli}; "
                         f"need different values")
    F = (y1 - y2) / (x1 - x2)
    bias = y1 - F * x1
    return F, bias


def correct_energy(zcounts: dict, xcounts: dict,
                   calibration: dict, ham: dict) -> tuple:
    """Re-compute energy from ZZ + XX counts using per-Pauli correction.
    zcounts = {'00': n, '01': n, ...}, same for xcounts.
    calibration = {'IZ': (F, bias), 'ZI': (F, bias), 'ZZ': (F, bias),
                   'XX': (F, bias)}
    """
    z = (zcounts.get("00", 0), zcounts.get("01", 0),
         zcounts.get("10", 0), zcounts.get("11", 0))
    x = (xcounts.get("00", 0), xcounts.get("01", 0),
         xcounts.get("10", 0), xcounts.get("11", 0))
    obs_iz_raw = expectation_from_counts(*z, "IZ")
    obs_zi_raw = expectation_from_counts(*z, "ZI")
    obs_zz_raw = expectation_from_counts(*z, "ZZ")
    obs_xx_raw = expectation_from_counts(*x, "XX")
    F_iz, b_iz = calibration["IZ"]
    F_zi, b_zi = calibration["ZI"]
    F_zz, b_zz = calibration["ZZ"]
    F_xx, b_xx = calibration["XX"]
    iz_corr = (obs_iz_raw - b_iz) / F_iz
    zi_corr = (obs_zi_raw - b_zi) / F_zi
    zz_corr = (obs_zz_raw - b_zz) / F_zz
    xx_corr = (obs_xx_raw - b_xx) / F_xx
    energy = (ham["II"] + ham["IZ"] * iz_corr + ham["ZI"] * zi_corr
              + ham["ZZ"] * zz_corr + ham["XX"] * xx_corr)
    energy_raw = (ham["II"] + ham["IZ"] * obs_iz_raw + ham["ZI"] * obs_zi_raw
                  + ham["ZZ"] * obs_zz_raw + ham["XX"] * obs_xx_raw)
    return energy, energy_raw, {
        "raw": {"IZ": obs_iz_raw, "ZI": obs_zi_raw,
                "ZZ": obs_zz_raw, "XX": obs_xx_raw},
        "corr": {"IZ": iz_corr, "ZI": zi_corr,
                 "ZZ": zz_corr, "XX": xx_corr},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="Tuna-17")
    parser.add_argument("--shots", type=int, default=8192)
    parser.add_argument("--h2-json", default=None,
                        help="Path to existing H2 diagnostic JSON to re-analyze")
    args = parser.parse_args()

    # --- 1. Find H2 diagnostic JSON ---
    if args.h2_json is None:
        backend_tag = args.backend.lower().replace(" ", "_").replace("-", "")
        cands = sorted(glob.glob(
            f"data/iqm_4platform_validation/h2_diag_{backend_tag}_*.json"))
        if not cands:
            print(f"ERROR: no H2 diagnostic JSON found for {args.backend}")
            sys.exit(1)
        args.h2_json = cands[-1]
    print(f"Re-analyzing H2 JSON: {args.h2_json}")
    with open(args.h2_json) as f:
        h2_data = json.load(f)

    backend_in_data = h2_data["metadata"]["backend"]
    initial_layout = [2, 5] if "tuna-17" in backend_in_data.lower() else (
        [0, 1] if "tuna" in backend_in_data.lower() else None)
    print(f"  layout used: {initial_layout}")

    e_fci = exact_ground_state_energy(H2_HAM)
    print(f"  E_FCI = {e_fci:+.6f} Ha")
    print()

    # --- 2. Run calibration on the same backend / layout ---
    from qiskit_quantuminspire.qi_provider import QIProvider
    backend = QIProvider().get_backend(args.backend)
    print(f"Connected: {backend.name}, running calibration ({args.shots} shots × 4 circuits)")
    obs_data = run_calibration(backend, initial_layout, args.shots)
    print()

    # --- 3. Fit (F, bias) per Pauli ---
    calibration = {
        "IZ": fit_F_bias(obs_data, "IZ", [("00", "ZZ"), ("11", "ZZ")]),
        "ZI": fit_F_bias(obs_data, "ZI", [("00", "ZZ"), ("11", "ZZ")]),
        "ZZ": None,  # both 00 and 11 give truth=+1; need different state
        "XX": fit_F_bias(obs_data, "XX", [("++", "XX"), ("+-", "XX")]),
    }
    # For ZZ, use 00 (truth=+1) and ++ measured in ZZ — but we didn't measure
    # |++> in ZZ basis. Approximation: assume bias_ZZ ≈ 0, F_ZZ from <ZZ>(00).
    F_zz = obs_data[("00", "ZZ")]["obs"]["ZZ"]
    calibration["ZZ"] = (F_zz, 0.0)
    print("Calibration parameters (F, bias):")
    for p, (F, b) in calibration.items():
        print(f"  {p}: F = {F:+.4f},  bias = {b:+.4f}")
    print()

    # --- 4. Re-analyze H2 scan ---
    print("Re-analyzing H2 theta scan with calibration ...")
    rescaled = []
    for entry in h2_data["scan"]:
        theta = entry["theta"]
        zc = entry["zcounts"]
        xc = entry["xcounts"]
        e_corr, e_raw, expvals = correct_energy(zc, xc, calibration, H2_HAM)
        rescaled.append({"theta": theta, "energy_raw": e_raw,
                         "energy_corr": e_corr,
                         "abs_err_raw_mHa": abs(e_raw - e_fci) * 1000,
                         "abs_err_corr_mHa": abs(e_corr - e_fci) * 1000,
                         **expvals})
    # Best theta after correction
    best = min(rescaled, key=lambda r: r["energy_corr"])
    print()
    print("Per-theta comparison (raw vs ABR-corrected):")
    print(f"{'theta':>9} {'E_raw (Ha)':>12} {'|err|_raw':>10} "
          f"{'E_corr (Ha)':>12} {'|err|_corr':>11}")
    for r in rescaled:
        print(f"{r['theta']:+8.3f} {r['energy_raw']:+12.6f} "
              f"{r['abs_err_raw_mHa']:>9.2f} mHa {r['energy_corr']:+12.6f} "
              f"{r['abs_err_corr_mHa']:>9.2f} mHa")
    print()
    print(f"Best theta after ABR: {best['theta']:+.4f} rad")
    print(f"  E_corr = {best['energy_corr']:+.6f} Ha")
    print(f"  E_FCI  = {e_fci:+.6f} Ha")
    print(f"  |err|  = {best['abs_err_corr_mHa']:.2f} mHa  "
          f"({'WITHIN 1 kcal/mol' if best['abs_err_corr_mHa'] <= 1.6 else 'over'})")
    print()
    raw_best = min(rescaled, key=lambda r: r["energy_raw"])
    print(f"Comparison: raw best E = {raw_best['energy_raw']:+.6f}, "
          f"|err| = {raw_best['abs_err_raw_mHa']:.2f} mHa")
    print(f"            corrected best E = {best['energy_corr']:+.6f}, "
          f"|err| = {best['abs_err_corr_mHa']:.2f} mHa")
    print(f"Reduction factor: {raw_best['abs_err_raw_mHa'] / max(best['abs_err_corr_mHa'], 1e-6):.2f}x")

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backend_tag = args.backend.lower().replace(" ", "_").replace("-", "")
    out_path = f"data/iqm_4platform_validation/h2_abr_calib_{backend_tag}_{ts}.json"
    payload = {
        "metadata": {
            "backend": args.backend,
            "initial_layout": initial_layout,
            "shots_per_calib_circuit": args.shots,
            "h2_diag_json": args.h2_json,
            "E_FCI": e_fci,
            "timestamp": ts,
        },
        "calibration_obs": {f"{s}_{b}": v for (s, b), v in obs_data.items()},
        "calibration_params": {p: {"F": float(F), "bias": float(b)}
                                for p, (F, b) in calibration.items()},
        "rescaled_scan": rescaled,
        "raw_best": raw_best,
        "corrected_best": best,
        "improvement_factor": float(raw_best["abs_err_raw_mHa"]
                                    / max(best["abs_err_corr_mHa"], 1e-6)),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
