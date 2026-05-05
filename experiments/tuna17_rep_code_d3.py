#!/usr/bin/env python3
"""
Phase A1: distance-3 repetition code memory experiment on Tuna-17.

Tests whether a logical bit (bit-flip protected) can be born on Tuna-17:
the logical error rate after R rounds of stabilizer extraction should be
LOWER than a single physical |0> qubit's idle decay rate over the same
wall time, if QEC is helping.

Code:
    Logical |0>_L = |000>, |1>_L = |111>
    Stabilizers: Z_d0 Z_d1, Z_d1 Z_d2
    Distance 3 (corrects 1 X error)

Chain (selected from Phase 1 F_anomaly map, mean F=0.707):
    physical: q0 - q1 - q4 - q2 - q5
    role:     d0 - a0 - d1 - a1 - d2
    edges:    F=0.746, F=0.605, F=0.676, F=0.799

Each round uses 4 CX gates + 2 ancilla measurements + 2 ancilla resets.
Final destructive readout of 3 data qubits.

Decoder: stim + pymatching (standard).

Usage:
    .venv/bin/python experiments/tuna17_rep_code_d3.py --backend "QX emulator" --rounds 1 --shots 1024  # smoke
    .venv/bin/python experiments/tuna17_rep_code_d3.py --backend "Tuna-17" --rounds 1 3 5 10 --shots 4096
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import stim
import pymatching

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.compiler import transpile

# Chain selection from Phase 1 (highest mean F linear path of length 5)
CHAIN = {
    "physical": [0, 1, 4, 2, 5],          # q0, q1, q4, q2, q5
    "role":     ["d0", "a0", "d1", "a1", "d2"],
    "F_edges":  [0.746, 0.605, 0.676, 0.799],  # for reference
}


def build_rep_code_circuit_qiskit(rounds: int) -> QuantumCircuit:
    """Build d=3 repetition code memory experiment in qiskit.
    Logical |0>_L preparation, R rounds of ZZ stabilizer extraction with
    mid-circuit measurement + reset, final destructive Z readout.

    Logical qubits ordered: d0, d1, d2, a0, a1.
    initial_layout will map logical [0..4] to physical chain qubits.
    """
    n_data, n_anc = 3, 2
    qr_d = QuantumRegister(n_data, "d")
    qr_a = QuantumRegister(n_anc, "a")
    cr_anc = ClassicalRegister(n_anc * rounds, "syn")
    cr_data = ClassicalRegister(n_data, "fin")
    qc = QuantumCircuit(qr_d, qr_a, cr_anc, cr_data)

    # Initial state: all data + ancilla in |0> (default)

    for r in range(rounds):
        # Stabilizer 1: Z_d0 Z_d1 -> ancilla a0
        qc.cx(qr_d[0], qr_a[0])
        qc.cx(qr_d[1], qr_a[0])
        # Stabilizer 2: Z_d1 Z_d2 -> ancilla a1
        qc.cx(qr_d[1], qr_a[1])
        qc.cx(qr_d[2], qr_a[1])
        # Measure both ancillas (mid-circuit)
        qc.measure(qr_a[0], cr_anc[r * 2 + 0])
        qc.measure(qr_a[1], cr_anc[r * 2 + 1])
        if r < rounds - 1:
            # Reset for next round
            qc.reset(qr_a[0])
            qc.reset(qr_a[1])
        qc.barrier()

    # Final destructive readout of data qubits
    qc.measure(qr_d[0], cr_data[0])
    qc.measure(qr_d[1], cr_data[1])
    qc.measure(qr_d[2], cr_data[2])
    return qc


def build_stim_circuit_for_decoder(rounds: int) -> stim.Circuit:
    """Use stim's built-in distance-3 repetition code memory circuit.

    The generator places small placeholder error rates on every gate,
    which are required for the detector_error_model() to have boundary
    edges that PyMatching needs. The actual error rates do not affect
    decoding outcome (only the matching-graph topology), so we use a
    uniform 0.01 to keep the graph well-conditioned.

    The qiskit hardware circuit (build_rep_code_circuit_qiskit) follows
    exactly the same structure: prepare 3 data qubits in |0>, alternate
    between data and ancilla in a chain, do `rounds` ZZ stabilizer
    extractions with mid-circuit measure+reset, then final destructive
    readout of the 3 data qubits."""
    return stim.Circuit.generated(
        "repetition_code:memory",
        distance=3,
        rounds=rounds,
        after_clifford_depolarization=0.01,
        before_round_data_depolarization=0.01,
        before_measure_flip_probability=0.01,
        after_reset_flip_probability=0.01,
    )


def parse_counts_to_shots(counts: dict, rounds: int):
    """Parse qiskit counts into stim-style shot arrays.

    Qiskit returns bitstrings as 'fin syn' separated by space (multiple
    classical registers). Within each register, leftmost char is the
    highest-indexed clbit. Need to extract syndrome history and final
    data bits in chronological measurement order.

    Returns:
      meas_array : shape (n_shots, 2*rounds + 3), each row is the
                   measurement history in the order they were measured:
                   [a0_round0, a1_round0, a0_round1, a1_round1, ...,
                    d0_final, d1_final, d2_final]
                   This matches stim's measurement record order.
    """
    rows = []
    for bs, n in counts.items():
        bs_clean = bs.replace(" ", "")
        # Total clbits = 2*rounds + 3
        total = 2 * rounds + 3
        if len(bs_clean) < total:
            bs_clean = bs_clean.zfill(total)
        # In qiskit, the leftmost char is the highest-index classical bit.
        # Classical registers were declared in order: cr_anc (size 2*rounds),
        # cr_data (size 3). So the bitstring is "data...syn..." with data
        # on the LEFT (higher indices) and syndromes on the RIGHT (lower).
        # Within syndromes (cr_anc): order is r0_a0, r0_a1, r1_a0, r1_a1, ...
        # at clbit indices 0, 1, 2, 3, ...; bs_clean[-1] = clbit 0.
        # Within data: order is d0, d1, d2 at clbit indices 0, 1, 2 of cr_data.
        syn_bits = []
        for r in range(rounds):
            for j in range(2):  # a0, a1
                idx = r * 2 + j
                bit = int(bs_clean[-(idx + 1)])
                syn_bits.append(bit)
        data_bits = []
        for j in range(3):
            idx = 2 * rounds + j
            bit = int(bs_clean[-(idx + 1)])
            data_bits.append(bit)
        meas = syn_bits + data_bits
        for _ in range(n):
            rows.append(meas)
    return np.array(rows, dtype=np.uint8)


def decode_and_count_logical_errors(meas_array: np.ndarray, rounds: int):
    """Use stim + pymatching to decode shots, count logical errors."""
    stim_circuit = build_stim_circuit_for_decoder(rounds)
    dem = stim_circuit.detector_error_model(decompose_errors=False)
    matching = pymatching.Matching.from_detector_error_model(dem)

    # Convert measurement array -> detector array
    # (stim has a helper for this)
    detector_array = np.zeros((meas_array.shape[0], stim_circuit.num_detectors),
                              dtype=np.uint8)
    obs_array = np.zeros((meas_array.shape[0], stim_circuit.num_observables),
                         dtype=np.uint8)

    n_meas = stim_circuit.num_measurements
    if meas_array.shape[1] != n_meas:
        raise ValueError(f"Measurement count mismatch: got {meas_array.shape[1]}, "
                         f"stim expects {n_meas}")

    sampler = stim_circuit.compile_m2d_converter()
    # dtype=bool_ tells stim the data is unpacked (1 byte per bit).
    detector_array, obs_array = sampler.convert(
        measurements=meas_array.astype(np.bool_),
        separate_observables=True)

    # Decode each shot
    predictions = matching.decode_batch(detector_array)
    # Logical error if predicted obs != actual obs (here actual is 0 since
    # we prepped |0>_L; the 'observable' = data parity should be 0)
    logical_errors = (predictions[:, 0] != obs_array[:, 0]).sum()
    n_shots = meas_array.shape[0]
    p_logical = logical_errors / n_shots
    return logical_errors, n_shots, p_logical


def majority_vote_logical(meas_array: np.ndarray, rounds: int):
    """Simpler decoder: majority vote on the 3 final data bits.
    Returns logical_errors, n_shots, p_logical."""
    final_data = meas_array[:, 2 * rounds:2 * rounds + 3]
    parity_sum = final_data.sum(axis=1)
    # Majority vote: parity_sum >= 2 -> logical 1, else logical 0
    decoded = (parity_sum >= 2).astype(int)
    # Expected logical: 0
    logical_errors = decoded.sum()
    return logical_errors, meas_array.shape[0], logical_errors / meas_array.shape[0]


def run_one_rounds(backend, rounds: int, shots: int):
    """Run rep-code memory experiment with `rounds` syndrome rounds."""
    qc = build_rep_code_circuit_qiskit(rounds)
    # Logical layout: [d0, d1, d2, a0, a1] -> physical [q0, q4, q5, q1, q2]
    initial_layout = [0, 4, 5, 1, 2]
    transpiled = transpile(qc, backend, initial_layout=initial_layout,
                           optimization_level=0)
    n_cz = sum(1 for i in transpiled.data if i.operation.name == "cz")
    n_cx = sum(1 for i in transpiled.data if i.operation.name == "cx")
    print(f"  rounds={rounds}: depth={transpiled.depth()}, CX={n_cx}, CZ={n_cz}")

    job = backend.run(transpiled, shots=shots)
    job.wait_for_final_state(timeout=1800)
    counts = job.result().get_counts(0)

    meas_array = parse_counts_to_shots(counts, rounds)

    # Two decoders (sanity cross-check)
    n_err_pm, n_shots, p_pm = decode_and_count_logical_errors(meas_array, rounds)
    n_err_mv, _, p_mv = majority_vote_logical(meas_array, rounds)
    return {
        "rounds": rounds,
        "shots": n_shots,
        "depth": transpiled.depth(),
        "n_cx": n_cx,
        "n_cz": n_cz,
        "logical_errors_pymatching": int(n_err_pm),
        "p_logical_pymatching": float(p_pm),
        "logical_errors_majvote": int(n_err_mv),
        "p_logical_majvote": float(p_mv),
        "raw_counts": dict(counts),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="QX emulator")
    parser.add_argument("--rounds", type=int, nargs="+", default=[1])
    parser.add_argument("--shots", type=int, default=4096)
    args = parser.parse_args()

    from qiskit_quantuminspire.qi_provider import QIProvider
    backend = QIProvider().get_backend(args.backend)
    print(f"Connected: {backend.name} ({backend.num_qubits} qubits)")
    print(f"Chain: physical qubits {CHAIN['physical']} (roles {CHAIN['role']})")
    print(f"Edge F_anomaly: {CHAIN['F_edges']}, mean = {sum(CHAIN['F_edges'])/4:.3f}")
    print()

    results = {}
    t0 = time.time()
    for r in args.rounds:
        print(f"=== {r} rounds ===")
        try:
            result = run_one_rounds(backend, r, args.shots)
            results[f"R{r}"] = result
            print(f"  PyMatching: {result['logical_errors_pymatching']}/{result['shots']} "
                  f"= {result['p_logical_pymatching']:.4f}")
            print(f"  MajorVote : {result['logical_errors_majvote']}/{result['shots']} "
                  f"= {result['p_logical_majvote']:.4f}")
            top = dict(sorted(result['raw_counts'].items(), key=lambda x: -x[1])[:5])
            print(f"  Top 5 counts: {top}")
        except Exception as e:
            results[f"R{r}"] = {"status": "fail", "error": str(e)}
            print(f"  FAILED: {e}")
        print()

    elapsed = time.time() - t0

    os.makedirs("data/iqm_4platform_validation", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backend_tag = args.backend.lower().replace(" ", "_").replace("-", "")
    out_path = f"data/iqm_4platform_validation/repcode_d3_{backend_tag}_{ts}.json"
    payload = {
        "metadata": {
            "backend": args.backend,
            "shots_per_run": args.shots,
            "rounds": args.rounds,
            "chain": CHAIN,
            "timestamp": ts,
            "elapsed_sec": elapsed,
        },
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"Saved: {out_path}")
    print(f"Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
