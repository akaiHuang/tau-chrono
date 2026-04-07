#!/usr/bin/env python3
"""
IQM Garnet — BV Depth Scaling (tau-chrono Mode 1 validation)

SYNTHESIS of 3 expert perspectives (physicist + benchmark engineer + skeptic):

Design decisions:
  - Backend: Garnet (Square 20q) — not Sirius (Star topology confounds comparison)
  - Depths: 5, 15, 30, 50, 75 — saturation effect only emerges at depth ≥ 30
  - Task framing: PREDICTION accuracy (f_naive vs f_tauchrono vs f_measured)
                  NOT error correction (Mitiq ZNE is different task)
  - Pre-registration: predictions committed to git BEFORE submitting
  - Budget: ~6-8 credits first run, keep buffer for reruns

Hypothesis:
  - Tuna-9 showed 2.7% at depth 2, 50% at depth 40
  - IQM (4× cleaner CNOT error) → saturation at depth ~4× deeper
  - Expected on Garnet: tau_error < 0.5× naive_error at depth ≥ 30

Failure modes (both still publishable):
  - tau-chrono wins by expected margin → cross-architecture validation ✓
  - tau-chrono systematically overshoots → IQM has non-depolarizing noise (coherent/ZZ)
    Pivot: "tau-chrono exposes non-depolarizing structure"

Author: Sheng-Kai Huang
"""

import warnings; warnings.filterwarnings('ignore')
import math
import sys
import json
import time
import hashlib
from pathlib import Path

# tau_chrono package
sys.path.insert(0, '/Users/akaihuangm1/Desktop/github/tau-chrono')
from tau_chrono.api import predict_gates

from iqm.iqm_client import (
    IQMClient, Circuit, CircuitOperation, validate_circuit,
)


# ============================================================
# CONFIG — COMMIT THIS TO GIT BEFORE SUBMITTING
# ============================================================
BACKEND = 'Garnet'
SHOTS = 2048                          # scaled for ~1% stat error
SECRET = '101'                        # 3-bit secret (2 ones → 2 CNOTs/round)
DEPTHS = [5, 15, 30, 50, 75]         # saturation regime scan
DATA_QUBITS = ['QB5', 'QB1', 'QB9']  # all have direct CZ to QB4 (verified)
ANCILLA_QUBIT = 'QB4'                # hub qubit with 3+ connections

# IQM-published nominal gate errors
GATE_ERRORS_IQM = {
    'h':  0.001,    # single-qubit ~0.1%
    'x':  0.001,
    'cx': 0.010,    # two-qubit ~1%
    'cz': 0.010,
}

# Pre-registered success criterion
SUCCESS_CRITERION = {
    'primary': 'At depth >= 30: tau_error < 0.5 * naive_error',
    'secondary': 'tau-chrono wins at least 3/5 depth points',
    'null_hypothesis': 'f_tauchrono has zero prediction advantage over f_naive',
}


# ============================================================
# Native gate builders (IQM: prx, cz, measure)
# ============================================================

def H(q):
    """Hadamard = X · Ry(π/2) via two prx gates.

    Verification:
      Ry(π/2) = (1/√2)[[1,-1],[1,1]]
      X       = [[0,1],[1,0]]
      X · Ry(π/2) = (1/√2)[[1,1],[1,-1]] = H ✓

    Single prx(π/2, π/2) = Ry(π/2) is NOT a Hadamard
    because Ry(π/2)² = Ry(π) = iY ≠ I.
    """
    return [
        CircuitOperation(name='prx', locus=(q,),
                         args={'angle': math.pi / 2, 'phase': math.pi / 2}),  # Ry(π/2)
        CircuitOperation(name='prx', locus=(q,),
                         args={'angle': math.pi, 'phase': 0.0}),              # X
    ]


def X(q):
    """Pauli-X via prx(π, 0)."""
    return CircuitOperation(
        name='prx', locus=(q,),
        args={'angle': math.pi, 'phase': 0.0},
    )


def CZ(q1, q2):
    """CZ gate."""
    return CircuitOperation(name='cz', locus=(q1, q2), args={})


def M(q, key):
    """Measurement."""
    return CircuitOperation(name='measure', locus=(q,), args={'key': key})


def CNOT(control, target):
    """CNOT = H(target) · CZ · H(target). H now returns a list."""
    return [*H(target), CZ(control, target), *H(target)]


# ============================================================
# BV Circuit Builder
# ============================================================

def build_bv(secret, depth_mult, data_qubits, ancilla):
    """Build Bernstein-Vazirani with configurable oracle depth.

    depth_mult = 1 means 1 oracle round (minimum BV)
    depth_mult = N means N oracle rounds (for testing noise scaling)
    """
    ops = []

    # Stage 1: H on data qubits
    for q in data_qubits:
        ops.extend(H(q))

    # Stage 2: Prep ancilla in |−⟩
    ops.append(X(ancilla))
    ops.extend(H(ancilla))

    # Stage 3: Oracle repeated depth_mult times
    for _ in range(depth_mult):
        for i, bit in enumerate(secret):
            if bit == '1':
                ops.extend(CNOT(data_qubits[i], ancilla))

    # Stage 4: H on data qubits
    for q in data_qubits:
        ops.extend(H(q))

    # Stage 5: Measure data qubits
    for i, q in enumerate(data_qubits):
        ops.append(M(q, f'm{i}'))

    return Circuit(name=f'bv_d{depth_mult}', instructions=tuple(ops))


def gate_sequence_for_prediction(secret, depth_mult):
    """Gate sequence for tau_chrono.predict_gates (standard gate names).

    Each logical H = X · Ry(π/2) = 2 prx gates (counted as 2 'h' for tau_chrono).
    """
    n_data = len(secret)
    n_ones = secret.count('1')

    gates = []
    gates.extend(['h', 'h'] * n_data)               # H on data (2 prx each)
    gates.append('x')                                # X ancilla
    gates.extend(['h', 'h'])                         # H ancilla (2 prx)
    for _ in range(depth_mult):                      # oracle (CNOT = H + CZ + H)
        for _ in range(n_ones):
            gates.extend(['h', 'h', 'cz', 'h', 'h'])
    gates.extend(['h', 'h'] * n_data)               # final H on data
    return gates


# ============================================================
# Pre-registration
# ============================================================

def compute_predictions():
    """Generate all predictions BEFORE submitting to hardware.

    Returns a dict that gets written to git-tracked file.
    Reviewers can verify predictions were made before measurement.
    """
    predictions = []
    for d in DEPTHS:
        gates = gate_sequence_for_prediction(SECRET, d)
        pred = predict_gates(gates, GATE_ERRORS_IQM)
        predictions.append({
            'depth': d,
            'n_gates_total': len(gates),
            'n_gates_1q': gates.count('h') + gates.count('x'),
            'n_gates_2q': gates.count('cz'),
            'f_naive': pred.f_naive,
            'f_tauchrono': pred.f_tauchrono,
            'saturation_bonus_pct': (pred.f_tauchrono - pred.f_naive) / pred.f_naive * 100,
        })
    return predictions


def pre_register(predictions, output_file):
    """Write predictions with timestamp + hash, then exit."""
    payload = {
        'preregistration_timestamp': time.time(),
        'timestamp_readable': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'backend': BACKEND,
        'shots': SHOTS,
        'secret': SECRET,
        'depths': DEPTHS,
        'data_qubits': DATA_QUBITS,
        'ancilla_qubit': ANCILLA_QUBIT,
        'gate_errors_iqm': GATE_ERRORS_IQM,
        'success_criterion': SUCCESS_CRITERION,
        'predictions': predictions,
    }
    content = json.dumps(payload, indent=2, sort_keys=True)
    payload_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
    payload['preregistration_hash'] = payload_hash

    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    return payload_hash


# ============================================================
# Main
# ============================================================

def main(submit=False):
    env_file = Path('/Users/akaihuangm1/Desktop/github/tau-chrono/.iqm_env')
    env = {}
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            env[k.strip()] = v.strip().strip('"').strip("'")

    print('=' * 78)
    print(f'  IQM {BACKEND} — BV DEPTH SCALING (tau-chrono Mode 1 Cross-Architecture)')
    print('=' * 78)
    print(f'  Backend:    {BACKEND} (Square topology, ~1% CNOT error)')
    print(f'  Data:       {DATA_QUBITS}')
    print(f'  Ancilla:    {ANCILLA_QUBIT}')
    print(f'  Secret:     {SECRET}')
    print(f'  Depths:     {DEPTHS}')
    print(f'  Shots:      {SHOTS} per circuit')
    print(f'  Circuits:   {len(DEPTHS)} in one batch')
    print('=' * 78)

    # Pre-register predictions (before any hardware access)
    predictions = compute_predictions()
    prereg_file = Path(__file__).parent.parent / 'results' / f'prereg_{BACKEND.lower()}_bv.json'
    prereg_hash = pre_register(predictions, prereg_file)

    print(f'\n[Pre-Registration]')
    print(f'  File: {prereg_file}')
    print(f'  Hash: {prereg_hash}')
    print(f'  ⚠️  COMMIT THIS FILE TO GIT BEFORE SUBMITTING TO HARDWARE')

    # Build circuits
    circuits = []
    print(f'\n[Circuit Design + Predictions]')
    print(f'  {"depth":>6} {"n_gates":>9} {"1q":>5} {"2q":>5} {"f_naive":>10} {"f_tau":>10} {"Δ (sat)":>10}')
    print(f'  {"-" * 66}')

    for pred in predictions:
        circ = build_bv(SECRET, pred['depth'], DATA_QUBITS, ANCILLA_QUBIT)
        try:
            validate_circuit(circ)
        except Exception as e:
            print(f'  ❌ d={pred["depth"]} invalid: {e}')
            return
        circuits.append(circ)

        print(f'  {pred["depth"]:>6} {pred["n_gates_total"]:>9} '
              f'{pred["n_gates_1q"]:>5} {pred["n_gates_2q"]:>5} '
              f'{pred["f_naive"]:>10.4f} {pred["f_tauchrono"]:>10.4f} '
              f'{pred["saturation_bonus_pct"]:>+9.2f}%')

    # Estimate budget
    n_2q_total = sum(p['n_gates_2q'] for p in predictions)
    print(f'\n[Budget Estimate]')
    print(f'  Batch overhead:   ~0.6 credits (one job for all {len(DEPTHS)} circuits)')
    print(f'  Execution:        ~{len(DEPTHS) * 0.3:.1f}-{len(DEPTHS) * 0.8:.1f} credits')
    print(f'  Total:            ~{0.6 + len(DEPTHS) * 0.3:.1f}-{0.6 + len(DEPTHS) * 0.8:.1f} credits')
    print(f'  Reserve:          keep 20+ credits for reruns/reviewer response')

    print(f'\n[Pre-registered Success Criterion]')
    for k, v in SUCCESS_CRITERION.items():
        print(f'  {k}: {v}')

    if not submit:
        print(f'\n💡 DRY RUN. To actually submit:')
        print(f'   1. git add {prereg_file}')
        print(f'   2. git commit -m "Pre-register tau-chrono IQM {BACKEND} predictions"')
        print(f'   3. python {Path(__file__).name} --submit')
        return

    # ==================== SUBMIT ====================
    print(f'\n→ Submitting {len(circuits)} circuits to {BACKEND}...')
    url_key = f'IQM_{BACKEND.upper()}_URL'
    if url_key not in env:
        print(f'❌ {url_key} not found in .iqm_env')
        return
    client = IQMClient(iqm_server_url=env[url_key], token=env['IQM_TOKEN'])

    t_submit = time.time()
    try:
        job = client.submit_circuits(circuits=circuits, shots=SHOTS)
        print(f'  ✅ Job: {job.job_id}')
    except Exception as e:
        print(f'  ❌ Submit failed: {type(e).__name__}: {e}')
        return

    # Wait for completion
    print(f'\n→ Waiting for completion...')
    t0 = time.time()
    for _ in range(240):  # up to 20 min
        try:
            info = client.get_job(job.job_id)
            status = str(getattr(info, 'status', '?')).lower()
            elapsed = time.time() - t0
            print(f'  [{elapsed:6.1f}s] {status}')
            if 'completed' in status or 'ready' in status:
                break
            if 'failed' in status or 'aborted' in status:
                print(f'  ❌ Failed')
                return
        except Exception as e:
            print(f'  Error: {e}')
        time.sleep(5)

    # Fetch results
    print(f'\n→ Fetching measurements...')
    try:
        measurements = client.get_job_measurements(job.job_id)
    except Exception as e:
        print(f'  ❌ Error: {e}')
        return

    # Analyze
    print(f'\n[Results]')
    print(f'  {"depth":>6} {"measured":>10} {"f_naive":>10} {"f_tau":>10} '
          f'{"naive_err":>11} {"tau_err":>10} {"stat_err":>10} {"winner":>8}')
    print(f'  {"-" * 82}')

    results = []
    naive_wins = 0
    tau_wins = 0
    deep_naive_err = []
    deep_tau_err = []

    for pred, meas in zip(predictions, measurements):
        shots_data = meas.measurements
        success = 0
        for shot in shots_data:
            bits = ''.join(str(shot.get(f'm{i}', [0])[0]) for i in range(len(SECRET)))
            if bits == SECRET:
                success += 1

        f_measured = success / len(shots_data)
        naive_err = abs(pred['f_naive'] - f_measured)
        tau_err = abs(pred['f_tauchrono'] - f_measured)
        # Binomial stat error
        stat_err = math.sqrt(f_measured * (1 - f_measured) / len(shots_data))

        if tau_err < naive_err:
            winner, tau_wins = 'tau ✓', tau_wins + 1
        elif naive_err < tau_err:
            winner, naive_wins = 'naive', naive_wins + 1
        else:
            winner = 'tie'

        # Collect deep-circuit errors for primary criterion
        if pred['depth'] >= 30:
            deep_naive_err.append(naive_err)
            deep_tau_err.append(tau_err)

        print(f'  {pred["depth"]:>6} {f_measured:>10.4f} {pred["f_naive"]:>10.4f} '
              f'{pred["f_tauchrono"]:>10.4f} {naive_err:>11.4f} {tau_err:>10.4f} '
              f'±{stat_err:>8.4f} {winner:>8}')

        results.append({
            'depth': pred['depth'],
            'f_measured': f_measured,
            'f_naive': pred['f_naive'],
            'f_tauchrono': pred['f_tauchrono'],
            'naive_error': naive_err,
            'tau_error': tau_err,
            'stat_error': stat_err,
            'winner': winner,
        })

    # Evaluate pre-registered criteria
    print(f'\n[Pre-Registered Criteria Check]')
    if deep_naive_err:
        avg_deep_naive = sum(deep_naive_err) / len(deep_naive_err)
        avg_deep_tau = sum(deep_tau_err) / len(deep_tau_err)
        primary_met = avg_deep_tau < 0.5 * avg_deep_naive
        print(f'  PRIMARY (depth≥30, tau<0.5×naive):')
        print(f'    avg naive_err (deep): {avg_deep_naive:.4f}')
        print(f'    avg tau_err (deep):   {avg_deep_tau:.4f}')
        print(f'    tau/naive ratio:      {avg_deep_tau / avg_deep_naive:.2f}')
        print(f'    {"✅ PASS" if primary_met else "❌ FAIL"}')

    secondary_met = tau_wins >= 3
    print(f'  SECONDARY (tau wins ≥3/5): tau={tau_wins}, naive={naive_wins}')
    print(f'    {"✅ PASS" if secondary_met else "❌ FAIL"}')

    # Save
    out_file = Path(__file__).parent.parent / 'results' / f'iqm_{BACKEND.lower()}_bv_depth_scan.json'
    with open(out_file, 'w') as f:
        json.dump({
            'job_id': job.job_id,
            'backend': BACKEND,
            'preregistration_hash': prereg_hash,
            'predictions': predictions,
            'results': results,
            'summary': {
                'tau_wins': tau_wins,
                'naive_wins': naive_wins,
                'avg_deep_naive_err': avg_deep_naive if deep_naive_err else None,
                'avg_deep_tau_err': avg_deep_tau if deep_tau_err else None,
                'primary_criterion_met': primary_met if deep_naive_err else False,
                'secondary_criterion_met': secondary_met,
            },
        }, f, indent=2)
    print(f'\n  Saved: {out_file}')


if __name__ == '__main__':
    import sys
    submit = '--submit' in sys.argv
    main(submit=submit)
