# Tuna-17 Chemistry Sprint — Plan

**Goal**: push ABR-corrected VQE on Tuna-17 toward 1 kcal/mol (= 1.6 mHa) chemical accuracy on at least one molecule beyond H₂.

**Why Tuna-17 (not Tuna-9)**:
- 17 physical qubits → can pick a 6–8 qubit cluster with low ΔF_anomaly (pair shopping at scale).
- Wider connectivity → frozen-core ansatz of LiH / BeH₂ / H₂O fits without long SWAP chains.
- We are not running QEC. The 17 qubits buy us *selection*, not error correction.

**Non-goals**: full QEC, full UCCSD, FeMoco, anything past ABR depth boundary (~4 CZ).

---

## Phase 0 — Pre-flight (M1, no hardware time)

1. Confirm exact Quantum Inspire backend name (likely `Spin-2+` or `Tuna-17`; check `provider.backends()`).
2. Pull connectivity graph: which physical pairs are coupler-connected.
3. Verify v2 ABR module (`tau_chrono_v2/anomaly.py`) accepts per-pair `(F, bias)` calibration objects.
4. Estimate shot budget — see Section 5.

Deliverable: `experiments/preflight_tuna17.py` that prints backend properties + connectivity map.

---

## Phase 1 — Pair shopping at scale

**Run**: anomalous-weak-value F estimator on every coupler-connected pair (~12–18 pairs depending on topology).

**Per pair**:
- 1 anomaly demo circuit (post-selection, `<Π_0>_w` measurement)
- 1024 shots
- Extract `F_anomaly` and `bias`

**Output**: `results/tuna17_pair_shopping_full.json` with per-pair `(F, bias, ΔF stderr)`.

**Selection rule**: pick the connected sub-graph of 6–8 qubits maximising
```
score(cluster) = mean(F_anomaly) − 2 * std(F_anomaly)
```
i.e. high mean *and* low non-uniformity. Reject any pair with `F_anomaly < 0.65`.

**Stop condition**: if no 6-qubit cluster has `std(F) < 0.05`, stop and report — Tuna-17 is too non-uniform for chemistry beyond H₂ today.

Shot cost: ~18 pairs × 1024 = **~18k shots**.

---

## Phase 2 — Per-pair ABR calibration on the selected cluster

**Run**: 4-Pauli calibration sweep (`I, X, Y, Z` Pauli-basis observables) on each pair in the selected cluster. This is what v2 already does globally — here we make it per-pair.

**Output**: `tau_chrono_v2/calibration/tuna17_<cluster_id>.json`.

Shot cost: 4 Paulis × 6 pairs × 4096 shots ≈ **~100k shots**.

---

## Phase 3 — H₂ anchor

**Run**: H₂ sto-3g, 7 bond lengths (R = 0.5 to 2.5 Å, step 0.33 Å), 2-qubit ansatz on the best pair.

- ansatz: hardware-efficient, depth 2 (1 CZ layer + 2 single-qubit layers), well within ABR boundary
- 100 VQE iterations, 4096 shots each
- both raw and ABR-corrected energies recorded

**Pass criterion**: ≥ 5/7 R points within 1.6 mHa of FCI.

If H₂ does not pass on Tuna-17, we stop and debug. No point pushing larger molecules.

Shot cost: 7 R × 100 iter × 4096 shots ≈ **~3M shots**.

---

## Phase 4 — Scale to LiH / BeH₂ / H₂O frozen-core

Order matters — go in increasing qubit count, stop at the first failure:

| Molecule | Qubits | Frozen-core? | Ansatz depth target | Pass criterion |
|---|---|---|---|---|
| LiH        | 4 | yes (1s frozen) | ≤ 3 CZ layers | ≥ 4/5 R within 1.6 mHa |
| BeH₂       | 6 | yes (1s frozen) | ≤ 3 CZ layers | ≥ 4/5 R within 1.6 mHa |
| H₂O        | 6 | aggressive frozen-core (drop virtual) | ≤ 4 CZ layers | ≥ 3/5 R within 1.6 mHa |

For each molecule:
- 5 R points sampling the dissociation curve
- 100 VQE iterations × 4096 shots
- record raw + ABR-corrected energies
- per-pair calibration from Phase 2 plugged into ABR

Shot cost per molecule: 5 R × 100 iter × 4096 shots ≈ **~2M shots**.

Total Phase 4: **~6M shots** if we hit all three.

---

## Phase 5 — Reporting

**Single output JSON**: `results/tuna17_chemistry_sprint.json` with schema
```json
{
  "backend": "...",
  "cluster": [q_id, ...],
  "calibration": { ... },
  "molecules": {
    "H2":  { "R": [...], "E_FCI": [...], "E_raw": [...], "E_abr": [...], "abs_err_mHa": [...], "hits_1.6mHa": k },
    "LiH": { ... },
    ...
  }
}
```

**Headline figure**: `fig_tuna17_chemistry.png` — 2×2 panel, each molecule, error bar plot of `(E - E_FCI) / mHa` vs R, raw vs ABR vs ±1.6 mHa band.

**No claims**. Just the numbers and the figure.

---

## Total resource estimate

- Hardware shots: **~9M shots** total → at typical Quantum Inspire Tuna-17 throughput (≈ 200 shots/sec including queue), this is **~13 hours** wall-clock.
- M1 prep + analysis: ~1 day.
- Free credit usage: should fit within 30 credits/month (1 credit ≈ 1M shots equivalent on QI).

---

## Risk register (honest)

1. **Tuna-17 too non-uniform** (Phase 1 fails): expected probability ~30% given v2's `ΔF = 0.22` over 3 pairs. Mitigation: report negative result, push experiment to a less non-uniform Spin-2+ revision when available.
2. **H₂ fails on Tuna-17 even with ABR** (Phase 3 fails): ~10%. Would mean Tuna-17 has worse 2-qubit gates than Tuna-9. Falsifies "more qubits → more selection freedom → better chemistry" thesis. Document and stop.
3. **BeH₂ / H₂O depth exceeds ABR boundary** (Phase 4 partial fail): ~50%. This is fine — it just maps the boundary on a real chip and sets the v3 problem ("how to extend ABR past depth 4").

---

## What this plan does NOT claim

- Does not claim QEC on Tuna-17.
- Does not claim distance-3 surface code.
- Does not claim universal chemical accuracy.
- Does not claim extrapolation to FeMoco / drug-design molecules.

The plan claims only: *with chip-aware pair selection + per-pair ABR, what is the largest molecule on Tuna-17 today within 1.6 mHa of FCI?*

That is the experiment.
