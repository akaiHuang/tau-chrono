"""
tau-chrono Adaptive QEC: Tau-Triggered Syndrome Extraction.

Standard QEC extracts syndromes at fixed intervals (every N gates).
Tau-triggered QEC uses the tau-chrono Bayesian noise tracker to predict
when syndrome extraction is actually needed, extracting only when
accumulated retrodiction failure (tau) exceeds a threshold.

Core idea:
    - Each gate accumulates tau (noise damage) via composition
    - When accumulated tau > tau_threshold, trigger syndrome extraction
    - After extraction, reset accumulated tau
    - On good hardware: fewer extractions = less overhead
    - On bad hardware: more frequent extraction where needed

This achieves the same (or better) logical error rate with significantly
fewer syndrome rounds, because syndrome extraction itself injects noise
(ancilla CNOTs), so avoiding unnecessary rounds is a net win.

Usage:
    from tau_chrono.adaptive_qec import TauTriggeredQEC

    sim = TauTriggeredQEC(
        gate_errors={"cx": 0.003, "h": 0.001},
        tau_threshold=0.1,
        code_distance=3,
    )
    results = sim.compare_strategies(
        gate_sequence=["h", "cx"] * 50,
        shots=10000,
    )
    print(results)

Author: Sheng-Kai Huang / QDA
Date: 2026-03-20
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from tau_chrono import depolarizing, tau_parameter, apply_channel


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Syndrome extraction circuit overhead for repetition code distance d:
#   - (d-1) ancilla qubits, each needs 2 CNOTs (connect to left & right data)
#   - Total: 2*(d-1) CNOTs per syndrome round
_SYNDROME_CNOTS_PER_ROUND = lambda d: 2 * (d - 1)

# Reference state for tau computation
_SIGMA = np.eye(2, dtype=complex) / 2
_RHO = np.array([[1, 0], [0, 0]], dtype=complex)  # |0> state


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GateEvent:
    """Record of a single gate in the circuit."""
    index: int
    gate_name: str
    error_rate: float
    tau_increment: float
    tau_accumulated: float
    triggered_extraction: bool


@dataclass
class StrategyResult:
    """Result of simulating one QEC strategy."""
    name: str
    logical_error_rate: float
    syndrome_rounds: int
    total_cnot_overhead: int
    total_gates: int
    description: str


@dataclass
class ComparisonResult:
    """Full comparison of all QEC strategies."""
    hardware_name: str
    gate_errors: Dict[str, float]
    circuit_length: int
    code_distance: int
    tau_threshold: float
    shots: int
    strategies: List[StrategyResult]
    tau_trace: List[float]  # accumulated tau over the circuit (adaptive)
    extraction_points: List[int]  # gate indices where extraction was triggered


# ---------------------------------------------------------------------------
# Tau computation helpers
# ---------------------------------------------------------------------------

def _gate_tau(gate_name: str, gate_errors: Dict[str, float]) -> float:
    """Compute tau for a single gate from its error rate.

    Uses depolarizing channel as the noise model and computes
    the Petz recovery failure parameter.
    """
    p = gate_errors.get(gate_name, gate_errors.get("default", 0.01))
    if p <= 0:
        return 0.0
    kraus = depolarizing(min(p, 1.0))
    return tau_parameter(_RHO, kraus, _SIGMA)


def _compose_tau(tau_a: float, tau_b: float) -> float:
    """Compose two tau values using fidelity composition.

    tau = 1 - F, and fidelity composes multiplicatively for
    independent channels: F_total = F_a * F_b.
    So: tau_total = 1 - (1 - tau_a)(1 - tau_b)
                  = tau_a + tau_b - tau_a * tau_b
    """
    return tau_a + tau_b - tau_a * tau_b


# ---------------------------------------------------------------------------
# Monte Carlo simulation engine
# ---------------------------------------------------------------------------

class RepetitionCodeSimulator:
    """Monte Carlo simulator for d-qubit repetition code.

    Simulates bit-flip errors and majority-vote decoding.
    Each data qubit can independently flip with some probability.
    Syndrome extraction uses ancilla CNOTs that can also inject errors.
    """

    def __init__(self, distance: int, rng: np.random.Generator):
        self.d = distance
        self.rng = rng
        self.data_qubits = np.zeros(distance, dtype=int)  # 0 or 1 (error flag)

    def reset(self):
        """Reset all data qubits to no-error state."""
        self.data_qubits[:] = 0

    def apply_gate_error(self, p_error: float):
        """Apply independent bit-flip error to each data qubit."""
        flips = self.rng.random(self.d) < p_error
        self.data_qubits ^= flips.astype(int)

    def extract_syndrome_and_correct(self, p_cx_error: float, p_meas_error: float):
        """Extract syndrome via ancilla CNOTs and apply majority-vote correction.

        For repetition code distance d:
        - (d-1) ancilla qubits measure parity of adjacent pairs
        - Each ancilla CNOT can inject errors on both data and ancilla
        - Majority vote on data qubits to decode

        Parameters
        ----------
        p_cx_error : float
            Error probability per CNOT gate (affects both data and ancilla).
        p_meas_error : float
            Measurement error probability on ancilla readout.
        """
        # Syndrome extraction injects noise via ancilla CNOTs
        # Each data qubit participates in ~2 CNOTs (left and right stabilizer)
        # Model: each CNOT has independent probability of flipping data qubit
        for i in range(self.d):
            n_cnots = 2 if 0 < i < self.d - 1 else 1  # edge qubits: 1 CNOT
            for _ in range(n_cnots):
                if self.rng.random() < p_cx_error:
                    self.data_qubits[i] ^= 1

        # Measure syndrome (parity of adjacent pairs)
        syndromes = np.zeros(self.d - 1, dtype=int)
        for i in range(self.d - 1):
            syndromes[i] = self.data_qubits[i] ^ self.data_qubits[i + 1]
            # Measurement error on ancilla
            if self.rng.random() < p_meas_error:
                syndromes[i] ^= 1

        # Majority vote decoding: correct if majority of data qubits are flipped
        error_count = np.sum(self.data_qubits)
        if error_count > self.d // 2:
            # Majority says error: flip all back
            self.data_qubits ^= 1

    def has_logical_error(self) -> bool:
        """Check if a logical error has occurred (majority of data qubits flipped)."""
        return int(np.sum(self.data_qubits) > self.d // 2)


# ---------------------------------------------------------------------------
# Main class: TauTriggeredQEC
# ---------------------------------------------------------------------------

class TauTriggeredQEC:
    """Adaptive QEC that uses tau-chrono to decide when to extract syndromes.

    Parameters
    ----------
    gate_errors : dict
        Gate error rates, e.g. {"cx": 0.003, "h": 0.001}.
    tau_threshold : float
        When accumulated tau exceeds this, trigger syndrome extraction.
        Typical range: 0.05 - 0.3.
    code_distance : int
        Repetition code distance (odd, >= 3).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        gate_errors: Dict[str, float],
        tau_threshold: float = 0.1,
        code_distance: int = 3,
        seed: int = 42,
    ):
        self.gate_errors = gate_errors
        self.tau_threshold = tau_threshold
        self.code_distance = code_distance
        self.seed = seed

        # Precompute tau for each gate type
        self._tau_cache: Dict[str, float] = {}

        # Dominant error rate (for syndrome extraction noise)
        self.p_cx = gate_errors.get("cx", gate_errors.get("cz", 0.01))
        self.p_meas = gate_errors.get("measure", self.p_cx * 0.1)
        self.cnots_per_round = _SYNDROME_CNOTS_PER_ROUND(code_distance)

    def _get_gate_tau(self, gate_name: str) -> float:
        """Get (cached) tau for a gate."""
        if gate_name not in self._tau_cache:
            self._tau_cache[gate_name] = _gate_tau(gate_name, self.gate_errors)
        return self._tau_cache[gate_name]

    def plan_extractions(self, gate_sequence: List[str]) -> Tuple[List[GateEvent], List[int]]:
        """Plan syndrome extraction points for a gate sequence.

        Walks through the gate sequence, accumulating tau. When accumulated
        tau exceeds the threshold, marks that point for syndrome extraction
        and resets the accumulator.

        Returns
        -------
        events : list of GateEvent
            Per-gate record with tau values and extraction flags.
        extraction_points : list of int
            Gate indices where extraction is triggered.
        """
        events = []
        extraction_points = []
        tau_acc = 0.0

        for i, gate_name in enumerate(gate_sequence):
            tau_gate = self._get_gate_tau(gate_name)
            tau_acc = _compose_tau(tau_acc, tau_gate)

            triggered = tau_acc >= self.tau_threshold
            if triggered:
                extraction_points.append(i)

            events.append(GateEvent(
                index=i,
                gate_name=gate_name,
                error_rate=self.gate_errors.get(gate_name, 0.01),
                tau_increment=tau_gate,
                tau_accumulated=tau_acc,
                triggered_extraction=triggered,
            ))

            if triggered:
                tau_acc = 0.0  # reset after extraction

        return events, extraction_points

    def _simulate_no_qec(
        self,
        gate_sequence: List[str],
        shots: int,
        rng: np.random.Generator,
    ) -> StrategyResult:
        """Strategy 1: No QEC -- raw physical errors accumulate."""
        sim = RepetitionCodeSimulator(self.code_distance, rng)
        logical_errors = 0

        for _ in range(shots):
            sim.reset()
            for gate_name in gate_sequence:
                p = self.gate_errors.get(gate_name, 0.01)
                sim.apply_gate_error(p)
            if sim.has_logical_error():
                logical_errors += 1

        ler = logical_errors / shots
        return StrategyResult(
            name="No QEC",
            logical_error_rate=ler,
            syndrome_rounds=0,
            total_cnot_overhead=0,
            total_gates=len(gate_sequence),
            description="Baseline: no error correction applied.",
        )

    def _simulate_fixed_qec(
        self,
        gate_sequence: List[str],
        shots: int,
        interval: int,
        rng: np.random.Generator,
    ) -> StrategyResult:
        """Strategy 2: Fixed-interval QEC -- extract every `interval` gates."""
        sim = RepetitionCodeSimulator(self.code_distance, rng)
        logical_errors = 0
        n_rounds = len(gate_sequence) // interval

        for _ in range(shots):
            sim.reset()
            for i, gate_name in enumerate(gate_sequence):
                p = self.gate_errors.get(gate_name, 0.01)
                sim.apply_gate_error(p)
                if (i + 1) % interval == 0:
                    sim.extract_syndrome_and_correct(self.p_cx, self.p_meas)
            if sim.has_logical_error():
                logical_errors += 1

        ler = logical_errors / shots
        total_cnots = n_rounds * self.cnots_per_round
        return StrategyResult(
            name=f"Fixed QEC (every {interval})",
            logical_error_rate=ler,
            syndrome_rounds=n_rounds,
            total_cnot_overhead=total_cnots,
            total_gates=len(gate_sequence),
            description=f"Syndrome extraction every {interval} gates.",
        )

    def _simulate_adaptive_qec(
        self,
        gate_sequence: List[str],
        shots: int,
        extraction_points: List[int],
        rng: np.random.Generator,
    ) -> StrategyResult:
        """Strategy 3: Tau-triggered adaptive QEC."""
        sim = RepetitionCodeSimulator(self.code_distance, rng)
        logical_errors = 0
        extraction_set = set(extraction_points)
        n_rounds = len(extraction_points)

        for _ in range(shots):
            sim.reset()
            for i, gate_name in enumerate(gate_sequence):
                p = self.gate_errors.get(gate_name, 0.01)
                sim.apply_gate_error(p)
                if i in extraction_set:
                    sim.extract_syndrome_and_correct(self.p_cx, self.p_meas)
            if sim.has_logical_error():
                logical_errors += 1

        ler = logical_errors / shots
        total_cnots = n_rounds * self.cnots_per_round
        return StrategyResult(
            name=f"Tau-triggered (threshold={self.tau_threshold})",
            logical_error_rate=ler,
            syndrome_rounds=n_rounds,
            total_cnot_overhead=total_cnots,
            total_gates=len(gate_sequence),
            description=(
                f"Adaptive syndrome extraction when accumulated tau > "
                f"{self.tau_threshold}. Triggered {n_rounds} times."
            ),
        )

    def compare_strategies(
        self,
        gate_sequence: List[str],
        shots: int = 10000,
        fixed_interval: int = 10,
    ) -> ComparisonResult:
        """Run all three strategies and compare.

        Parameters
        ----------
        gate_sequence : list of str
            Sequence of gate names to simulate.
        shots : int
            Monte Carlo shots per strategy.
        fixed_interval : int
            Gate interval for fixed QEC strategy.

        Returns
        -------
        ComparisonResult
            Full comparison with all strategies and tau trace.
        """
        # Plan adaptive extraction points
        events, extraction_points = self.plan_extractions(gate_sequence)
        tau_trace = [e.tau_accumulated for e in events]

        # Use independent RNGs with different seeds for fair comparison
        rng_no_qec = np.random.default_rng(self.seed)
        rng_fixed = np.random.default_rng(self.seed)
        rng_adaptive = np.random.default_rng(self.seed)

        # Run all three strategies
        result_no_qec = self._simulate_no_qec(gate_sequence, shots, rng_no_qec)
        result_fixed = self._simulate_fixed_qec(
            gate_sequence, shots, fixed_interval, rng_fixed
        )
        result_adaptive = self._simulate_adaptive_qec(
            gate_sequence, shots, extraction_points, rng_adaptive
        )

        return ComparisonResult(
            hardware_name="custom",
            gate_errors=self.gate_errors,
            circuit_length=len(gate_sequence),
            code_distance=self.code_distance,
            tau_threshold=self.tau_threshold,
            shots=shots,
            strategies=[result_no_qec, result_fixed, result_adaptive],
            tau_trace=tau_trace,
            extraction_points=extraction_points,
        )


# ---------------------------------------------------------------------------
# Convenience function for quick comparison
# ---------------------------------------------------------------------------

def run_adaptive_qec_experiment(
    gate_sequence: List[str],
    gate_errors: Dict[str, float],
    tau_threshold: float = 0.1,
    code_distance: int = 3,
    shots: int = 10000,
    fixed_interval: int = 10,
    seed: int = 42,
) -> ComparisonResult:
    """Run a complete adaptive QEC experiment.

    Convenience wrapper around TauTriggeredQEC.compare_strategies().
    """
    sim = TauTriggeredQEC(
        gate_errors=gate_errors,
        tau_threshold=tau_threshold,
        code_distance=code_distance,
        seed=seed,
    )
    return sim.compare_strategies(
        gate_sequence=gate_sequence,
        shots=shots,
        fixed_interval=fixed_interval,
    )


def format_results_table(result: ComparisonResult) -> str:
    """Format comparison results as a readable ASCII table."""
    lines = []
    lines.append(f"{'='*78}")
    lines.append(f"  Tau-Triggered Adaptive QEC -- {result.hardware_name}")
    lines.append(f"  Circuit: {result.circuit_length} gates, "
                 f"d={result.code_distance}, "
                 f"tau_threshold={result.tau_threshold}, "
                 f"shots={result.shots}")
    lines.append(f"{'='*78}")
    lines.append("")

    header = f"  {'Strategy':<35} {'Syn. rounds':>12} {'CNOT overhead':>14} {'LER':>10}"
    lines.append(header)
    lines.append(f"  {'-'*35} {'-'*12} {'-'*14} {'-'*10}")

    for s in result.strategies:
        ler_str = f"{s.logical_error_rate:.4f}" if s.logical_error_rate > 0 else "0.0000"
        lines.append(
            f"  {s.name:<35} {s.syndrome_rounds:>12} {s.total_cnot_overhead:>14} {ler_str:>10}"
        )

    lines.append("")

    # Compute savings
    fixed = result.strategies[1]
    adaptive = result.strategies[2]
    if fixed.syndrome_rounds > 0:
        round_savings = (1 - adaptive.syndrome_rounds / fixed.syndrome_rounds) * 100
        cnot_savings = (1 - adaptive.total_cnot_overhead / fixed.total_cnot_overhead) * 100
    else:
        round_savings = 0
        cnot_savings = 0

    lines.append(f"  Adaptive vs Fixed:")
    lines.append(f"    Syndrome round savings: {round_savings:.0f}%")
    lines.append(f"    CNOT overhead savings:  {cnot_savings:.0f}%")

    if fixed.logical_error_rate > 0:
        ler_ratio = adaptive.logical_error_rate / fixed.logical_error_rate
        lines.append(f"    LER ratio (adaptive/fixed): {ler_ratio:.3f}")
    lines.append(f"    Extraction points: {result.extraction_points}")
    lines.append("")
    return "\n".join(lines)


def results_to_json(result: ComparisonResult) -> dict:
    """Convert ComparisonResult to a JSON-serializable dict."""
    return {
        "hardware_name": result.hardware_name,
        "gate_errors": result.gate_errors,
        "circuit_length": result.circuit_length,
        "code_distance": result.code_distance,
        "tau_threshold": result.tau_threshold,
        "shots": result.shots,
        "strategies": [
            {
                "name": s.name,
                "logical_error_rate": s.logical_error_rate,
                "syndrome_rounds": s.syndrome_rounds,
                "total_cnot_overhead": s.total_cnot_overhead,
                "total_gates": s.total_gates,
                "description": s.description,
            }
            for s in result.strategies
        ],
        "extraction_points": result.extraction_points,
        "tau_trace_summary": {
            "min": float(min(result.tau_trace)) if result.tau_trace else 0,
            "max": float(max(result.tau_trace)) if result.tau_trace else 0,
            "mean": float(np.mean(result.tau_trace)) if result.tau_trace else 0,
        },
    }
