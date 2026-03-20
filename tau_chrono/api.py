"""
tau-chrono high-level API.

Usage:
    from tau_chrono.api import predict_circuit, predict_gates

    # Option 1: From a Qiskit circuit
    result = predict_circuit(qc, gate_errors={"cx": 0.01, "h": 0.005})
    print(result.should_run)       # True
    print(result.f_tauchrono)      # 0.82
    print(result.f_naive)          # 0.45
    print(result.savings_pct)      # 45.2%

    # Option 2: Just a list of gate names
    result = predict_gates(["h", "cx", "cx", "h", "cx", "cx", "h"])
    print(result.should_run)       # True

    # Option 3: QEC intelligence
    from tau_chrono.api import should_enable_qec, qec_decoder_weights
    result = should_enable_qec({"cx": 0.05, "h": 0.02})
    print(result.enable)           # False
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import numpy as np

from tau_chrono import depolarizing, tau_chrono_compose

# Default error rates (from QuTech Tuna-9 process tomography)
DEFAULT_ERRORS: Dict[str, float] = {
    "h": 0.0214,
    "x": 0.0269,
    "y": 0.0269,
    "z": 0.001,
    "sx": 0.0110,
    "s": 0.0170,
    "sdg": 0.0170,
    "t": 0.0170,
    "tdg": 0.0170,
    "rz": 0.005,
    "rx": 0.015,
    "ry": 0.015,
    "id": 0.0226,
    "cx": 0.0400,
    "cz": 0.0400,
    "swap": 0.0600,
    "measure": 0.001,
    "barrier": 0.0,
}


@dataclass
class PredictionResult:
    """Result of a tau-chrono circuit prediction."""
    f_tauchrono: float
    f_naive: float
    should_run: bool
    n_gates: int
    improvement_pct: float
    threshold: float
    naive_says: str
    tauchrono_says: str
    savings_pct: float

    def __repr__(self):
        return (
            f"PredictionResult(\n"
            f"  f_tauchrono = {self.f_tauchrono:.4f}  ({self.tauchrono_says})\n"
            f"  f_naive     = {self.f_naive:.4f}  ({self.naive_says})\n"
            f"  improvement = {self.improvement_pct:.1f}%\n"
            f"  gates       = {self.n_gates}\n"
            f"  should_run  = {self.should_run}\n"
            f")"
        )


def predict_gates(
    gate_names: List[str],
    gate_errors: Optional[Dict[str, float]] = None,
    threshold: float = 0.5,
) -> PredictionResult:
    """Predict circuit fidelity from a list of gate names.

    Args:
        gate_names: List of gate names, e.g. ["h", "cx", "cx", "h"]
        gate_errors: Dict mapping gate name to error rate.
                     If None, uses default Tuna-9 calibration data.
        threshold: Fidelity threshold for should_run (default 0.5)

    Returns:
        PredictionResult with f_tauchrono, f_naive, should_run, etc.
    """
    errors = {**DEFAULT_ERRORS, **(gate_errors or {})}

    # Filter out non-gate operations
    skip = {"measure", "barrier", "reset", "delay"}
    gates = [g.lower() for g in gate_names if g.lower() not in skip]

    if not gates:
        return PredictionResult(
            f_tauchrono=1.0, f_naive=1.0, should_run=True,
            n_gates=0, improvement_pct=0.0, threshold=threshold,
            naive_says="GO", tauchrono_says="GO", savings_pct=0.0,
        )

    # Build channels
    channels = []
    for g in gates:
        p = errors.get(g, 0.03)  # default 3% for unknown gates
        channels.append(depolarizing(p))

    rho = np.array([[1, 0], [0, 0]], dtype=complex)
    sigma = np.eye(2, dtype=complex) / 2

    result = tau_chrono_compose(channels, sigma_0=sigma, rho=rho)

    f_naive = 1 - result.tau_multiplicative_total
    f_chrono = 1 - result.tau_bayesian_total

    naive_go = f_naive >= threshold
    chrono_go = f_chrono >= threshold

    improvement = (f_chrono - f_naive) / max(abs(f_naive), 1e-10) * 100

    # Savings: if naive says STOP, user would need 3x shots for majority voting
    if not naive_go and chrono_go:
        savings = 67.0  # save 2/3 of shots
    elif not naive_go and not chrono_go:
        savings = 0.0  # both say stop
    else:
        savings = 0.0  # both say go, no savings

    return PredictionResult(
        f_tauchrono=f_chrono,
        f_naive=f_naive,
        should_run=chrono_go,
        n_gates=len(gates),
        improvement_pct=improvement,
        threshold=threshold,
        naive_says="GO" if naive_go else "STOP",
        tauchrono_says="GO" if chrono_go else "STOP",
        savings_pct=savings,
    )


def predict_circuit(
    circuit,
    gate_errors: Optional[Dict[str, float]] = None,
    threshold: float = 0.5,
) -> PredictionResult:
    """Predict circuit fidelity from a Qiskit QuantumCircuit.

    Args:
        circuit: A qiskit.QuantumCircuit object
        gate_errors: Optional dict of gate error rates.
        threshold: Fidelity threshold (default 0.5)

    Returns:
        PredictionResult
    """
    # Extract gate names from Qiskit circuit
    gate_names = []
    for instruction in circuit.data:
        name = instruction.operation.name.lower()
        gate_names.append(name)

    return predict_gates(gate_names, gate_errors=gate_errors, threshold=threshold)


# ---------------------------------------------------------------------------
# QEC Intelligence (re-exported from tau_chrono.qec)
# ---------------------------------------------------------------------------

from tau_chrono.qec import (  # noqa: E402
    should_enable_qec,
    qec_decoder_weights,
    qec_health_monitor,
    QECRecommendation,
    QECHealthAlert,
)
