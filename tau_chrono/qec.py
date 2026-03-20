"""
tau-chrono QEC intelligence module.

Predicts whether quantum error correction will help or hurt on given hardware,
generates decoder weights from tau characterization, and monitors QEC health
from syndrome statistics.

Usage:
    from tau_chrono.qec import should_enable_qec, qec_decoder_weights, qec_health_monitor

    result = should_enable_qec({"cx": 0.05, "h": 0.02})
    print(result.enable)   # False
    print(result.reason)   # "Physical error rate 5.0% exceeds threshold 3.0%..."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np

from tau_chrono import depolarizing, tau_parameter

# ---------------------------------------------------------------------------
# Threshold constants for different QEC code families
# ---------------------------------------------------------------------------

_CODE_THRESHOLDS: Dict[str, float] = {
    "repetition": 0.03,   # ~3% for repetition code
    "surface": 0.01,      # ~1% for surface code
    "steane": 0.01,       # ~1% for Steane [[7,1,3]]
    "shor": 0.01,         # ~1% for Shor [[9,1,3]]
}

# Syndrome CNOT count per stabilizer round (approximate formulas)
# repetition code distance d: 2*(d-1) CNOT gates for one round
# surface code distance d: ~4*d^2 CNOT gates for one round
_SYNDROME_CNOTS: Dict[str, callable] = {
    "repetition": lambda d: 2 * (d - 1),
    "surface": lambda d: 4 * d * d,
    "steane": lambda d: 4 * d,
    "shor": lambda d: 8 * d,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class QECRecommendation:
    """Result of the QEC enable/disable prediction."""
    enable: bool
    predicted_ler_with_qec: float
    predicted_ler_without_qec: float
    reason: str
    threshold_error_rate: float

    def __repr__(self):
        status = "ENABLE" if self.enable else "DISABLE"
        return (
            f"QECRecommendation(\n"
            f"  enable = {self.enable}\n"
            f"  predicted_ler_with_qec    = {self.predicted_ler_with_qec:.6f}\n"
            f"  predicted_ler_without_qec = {self.predicted_ler_without_qec:.6f}\n"
            f"  threshold_error_rate      = {self.threshold_error_rate:.4f}\n"
            f"  reason = \"{self.reason}\"\n"
            f")"
        )


@dataclass
class QECHealthAlert:
    """Result of QEC health monitoring from syndrome data."""
    healthy: bool
    delta_D: float
    mean_syndrome_rate: float
    drift_detected: bool
    message: str

    def __repr__(self):
        status = "HEALTHY" if self.healthy else "DEGRADED"
        return (
            f"QECHealthAlert(\n"
            f"  healthy        = {self.healthy}  ({status})\n"
            f"  delta_D        = {self.delta_D:.6f}\n"
            f"  syndrome_rate  = {self.mean_syndrome_rate:.4f}\n"
            f"  drift_detected = {self.drift_detected}\n"
            f"  message = \"{self.message}\"\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def should_enable_qec(
    gate_errors: Dict[str, float],
    code_type: str = "repetition",
    code_distance: int = 3,
) -> QECRecommendation:
    """Predict whether QEC will help or hurt on this hardware.

    Uses the physical error rates from tau characterization to estimate:
    1. The logical error rate WITHOUT QEC (= physical error rate)
    2. The logical error rate WITH QEC (accounting for syndrome extraction overhead)
    3. Whether QEC improves things or makes them worse

    Parameters
    ----------
    gate_errors : dict
        Gate error rates, e.g. {"cx": 0.04, "h": 0.02, "measure": 0.01}.
        At minimum, "cx" should be provided (dominant error source).
    code_type : str
        QEC code family: "repetition", "surface", "steane", "shor".
    code_distance : int
        Code distance (must be odd, >= 3).

    Returns
    -------
    QECRecommendation
        Recommendation with predicted logical error rates and explanation.

    Examples
    --------
    >>> result = should_enable_qec({"cx": 0.001, "h": 0.0005})
    >>> result.enable
    True

    >>> result = should_enable_qec({"cx": 0.05, "h": 0.02})
    >>> result.enable
    False
    """
    if code_type not in _CODE_THRESHOLDS:
        raise ValueError(
            f"Unknown code type '{code_type}'. "
            f"Supported: {list(_CODE_THRESHOLDS.keys())}"
        )
    if code_distance < 3 or code_distance % 2 == 0:
        raise ValueError(
            f"Code distance must be odd and >= 3, got {code_distance}"
        )

    # Extract the dominant error rate (CNOT is usually the bottleneck)
    p_cx = gate_errors.get("cx", gate_errors.get("cz", 0.01))
    p_meas = gate_errors.get("measure", 0.001)
    p_single = np.mean([
        v for k, v in gate_errors.items()
        if k not in ("cx", "cz", "swap", "measure", "barrier")
    ]) if any(k not in ("cx", "cz", "swap", "measure", "barrier")
              for k in gate_errors) else p_cx / 2

    # Physical error rate (effective per-qubit per-round error)
    # Each data qubit participates in a fixed number of CNOTs per syndrome
    # round, regardless of total code size:
    #   - Repetition code: each data qubit is in ~2 CNOTs (left + right stabilizer)
    #   - Surface code: each data qubit is in ~4 CNOTs (4 neighboring stabilizers)
    # The overhead is per-qubit, NOT total circuit size.
    # Including ancilla back-action: each CNOT can also propagate an ancilla
    # error back to the data qubit, roughly doubling the effective error.
    _CNOTS_PER_DATA_QUBIT = {
        "repetition": 2,   # each data qubit in 2 stabilizers
        "surface": 4,      # each data qubit in 4 stabilizers
        "steane": 4,
        "shor": 4,
    }
    n_cnots_per_qubit = _CNOTS_PER_DATA_QUBIT.get(code_type, 4)
    n_syndrome_cnots = _SYNDROME_CNOTS[code_type](code_distance)

    # Effective per-qubit error rate per syndrome round:
    # - n_cnots_per_qubit CNOTs, each with error ~ p_cx on data qubit
    # - ancilla back-action contributes ~50% additional error (accounted
    #   for by using 1.5x multiplier rather than 2x, since threshold
    #   constants already absorb some of this)
    # - single-qubit idle noise ~ p_single
    p_per_round = 1.0 - (1.0 - p_cx) ** int(1.5 * n_cnots_per_qubit + 0.5)
    p_effective = min(p_per_round + p_single * 0.1, 1.0)

    # Threshold for this code
    p_threshold = _CODE_THRESHOLDS[code_type]

    # Logical error rate WITHOUT QEC = physical error rate
    p_logical_no_qec = p_cx

    # Logical error rate WITH QEC
    # Standard scaling: p_L ~ (p_eff / p_threshold)^floor((d+1)/2)
    # This is the standard result from threshold theorem:
    #   p_L ~ p_threshold * (p_eff / p_threshold)^((d+1)/2)
    # Below threshold (ratio < 1): higher d -> exponentially lower p_L
    # Above threshold (ratio > 1): higher d -> exponentially higher p_L
    t = (code_distance + 1) // 2  # correction capability

    if p_effective < p_threshold:
        # Below threshold: QEC helps
        ratio = p_effective / p_threshold
        p_logical_with_qec = float(p_threshold * ratio ** t)
        p_logical_with_qec = min(p_logical_with_qec, 1.0)
    else:
        # Above threshold: QEC hurts
        # Syndrome extraction errors compound faster than correction helps
        # Empirical model: p_L ~ p_eff * (n_syndrome_cnots overhead)
        overhead_factor = 1.0 + n_syndrome_cnots * p_cx
        p_logical_with_qec = min(p_cx * overhead_factor, 1.0)

    # Decision
    enable = p_logical_with_qec < p_logical_no_qec

    # Build explanation
    if enable:
        improvement = (1.0 - p_logical_with_qec / p_logical_no_qec) * 100
        reason = (
            f"Physical error rate {p_cx*100:.1f}% is below threshold "
            f"{p_threshold*100:.1f}%. "
            f"QEC will reduce logical error rate by ~{improvement:.0f}%."
        )
    else:
        if p_effective >= p_threshold:
            reason = (
                f"Physical error rate {p_cx*100:.1f}% exceeds threshold "
                f"{p_threshold*100:.1f}%. "
                f"QEC will likely INCREASE logical error rate."
            )
        else:
            reason = (
                f"Physical error rate {p_cx*100:.1f}% is near threshold "
                f"{p_threshold*100:.1f}%. "
                f"QEC overhead outweighs correction benefit at distance "
                f"{code_distance}."
            )

    return QECRecommendation(
        enable=enable,
        predicted_ler_with_qec=p_logical_with_qec,
        predicted_ler_without_qec=p_logical_no_qec,
        reason=reason,
        threshold_error_rate=p_threshold,
    )


def qec_decoder_weights(
    gate_errors: Dict[str, float],
    qubit_ids: Optional[List[int]] = None,
    per_qubit_errors: Optional[Dict[int, Dict[str, float]]] = None,
) -> Dict[int, float]:
    """Generate per-qubit weights for MWPM decoder from tau characterization.

    Maps tau characterization data to edge weights for PyMatching's
    minimum-weight perfect matching decoder. Noisier qubits (higher tau)
    get lower weight, so the decoder is more likely to assign errors there.

    Zero additional circuits needed -- uses existing calibration data.

    Parameters
    ----------
    gate_errors : dict
        Default gate error rates (used for qubits without per-qubit data).
    qubit_ids : list of int, optional
        Qubit IDs to generate weights for. Default: [0, 1, ..., 8] (T-9).
    per_qubit_errors : dict, optional
        Per-qubit error rates: {qubit_id: {"cx": p_cx, "h": p_h, ...}}.
        If None, all qubits use gate_errors (uniform weights).

    Returns
    -------
    dict
        Mapping qubit_id -> weight (float). Higher weight = more reliable.
        Suitable for passing to PyMatching.

    Examples
    --------
    >>> weights = qec_decoder_weights(
    ...     {"cx": 0.01},
    ...     per_qubit_errors={0: {"cx": 0.005}, 1: {"cx": 0.02}, 2: {"cx": 0.01}}
    ... )
    >>> weights[0] > weights[1]  # qubit 0 is more reliable -> higher weight
    True
    """
    if qubit_ids is None:
        if per_qubit_errors is not None:
            qubit_ids = sorted(per_qubit_errors.keys())
        else:
            qubit_ids = list(range(9))  # T-9 default

    rho = np.array([[1, 0], [0, 0]], dtype=complex)
    sigma = np.eye(2, dtype=complex) / 2

    weights = {}
    for qid in qubit_ids:
        # Get this qubit's error rates
        if per_qubit_errors is not None and qid in per_qubit_errors:
            q_errors = per_qubit_errors[qid]
        else:
            q_errors = gate_errors

        # Compute tau from the dominant error (CNOT)
        p_cx = q_errors.get("cx", q_errors.get("cz", 0.01))
        channel = depolarizing(min(p_cx, 1.0))
        tau = tau_parameter(rho, channel, sigma)

        # Weight = -log(p_error) for MWPM, but using tau as the error proxy
        # Higher tau = noisier = lower weight (decoder assigns errors here)
        # Clamp tau to avoid log(0)
        tau_clamped = max(tau, 1e-10)
        weight = -np.log(tau_clamped)

        weights[qid] = float(weight)

    return weights


def qec_health_monitor(
    syndrome_history: List[List[int]],
    window_size: int = 50,
    drift_threshold: float = 0.3,
) -> QECHealthAlert:
    """Monitor QEC health from syndrome measurement statistics.

    Detects noise drift without any additional quantum circuits by analyzing
    the syndrome measurement record. Uses the retrodiction gap delta_D:
    if syndrome rates change significantly over time, noise has drifted
    and QEC parameters may need recalibration.

    Parameters
    ----------
    syndrome_history : list of list of int
        Each inner list is one round of syndrome measurements (0 or 1).
        Outer list is time-ordered rounds.
        Example: [[0,1,0,1], [0,0,0,1], [1,1,0,1], ...]
    window_size : int
        Number of rounds per analysis window. Default 50.
    drift_threshold : float
        Relative change in syndrome rate that triggers an alert. Default 0.3.

    Returns
    -------
    QECHealthAlert
        Health status with drift detection results.

    Examples
    --------
    >>> # Stable noise: syndrome rate ~ 0.3 throughout
    >>> history = [[int(np.random.random() < 0.3) for _ in range(4)]
    ...            for _ in range(100)]
    >>> alert = qec_health_monitor(history)
    >>> alert.healthy
    True
    """
    if not syndrome_history or len(syndrome_history) < 2:
        return QECHealthAlert(
            healthy=True,
            delta_D=0.0,
            mean_syndrome_rate=0.0,
            drift_detected=False,
            message="Insufficient data for analysis (need >= 2 rounds).",
        )

    n_rounds = len(syndrome_history)
    n_stabilizers = len(syndrome_history[0])

    # Compute per-round syndrome rates
    round_rates = []
    for round_data in syndrome_history:
        rate = np.mean(round_data) if len(round_data) > 0 else 0.0
        round_rates.append(rate)

    round_rates = np.array(round_rates)
    overall_mean = float(np.mean(round_rates))

    # Split into windows and compare
    if n_rounds < 2 * window_size:
        # Not enough data for windowed analysis; compare first/second half
        half = n_rounds // 2
        rate_early = float(np.mean(round_rates[:half]))
        rate_late = float(np.mean(round_rates[half:]))
    else:
        # Compare first and last windows
        rate_early = float(np.mean(round_rates[:window_size]))
        rate_late = float(np.mean(round_rates[-window_size:]))

    # Retrodiction gap: measures how much the noise has changed
    # delta_D = |rate_late - rate_early| / max(rate_early, eps)
    denominator = max(rate_early, 1e-10)
    delta_D = abs(rate_late - rate_early) / denominator

    # Drift detection
    drift_detected = delta_D > drift_threshold

    # Health assessment
    if drift_detected:
        direction = "increased" if rate_late > rate_early else "decreased"
        healthy = False
        message = (
            f"Noise drift detected: syndrome rate {direction} by "
            f"{delta_D*100:.1f}% (early={rate_early:.4f}, "
            f"late={rate_late:.4f}). Consider recalibrating QEC parameters."
        )
    elif overall_mean > 0.45:
        healthy = False
        message = (
            f"High syndrome rate ({overall_mean:.4f}). "
            f"Physical error rate may be above QEC threshold."
        )
    else:
        healthy = True
        message = (
            f"QEC operating normally. Syndrome rate stable at "
            f"{overall_mean:.4f}."
        )

    return QECHealthAlert(
        healthy=healthy,
        delta_D=float(delta_D),
        mean_syndrome_rate=overall_mean,
        drift_detected=drift_detected,
        message=message,
    )
