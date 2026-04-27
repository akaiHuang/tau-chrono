"""
Anomaly-Based Recovery (ABR): channel-agnostic process fidelity from
anomalous weak values.

Provides a single-parameter `F_anomaly` estimator that:
  1. Does not require choosing a Petz reference state sigma in advance
     (avoids the IQM Garnet failure mode of v0.1).
  2. Works on depolarizing-, amplitude-damping-, and resonator-coupled
     transmons with a single formula.
  3. Reduces to standard Petz behaviour in the unital limit.

Validated experimentally on Tuna-9, IQM Garnet, IQM Sirius, IQM Emerald
in 2026-04-27 (see `data/iqm_4platform_validation/`).

Public API:
    extract_F_anomaly(pointer_obs, pointer_theory, weights=None)
        -> ExtractionResult
    predict_observation(F_anomaly, theta_psi, theta_phi, g)
        -> dict with predicted pointer + projector weak values

Universal cross-platform formula:

    <Pi_0>_w_observed = 1 - F_anomaly * <Pi_1>_w_theory(g)

where F_anomaly is the single-parameter Bayesian retrodiction-failure
estimator extracted from a one-shot weak-value g-sweep.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np


@dataclass
class ExtractionResult:
    """Result of fitting a single-parameter F_anomaly to a g-sweep."""
    F_anomaly: float
    F_anomaly_err: float
    chi2: float
    dof: int
    chi2_per_dof: float
    ratios: List[float]
    consistent: bool

    @property
    def tau_anomaly(self) -> float:
        return 1.0 - self.F_anomaly

    def __repr__(self) -> str:
        verdict = "consistent" if self.consistent else "inconsistent"
        return (f"ExtractionResult(F_anomaly={self.F_anomaly:.4f} "
                f"+/- {self.F_anomaly_err:.4f}, "
                f"tau_anomaly={self.tau_anomaly:.4f}, "
                f"chi2/dof={self.chi2_per_dof:.2f}, {verdict})")


def extract_F_anomaly(
    pointer_obs: Sequence[float],
    pointer_theory: Sequence[float],
    pointer_obs_err: Optional[Sequence[float]] = None,
    chi2_threshold: float = 3.0,
) -> ExtractionResult:
    """Single-parameter weighted-mean fit of pointer_obs / pointer_theory.

    Parameters
    ----------
    pointer_obs       : measured post-selected <sigma_x>_meter for each g
    pointer_theory    : theoretical exact pointer for each g
                        (computed from psi, phi, g via the Aharonov formula)
    pointer_obs_err   : 1-sigma uncertainty on each pointer_obs (optional)
    chi2_threshold    : chi^2/dof above which model is flagged inconsistent

    Returns
    -------
    ExtractionResult with F_anomaly, error, chi^2 / dof, consistency verdict.
    """
    obs = np.asarray(pointer_obs, dtype=float)
    th = np.asarray(pointer_theory, dtype=float)
    if obs.shape != th.shape:
        raise ValueError("pointer_obs and pointer_theory must have same shape")
    ratios = obs / th
    if pointer_obs_err is not None:
        err = np.asarray(pointer_obs_err, dtype=float) / np.abs(th)
        weights = 1.0 / err**2
    else:
        weights = np.ones_like(ratios)
    F_mean = float(np.average(ratios, weights=weights))
    F_err = float(1.0 / math.sqrt(weights.sum()))
    chi2 = float(((ratios - F_mean) ** 2 * weights).sum())
    dof = max(len(ratios) - 1, 1)
    chi2_dof = chi2 / dof
    return ExtractionResult(
        F_anomaly=F_mean,
        F_anomaly_err=F_err,
        chi2=chi2,
        dof=dof,
        chi2_per_dof=chi2_dof,
        ratios=[float(r) for r in ratios],
        consistent=chi2_dof <= chi2_threshold,
    )


def theory_pointer(theta_psi: float, theta_phi: float, g: float) -> dict:
    """Exact (any g) theoretical pointer + post-select probability +
    projector weak values."""
    a = math.cos(theta_psi / 2); b = math.sin(theta_psi / 2)
    c = math.cos(theta_phi / 2); d = math.sin(theta_phi / 2)
    cg, sg = math.cos(g), math.sin(g)
    real_part = a * c + b * d * cg
    imag_part = b * d * sg
    norm2 = real_part**2 + imag_part**2
    pointer = 2 * real_part * imag_part / norm2
    sin2g = math.sin(2 * g)
    pi1_obs = pointer / sin2g
    pi0_obs = 1.0 - pi1_obs
    sigma_z_obs = 1.0 - 2.0 * pi1_obs
    return {
        "pointer": pointer,
        "p_post": norm2,
        "pi1_observed": pi1_obs,
        "pi0_observed": pi0_obs,
        "sigma_z_observed": sigma_z_obs,
    }


def predict_observation(
    F_anomaly: float,
    theta_psi: float,
    theta_phi: float,
    g: float,
) -> dict:
    """Universal cross-platform prediction:
        pointer_observed = F_anomaly * pointer_theory(g)
        <Pi_0>_w_observed = 1 - F_anomaly * <Pi_1>_w_theory(g)

    Tested across Tuna-9 (depolarizing), IQM Garnet/Emerald (amplitude
    damping), and IQM Sirius (resonator-coupled transmon) in 2026-04-27.
    """
    th = theory_pointer(theta_psi, theta_phi, g)
    sin2g = math.sin(2 * g)
    pred_pointer = F_anomaly * th["pointer"]
    pred_pi1 = pred_pointer / sin2g
    pred_pi0 = 1.0 - pred_pi1
    pred_sz = 1.0 - 2.0 * pred_pi1
    return {
        "F_anomaly": F_anomaly,
        "g": g,
        "pred_pointer": pred_pointer,
        "pred_pi1_w": pred_pi1,
        "pred_pi0_w": pred_pi0,
        "pred_sigma_z_w": pred_sz,
        "theory_pointer": th["pointer"],
        "theory_pi0_w": th["pi0_observed"],
    }


# ---------------------------------------------------------------------------
# Validated platform F_anomaly registry (from 4-platform validation, g=0.30,
# 8192 shots each, 2026-04-27)
# ---------------------------------------------------------------------------

VALIDATED_PLATFORMS = {
    "tuna9":   {"F_anomaly": 0.793, "noise": "depolarizing",
                "n_cz": 16, "depth": 30, "platform": "QuTech Tuna-9"},
    "garnet":  {"F_anomaly": 0.884, "noise": "amplitude_damping",
                "n_cz": 2,  "depth": 6,  "platform": "IQM Garnet"},
    "sirius":  {"F_anomaly": 0.838, "noise": "amp_damp + resonator MOVE",
                "n_cz": 2,  "depth": 8,  "platform": "IQM Sirius"},
    "emerald": {"F_anomaly": 0.836, "noise": "amplitude_damping",
                "n_cz": 2,  "depth": 6,  "platform": "IQM Emerald"},
}


def get_platform_F_anomaly(platform_name: str) -> float:
    """Return validated F_anomaly for a known platform (lowercase name)."""
    p = VALIDATED_PLATFORMS.get(platform_name.lower())
    if p is None:
        raise KeyError(
            f"Unknown platform {platform_name}; "
            f"validated platforms: {sorted(VALIDATED_PLATFORMS.keys())}"
        )
    return p["F_anomaly"]


__all__ = [
    "ExtractionResult",
    "extract_F_anomaly",
    "predict_observation",
    "theory_pointer",
    "VALIDATED_PLATFORMS",
    "get_platform_F_anomaly",
]
