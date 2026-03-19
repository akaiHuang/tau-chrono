"""
tau-chrono: Bayesian Noise Tracker for Quantum Circuits.

Self-contained module for the qec-retrodiction-decoder project.
Provides core quantum channel, Petz recovery, and Bayesian composition
primitives using only numpy.

Public API
----------
Channels:
    depolarizing, amplitude_damping, dephasing, verify_cptp

Petz / Information:
    apply_channel, fidelity, tau_parameter, relative_entropy

Composition:
    bayesian_compose, compose_kraus, GateResult, CompositionResult
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import List, Optional
from dataclasses import dataclass

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

DensityMatrix = NDArray[np.complexfloating]
KrausList = List[NDArray[np.complexfloating]]

# ---------------------------------------------------------------------------
# Pauli constants
# ---------------------------------------------------------------------------

I2 = np.eye(2, dtype=complex)
X_GATE = np.array([[0, 1], [1, 0]], dtype=complex)
Y_GATE = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z_GATE = np.array([[1, 0], [0, -1]], dtype=complex)

_EPS = 1e-14

# ---------------------------------------------------------------------------
# Linear algebra helpers
# ---------------------------------------------------------------------------


def matrix_sqrt(A: NDArray) -> NDArray:
    """Matrix square root of a PSD matrix via eigendecomposition."""
    A = (A + A.conj().T) / 2
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, 0.0)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.conj().T


def matrix_inv_sqrt(A: NDArray, tol: float = 1e-12) -> NDArray:
    """Pseudoinverse square root of a PSD matrix."""
    A = (A + A.conj().T) / 2
    eigvals, eigvecs = np.linalg.eigh(A)
    inv_sqrt = np.zeros_like(eigvals)
    for i, ev in enumerate(eigvals):
        if ev > tol:
            inv_sqrt[i] = 1.0 / np.sqrt(ev)
    return eigvecs @ np.diag(inv_sqrt) @ eigvecs.conj().T


def commutator_norm(A: NDArray, B: NDArray) -> float:
    """Frobenius norm of the commutator [A, B]."""
    comm = A @ B - B @ A
    return float(np.linalg.norm(comm, "fro"))


# ---------------------------------------------------------------------------
# Quantum channels
# ---------------------------------------------------------------------------


def depolarizing(p: float) -> KrausList:
    """Depolarizing channel: N(rho) = (1-p) rho + (p/3)(X rho X + Y rho Y + Z rho Z).

    Parameters
    ----------
    p : float
        Depolarizing probability in [0, 1].

    Returns
    -------
    KrausList
        Four 2x2 Kraus operators.
    """
    assert 0.0 <= p <= 1.0, f"p must be in [0,1], got {p}"
    K0 = np.sqrt(1.0 - 3.0 * p / 4.0) * I2
    K1 = np.sqrt(p / 4.0) * X_GATE
    K2 = np.sqrt(p / 4.0) * Y_GATE
    K3 = np.sqrt(p / 4.0) * Z_GATE
    return [K0, K1, K2, K3]


def amplitude_damping(gamma: float) -> KrausList:
    """Amplitude damping channel (T1 decay).

    Parameters
    ----------
    gamma : float
        Damping probability in [0, 1].

    Returns
    -------
    KrausList
        Two 2x2 Kraus operators.
    """
    assert 0.0 <= gamma <= 1.0, f"gamma must be in [0,1], got {gamma}"
    K0 = np.array([[1.0, 0.0],
                    [0.0, np.sqrt(1.0 - gamma)]], dtype=complex)
    K1 = np.array([[0.0, np.sqrt(gamma)],
                    [0.0, 0.0]], dtype=complex)
    return [K0, K1]


def dephasing(p: float) -> KrausList:
    """Dephasing channel: N(rho) = (1-p) rho + p Z rho Z.

    Parameters
    ----------
    p : float
        Dephasing probability in [0, 1].

    Returns
    -------
    KrausList
        Two 2x2 Kraus operators.
    """
    assert 0.0 <= p <= 1.0, f"p must be in [0,1], got {p}"
    K0 = np.sqrt(1.0 - p) * I2
    K1 = np.sqrt(p) * Z_GATE
    return [K0, K1]


def verify_cptp(kraus_ops: KrausList, tol: float = 1e-10) -> bool:
    """Verify that Kraus operators satisfy CPTP condition: sum K_i^dag K_i = I.

    Parameters
    ----------
    kraus_ops : KrausList
        List of Kraus operators.
    tol : float
        Tolerance for the identity check.

    Returns
    -------
    bool
        True if CPTP condition is satisfied within tolerance.
    """
    d = kraus_ops[0].shape[1]
    total = np.zeros((d, d), dtype=complex)
    for K in kraus_ops:
        total += K.conj().T @ K
    return bool(np.linalg.norm(total - np.eye(d, dtype=complex)) < tol)


# ---------------------------------------------------------------------------
# Channel application and adjoint
# ---------------------------------------------------------------------------


def apply_channel(rho: DensityMatrix, kraus_ops: KrausList) -> DensityMatrix:
    """Apply a quantum channel: N(rho) = sum_k K_k rho K_k^dag.

    Parameters
    ----------
    rho : DensityMatrix
        Input density matrix.
    kraus_ops : KrausList
        Kraus operators.

    Returns
    -------
    DensityMatrix
        Output density matrix.
    """
    result = np.zeros((kraus_ops[0].shape[0], kraus_ops[0].shape[0]),
                       dtype=complex)
    for K in kraus_ops:
        result += K @ rho @ K.conj().T
    return result


def adjoint_channel(X: NDArray, kraus_ops: KrausList) -> NDArray:
    """Apply the adjoint (Heisenberg picture) channel: N^dag(X) = sum_k K_k^dag X K_k."""
    d_in = kraus_ops[0].shape[1]
    result = np.zeros((d_in, d_in), dtype=complex)
    for K in kraus_ops:
        result += K.conj().T @ X @ K
    return result


# ---------------------------------------------------------------------------
# Fidelity and tau parameter
# ---------------------------------------------------------------------------


def fidelity(rho: DensityMatrix, sigma: DensityMatrix) -> float:
    """Uhlmann fidelity (squared convention): F = (Tr sqrt(sqrt(rho) sigma sqrt(rho)))^2.

    Returns
    -------
    float
        Fidelity in [0, 1].
    """
    sqrt_rho = matrix_sqrt(rho)
    M = sqrt_rho @ sigma @ sqrt_rho
    M = (M + M.conj().T) / 2
    eigvals = np.linalg.eigvalsh(M)
    eigvals = np.maximum(eigvals, 0.0)
    F = np.sum(np.sqrt(eigvals)) ** 2
    return float(np.clip(F.real, 0.0, 1.0))


def petz_recovery_map(kraus_ops: KrausList, sigma: DensityMatrix):
    """Construct the Petz recovery map as a superoperator matrix.

    R_{sigma,N}(X) = sigma^{1/2} N^dag(N(sigma)^{-1/2} X N(sigma)^{-1/2}) sigma^{1/2}

    Returns
    -------
    NDArray
        Column-vectorised superoperator.
    """
    d_in = sigma.shape[0]
    d_out = kraus_ops[0].shape[0]

    sigma_sqrt = matrix_sqrt(sigma)
    tau_N = apply_channel(sigma, kraus_ops)
    tau_N_inv_sqrt = matrix_inv_sqrt(tau_N)

    S_R = np.zeros((d_in * d_in, d_out * d_out), dtype=complex)
    for i in range(d_out):
        for j in range(d_out):
            E_ij = np.zeros((d_out, d_out), dtype=complex)
            E_ij[i, j] = 1.0
            sandwiched = tau_N_inv_sqrt @ E_ij @ tau_N_inv_sqrt
            adj_result = adjoint_channel(sandwiched, kraus_ops)
            R_Eij = sigma_sqrt @ adj_result @ sigma_sqrt
            S_R[:, j * d_out + i] = R_Eij.flatten(order="F")
    return S_R


def apply_petz_recovery(
    X: DensityMatrix,
    kraus_ops: KrausList,
    sigma: DensityMatrix,
    S_R=None,
) -> DensityMatrix:
    """Apply the Petz recovery map to an operator."""
    d_in = sigma.shape[0]
    if S_R is None:
        S_R = petz_recovery_map(kraus_ops, sigma)
    vec_X = X.flatten(order="F")
    vec_RX = S_R @ vec_X
    result = vec_RX.reshape(d_in, d_in, order="F")
    return (result + result.conj().T) / 2


def tau_parameter(
    rho: DensityMatrix,
    kraus_ops: KrausList,
    sigma: DensityMatrix,
) -> float:
    """Recovery failure parameter: tau = 1 - F(rho, R_{sigma,N}(N(rho))).

    tau = 0 means perfect recovery.
    tau = 1 means complete failure.

    Parameters
    ----------
    rho : DensityMatrix
        Input state.
    kraus_ops : KrausList
        Kraus operators of the channel.
    sigma : DensityMatrix
        Reference state.

    Returns
    -------
    float
        tau in [0, 1].
    """
    omega = apply_channel(rho, kraus_ops)
    S_R = petz_recovery_map(kraus_ops, sigma)
    recovered = apply_petz_recovery(omega, kraus_ops, sigma, S_R)
    F = fidelity(rho, recovered)
    return float(np.clip(1.0 - F, 0.0, 1.0))


def relative_entropy(rho: DensityMatrix, sigma: DensityMatrix) -> float:
    """Quantum relative entropy D(rho || sigma) = Tr[rho (log rho - log sigma)].

    Returns +inf when supp(rho) is not contained in supp(sigma).
    """
    rho = (rho + rho.conj().T) / 2
    sigma = (sigma + sigma.conj().T) / 2
    eigvals_rho, U_rho = np.linalg.eigh(rho)
    eigvals_sigma, U_sigma = np.linalg.eigh(sigma)

    overlap = U_sigma.conj().T @ U_rho
    T = np.abs(overlap) ** 2

    for k in range(len(eigvals_rho)):
        if eigvals_rho[k] > _EPS:
            weight_outside = np.sum(T[eigvals_sigma <= _EPS, k])
            if weight_outside > 1e-8:
                return float("inf")

    mask_rho = eigvals_rho > _EPS
    term1 = np.sum(eigvals_rho[mask_rho] * np.log(eigvals_rho[mask_rho]))

    mask_sigma = eigvals_sigma > _EPS
    log_q = np.zeros_like(eigvals_sigma)
    log_q[mask_sigma] = np.log(eigvals_sigma[mask_sigma])
    term2 = 0.0
    for k in range(len(eigvals_rho)):
        if eigvals_rho[k] > _EPS:
            term2 += eigvals_rho[k] * np.sum(T[mask_sigma, k] * log_q[mask_sigma])

    result = float(term1 - term2)
    if result < 0 and result > -1e-10:
        result = 0.0
    return result


# ---------------------------------------------------------------------------
# Bayesian composition
# ---------------------------------------------------------------------------


@dataclass
class GateResult:
    """Result for a single gate in the circuit."""
    gate_index: int
    channel_name: str
    tau_naive: float
    tau_eff: float
    sigma_before: DensityMatrix
    rho_before: DensityMatrix
    rho_after: DensityMatrix
    commutator_norm: float
    pre_commutator_norm: float
    classification: str


@dataclass
class CompositionResult:
    """Full result of a Bayesian composition analysis."""
    gate_results: List[GateResult]
    tau_bayesian_total: float
    tau_multiplicative_total: float
    improvement_percent: float
    composition_lhs: float
    composition_rhs: float
    composition_holds: bool
    composition_slack: float


def compose_kraus(channels: List[KrausList]) -> KrausList:
    """Compose a sequence of Kraus channels into a single channel.

    Parameters
    ----------
    channels : List[KrausList]
        Ordered list of channels (first applied first).

    Returns
    -------
    KrausList
        Composed Kraus operators.
    """
    if len(channels) == 0:
        return [np.eye(2, dtype=complex)]
    if len(channels) == 1:
        return channels[0]

    composed = channels[0]
    for i in range(1, len(channels)):
        new_composed = []
        for K_new in channels[i]:
            for K_old in composed:
                new_composed.append(K_new @ K_old)
        composed = new_composed
    return composed


def compose_kraus_compressed(
    channels: List[KrausList],
    max_ops: int = 64,
    compress_threshold: int = 256,
) -> KrausList:
    """Compose channels with SVD compression to limit Kraus count."""
    if len(channels) == 0:
        return [np.eye(2, dtype=complex)]
    if len(channels) == 1:
        return channels[0]

    d = channels[0][0].shape[0]
    composed = channels[0]
    for i in range(1, len(channels)):
        new_composed = []
        for K_new in channels[i]:
            for K_old in composed:
                prod = K_new @ K_old
                if np.linalg.norm(prod) > 1e-14:
                    new_composed.append(prod)
        composed = new_composed
        if len(composed) > compress_threshold:
            n = len(composed)
            M = np.zeros((n, d * d), dtype=complex)
            for idx, K in enumerate(composed):
                M[idx] = K.flatten()
            U, S, Vh = np.linalg.svd(M, full_matrices=False)
            threshold = 1e-12 * S[0] if len(S) > 0 else 0
            keep = min(max_ops, int(np.sum(S > threshold)))
            keep = max(keep, 1)
            composed = [(S[idx] * Vh[idx]).reshape(d, d) for idx in range(keep)]
    return composed


def bayesian_compose(
    channels: List[KrausList],
    sigma_0: DensityMatrix,
    rho: DensityMatrix,
    channel_names: Optional[List[str]] = None,
    comm_threshold: float = 0.01,
    memory_alpha: Optional[float] = None,
    min_depth: int = 0,
) -> CompositionResult:
    """Bayesian composition engine: per-gate tau with updated sigma.

    For each gate i:
      1. tau_naive = tau(rho_0, N_i, sigma_0)
      2. tau_eff   = tau(rho_i, N_i, sigma_i)  (Bayesian-updated)
      3. Update rho and sigma through the channel

    Parameters
    ----------
    channels : List[KrausList]
        Sequence of noise channels.
    sigma_0 : DensityMatrix
        Initial reference state.
    rho : DensityMatrix
        Initial input state.
    channel_names : list[str], optional
        Names for each channel.
    comm_threshold : float
        Threshold for gate classification.
    memory_alpha : float, optional
        Non-Markovian memory parameter in [0, 1].
    min_depth : int
        Minimum depth for Bayesian tracking.

    Returns
    -------
    CompositionResult
        Full analysis.
    """
    n_gates = len(channels)
    if channel_names is None:
        channel_names = [f"Gate_{i}" for i in range(n_gates)]

    gate_results: List[GateResult] = []
    rho_current = rho.copy()
    sigma_current = sigma_0.copy()

    for i in range(n_gates):
        kraus = channels[i]
        pre_comm = commutator_norm(rho_current, sigma_current)
        tau_naive = tau_parameter(rho, kraus, sigma_0)

        if n_gates < min_depth:
            tau_eff = tau_naive
        else:
            tau_eff = tau_parameter(rho_current, kraus, sigma_current)

        rho_after = apply_channel(rho_current, kraus)
        sigma_after = apply_channel(sigma_current, kraus)

        if memory_alpha is not None and i > 0:
            sigma_after = (memory_alpha * sigma_after
                           + (1.0 - memory_alpha) * sigma_current)
            sigma_after = sigma_after / np.trace(sigma_after).real

        comm = commutator_norm(rho_after, sigma_after)

        if tau_eff < 1e-10:
            classification = "SATURATED"
        elif comm < comm_threshold or pre_comm < comm_threshold:
            classification = "NEAR_SATURATED"
        else:
            classification = "NORMAL"

        gate_results.append(GateResult(
            gate_index=i,
            channel_name=channel_names[i],
            tau_naive=tau_naive,
            tau_eff=tau_eff,
            sigma_before=sigma_current.copy(),
            rho_before=rho_current.copy(),
            rho_after=rho_after.copy(),
            commutator_norm=comm,
            pre_commutator_norm=pre_comm,
            classification=classification,
        ))

        rho_current = rho_after
        sigma_current = sigma_after

    # Total tau for the composed channel
    if len(channels) > 8:
        composed_kraus = compose_kraus_compressed(channels)
    else:
        composed_kraus = compose_kraus(channels)
    tau_bayesian_total = tau_parameter(rho, composed_kraus, sigma_0)

    # Multiplicative baseline
    prod_fidelity = 1.0
    for gr in gate_results:
        prod_fidelity *= (1.0 - gr.tau_naive)
    tau_multiplicative_total = 1.0 - prod_fidelity

    # Composition inequality: sqrt(tau_total) <= sum sqrt(tau_i^eff)
    sum_sqrt_tau_eff = sum(
        np.sqrt(max(gr.tau_eff, 0.0)) for gr in gate_results
    )
    sqrt_tau_total = np.sqrt(max(tau_bayesian_total, 0.0))
    composition_holds = sqrt_tau_total <= sum_sqrt_tau_eff + 1e-10
    composition_slack = sum_sqrt_tau_eff - sqrt_tau_total

    if tau_multiplicative_total > 1e-15:
        improvement = (1.0 - tau_bayesian_total / tau_multiplicative_total) * 100.0
    else:
        improvement = 0.0

    return CompositionResult(
        gate_results=gate_results,
        tau_bayesian_total=tau_bayesian_total,
        tau_multiplicative_total=tau_multiplicative_total,
        improvement_percent=improvement,
        composition_lhs=sqrt_tau_total,
        composition_rhs=sum_sqrt_tau_eff,
        composition_holds=composition_holds,
        composition_slack=composition_slack,
    )


__all__ = [
    # channels
    "depolarizing",
    "amplitude_damping",
    "dephasing",
    "verify_cptp",
    # core
    "apply_channel",
    "adjoint_channel",
    "fidelity",
    "tau_parameter",
    "petz_recovery_map",
    "apply_petz_recovery",
    "relative_entropy",
    # composition
    "bayesian_compose",
    "tau_chrono_compose",
    "compose_kraus",
    "compose_kraus_compressed",
    "GateResult",
    "CompositionResult",
    # utils
    "matrix_sqrt",
    "matrix_inv_sqrt",
    "commutator_norm",
]
