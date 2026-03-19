"""
Core tests for the tau-chrono quantum noise tracking library.

Verifies channel construction, tau computation, Bayesian composition,
and information-theoretic bounds.

Run with:
    python -m pytest tests/ -v
"""

import sys
import os
import numpy as np
import pytest

# Ensure the repo root is on the path so tau_chrono is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tau_chrono import (
    depolarizing,
    amplitude_damping,
    dephasing,
    verify_cptp,
    apply_channel,
    fidelity,
    tau_parameter,
    bayesian_compose,
    compose_kraus,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sigma():
    """Maximally mixed qubit state I/2."""
    return np.eye(2, dtype=complex) / 2


@pytest.fixture
def rho_plus():
    """|+> state density matrix."""
    return np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)


@pytest.fixture
def rho_zero():
    """|0> state density matrix."""
    return np.array([[1, 0], [0, 0]], dtype=complex)


# ---------------------------------------------------------------------------
# Test: depolarizing channel is CPTP
# ---------------------------------------------------------------------------

class TestDepolarizingChannelCPTP:
    """Verify the depolarizing channel satisfies the CPTP condition."""

    @pytest.mark.parametrize("p", [0.0, 0.01, 0.05, 0.1, 0.5, 1.0])
    def test_depolarizing_channel_cptp(self, p):
        kraus = depolarizing(p)
        assert verify_cptp(kraus), (
            f"Depolarizing channel with p={p} failed CPTP check"
        )

    def test_amplitude_damping_cptp(self):
        for gamma in [0.0, 0.01, 0.1, 0.5, 1.0]:
            kraus = amplitude_damping(gamma)
            assert verify_cptp(kraus), (
                f"Amplitude damping with gamma={gamma} failed CPTP check"
            )

    def test_dephasing_cptp(self):
        for p in [0.0, 0.1, 0.5, 1.0]:
            kraus = dephasing(p)
            assert verify_cptp(kraus), (
                f"Dephasing with p={p} failed CPTP check"
            )


# ---------------------------------------------------------------------------
# Test: tau of identity channel is zero
# ---------------------------------------------------------------------------

class TestTauIdentity:
    """The identity channel should have tau = 0 (perfect recovery)."""

    def test_tau_identity_is_zero(self, sigma, rho_plus):
        identity_kraus = [np.eye(2, dtype=complex)]
        tau = tau_parameter(rho_plus, identity_kraus, sigma)
        assert tau < 1e-10, (
            f"Expected tau ~ 0 for identity channel, got {tau}"
        )

    def test_tau_identity_zero_state(self, sigma, rho_zero):
        identity_kraus = [np.eye(2, dtype=complex)]
        tau = tau_parameter(rho_zero, identity_kraus, sigma)
        assert tau < 1e-10, (
            f"Expected tau ~ 0 for identity channel on |0>, got {tau}"
        )


# ---------------------------------------------------------------------------
# Test: tau increases with noise strength
# ---------------------------------------------------------------------------

class TestTauIncreasesWithNoise:
    """Higher noise parameter p should yield higher tau."""

    def test_tau_increases_with_noise(self, sigma, rho_plus):
        p_values = [0.01, 0.05, 0.1, 0.2, 0.5]
        taus = []
        for p in p_values:
            kraus = depolarizing(p)
            tau = tau_parameter(rho_plus, kraus, sigma)
            taus.append(tau)

        for i in range(len(taus) - 1):
            assert taus[i] <= taus[i + 1] + 1e-10, (
                f"tau did not increase: tau(p={p_values[i]})={taus[i]:.6f} > "
                f"tau(p={p_values[i+1]})={taus[i+1]:.6f}"
            )

    def test_tau_increases_amplitude_damping(self, sigma, rho_zero):
        gammas = [0.01, 0.05, 0.1, 0.3]
        taus = []
        for gamma in gammas:
            kraus = amplitude_damping(gamma)
            tau = tau_parameter(rho_zero, kraus, sigma)
            taus.append(tau)

        for i in range(len(taus) - 1):
            assert taus[i] <= taus[i + 1] + 1e-10, (
                f"tau did not increase for amplitude damping"
            )


# ---------------------------------------------------------------------------
# Test: Bayesian composition improves over naive estimate at depth > 6
# ---------------------------------------------------------------------------

class TestBayesianImprovesOverNaive:
    """Bayesian tau should be lower than naive multiplicative tau for deep circuits."""

    def test_bayesian_improves_over_naive(self, sigma, rho_plus):
        depth = 10
        p = 0.05
        channels = [depolarizing(p) for _ in range(depth)]
        names = [f"depol_{i}" for i in range(depth)]

        result = bayesian_compose(channels, sigma, rho_plus, channel_names=names)

        assert result.tau_bayesian_total < result.tau_multiplicative_total, (
            f"Bayesian tau ({result.tau_bayesian_total:.6f}) should be less "
            f"than multiplicative tau ({result.tau_multiplicative_total:.6f}) "
            f"at depth {depth}"
        )
        assert result.improvement_percent > 0, (
            f"Expected positive improvement, got {result.improvement_percent:.2f}%"
        )

    def test_bayesian_matches_naive_at_depth_1(self, sigma, rho_plus):
        """At depth 1, Bayesian and naive should be nearly identical."""
        channels = [depolarizing(0.05)]
        result = bayesian_compose(channels, sigma, rho_plus)
        # At depth 1, they should match closely
        assert abs(result.tau_bayesian_total - result.tau_multiplicative_total) < 0.01


# ---------------------------------------------------------------------------
# Test: composition inequality holds: sqrt(tau_total) <= sum sqrt(tau_i)
# ---------------------------------------------------------------------------

class TestCompositionInequalityHolds:
    """The sub-additivity inequality for sqrt(tau) must hold."""

    @pytest.mark.parametrize("depth", [2, 5, 8, 12])
    def test_composition_inequality_holds(self, sigma, rho_plus, depth):
        p = 0.05
        channels = [depolarizing(p) for _ in range(depth)]
        result = bayesian_compose(channels, sigma, rho_plus)

        assert result.composition_holds, (
            f"Composition inequality violated at depth {depth}: "
            f"sqrt(tau_total)={result.composition_lhs:.6f} > "
            f"sum sqrt(tau_i)={result.composition_rhs:.6f}"
        )
        assert result.composition_slack >= -1e-10, (
            f"Negative slack: {result.composition_slack:.6f}"
        )

    def test_composition_inequality_mixed_channels(self, sigma, rho_plus):
        """Test with a mix of different channel types."""
        channels = [
            depolarizing(0.03),
            amplitude_damping(0.05),
            dephasing(0.04),
            depolarizing(0.06),
            amplitude_damping(0.02),
        ]
        result = bayesian_compose(channels, sigma, rho_plus)
        assert result.composition_holds, (
            f"Composition inequality violated for mixed channels"
        )


# ---------------------------------------------------------------------------
# Test: fidelity bounds 0 <= F <= 1
# ---------------------------------------------------------------------------

class TestFidelityBounds:
    """Fidelity must always be in [0, 1]."""

    def test_fidelity_bounds(self, sigma, rho_plus, rho_zero):
        states = [rho_plus, rho_zero, sigma]
        channels = [
            depolarizing(0.01),
            depolarizing(0.5),
            amplitude_damping(0.1),
            dephasing(0.3),
        ]

        for rho in states:
            for kraus in channels:
                rho_out = apply_channel(rho, kraus)
                F = fidelity(rho, rho_out)
                assert 0.0 <= F <= 1.0 + 1e-10, (
                    f"Fidelity out of bounds: F={F}"
                )

    def test_fidelity_identity(self, rho_plus):
        """Fidelity of a state with itself should be 1."""
        F = fidelity(rho_plus, rho_plus)
        assert abs(F - 1.0) < 1e-10, (
            f"F(rho, rho) should be 1, got {F}"
        )

    def test_fidelity_orthogonal(self, rho_zero):
        """|0> and |1> should have fidelity 0."""
        rho_one = np.array([[0, 0], [0, 1]], dtype=complex)
        F = fidelity(rho_zero, rho_one)
        assert F < 1e-10, (
            f"F(|0>, |1>) should be ~0, got {F}"
        )

    def test_fidelity_random_states(self):
        """Fidelity of random density matrices must stay in [0, 1]."""
        rng = np.random.default_rng(42)
        for _ in range(20):
            psi = rng.standard_normal(2) + 1j * rng.standard_normal(2)
            psi /= np.linalg.norm(psi)
            rho = np.outer(psi, psi.conj())

            phi = rng.standard_normal(2) + 1j * rng.standard_normal(2)
            phi /= np.linalg.norm(phi)
            sig = np.outer(phi, phi.conj())

            F = fidelity(rho, sig)
            assert 0.0 - 1e-10 <= F <= 1.0 + 1e-10, (
                f"Fidelity out of bounds: F={F}"
            )


# ---------------------------------------------------------------------------
# Test: compose_kraus preserves CPTP
# ---------------------------------------------------------------------------

class TestComposeKraus:
    """Composed channels should remain CPTP."""

    def test_composed_channel_cptp(self):
        channels = [depolarizing(0.05) for _ in range(5)]
        composed = compose_kraus(channels)
        assert verify_cptp(composed, tol=1e-6), (
            "Composed depolarizing channels are not CPTP"
        )

    def test_composed_mixed_cptp(self):
        channels = [
            depolarizing(0.03),
            amplitude_damping(0.05),
            dephasing(0.04),
        ]
        composed = compose_kraus(channels)
        assert verify_cptp(composed, tol=1e-6), (
            "Composed mixed channels are not CPTP"
        )


# ---------------------------------------------------------------------------
# Test: depolarizing channel output is correct
# ---------------------------------------------------------------------------

class TestChannelOutput:
    """Verify channel outputs match expected analytical results."""

    def test_depolarizing_maximally_mixed(self, sigma):
        """Depolarizing channel preserves the maximally mixed state."""
        kraus = depolarizing(0.3)
        out = apply_channel(sigma, kraus)
        assert np.allclose(out, sigma, atol=1e-10), (
            "Depolarizing channel should preserve I/2"
        )

    def test_full_depolarizing_gives_maximally_mixed(self, rho_zero):
        """p=1 depolarizing maps everything to I/2."""
        kraus = depolarizing(1.0)
        out = apply_channel(rho_zero, kraus)
        expected = np.eye(2, dtype=complex) / 2
        assert np.allclose(out, expected, atol=1e-10), (
            "Full depolarizing should give I/2"
        )

    def test_amplitude_damping_ground_state(self, rho_zero):
        """Amplitude damping preserves |0>."""
        kraus = amplitude_damping(0.5)
        out = apply_channel(rho_zero, kraus)
        assert np.allclose(out, rho_zero, atol=1e-10), (
            "Amplitude damping should preserve |0>"
        )
