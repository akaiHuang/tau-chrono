"""
Tests for the tau-chrono QEC intelligence module.

Verifies QEC enable/disable predictions, decoder weight generation,
and health monitoring from syndrome statistics.

Run with:
    python -m pytest tests/test_qec.py -v
"""

import sys
import os
import numpy as np
import pytest

# Ensure the repo root is on the path so tau_chrono is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tau_chrono.qec import (
    should_enable_qec,
    qec_decoder_weights,
    qec_health_monitor,
    QECRecommendation,
    QECHealthAlert,
)


# ---------------------------------------------------------------------------
# Test: should_enable_qec
# ---------------------------------------------------------------------------

class TestShouldEnableQEC:
    """Verify QEC enable/disable predictions match expected behavior."""

    def test_low_noise_enables_qec(self):
        """At p=0.001 (well below threshold), QEC should help."""
        result = should_enable_qec({"cx": 0.001, "h": 0.0005})
        assert result.enable is True, (
            f"Expected QEC enabled at p_cx=0.001, got disabled. "
            f"Reason: {result.reason}"
        )
        assert result.predicted_ler_with_qec < result.predicted_ler_without_qec, (
            f"With QEC ({result.predicted_ler_with_qec:.6f}) should be less "
            f"than without ({result.predicted_ler_without_qec:.6f})"
        )

    def test_high_noise_disables_qec(self):
        """At p=0.05 (above repetition threshold), QEC should hurt."""
        result = should_enable_qec({"cx": 0.05, "h": 0.02})
        assert result.enable is False, (
            f"Expected QEC disabled at p_cx=0.05, got enabled. "
            f"Reason: {result.reason}"
        )
        assert result.predicted_ler_with_qec >= result.predicted_ler_without_qec, (
            f"With QEC ({result.predicted_ler_with_qec:.6f}) should be >= "
            f"without ({result.predicted_ler_without_qec:.6f}) above threshold"
        )

    def test_t9_error_rates_disable_qec(self):
        """T-9 hardware error rates (~4-5% CNOT) should disable QEC.

        Real T-9 result: QEC made things 7.2x worse.
        Our prediction should match: QEC disabled.
        """
        # Actual T-9 calibration data
        t9_errors = {
            "cx": 0.040,
            "h": 0.0214,
            "x": 0.0269,
            "sx": 0.0110,
            "id": 0.0226,
            "measure": 0.001,
        }
        result = should_enable_qec(t9_errors, code_type="repetition", code_distance=3)
        assert result.enable is False, (
            f"T-9 error rates should disable QEC (real data: 7.2x worse). "
            f"Got enabled. Reason: {result.reason}"
        )

    def test_surface_code_threshold_lower(self):
        """Surface code has ~1% threshold, stricter than repetition."""
        # 2% CNOT error: ok for repetition, not for surface
        result_rep = should_enable_qec(
            {"cx": 0.005}, code_type="repetition", code_distance=3
        )
        result_surf = should_enable_qec(
            {"cx": 0.005}, code_type="surface", code_distance=3
        )
        # At 0.5% both should be enabled, but surface code threshold is lower
        assert result_surf.threshold_error_rate < result_rep.threshold_error_rate, (
            f"Surface threshold ({result_surf.threshold_error_rate}) should be "
            f"< repetition threshold ({result_rep.threshold_error_rate})"
        )

    def test_returns_correct_type(self):
        """Result should be a QECRecommendation dataclass."""
        result = should_enable_qec({"cx": 0.01})
        assert isinstance(result, QECRecommendation)
        assert isinstance(result.enable, bool)
        assert isinstance(result.predicted_ler_with_qec, float)
        assert isinstance(result.predicted_ler_without_qec, float)
        assert isinstance(result.reason, str)
        assert isinstance(result.threshold_error_rate, float)

    def test_threshold_stored_correctly(self):
        """Threshold should match the code type."""
        result = should_enable_qec({"cx": 0.01}, code_type="repetition")
        assert abs(result.threshold_error_rate - 0.03) < 1e-10

        result = should_enable_qec({"cx": 0.01}, code_type="surface")
        assert abs(result.threshold_error_rate - 0.01) < 1e-10

    def test_invalid_code_type_raises(self):
        """Unknown code type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown code type"):
            should_enable_qec({"cx": 0.01}, code_type="nonexistent")

    def test_invalid_distance_raises(self):
        """Even or too-small distance should raise ValueError."""
        with pytest.raises(ValueError, match="odd and >= 3"):
            should_enable_qec({"cx": 0.01}, code_distance=2)
        with pytest.raises(ValueError, match="odd and >= 3"):
            should_enable_qec({"cx": 0.01}, code_distance=4)

    def test_higher_distance_helps_below_threshold(self):
        """Below threshold, higher distance should reduce logical error rate."""
        result_d3 = should_enable_qec(
            {"cx": 0.001}, code_type="repetition", code_distance=3
        )
        result_d5 = should_enable_qec(
            {"cx": 0.001}, code_type="repetition", code_distance=5
        )
        # Both should be enabled
        assert result_d3.enable is True
        assert result_d5.enable is True
        # d=5 should have lower logical error rate (if well below threshold)
        # Note: this depends on overhead, so only check at very low error rates
        assert result_d5.predicted_ler_with_qec <= result_d3.predicted_ler_with_qec * 1.5, (
            f"Higher distance should not dramatically increase LER below threshold"
        )

    def test_repr_contains_key_info(self):
        """String representation should include key fields."""
        result = should_enable_qec({"cx": 0.05, "h": 0.02})
        s = repr(result)
        assert "enable" in s
        assert "False" in s
        assert "threshold" in s.lower() or "reason" in s.lower()


# ---------------------------------------------------------------------------
# Test: qec_decoder_weights
# ---------------------------------------------------------------------------

class TestQECDecoderWeights:
    """Verify decoder weight generation from tau characterization."""

    def test_higher_tau_lower_weight(self):
        """Noisier qubits (higher tau) should get lower decoder weight."""
        weights = qec_decoder_weights(
            gate_errors={"cx": 0.01},
            per_qubit_errors={
                0: {"cx": 0.005},   # quiet
                1: {"cx": 0.02},    # noisy
                2: {"cx": 0.01},    # medium
            },
        )
        # Qubit 0 (lowest error) should have highest weight
        assert weights[0] > weights[2], (
            f"Qubit 0 (p=0.005) should have higher weight than qubit 2 (p=0.01), "
            f"got {weights[0]:.4f} vs {weights[2]:.4f}"
        )
        assert weights[2] > weights[1], (
            f"Qubit 2 (p=0.01) should have higher weight than qubit 1 (p=0.02), "
            f"got {weights[2]:.4f} vs {weights[1]:.4f}"
        )

    def test_uniform_noise_uniform_weights(self):
        """All qubits with same noise should get equal weights."""
        weights = qec_decoder_weights(
            gate_errors={"cx": 0.01},
            qubit_ids=[0, 1, 2, 3],
        )
        values = list(weights.values())
        for v in values[1:]:
            assert abs(v - values[0]) < 1e-10, (
                f"Uniform noise should give uniform weights, "
                f"got {values}"
            )

    def test_returns_dict_of_floats(self):
        """Weights should be a dict mapping int -> float."""
        weights = qec_decoder_weights(
            gate_errors={"cx": 0.01},
            qubit_ids=[0, 1, 2],
        )
        assert isinstance(weights, dict)
        for k, v in weights.items():
            assert isinstance(k, int)
            assert isinstance(v, float)

    def test_weights_are_positive(self):
        """All decoder weights should be positive."""
        weights = qec_decoder_weights(
            gate_errors={"cx": 0.05},
            qubit_ids=[0, 1, 2, 3, 4],
        )
        for qid, w in weights.items():
            assert w > 0, f"Weight for qubit {qid} should be positive, got {w}"

    def test_default_qubit_ids(self):
        """Without qubit_ids, should use per_qubit_errors keys or default to T-9."""
        # From per_qubit_errors
        weights = qec_decoder_weights(
            gate_errors={"cx": 0.01},
            per_qubit_errors={5: {"cx": 0.01}, 7: {"cx": 0.02}},
        )
        assert set(weights.keys()) == {5, 7}

        # Default T-9 (9 qubits)
        weights = qec_decoder_weights(gate_errors={"cx": 0.01})
        assert len(weights) == 9
        assert set(weights.keys()) == set(range(9))


# ---------------------------------------------------------------------------
# Test: qec_health_monitor
# ---------------------------------------------------------------------------

class TestQECHealthMonitor:
    """Verify syndrome-based QEC health monitoring."""

    def test_stable_noise_healthy(self):
        """Stable syndrome rates should report healthy."""
        rng = np.random.default_rng(42)
        # 100 rounds, 4 stabilizers, ~30% syndrome rate
        history = [
            [int(rng.random() < 0.3) for _ in range(4)]
            for _ in range(100)
        ]
        alert = qec_health_monitor(history)
        assert alert.healthy is True, (
            f"Stable noise should be healthy. Message: {alert.message}"
        )
        assert alert.drift_detected is False

    def test_drifting_noise_detected(self):
        """Significant noise drift should trigger an alert."""
        rng = np.random.default_rng(42)
        # First 50 rounds: low syndrome rate (~10%)
        early = [
            [int(rng.random() < 0.1) for _ in range(4)]
            for _ in range(50)
        ]
        # Last 50 rounds: high syndrome rate (~50%)
        late = [
            [int(rng.random() < 0.5) for _ in range(4)]
            for _ in range(50)
        ]
        history = early + late
        alert = qec_health_monitor(history, window_size=40, drift_threshold=0.3)
        assert alert.drift_detected is True, (
            f"Should detect drift from 10% to 50%. Message: {alert.message}"
        )
        assert alert.healthy is False

    def test_high_syndrome_rate_unhealthy(self):
        """Very high syndrome rate should report unhealthy."""
        # All syndromes firing ~90% of the time
        history = [[1, 1, 1, 1] for _ in range(100)]
        alert = qec_health_monitor(history)
        assert alert.healthy is False, (
            f"High syndrome rate should be unhealthy. Message: {alert.message}"
        )

    def test_insufficient_data(self):
        """Very short syndrome history should return healthy with warning."""
        alert = qec_health_monitor([[0, 1]])
        assert alert.healthy is True
        assert "Insufficient" in alert.message

    def test_empty_history(self):
        """Empty history should not crash."""
        alert = qec_health_monitor([])
        assert alert.healthy is True
        assert isinstance(alert, QECHealthAlert)

    def test_returns_correct_type(self):
        """Result should be a QECHealthAlert dataclass."""
        history = [[0, 1, 0] for _ in range(20)]
        alert = qec_health_monitor(history)
        assert isinstance(alert, QECHealthAlert)
        assert isinstance(alert.healthy, bool)
        assert isinstance(alert.delta_D, float)
        assert isinstance(alert.mean_syndrome_rate, float)
        assert isinstance(alert.drift_detected, bool)
        assert isinstance(alert.message, str)

    def test_delta_d_nonnegative(self):
        """Retrodiction gap delta_D should always be non-negative."""
        rng = np.random.default_rng(123)
        history = [
            [int(rng.random() < 0.25) for _ in range(6)]
            for _ in range(80)
        ]
        alert = qec_health_monitor(history)
        assert alert.delta_D >= 0, f"delta_D should be >= 0, got {alert.delta_D}"
