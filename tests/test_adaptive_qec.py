"""
Tests for tau-chrono adaptive QEC module.

Verifies:
1. Tau composition is correct (subadditivity)
2. Extraction planning triggers at the right points
3. Adaptive strategy produces fewer syndrome rounds than fixed
4. Monte Carlo simulation produces reasonable error rates
5. Edge cases (zero noise, very high noise)

Run with:
    python -m pytest tests/test_adaptive_qec.py -v
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tau_chrono.adaptive_qec import (
    TauTriggeredQEC,
    RepetitionCodeSimulator,
    _gate_tau,
    _compose_tau,
    run_adaptive_qec_experiment,
    format_results_table,
    results_to_json,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def willow_errors():
    """Google Willow hardware error profile."""
    return {
        "cx": 0.003, "cz": 0.003, "h": 0.001,
        "x": 0.001, "z": 0.0005, "measure": 0.003, "default": 0.001,
    }


@pytest.fixture
def ibm_errors():
    """IBM Eagle hardware error profile."""
    return {
        "cx": 0.008, "cz": 0.008, "h": 0.003,
        "x": 0.003, "z": 0.001, "measure": 0.005, "default": 0.003,
    }


@pytest.fixture
def simple_circuit():
    """A simple 20-gate circuit."""
    return ["h", "cx", "x", "cx", "z"] * 4


@pytest.fixture
def long_circuit():
    """A 100-gate circuit."""
    rng = np.random.default_rng(123)
    gates = []
    for _ in range(100):
        if rng.random() < 0.3:
            gates.append("cx")
        else:
            gates.append(rng.choice(["h", "x", "z"]))
    return gates


# ---------------------------------------------------------------------------
# Test: tau composition
# ---------------------------------------------------------------------------

class TestTauComposition:
    """Verify tau composition properties."""

    def test_compose_tau_zero(self):
        """Composing with zero tau should return the other tau."""
        assert abs(_compose_tau(0.0, 0.0)) < 1e-15
        assert abs(_compose_tau(0.1, 0.0) - 0.1) < 1e-15
        assert abs(_compose_tau(0.0, 0.2) - 0.2) < 1e-15

    def test_compose_tau_subadditive(self):
        """Composed tau should be <= sum of individual taus."""
        tau_a, tau_b = 0.1, 0.2
        composed = _compose_tau(tau_a, tau_b)
        assert composed <= tau_a + tau_b + 1e-10

    def test_compose_tau_formula(self):
        """Check exact formula: tau_ab = tau_a + tau_b - tau_a * tau_b."""
        tau_a, tau_b = 0.15, 0.25
        expected = tau_a + tau_b - tau_a * tau_b
        assert abs(_compose_tau(tau_a, tau_b) - expected) < 1e-15

    def test_compose_tau_one(self):
        """Composing with tau=1 should give 1."""
        assert abs(_compose_tau(1.0, 0.5) - 1.0) < 1e-15
        assert abs(_compose_tau(0.5, 1.0) - 1.0) < 1e-15

    def test_compose_tau_monotonic(self):
        """More noise should always increase composed tau."""
        tau = 0.0
        for _ in range(20):
            tau_new = _compose_tau(tau, 0.05)
            assert tau_new >= tau - 1e-15
            tau = tau_new


# ---------------------------------------------------------------------------
# Test: gate tau computation
# ---------------------------------------------------------------------------

class TestGateTau:
    """Verify single-gate tau values are sensible."""

    def test_gate_tau_positive(self, willow_errors):
        """Gate tau should be positive for nonzero error rate."""
        for gate_name in ["cx", "h", "x"]:
            tau = _gate_tau(gate_name, willow_errors)
            assert tau > 0, f"tau({gate_name}) should be > 0"

    def test_gate_tau_ordering(self, willow_errors):
        """Higher error rate gates should have higher tau."""
        tau_z = _gate_tau("z", willow_errors)
        tau_h = _gate_tau("h", willow_errors)
        tau_cx = _gate_tau("cx", willow_errors)
        assert tau_z <= tau_h + 1e-10
        assert tau_h <= tau_cx + 1e-10

    def test_gate_tau_bounded(self, willow_errors):
        """Tau should be in [0, 1]."""
        for gate_name in ["cx", "h", "x", "z"]:
            tau = _gate_tau(gate_name, willow_errors)
            assert 0.0 <= tau <= 1.0


# ---------------------------------------------------------------------------
# Test: extraction planning
# ---------------------------------------------------------------------------

class TestExtractionPlanning:
    """Verify the tau-triggered extraction planning."""

    def test_extraction_triggers(self, willow_errors, simple_circuit):
        """Should trigger at least one extraction for a long enough circuit."""
        sim = TauTriggeredQEC(
            gate_errors=willow_errors,
            tau_threshold=0.01,  # low threshold -> frequent triggers
        )
        events, points = sim.plan_extractions(simple_circuit)
        assert len(points) > 0, "Should trigger at least one extraction"

    def test_high_threshold_no_extraction(self, willow_errors, simple_circuit):
        """Very high threshold should produce zero extractions on short circuit."""
        sim = TauTriggeredQEC(
            gate_errors=willow_errors,
            tau_threshold=0.99,  # very high
        )
        events, points = sim.plan_extractions(simple_circuit)
        assert len(points) == 0, "Should not trigger with threshold=0.99"

    def test_tau_resets_after_extraction(self, willow_errors):
        """After extraction, accumulated tau should reset to near zero."""
        sim = TauTriggeredQEC(
            gate_errors=willow_errors,
            tau_threshold=0.01,
        )
        circuit = ["cx"] * 50
        events, points = sim.plan_extractions(circuit)
        for p in points:
            # The gate right after an extraction should have low accumulated tau
            if p + 1 < len(events):
                assert events[p + 1].tau_accumulated < sim.tau_threshold

    def test_more_extractions_on_noisy_hardware(self, willow_errors, ibm_errors, long_circuit):
        """Noisy hardware should trigger more extractions at the same threshold."""
        tau_th = 0.10
        sim_good = TauTriggeredQEC(gate_errors=willow_errors, tau_threshold=tau_th)
        sim_bad = TauTriggeredQEC(gate_errors=ibm_errors, tau_threshold=tau_th)

        _, points_good = sim_good.plan_extractions(long_circuit)
        _, points_bad = sim_bad.plan_extractions(long_circuit)

        assert len(points_bad) >= len(points_good), (
            f"Noisy hardware should trigger >= extractions: "
            f"good={len(points_good)}, bad={len(points_bad)}"
        )


# ---------------------------------------------------------------------------
# Test: repetition code simulator
# ---------------------------------------------------------------------------

class TestRepetitionCodeSimulator:
    """Verify the Monte Carlo repetition code simulator."""

    def test_no_error_no_logical_error(self):
        """With zero noise, no logical error should occur."""
        rng = np.random.default_rng(42)
        sim = RepetitionCodeSimulator(3, rng)
        sim.reset()
        assert not sim.has_logical_error()

    def test_single_flip_no_logical_error(self):
        """A single bit flip should be correctable with d=3."""
        rng = np.random.default_rng(42)
        sim = RepetitionCodeSimulator(3, rng)
        sim.reset()
        sim.data_qubits[0] = 1  # flip one qubit
        assert not sim.has_logical_error()  # majority vote: 0,0 wins

    def test_majority_flip_logical_error(self):
        """Flipping majority of qubits should cause logical error."""
        rng = np.random.default_rng(42)
        sim = RepetitionCodeSimulator(3, rng)
        sim.reset()
        sim.data_qubits[0] = 1
        sim.data_qubits[1] = 1  # 2 out of 3 flipped
        assert sim.has_logical_error()

    def test_high_noise_high_ler(self):
        """Very high error rate should produce high logical error rate."""
        rng = np.random.default_rng(42)
        sim = RepetitionCodeSimulator(3, rng)
        n_errors = 0
        shots = 1000
        for _ in range(shots):
            sim.reset()
            sim.apply_gate_error(0.4)  # 40% per qubit
            if sim.has_logical_error():
                n_errors += 1
        ler = n_errors / shots
        assert ler > 0.05, f"LER should be significant at 40% noise, got {ler}"

    def test_low_noise_low_ler(self):
        """Very low error rate should produce low logical error rate."""
        rng = np.random.default_rng(42)
        sim = RepetitionCodeSimulator(3, rng)
        n_errors = 0
        shots = 1000
        for _ in range(shots):
            sim.reset()
            sim.apply_gate_error(0.001)  # 0.1% per qubit per gate
            if sim.has_logical_error():
                n_errors += 1
        ler = n_errors / shots
        assert ler < 0.05, f"LER should be very low at 0.1% noise, got {ler}"


# ---------------------------------------------------------------------------
# Test: full strategy comparison
# ---------------------------------------------------------------------------

class TestStrategyComparison:
    """Verify full strategy comparison produces reasonable results."""

    def test_adaptive_fewer_rounds_than_fixed(self, willow_errors, long_circuit):
        """Adaptive should use fewer syndrome rounds than fixed on good hardware."""
        sim = TauTriggeredQEC(
            gate_errors=willow_errors,
            tau_threshold=0.15,
            seed=42,
        )
        result = sim.compare_strategies(
            gate_sequence=long_circuit,
            shots=2000,
            fixed_interval=10,
        )
        fixed = result.strategies[1]
        adaptive = result.strategies[2]
        assert adaptive.syndrome_rounds < fixed.syndrome_rounds, (
            f"Adaptive ({adaptive.syndrome_rounds} rounds) should use "
            f"fewer rounds than fixed ({fixed.syndrome_rounds} rounds)"
        )

    def test_adaptive_fewer_cnots_than_fixed(self, willow_errors, long_circuit):
        """Adaptive should have less CNOT overhead than fixed."""
        sim = TauTriggeredQEC(
            gate_errors=willow_errors,
            tau_threshold=0.15,
            seed=42,
        )
        result = sim.compare_strategies(
            gate_sequence=long_circuit,
            shots=2000,
            fixed_interval=10,
        )
        fixed = result.strategies[1]
        adaptive = result.strategies[2]
        assert adaptive.total_cnot_overhead < fixed.total_cnot_overhead

    def test_no_qec_worse_than_fixed(self, ibm_errors, long_circuit):
        """On noisy hardware, no QEC should have higher LER than fixed QEC."""
        sim = TauTriggeredQEC(
            gate_errors=ibm_errors,
            tau_threshold=0.08,
            seed=42,
        )
        result = sim.compare_strategies(
            gate_sequence=long_circuit,
            shots=5000,
            fixed_interval=10,
        )
        no_qec = result.strategies[0]
        fixed = result.strategies[1]
        # At 0.8% CZ with 100 gates, errors accumulate significantly
        # QEC with correction should help
        # Note: at very low error rates this might not always hold
        # due to syndrome extraction overhead, so we use IBM profile
        assert no_qec.logical_error_rate >= fixed.logical_error_rate - 0.01, (
            f"No QEC LER ({no_qec.logical_error_rate:.4f}) should be >= "
            f"fixed QEC LER ({fixed.logical_error_rate:.4f}) on noisy hardware"
        )

    def test_result_has_correct_structure(self, willow_errors, simple_circuit):
        """Comparison result should have all expected fields."""
        sim = TauTriggeredQEC(gate_errors=willow_errors, seed=42)
        result = sim.compare_strategies(simple_circuit, shots=100)

        assert len(result.strategies) == 3
        assert result.strategies[0].name == "No QEC"
        assert "Fixed" in result.strategies[1].name
        assert "Tau-triggered" in result.strategies[2].name
        assert result.circuit_length == len(simple_circuit)
        assert result.shots == 100
        assert isinstance(result.tau_trace, list)
        assert isinstance(result.extraction_points, list)


# ---------------------------------------------------------------------------
# Test: convenience function
# ---------------------------------------------------------------------------

class TestConvenienceFunction:
    """Test the run_adaptive_qec_experiment convenience wrapper."""

    def test_convenience_function(self, willow_errors, simple_circuit):
        """Convenience function should return valid results."""
        result = run_adaptive_qec_experiment(
            gate_sequence=simple_circuit,
            gate_errors=willow_errors,
            tau_threshold=0.1,
            shots=100,
        )
        assert len(result.strategies) == 3
        assert result.hardware_name == "custom"


# ---------------------------------------------------------------------------
# Test: output formatting
# ---------------------------------------------------------------------------

class TestOutputFormatting:
    """Test result formatting and serialization."""

    def test_format_table(self, willow_errors, simple_circuit):
        """Table formatting should produce non-empty string."""
        sim = TauTriggeredQEC(gate_errors=willow_errors, seed=42)
        result = sim.compare_strategies(simple_circuit, shots=100)
        table = format_results_table(result)
        assert len(table) > 100
        assert "No QEC" in table
        assert "Fixed" in table
        assert "Tau-triggered" in table

    def test_json_serializable(self, willow_errors, simple_circuit):
        """Results should be JSON-serializable."""
        sim = TauTriggeredQEC(gate_errors=willow_errors, seed=42)
        result = sim.compare_strategies(simple_circuit, shots=100)
        data = results_to_json(result)

        import json
        serialized = json.dumps(data)
        assert len(serialized) > 0

        parsed = json.loads(serialized)
        assert len(parsed["strategies"]) == 3
