# tau-chrono

[![Tests](https://github.com/akaiHuang/tau-chrono/actions/workflows/test.yml/badge.svg)](https://github.com/akaiHuang/tau-chrono/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**τ-chrono: noise tracking for quantum circuits via Petz recovery maps.**
26.4% more accurate on average than independent gate models on real hardware.

**Author:** Sheng-Kai Huang (akai@fawstudio.com)
**Website:** [tau-chrono.pages.dev](https://tau-chrono.pages.dev)

## Quickstart

```bash
pip install tau-chrono
```

### Option 1: With Qiskit circuit (recommended)

```python
from qiskit import QuantumCircuit
from tau_chrono.api import predict_circuit

# Your quantum circuit
qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)
# ... add more gates ...

# One line: should I run this circuit?
result = predict_circuit(qc)
print(result)
# PredictionResult(
#   f_tauchrono = 0.8308  (GO)
#   f_naive     = 0.8165  (GO)
#   should_run  = True
# )

if result.should_run:
    backend.run(qc)  # run with confidence
else:
    print("Circuit too noisy, skip")
```

### Option 2: Just gate names (no Qiskit needed)

```python
from tau_chrono.api import predict_gates

result = predict_gates(["h", "cx", "cx", "h", "cx", "cx", "h"])
print(result.should_run)      # True
print(result.f_tauchrono)     # 0.82
print(result.f_naive)         # 0.80
```

### Option 3: Custom gate error rates

```python
from tau_chrono.api import predict_gates

# Use your own hardware calibration data
my_errors = {"cx": 0.008, "h": 0.002, "sx": 0.001}
result = predict_gates(["h", "cx", "cx", "h"] * 20, gate_errors=my_errors)
print(result)
```

### Option 4: Low-level API

```python
import numpy as np
from tau_chrono import depolarizing, tau_chrono_compose

gates = [depolarizing(0.05) for _ in range(20)]
rho = np.array([[1, 0], [0, 0]], dtype=complex)
sigma = np.eye(2, dtype=complex) / 2

result = tau_chrono_compose(gates, sigma_0=sigma, rho=rho)
print(f"Naive:      tau = {result.tau_multiplicative_total:.3f}")
print(f"tau-chrono: tau = {result.tau_bayesian_total:.3f}")
print(f"Improvement: {result.improvement_percent:.1f}%")
```

## Key Results (QuTech Tuna-9 Hardware)

All results from real NISQ hardware experiments on the QuTech Tuna-9 superconducting processor (9 transmon qubits, 4096 shots).

| Experiment | Result |
|---|---|
| Depth scaling (all depths) | τ-chrono is closer to measured fidelity at ALL 10 tested depths |
| Depth scaling (average) | 26.4% more accurate than independent model on average |
| Depth scaling (depth 50) | 48.3% more accurate than independent model |
| Bernstein-Vazirani (4 qubits) | P_success from 0.68 (n_rep=1) down to 0.08 (n_rep=12) — real measured values |
| H2 VQE | τ-chrono keeps depth 4 viable (tau=0.49); naive says stop (tau=0.60) |
| Composition inequality | Verified across all 65 circuit configurations |

### Depth Scaling

![Depth scaling results](results/fig_depth_scaling.png)

τ-chrono prediction is closer to actual measured fidelity than the independent model at ALL tested depths. Average improvement: 26.4%. Peak improvement at depth 50: 48.3%.

### Bernstein-Vazirani

![Bernstein-Vazirani results](results/fig_bernstein_vazirani.png)

Real measured P_success values: 0.68 at n_rep=1, decreasing to 0.08 at n_rep=12.

### H2 VQE

![H2 VQE results](results/fig_h2_vqe.png)

τ-chrono tracking doubles usable ansatz depth (2 to 4). At depth 4: naive tau=0.60 (STOP), τ-chrono tau=0.49 (GO).

### Experiment A: Cost Savings

![Cost savings](results/fig_expA_cost_savings.png)

τ-chrono saves 29% total QPU shots on Bernstein-Vazirani. At n_rep=8, naive requires 3x shots for majority voting; τ-chrono knows the circuit is reliable and runs once — saving 67%.

### Experiment B: Depth Ceiling

![Depth ceiling](results/fig_expB_depth_ceiling.png)

3-qubit entangling mirror circuit on T-9. Naive says STOP at 20 gates; τ-chrono correctly identifies that 50-gate circuits still work (F=0.67). Depth extension: 2.5x. Two circuits saved that naive would have rejected.

## QEC Intelligence

tau-chrono can predict whether quantum error correction will help or hurt on your hardware, using the same calibration data you already have. Zero additional circuits needed.

```python
from tau_chrono.api import should_enable_qec

# Check if QEC will help on your hardware
result = should_enable_qec({"cx": 0.05, "h": 0.02})
print(result)
# QECRecommendation(
#   enable = False
#   predicted_ler_with_qec    = 0.130000
#   predicted_ler_without_qec = 0.050000
#   threshold_error_rate      = 0.0300
#   reason = "Physical error rate 5.0% exceeds threshold 3.0%.
#             QEC will likely INCREASE logical error rate."
# )
```

**Validated against real T-9 data:** At 4-5% CNOT error, QEC made things 7.2x worse on real hardware. `should_enable_qec` correctly predicts this -- it returns `enable=False` for T-9 error rates.

### Decoder Weights

Generate per-qubit MWPM decoder weights directly from tau characterization:

```python
from tau_chrono.api import qec_decoder_weights

# Per-qubit calibration -> decoder weights for PyMatching
weights = qec_decoder_weights(
    gate_errors={"cx": 0.01},
    per_qubit_errors={0: {"cx": 0.005}, 1: {"cx": 0.02}, 2: {"cx": 0.01}},
)
# weights[0] > weights[2] > weights[1]  (quieter qubit = higher weight)
```

### Health Monitoring

Monitor QEC health from syndrome statistics without additional circuits:

```python
from tau_chrono.api import qec_health_monitor

alert = qec_health_monitor(syndrome_history)
if not alert.healthy:
    print(alert.message)  # "Noise drift detected: syndrome rate increased by 45%..."
```

## Why It Works

Independent gate noise models assume each gate fails independently. In reality, noise saturates: a qubit that's already noisy can't get much noisier. The Petz recovery map (Petz, 1986) tracks this saturation through the circuit by propagating a Bayesian reference state alongside the signal state. τ-chrono uses this retrodiction structure to give more accurate fidelity predictions.

## Interactive Demo

```bash
pip install tau-chrono[demo]
streamlit run demo.py
```

Adjust noise type, error rate, and circuit depth interactively.

## Honest Limitations

1. **Small hardware only.** All results from QuTech Tuna-9 (5-9 qubits). Not tested on larger devices.
2. **Improvement is noise-dependent.** Large on noisy hardware, diminishes on low-noise hardware.
3. **Depolarizing approximation.** Structured noise (coherent errors, leakage) is not captured.
4. **Not tested on IBM or Google hardware.** Portability is plausible but unverified.

## Theoretical Foundation

All theoretical tools are due to their original authors:

- D. Petz, *Commun. Math. Phys.* **105**, 123 (1986) -- Petz recovery map
- A. J. Parzygnat and F. Buscemi, *Quantum* **7**, 1013 (2023) -- Unique retrodiction functor
- M. Junge et al., *Ann. Henri Poincare* **19**, 2955 (2018) -- Strengthened data processing inequality

## Development

```bash
git clone https://github.com/akaiHuang/tau-chrono.git
cd tau-chrono
pip install -e ".[dev]"
pytest tests/
```

## Citation

```bibtex
@software{Huang2026TauChrono,
  author  = {Huang, Sheng-Kai},
  title   = {\tau-chrono: Noise Tracking via Petz Recovery Maps},
  year    = {2026},
  url     = {https://github.com/akaiHuang/tau-chrono}
}
```

## License

MIT License. See [LICENSE](LICENSE).
