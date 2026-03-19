# Email to QuTech / Quantum Inspire

**To:** Richard.versluis@tno.nl
**CC:** l.dicarlo@tudelft.nl
**Subject:** τ-chrono: open-source tool extends usable circuit depth 2.5x on Tuna-9 — validated with real data

---

Dear Dr. Versluis,

My name is Sheng-Kai Huang, an independent quantum computing researcher based in Taiwan. I have been using Quantum Inspire and the Tuna-9 processor for my research, and I wanted to share a result that may benefit your platform users.

I developed τ-chrono, an open-source noise prediction tool based on the Petz recovery map. It predicts circuit fidelity more accurately than the standard independent gate model. All validation was performed on your Tuna-9 hardware with 4096 shots per circuit.

**Key results (real Tuna-9 data, all open source):**

- Fidelity prediction: τ-chrono is closer to actual measured fidelity than the naive model at ALL 10 tested depths, with 26.4% average improvement
- Cost savings: 29% fewer QPU shots needed for Bernstein-Vazirani (67% at 8 oracle repetitions)
- Depth ceiling: 3-qubit entangling mirror circuit usable up to 50 gates (vs 20 gates with naive model) — 2.5x extension
- Overhead: zero additional calibration circuits, ~20ms CPU computation
- API: `pip install tau-chrono`, then `predict_circuit(qc)` returns whether the circuit will produce useful output

The practical benefit: Quantum Inspire users can run deeper circuits with confidence, because τ-chrono tells them more accurately which circuits will still work. This means better VQE energies, more algorithm iterations, and less wasted QPU time.

Everything is MIT licensed and open source:
- GitHub: https://github.com/akaiHuang/tau-chrono
- Website: https://tau-chrono.pages.dev
- PyPI: https://pypi.org/project/tau-chrono/
- Raw T-9 data included in the repository

**I see three possible levels of engagement:**

1. **Mention in documentation** (minimal effort): Add a note in QI docs that τ-chrono is available for noise prediction. I can write the tutorial.
2. **Platform integration** (medium effort): Integrate τ-chrono's `predict_circuit()` into the QI web interface, giving users a fidelity estimate before submitting jobs.
3. **Research collaboration** (deeper): I would be glad to contribute to Quantum Inspire's software development, particularly in noise characterization and circuit optimization.

I am happy to share any additional data or answer questions about the methodology. The paper (included in the repository) explains the approach and honestly states limitations.

Best regards,
Sheng-Kai Huang
akai@fawstudio.com
GitHub: https://github.com/akaiHuang
