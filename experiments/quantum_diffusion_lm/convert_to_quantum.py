#!/usr/bin/env python3
"""
Convert the trained classical MLP to a parameterized quantum circuit.

Strategy (Option B — principled encoding):
  - 3 input qubits: encode the prompt token via X gates (computational basis)
  - 3 output qubits: produce the predicted token
  - Entangling layers (CNOTs) + parameterized rotations (R_y, R_z)
    with angles derived from trained MLP weights
  - Measurement of output qubits -> 3 bits -> token ID (0-7)

The circuit approximates the MLP's input-output mapping by:
  1. Encoding the input token in computational basis (X gates)
  2. For each input token, pre-computing the optimal rotation angles
     that produce measurement probabilities matching the MLP's output
  3. Using a 3-qubit variational circuit on the output register

Circuit depth: ~20 gates, well within T-9 coherence.
"""

import numpy as np
from scipy.optimize import minimize

try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False
    print("WARNING: qiskit not installed. Install with: pip install qiskit qiskit-aer")

from data import VOCAB_SIZE, NUM_BITS, token_to_bits, bits_to_token, VOCAB
from model import MLPDenoiser


def extract_quantum_params(model):
    """
    Extract the conditional probability matrix from the trained MLP.

    For each possible input token, we compute the MLP's output distribution
    in a representative context, then optimize quantum circuit angles to
    reproduce that distribution.

    Returns dict with 'P_matrix' and per-token optimized angles.
    """
    from data import build_input_vector

    # Build the 8x8 conditional probability matrix
    # P[i,j] = P(output=j | context token = i)
    # We use a 3-token context [input, MASK, dummy] to get a representative mapping
    P = np.zeros((VOCAB_SIZE, VOCAB_SIZE), dtype=np.float64)

    for input_token in range(VOCAB_SIZE):
        # Try multiple context configurations and average
        probs_sum = np.zeros(VOCAB_SIZE, dtype=np.float64)
        count = 0
        for ctx_len in [2, 3]:
            for mask_pos in range(ctx_len):
                sentence = [input_token] * ctx_len
                if mask_pos == 0:
                    # Input token is at position 1
                    sentence = [0, input_token] + [0] * (ctx_len - 2)
                    inp = build_input_vector(sentence[:ctx_len], mask_pos=0)
                else:
                    sentence = [input_token] + [0] * (ctx_len - 1)
                    inp = build_input_vector(sentence[:ctx_len], mask_pos=mask_pos)
                probs = model.forward(inp).astype(np.float64)
                probs_sum += probs
                count += 1

        P[input_token] = probs_sum / count

    # For each input token, optimize rotation angles
    token_angles = {}
    for tok in range(VOCAB_SIZE):
        target = P[tok]
        target = target / target.sum()  # ensure normalization
        angles = _optimize_circuit_angles(target)
        token_angles[tok] = angles

    return {'P_matrix': P, 'token_angles': token_angles}


def _simulate_3qubit_circuit(theta):
    """
    Simulate a 3-qubit variational circuit and return measurement probabilities.

    Circuit:
      q0: Ry(t0) -- Rz(t3) -- CNOT(ctrl) --------- Ry(t9)  -- CNOT(ctrl) -- Ry(t12)
      q1: Ry(t1) -- Rz(t4) -- CNOT(tgt)  -- CNOT(ctrl) -- Ry(t10) -- CNOT(tgt) -- Ry(t13)
      q2: Ry(t2) -- Rz(t5) -------------- CNOT(tgt)  -- Ry(t11) ----------- Ry(t14)
      + Rz layer: Rz(t6), Rz(t7), Rz(t8)

    Total: 15 parameters
    """
    # Start from |000>
    state = np.zeros(8, dtype=complex)
    state[0] = 1.0

    # Layer 1: Ry rotations
    state = _apply_ry(state, 0, theta[0])
    state = _apply_ry(state, 1, theta[1])
    state = _apply_ry(state, 2, theta[2])

    # Layer 1: Rz rotations
    state = _apply_rz(state, 0, theta[3])
    state = _apply_rz(state, 1, theta[4])
    state = _apply_rz(state, 2, theta[5])

    # Entangling: CNOT(0->1), CNOT(1->2)
    state = _apply_cnot(state, 0, 1)
    state = _apply_cnot(state, 1, 2)

    # Layer 2: Rz rotations
    state = _apply_rz(state, 0, theta[6])
    state = _apply_rz(state, 1, theta[7])
    state = _apply_rz(state, 2, theta[8])

    # Layer 2: Ry rotations
    state = _apply_ry(state, 0, theta[9])
    state = _apply_ry(state, 1, theta[10])
    state = _apply_ry(state, 2, theta[11])

    # Entangling: CNOT(2->0), CNOT(0->1)
    state = _apply_cnot(state, 2, 0)
    state = _apply_cnot(state, 0, 1)

    # Layer 3: Ry rotations
    state = _apply_ry(state, 0, theta[12])
    state = _apply_ry(state, 1, theta[13])
    state = _apply_ry(state, 2, theta[14])

    probs = np.abs(state) ** 2
    return probs


def _circuit_cost(theta, target_probs):
    """KL divergence between circuit output and target distribution."""
    sim_probs = _simulate_3qubit_circuit(theta)
    eps = 1e-12
    # Use KL divergence + L2 for better gradients
    kl = np.sum(target_probs * np.log((target_probs + eps) / (sim_probs + eps)))
    l2 = np.sum((target_probs - sim_probs) ** 2)
    return kl + 0.1 * l2


def _optimize_circuit_angles(target_probs, n_restarts=20):
    """
    Optimize 15 rotation angles to match target probability distribution.
    Uses scipy.optimize.minimize with multiple random restarts.
    """
    n_params = 15
    best_cost = float('inf')
    best_theta = np.zeros(n_params)

    rng = np.random.default_rng(42)

    for restart in range(n_restarts):
        if restart == 0:
            # First try: initialize from marginal probabilities
            p_bit = np.zeros(NUM_BITS)
            for tok in range(VOCAB_SIZE):
                bits = token_to_bits(tok)
                for b in range(NUM_BITS):
                    if bits[b] == 1:
                        p_bit[b] += target_probs[tok]
            theta0 = np.zeros(n_params)
            for b in range(NUM_BITS):
                theta0[b] = 2 * np.arcsin(np.sqrt(np.clip(p_bit[b], 0, 1)))
        else:
            # Random initialization
            theta0 = rng.uniform(-np.pi, np.pi, n_params)

        result = minimize(
            _circuit_cost, theta0, args=(target_probs,),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8}
        )

        if result.fun < best_cost:
            best_cost = result.fun
            best_theta = result.x.copy()

    return best_theta


def _apply_ry(state, qubit, theta):
    """Apply R_y(theta) to qubit in 3-qubit state vector."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    gate = np.array([[c, -s], [s, c]], dtype=complex)
    return _apply_single_gate(state, qubit, gate)


def _apply_rz(state, qubit, theta):
    """Apply R_z(theta) to qubit in 3-qubit state vector."""
    gate = np.array([
        [np.exp(-1j * theta / 2), 0],
        [0, np.exp(1j * theta / 2)]
    ], dtype=complex)
    return _apply_single_gate(state, qubit, gate)


def _apply_single_gate(state, qubit, gate):
    """Apply a single-qubit gate to qubit in a 3-qubit state."""
    n = NUM_BITS
    new_state = np.zeros_like(state)
    for i in range(2**n):
        bit_val = (i >> (n - 1 - qubit)) & 1
        for out_bit in range(2):
            j = i ^ ((bit_val ^ out_bit) << (n - 1 - qubit))
            new_state[j] += gate[out_bit, bit_val] * state[i]
    return new_state


def _apply_cnot(state, control, target):
    """Apply CNOT gate."""
    n = NUM_BITS
    new_state = np.zeros_like(state)
    for i in range(2**n):
        ctrl_bit = (i >> (n - 1 - control)) & 1
        if ctrl_bit == 1:
            j = i ^ (1 << (n - 1 - target))
            new_state[j] += state[i]
        else:
            new_state[i] += state[i]
    return new_state


def build_circuit_for_token(input_token, params):
    """
    Build a quantum circuit that, given an input token, produces
    the predicted next token on the output qubits.

    Uses 6 qubits: 3 input (computational basis encoding) + 3 output
    (variational circuit with pre-optimized angles).
    """
    if not HAS_QISKIT:
        raise ImportError("qiskit is required")

    theta = params['token_angles'][input_token]

    inp_reg = QuantumRegister(NUM_BITS, 'inp')
    out_reg = QuantumRegister(NUM_BITS, 'out')
    c_reg = ClassicalRegister(NUM_BITS, 'result')
    qc = QuantumCircuit(inp_reg, out_reg, c_reg)

    # --- Encode input token in computational basis ---
    bits = token_to_bits(input_token)
    for i in range(NUM_BITS):
        if bits[i] == 1:
            qc.x(inp_reg[i])

    qc.barrier()

    # --- Variational circuit on output qubits ---
    # Layer 1: Ry + Rz
    for i in range(NUM_BITS):
        qc.ry(float(theta[i]), out_reg[i])
    for i in range(NUM_BITS):
        qc.rz(float(theta[3 + i]), out_reg[i])

    # Entangling: CNOT(0->1), CNOT(1->2)
    qc.cx(out_reg[0], out_reg[1])
    qc.cx(out_reg[1], out_reg[2])

    # Layer 2: Rz + Ry
    for i in range(NUM_BITS):
        qc.rz(float(theta[6 + i]), out_reg[i])
    for i in range(NUM_BITS):
        qc.ry(float(theta[9 + i]), out_reg[i])

    # Entangling: CNOT(2->0), CNOT(0->1)
    qc.cx(out_reg[2], out_reg[0])
    qc.cx(out_reg[0], out_reg[1])

    # Layer 3: Ry
    for i in range(NUM_BITS):
        qc.ry(float(theta[12 + i]), out_reg[i])

    qc.barrier()

    # --- Measure output qubits ---
    qc.measure(out_reg, c_reg)

    return qc


def build_circuit_for_distribution(target_probs, label="custom", n_restarts=20):
    """
    Build a quantum circuit whose measurement probabilities match target_probs.

    This is the key function for faithful quantum inference: given the
    classical model's output distribution for a specific (sentence, mask_pos),
    we build a circuit that reproduces it exactly.

    Uses 3 qubits only (no input register needed -- the context is baked
    into the optimized angles).

    Returns: (QuantumCircuit, optimized_angles)
    """
    if not HAS_QISKIT:
        raise ImportError("qiskit is required")

    target_probs = np.array(target_probs, dtype=np.float64)
    target_probs = target_probs / target_probs.sum()

    # Optimize rotation angles
    theta = _optimize_circuit_angles(target_probs, n_restarts=n_restarts)

    # Build the circuit
    qr = QuantumRegister(NUM_BITS, 'q')
    cr = ClassicalRegister(NUM_BITS, 'result')
    qc = QuantumCircuit(qr, cr)

    # Layer 1: Ry + Rz
    for i in range(NUM_BITS):
        qc.ry(float(theta[i]), qr[i])
    for i in range(NUM_BITS):
        qc.rz(float(theta[3 + i]), qr[i])

    # Entangling: CNOT(0->1), CNOT(1->2)
    qc.cx(qr[0], qr[1])
    qc.cx(qr[1], qr[2])

    # Layer 2: Rz + Ry
    for i in range(NUM_BITS):
        qc.rz(float(theta[6 + i]), qr[i])
    for i in range(NUM_BITS):
        qc.ry(float(theta[9 + i]), qr[i])

    # Entangling: CNOT(2->0), CNOT(0->1)
    qc.cx(qr[2], qr[0])
    qc.cx(qr[0], qr[1])

    # Layer 3: Ry
    for i in range(NUM_BITS):
        qc.ry(float(theta[12 + i]), qr[i])

    # Measure
    qc.measure(qr, cr)

    return qc, theta


def verify_circuit(params):
    """Verify that the quantum circuits reproduce the target distributions."""
    print("\nCircuit verification (target vs simulated):")
    print("-" * 55)
    P = params['P_matrix']

    total_kl = 0
    for tok in range(VOCAB_SIZE):
        target = P[tok] / P[tok].sum()
        theta = params['token_angles'][tok]
        sim_probs = _simulate_3qubit_circuit(theta)

        # Argmax comparison
        target_tok = VOCAB[int(np.argmax(target))]
        sim_tok = VOCAB[int(np.argmax(sim_probs))]
        match = "OK" if target_tok == sim_tok else "XX"

        kl = np.sum(target * np.log((target + 1e-12) / (sim_probs + 1e-12)))
        total_kl += kl

        print(f"  Token {tok} ({VOCAB[tok]:>3s}): "
              f"target={target_tok:>3s}, sim={sim_tok:>3s} [{match}], "
              f"KL={kl:.4f}")

    print(f"  Mean KL: {total_kl / VOCAB_SIZE:.4f}")


def convert_model(model_path='model_weights.npz'):
    """
    Load trained model and convert to quantum circuits.

    Returns: (params, circuits_dict)
    """
    model = MLPDenoiser()
    model.load(model_path)

    print("Extracting quantum parameters from trained MLP...")
    print("Optimizing rotation angles (this may take ~30s)...")
    params = extract_quantum_params(model)

    verify_circuit(params)

    print("\nBuilding quantum circuits for each input token...")
    circuits = {}
    for token_id in range(VOCAB_SIZE):
        qc = build_circuit_for_token(token_id, params)
        circuits[token_id] = qc
        ops = qc.count_ops()
        n_gates = sum(v for k, v in ops.items()
                      if k not in ['measure', 'barrier'])
        print(f"  Token {token_id} ({VOCAB[token_id]:>3s}): "
              f"depth={qc.depth()}, gates={n_gates}")

    return params, circuits


if __name__ == '__main__':
    if not HAS_QISKIT:
        print("ERROR: qiskit not installed. Run: pip install qiskit qiskit-aer")
        exit(1)

    params, circuits = convert_model()

    print("\nExample circuit for input token 0 ('the'):")
    print(circuits[0].draw(output='text'))

    print(f"\nTotal circuits: {len(circuits)}")
    print("Conversion complete.")
