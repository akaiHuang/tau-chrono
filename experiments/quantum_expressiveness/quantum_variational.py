"""
Variational Quantum Circuit for masked token prediction.

Architecture: 6-qubit parameterized circuit
  - Input encoding: R_y rotations on all 6 qubits based on input features
  - Variational layers: R_y + R_z on each qubit + CNOT entangling
  - Output: measure qubits 0,1,2 → 8 possible outcomes → token prediction

Parameter count:
  - Input encoding: 6 R_y gates (but these are DATA-dependent, not trainable)
  - Variational layers: each layer = 6 R_y + 6 R_z = 12 params + CNOTs (no params)
  - With 6 layers: 6 * 12 = 72 params
  - Final R_y layer: 6 R_y = 6 params  (removed R_z to hit exactly 78)
  - Plus 2 extra params: input scaling weights = 80 total

  Actually let's be precise:
  - 6 variational layers x (6 R_y + 6 R_z) = 72
  - 1 final layer x 6 R_y = 6
  - 2 trainable input scaling params = 2
  Total: 72 + 6 + 2 = 80 parameters exactly
"""

import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'quantum_diffusion_lm'))
from data import SENTENCES, VOCAB, VOCAB_SIZE

# Import the same encoding
from classical_tiny import encode_input_simple, make_dataset


# ─── Statevector-based quantum simulation ─────────────────────────────────────
# For training efficiency, we simulate the circuit using matrix multiplication
# on statevectors rather than Qiskit circuit execution.

N_QUBITS = 6
DIM = 2**N_QUBITS  # 64

def ry_matrix(theta):
    """Single-qubit R_y(theta) matrix."""
    c, s = np.cos(theta/2), np.sin(theta/2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)

def rz_matrix(theta):
    """Single-qubit R_z(theta) matrix."""
    return np.array([
        [np.exp(-1j*theta/2), 0],
        [0, np.exp(1j*theta/2)]
    ], dtype=np.complex128)

def apply_single_qubit_gate(state, gate, qubit, n_qubits):
    """Apply a single-qubit gate to a statevector efficiently."""
    # Reshape state into tensor form
    n = n_qubits
    state = state.reshape([2]*n)

    # Apply gate to the specified qubit axis
    # Move target qubit to last axis, apply gate, move back
    state = np.moveaxis(state, qubit, -1)
    shape = state.shape
    state = state.reshape(-1, 2)
    state = (gate @ state.T).T
    state = state.reshape(shape)
    state = np.moveaxis(state, -1, qubit)

    return state.reshape(DIM)

def apply_cnot(state, control, target, n_qubits):
    """Apply CNOT gate to a statevector."""
    n = n_qubits
    state_tensor = state.reshape([2]*n)
    new_state = state_tensor.copy()

    # When control qubit is |1>, flip target qubit
    # Build index slices
    idx_ctrl_1 = [slice(None)] * n
    idx_ctrl_1[control] = 1

    idx_ctrl_1_tgt_0 = list(idx_ctrl_1)
    idx_ctrl_1_tgt_0[target] = 0
    idx_ctrl_1_tgt_1 = list(idx_ctrl_1)
    idx_ctrl_1_tgt_1[target] = 1

    # Swap amplitudes where control=1
    new_state[tuple(idx_ctrl_1_tgt_0)], new_state[tuple(idx_ctrl_1_tgt_1)] = \
        state_tensor[tuple(idx_ctrl_1_tgt_1)].copy(), state_tensor[tuple(idx_ctrl_1_tgt_0)].copy()

    return new_state.reshape(DIM)


class VariationalQuantumCircuit:
    """
    6-qubit variational quantum circuit for token prediction.

    Structure:
      1. Input encoding: R_y(f(x)) on each qubit
      2. Variational layers: [R_y, R_z on each qubit] + [CNOT chain]
      3. Measurement of qubits 0,1,2 → 8 outcomes → token probabilities

    Total params: 80
    """

    def __init__(self, n_layers=6, rng=None):
        if rng is None:
            rng = np.random.default_rng(42)

        self.n_qubits = N_QUBITS
        self.n_layers = n_layers

        # Variational parameters
        # Each layer: 6 R_y + 6 R_z = 12 params
        self.layer_params = rng.standard_normal((n_layers, N_QUBITS, 2)).astype(np.float64) * 0.3
        # n_layers * N_QUBITS * 2 = 6 * 6 * 2 = 72

        # Final rotation layer: 6 R_y only
        self.final_ry = rng.standard_normal(N_QUBITS).astype(np.float64) * 0.3
        # 6 params

        # Input scaling parameters (trainable)
        self.input_scale = np.ones(2, dtype=np.float64)  # 2 params
        # Total: 72 + 6 + 2 = 80

    def count_params(self):
        return self.layer_params.size + self.final_ry.size + self.input_scale.size

    def get_params_flat(self):
        """Get all parameters as a flat array."""
        return np.concatenate([
            self.layer_params.ravel(),
            self.final_ry.ravel(),
            self.input_scale.ravel()
        ])

    def set_params_flat(self, flat):
        """Set all parameters from a flat array."""
        n1 = self.layer_params.size
        n2 = self.final_ry.size
        n3 = self.input_scale.size
        self.layer_params = flat[:n1].reshape(self.layer_params.shape).copy()
        self.final_ry = flat[n1:n1+n2].copy()
        self.input_scale = flat[n1+n2:n1+n2+n3].copy()

    def encode_input(self, x):
        """
        Convert 8-dim input vector to 6 rotation angles for input encoding.

        Strategy: linear combination of input features → 6 angles
        Using a fixed (non-trainable) mapping + trainable scaling.
        """
        # Map 8-dim input to 6 angles via a fixed pattern
        angles = np.zeros(self.n_qubits, dtype=np.float64)
        # Qubit 0-5: weighted sums of input features
        # Use a deterministic mixing pattern
        for q in range(self.n_qubits):
            # Each qubit gets a different linear combination
            for j in range(VOCAB_SIZE):
                # Use a fixed mixing matrix (Hadamard-like)
                weight = np.cos(np.pi * (2*q+1) * (2*j+1) / (4 * max(self.n_qubits, VOCAB_SIZE)))
                angles[q] += weight * x[j]

        # Apply trainable scaling
        angles[:3] *= self.input_scale[0]
        angles[3:] *= self.input_scale[1]

        # Scale to [0, pi] range
        angles = np.pi * np.tanh(angles)
        return angles

    def run_circuit(self, x):
        """
        Execute the quantum circuit and return measurement probabilities.

        Returns: 8-element array of probabilities (from measuring qubits 0,1,2)
        """
        # Initialize |000000⟩
        state = np.zeros(DIM, dtype=np.complex128)
        state[0] = 1.0

        # Input encoding
        input_angles = self.encode_input(x)
        for q in range(self.n_qubits):
            gate = ry_matrix(input_angles[q])
            state = apply_single_qubit_gate(state, gate, q, self.n_qubits)

        # Variational layers
        for layer in range(self.n_layers):
            # R_y and R_z on each qubit
            for q in range(self.n_qubits):
                ry_angle = self.layer_params[layer, q, 0]
                rz_angle = self.layer_params[layer, q, 1]
                state = apply_single_qubit_gate(state, ry_matrix(ry_angle), q, self.n_qubits)
                state = apply_single_qubit_gate(state, rz_matrix(rz_angle), q, self.n_qubits)

            # CNOT chain: 0→1, 1→2, 2→3, 3→4, 4→5, 5→0
            for q in range(self.n_qubits):
                control = q
                target = (q + 1) % self.n_qubits
                state = apply_cnot(state, control, target, self.n_qubits)

        # Final R_y layer
        for q in range(self.n_qubits):
            state = apply_single_qubit_gate(state, ry_matrix(self.final_ry[q]), q, self.n_qubits)

        # Measurement: trace out qubits 3,4,5 to get probabilities on qubits 0,1,2
        probs_full = np.abs(state)**2
        # Reshape into (2,2,2, 2,2,2) for qubits 0-5
        probs_tensor = probs_full.reshape([2]*self.n_qubits)
        # Sum over qubits 3,4,5 (last 3 axes)
        probs_8 = np.sum(probs_tensor, axis=(3, 4, 5))  # shape (2,2,2)
        probs_8 = probs_8.ravel()  # 8 probabilities

        # Ensure normalization
        probs_8 = np.maximum(probs_8, 1e-10)
        probs_8 /= np.sum(probs_8)

        return probs_8

    def predict_probs(self, x):
        """Return token prediction probabilities."""
        return self.run_circuit(x)

    def compute_loss(self, x, target):
        """Cross-entropy loss."""
        probs = self.predict_probs(x)
        return -np.log(probs[target] + 1e-10)


def compute_gradient_param_shift(model, x, target, shift=np.pi/2):
    """
    Compute gradients using the parameter shift rule.

    For each parameter θ_i:
      ∂L/∂θ_i = [L(θ_i + π/2) - L(θ_i - π/2)] / 2

    This requires 2 circuit evaluations per parameter.
    """
    params = model.get_params_flat()
    n_params = len(params)
    grads = np.zeros(n_params, dtype=np.float64)

    for i in range(n_params):
        # Forward shift
        params_plus = params.copy()
        params_plus[i] += shift
        model.set_params_flat(params_plus)
        loss_plus = model.compute_loss(x, target)

        # Backward shift
        params_minus = params.copy()
        params_minus[i] -= shift
        model.set_params_flat(params_minus)
        loss_minus = model.compute_loss(x, target)

        grads[i] = (loss_plus - loss_minus) / (2 * np.sin(shift))

    # Restore original params
    model.set_params_flat(params)
    return grads


class AdamOptimizer:
    """Adam optimizer for flat parameter array."""

    def __init__(self, n_params, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = np.zeros(n_params, dtype=np.float64)
        self.v = np.zeros(n_params, dtype=np.float64)

    def step(self, params, grads):
        """Update params and return new params array."""
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * grads**2
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def evaluate_model(model, data):
    """Evaluate model on a dataset."""
    total_loss = 0.0
    correct = 0
    for x, y in data:
        probs = model.predict_probs(x)
        total_loss += -np.log(probs[y] + 1e-10)
        if np.argmax(probs) == y:
            correct += 1
    return total_loss / len(data), correct / len(data)


def train_quantum(n_steps=1000, lr=0.01, seed=42, verbose=True, eval_every=50):
    """Train the variational quantum circuit and return training history."""
    rng = np.random.default_rng(seed)
    model = VariationalQuantumCircuit(n_layers=6, rng=rng)

    if verbose:
        print(f"Quantum VQC: {model.count_params()} parameters")

    # Create dataset (same encoding as classical)
    all_examples = make_dataset(SENTENCES)
    n_total = len(all_examples)

    # Split: 80% train, 20% test (SAME split as classical — same seed)
    split_rng = np.random.default_rng(42)  # deterministic split
    indices = split_rng.permutation(n_total)
    n_train = int(0.8 * n_total)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    train_data = [all_examples[i] for i in train_idx]
    test_data = [all_examples[i] for i in test_idx]

    if verbose:
        print(f"Train examples: {len(train_data)}, Test examples: {len(test_data)}")

    # Optimizer
    n_params = model.count_params()
    optimizer = AdamOptimizer(n_params, lr=lr)

    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': [],
        'steps': []
    }

    for step in range(n_steps):
        # Random example (SGD)
        idx = rng.integers(len(train_data))
        x, y = train_data[idx]

        # Compute gradient via parameter shift rule
        grads = compute_gradient_param_shift(model, x, y)

        # Update
        params = model.get_params_flat()
        new_params = optimizer.step(params, grads)
        model.set_params_flat(new_params)

        # Evaluate
        if step % eval_every == 0 or step == n_steps - 1:
            train_loss, train_acc = evaluate_model(model, train_data)
            test_loss, test_acc = evaluate_model(model, test_data)

            history['steps'].append(step)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)

            if verbose:
                print(f"  Step {step:4d}: train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
                      f"test_loss={test_loss:.4f} test_acc={test_acc:.3f}  "
                      f"[{2*n_params} circuits/step]")

    return model, history, train_data, test_data


if __name__ == '__main__':
    print("Testing quantum circuit...")
    model = VariationalQuantumCircuit(rng=np.random.default_rng(42))
    print(f"Parameters: {model.count_params()}")

    # Quick test
    x = encode_input_simple([0, 1, 2, 3, 0, 5], mask_pos=1)
    probs = model.predict_probs(x)
    print(f"Test probs: {probs}")
    print(f"Sum: {np.sum(probs):.6f}")

    print("\nStarting training (this will be slow due to parameter shift rule)...")
    model, history, _, _ = train_quantum(n_steps=200, lr=0.02, eval_every=50)
    print(f"\nFinal: train_acc={history['train_acc'][-1]:.3f}, test_acc={history['test_acc'][-1]:.3f}")
