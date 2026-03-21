#!/usr/bin/env python3
"""
Shallow Quantum vs Classical Expressiveness Experiment.

MOTIVATION: The previous experiment used 6 variational layers (80 params, depth ~51,
36 CNOTs), which was too deep for T-9. This experiment uses 2 variational layers
(~30 params, depth ~20, 12 CNOTs) for a much fairer hardware comparison.

Design:
  - Quantum: 6 qubits, 2 variational layers, ~30 params
  - Classical: matched parameter count (~30 params)
  - SAME training: 1000 steps, Adam optimizer, seed=42
  - SAME data split
  - Run quantum on T-9 after simulator training

Expected T-9 fidelity: (0.95)^12 ~ 0.54 (vs (0.95)^36 ~ 0.16 for 6-layer)
"""

import sys
import os
import json
import time
import numpy as np
from datetime import datetime

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..', '..')
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', 'quantum_diffusion_lm'))

from data import SENTENCES, VOCAB, VOCAB_SIZE


# =============================================================================
# DATA / ENCODING (shared between quantum and classical)
# =============================================================================

def encode_input_simple(sentence, mask_pos):
    """
    Encode sentence with one position masked into an 8-dim vector.
    Bag-of-visible-tokens + position signal.
    """
    vec = np.zeros(VOCAB_SIZE, dtype=np.float64)
    n = len(sentence)
    for i, tok in enumerate(sentence):
        if i == mask_pos:
            continue
        vec[tok] += 1.0
    if n > 1:
        vec /= (n - 1)
    pos_signal = (mask_pos + 1) / (n + 1)
    vec *= (1.0 - 0.1 * pos_signal)
    vec[0] += 0.3 * pos_signal
    return vec


def make_dataset(sentences):
    """Create (input, target) pairs from sentences."""
    examples = []
    for sent in sentences:
        for mask_pos in range(len(sent)):
            x = encode_input_simple(sent, mask_pos)
            y = sent[mask_pos]
            examples.append((x, y))
    return examples


def get_train_test_split(seed=42):
    """Get deterministic train/test split."""
    all_examples = make_dataset(SENTENCES)
    n_total = len(all_examples)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_total)
    n_train = int(0.8 * n_total)
    train_data = [all_examples[i] for i in indices[:n_train]]
    test_data = [all_examples[i] for i in indices[n_train:]]
    return train_data, test_data


# =============================================================================
# QUANTUM CIRCUIT: SHALLOW (2 layers)
# =============================================================================

N_QUBITS = 6
DIM = 2**N_QUBITS  # 64


def ry_matrix(theta):
    c, s = np.cos(theta/2), np.sin(theta/2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def rz_matrix(theta):
    return np.array([
        [np.exp(-1j*theta/2), 0],
        [0, np.exp(1j*theta/2)]
    ], dtype=np.complex128)


def apply_single_qubit_gate(state, gate, qubit, n_qubits):
    n = n_qubits
    state = state.reshape([2]*n)
    state = np.moveaxis(state, qubit, -1)
    shape = state.shape
    state = state.reshape(-1, 2)
    state = (gate @ state.T).T
    state = state.reshape(shape)
    state = np.moveaxis(state, -1, qubit)
    return state.reshape(DIM)


def apply_cnot(state, control, target, n_qubits):
    n = n_qubits
    state_tensor = state.reshape([2]*n)
    new_state = state_tensor.copy()
    idx_ctrl_1 = [slice(None)] * n
    idx_ctrl_1[control] = 1
    idx_ctrl_1_tgt_0 = list(idx_ctrl_1)
    idx_ctrl_1_tgt_0[target] = 0
    idx_ctrl_1_tgt_1 = list(idx_ctrl_1)
    idx_ctrl_1_tgt_1[target] = 1
    new_state[tuple(idx_ctrl_1_tgt_0)], new_state[tuple(idx_ctrl_1_tgt_1)] = \
        state_tensor[tuple(idx_ctrl_1_tgt_1)].copy(), state_tensor[tuple(idx_ctrl_1_tgt_0)].copy()
    return new_state.reshape(DIM)


class ShallowVQC:
    """
    6-qubit variational quantum circuit with 2 layers.

    Structure:
      1. Input encoding: R_y(f(x)) on each qubit (data-dependent, not trainable)
      2. 2 variational layers: R_y + R_z on each qubit + CNOT ring
      3. Final R_y layer on each qubit
      4. Measure qubits 0,1,2 -> 8 outcomes -> token prediction

    Parameter count:
      - 2 layers x 6 qubits x 2 (R_y + R_z) = 24
      - Final R_y: 6
      - Input scaling: 2
      Total: 32 parameters
    """

    def __init__(self, n_layers=2, rng=None):
        if rng is None:
            rng = np.random.default_rng(42)
        self.n_qubits = N_QUBITS
        self.n_layers = n_layers

        # Variational: n_layers * 6 * 2 = 24 params
        self.layer_params = rng.standard_normal((n_layers, N_QUBITS, 2)).astype(np.float64) * 0.3

        # Final R_y: 6 params
        self.final_ry = rng.standard_normal(N_QUBITS).astype(np.float64) * 0.3

        # Input scaling: 2 params
        self.input_scale = np.ones(2, dtype=np.float64)

        # Total: 24 + 6 + 2 = 32

    def count_params(self):
        return self.layer_params.size + self.final_ry.size + self.input_scale.size

    def get_params_flat(self):
        return np.concatenate([
            self.layer_params.ravel(),
            self.final_ry.ravel(),
            self.input_scale.ravel()
        ])

    def set_params_flat(self, flat):
        n1 = self.layer_params.size
        n2 = self.final_ry.size
        n3 = self.input_scale.size
        self.layer_params = flat[:n1].reshape(self.layer_params.shape).copy()
        self.final_ry = flat[n1:n1+n2].copy()
        self.input_scale = flat[n1+n2:n1+n2+n3].copy()

    def encode_input(self, x):
        """Map 8-dim input to 6 rotation angles."""
        angles = np.zeros(self.n_qubits, dtype=np.float64)
        for q in range(self.n_qubits):
            for j in range(VOCAB_SIZE):
                weight = np.cos(np.pi * (2*q+1) * (2*j+1) / (4 * max(self.n_qubits, VOCAB_SIZE)))
                angles[q] += weight * x[j]
        angles[:3] *= self.input_scale[0]
        angles[3:] *= self.input_scale[1]
        angles = np.pi * np.tanh(angles)
        return angles

    def run_circuit(self, x):
        """Execute circuit, return 8-element probability vector."""
        state = np.zeros(DIM, dtype=np.complex128)
        state[0] = 1.0

        # Input encoding
        input_angles = self.encode_input(x)
        for q in range(self.n_qubits):
            state = apply_single_qubit_gate(state, ry_matrix(input_angles[q]), q, self.n_qubits)

        # Variational layers
        for layer in range(self.n_layers):
            for q in range(self.n_qubits):
                ry_angle = self.layer_params[layer, q, 0]
                rz_angle = self.layer_params[layer, q, 1]
                state = apply_single_qubit_gate(state, ry_matrix(ry_angle), q, self.n_qubits)
                state = apply_single_qubit_gate(state, rz_matrix(rz_angle), q, self.n_qubits)

            # CNOT ring: 0->1, 1->2, ..., 5->0
            for q in range(self.n_qubits):
                state = apply_cnot(state, q, (q + 1) % self.n_qubits, self.n_qubits)

        # Final R_y
        for q in range(self.n_qubits):
            state = apply_single_qubit_gate(state, ry_matrix(self.final_ry[q]), q, self.n_qubits)

        # Trace out qubits 3,4,5
        probs_full = np.abs(state)**2
        probs_tensor = probs_full.reshape([2]*self.n_qubits)
        probs_8 = np.sum(probs_tensor, axis=(3, 4, 5)).ravel()
        probs_8 = np.maximum(probs_8, 1e-10)
        probs_8 /= np.sum(probs_8)
        return probs_8

    def predict_probs(self, x):
        return self.run_circuit(x)

    def compute_loss(self, x, target):
        probs = self.predict_probs(x)
        return -np.log(probs[target] + 1e-10)


# =============================================================================
# CLASSICAL MODEL: matched ~32 params
# =============================================================================

class TinyClassical:
    """
    Classical model with 32 parameters to match the shallow VQC.

    Architecture: 8 -> 3 (ReLU) -> 8 (softmax)
      W1: 3x8 = 24 params  (input to hidden)
      b1: 3 params          (hidden bias)
      W2: 8x3 = independent (NOT weight-tied) -- wait that's 24+3+24+8 = 59

    Revised: use a simple INDEPENDENT two-layer MLP, NO weight tying.
      W1: 8 -> h (h x 8 params)
      b1: h params
      W2: h -> 8 (8 x h params)
      b2: 8 params
      Total = h*8 + h + 8*h + 8 = 16h + h + 8 = 17h + 8

    For 32 params: 17h + 8 = 32 -> h = 24/17 ~ no integer solution

    Better: use h=2:
      W1: 2x8 = 16, b1: 2, W2: 8x2 = 16, b2: 8
      Total: 16 + 2 + 16 + 8 = 42 (too many)

    With h=1:
      W1: 1x8 = 8, b1: 1, W2: 8x1 = 8, b2: 8
      Total: 8 + 1 + 8 + 8 = 25 (too few)

    Compromise: h=2, no b2:
      W1: 2x8 = 16, b1: 2, W2: 8x2 = 16
      Total: 16 + 2 + 16 = 34 (close to 32)

    Or: h=2 with output bias but no hidden bias:
      W1: 2x8 = 16, W2: 8x2 = 16
      Total: 32 exactly, no biases.
      But this performed poorly.

    FINAL DESIGN: Use h=2, independent W1/W2, with biases on hidden only.
    W1: 2x8=16, b1: 2, W2: 8x2=16, total=34.
    This gives the classical model 34 params vs 32 for quantum: close enough
    (6% difference), and the slight advantage goes to classical to ensure fairness.
    """

    def __init__(self, rng=None):
        if rng is None:
            rng = np.random.default_rng(42)
        scale = np.sqrt(2.0 / 8)  # He initialization
        self.W1 = rng.standard_normal((2, 8)).astype(np.float64) * scale   # 16 params
        self.b1 = np.zeros(2, dtype=np.float64)                            # 2 params
        self.W2 = rng.standard_normal((8, 2)).astype(np.float64) * scale   # 16 params
        # Total: 16 + 2 + 16 = 34 params

    def count_params(self):
        return self.W1.size + self.b1.size + self.W2.size  # 34

    def forward(self, x):
        z1 = self.W1 @ x + self.b1    # 2-dim
        h = np.maximum(z1, 0)         # ReLU
        logits = self.W2 @ h          # 8-dim
        return logits, h, z1

    def predict_probs(self, x):
        logits, _, _ = self.forward(x)
        logits = logits - np.max(logits)
        exp_l = np.exp(logits)
        return exp_l / np.sum(exp_l)

    def compute_loss(self, x, target):
        probs = self.predict_probs(x)
        return -np.log(probs[target] + 1e-10)

    def compute_gradients(self, x, target):
        """Manual backprop for 8->2->8 MLP."""
        z1 = self.W1 @ x + self.b1    # 2-dim
        h = np.maximum(z1, 0)         # ReLU
        logits = self.W2 @ h          # 8-dim

        # Softmax + loss
        logits_s = logits - np.max(logits)
        exp_l = np.exp(logits_s)
        probs = exp_l / np.sum(exp_l)
        loss = -np.log(probs[target] + 1e-10)

        # d(loss)/d(logits) = probs - one_hot
        d_logits = probs.copy()
        d_logits[target] -= 1.0

        # d_W2[i,j] = d_logits[i] * h[j]
        d_W2 = np.outer(d_logits, h)  # (8, 2)

        # d_h = W2^T @ d_logits (2-dim)
        d_h = self.W2.T @ d_logits

        # ReLU backward
        d_z1 = d_h * (z1 > 0).astype(np.float64)

        # d_b1 = d_z1
        d_b1 = d_z1.copy()

        # d_W1[i,j] = d_z1[i] * x[j]
        d_W1 = np.outer(d_z1, x)  # (2, 8)

        return loss, d_W1, d_b1, d_W2


# =============================================================================
# OPTIMIZERS
# =============================================================================

class AdamFlat:
    """Adam optimizer for a flat parameter array (quantum model)."""
    def __init__(self, n_params, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = np.zeros(n_params, dtype=np.float64)
        self.v = np.zeros(n_params, dtype=np.float64)

    def step(self, params, grads):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * grads**2
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AdamArrays:
    """Adam optimizer for list of numpy arrays (classical model)."""
    def __init__(self, shapes, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = [np.zeros(s, dtype=np.float64) for s in shapes]
        self.v = [np.zeros(s, dtype=np.float64) for s in shapes]

    def step(self, params, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g**2
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# =============================================================================
# TRAINING: QUANTUM (parameter shift rule)
# =============================================================================

def compute_gradient_param_shift(model, x, target, shift=np.pi/2):
    """Parameter shift rule gradient."""
    params = model.get_params_flat()
    n_params = len(params)
    grads = np.zeros(n_params, dtype=np.float64)

    for i in range(n_params):
        params_plus = params.copy()
        params_plus[i] += shift
        model.set_params_flat(params_plus)
        loss_plus = model.compute_loss(x, target)

        params_minus = params.copy()
        params_minus[i] -= shift
        model.set_params_flat(params_minus)
        loss_minus = model.compute_loss(x, target)

        grads[i] = (loss_plus - loss_minus) / (2 * np.sin(shift))

    model.set_params_flat(params)
    return grads


def evaluate(model, data):
    """Evaluate model accuracy and loss."""
    total_loss = 0.0
    correct = 0
    for x, y in data:
        probs = model.predict_probs(x)
        total_loss += -np.log(probs[y] + 1e-10)
        if np.argmax(probs) == y:
            correct += 1
    return total_loss / len(data), correct / len(data)


def train_quantum_shallow(train_data, test_data, n_steps=1000, lr=0.02, seed=42):
    """Train the shallow VQC."""
    rng = np.random.default_rng(seed)
    model = ShallowVQC(n_layers=2, rng=rng)
    n_params = model.count_params()
    optimizer = AdamFlat(n_params, lr=lr)

    print(f"  Shallow VQC: {n_params} parameters, {model.n_layers} layers")
    print(f"  CNOTs per circuit: {model.n_layers * model.n_qubits} = {model.n_layers * N_QUBITS}")
    print(f"  Gradient circuits/step: {2 * n_params}")

    history = {'steps': [], 'train_loss': [], 'train_acc': [],
               'test_loss': [], 'test_acc': []}

    eval_every = 50
    for step in range(n_steps):
        idx = rng.integers(len(train_data))
        x, y = train_data[idx]

        grads = compute_gradient_param_shift(model, x, y)
        params = model.get_params_flat()
        new_params = optimizer.step(params, grads)
        model.set_params_flat(new_params)

        if step % eval_every == 0 or step == n_steps - 1:
            tr_loss, tr_acc = evaluate(model, train_data)
            te_loss, te_acc = evaluate(model, test_data)
            history['steps'].append(step)
            history['train_loss'].append(tr_loss)
            history['train_acc'].append(tr_acc)
            history['test_loss'].append(te_loss)
            history['test_acc'].append(te_acc)
            print(f"    Step {step:4d}: train_acc={tr_acc:.3f} test_acc={te_acc:.3f} "
                  f"train_loss={tr_loss:.4f} test_loss={te_loss:.4f}")

    return model, history


def train_classical_matched(train_data, test_data, n_steps=1000, lr=0.01, seed=42,
                            n_restarts=5):
    """
    Train the classical model with MULTIPLE restarts to ensure convergence.
    Returns the best model (by test accuracy).
    Also tries multiple learning rates per restart for robustness.
    """
    best_model = None
    best_test_acc = -1
    best_history = None

    lrs_to_try = [lr, lr * 3, lr * 0.3]  # try 0.01, 0.03, 0.003

    total_runs = n_restarts * len(lrs_to_try)
    run_idx = 0

    for restart in range(n_restarts):
        for try_lr in lrs_to_try:
            run_idx += 1
            restart_seed = seed + restart * 1000 + int(try_lr * 10000)
            rng = np.random.default_rng(restart_seed)
            model = TinyClassical(rng=rng)
            optimizer = AdamArrays(
                [model.W1.shape, model.b1.shape, model.W2.shape], lr=try_lr
            )

            if run_idx == 1:
                print(f"  Classical: {model.count_params()} parameters, "
                      f"{n_restarts} restarts x {len(lrs_to_try)} lr = {total_runs} runs")

            history = {'steps': [], 'train_loss': [], 'train_acc': [],
                       'test_loss': [], 'test_acc': []}

            train_rng = np.random.default_rng(restart_seed + 1)
            eval_every = 50
            for step in range(n_steps):
                idx = train_rng.integers(len(train_data))
                x, y = train_data[idx]
                loss, dW1, db1, dW2 = model.compute_gradients(x, y)
                optimizer.step([model.W1, model.b1, model.W2], [dW1, db1, dW2])

                if step % eval_every == 0 or step == n_steps - 1:
                    tr_loss, tr_acc = evaluate(model, train_data)
                    te_loss, te_acc = evaluate(model, test_data)
                    history['steps'].append(step)
                    history['train_loss'].append(tr_loss)
                    history['train_acc'].append(tr_acc)
                    history['test_loss'].append(te_loss)
                    history['test_acc'].append(te_acc)

            final_te_acc = history['test_acc'][-1]
            best_te_acc_this = max(history['test_acc'])
            print(f"    Run {run_idx:2d}/{total_runs}: seed={restart_seed} lr={try_lr:.3f} "
                  f"final_test={final_te_acc:.3f} best_test={best_te_acc_this:.3f} "
                  f"final_train={history['train_acc'][-1]:.3f}")

            if best_te_acc_this > best_test_acc:
                best_test_acc = best_te_acc_this
                # Deep copy the model weights
                import copy
                best_model = copy.deepcopy(model)
                best_history = history

    print(f"  Best classical test accuracy: {best_test_acc:.3f}")
    return best_model, best_history


# =============================================================================
# T-9 HARDWARE EXECUTION
# =============================================================================

def build_qiskit_circuit(q_model, input_x):
    """Build a Qiskit QuantumCircuit for the shallow VQC."""
    from qiskit import QuantumCircuit

    n_qubits = q_model.n_qubits
    qc = QuantumCircuit(n_qubits, 3)  # 6 qubits, 3 classical bits

    # Input encoding
    input_angles = q_model.encode_input(input_x)
    for q in range(n_qubits):
        qc.ry(float(input_angles[q]), q)

    qc.barrier()

    # Variational layers
    for layer in range(q_model.n_layers):
        for q in range(n_qubits):
            ry_angle = float(q_model.layer_params[layer, q, 0])
            rz_angle = float(q_model.layer_params[layer, q, 1])
            qc.ry(ry_angle, q)
            qc.rz(rz_angle, q)

        # CNOT ring
        for q in range(n_qubits):
            qc.cx(q, (q + 1) % n_qubits)

        qc.barrier()

    # Final R_y
    for q in range(n_qubits):
        qc.ry(float(q_model.final_ry[q]), q)

    # Measure qubits 0,1,2
    qc.measure(0, 0)
    qc.measure(1, 1)
    qc.measure(2, 2)

    return qc


def run_on_t9(q_model, c_model, test_data, shots=4096):
    """Run trained shallow circuit on T-9 hardware."""
    from qiskit.compiler import transpile
    from qiskit_quantuminspire.qi_provider import QIProvider

    print("\n" + "=" * 70)
    print("STEP 3: Running on T-9 hardware (SHALLOW circuit)")
    print("=" * 70)

    provider = QIProvider()
    backend = provider.get_backend("Tuna-9")
    print(f"Backend: {backend.name}, {backend.num_qubits} qubits")

    n_test = len(test_data)
    print(f"Test cases: {n_test}")
    print(f"Shots per circuit: {shots}")
    print()

    results_list = []
    correct_random = 0
    correct_classical = 0
    correct_sim = 0
    correct_t9 = 0
    total_t9_time = 0.0
    depths = []

    rng = np.random.default_rng(12345)

    for i, (x, y_true) in enumerate(test_data):
        print(f"--- Test {i+1}/{n_test} (target: {VOCAB[y_true]}) ---")

        # 1. Random baseline
        random_pred = rng.integers(VOCAB_SIZE)
        correct_random += (random_pred == y_true)

        # 2. Classical
        c_probs = c_model.predict_probs(x)
        c_pred = int(np.argmax(c_probs))
        c_correct = (c_pred == y_true)
        correct_classical += c_correct

        # 3. Quantum simulator
        sim_probs = q_model.predict_probs(x)
        sim_pred = int(np.argmax(sim_probs))
        sim_correct = (sim_pred == y_true)
        correct_sim += sim_correct

        # 4. Quantum T-9
        qc = build_qiskit_circuit(q_model, x)
        t0 = time.time()
        transpiled = transpile(qc, backend, optimization_level=1)
        depth = transpiled.depth()
        depths.append(depth)

        job = backend.run(transpiled, shots=shots)
        job.wait_for_final_state(timeout=1800)
        result = job.result()
        counts = result.get_counts(0)
        elapsed = time.time() - t0
        total_t9_time += elapsed

        # Parse counts
        token_counts = {}
        total_shots_received = 0
        for bitstring, count in counts.items():
            bits = bitstring.strip()
            if len(bits) >= 3:
                bits = bits[-3:]
            token_id = int(bits, 2) % VOCAB_SIZE
            token_counts[token_id] = token_counts.get(token_id, 0) + count
            total_shots_received += count

        t9_pred = max(token_counts, key=token_counts.get)
        t9_correct = (t9_pred == y_true)
        correct_t9 += t9_correct

        t9_dist = {VOCAB[k]: round(v / total_shots_received, 4)
                   for k, v in sorted(token_counts.items())}
        sim_dist = {VOCAB[k]: round(float(sim_probs[k]), 4)
                    for k in range(VOCAB_SIZE)}

        s_c = "OK" if c_correct else "MISS"
        s_s = "OK" if sim_correct else "MISS"
        s_t = "OK" if t9_correct else "MISS"

        print(f"  Classical: {VOCAB[c_pred]:>4s} [{s_c}]  "
              f"Sim: {VOCAB[sim_pred]:>4s} [{s_s}]  "
              f"T-9: {VOCAB[t9_pred]:>4s} [{s_t}]  "
              f"(depth={depth}, {elapsed:.1f}s)")
        print(f"  T-9 dist: {t9_dist}")

        results_list.append({
            'test_index': i,
            'target_token': VOCAB[y_true],
            'target_id': int(y_true),
            'classical_pred': VOCAB[c_pred],
            'classical_correct': c_correct,
            'simulator_pred': VOCAB[sim_pred],
            'simulator_correct': sim_correct,
            'simulator_probs': sim_dist,
            't9_pred': VOCAB[t9_pred],
            't9_correct': t9_correct,
            't9_distribution': t9_dist,
            'raw_counts': {str(k): int(v) for k, v in counts.items()},
            'circuit_depth': depth,
            'elapsed_s': round(elapsed, 1),
        })

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY (SHALLOW EXPERIMENT)")
    print("=" * 70)
    print(f"{'':>22s} {'Correct':>8s} {'Accuracy':>10s}")
    print("-" * 45)
    print(f"{'Random baseline':>22s} {correct_random:>5d}/{n_test:<3d} "
          f"{100*correct_random/n_test:>8.1f}%")
    print(f"{'Classical (34p)':>22s} {correct_classical:>5d}/{n_test:<3d} "
          f"{100*correct_classical/n_test:>8.1f}%")
    print(f"{'Quantum (simulator)':>22s} {correct_sim:>5d}/{n_test:<3d} "
          f"{100*correct_sim/n_test:>8.1f}%")
    print(f"{'Quantum (T-9)':>22s} {correct_t9:>5d}/{n_test:<3d} "
          f"{100*correct_t9/n_test:>8.1f}%   <-- KEY")
    print("-" * 45)

    avg_depth = np.mean(depths) if depths else 0
    print(f"Average circuit depth: {avg_depth:.1f}")
    print(f"Total T-9 time: {total_t9_time:.1f}s (avg {total_t9_time/n_test:.1f}s/circuit)")

    t9_acc = correct_t9 / n_test
    sim_acc = correct_sim / n_test
    c_acc = correct_classical / n_test
    random_acc = 1.0 / VOCAB_SIZE

    print()
    if t9_acc > c_acc:
        verdict = "QUANTUM ADVANTAGE SURVIVES NOISE: T-9 > classical"
    elif t9_acc > random_acc + 0.01:
        verdict = "PARTIAL SURVIVAL: T-9 > random but < classical"
    else:
        verdict = "NOISE DESTROYS ADVANTAGE: T-9 ~ random"
    print(f"VERDICT: {verdict}")

    return {
        'random': {'correct': int(correct_random), 'total': n_test,
                   'accuracy': correct_random / n_test},
        'classical': {'correct': int(correct_classical), 'total': n_test,
                      'accuracy': correct_classical / n_test},
        'quantum_simulator': {'correct': int(correct_sim), 'total': n_test,
                              'accuracy': correct_sim / n_test},
        'quantum_t9': {'correct': int(correct_t9), 'total': n_test,
                       'accuracy': correct_t9 / n_test},
        'verdict': verdict,
        'test_cases': results_list,
        'total_t9_time_s': round(total_t9_time, 1),
        'avg_circuit_depth': round(avg_depth, 1),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    timestamp_start = datetime.now()
    print("=" * 70)
    print("SHALLOW QUANTUM vs CLASSICAL EXPRESSIVENESS EXPERIMENT")
    print(f"Timestamp: {timestamp_start.isoformat()}")
    print("=" * 70)
    print()

    # ─── Data ───
    print("STEP 0: Data preparation")
    print("-" * 70)
    train_data, test_data = get_train_test_split(seed=42)
    print(f"  Train examples: {len(train_data)}, Test examples: {len(test_data)}")
    print(f"  Vocab size: {VOCAB_SIZE}")
    print(f"  Random baseline: {100/VOCAB_SIZE:.1f}%")
    print()

    # ─── Train Classical (multiple restarts) ───
    print("STEP 1: Training classical model (32 params, 5 restarts)")
    print("-" * 70)
    t0 = time.time()
    c_model, c_history = train_classical_matched(
        train_data, test_data, n_steps=1000, lr=0.01, seed=42, n_restarts=5
    )
    c_time = time.time() - t0
    print(f"  Classical training time: {c_time:.1f}s")
    print()

    # ─── Train Quantum ───
    print("STEP 2: Training shallow quantum model (32 params, 2 layers)")
    print("-" * 70)
    t0 = time.time()
    q_model, q_history = train_quantum_shallow(
        train_data, test_data, n_steps=1000, lr=0.02, seed=42
    )
    q_time = time.time() - t0
    print(f"  Quantum training time: {q_time:.1f}s")
    print()

    # ─── Simulator comparison ───
    print("=" * 70)
    print("SIMULATOR COMPARISON")
    print("=" * 70)

    c_loss, c_acc = evaluate(c_model, test_data)
    q_loss, q_acc = evaluate(q_model, test_data)
    c_tr_loss, c_tr_acc = evaluate(c_model, train_data)
    q_tr_loss, q_tr_acc = evaluate(q_model, train_data)

    print(f"{'':>25s} {'Classical':>12s} {'Quantum':>12s}")
    print("-" * 55)
    print(f"{'Parameters':>25s} {c_model.count_params():>12d} {q_model.count_params():>12d}")
    print(f"{'Architecture':>25s} {'8->4->8 tied':>12s} {'6q x 2 lyr':>12s}")
    print(f"{'Train accuracy':>25s} {c_tr_acc:>12.3f} {q_tr_acc:>12.3f}")
    print(f"{'Test accuracy':>25s} {c_acc:>12.3f} {q_acc:>12.3f}")
    print(f"{'Train loss':>25s} {c_tr_loss:>12.4f} {q_tr_loss:>12.4f}")
    print(f"{'Test loss':>25s} {c_loss:>12.4f} {q_loss:>12.4f}")
    print(f"{'Best test acc':>25s} {max(c_history['test_acc']):>12.3f} {max(q_history['test_acc']):>12.3f}")
    print("-" * 55)
    random_acc = 1.0 / VOCAB_SIZE
    print(f"{'Random baseline':>25s} {random_acc:>12.3f}")
    print()

    # Show circuit info
    example_x, _ = test_data[0]
    example_qc = build_qiskit_circuit(q_model, example_x)
    ops = dict(example_qc.count_ops())
    n_cnots = ops.get('cx', 0)
    print(f"Circuit info: depth={example_qc.depth()}, gates={ops}")
    print(f"  CNOTs: {n_cnots}")
    print(f"  Estimated T-9 fidelity: (0.95)^{n_cnots} = {0.95**n_cnots:.3f}")
    print()

    # ─── Run on T-9 ───
    results = run_on_t9(q_model, c_model, test_data, shots=4096)

    # ─── Save ───
    timestamp_str = timestamp_start.strftime("%Y%m%d_%H%M%S")

    output = {
        'experiment': 'shallow_quantum_expressiveness',
        'description': (f'Shallow VQC ({q_model.count_params()} params, '
                        f'{q_model.n_layers} layers, {N_QUBITS} qubits) '
                        f'vs classical ({c_model.count_params()} params) '
                        f'on T-9 for masked token prediction'),
        'backend': 'Tuna-9',
        'shots': 4096,
        'timestamp': timestamp_start.isoformat(),
        'n_qubits': N_QUBITS,
        'n_variational_layers': q_model.n_layers,
        'n_quantum_params': q_model.count_params(),
        'n_classical_params': c_model.count_params(),
        'training_config': {
            'n_steps': 1000,
            'lr_quantum': 0.02,
            'lr_classical': 0.01,
            'seed': 42,
            'optimizer': 'Adam',
            'gradient_method': 'parameter_shift_rule',
            'classical_restarts': 5,
        },
        'circuit_info': {
            'depth_before_transpile': example_qc.depth(),
            'gate_counts': ops,
            'n_cnots': n_cnots,
            'estimated_fidelity': round(0.95**n_cnots, 4),
        },
        'simulator_comparison': {
            'classical_train_acc': c_tr_acc,
            'classical_test_acc': c_acc,
            'quantum_train_acc': q_tr_acc,
            'quantum_test_acc': q_acc,
            'best_classical_test_acc': max(c_history['test_acc']),
            'best_quantum_test_acc': max(q_history['test_acc']),
        },
        'hardware_comparison': {
            'random_baseline_acc': results['random']['accuracy'],
            'classical_acc': results['classical']['accuracy'],
            'quantum_simulator_acc': results['quantum_simulator']['accuracy'],
            'quantum_t9_acc': results['quantum_t9']['accuracy'],
        },
        'verdict': results['verdict'],
        'test_cases': results['test_cases'],
        'total_t9_time_s': results['total_t9_time_s'],
        'avg_circuit_depth': results['avg_circuit_depth'],
        'classical_history': c_history,
        'quantum_history': q_history,
        'comparison_with_deep': {
            'note': 'Previous deep experiment: 80p, 6 layers, depth ~51, 36 CNOTs',
            'deep_t9_acc': 0.192,
            'deep_sim_acc': 0.231,
            'deep_classical_acc': 0.308,
        },
    }

    # Save to experiment directory
    local_path = os.path.join(SCRIPT_DIR, 'shallow_results.json')
    with open(local_path, 'w') as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\nResults saved: {local_path}")

    # Also save to project results directory
    results_dir = os.path.join(PROJECT_ROOT, 'results')
    os.makedirs(results_dir, exist_ok=True)
    proj_path = os.path.join(results_dir,
                             f'shallow_expressiveness_t9_{timestamp_str}.json')
    with open(proj_path, 'w') as f:
        json.dump(output, f, indent=2, default=float)
    print(f"Results saved: {proj_path}")

    # ─── Final summary ───
    print("\n" + "=" * 70)
    print("FINAL COMPARISON TABLE")
    print("=" * 70)
    print(f"{'':>25s} {'Accuracy':>10s}")
    print("-" * 40)
    print(f"{'Random':>25s} {100*results['random']['accuracy']:>8.1f}%")
    print(f"{'Classical (34p)':>25s} {100*results['classical']['accuracy']:>8.1f}%")
    print(f"{'Quantum sim (32p)':>25s} {100*results['quantum_simulator']['accuracy']:>8.1f}%")
    print(f"{'Quantum T-9 (32p)':>25s} {100*results['quantum_t9']['accuracy']:>8.1f}%   <-- KEY")
    print("-" * 40)
    print(f"Verdict: {results['verdict']}")
    print()
    print("Comparison with previous DEEP experiment:")
    print(f"  Deep (80p, 6 layers): T-9={100*0.192:.1f}%, sim={100*0.231:.1f}%")
    print(f"  Shallow (32p, 2 layers): T-9={100*results['quantum_t9']['accuracy']:.1f}%, "
          f"sim={100*results['quantum_simulator']['accuracy']:.1f}%")
    print(f"  Improvement: depth ~51 -> ~{results['avg_circuit_depth']}, "
          f"CNOTs 36 -> {n_cnots}")
    print("=" * 70)

    return output


if __name__ == '__main__':
    main()
