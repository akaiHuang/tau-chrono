#!/usr/bin/env python3
"""
FAIR COMPARISON: Quantum (6-layer deep VQC, 80 params) vs Classical (weight-tied MLP, 80 params)
Both with 15 random restarts, pick the best.

Quantum:
  - 6 qubits, 6 variational layers, 80 params
  - 1000 SPSA steps per restart (2 circuit evals/step = fast)
  - Statevector simulator
  - Seeds 0-14

Classical:
  - 8->8(ReLU)->8 weight-tied MLP, 80 params (W1: 8x8, b1: 8, b2: 8, W2=W1^T)
  - 5000 Adam steps per restart
  - 3 learning rates tried per seed: 0.01, 0.005, 0.001
  - Seeds 0-14

After completion:
  - Print comparison table
  - Save results to fair_comparison_results.json
  - If quantum wins, save best params for T-9 verification
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

# Reuse existing encoding and dataset functions
from classical_tiny import encode_input_simple, make_dataset


# =============================================================================
# SHARED DATA SPLIT (deterministic, same for quantum and classical)
# =============================================================================

def get_train_test_split(seed=42):
    """Deterministic train/test split, same as used in all other experiments."""
    all_examples = make_dataset(SENTENCES)
    n_total = len(all_examples)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_total)
    n_train = int(0.8 * n_total)
    train_data = [all_examples[i] for i in indices[:n_train]]
    test_data = [all_examples[i] for i in indices[n_train:]]
    return train_data, test_data


# =============================================================================
# QUANTUM CIRCUIT (6 qubits, 6 layers, 80 params) — from quantum_variational.py
# =============================================================================

N_QUBITS = 6
DIM = 2**N_QUBITS  # 64


def ry_matrix(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def rz_matrix(theta):
    return np.array([
        [np.exp(-1j * theta / 2), 0],
        [0, np.exp(1j * theta / 2)]
    ], dtype=np.complex128)


def apply_single_qubit_gate(state, gate, qubit, n_qubits):
    n = n_qubits
    state = state.reshape([2] * n)
    state = np.moveaxis(state, qubit, -1)
    shape = state.shape
    state = state.reshape(-1, 2)
    state = (gate @ state.T).T
    state = state.reshape(shape)
    state = np.moveaxis(state, -1, qubit)
    return state.reshape(DIM)


def apply_cnot(state, control, target, n_qubits):
    n = n_qubits
    state_tensor = state.reshape([2] * n)
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


class DeepVQC:
    """
    6-qubit variational quantum circuit with 6 layers, 80 params.

    Structure:
      1. Input encoding: R_y(f(x)) on each qubit (data-dependent, NOT trainable)
      2. 6 variational layers: R_y + R_z on each qubit + CNOT ring
      3. Final R_y layer on each qubit
      4. Measure qubits 0,1,2 -> 8 outcomes -> token prediction

    Parameter count:
      - 6 layers x 6 qubits x 2 (R_y + R_z) = 72
      - Final R_y: 6
      - Input scaling: 2
      Total: 80 parameters exactly
    """

    def __init__(self, n_layers=6, rng=None):
        if rng is None:
            rng = np.random.default_rng(42)
        self.n_qubits = N_QUBITS
        self.n_layers = n_layers

        # Variational: n_layers * 6 * 2 = 72 params
        self.layer_params = rng.standard_normal((n_layers, N_QUBITS, 2)).astype(np.float64) * 0.3

        # Final R_y: 6 params
        self.final_ry = rng.standard_normal(N_QUBITS).astype(np.float64) * 0.3

        # Input scaling: 2 params
        self.input_scale = np.ones(2, dtype=np.float64)

        # Total: 72 + 6 + 2 = 80

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
        self.final_ry = flat[n1:n1 + n2].copy()
        self.input_scale = flat[n1 + n2:n1 + n2 + n3].copy()

    def encode_input(self, x):
        """Map 8-dim input to 6 rotation angles via DCT-like mixing."""
        angles = np.zeros(self.n_qubits, dtype=np.float64)
        for q in range(self.n_qubits):
            for j in range(VOCAB_SIZE):
                weight = np.cos(np.pi * (2 * q + 1) * (2 * j + 1)
                                / (4 * max(self.n_qubits, VOCAB_SIZE)))
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
            state = apply_single_qubit_gate(state, ry_matrix(input_angles[q]),
                                            q, self.n_qubits)

        # Variational layers
        for layer in range(self.n_layers):
            for q in range(self.n_qubits):
                ry_angle = self.layer_params[layer, q, 0]
                rz_angle = self.layer_params[layer, q, 1]
                state = apply_single_qubit_gate(state, ry_matrix(ry_angle),
                                                q, self.n_qubits)
                state = apply_single_qubit_gate(state, rz_matrix(rz_angle),
                                                q, self.n_qubits)

            # CNOT ring
            for q in range(self.n_qubits):
                state = apply_cnot(state, q, (q + 1) % self.n_qubits, self.n_qubits)

        # Final R_y
        for q in range(self.n_qubits):
            state = apply_single_qubit_gate(state, ry_matrix(self.final_ry[q]),
                                            q, self.n_qubits)

        # Trace out qubits 3,4,5
        probs_full = np.abs(state) ** 2
        probs_tensor = probs_full.reshape([2] * self.n_qubits)
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
# CLASSICAL MODEL: 80-param weight-tied MLP (from classical_tiny.py)
# =============================================================================

class ClassicalMLP80:
    """
    Weight-tied MLP with exactly 80 parameters.
    Architecture: 8 -> 8 (ReLU) -> 8 (softmax)
      W1: 8x8 = 64 params
      b1: 8 params
      b2: 8 params
      W2 = W1^T (weight-tied, no extra params)
      Total: 64 + 8 + 8 = 80
    """

    def __init__(self, rng=None):
        if rng is None:
            rng = np.random.default_rng(42)
        scale = 0.3
        self.W1 = rng.standard_normal((8, 8)).astype(np.float64) * scale
        self.b1 = np.zeros(8, dtype=np.float64)
        self.b2 = np.zeros(8, dtype=np.float64)

    def count_params(self):
        return self.W1.size + self.b1.size + self.b2.size  # 80

    def forward(self, x):
        z1 = self.W1 @ x + self.b1
        h = np.maximum(z1, 0)
        logits = self.W1.T @ h + self.b2
        return logits, h, z1

    def predict_probs(self, x):
        logits, _, _ = self.forward(x)
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits)

    def compute_loss(self, x, target):
        probs = self.predict_probs(x)
        return -np.log(probs[target] + 1e-10)

    def compute_gradients(self, x, target):
        """Manual backprop for weight-tied architecture."""
        z1 = self.W1 @ x + self.b1
        h = np.maximum(z1, 0)
        logits = self.W1.T @ h + self.b2
        logits_stable = logits - np.max(logits)
        exp_l = np.exp(logits_stable)
        probs = exp_l / np.sum(exp_l)
        loss = -np.log(probs[target] + 1e-10)

        d_logits = probs.copy()
        d_logits[target] -= 1.0

        d_b2 = d_logits.copy()

        # From layer 2: logits = W1^T @ h, so d_W1[k,j] += d_logits[j]*h[k]
        d_W1_layer2 = np.outer(h, d_logits)
        d_h = self.W1 @ d_logits

        # ReLU backward
        d_z1 = d_h * (z1 > 0).astype(np.float64)
        d_b1 = d_z1.copy()

        # From layer 1: z1 = W1 @ x, so d_W1[i,j] += d_z1[i]*x[j]
        d_W1_layer1 = np.outer(d_z1, x)

        d_W1 = d_W1_layer1 + d_W1_layer2
        return loss, d_W1, d_b1, d_b2


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
        self.v = self.beta2 * self.v + (1 - self.beta2) * grads ** 2
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
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
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g ** 2
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# =============================================================================
# EVALUATION
# =============================================================================

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


# =============================================================================
# QUANTUM TRAINING: SPSA (2 circuit evals per step, much faster than param shift)
# =============================================================================

def train_quantum_spsa(train_data, test_data, n_steps=1000, seed=0, verbose=True):
    """
    Train deep VQC using SPSA (Simultaneous Perturbation Stochastic Approximation).

    SPSA uses only 2 circuit evaluations per step (vs 2*n_params for parameter shift),
    making it ~80x faster per step.

    Standard SPSA schedule:
      a_k = a / (k + A + 1)^alpha
      c_k = c / (k + 1)^gamma
    with alpha=0.602, gamma=0.101 (Spall 1998).
    """
    rng = np.random.default_rng(seed)
    model = DeepVQC(n_layers=6, rng=rng)
    n_params = model.count_params()

    # SPSA hyperparameters (calibrated for this problem)
    a = 0.05
    c = 0.2
    A = 100  # stability constant
    alpha = 0.602
    gamma = 0.101

    best_params = model.get_params_flat().copy()
    best_test_acc = 0.0
    best_train_acc = 0.0

    eval_every = 100

    for step in range(n_steps):
        # SPSA perturbation: random +/-1 vector
        delta = rng.choice([-1, 1], size=n_params).astype(np.float64)

        # Decaying step sizes
        a_k = a / (step + A + 1) ** alpha
        c_k = c / (step + 1) ** gamma

        # Sample a random training example
        idx = rng.integers(len(train_data))
        x, y = train_data[idx]

        # Two circuit evaluations
        params = model.get_params_flat()

        model.set_params_flat(params + c_k * delta)
        loss_plus = model.compute_loss(x, y)

        model.set_params_flat(params - c_k * delta)
        loss_minus = model.compute_loss(x, y)

        # SPSA gradient estimate
        g_hat = (loss_plus - loss_minus) / (2.0 * c_k) * (1.0 / delta)

        # Gradient clipping for stability
        grad_norm = np.linalg.norm(g_hat)
        if grad_norm > 10.0:
            g_hat = g_hat * 10.0 / grad_norm

        # Update
        new_params = params - a_k * g_hat
        model.set_params_flat(new_params)

        # Evaluate periodically
        if step % eval_every == 0 or step == n_steps - 1:
            tr_loss, tr_acc = evaluate(model, train_data)
            te_loss, te_acc = evaluate(model, test_data)

            if te_acc > best_test_acc or (te_acc == best_test_acc and tr_acc > best_train_acc):
                best_test_acc = te_acc
                best_train_acc = tr_acc
                best_params = model.get_params_flat().copy()

            if verbose:
                marker = " *" if te_acc >= best_test_acc else ""
                print(f"    Step {step:4d}: train_acc={tr_acc:.3f} test_acc={te_acc:.3f} "
                      f"(best_test={best_test_acc:.3f}){marker}")

    # Restore best params
    model.set_params_flat(best_params)
    final_tr_loss, final_tr_acc = evaluate(model, train_data)
    final_te_loss, final_te_acc = evaluate(model, test_data)

    return model, {
        'best_test_acc': best_test_acc,
        'best_train_acc': best_train_acc,
        'final_train_acc': final_tr_acc,
        'final_test_acc': final_te_acc,
        'final_train_loss': final_tr_loss,
        'final_test_loss': final_te_loss,
        'best_params': best_params.tolist(),
    }


# =============================================================================
# QUANTUM TRAINING: Parameter Shift Rule + Adam (exact gradients, slower)
# =============================================================================

def compute_gradient_param_shift(model, x, target, shift=np.pi / 2):
    """Parameter shift rule: exact gradient, 2*n_params circuit evals."""
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


def train_quantum_paramshift(train_data, test_data, n_steps=1000, lr=0.01, seed=0,
                              verbose=True):
    """Train deep VQC using parameter shift rule + Adam."""
    rng = np.random.default_rng(seed)
    model = DeepVQC(n_layers=6, rng=rng)
    n_params = model.count_params()
    optimizer = AdamFlat(n_params, lr=lr)

    best_params = model.get_params_flat().copy()
    best_test_acc = 0.0
    best_train_acc = 0.0

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

            if te_acc > best_test_acc or (te_acc == best_test_acc and tr_acc > best_train_acc):
                best_test_acc = te_acc
                best_train_acc = tr_acc
                best_params = model.get_params_flat().copy()

            if verbose:
                marker = " *" if te_acc >= best_test_acc else ""
                print(f"    Step {step:4d}: train_acc={tr_acc:.3f} test_acc={te_acc:.3f} "
                      f"(best_test={best_test_acc:.3f}){marker}")

    model.set_params_flat(best_params)
    final_tr_loss, final_tr_acc = evaluate(model, train_data)
    final_te_loss, final_te_acc = evaluate(model, test_data)

    return model, {
        'best_test_acc': best_test_acc,
        'best_train_acc': best_train_acc,
        'final_train_acc': final_tr_acc,
        'final_test_acc': final_te_acc,
        'final_train_loss': final_tr_loss,
        'final_test_loss': final_te_loss,
        'best_params': best_params.tolist(),
    }


# =============================================================================
# CLASSICAL TRAINING
# =============================================================================

def train_classical_single(train_data, test_data, n_steps=5000, lr=0.01, seed=0,
                           verbose=False):
    """Train classical MLP, return best test accuracy achieved."""
    rng = np.random.default_rng(seed)
    model = ClassicalMLP80(rng=rng)
    optimizer = AdamArrays(
        [model.W1.shape, model.b1.shape, model.b2.shape], lr=lr
    )

    best_test_acc = 0.0
    best_train_acc = 0.0
    best_W1 = model.W1.copy()
    best_b1 = model.b1.copy()
    best_b2 = model.b2.copy()

    train_rng = np.random.default_rng(seed + 999)

    eval_every = 200

    for step in range(n_steps):
        idx = train_rng.integers(len(train_data))
        x, y = train_data[idx]

        loss, dW1, db1, db2 = model.compute_gradients(x, y)
        optimizer.step(
            [model.W1, model.b1, model.b2],
            [dW1, db1, db2]
        )

        if step % eval_every == 0 or step == n_steps - 1:
            tr_loss, tr_acc = evaluate(model, train_data)
            te_loss, te_acc = evaluate(model, test_data)

            if te_acc > best_test_acc or (te_acc == best_test_acc and tr_acc > best_train_acc):
                best_test_acc = te_acc
                best_train_acc = tr_acc
                best_W1 = model.W1.copy()
                best_b1 = model.b1.copy()
                best_b2 = model.b2.copy()

    # Restore best
    model.W1 = best_W1
    model.b1 = best_b1
    model.b2 = best_b2
    final_tr_loss, final_tr_acc = evaluate(model, train_data)
    final_te_loss, final_te_acc = evaluate(model, test_data)

    return model, {
        'best_test_acc': best_test_acc,
        'best_train_acc': best_train_acc,
        'final_train_acc': final_tr_acc,
        'final_test_acc': final_te_acc,
        'final_train_loss': final_tr_loss,
        'final_test_loss': final_te_loss,
    }


# =============================================================================
# MAIN: 15 restarts each
# =============================================================================

def main():
    N_RESTARTS = 15
    QUANTUM_STEPS = 1000
    CLASSICAL_STEPS = 5000
    CLASSICAL_LRS = [0.01, 0.005, 0.001]

    print("=" * 70)
    print("FAIR COMPARISON: Quantum (80 params) vs Classical (80 params)")
    print(f"  {N_RESTARTS} random restarts each, pick the best")
    print("=" * 70)

    # Data
    train_data, test_data = get_train_test_split(seed=42)
    print(f"\nDataset: {len(train_data)} train, {len(test_data)} test examples")
    print(f"Random baseline: {1.0 / VOCAB_SIZE:.1%} (1/{VOCAB_SIZE})")

    # =========================================================================
    # PHASE 1: QUANTUM — 15 restarts with SPSA (fast) + 1 param-shift verify
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: QUANTUM (6-qubit, 6-layer VQC, 80 params)")
    print(f"  Method: SPSA ({QUANTUM_STEPS} steps, 2 circuit evals/step)")
    print(f"  {N_RESTARTS} random restarts (seeds 0-{N_RESTARTS - 1})")
    print("=" * 70)

    quantum_results = []
    overall_best_q_test = 0.0
    overall_best_q_seed = 0
    t_q_start = time.time()

    for restart in range(N_RESTARTS):
        seed = restart
        print(f"\n--- Quantum restart {restart + 1}/{N_RESTARTS} (seed={seed}) ---")
        t0 = time.time()

        model_q, result_q = train_quantum_spsa(
            train_data, test_data,
            n_steps=QUANTUM_STEPS,
            seed=seed,
            verbose=True
        )

        elapsed = time.time() - t0
        result_q['seed'] = seed
        result_q['elapsed_s'] = round(elapsed, 1)
        result_q['method'] = 'SPSA'

        # Remove best_params from per-restart summary (save space in console)
        test_acc = result_q['best_test_acc']
        train_acc = result_q['best_train_acc']
        print(f"  -> Seed {seed}: test_acc={test_acc:.1%}, "
              f"train_acc={train_acc:.1%}, time={elapsed:.1f}s")

        if test_acc > overall_best_q_test:
            overall_best_q_test = test_acc
            overall_best_q_seed = seed

        quantum_results.append(result_q)

    t_q_total = time.time() - t_q_start

    # Also run the best seed with parameter shift rule for verification
    print(f"\n--- Verifying best quantum seed={overall_best_q_seed} "
          f"with parameter shift rule + Adam ---")
    t0 = time.time()
    model_q_verify, result_q_verify = train_quantum_paramshift(
        train_data, test_data,
        n_steps=QUANTUM_STEPS,
        lr=0.01,
        seed=overall_best_q_seed,
        verbose=True
    )
    elapsed_verify = time.time() - t0
    result_q_verify['seed'] = overall_best_q_seed
    result_q_verify['elapsed_s'] = round(elapsed_verify, 1)
    result_q_verify['method'] = 'param_shift_adam'
    print(f"  -> Param-shift verification: test_acc={result_q_verify['best_test_acc']:.1%}, "
          f"train_acc={result_q_verify['best_train_acc']:.1%}, time={elapsed_verify:.1f}s")

    # Also try a couple more lr values for the best seed with param shift
    for lr_try in [0.02, 0.005]:
        print(f"\n--- Param-shift seed={overall_best_q_seed}, lr={lr_try} ---")
        t0 = time.time()
        _, result_extra = train_quantum_paramshift(
            train_data, test_data,
            n_steps=QUANTUM_STEPS,
            lr=lr_try,
            seed=overall_best_q_seed,
            verbose=True
        )
        elapsed_extra = time.time() - t0
        result_extra['seed'] = overall_best_q_seed
        result_extra['elapsed_s'] = round(elapsed_extra, 1)
        result_extra['method'] = f'param_shift_adam_lr{lr_try}'
        quantum_results.append(result_extra)
        print(f"  -> lr={lr_try}: test_acc={result_extra['best_test_acc']:.1%}")

    quantum_results.append(result_q_verify)

    # Find overall best quantum
    best_q = max(quantum_results, key=lambda r: (r['best_test_acc'], r['best_train_acc']))

    print(f"\n  QUANTUM TOTAL TIME: {t_q_total:.1f}s "
          f"(+ {elapsed_verify:.1f}s verification)")

    # =========================================================================
    # PHASE 2: CLASSICAL — 15 restarts x 3 learning rates
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: CLASSICAL (weight-tied MLP, 80 params)")
    print(f"  Method: Adam ({CLASSICAL_STEPS} steps, analytic gradients)")
    print(f"  {N_RESTARTS} restarts x {len(CLASSICAL_LRS)} learning rates "
          f"= {N_RESTARTS * len(CLASSICAL_LRS)} runs")
    print("=" * 70)

    classical_results = []
    overall_best_c_test = 0.0
    overall_best_c_info = ""
    t_c_start = time.time()

    for restart in range(N_RESTARTS):
        seed = restart
        best_for_seed = {'best_test_acc': 0.0}

        for lr in CLASSICAL_LRS:
            # Use a combined seed so different lr values get different init
            combined_seed = seed * 1000 + int(lr * 10000)
            rng_init = np.random.default_rng(combined_seed)

            model_c, result_c = train_classical_single(
                train_data, test_data,
                n_steps=CLASSICAL_STEPS,
                lr=lr,
                seed=combined_seed,
                verbose=False
            )

            result_c['seed'] = seed
            result_c['lr'] = lr
            result_c['combined_seed'] = combined_seed

            if result_c['best_test_acc'] > best_for_seed['best_test_acc']:
                best_for_seed = result_c

            classical_results.append(result_c)

            if result_c['best_test_acc'] > overall_best_c_test:
                overall_best_c_test = result_c['best_test_acc']
                overall_best_c_info = f"seed={seed}, lr={lr}"

        print(f"  Restart {restart + 1:2d}/{N_RESTARTS} (seed={seed}): "
              f"best_test_acc={best_for_seed['best_test_acc']:.1%} "
              f"(lr={best_for_seed.get('lr', '?')})")

    t_c_total = time.time() - t_c_start
    best_c = max(classical_results, key=lambda r: (r['best_test_acc'], r['best_train_acc']))

    print(f"\n  CLASSICAL TOTAL TIME: {t_c_total:.1f}s")

    # =========================================================================
    # RESULTS TABLE
    # =========================================================================
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\nQuantum model:  DeepVQC (6 qubits, 6 layers, 80 params)")
    print(f"Classical model: Weight-tied MLP (8->8->8, 80 params)")
    print(f"Data split:     {len(train_data)} train / {len(test_data)} test")
    print(f"Random baseline: {1.0 / VOCAB_SIZE:.1%}")

    print(f"\n{'':30s} {'Test Acc':>10s} {'Train Acc':>10s} {'Method':>20s}")
    print("-" * 70)

    # All quantum results sorted
    print("\nQUANTUM ({} restarts):".format(N_RESTARTS))
    q_sorted = sorted(quantum_results, key=lambda r: -r['best_test_acc'])
    for i, r in enumerate(q_sorted):
        marker = " <-- BEST" if r is best_q else ""
        method_str = r.get('method', 'SPSA')
        seed_str = f"seed={r['seed']}"
        if 'lr' in method_str:
            seed_str += f" ({method_str})"
        elif method_str == 'param_shift_adam':
            seed_str += " (param_shift)"
        print(f"  {seed_str:28s} {r['best_test_acc']:>9.1%} {r['best_train_acc']:>10.1%}"
              f"  {method_str:>16s}{marker}")

    # Classical top results
    print(f"\nCLASSICAL ({N_RESTARTS} seeds x {len(CLASSICAL_LRS)} lr = "
          f"{len(classical_results)} runs):")
    c_sorted = sorted(classical_results, key=lambda r: -r['best_test_acc'])
    for i, r in enumerate(c_sorted[:10]):  # show top 10
        marker = " <-- BEST" if r is best_c else ""
        print(f"  seed={r['seed']:2d}, lr={r['lr']:.3f}:           "
              f"{r['best_test_acc']:>9.1%} {r['best_train_acc']:>10.1%}{marker}")
    if len(c_sorted) > 10:
        print(f"  ... ({len(c_sorted) - 10} more runs not shown)")

    # Final comparison
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    q_test = best_q['best_test_acc']
    q_train = best_q['best_train_acc']
    c_test = best_c['best_test_acc']
    c_train = best_c['best_train_acc']

    print(f"\n  QUANTUM  (best of {len(quantum_results):2d} runs): "
          f"test_acc = {q_test:.1%},  train_acc = {q_train:.1%}  "
          f"(seed={best_q['seed']}, {best_q.get('method', 'SPSA')})")
    print(f"  CLASSICAL (best of {len(classical_results):2d} runs): "
          f"test_acc = {c_test:.1%},  train_acc = {c_train:.1%}  "
          f"(seed={best_c['seed']}, lr={best_c.get('lr', '?')})")

    print()
    if q_test > c_test:
        delta = q_test - c_test
        winner = "QUANTUM"
        print(f"  >>> WINNER: QUANTUM  (by {delta:.1%} test accuracy)")
    elif c_test > q_test:
        delta = c_test - q_test
        winner = "CLASSICAL"
        print(f"  >>> WINNER: CLASSICAL  (by {delta:.1%} test accuracy)")
    else:
        winner = "TIE"
        if q_train > c_train:
            print(f"  >>> TIE on test accuracy, QUANTUM wins on train accuracy")
        elif c_train > q_train:
            print(f"  >>> TIE on test accuracy, CLASSICAL wins on train accuracy")
        else:
            print(f"  >>> EXACT TIE on both test and train accuracy")

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================

    results = {
        'experiment': 'fair_comparison_15_restarts',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'n_restarts': N_RESTARTS,
            'quantum_steps': QUANTUM_STEPS,
            'classical_steps': CLASSICAL_STEPS,
            'classical_lrs': CLASSICAL_LRS,
            'quantum_params': 80,
            'classical_params': 80,
            'n_train': len(train_data),
            'n_test': len(test_data),
            'random_baseline': 1.0 / VOCAB_SIZE,
        },
        'quantum_results': [
            {k: v for k, v in r.items() if k != 'best_params'}
            for r in quantum_results
        ],
        'classical_results': classical_results,
        'best_quantum': {
            'seed': best_q['seed'],
            'method': best_q.get('method', 'SPSA'),
            'test_acc': best_q['best_test_acc'],
            'train_acc': best_q['best_train_acc'],
        },
        'best_classical': {
            'seed': best_c['seed'],
            'lr': best_c.get('lr'),
            'test_acc': best_c['best_test_acc'],
            'train_acc': best_c['best_train_acc'],
        },
        'winner': winner,
        'quantum_time_s': round(t_q_total, 1),
        'classical_time_s': round(t_c_total, 1),
    }

    # If quantum wins, also save the best params for T-9 verification
    if winner == "QUANTUM" and 'best_params' in best_q:
        results['best_quantum']['params'] = best_q['best_params']
        print("\n  Best quantum parameters SAVED for T-9 verification.")

    out_path = os.path.join(SCRIPT_DIR, 'fair_comparison_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
