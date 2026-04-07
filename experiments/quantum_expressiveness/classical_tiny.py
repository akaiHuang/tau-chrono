"""
Classical tiny MLP for masked token prediction.

Architecture: 8-dim input → 8 hidden (ReLU) → 8 output (softmax)
Parameters: 8*8 + 8 + 8*8 + 8 = 80 parameters exactly

Input encoding: "bag of visible tokens" — sum of one-hot vectors for all
non-masked tokens in the sentence, giving an 8-dimensional vector.
This is a deliberately simple encoding that both models can use.
"""

import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'quantum_diffusion_lm'))
from data import SENTENCES, VOCAB, VOCAB_SIZE


# ─── Simplified input encoding ───────────────────────────────────────────────

def encode_input_simple(sentence, mask_pos):
    """
    Encode sentence with one position masked into an 8-dim vector.

    Strategy: bag-of-visible-tokens (sum of one-hot vectors for visible tokens)
    plus a position signal encoded into the bag.

    Returns: 8-dimensional float vector.
    """
    vec = np.zeros(VOCAB_SIZE, dtype=np.float64)
    n = len(sentence)
    for i, tok in enumerate(sentence):
        if i == mask_pos:
            continue  # skip masked token
        vec[tok] += 1.0
    # Normalize by sentence length to keep values in [0,1] range
    if n > 1:
        vec /= (n - 1)
    # Encode mask position as a small signal in all dimensions
    # This gives the model information about WHERE in the sentence to predict
    pos_signal = (mask_pos + 1) / (n + 1)  # in (0, 1)
    vec *= (1.0 - 0.1 * pos_signal)  # subtle modulation
    # Also add position info to a specific dimension based on mask_pos
    # Use a separate encoding: add 0.5 * position_fraction to dim 0
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


# ─── Classical MLP ────────────────────────────────────────────────────────────

class TinyMLP:
    """
    2-layer MLP: 8 → 8 (ReLU) → 8 (softmax)

    Parameters:
      W1: 8x8 = 64
      b1: 8
      W2: 8x8 = 64  (wait — that's 144, too many)

    Revised to hit exactly ~80:
      W1: 8x8 = 64
      b1: 8
      W2: 8x1 diagonal-like structure won't work...

    Actually, let's use:
      W1: 8x8 = 64  (input to hidden)
      b1: 8          (hidden bias)
      W2: 8x1 = 8   (hidden to single scalar per output via shared structure)

    No — let's just do:
      W1: 8x8 = 64
      b1: 8
      b2: 8
    Total: 80 params
    Output = softmax(W1^T @ ReLU(W1 @ x + b1) + b2)
    (Reusing W1 transposed for the second layer — weight tying)
    """

    def __init__(self, rng=None):
        if rng is None:
            rng = np.random.default_rng(42)

        scale = 0.3
        self.W1 = rng.standard_normal((8, 8)).astype(np.float64) * scale  # 64 params
        self.b1 = np.zeros(8, dtype=np.float64)                           # 8 params
        self.b2 = np.zeros(8, dtype=np.float64)                           # 8 params
        # Total: 64 + 8 + 8 = 80 params

    def count_params(self):
        return self.W1.size + self.b1.size + self.b2.size

    def forward(self, x):
        """Forward pass. Returns (logits, hidden, pre_activation)."""
        z1 = self.W1 @ x + self.b1          # 8-dim
        h = np.maximum(z1, 0)               # ReLU
        logits = self.W1.T @ h + self.b2    # Weight-tied second layer
        return logits, h, z1

    def predict_probs(self, x):
        """Forward pass → softmax probabilities."""
        logits, _, _ = self.forward(x)
        # Stable softmax
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits)
        return probs

    def compute_loss(self, x, target):
        """Cross-entropy loss."""
        probs = self.predict_probs(x)
        return -np.log(probs[target] + 1e-10)

    def compute_gradients(self, x, target):
        """Manual backprop for weight-tied architecture."""
        # Forward
        z1 = self.W1 @ x + self.b1
        h = np.maximum(z1, 0)
        logits = self.W1.T @ h + self.b2

        # Softmax
        logits_stable = logits - np.max(logits)
        exp_l = np.exp(logits_stable)
        probs = exp_l / np.sum(exp_l)

        loss = -np.log(probs[target] + 1e-10)

        # Backward: dL/d(logits) = probs - one_hot(target)
        d_logits = probs.copy()
        d_logits[target] -= 1.0

        # d_b2 = d_logits
        d_b2 = d_logits.copy()

        # logits = W1^T @ h + b2
        # d_h = W1 @ d_logits
        d_h = self.W1 @ d_logits

        # d_W1 (from second layer, W1^T @ h):
        # d(W1^T @ h)/dW1 — since W1 appears transposed:
        # logits_j = sum_k W1[k,j] * h[k] + b2[j]
        # d_W1[k,j] += d_logits[j] * h[k]
        d_W1_layer2 = np.outer(h, d_logits)  # shape (8, 8), but W1 is (8,8)
        # Wait: W1^T has shape (8,8), W1^T[j,k] = W1[k,j]
        # logits = W1^T @ h means logits[j] = sum_k W1^T[j,k] * h[k] = sum_k W1[k,j]*h[k]
        # dL/dW1[k,j] from layer2 = d_logits[j] * h[k]
        # So d_W1_layer2[k,j] = h[k] * d_logits[j]
        # = outer(h, d_logits) — correct

        # ReLU backward
        d_z1 = d_h * (z1 > 0).astype(np.float64)

        # d_b1 = d_z1
        d_b1 = d_z1.copy()

        # z1 = W1 @ x + b1
        # d_W1[i,j] from layer1 = d_z1[i] * x[j]
        d_W1_layer1 = np.outer(d_z1, x)

        # Total gradient for W1 (shared)
        d_W1 = d_W1_layer1 + d_W1_layer2

        return loss, d_W1, d_b1, d_b2


class AdamOptimizer:
    """Simple Adam optimizer for numpy arrays."""

    def __init__(self, params_shapes, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = [np.zeros(s, dtype=np.float64) for s in params_shapes]
        self.v = [np.zeros(s, dtype=np.float64) for s in params_shapes]

    def step(self, params, grads):
        """Update params in-place."""
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g**2
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def train_classical(n_steps=1000, lr=0.01, seed=42, verbose=True):
    """Train the classical MLP and return training history."""
    rng = np.random.default_rng(seed)
    model = TinyMLP(rng=rng)

    if verbose:
        print(f"Classical MLP: {model.count_params()} parameters")

    # Create dataset
    all_examples = make_dataset(SENTENCES)
    n_total = len(all_examples)

    # Split: 80% train, 20% test
    indices = rng.permutation(n_total)
    n_train = int(0.8 * n_total)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    train_data = [all_examples[i] for i in train_idx]
    test_data = [all_examples[i] for i in test_idx]

    if verbose:
        print(f"Train examples: {len(train_data)}, Test examples: {len(test_data)}")

    # Optimizer
    optimizer = AdamOptimizer(
        [model.W1.shape, model.b1.shape, model.b2.shape],
        lr=lr
    )

    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': [],
        'steps': []
    }

    for step in range(n_steps):
        # Random mini-batch (single example SGD)
        idx = rng.integers(len(train_data))
        x, y = train_data[idx]

        loss, dW1, db1, db2 = model.compute_gradients(x, y)
        optimizer.step(
            [model.W1, model.b1, model.b2],
            [dW1, db1, db2]
        )

        # Evaluate every 100 steps
        if step % 100 == 0 or step == n_steps - 1:
            # Train metrics
            train_losses = []
            train_correct = 0
            for x_t, y_t in train_data:
                probs = model.predict_probs(x_t)
                train_losses.append(-np.log(probs[y_t] + 1e-10))
                if np.argmax(probs) == y_t:
                    train_correct += 1
            train_loss = np.mean(train_losses)
            train_acc = train_correct / len(train_data)

            # Test metrics
            test_losses = []
            test_correct = 0
            for x_t, y_t in test_data:
                probs = model.predict_probs(x_t)
                test_losses.append(-np.log(probs[y_t] + 1e-10))
                if np.argmax(probs) == y_t:
                    test_correct += 1
            test_loss = np.mean(test_losses)
            test_acc = test_correct / len(test_data)

            history['steps'].append(step)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)

            if verbose and (step % 200 == 0 or step == n_steps - 1):
                print(f"  Step {step:4d}: train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
                      f"test_loss={test_loss:.4f} test_acc={test_acc:.3f}")

    return model, history, train_data, test_data


if __name__ == '__main__':
    model, history, train_data, test_data = train_classical(n_steps=1000, lr=0.01)
    print(f"\nFinal: train_acc={history['train_acc'][-1]:.3f}, test_acc={history['test_acc'][-1]:.3f}")
