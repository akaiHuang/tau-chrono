"""
Classical MLP denoiser for the toy quantum diffusion LM.

A small 2-hidden-layer MLP that predicts masked tokens.
Implemented in pure numpy (no PyTorch needed for ~800 params).
"""

import numpy as np
from data import VOCAB_SIZE, INPUT_DIM


class MLPDenoiser:
    """
    2-hidden-layer MLP: INPUT_DIM -> 32 -> 16 -> 8 (VOCAB_SIZE)

    Activations: ReLU for hidden, softmax for output.
    """

    def __init__(self, hidden1=32, hidden2=16, seed=42):
        rng = np.random.default_rng(seed)
        self.hidden1 = hidden1
        self.hidden2 = hidden2

        # Xavier initialization
        scale1 = np.sqrt(2.0 / (INPUT_DIM + hidden1))
        self.W1 = rng.normal(0, scale1, (INPUT_DIM, hidden1)).astype(np.float32)
        self.b1 = np.zeros(hidden1, dtype=np.float32)

        scale2 = np.sqrt(2.0 / (hidden1 + hidden2))
        self.W2 = rng.normal(0, scale2, (hidden1, hidden2)).astype(np.float32)
        self.b2 = np.zeros(hidden2, dtype=np.float32)

        scale3 = np.sqrt(2.0 / (hidden2 + VOCAB_SIZE))
        self.W3 = rng.normal(0, scale3, (hidden2, VOCAB_SIZE)).astype(np.float32)
        self.b3 = np.zeros(VOCAB_SIZE, dtype=np.float32)

        self._count_params()

    def _count_params(self):
        self.num_params = (
            self.W1.size + self.b1.size +
            self.W2.size + self.b2.size +
            self.W3.size + self.b3.size
        )

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def softmax(x):
        # Numerically stable softmax
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e / np.sum(e, axis=-1, keepdims=True)

    def forward(self, x):
        """
        Forward pass. x: (batch, INPUT_DIM) or (INPUT_DIM,)

        Returns: probabilities (batch, VOCAB_SIZE) or (VOCAB_SIZE,)
        """
        single = (x.ndim == 1)
        if single:
            x = x[np.newaxis, :]

        # Layer 1
        z1 = x @ self.W1 + self.b1
        h1 = self.relu(z1)

        # Layer 2
        z2 = h1 @ self.W2 + self.b2
        h2 = self.relu(z2)

        # Output layer
        z3 = h2 @ self.W3 + self.b3
        probs = self.softmax(z3)

        if single:
            return probs[0]
        return probs

    def forward_with_cache(self, x):
        """Forward pass, returning intermediate values for backprop."""
        single = (x.ndim == 1)
        if single:
            x = x[np.newaxis, :]

        z1 = x @ self.W1 + self.b1
        h1 = self.relu(z1)
        z2 = h1 @ self.W2 + self.b2
        h2 = self.relu(z2)
        z3 = h2 @ self.W3 + self.b3
        probs = self.softmax(z3)

        cache = {'x': x, 'z1': z1, 'h1': h1, 'z2': z2, 'h2': h2, 'z3': z3, 'probs': probs}
        return probs, cache

    def backward(self, cache, targets):
        """
        Backward pass. Computes gradients of cross-entropy loss.

        targets: (batch,) integer token IDs
        Returns: dict of gradients
        """
        batch_size = cache['x'].shape[0]
        probs = cache['probs'].copy()

        # d(loss)/d(z3) = probs - one_hot(targets)
        dz3 = probs
        dz3[np.arange(batch_size), targets] -= 1.0
        dz3 /= batch_size

        # Layer 3 gradients
        dW3 = cache['h2'].T @ dz3
        db3 = np.sum(dz3, axis=0)

        # Backprop through layer 2
        dh2 = dz3 @ self.W3.T
        dz2 = dh2 * (cache['z2'] > 0).astype(np.float32)
        dW2 = cache['h1'].T @ dz2
        db2 = np.sum(dz2, axis=0)

        # Backprop through layer 1
        dh1 = dz2 @ self.W2.T
        dz1 = dh1 * (cache['z1'] > 0).astype(np.float32)
        dW1 = cache['x'].T @ dz1
        db1 = np.sum(dz1, axis=0)

        return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3}

    def predict(self, x):
        """Return the most likely token ID."""
        probs = self.forward(x)
        return int(np.argmax(probs))

    def get_params(self):
        """Return all parameters as a flat dict."""
        return {
            'W1': self.W1.copy(), 'b1': self.b1.copy(),
            'W2': self.W2.copy(), 'b2': self.b2.copy(),
            'W3': self.W3.copy(), 'b3': self.b3.copy(),
        }

    def set_params(self, params):
        """Load parameters from a flat dict."""
        self.W1 = params['W1'].copy()
        self.b1 = params['b1'].copy()
        self.W2 = params['W2'].copy()
        self.b2 = params['b2'].copy()
        self.W3 = params['W3'].copy()
        self.b3 = params['b3'].copy()

    def save(self, path):
        """Save model parameters to .npz file."""
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                 W3=self.W3, b3=self.b3)

    def load(self, path):
        """Load model parameters from .npz file."""
        data = np.load(path)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
        self.W3 = data['W3']
        self.b3 = data['b3']


class AdamOptimizer:
    """Adam optimizer for the MLP."""

    def __init__(self, model, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.model = model
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

        # Initialize moments
        self.m = {}
        self.v = {}
        for name in ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']:
            param = getattr(model, name)
            self.m[name] = np.zeros_like(param)
            self.v[name] = np.zeros_like(param)

    def step(self, grads):
        """Update parameters using Adam."""
        self.t += 1
        for name in ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']:
            g = grads[name]
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * g
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * g ** 2

            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)

            param = getattr(self.model, name)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


if __name__ == '__main__':
    model = MLPDenoiser()
    print(f"Model: {INPUT_DIM} -> {model.hidden1} -> {model.hidden2} -> {VOCAB_SIZE}")
    print(f"Total parameters: {model.num_params}")

    # Quick forward pass test
    x = np.random.randn(INPUT_DIM).astype(np.float32)
    probs = model.forward(x)
    print(f"Output probs shape: {probs.shape}, sum: {probs.sum():.4f}")
    print(f"Output probs: {probs}")
