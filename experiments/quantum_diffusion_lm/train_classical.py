#!/usr/bin/env python3
"""
Train the classical MLP denoiser on masked token prediction.

Usage: python train_classical.py
Output: saves trained weights to model_weights.npz
"""

import sys
import time
import numpy as np

from data import (SENTENCES, VOCAB, VOCAB_SIZE, INPUT_DIM,
                  make_masked_examples, decode_sentence, build_input_vector)
from model import MLPDenoiser, AdamOptimizer


def compute_loss_and_accuracy(model, examples):
    """Compute cross-entropy loss and accuracy over examples."""
    inputs = np.array([e[0] for e in examples], dtype=np.float32)
    targets = np.array([e[1] for e in examples], dtype=np.int32)

    probs = model.forward(inputs)
    # Cross-entropy loss
    log_probs = np.log(probs[np.arange(len(targets)), targets] + 1e-10)
    loss = -np.mean(log_probs)

    # Accuracy
    preds = np.argmax(probs, axis=-1)
    accuracy = np.mean(preds == targets)

    return loss, accuracy


def train(num_steps=5000, lr=0.01, batch_size=32, seed=42, verbose=True):
    """Train the MLP denoiser."""
    rng = np.random.default_rng(seed)

    # Create model
    model = MLPDenoiser(hidden1=32, hidden2=16, seed=seed)
    optimizer = AdamOptimizer(model, lr=lr)

    # Generate training examples
    examples = make_masked_examples(SENTENCES, rng=rng)
    n_examples = len(examples)

    if verbose:
        print("=" * 60)
        print("Classical MLP Denoiser Training")
        print("=" * 60)
        print(f"Model: {INPUT_DIM} -> 32 -> 16 -> {VOCAB_SIZE}")
        print(f"Parameters: {model.num_params}")
        print(f"Training examples: {n_examples}")
        print(f"Learning rate: {lr}")
        print(f"Batch size: {batch_size}")
        print(f"Steps: {num_steps}")
        print("-" * 60)

    start_time = time.time()
    best_acc = 0.0

    for step in range(1, num_steps + 1):
        # Sample mini-batch
        idx = rng.choice(n_examples, size=min(batch_size, n_examples), replace=False)
        batch_inputs = np.array([examples[i][0] for i in idx], dtype=np.float32)
        batch_targets = np.array([examples[i][1] for i in idx], dtype=np.int32)

        # Forward + backward
        probs, cache = model.forward_with_cache(batch_inputs)
        grads = model.backward(cache, batch_targets)
        optimizer.step(grads)

        # Logging
        if verbose and (step % 500 == 0 or step == 1 or step == num_steps):
            loss, acc = compute_loss_and_accuracy(model, examples)
            elapsed = time.time() - start_time
            print(f"Step {step:5d} | Loss: {loss:.4f} | Acc: {acc:.3f} | Time: {elapsed:.1f}s")
            best_acc = max(best_acc, acc)

    elapsed = time.time() - start_time

    # Final evaluation
    final_loss, final_acc = compute_loss_and_accuracy(model, examples)

    if verbose:
        print("-" * 60)
        print(f"Final accuracy: {final_acc:.3f}")
        print(f"Best accuracy:  {best_acc:.3f}")
        print(f"Total time:     {elapsed:.2f}s")
        print()

        # Show some predictions
        print("Sample predictions:")
        print("-" * 60)
        for sent in SENTENCES[:6]:
            for mask_pos in range(len(sent)):
                inp = build_input_vector(sent, mask_pos)
                pred = model.predict(inp)
                actual = sent[mask_pos]
                status = "OK" if pred == actual else "XX"
                masked_sent = sent.copy()
                masked_sent[mask_pos] = -1
                display = ' '.join(
                    '___' if t == -1 else VOCAB[t] for t in masked_sent
                )
                print(f"  [{status}] {display} -> "
                      f"pred={VOCAB[pred]}, actual={VOCAB[actual]}")
            print()

    # Save weights
    save_path = 'model_weights.npz'
    model.save(save_path)
    if verbose:
        print(f"Weights saved to {save_path}")

    return model, final_acc


if __name__ == '__main__':
    model, acc = train()
    print(f"\nDone. Final accuracy: {acc:.1%}")
