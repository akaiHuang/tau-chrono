#!/usr/bin/env python3
"""Interactive demo for the quantum diffusion language model."""

import sys, os, argparse, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import VOCAB, TOKEN2ID, VOCAB_SIZE, build_input_vector, bits_to_token
from model import MLPDenoiser
from convert_to_quantum import build_circuit_for_distribution

HAS_AER = False
try:
    from qiskit_aer import AerSimulator
    HAS_AER = True
except ImportError:
    pass

BLOCK, WORDS = "\u2588", sorted(TOKEN2ID.keys())
_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH, SHOTS = os.path.join(_DIR, "model_weights.npz"), 4096

def load_model():
    m = MLPDenoiser(); m.load(MODEL_PATH); return m

def parse_input(text):
    """Parse user input, return (token_ids, mask_pos) or raise ValueError."""
    tokens = text.strip().lower().split()
    if not tokens:
        raise ValueError("Empty input.")
    mask_pos = None
    ids = []
    for i, tok in enumerate(tokens):
        if tok in ("___", "_", "???", "[mask]", "blank"):
            if mask_pos is not None:
                raise ValueError("Only one blank (___) is allowed.")
            mask_pos = i
            ids.append(0)  # placeholder
        elif tok in TOKEN2ID:
            ids.append(TOKEN2ID[tok])
        else:
            raise ValueError(
                f"Unknown word '{tok}'. Vocabulary: {', '.join(WORDS)}"
            )
    if mask_pos is None:
        raise ValueError("No blank found. Use ___ to mark the position to predict.")
    return ids, mask_pos


def classical_predict(model, token_ids, mask_pos):
    inp = build_input_vector(token_ids, mask_pos)
    probs = model.forward(inp).astype(np.float64)
    probs /= probs.sum()
    return probs, int(np.argmax(probs))

def quantum_predict(probs_target):
    """Build a quantum circuit matching probs_target, run on Aer."""
    qc, _ = build_circuit_for_distribution(probs_target, n_restarts=10)
    counts = AerSimulator().run(qc, shots=SHOTS).result().get_counts()
    q_probs = np.zeros(VOCAB_SIZE)
    for bitstr, count in counts.items():
        q_probs[bits_to_token([int(b) for b in bitstr][::-1])] += count
    q_probs /= SHOTS
    return q_probs, int(np.argmax(q_probs))

def bar_chart(probs):
    """Return a sorted ASCII bar chart string."""
    lines = []
    for tok_id in np.argsort(-probs):
        p = probs[tok_id]
        bar = BLOCK * int(p * 30) or "\u258f"
        lines.append(f"    {VOCAB[tok_id]:<4s} {bar} {p:5.1%}")
    return "\n".join(lines)

def run_once(model, text, quantum_mode):
    try:
        token_ids, mask_pos = parse_input(text)
    except ValueError as e:
        print(f"  Error: {e}\n")
        return

    c_probs, c_pred = classical_predict(model, token_ids, mask_pos)
    print(f"  Classical: {VOCAB[c_pred]} ({c_probs[c_pred]:.1%} confidence)")

    q_probs = None
    if quantum_mode and HAS_AER:
        q_probs, q_pred = quantum_predict(c_probs)
        print(f"  Quantum:   {VOCAB[q_pred]} ({q_probs[q_pred]:.1%} confidence, {SHOTS} shots)")
    elif quantum_mode:
        print("  Quantum:   [skipped -- qiskit-aer not installed]")

    print()
    disp = q_probs if q_probs is not None else c_probs
    label = "Quantum" if q_probs is not None else "Classical"
    print(f"  {label} distribution:")
    print(bar_chart(disp))
    print()


def main():
    ap = argparse.ArgumentParser(description="Quantum Diffusion LM Demo")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--quantum", action="store_true", default=True,
                      help="Run both classical and quantum (default)")
    mode.add_argument("--classical", action="store_true",
                      help="Classical only, skip quantum circuit")
    ap.add_argument("--non-interactive", metavar="SENTENCE",
                    help="Run once on SENTENCE and exit")
    args = ap.parse_args()
    quantum_mode = not args.classical

    if not os.path.exists(MODEL_PATH):
        print(f"Error: model weights not found at {MODEL_PATH}")
        print("Run train_classical.py first.")
        sys.exit(1)

    model = load_model()

    print()
    print("  Quantum Diffusion Language Model Demo")
    print("  " + "=" * 40)
    print(f"  Vocabulary: {', '.join(WORDS)}")
    print(f"  Mode: {'classical + quantum' if quantum_mode else 'classical only'}")
    print()
    print("  Type a sentence using these words. Use ___ for the blank.")
    print("  Examples:")
    print("    the cat ___")
    print("    the ___ ran on the mat")
    print("    ___ sat on a mat")
    print()

    if args.non_interactive:
        print(f"> {args.non_interactive}")
        run_once(model, args.non_interactive, quantum_mode)
        return

    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if text.lower() in ("quit", "exit", "q"):
            break
        if not text:
            continue
        run_once(model, text, quantum_mode)


if __name__ == "__main__":
    main()
