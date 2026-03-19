#!/usr/bin/env python3
"""
Master pipeline: Diffusion QEC Decoder -- Train, Evaluate, Distill
====================================================================

Usage:
    python run_all.py --quick    # d=3, 5 epochs, ~5 min
    python run_all.py            # d=3,5, 50 epochs, ~30 min
    python run_all.py --full     # d=3,5,7, 100 epochs, ~2 hr

Steps:
    1. Generate data (Stim)
    2. Train diffusion decoder (with and without tau)
    3. Evaluate diffusion decoder
    4. Distill to GNN student
    5. Compare all decoders
    6. Generate plots and JSON results

All results are saved to results/exp9_diffusion/.

Requirements:
    pip install stim pymatching torch matplotlib numpy

Compatible with Mac M1 Max (MPS or CPU).

Author: Sheng-Kai Huang
Date: 2026-03-19
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# ─────────────────────────────────────────────────────────────────────
# Path setup
# ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(CODE_DIR)

# Ensure imports work
sys.path.insert(0, CODE_DIR)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "exp9_diffusion")


def check_dependencies():
    """Verify all required packages are installed."""
    missing = []

    try:
        import numpy  # noqa: F401
    except ImportError:
        missing.append("numpy")

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        missing.append("matplotlib")

    try:
        import stim  # noqa: F401
    except ImportError:
        missing.append("stim")

    try:
        import pymatching  # noqa: F401
    except ImportError:
        missing.append("pymatching")

    has_torch = True
    try:
        import torch  # noqa: F401
    except ImportError:
        has_torch = False
        print("WARNING: PyTorch not installed. Only MWPM baselines will run.")
        print("         Install with: pip install torch")

    if missing:
        print(f"ERROR: Missing required packages: {', '.join(missing)}")
        print(f"       Install with: pip install {' '.join(missing)}")
        sys.exit(1)

    return has_torch


def print_banner(mode: str, has_torch: bool):
    """Print the pipeline banner."""
    print()
    print("=" * 70)
    print("  Diffusion QEC Decoder: Master Pipeline")
    print("=" * 70)
    print(f"  Mode:     {mode}")
    print(f"  PyTorch:  {'yes' if has_torch else 'NO (MWPM only)'}")

    if has_torch:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "MPS (Apple Silicon)"
        elif torch.cuda.is_available():
            device = f"CUDA ({torch.cuda.get_device_name(0)})"
        else:
            device = "CPU"
        print(f"  Device:   {device}")

    print(f"  Results:  {RESULTS_DIR}")
    print("=" * 70)

    # Estimate time
    if mode == "quick":
        print(f"  Estimated time: ~5 minutes")
        print(f"  Configuration:  d=3, 5 epochs, 1K train, 1K test")
    elif mode == "normal":
        print(f"  Estimated time: ~30 minutes")
        print(f"  Configuration:  d=3,5, 50 epochs, 5K train, 5K test")
    else:
        print(f"  Estimated time: ~2 hours")
        print(f"  Configuration:  d=3,5,7, 100 epochs, 10K train, 10K test")
    print("=" * 70)


def run_pipeline(mode: str, noise_rate: float):
    """Execute the full pipeline.

    Parameters
    ----------
    mode : str
        "quick", "normal", or "full"
    noise_rate : float
        Physical error rate for the surface code.
    """
    t_start = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Import the evaluation module
    from diffusion_decoder.evaluate import main as eval_main

    # ─── Step 1-6: Delegate to evaluate.main() ────────────────────
    # evaluate.main() handles all steps internally:
    #   1. Generate data
    #   2. Train diffusion decoder (with and without tau)
    #   3. Evaluate diffusion decoder
    #   4. Distill to GNN student
    #   5. Compare all decoders
    #   6. Generate plots and JSON results

    results = eval_main(mode=mode, noise_rate=noise_rate)

    # ─── Step 7: Write final summary ──────────────────────────────
    total_time = time.time() - t_start

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Total wall time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Results saved to: {RESULTS_DIR}/")
    print()

    # List output files
    print("  Output files:")
    for f in sorted(os.listdir(RESULTS_DIR)):
        fpath = os.path.join(RESULTS_DIR, f)
        size = os.path.getsize(fpath)
        if size > 1024:
            print(f"    {f}  ({size/1024:.1f} KB)")
        else:
            print(f"    {f}  ({size} B)")

    print()

    # Quick summary of key results
    eval_results = results.get("evaluation", {})
    for d_str in sorted((k for k in eval_results if k.isdigit()), key=int):
        d_res = eval_results[d_str]
        d = d_res.get("distance", d_str)

        mwpm_ler = d_res.get("mwpm", {}).get("logical_error_rate", "N/A")
        mwpm_tau_ler = d_res.get("mwpm_tau", {}).get("logical_error_rate", "N/A")
        diff_ler = d_res.get("diffusion_tau", {}).get("logical_error_rate", "N/A")
        gnn_ler = d_res.get("gnn_distilled", {}).get("logical_error_rate", "N/A")

        print(f"  d={d}: MWPM={_fmt(mwpm_ler)}, MWPM+tau={_fmt(mwpm_tau_ler)}, "
              f"Diff+tau={_fmt(diff_ler)}, GNN(dist)={_fmt(gnn_ler)}")

    print()
    print("=" * 70)

    return results


def _fmt(val):
    """Format a value for display."""
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def main():
    parser = argparse.ArgumentParser(
        description="Diffusion QEC Decoder: Master Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all.py --quick    # d=3, 5 epochs, ~5 min
  python run_all.py            # d=3,5, 50 epochs, ~30 min
  python run_all.py --full     # d=3,5,7, 100 epochs, ~2 hr

All results saved to results/exp9_diffusion/.
        """,
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: d=3, 5 epochs, ~5 minutes",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Full mode: d=3,5,7, 100 epochs, ~2 hours",
    )
    parser.add_argument(
        "--noise", type=float, default=0.005,
        help="Physical error rate (default: 0.005)",
    )

    args = parser.parse_args()

    if args.quick:
        mode = "quick"
    elif args.full:
        mode = "full"
    else:
        mode = "normal"

    # Check dependencies
    has_torch = check_dependencies()
    print_banner(mode, has_torch)

    # Run
    run_pipeline(mode, args.noise)


if __name__ == "__main__":
    main()
