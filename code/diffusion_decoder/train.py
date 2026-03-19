#!/usr/bin/env python3
"""
Training pipeline for the diffusion QEC decoder.
==================================================

Trains a discrete diffusion model to decode surface code syndromes.
The score network learns to predict correction vectors from syndrome
measurements, conditioned on diffusion timestep and optional
retrodiction (tau) features extracted from the detector error model.

Three decoding modes are compared:
  1. MWPM (pymatching) -- classical baseline
  2. Diffusion (no tau) -- pure ML diffusion decoder
  3. Diffusion (+tau)   -- diffusion decoder with retrodiction features

Usage:
  python train.py                    # Full training (d=3,5)
  python train.py --quick            # Quick test (d=3, 5 epochs, ~5 min)
  python train.py --distance 5       # Single distance
  python train.py --epochs 100       # More training

Output:
  - Trained model checkpoint (results/diffusion_decoder_d{d}.pt)
  - Training loss curve    (results/training_curve_d{d}.png)
  - Comparison table       (results/comparison_d{d}.json)
  - Combined summary plot  (results/diffusion_decoder_summary.png)

Requirements:
  pip install torch stim pymatching matplotlib numpy

Compatible with Mac M1/M2/M3 (MPS backend) and CPU.

Author: Sheng-Kai Huang / QDA
Date: 2026-03-19
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import stim
import pymatching

# Local imports -- handle both package and script execution
try:
    from .score_network import ScoreNet, DiffusionDecoder
    from .noise_schedule import NoiseSchedule
except ImportError:
    from score_network import ScoreNet, DiffusionDecoder
    from noise_schedule import NoiseSchedule

# ---------------------------------------------------------------------------
# Matplotlib backend (must be set before import)
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Select the best available device: MPS > CPU."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Data generation with Stim
# ---------------------------------------------------------------------------

@dataclass
class QECDataConfig:
    """Configuration for QEC data generation."""
    distance: int = 3
    rounds: int = 3
    noise_p: float = 0.005
    n_train: int = 50_000
    n_test: int = 10_000
    seed: int = 42


def build_circuit(distance: int, rounds: int, noise_p: float) -> stim.Circuit:
    """Build a rotated surface code memory circuit with depolarizing noise."""
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=noise_p,
        after_reset_flip_probability=noise_p * 0.5,
        before_measure_flip_probability=noise_p * 0.5,
        before_round_data_depolarization=0,
    )
    return circuit


def extract_tau_features(circuit: stim.Circuit) -> np.ndarray:
    """Extract per-detector retrodiction features from the detector error model.

    The tau feature for each detector is derived from the DEM error probabilities.
    For each detector, we compute:
      - The total error probability affecting that detector (sum over all DEM
        error mechanisms that include this detector)
      - tau_i = 1 - exp(-sum_p_i), a monotone transform that maps [0, inf) -> [0, 1)

    This captures how "noisy" each detector is -- detectors triggered by more
    or higher-probability error mechanisms have higher tau values, meaning
    their outcomes are less reliable for decoding.

    Parameters
    ----------
    circuit : stim.Circuit
        Stim circuit with noise instructions.

    Returns
    -------
    tau : ndarray, shape (n_detectors,)
        Per-detector tau features in [0, 1).
    """
    dem = circuit.detector_error_model(decompose_errors=True)
    n_detectors = circuit.num_detectors

    # Accumulate error probabilities per detector
    detector_error_sum = np.zeros(n_detectors, dtype=np.float64)

    for instruction in dem:
        if instruction.type == "error":
            p = instruction.args_copy()[0]
            for target in instruction.targets_copy():
                if target.is_relative_detector_id():
                    det_id = target.val
                    if 0 <= det_id < n_detectors:
                        detector_error_sum[det_id] += p

    # Transform to tau: tau_i = 1 - exp(-sum_p_i)
    # This ensures tau in [0, 1) and is monotone in total error probability
    tau = 1.0 - np.exp(-detector_error_sum)

    return tau.astype(np.float32)


def generate_dataset(
    circuit: stim.Circuit,
    n_samples: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate (syndrome, observable_flip) pairs from a Stim circuit.

    Parameters
    ----------
    circuit : stim.Circuit
        The QEC circuit to sample from.
    n_samples : int
        Number of syndrome samples to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    syndromes : ndarray, shape (n_samples, n_detectors), dtype=uint8
        Detector outcomes (0 or 1).
    observables : ndarray, shape (n_samples,), dtype=uint8
        True observable flip values (0 or 1).
    """
    sampler = circuit.compile_detector_sampler(seed=seed)
    syndromes, observables = sampler.sample(n_samples, separate_observables=True)
    return syndromes.astype(np.uint8), observables.flatten().astype(np.uint8)


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class SyndromeDataset(Dataset):
    """PyTorch dataset wrapping (syndrome, observable, tau) triples."""

    def __init__(
        self,
        syndromes: np.ndarray,
        observables: np.ndarray,
        tau_features: Optional[np.ndarray] = None,
    ):
        self.syndromes = torch.tensor(syndromes, dtype=torch.float32)
        self.observables = torch.tensor(observables, dtype=torch.float32)
        self.tau_features = (
            torch.tensor(tau_features, dtype=torch.float32)
            if tau_features is not None
            else None
        )

    def __len__(self) -> int:
        return len(self.syndromes)

    def __getitem__(self, idx):
        syn = self.syndromes[idx]
        obs = self.observables[idx]
        if self.tau_features is not None:
            tau = self.tau_features  # same tau for all samples (circuit-level)
            return syn, obs, tau
        return syn, obs


# ---------------------------------------------------------------------------
# MWPM baseline decoder
# ---------------------------------------------------------------------------

def build_mwpm_decoder(circuit: stim.Circuit) -> pymatching.Matching:
    """Build MWPM decoder from the circuit's detector error model."""
    dem = circuit.detector_error_model(decompose_errors=True)
    return pymatching.Matching.from_detector_error_model(dem)


def evaluate_mwpm(
    matcher: pymatching.Matching,
    syndromes: np.ndarray,
    observables: np.ndarray,
) -> float:
    """Evaluate MWPM logical error rate.

    Returns
    -------
    ler : float
        Logical error rate.
    """
    predictions = matcher.decode_batch(syndromes)
    if predictions.ndim > 1:
        predictions = predictions.flatten()
    n_errors = int(np.sum(predictions != observables))
    return n_errors / len(observables)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Training hyperparameters."""
    epochs: int = 50
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    T_diffusion: int = 100
    n_sample_steps: int = 20
    d_hidden: int = 256
    n_layers: int = 4
    d_time: int = 64
    dropout: float = 0.1
    log_interval: int = 10
    eval_interval: int = 5  # evaluate every N epochs
    grad_clip: float = 1.0


def train_epoch(
    model: ScoreNet,
    noise_schedule: NoiseSchedule,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_tau: bool = False,
    grad_clip: float = 1.0,
) -> float:
    """Train for one epoch.

    The training objective: given a syndrome and a noisy correction c_t at
    timestep t, predict the clean correction c_0. We use binary cross-entropy
    as the loss since corrections are binary.

    Returns
    -------
    avg_loss : float
        Average training loss over the epoch.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        if use_tau and len(batch) == 3:
            syndrome, observable, tau = batch
            tau = tau.to(device)
        else:
            if len(batch) == 3:
                syndrome, observable, _ = batch
            else:
                syndrome, observable = batch
            tau = None

        syndrome = syndrome.to(device)
        observable = observable.to(device)

        batch_size = syndrome.shape[0]
        correction_dim = 1  # for surface code observable: single bit

        # Clean correction: the observable flip (0 or 1), expanded to correction_dim
        c_0 = observable.unsqueeze(-1)  # (batch, 1)

        # Sample random timesteps
        t = noise_schedule.sample_timesteps(batch_size, device)

        # Forward diffusion: corrupt the clean correction
        c_t = noise_schedule.forward_process(c_0, t)

        # Score network predicts clean correction logits
        logits = model(syndrome, c_t, t, tau)

        # Loss: binary cross-entropy between prediction and clean correction
        loss = F.binary_cross_entropy_with_logits(logits, c_0)

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_diffusion(
    model: ScoreNet,
    noise_schedule: NoiseSchedule,
    syndromes: np.ndarray,
    observables: np.ndarray,
    device: torch.device,
    n_sample_steps: int = 20,
    tau_features: Optional[np.ndarray] = None,
    use_tau: bool = False,
    eval_batch_size: int = 512,
) -> float:
    """Evaluate diffusion decoder logical error rate.

    Parameters
    ----------
    model : ScoreNet
        Trained score network.
    noise_schedule : NoiseSchedule
        Noise schedule matching training.
    syndromes : ndarray, shape (N, n_detectors)
    observables : ndarray, shape (N,)
    device : torch.device
    n_sample_steps : int
        Number of reverse diffusion steps.
    tau_features : ndarray or None, shape (n_detectors,)
    use_tau : bool
        Whether to use tau features.
    eval_batch_size : int

    Returns
    -------
    ler : float
        Logical error rate.
    """
    model.eval()
    decoder = DiffusionDecoder(model, noise_schedule, n_sample_steps)

    syn_tensor = torch.tensor(syndromes, dtype=torch.float32, device=device)
    obs_tensor = torch.tensor(observables, dtype=torch.long, device=device)

    if use_tau and tau_features is not None:
        tau_tensor = torch.tensor(
            tau_features, dtype=torch.float32, device=device
        ).unsqueeze(0).expand(len(syndromes), -1)
    else:
        tau_tensor = None

    # Decode in batches
    all_preds = []
    N = len(syndromes)

    for start in range(0, N, eval_batch_size):
        end = min(start + eval_batch_size, N)
        syn_batch = syn_tensor[start:end]
        tau_batch = tau_tensor[start:end] if tau_tensor is not None else None

        corr = decoder.sample(syn_batch, tau_batch)  # (batch, correction_dim)
        # Observable prediction: for correction_dim=1, just take the bit
        pred = corr.squeeze(-1)  # (batch,)
        all_preds.append(pred)

    predictions = torch.cat(all_preds, dim=0)
    n_errors = int((predictions != obs_tensor).sum().item())
    return n_errors / N


# ---------------------------------------------------------------------------
# Full training pipeline
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    """Results from training and evaluation."""
    distance: int
    rounds: int
    noise_p: float
    n_train: int
    n_test: int
    n_detectors: int
    training_losses: List[float] = field(default_factory=list)
    mwpm_ler: float = 0.0
    diffusion_no_tau_ler: float = 0.0
    diffusion_tau_ler: float = 0.0
    train_time_s: float = 0.0
    epochs_trained: int = 0


def run_training(
    data_config: QECDataConfig,
    train_config: TrainConfig,
    device: torch.device,
    results_dir: str,
) -> ExperimentResult:
    """Run the full training pipeline for one distance.

    Steps:
      1. Generate surface code circuit and sample data
      2. Extract tau features from detector error model
      3. Build MWPM baseline
      4. Train diffusion decoder (no tau)
      5. Train diffusion decoder (with tau)
      6. Evaluate all three decoders on test set
      7. Save checkpoint and results

    Parameters
    ----------
    data_config : QECDataConfig
        Data generation parameters.
    train_config : TrainConfig
        Training hyperparameters.
    device : torch.device
        Compute device.
    results_dir : str
        Output directory.

    Returns
    -------
    result : ExperimentResult
    """
    d = data_config.distance
    print(f"\n{'='*70}")
    print(f"  Distance d={d} Surface Code Training Pipeline")
    print(f"{'='*70}")

    # -----------------------------------------------------------------------
    # Step 1: Build circuit and generate data
    # -----------------------------------------------------------------------
    print(f"\n[1] Building d={d} rotated surface code circuit...")
    circuit = build_circuit(d, data_config.rounds, data_config.noise_p)
    n_detectors = circuit.num_detectors
    n_observables = circuit.num_observables
    print(f"    Qubits: {circuit.num_qubits}")
    print(f"    Detectors: {n_detectors}")
    print(f"    Observables: {n_observables}")

    print(f"\n[2] Generating {data_config.n_train} training + "
          f"{data_config.n_test} test samples...")
    t0 = time.time()
    syn_train, obs_train = generate_dataset(
        circuit, data_config.n_train, seed=data_config.seed
    )
    syn_test, obs_test = generate_dataset(
        circuit, data_config.n_test, seed=data_config.seed + 1000
    )
    dt_data = time.time() - t0
    print(f"    Data generation: {dt_data:.1f}s")
    print(f"    Training set observable flip rate: {obs_train.mean():.4f}")
    print(f"    Test set observable flip rate:     {obs_test.mean():.4f}")

    # -----------------------------------------------------------------------
    # Step 2: Extract tau features
    # -----------------------------------------------------------------------
    print(f"\n[3] Extracting tau features from detector error model...")
    tau_features = extract_tau_features(circuit)
    print(f"    tau shape: {tau_features.shape}")
    print(f"    tau range: [{tau_features.min():.6f}, {tau_features.max():.6f}]")
    print(f"    tau mean:  {tau_features.mean():.6f}")

    # -----------------------------------------------------------------------
    # Step 3: MWPM baseline
    # -----------------------------------------------------------------------
    print(f"\n[4] Building and evaluating MWPM baseline...")
    t0 = time.time()
    matcher = build_mwpm_decoder(circuit)
    mwpm_ler = evaluate_mwpm(matcher, syn_test, obs_test)
    dt_mwpm = time.time() - t0
    print(f"    MWPM logical error rate: {mwpm_ler:.6f}")
    print(f"    MWPM evaluation time:    {dt_mwpm:.2f}s")

    # -----------------------------------------------------------------------
    # Step 4 & 5: Train diffusion decoders
    # -----------------------------------------------------------------------
    result = ExperimentResult(
        distance=d,
        rounds=data_config.rounds,
        noise_p=data_config.noise_p,
        n_train=data_config.n_train,
        n_test=data_config.n_test,
        n_detectors=n_detectors,
        mwpm_ler=mwpm_ler,
    )

    # Correction dimension = 1 (single observable flip prediction)
    correction_dim = 1
    tau_dim = n_detectors  # tau has one value per detector

    for mode_name, use_tau in [("no_tau", False), ("with_tau", True)]:
        print(f"\n{'─'*70}")
        print(f"  Training diffusion decoder ({mode_name})")
        print(f"{'─'*70}")

        # Build model
        model = ScoreNet(
            syndrome_dim=n_detectors,
            correction_dim=correction_dim,
            tau_dim=tau_dim if use_tau else 0,
            d_hidden=train_config.d_hidden,
            n_layers=train_config.n_layers,
            d_time=train_config.d_time,
            dropout=train_config.dropout,
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"    Model parameters: {n_params:,}")

        # Build noise schedule
        noise_schedule = NoiseSchedule(T=train_config.T_diffusion)
        noise_schedule.to(device)

        # Build dataset and dataloader
        dataset = SyndromeDataset(
            syn_train, obs_train,
            tau_features=tau_features if use_tau else None,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=train_config.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,  # MPS doesn't support multiprocess workers well
            pin_memory=False,
        )

        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )

        # Learning rate scheduler: cosine annealing
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=train_config.epochs,
            eta_min=train_config.lr * 0.01,
        )

        # Training loop
        losses = []
        best_ler = float("inf")
        best_state = None
        t_train_start = time.time()

        for epoch in range(1, train_config.epochs + 1):
            t_epoch = time.time()
            avg_loss = train_epoch(
                model, noise_schedule, dataloader, optimizer, device,
                use_tau=use_tau,
                grad_clip=train_config.grad_clip,
            )
            scheduler.step()
            losses.append(avg_loss)

            dt_epoch = time.time() - t_epoch

            if epoch % train_config.log_interval == 0 or epoch == 1:
                lr_current = scheduler.get_last_lr()[0]
                print(f"    Epoch {epoch:>4d}/{train_config.epochs}: "
                      f"loss = {avg_loss:.6f}, "
                      f"lr = {lr_current:.2e}, "
                      f"time = {dt_epoch:.1f}s")

            # Periodic evaluation
            if epoch % train_config.eval_interval == 0 or epoch == train_config.epochs:
                ler = evaluate_diffusion(
                    model, noise_schedule, syn_test, obs_test, device,
                    n_sample_steps=train_config.n_sample_steps,
                    tau_features=tau_features if use_tau else None,
                    use_tau=use_tau,
                )
                print(f"           eval LER = {ler:.6f} "
                      f"(MWPM: {mwpm_ler:.6f})")

                if ler < best_ler:
                    best_ler = ler
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        train_time = time.time() - t_train_start
        print(f"\n    Training complete: {train_time:.1f}s total")
        print(f"    Best eval LER: {best_ler:.6f}")

        # Restore best model and do final evaluation
        if best_state is not None:
            model.load_state_dict(best_state)
            model = model.to(device)

        final_ler = evaluate_diffusion(
            model, noise_schedule, syn_test, obs_test, device,
            n_sample_steps=train_config.n_sample_steps,
            tau_features=tau_features if use_tau else None,
            use_tau=use_tau,
        )
        print(f"    Final LER (best checkpoint): {final_ler:.6f}")

        # Save checkpoint
        ckpt_path = os.path.join(results_dir, f"diffusion_decoder_d{d}_{mode_name}.pt")
        torch.save({
            "model_state_dict": best_state or model.state_dict(),
            "config": {
                "syndrome_dim": n_detectors,
                "correction_dim": correction_dim,
                "tau_dim": tau_dim if use_tau else 0,
                "d_hidden": train_config.d_hidden,
                "n_layers": train_config.n_layers,
                "d_time": train_config.d_time,
                "dropout": train_config.dropout,
            },
            "noise_schedule": {
                "T": train_config.T_diffusion,
            },
            "final_ler": final_ler,
            "training_losses": losses,
        }, ckpt_path)
        print(f"    Checkpoint saved: {ckpt_path}")

        # Record results
        if use_tau:
            result.diffusion_tau_ler = final_ler
        else:
            result.diffusion_no_tau_ler = final_ler
            result.training_losses = losses

        result.train_time_s += train_time
        result.epochs_trained = train_config.epochs

    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training_curve(
    losses: List[float],
    distance: int,
    output_path: str,
) -> None:
    """Plot training loss curve."""
    fig, ax = plt.subplots(figsize=(8, 5))

    epochs = range(1, len(losses) + 1)
    ax.plot(epochs, losses, "-", color="#2196F3", linewidth=2, label="Training loss (BCE)")

    # Smoothed curve (exponential moving average)
    if len(losses) > 10:
        alpha_ema = 0.1
        smoothed = [losses[0]]
        for loss in losses[1:]:
            smoothed.append(alpha_ema * loss + (1 - alpha_ema) * smoothed[-1])
        ax.plot(epochs, smoothed, "-", color="#F44336", linewidth=2,
                alpha=0.8, label="Smoothed (EMA)")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss (BCE)", fontsize=12)
    ax.set_title(f"Diffusion Decoder Training Curve (d={distance})",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, len(losses))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Training curve saved: {output_path}")


def plot_comparison(
    results: List[ExperimentResult],
    output_path: str,
) -> None:
    """Plot comparison of decoder logical error rates."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left: bar chart comparison ---
    ax = axes[0]

    distances = [r.distance for r in results]
    mwpm_lers = [r.mwpm_ler for r in results]
    no_tau_lers = [r.diffusion_no_tau_ler for r in results]
    tau_lers = [r.diffusion_tau_ler for r in results]

    x = np.arange(len(distances))
    width = 0.25

    bars1 = ax.bar(x - width, mwpm_lers, width, color="#2196F3",
                   edgecolor="black", linewidth=0.5, label="MWPM")
    bars2 = ax.bar(x, no_tau_lers, width, color="#FF9800",
                   edgecolor="black", linewidth=0.5, label="Diffusion (no tau)")
    bars3 = ax.bar(x + width, tau_lers, width, color="#4CAF50",
                   edgecolor="black", linewidth=0.5, label="Diffusion (+tau)")

    # Value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.002,
                    f"{height:.4f}", ha="center", va="bottom", fontsize=8,
                    fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"d={d}" for d in distances], fontsize=12)
    ax.set_ylabel("Logical Error Rate", fontsize=12)
    ax.set_title("Decoder Comparison: Logical Error Rate",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    # --- Right: training curve for largest distance ---
    ax2 = axes[1]
    best_result = max(results, key=lambda r: r.distance)
    if best_result.training_losses:
        epochs = range(1, len(best_result.training_losses) + 1)
        ax2.plot(epochs, best_result.training_losses, "-", color="#2196F3",
                 linewidth=1.5, alpha=0.5, label="Raw loss")

        # Smoothed
        if len(best_result.training_losses) > 5:
            alpha_ema = 0.15
            smoothed = [best_result.training_losses[0]]
            for loss in best_result.training_losses[1:]:
                smoothed.append(alpha_ema * loss + (1 - alpha_ema) * smoothed[-1])
            ax2.plot(epochs, smoothed, "-", color="#F44336", linewidth=2,
                     label="Smoothed")

    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Loss (BCE)", fontsize=12)
    ax2.set_title(f"Training Curve (d={best_result.distance})",
                  fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Summary plot saved: {output_path}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(results: List[ExperimentResult]) -> None:
    """Print a formatted comparison table."""
    print(f"\n{'='*70}")
    print("DECODER COMPARISON SUMMARY")
    print(f"{'='*70}")

    print(f"\n  {'Distance':>8s}  {'MWPM':>10s}  {'Diff(no tau)':>12s}  "
          f"{'Diff(+tau)':>12s}  {'tau gain':>10s}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*12}  {'─'*12}  {'─'*10}")

    for r in results:
        # tau gain: relative improvement of +tau over no-tau
        if r.diffusion_no_tau_ler > 0:
            gain = (r.diffusion_no_tau_ler - r.diffusion_tau_ler) / r.diffusion_no_tau_ler * 100
        else:
            gain = 0.0

        print(f"  d={r.distance:>5d}  {r.mwpm_ler:>10.6f}  "
              f"{r.diffusion_no_tau_ler:>12.6f}  {r.diffusion_tau_ler:>12.6f}  "
              f"{gain:>+9.1f}%")

    print(f"\n  Total training time: "
          f"{sum(r.train_time_s for r in results):.1f}s")

    # Interpretation
    print(f"\n  INTERPRETATION:")
    for r in results:
        if r.diffusion_tau_ler < r.diffusion_no_tau_ler:
            print(f"    d={r.distance}: tau features IMPROVE diffusion decoder "
                  f"(LER: {r.diffusion_no_tau_ler:.6f} -> {r.diffusion_tau_ler:.6f})")
        else:
            print(f"    d={r.distance}: tau features do not improve at this distance "
                  f"(may need more training or higher noise)")

        if r.diffusion_tau_ler < r.mwpm_ler:
            print(f"    d={r.distance}: Diffusion(+tau) BEATS MWPM! "
                  f"({r.diffusion_tau_ler:.6f} vs {r.mwpm_ler:.6f})")
        else:
            gap = r.diffusion_tau_ler - r.mwpm_ler
            print(f"    d={r.distance}: MWPM leads by {gap:.6f} "
                  f"(expected -- MWPM is optimal for uniform noise)")

    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train diffusion QEC decoder with retrodiction features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --quick            # Fast test (~5 min on M1)
  python train.py                    # Full training
  python train.py --distance 5       # Single distance
  python train.py --epochs 100       # More training
        """,
    )
    parser.add_argument("--quick", action="store_true",
                        help="Quick test mode: d=3, fewer samples, 5 epochs")
    parser.add_argument("--distance", type=int, nargs="+", default=None,
                        help="Surface code distance(s) to train (default: 3,5)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Training batch size (default: 256)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--n-train", type=int, default=None,
                        help="Number of training samples")
    parser.add_argument("--n-test", type=int, default=None,
                        help="Number of test samples")
    parser.add_argument("--noise", type=float, default=0.005,
                        help="Physical noise rate (default: 0.005)")
    parser.add_argument("--T", type=int, default=100,
                        help="Diffusion timesteps (default: 100)")
    parser.add_argument("--sample-steps", type=int, default=20,
                        help="Sampling steps (default: 20)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override (e.g., 'cpu', 'mps')")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Output directory for results")

    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()

    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")

    # Results directory
    if args.results_dir:
        results_dir = args.results_dir
    else:
        results_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "results",
        )
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results directory: {results_dir}")

    # Quick mode overrides
    if args.quick:
        print("\n*** QUICK MODE: reduced parameters for fast testing ***\n")
        distances = [3]
        data_configs = {3: QECDataConfig(distance=3, rounds=2, noise_p=args.noise,
                                          n_train=5_000, n_test=2_000, seed=42)}
        train_cfg = TrainConfig(
            epochs=5,
            batch_size=args.batch_size,
            lr=args.lr,
            T_diffusion=50,
            n_sample_steps=10,
            d_hidden=128,
            n_layers=3,
            d_time=32,
            log_interval=1,
            eval_interval=2,
        )
    else:
        # Full training
        distances = args.distance or [3, 5]
        n_train = args.n_train or 50_000
        n_test = args.n_test or 10_000
        epochs = args.epochs or 50

        data_configs = {}
        for dist in distances:
            rounds = max(dist, 3)
            data_configs[dist] = QECDataConfig(
                distance=dist,
                rounds=rounds,
                noise_p=args.noise,
                n_train=n_train,
                n_test=n_test,
                seed=42,
            )

        train_cfg = TrainConfig(
            epochs=epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            T_diffusion=args.T,
            n_sample_steps=args.sample_steps,
            d_hidden=256,
            n_layers=4,
            d_time=64,
            log_interval=max(1, epochs // 10),
            eval_interval=max(1, epochs // 10),
        )

    print(f"Distances: {distances}")
    print(f"Epochs: {train_cfg.epochs}")
    print(f"Batch size: {train_cfg.batch_size}")
    print(f"Diffusion steps: {train_cfg.T_diffusion}")
    print(f"Sampling steps: {train_cfg.n_sample_steps}")

    # Run training for each distance
    all_results = []
    t_total_start = time.time()

    for dist in distances:
        dc = data_configs[dist]
        result = run_training(dc, train_cfg, device, results_dir)
        all_results.append(result)

        # Save individual training curve
        if result.training_losses:
            curve_path = os.path.join(results_dir, f"training_curve_d{dist}.png")
            plot_training_curve(result.training_losses, dist, curve_path)

        # Save individual results JSON
        json_path = os.path.join(results_dir, f"comparison_d{dist}.json")
        with open(json_path, "w") as f:
            json.dump(asdict(result), f, indent=2)
        print(f"    Results JSON saved: {json_path}")

    total_time = time.time() - t_total_start

    # Summary
    print_summary(all_results)

    # Summary plot
    summary_path = os.path.join(results_dir, "diffusion_decoder_summary.png")
    plot_comparison(all_results, summary_path)

    # Save combined results
    combined_path = os.path.join(results_dir, "diffusion_decoder_all_results.json")
    with open(combined_path, "w") as f:
        json.dump({
            "results": [asdict(r) for r in all_results],
            "total_time_s": total_time,
            "device": str(device),
            "torch_version": torch.__version__,
        }, f, indent=2)
    print(f"\n  Combined results saved: {combined_path}")
    print(f"\n  Total pipeline time: {total_time:.1f}s")
    print("\nDone.")


if __name__ == "__main__":
    main()
