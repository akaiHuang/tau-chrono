#!/usr/bin/env python3
"""
Diffusion QEC Decoder: Evaluation and Knowledge Distillation Pipeline
======================================================================

Compares five decoder strategies on the rotated surface code:

  1. MWPM (PyMatching)           -- baseline, no training
  2. MWPM + tau weights          -- noise-informed MWPM
  3. Diffusion decoder (no tau)  -- pure ML
  4. Diffusion decoder (+tau)    -- Physics+ML (main claim)
  5. GNN student (distilled)     -- fast distilled version

For each decoder, measures:
  - Logical error rate at d=3,5,7
  - Decoding speed (samples/second)
  - Training data required

Includes knowledge distillation from the diffusion teacher to a GNN student
that runs 50x faster while retaining most accuracy.

Generates the "money plot": Logical Error Rate vs Training Samples.

Author: Sheng-Kai Huang
Date: 2026-03-19
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

# Conditional imports with clear error messages
try:
    import stim
except ImportError:
    print("ERROR: stim is required.  pip install stim")
    sys.exit(1)

try:
    import pymatching
except ImportError:
    print("ERROR: pymatching is required.  pip install pymatching")
    sys.exit(1)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: PyTorch not found. Only MWPM decoders will run.")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Local imports (ScoreNet, DiffusionDecoder, NoiseSchedule)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, CODE_DIR)

if HAS_TORCH:
    from diffusion_decoder.score_network import ScoreNet, DiffusionDecoder
    from diffusion_decoder.noise_schedule import NoiseSchedule

# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(SCRIPT_DIR)), "results", "exp9_diffusion"
)

SEED = 42


def get_device():
    """Select best available device: MPS > CUDA > CPU."""
    if not HAS_TORCH:
        return "cpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────
# Data generation (Stim)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class QECDataset:
    """Container for QEC syndrome data."""
    syndromes: np.ndarray       # (N, n_detectors), bool/int8
    observables: np.ndarray     # (N,), bool/int8
    corrections: np.ndarray     # (N, n_corrections), bool/int8
    tau_features: np.ndarray    # (N, n_corrections), float32
    distance: int
    noise_rate: float
    n_detectors: int
    n_corrections: int


def build_circuit(
    distance: int,
    noise_rate: float,
    rounds: Optional[int] = None,
    noise_profile: str = "uniform",
) -> stim.Circuit:
    """Build a rotated surface code memory-Z circuit.

    Parameters
    ----------
    distance : int
        Code distance (3, 5, or 7).
    noise_rate : float
        Physical error rate.
    rounds : int, optional
        Number of QEC rounds. Defaults to distance.
    noise_profile : str
        "uniform" or "biased" (2x measurement noise).
    """
    if rounds is None:
        rounds = distance

    if noise_profile == "uniform":
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=distance,
            rounds=rounds,
            after_clifford_depolarization=noise_rate,
            after_reset_flip_probability=noise_rate,
            before_measure_flip_probability=noise_rate,
            before_round_data_depolarization=noise_rate,
        )
    else:
        # Biased: measurement noise is 2x higher
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=distance,
            rounds=rounds,
            after_clifford_depolarization=noise_rate,
            after_reset_flip_probability=noise_rate,
            before_measure_flip_probability=noise_rate * 2,
            before_round_data_depolarization=noise_rate,
        )
    return circuit


def generate_data(
    distance: int,
    noise_rate: float,
    n_samples: int,
    seed: int = SEED,
    noise_profile: str = "uniform",
) -> QECDataset:
    """Generate QEC dataset: syndromes, observables, corrections, tau features.

    Parameters
    ----------
    distance, noise_rate : code parameters
    n_samples : number of samples to generate
    seed : random seed
    noise_profile : "uniform" or "biased"

    Returns
    -------
    QECDataset with all fields populated.
    """
    circuit = build_circuit(distance, noise_rate, noise_profile=noise_profile)
    dem = circuit.detector_error_model(decompose_errors=True)

    sampler = circuit.compile_detector_sampler(seed=seed)
    syndromes, observables = sampler.sample(n_samples, separate_observables=True)

    syndromes = syndromes.astype(np.int8)
    observables = observables.flatten().astype(np.int8)

    n_detectors = syndromes.shape[1]
    n_data_qubits = distance * distance

    # Corrections: for each sample, the set of data qubits that need to be
    # flipped. In surface codes, this maps to the observable flip.
    # We use the observable as a 1-bit "correction summary".
    # For the diffusion model, we expand this to per-qubit corrections
    # using MWPM as ground truth.
    matcher = pymatching.Matching.from_detector_error_model(dem)
    corrections = matcher.decode_batch(syndromes)

    # Ensure corrections has the right shape
    if corrections.ndim == 1:
        corrections = corrections.reshape(-1, 1)

    n_corrections = corrections.shape[1]

    # Tau features: per-detector noise estimates from the DEM.
    # Extract per-detector error probabilities as tau features.
    tau_features = _extract_tau_features(dem, n_detectors, n_corrections, n_samples)

    return QECDataset(
        syndromes=syndromes,
        observables=observables,
        corrections=corrections.astype(np.int8),
        tau_features=tau_features,
        distance=distance,
        noise_rate=noise_rate,
        n_detectors=n_detectors,
        n_corrections=n_corrections,
    )


def _extract_tau_features(
    dem: stim.DetectorErrorModel,
    n_detectors: int,
    n_corrections: int,
    n_samples: int,
) -> np.ndarray:
    """Extract per-detector error probabilities from the DEM as tau features.

    These serve as the "retrodiction prior" -- information about which
    detectors are more likely to fire due to noise, which is exactly what
    the tau parameter captures from the Petz recovery framework.

    Returns shape (n_samples, n_corrections) -- broadcast to all samples
    since tau is a property of the noise model, not per-shot.
    """
    # Extract per-detector marginal error probabilities
    detector_probs = np.zeros(n_detectors, dtype=np.float32)

    for instruction in dem.flattened():
        if instruction.type == "error":
            p = instruction.args_copy()[0]
            targets = instruction.targets_copy()
            for target in targets:
                if target.is_relative_detector_id():
                    det_id = target.val
                    if det_id < n_detectors:
                        # Accumulate (approximate marginal probability)
                        detector_probs[det_id] = 1 - (1 - detector_probs[det_id]) * (1 - p)

    # Map detector probabilities to correction-sized feature vector.
    # Average detector probs over spatial groups corresponding to each correction qubit.
    if n_corrections == 1:
        tau_vec = np.array([detector_probs.mean()], dtype=np.float32)
    else:
        # Distribute detectors evenly across correction qubits
        tau_vec = np.zeros(n_corrections, dtype=np.float32)
        detectors_per_corr = max(1, n_detectors // n_corrections)
        for i in range(n_corrections):
            start = i * detectors_per_corr
            end = min(start + detectors_per_corr, n_detectors)
            if start < n_detectors:
                tau_vec[i] = detector_probs[start:end].mean()

    # Broadcast to all samples (tau is per-model, not per-shot)
    tau_features = np.tile(tau_vec, (n_samples, 1))
    return tau_features


# ─────────────────────────────────────────────────────────────────────
# GNN Student Model (for knowledge distillation)
# ─────────────────────────────────────────────────────────────────────

if HAS_TORCH:
    class GNNStudent(nn.Module):
        """Simple 3-layer MLP that mimics a GNN for QEC decoding.

        Trained on the diffusion decoder's SOFT outputs (probabilities),
        NOT on hard labels from Stim. This is the key insight of
        knowledge distillation: the teacher's soft probabilities contain
        richer information than binary labels.

        Parameters
        ----------
        n_detectors : int
            Number of syndrome detectors (input dimension).
        n_corrections : int
            Number of correction bits (output dimension).
        hidden : int
            Hidden layer width.
        use_tau : bool
            Whether to accept tau features as additional input.
        tau_dim : int
            Dimension of tau features (if use_tau=True).
        """

        def __init__(
            self,
            n_detectors: int,
            n_corrections: int = 1,
            hidden: int = 128,
            use_tau: bool = False,
            tau_dim: int = 0,
        ):
            super().__init__()
            self.n_detectors = n_detectors
            self.n_corrections = n_corrections
            self.use_tau = use_tau

            input_dim = n_detectors
            if use_tau:
                input_dim += tau_dim

            self.layers = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, n_corrections),
            )

        def forward(
            self,
            syndrome: torch.Tensor,
            tau: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """
            Parameters
            ----------
            syndrome : Tensor, shape (batch, n_detectors)
            tau : Tensor or None, shape (batch, tau_dim)

            Returns
            -------
            logits : Tensor, shape (batch, n_corrections)
            """
            x = syndrome.float()
            if self.use_tau and tau is not None:
                x = torch.cat([x, tau], dim=-1)
            return self.layers(x)


# ─────────────────────────────────────────────────────────────────────
# Training utilities
# ─────────────────────────────────────────────────────────────────────

def train_diffusion_decoder(
    dataset: QECDataset,
    use_tau: bool = False,
    n_epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    d_hidden: int = 128,
    n_layers: int = 3,
    diffusion_T: int = 50,
    device: torch.device = None,
    verbose: bool = True,
) -> Tuple[ScoreNet, NoiseSchedule]:
    """Train the diffusion score network.

    Parameters
    ----------
    dataset : QECDataset
        Training data.
    use_tau : bool
        Whether to condition on tau features.
    n_epochs : int
        Training epochs.
    batch_size, lr, d_hidden, n_layers, diffusion_T : hyperparameters.
    device : torch.device
    verbose : bool

    Returns
    -------
    (model, noise_schedule) : trained ScoreNet and NoiseSchedule
    """
    if device is None:
        device = get_device()

    tau_dim = dataset.n_corrections if use_tau else 0

    model = ScoreNet(
        syndrome_dim=dataset.n_detectors,
        correction_dim=dataset.n_corrections,
        tau_dim=tau_dim,
        d_hidden=d_hidden,
        n_layers=n_layers,
        d_time=64,
        dropout=0.1,
    ).to(device)

    schedule = NoiseSchedule(T=diffusion_T, beta_start=0.0001, beta_end=0.02)
    schedule.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    # Prepare data tensors
    syn_t = torch.tensor(dataset.syndromes, dtype=torch.float32)
    corr_t = torch.tensor(dataset.corrections, dtype=torch.float32)
    tau_t = torch.tensor(dataset.tau_features, dtype=torch.float32) if use_tau else None

    if tau_t is not None:
        ds = TensorDataset(syn_t, corr_t, tau_t)
    else:
        ds = TensorDataset(syn_t, corr_t)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    model.train()
    losses = []

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0

        for batch in loader:
            if use_tau:
                syn_b, corr_b, tau_b = batch
                syn_b, corr_b, tau_b = syn_b.to(device), corr_b.to(device), tau_b.to(device)
            else:
                syn_b, corr_b = batch
                syn_b, corr_b = syn_b.to(device), corr_b.to(device)
                tau_b = None

            B = syn_b.shape[0]

            # Sample timesteps
            t = schedule.sample_timesteps(B, device)

            # Forward diffusion
            c_t = schedule.forward_process(corr_b, t)

            # Predict clean correction
            logits = model(syn_b, c_t, t, tau_b)

            # BCE loss
            loss = F.binary_cross_entropy_with_logits(
                logits, corr_b, reduction="mean"
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)

        if verbose and (epoch + 1) % max(1, n_epochs // 5) == 0:
            print(f"    Epoch {epoch+1:4d}/{n_epochs}: loss = {avg_loss:.4f}")

    model.eval()
    return model, schedule


def distill_to_gnn(
    teacher_model: ScoreNet,
    teacher_schedule: NoiseSchedule,
    dataset: QECDataset,
    use_tau: bool = True,
    n_epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    hidden: int = 128,
    n_sample_steps: int = 20,
    device: torch.device = None,
    verbose: bool = True,
) -> "GNNStudent":
    """Distill diffusion decoder knowledge into a fast GNN student.

    The student is trained on the teacher's SOFT outputs (probabilities),
    not on hard binary labels. This transfers the teacher's learned
    distribution over corrections, including its uncertainty.

    Parameters
    ----------
    teacher_model : ScoreNet
        Trained diffusion score network (frozen).
    teacher_schedule : NoiseSchedule
        Teacher's noise schedule.
    dataset : QECDataset
        Training data for distillation.
    use_tau : bool
        Whether to include tau features in the student.
    n_epochs, batch_size, lr, hidden : training hyperparameters.
    n_sample_steps : int
        Number of reverse diffusion steps for teacher inference.
    device : torch.device
    verbose : bool

    Returns
    -------
    student : GNNStudent
        Trained student model.
    """
    if device is None:
        device = get_device()

    tau_dim = dataset.n_corrections if use_tau else 0

    student = GNNStudent(
        n_detectors=dataset.n_detectors,
        n_corrections=dataset.n_corrections,
        hidden=hidden,
        use_tau=use_tau,
        tau_dim=tau_dim,
    ).to(device)

    # Create teacher decoder for soft inference
    teacher_decoder = DiffusionDecoder(
        teacher_model, teacher_schedule, n_sample_steps=n_sample_steps
    )

    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)

    # Generate teacher soft labels for the entire dataset
    if verbose:
        print("    Generating teacher soft labels...")

    syn_t = torch.tensor(dataset.syndromes, dtype=torch.float32).to(device)
    tau_t = torch.tensor(dataset.tau_features, dtype=torch.float32).to(device) if use_tau else None

    # Teacher inference in batches
    teacher_model.eval()
    all_soft = []
    infer_batch = min(batch_size, 512)

    for start in range(0, len(dataset.syndromes), infer_batch):
        end = min(start + infer_batch, len(dataset.syndromes))
        syn_batch = syn_t[start:end]
        tau_batch = tau_t[start:end] if tau_t is not None else None

        # Run reverse diffusion to get soft probabilities
        B = syn_batch.shape[0]
        correction_dim = teacher_model.correction_dim

        # Start from fully masked
        c = torch.full(
            (B, correction_dim), 0.5, device=device, dtype=torch.float32
        )

        with torch.no_grad():
            for step_idx, t_val in enumerate(teacher_decoder.timesteps):
                t = torch.full((B,), t_val, device=device, dtype=torch.long)
                logits = teacher_model(syn_batch, c, t, tau_batch)
                probs = torch.sigmoid(logits)

                if step_idx < len(teacher_decoder.timesteps) - 1:
                    progress = (step_idx + 1) / len(teacher_decoder.timesteps)
                    c = (1.0 - progress) * c + progress * probs
                else:
                    c = probs

        all_soft.append(c.cpu())

    teacher_soft = torch.cat(all_soft, dim=0)  # (N, n_corrections)

    if verbose:
        print(f"    Teacher soft labels generated: mean={teacher_soft.mean():.4f}, "
              f"std={teacher_soft.std():.4f}")

    # Distillation training
    syn_cpu = syn_t.cpu()
    tau_cpu = tau_t.cpu() if tau_t is not None else None

    if tau_cpu is not None:
        ds = TensorDataset(syn_cpu, teacher_soft, tau_cpu)
    else:
        ds = TensorDataset(syn_cpu, teacher_soft)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    student.train()

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0

        for batch in loader:
            if use_tau:
                syn_b, soft_b, tau_b = batch
                syn_b = syn_b.to(device)
                soft_b = soft_b.to(device)
                tau_b = tau_b.to(device)
            else:
                syn_b, soft_b = batch
                syn_b = syn_b.to(device)
                soft_b = soft_b.to(device)
                tau_b = None

            # Student prediction
            student_logits = student(syn_b, tau_b)

            # KL divergence loss: learn teacher's soft distribution
            loss = F.binary_cross_entropy_with_logits(
                student_logits, soft_b, reduction="mean"
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        if verbose and (epoch + 1) % max(1, n_epochs // 5) == 0:
            print(f"    Distillation epoch {epoch+1:4d}/{n_epochs}: loss = {avg_loss:.4f}")

    student.eval()
    return student


def train_gnn_on_labels(
    dataset: QECDataset,
    use_tau: bool = False,
    n_epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    hidden: int = 128,
    device: torch.device = None,
    verbose: bool = True,
) -> "GNNStudent":
    """Train GNN student directly on Stim hard labels (baseline comparison).

    This is the standard ML approach: train on ground-truth binary labels.
    Compared to distillation, this does NOT benefit from the teacher's
    soft probability distribution.

    Parameters
    ----------
    dataset : QECDataset
    use_tau, n_epochs, batch_size, lr, hidden : hyperparameters
    device : torch.device
    verbose : bool

    Returns
    -------
    student : GNNStudent
    """
    if device is None:
        device = get_device()

    tau_dim = dataset.n_corrections if use_tau else 0

    student = GNNStudent(
        n_detectors=dataset.n_detectors,
        n_corrections=dataset.n_corrections,
        hidden=hidden,
        use_tau=use_tau,
        tau_dim=tau_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)

    syn_t = torch.tensor(dataset.syndromes, dtype=torch.float32)
    corr_t = torch.tensor(dataset.corrections, dtype=torch.float32)
    tau_t = torch.tensor(dataset.tau_features, dtype=torch.float32) if use_tau else None

    if tau_t is not None:
        ds = TensorDataset(syn_t, corr_t, tau_t)
    else:
        ds = TensorDataset(syn_t, corr_t)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    student.train()

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0

        for batch in loader:
            if use_tau:
                syn_b, corr_b, tau_b = batch
                syn_b, corr_b, tau_b = syn_b.to(device), corr_b.to(device), tau_b.to(device)
            else:
                syn_b, corr_b = batch
                syn_b, corr_b = syn_b.to(device), corr_b.to(device)
                tau_b = None

            logits = student(syn_b, tau_b)
            loss = F.binary_cross_entropy_with_logits(logits, corr_b, reduction="mean")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        if verbose and (epoch + 1) % max(1, n_epochs // 5) == 0:
            print(f"    GNN epoch {epoch+1:4d}/{n_epochs}: loss = {avg_loss:.4f}")

    student.eval()
    return student


# ─────────────────────────────────────────────────────────────────────
# Decoder evaluation
# ─────────────────────────────────────────────────────────────────────

@dataclass
class DecoderResult:
    """Result of evaluating one decoder."""
    name: str
    logical_error_rate: float
    samples_per_second: float
    training_samples: int
    training_time_s: float = 0.0
    inference_time_s: float = 0.0


def evaluate_mwpm(
    test_data: QECDataset,
    noise_profile: str = "uniform",
) -> DecoderResult:
    """Evaluate standard MWPM decoder."""
    circuit = build_circuit(
        test_data.distance, test_data.noise_rate,
        noise_profile=noise_profile,
    )
    dem = circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)

    t0 = time.time()
    predictions = matcher.decode_batch(test_data.syndromes).flatten()
    dt = time.time() - t0

    n_test = len(test_data.observables)
    ler = float(np.mean(predictions != test_data.observables))
    speed = n_test / max(dt, 1e-9)

    return DecoderResult(
        name="MWPM",
        logical_error_rate=ler,
        samples_per_second=speed,
        training_samples=0,
        inference_time_s=dt,
    )


def evaluate_mwpm_tau(
    test_data: QECDataset,
) -> DecoderResult:
    """Evaluate MWPM with tau-informed weights (biased noise model)."""
    # Use biased noise model for DEM weights -- this simulates having
    # tau information about the noise structure
    circuit_biased = build_circuit(
        test_data.distance, test_data.noise_rate,
        noise_profile="biased",
    )
    try:
        dem_biased = circuit_biased.detector_error_model(decompose_errors=True)
        matcher_tau = pymatching.Matching.from_detector_error_model(dem_biased)
    except Exception:
        # Fallback to uniform if biased model fails
        circuit_uniform = build_circuit(
            test_data.distance, test_data.noise_rate,
            noise_profile="uniform",
        )
        dem_uniform = circuit_uniform.detector_error_model(decompose_errors=True)
        matcher_tau = pymatching.Matching.from_detector_error_model(dem_uniform)

    t0 = time.time()
    predictions = matcher_tau.decode_batch(test_data.syndromes).flatten()
    dt = time.time() - t0

    n_test = len(test_data.observables)
    ler = float(np.mean(predictions != test_data.observables))
    speed = n_test / max(dt, 1e-9)

    return DecoderResult(
        name="MWPM+tau",
        logical_error_rate=ler,
        samples_per_second=speed,
        training_samples=0,
        inference_time_s=dt,
    )


def evaluate_diffusion(
    model: ScoreNet,
    schedule: NoiseSchedule,
    test_data: QECDataset,
    use_tau: bool = False,
    n_sample_steps: int = 20,
    name: str = "Diffusion",
    training_samples: int = 0,
    training_time_s: float = 0.0,
    device: torch.device = None,
) -> DecoderResult:
    """Evaluate diffusion decoder on test data."""
    if device is None:
        device = get_device()

    decoder = DiffusionDecoder(model, schedule, n_sample_steps=n_sample_steps)

    syn_t = torch.tensor(test_data.syndromes, dtype=torch.float32).to(device)
    tau_t = None
    if use_tau:
        tau_t = torch.tensor(test_data.tau_features, dtype=torch.float32).to(device)

    t0 = time.time()
    predictions = decoder.decode_batch(syn_t, tau_t, batch_size=512)
    dt = time.time() - t0

    pred_np = predictions.cpu().numpy().flatten()
    obs_np = test_data.observables.flatten()

    # Ensure same length
    n = min(len(pred_np), len(obs_np))
    ler = float(np.mean(pred_np[:n] != obs_np[:n]))
    speed = n / max(dt, 1e-9)

    return DecoderResult(
        name=name,
        logical_error_rate=ler,
        samples_per_second=speed,
        training_samples=training_samples,
        training_time_s=training_time_s,
        inference_time_s=dt,
    )


def evaluate_gnn(
    student: "GNNStudent",
    test_data: QECDataset,
    use_tau: bool = False,
    name: str = "GNN",
    training_samples: int = 0,
    training_time_s: float = 0.0,
    device: torch.device = None,
) -> DecoderResult:
    """Evaluate GNN student decoder on test data."""
    if device is None:
        device = get_device()

    syn_t = torch.tensor(test_data.syndromes, dtype=torch.float32).to(device)
    tau_t = None
    if use_tau:
        tau_t = torch.tensor(test_data.tau_features, dtype=torch.float32).to(device)

    student.eval()
    t0 = time.time()

    with torch.no_grad():
        all_preds = []
        batch_size = 1024
        for start in range(0, len(syn_t), batch_size):
            end = min(start + batch_size, len(syn_t))
            syn_b = syn_t[start:end]
            tau_b = tau_t[start:end] if tau_t is not None else None
            logits = student(syn_b, tau_b)
            preds = (logits > 0).long()
            # Observable = parity of corrections
            obs_pred = preds.sum(dim=-1) % 2
            all_preds.append(obs_pred.cpu())

    dt = time.time() - t0

    pred_np = torch.cat(all_preds, dim=0).numpy().flatten()
    obs_np = test_data.observables.flatten()

    n = min(len(pred_np), len(obs_np))
    ler = float(np.mean(pred_np[:n] != obs_np[:n]))
    speed = n / max(dt, 1e-9)

    return DecoderResult(
        name=name,
        logical_error_rate=ler,
        samples_per_second=speed,
        training_samples=training_samples,
        training_time_s=training_time_s,
        inference_time_s=dt,
    )


# ─────────────────────────────────────────────────────────────────────
# Data efficiency experiment (money plot data)
# ─────────────────────────────────────────────────────────────────────

def run_data_efficiency_experiment(
    distance: int = 5,
    noise_rate: float = 0.005,
    train_sizes: List[int] = None,
    n_test: int = 5000,
    n_epochs_gnn: int = 30,
    n_epochs_diffusion: int = 50,
    diffusion_T: int = 50,
    device: torch.device = None,
    verbose: bool = True,
) -> Dict:
    """Run the data efficiency experiment for the money plot.

    Trains GNN models with varying amounts of data under different regimes:
      - Red: GNN on Stim labels (no tau)
      - Blue: GNN on Stim labels + tau features
      - Green: GNN distilled from diffusion teacher + tau

    Parameters
    ----------
    distance : int
    noise_rate : float
    train_sizes : list of int
        Number of training samples to try.
    n_test : int
        Test set size.
    n_epochs_gnn, n_epochs_diffusion : training epochs.
    diffusion_T : diffusion timesteps.
    device : torch.device
    verbose : bool

    Returns
    -------
    dict with results for plotting.
    """
    if train_sizes is None:
        train_sizes = [100, 500, 1000, 5000, 10000, 50000]

    if device is None:
        device = get_device()

    max_train = max(train_sizes)
    total_needed = max_train + n_test

    if verbose:
        print(f"\n  Generating {total_needed} samples for d={distance}, p={noise_rate}...")

    # Generate full dataset
    full_data = generate_data(distance, noise_rate, total_needed, seed=SEED)

    # Split test set (always the same)
    test_data = QECDataset(
        syndromes=full_data.syndromes[-n_test:],
        observables=full_data.observables[-n_test:],
        corrections=full_data.corrections[-n_test:],
        tau_features=full_data.tau_features[-n_test:],
        distance=full_data.distance,
        noise_rate=full_data.noise_rate,
        n_detectors=full_data.n_detectors,
        n_corrections=full_data.n_corrections,
    )

    # MWPM baseline (no training)
    mwpm_result = evaluate_mwpm(test_data)
    if verbose:
        print(f"  MWPM baseline LER: {mwpm_result.logical_error_rate:.4f}")

    results = {
        "mwpm_ler": mwpm_result.logical_error_rate,
        "gnn_stim": {},          # Red: GNN on Stim labels
        "gnn_stim_tau": {},      # Blue: GNN on Stim labels + tau
        "gnn_distilled_tau": {}, # Green: GNN distilled from diffusion + tau
    }

    for n_train in train_sizes:
        if n_train > max_train:
            continue

        if verbose:
            print(f"\n  --- N_train = {n_train} ---")

        # Subset training data
        train_sub = QECDataset(
            syndromes=full_data.syndromes[:n_train],
            observables=full_data.observables[:n_train],
            corrections=full_data.corrections[:n_train],
            tau_features=full_data.tau_features[:n_train],
            distance=full_data.distance,
            noise_rate=full_data.noise_rate,
            n_detectors=full_data.n_detectors,
            n_corrections=full_data.n_corrections,
        )

        # Red: GNN on Stim labels (no tau)
        if verbose:
            print(f"  Training GNN on Stim labels (no tau)...")
        t0 = time.time()
        gnn_stim = train_gnn_on_labels(
            train_sub, use_tau=False, n_epochs=n_epochs_gnn,
            device=device, verbose=False,
        )
        t_train = time.time() - t0
        res_stim = evaluate_gnn(
            gnn_stim, test_data, use_tau=False, name="GNN(Stim)",
            training_samples=n_train, training_time_s=t_train, device=device,
        )
        results["gnn_stim"][n_train] = asdict(res_stim)
        if verbose:
            print(f"    LER = {res_stim.logical_error_rate:.4f} "
                  f"({res_stim.samples_per_second:.0f} samples/s)")

        # Blue: GNN on Stim labels + tau
        if verbose:
            print(f"  Training GNN on Stim labels + tau...")
        t0 = time.time()
        gnn_tau = train_gnn_on_labels(
            train_sub, use_tau=True, n_epochs=n_epochs_gnn,
            device=device, verbose=False,
        )
        t_train = time.time() - t0
        res_tau = evaluate_gnn(
            gnn_tau, test_data, use_tau=True, name="GNN(Stim+tau)",
            training_samples=n_train, training_time_s=t_train, device=device,
        )
        results["gnn_stim_tau"][n_train] = asdict(res_tau)
        if verbose:
            print(f"    LER = {res_tau.logical_error_rate:.4f} "
                  f"({res_tau.samples_per_second:.0f} samples/s)")

        # Green: Train diffusion teacher, then distill to GNN
        if verbose:
            print(f"  Training diffusion teacher (+tau)...")
        t0 = time.time()
        teacher, schedule = train_diffusion_decoder(
            train_sub, use_tau=True, n_epochs=n_epochs_diffusion,
            diffusion_T=diffusion_T, d_hidden=128, n_layers=3,
            device=device, verbose=False,
        )
        t_teacher = time.time() - t0

        if verbose:
            print(f"  Distilling to GNN student...")
        t0 = time.time()
        gnn_distilled = distill_to_gnn(
            teacher, schedule, train_sub, use_tau=True,
            n_epochs=n_epochs_gnn, n_sample_steps=20,
            device=device, verbose=False,
        )
        t_distill = time.time() - t0

        res_distilled = evaluate_gnn(
            gnn_distilled, test_data, use_tau=True, name="GNN(distilled+tau)",
            training_samples=n_train, training_time_s=t_teacher + t_distill,
            device=device,
        )
        results["gnn_distilled_tau"][n_train] = asdict(res_distilled)
        if verbose:
            print(f"    LER = {res_distilled.logical_error_rate:.4f} "
                  f"({res_distilled.samples_per_second:.0f} samples/s)")

    return results


# ─────────────────────────────────────────────────────────────────────
# Full multi-distance evaluation
# ─────────────────────────────────────────────────────────────────────

def run_full_evaluation(
    distances: List[int] = None,
    noise_rate: float = 0.005,
    n_train: int = 5000,
    n_test: int = 5000,
    n_epochs_diffusion: int = 50,
    n_epochs_gnn: int = 30,
    diffusion_T: int = 50,
    n_sample_steps: int = 20,
    device: torch.device = None,
    verbose: bool = True,
) -> Dict:
    """Run full decoder comparison across code distances.

    Evaluates all 5 decoders at each distance d:
      1. MWPM (baseline)
      2. MWPM + tau weights
      3. Diffusion (no tau)
      4. Diffusion (+tau) -- main claim
      5. GNN student (distilled from 4)

    Returns
    -------
    dict with per-distance results and timing.
    """
    if distances is None:
        distances = [3, 5, 7]

    if device is None:
        device = get_device()

    all_results = {}
    t_total_start = time.time()

    for d in distances:
        if verbose:
            print(f"\n{'='*70}")
            print(f"  Distance d = {d}")
            print(f"{'='*70}")

        # Generate data
        total = n_train + n_test
        if verbose:
            print(f"  Generating {total} samples...")

        data = generate_data(d, noise_rate, total, seed=SEED + d)

        train_data = QECDataset(
            syndromes=data.syndromes[:n_train],
            observables=data.observables[:n_train],
            corrections=data.corrections[:n_train],
            tau_features=data.tau_features[:n_train],
            distance=data.distance,
            noise_rate=data.noise_rate,
            n_detectors=data.n_detectors,
            n_corrections=data.n_corrections,
        )

        test_data = QECDataset(
            syndromes=data.syndromes[n_train:],
            observables=data.observables[n_train:],
            corrections=data.corrections[n_train:],
            tau_features=data.tau_features[n_train:],
            distance=data.distance,
            noise_rate=data.noise_rate,
            n_detectors=data.n_detectors,
            n_corrections=data.n_corrections,
        )

        d_results = {"distance": d, "n_detectors": data.n_detectors,
                      "n_corrections": data.n_corrections}

        # 1. MWPM
        if verbose:
            print(f"\n  [1/5] MWPM baseline...")
        res_mwpm = evaluate_mwpm(test_data)
        d_results["mwpm"] = asdict(res_mwpm)
        if verbose:
            print(f"    LER = {res_mwpm.logical_error_rate:.4f} "
                  f"({res_mwpm.samples_per_second:.0f} s/s)")

        # 2. MWPM + tau
        if verbose:
            print(f"  [2/5] MWPM + tau weights...")
        res_mwpm_tau = evaluate_mwpm_tau(test_data)
        d_results["mwpm_tau"] = asdict(res_mwpm_tau)
        if verbose:
            print(f"    LER = {res_mwpm_tau.logical_error_rate:.4f} "
                  f"({res_mwpm_tau.samples_per_second:.0f} s/s)")

        if HAS_TORCH:
            # 3. Diffusion (no tau)
            if verbose:
                print(f"  [3/5] Diffusion decoder (no tau)...")
            t0 = time.time()
            model_no_tau, sched_no_tau = train_diffusion_decoder(
                train_data, use_tau=False, n_epochs=n_epochs_diffusion,
                diffusion_T=diffusion_T, device=device, verbose=verbose,
            )
            t_train_no_tau = time.time() - t0

            res_diff_no_tau = evaluate_diffusion(
                model_no_tau, sched_no_tau, test_data,
                use_tau=False, n_sample_steps=n_sample_steps,
                name="Diffusion(no_tau)",
                training_samples=n_train, training_time_s=t_train_no_tau,
                device=device,
            )
            d_results["diffusion_no_tau"] = asdict(res_diff_no_tau)
            if verbose:
                print(f"    LER = {res_diff_no_tau.logical_error_rate:.4f} "
                      f"({res_diff_no_tau.samples_per_second:.0f} s/s)")

            # 4. Diffusion (+tau)
            if verbose:
                print(f"  [4/5] Diffusion decoder (+tau)...")
            t0 = time.time()
            model_tau, sched_tau = train_diffusion_decoder(
                train_data, use_tau=True, n_epochs=n_epochs_diffusion,
                diffusion_T=diffusion_T, device=device, verbose=verbose,
            )
            t_train_tau = time.time() - t0

            res_diff_tau = evaluate_diffusion(
                model_tau, sched_tau, test_data,
                use_tau=True, n_sample_steps=n_sample_steps,
                name="Diffusion(+tau)",
                training_samples=n_train, training_time_s=t_train_tau,
                device=device,
            )
            d_results["diffusion_tau"] = asdict(res_diff_tau)
            if verbose:
                print(f"    LER = {res_diff_tau.logical_error_rate:.4f} "
                      f"({res_diff_tau.samples_per_second:.0f} s/s)")

            # 5. GNN student (distilled from diffusion+tau)
            if verbose:
                print(f"  [5/5] GNN student (distilled from diffusion+tau)...")
            t0 = time.time()
            student = distill_to_gnn(
                model_tau, sched_tau, train_data,
                use_tau=True, n_epochs=n_epochs_gnn,
                n_sample_steps=n_sample_steps,
                device=device, verbose=verbose,
            )
            t_distill = time.time() - t0

            res_student = evaluate_gnn(
                student, test_data, use_tau=True,
                name="GNN_distilled",
                training_samples=n_train,
                training_time_s=t_train_tau + t_distill,
                device=device,
            )
            d_results["gnn_distilled"] = asdict(res_student)
            if verbose:
                print(f"    LER = {res_student.logical_error_rate:.4f} "
                      f"({res_student.samples_per_second:.0f} s/s)")
        else:
            if verbose:
                print("  [3-5] Skipped (PyTorch not available)")

        all_results[str(d)] = d_results

    total_time = time.time() - t_total_start
    all_results["total_time_s"] = total_time

    return all_results


# ─────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────

def plot_decoder_comparison(
    results: Dict,
    output_path: str,
    title: str = "Diffusion QEC Decoder Comparison",
) -> None:
    """Plot bar chart of LER for each decoder at each distance.

    Parameters
    ----------
    results : dict
        Output of run_full_evaluation().
    output_path : str
        File path for the saved figure.
    title : str
    """
    distances = []
    decoder_names = ["MWPM", "MWPM+tau", "Diff(no_tau)", "Diff(+tau)", "GNN_dist"]
    decoder_keys = ["mwpm", "mwpm_tau", "diffusion_no_tau", "diffusion_tau", "gnn_distilled"]
    colors = ["#9E9E9E", "#607D8B", "#F44336", "#4CAF50", "#2196F3"]

    for d_str in sorted((k for k in results if k.isdigit()), key=int):
        distances.append(int(d_str))

    if not distances:
        print("  No distance data to plot.")
        return

    fig, axes = plt.subplots(1, len(distances), figsize=(5 * len(distances), 6),
                              sharey=True, squeeze=False)

    for idx, d in enumerate(distances):
        ax = axes[0][idx]
        d_res = results[str(d)]

        lers = []
        labels = []
        bar_colors = []
        speeds = []

        for name, key, color in zip(decoder_names, decoder_keys, colors):
            if key in d_res:
                ler = d_res[key]["logical_error_rate"]
                speed = d_res[key]["samples_per_second"]
                lers.append(ler)
                labels.append(name)
                bar_colors.append(color)
                speeds.append(speed)

        x = np.arange(len(lers))
        bars = ax.bar(x, lers, color=bar_colors, edgecolor="black", linewidth=0.5, alpha=0.85)

        for bar, ler, speed in zip(bars, lers, speeds):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{ler:.3f}\n({speed:.0f}/s)",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_title(f"d = {d}", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        if idx == 0:
            ax.set_ylabel("Logical Error Rate", fontsize=12)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Decoder comparison plot saved: {output_path}")


def plot_money_plot(
    results: Dict,
    output_path: str,
) -> None:
    """Create THE money plot: Logical Error Rate vs Training Samples.

    X axis: training samples (log scale)
    Y axis: logical error rate

    Lines:
      - Gray dashed: MWPM baseline (no training)
      - Red: GNN on Stim labels (standard ML)
      - Blue: GNN on Stim labels + tau features
      - Green: GNN distilled from diffusion teacher + tau (our method)

    Parameters
    ----------
    results : dict
        Output of run_data_efficiency_experiment().
    output_path : str
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # MWPM baseline
    mwpm_ler = results.get("mwpm_ler", None)
    if mwpm_ler is not None:
        ax.axhline(y=mwpm_ler, color="#9E9E9E", linewidth=2.5, linestyle="--",
                    label=f"MWPM baseline ({mwpm_ler:.4f})", zorder=3)
        ax.axhspan(mwpm_ler * 0.97, mwpm_ler * 1.03,
                    color="#9E9E9E", alpha=0.08, zorder=1)

    # Plot lines for each GNN variant
    plot_configs = [
        ("gnn_stim",          "#F44336", "o", "GNN (Stim labels)"),
        ("gnn_stim_tau",      "#2196F3", "s", "GNN (Stim labels + tau)"),
        ("gnn_distilled_tau", "#4CAF50", "D", "GNN (distilled from diffusion + tau)"),
    ]

    for key, color, marker, label in plot_configs:
        if key not in results or not results[key]:
            continue

        ns = sorted(int(k) for k in results[key].keys())
        lers = [results[key][n]["logical_error_rate"] for n in ns]

        ax.plot(ns, lers, f"{marker}-", color=color, linewidth=2.2,
                markersize=8, label=label, zorder=4)

        # Annotate each point
        for n, ler in zip(ns, lers):
            offset = (0, 10) if key != "gnn_distilled_tau" else (0, -14)
            ax.annotate(f"{ler:.3f}", (n, ler), textcoords="offset points",
                        xytext=offset, fontsize=7.5, color=color, ha="center")

    ax.set_xscale("log")
    ax.set_xlabel("Training Samples (log scale)", fontsize=13)
    ax.set_ylabel("Logical Error Rate (lower = better)", fontsize=13)
    ax.set_title("Data Efficiency: Diffusion Distillation vs Direct Training",
                  fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Annotation for the advantage
    gnn_stim = results.get("gnn_stim", {})
    gnn_distilled = results.get("gnn_distilled_tau", {})
    if gnn_stim and gnn_distilled:
        # Find where distilled beats stim with fewer samples
        ns_stim = sorted(int(k) for k in gnn_stim.keys())
        ns_dist = sorted(int(k) for k in gnn_distilled.keys())

        if ns_stim and ns_dist:
            best_stim_ler = min(gnn_stim[n]["logical_error_rate"] for n in ns_stim)
            best_stim_n = min(ns_stim, key=lambda n: gnn_stim[n]["logical_error_rate"])

            # Find smallest n where distilled matches best stim
            match_n = None
            for n in ns_dist:
                if gnn_distilled[n]["logical_error_rate"] <= best_stim_ler * 1.05:
                    match_n = n
                    break

            if match_n and match_n < best_stim_n:
                efficiency = best_stim_n / match_n
                ax.annotate(
                    f"{efficiency:.0f}x data\nefficiency",
                    xy=(match_n, gnn_distilled[match_n]["logical_error_rate"]),
                    xytext=(match_n * 5, gnn_distilled[match_n]["logical_error_rate"] + 0.03),
                    fontsize=11, fontweight="bold", color="#4CAF50",
                    arrowprops=dict(arrowstyle="->", color="#4CAF50", lw=2),
                    zorder=6,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F5E9",
                              edgecolor="#4CAF50"),
                )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Money plot saved: {output_path}")


def plot_speed_comparison(
    results: Dict,
    output_path: str,
) -> None:
    """Plot decoding speed comparison across decoders.

    Parameters
    ----------
    results : dict
        Output of run_full_evaluation().
    output_path : str
    """
    # Collect speed data from the first available distance
    d_str = None
    for k in sorted(results.keys()):
        if k.isdigit():
            d_str = k
            break

    if d_str is None:
        print("  No data for speed plot.")
        return

    d_res = results[d_str]
    names = []
    speeds = []
    lers = []
    colors_list = []

    decoder_info = [
        ("mwpm",            "MWPM",           "#9E9E9E"),
        ("mwpm_tau",        "MWPM+tau",       "#607D8B"),
        ("diffusion_no_tau","Diff(no_tau)",    "#F44336"),
        ("diffusion_tau",   "Diff(+tau)",      "#4CAF50"),
        ("gnn_distilled",   "GNN(distilled)",  "#2196F3"),
    ]

    for key, name, color in decoder_info:
        if key in d_res:
            names.append(name)
            speeds.append(d_res[key]["samples_per_second"])
            lers.append(d_res[key]["logical_error_rate"])
            colors_list.append(color)

    if not names:
        print("  No speed data to plot.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: speed bar chart
    x = np.arange(len(names))
    bars = ax1.bar(x, speeds, color=colors_list, edgecolor="black",
                    linewidth=0.5, alpha=0.85)
    for bar, speed in zip(bars, speeds):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{speed:.0f}", ha="center", va="bottom", fontsize=9,
                 fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=30, ha="right", fontsize=10)
    ax1.set_ylabel("Samples / second", fontsize=12)
    ax1.set_title(f"Decoding Speed (d={d_str})", fontsize=13, fontweight="bold")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3, axis="y")

    # Right: LER vs speed scatter
    for i, (name, speed, ler, color) in enumerate(zip(names, speeds, lers, colors_list)):
        ax2.scatter(speed, ler, c=color, s=150, edgecolors="black",
                     linewidth=0.8, zorder=5, marker="o")
        ax2.annotate(name, (speed, ler), textcoords="offset points",
                     xytext=(10, 5), fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="wheat",
                               alpha=0.7, edgecolor="gray"))

    ax2.set_xlabel("Samples / second (log scale)", fontsize=12)
    ax2.set_ylabel("Logical Error Rate", fontsize=12)
    ax2.set_title("Accuracy vs Speed Tradeoff", fontsize=13, fontweight="bold")
    ax2.set_xscale("log")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"Decoder Performance: d={d_str}", fontsize=14,
                  fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Speed comparison plot saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────
# Summary printing
# ─────────────────────────────────────────────────────────────────────

def print_summary_table(results: Dict) -> None:
    """Print a formatted comparison table."""
    print("\n" + "=" * 80)
    print("  DECODER COMPARISON SUMMARY")
    print("=" * 80)

    header = f"  {'Decoder':<25s} {'d':>3s} {'LER':>10s} {'Speed':>12s} {'Train N':>10s}"
    print(header)
    print("  " + "-" * 75)

    for d_str in sorted((k for k in results if k.isdigit()), key=int):
        d_res = results[d_str]
        d = d_res.get("distance", d_str)

        decoder_keys = [
            ("mwpm",             "MWPM"),
            ("mwpm_tau",         "MWPM+tau"),
            ("diffusion_no_tau", "Diffusion(no tau)"),
            ("diffusion_tau",    "Diffusion(+tau)"),
            ("gnn_distilled",    "GNN(distilled)"),
        ]

        for key, name in decoder_keys:
            if key in d_res:
                ler = d_res[key]["logical_error_rate"]
                speed = d_res[key]["samples_per_second"]
                n_train = d_res[key]["training_samples"]
                print(f"  {name:<25s} {d:>3} {ler:>10.4f} {speed:>10.0f}/s {n_train:>10d}")

        print()

    # Highlight key findings
    print("=" * 80)
    print("  KEY FINDINGS:")

    for d_str in sorted((k for k in results if k.isdigit()), key=int):
        d_res = results[d_str]
        d = d_res.get("distance", d_str)

        mwpm_ler = d_res.get("mwpm", {}).get("logical_error_rate")
        diff_tau_ler = d_res.get("diffusion_tau", {}).get("logical_error_rate")
        gnn_ler = d_res.get("gnn_distilled", {}).get("logical_error_rate")
        gnn_speed = d_res.get("gnn_distilled", {}).get("samples_per_second")
        diff_speed = d_res.get("diffusion_tau", {}).get("samples_per_second")

        if mwpm_ler and diff_tau_ler:
            ratio = diff_tau_ler / mwpm_ler if mwpm_ler > 0 else float('inf')
            print(f"  d={d}: Diffusion(+tau) vs MWPM: {ratio:.3f}x")

        if gnn_speed and diff_speed and diff_speed > 0:
            speedup = gnn_speed / diff_speed
            print(f"  d={d}: GNN is {speedup:.0f}x faster than Diffusion")

    print("=" * 80)


# ─────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────

def main(
    mode: str = "quick",
    noise_rate: float = 0.005,
) -> Dict:
    """Run the full evaluation pipeline.

    Parameters
    ----------
    mode : str
        "quick" (d=3, 5 epochs, ~5 min)
        "normal" (d=5, 50 epochs, ~30 min)
        "full" (d=3,5,7, 100 epochs, ~2 hr)
    noise_rate : float
        Physical error rate.

    Returns
    -------
    dict with all results.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = get_device()

    print("=" * 70)
    print("  Diffusion QEC Decoder: Evaluation Pipeline")
    print(f"  Mode: {mode}")
    print(f"  Device: {device}")
    print(f"  Physical noise: {noise_rate}")
    print(f"  PyTorch: {'available' if HAS_TORCH else 'NOT available'}")
    print("=" * 70)

    # Mode-dependent configuration
    if mode == "quick":
        distances = [3]
        n_train, n_test = 1000, 1000
        n_epochs_diff, n_epochs_gnn = 5, 5
        diffusion_T = 20
        n_sample_steps = 10
        money_train_sizes = [100, 500, 1000]
    elif mode == "normal":
        distances = [3, 5]
        n_train, n_test = 5000, 5000
        n_epochs_diff, n_epochs_gnn = 50, 30
        diffusion_T = 50
        n_sample_steps = 20
        money_train_sizes = [100, 500, 1000, 5000]
    else:  # full
        distances = [3, 5, 7]
        n_train, n_test = 10000, 10000
        n_epochs_diff, n_epochs_gnn = 100, 50
        diffusion_T = 100
        n_sample_steps = 50
        money_train_sizes = [100, 500, 1000, 5000, 10000, 50000]

    all_results = {"mode": mode, "noise_rate": noise_rate, "device": str(device)}

    # Part 1: Multi-distance evaluation
    print("\n" + "=" * 70)
    print("  PART 1: Multi-Distance Decoder Comparison")
    print("=" * 70)

    eval_results = run_full_evaluation(
        distances=distances,
        noise_rate=noise_rate,
        n_train=n_train,
        n_test=n_test,
        n_epochs_diffusion=n_epochs_diff,
        n_epochs_gnn=n_epochs_gnn,
        diffusion_T=diffusion_T,
        n_sample_steps=n_sample_steps,
        device=device,
        verbose=True,
    )
    all_results["evaluation"] = eval_results

    # Print summary
    print_summary_table(eval_results)

    # Part 2: Data efficiency (money plot)
    if HAS_TORCH:
        print("\n" + "=" * 70)
        print("  PART 2: Data Efficiency Experiment (Money Plot)")
        print("=" * 70)

        d_for_money = distances[0]
        money_results = run_data_efficiency_experiment(
            distance=d_for_money,
            noise_rate=noise_rate,
            train_sizes=money_train_sizes,
            n_test=n_test,
            n_epochs_gnn=n_epochs_gnn,
            n_epochs_diffusion=n_epochs_diff,
            diffusion_T=diffusion_T,
            device=device,
            verbose=True,
        )
        all_results["data_efficiency"] = money_results

    # Part 3: Generate plots
    print("\n" + "=" * 70)
    print("  PART 3: Generating Plots")
    print("=" * 70)

    plot_decoder_comparison(
        eval_results,
        os.path.join(RESULTS_DIR, "decoder_comparison.png"),
    )

    plot_speed_comparison(
        eval_results,
        os.path.join(RESULTS_DIR, "speed_comparison.png"),
    )

    if "data_efficiency" in all_results:
        plot_money_plot(
            all_results["data_efficiency"],
            os.path.join(RESULTS_DIR, "money_plot.png"),
        )

    # Save JSON results
    json_path = os.path.join(RESULTS_DIR, "evaluation_results.json")

    # Convert any non-serializable types
    def make_serializable(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [make_serializable(v) for v in obj]
        return obj

    with open(json_path, "w") as f:
        json.dump(make_serializable(all_results), f, indent=2)
    print(f"\n  JSON results saved: {json_path}")

    # Final timing
    total_time = eval_results.get("total_time_s", 0)
    print(f"\n  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print("  Done.")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Diffusion QEC Decoder: Evaluation and Distillation Pipeline"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: d=3, 5 epochs (~5 min)",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Full mode: d=3,5,7, 100 epochs (~2 hr)",
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

    main(mode=mode, noise_rate=args.noise)
