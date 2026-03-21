"""
Tau-Conditioned Denoising for Diffusion Models.
=================================================

Implements the TCD (Tau-Conditioned Denoising) method:
  1. At each denoising step, compute per-position recovery difficulty tau_j
  2. Feed tau as additional conditioning to the score network
  3. The model learns cross-position correlations via the tau signal

This module extends the existing ScoreNet architecture with:
  - TauEmbedding: maps scalar tau per position to d_model-dimensional vector
  - TauConditionedScoreNet: wraps ScoreNet with self-computed tau
  - TauConditionedDecoder: modified sampling loop with amortized tau

Design: see research/tau_conditioned_denoising_design.md

Compatible with CPU and MPS (Apple Silicon). No CUDA dependency.

Author: Sheng-Kai Huang
Date: 2026-03-20
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .noise_schedule import NoiseSchedule
    from .score_network import (
        SinusoidalTimeEmbedding,
        FiLMLayer,
        ResidualFiLMBlock,
    )
except ImportError:
    from noise_schedule import NoiseSchedule
    from score_network import (
        SinusoidalTimeEmbedding,
        FiLMLayer,
        ResidualFiLMBlock,
    )


# ---------------------------------------------------------------------------
# Tau Embedding: maps per-position scalar tau -> d_model vector
# ---------------------------------------------------------------------------

class TauEmbedding(nn.Module):
    """Per-position tau embedding.

    Maps a scalar tau_j in [0, 1] (recovery difficulty) to a d_model-
    dimensional vector via a small MLP with SiLU activation.

    The embedding is designed so that:
      - tau = 0 (trivially recoverable) maps to near-zero vector
      - tau = 1 (maximally uncertain) maps to a learned maximum-signal vector
      - The intermediate range captures the "type" information (function
        word vs content word, etc.)

    Parameters
    ----------
    d_model : int
        Output embedding dimension (must match the hidden dim of the network).
    d_hidden : int
        Hidden dimension of the embedding MLP.
    """

    def __init__(self, d_model: int, d_hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_model),
        )
        # Initialize final layer small (but nonzero!) so tau embedding
        # starts with minimal impact but gradients still flow.
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        tau : Tensor, shape (batch, seq_len)
            Per-position recovery difficulty in [0, 1].

        Returns
        -------
        emb : Tensor, shape (batch, seq_len, d_model)
            Per-position tau embeddings.
        """
        # (B, L) -> (B, L, 1) -> (B, L, d_model)
        return self.net(tau.unsqueeze(-1))


# ---------------------------------------------------------------------------
# Tau-Conditioned Score Network
# ---------------------------------------------------------------------------

class TauConditionedScoreNet(nn.Module):
    """Score network with per-position tau conditioning.

    Extends the base ScoreNet architecture:
    - Standard FiLM conditioning from timestep t (global)
    - PLUS per-position tau embedding added to the input representation
    - Learnable gate controls how much tau influences predictions

    The tau signal provides INDIRECT cross-position information:
    - Low tau at position j => probably a function word => constrains
      what other positions can be
    - High tau at position k => content word => more semantic freedom

    Parameters
    ----------
    input_dim : int
        Dimension of the primary input (e.g., syndrome + noisy correction).
    output_dim : int
        Number of output logits (e.g., correction bits or vocab size).
    tau_mode : str
        How tau is used:
        - "embedding": Add tau embedding to input (recommended)
        - "film": FiLM modulation at every layer
        - "both": Embedding + FiLM
    d_hidden : int
        Hidden layer dimension.
    n_layers : int
        Number of residual blocks.
    d_time : int
        Timestep embedding dimension.
    d_tau_hidden : int
        Hidden dimension for tau embedding MLP.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        tau_mode: str = "embedding",
        d_hidden: int = 256,
        n_layers: int = 4,
        d_time: int = 64,
        d_tau_hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.tau_mode = tau_mode

        # Timestep embedding
        self.time_embed = SinusoidalTimeEmbedding(d_time)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_time, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
        )

        # Tau embedding (per-position)
        self.tau_embed = TauEmbedding(d_hidden, d_tau_hidden)

        # Learnable gate for tau contribution -- initialized at -3.0 so
        # sigmoid(-3) ≈ 0.05, meaning tau starts with ~5% influence.
        # This allows gradients to flow while keeping initial impact small.
        self.tau_gate = nn.Parameter(torch.tensor(-3.0))

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
        )

        # FiLM conditioning dimension
        if tau_mode == "film" or tau_mode == "both":
            d_cond = d_hidden * 2  # time + tau
        else:
            d_cond = d_hidden  # time only

        # Residual FiLM blocks
        self.blocks = nn.ModuleList([
            ResidualFiLMBlock(d_hidden, d_cond, dropout)
            for _ in range(n_layers)
        ])

        # If using FiLM tau: project per-position tau to conditioning dim
        if tau_mode in ("film", "both"):
            self.tau_film_proj = nn.Sequential(
                nn.Linear(1, d_hidden),
                nn.GELU(),
                nn.Linear(d_hidden, d_hidden),
            )
        else:
            self.tau_film_proj = None

        # Output head
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, output_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        tau: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with optional tau conditioning.

        Parameters
        ----------
        x : Tensor, shape (batch, input_dim)
            Primary input (e.g., syndrome + noisy correction concatenated).
        t : Tensor, shape (batch,)
            Diffusion timestep indices.
        tau : Tensor or None, shape (batch, output_dim) or (batch,)
            Per-position recovery difficulty in [0, 1].
            If None, runs without tau conditioning (graceful degradation).

        Returns
        -------
        logits : Tensor, shape (batch, output_dim)
            Output logits.
        """
        batch_size = x.shape[0]

        # Time conditioning
        t_emb = self.time_embed(t)
        cond = self.time_mlp(t_emb)  # (batch, d_hidden)

        # Input projection
        h = self.input_proj(x.float())  # (batch, d_hidden)

        # Tau injection: embedding mode
        if tau is not None and self.tau_mode in ("embedding", "both"):
            # Ensure tau has the right shape
            if tau.dim() == 1:
                tau = tau.unsqueeze(-1)  # (batch, 1)

            # Per-position tau embedding
            tau_emb = self.tau_embed(tau)  # (batch, L, d_hidden) or (batch, 1, d_hidden)

            # If tau has multiple positions, average for now (QEC case: single position)
            if tau_emb.dim() == 3:
                tau_emb = tau_emb.mean(dim=1)  # (batch, d_hidden)

            # Gated addition
            gate = torch.sigmoid(self.tau_gate)
            h = h + gate * tau_emb

        # Tau injection: FiLM mode -- concatenate tau to conditioning vector
        if tau is not None and self.tau_mode in ("film", "both") and self.tau_film_proj is not None:
            if tau.dim() == 1:
                tau_for_film = tau.unsqueeze(-1)
            else:
                tau_for_film = tau.mean(dim=-1, keepdim=True)  # average to scalar
            tau_cond = self.tau_film_proj(tau_for_film)  # (batch, d_hidden)
            cond = torch.cat([cond, tau_cond], dim=-1)  # (batch, 2*d_hidden)
        elif self.tau_mode in ("film", "both"):
            # No tau: pad with zeros
            cond = torch.cat([cond, torch.zeros_like(cond)], dim=-1)

        # Residual blocks
        for block in self.blocks:
            h = block(h, cond)

        # Output
        return self.output_head(h)


# ---------------------------------------------------------------------------
# Tau computation utilities
# ---------------------------------------------------------------------------

def compute_tau_from_logits(
    logits: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute per-position recovery difficulty from model logits.

    tau_j = 1 - max_v P(x_j = v)

    For binary outputs (QEC): tau = 1 - max(sigmoid(logit), 1 - sigmoid(logit))
    For categorical (LLM): tau = 1 - max_v softmax(logits)[v]

    Parameters
    ----------
    logits : Tensor, shape (batch, output_dim) or (batch, seq_len, vocab_size)
        Model output logits.
    mask : Tensor or None, shape matching logits' spatial dims
        If provided, only compute tau for masked positions. Others get tau=0.

    Returns
    -------
    tau : Tensor, same spatial shape as logits
        Per-position tau in [0, 1].
    """
    if logits.dim() == 2:
        # Binary case (QEC): logits are (batch, output_dim)
        probs = torch.sigmoid(logits)
        max_prob = torch.max(probs, 1.0 - probs)
        tau = 1.0 - max_prob
    elif logits.dim() == 3:
        # Categorical case (LLM): logits are (batch, seq_len, vocab_size)
        probs = F.softmax(logits, dim=-1)
        max_prob = probs.max(dim=-1).values  # (batch, seq_len)
        tau = 1.0 - max_prob
    else:
        raise ValueError(f"Expected 2D or 3D logits, got {logits.dim()}D")

    if mask is not None:
        tau = tau * mask.float()

    return tau


def compute_tau_from_probs(
    probs: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute tau from probability tensor (avoid recomputing softmax).

    Parameters
    ----------
    probs : Tensor
        Probability distribution. For binary: (batch, dim) in [0,1].
        For categorical: (batch, seq_len, vocab) summing to 1 along last dim.
    mask : optional Tensor
        Mask for which positions to compute tau.

    Returns
    -------
    tau : Tensor
    """
    if probs.dim() == 2:
        max_prob = torch.max(probs, 1.0 - probs)
        tau = 1.0 - max_prob
    elif probs.dim() == 3:
        max_prob = probs.max(dim=-1).values
        tau = 1.0 - max_prob
    else:
        raise ValueError(f"Expected 2D or 3D probs, got {probs.dim()}D")

    if mask is not None:
        tau = tau * mask.float()

    return tau


# ---------------------------------------------------------------------------
# Tau-Conditioned Decoder: inference wrapper
# ---------------------------------------------------------------------------

class TauConditionedDecoder:
    """Diffusion decoder with amortized tau conditioning.

    At each denoising step:
    1. Use tau from the PREVIOUS step's predictions (amortized, zero overhead)
    2. Run the tau-conditioned score network
    3. Save current predictions for next step's tau

    Parameters
    ----------
    model : TauConditionedScoreNet
        Trained tau-conditioned score network.
    noise_schedule : NoiseSchedule
        Must match training configuration.
    n_sample_steps : int
        Number of reverse diffusion steps.
    tau_warmup_steps : int
        Number of initial steps WITHOUT tau (to avoid noisy tau at start).
    """

    def __init__(
        self,
        model: TauConditionedScoreNet,
        noise_schedule: NoiseSchedule,
        n_sample_steps: int = 20,
        tau_warmup_steps: int = 2,
    ):
        self.model = model
        self.schedule = noise_schedule
        self.n_sample_steps = n_sample_steps
        self.tau_warmup_steps = tau_warmup_steps

        # Build skip-step indices
        T = noise_schedule.T
        if n_sample_steps >= T:
            self.timesteps = list(range(T - 1, -1, -1))
        else:
            step_size = T / n_sample_steps
            self.timesteps = [int(round(T - 1 - i * step_size)) for i in range(n_sample_steps)]
            self.timesteps = [max(0, t) for t in self.timesteps]

    @torch.no_grad()
    def sample(
        self,
        primary_input: torch.Tensor,
        external_tau: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, list]:
        """Sample via reverse diffusion with amortized tau.

        Parameters
        ----------
        primary_input : Tensor, shape (batch, input_dim_partial)
            The fixed part of the input (e.g., syndrome measurements).
        external_tau : Tensor or None
            External tau features (e.g., from DEM). If provided, used as
            INITIAL tau before self-computed tau takes over.

        Returns
        -------
        output : Tensor, shape (batch, output_dim)
            Predicted binary output vectors.
        tau_history : list of Tensor
            Per-step tau values for analysis.
        """
        self.model.eval()
        device = primary_input.device
        batch_size = primary_input.shape[0]
        output_dim = self.model.output_dim

        # Start from fully masked
        c = torch.full(
            (batch_size, output_dim), 0.5,
            device=device, dtype=torch.float32,
        )

        prev_tau = external_tau  # Use external tau if available
        tau_history = []

        for step_idx, t_val in enumerate(self.timesteps):
            t = torch.full((batch_size,), t_val, device=device, dtype=torch.long)

            # Determine tau for this step
            if step_idx < self.tau_warmup_steps:
                tau = external_tau  # Use external or None during warmup
            else:
                tau = prev_tau  # Amortized from previous step

            # Concatenate primary input with current noisy output
            x = torch.cat([primary_input.float(), c.float()], dim=-1)

            # Score network with tau
            logits = self.model(x, t, tau=tau)
            probs = torch.sigmoid(logits)

            # Compute tau for next step (amortized)
            prev_tau = compute_tau_from_logits(logits.detach())
            tau_history.append(prev_tau.clone())

            # Progressive unmasking
            if step_idx < len(self.timesteps) - 1:
                progress = (step_idx + 1) / len(self.timesteps)
                c = (1.0 - progress) * c + progress * probs
            else:
                c = probs

        output = (c > 0.5).long()
        return output, tau_history


# ---------------------------------------------------------------------------
# Training loss with self-tau
# ---------------------------------------------------------------------------

def compute_tcd_loss(
    model: TauConditionedScoreNet,
    noise_schedule: NoiseSchedule,
    primary_input: torch.Tensor,
    target: torch.Tensor,
    external_tau: Optional[torch.Tensor] = None,
    use_self_tau: bool = True,
    tau_dropout: float = 0.0,
) -> Tuple[torch.Tensor, dict]:
    """Compute tau-conditioned diffusion training loss.

    Two-pass training:
    1. Forward WITHOUT tau (stop gradient) -> compute self-tau
    2. Forward WITH tau -> compute loss

    Parameters
    ----------
    model : TauConditionedScoreNet
    noise_schedule : NoiseSchedule
    primary_input : Tensor, shape (batch, primary_dim)
        Fixed input (e.g., syndrome).
    target : Tensor, shape (batch, output_dim)
        Ground-truth output in {0, 1}.
    external_tau : Tensor or None
        External tau features.
    use_self_tau : bool
        If True, compute tau from model's own predictions.
        If False, only use external_tau (or None).
    tau_dropout : float
        Probability of dropping tau (training without it).

    Returns
    -------
    loss : scalar Tensor
        Mean BCE over masked positions.
    info : dict
        Diagnostic information (tau stats, etc.).
    """
    device = primary_input.device
    batch_size = primary_input.shape[0]

    noise_schedule.to(device)
    t_idx = noise_schedule.sample_timesteps(batch_size, device)
    c_t = noise_schedule.forward_process(target.float(), t_idx)

    # Identify masked positions
    is_masked = (c_t == NoiseSchedule.MASK_VALUE).float()

    # Concatenate input
    x = torch.cat([primary_input.float(), c_t.float()], dim=-1)

    # Phase 1: Get self-tau (no gradient)
    tau = external_tau
    if use_self_tau:
        with torch.no_grad():
            logits_notau = model(x, t_idx, tau=None)
            self_tau = compute_tau_from_logits(logits_notau.detach())
            # Merge: prefer self-tau, fall back to external
            if tau is not None:
                tau = 0.5 * self_tau + 0.5 * tau  # blend
            else:
                tau = self_tau

    # Tau dropout: randomly drop tau to prevent over-reliance
    if tau is not None and tau_dropout > 0:
        drop_mask = torch.rand(batch_size, device=device) < tau_dropout
        tau = tau * (~drop_mask).float().unsqueeze(-1) if tau.dim() > 1 else tau * (~drop_mask).float()

    # Phase 2: Forward with tau -> loss
    logits = model(x, t_idx, tau=tau)

    # BCE loss on masked positions
    bce = F.binary_cross_entropy_with_logits(
        logits, target.float(), reduction="none"
    )
    masked_bce = bce * is_masked
    n_masked = is_masked.sum().clamp(min=1.0)
    loss = masked_bce.sum() / n_masked

    # Diagnostics
    info = {
        "loss": loss.item(),
        "n_masked_mean": is_masked.sum(dim=-1).mean().item(),
        "tau_gate": torch.sigmoid(model.tau_gate).item(),
    }
    if tau is not None:
        info["tau_mean"] = tau.mean().item()
        info["tau_std"] = tau.std().item()
        info["tau_max"] = tau.max().item()
        info["tau_min"] = tau.min().item()

    return loss, info


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def build_tcd_decoder(
    primary_dim: int,
    output_dim: int,
    tau_mode: str = "embedding",
    d_hidden: int = 256,
    n_layers: int = 4,
    d_time: int = 64,
    d_tau_hidden: int = 64,
    dropout: float = 0.1,
    T: int = 100,
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
    n_sample_steps: int = 20,
    tau_warmup_steps: int = 2,
    device: Optional[torch.device] = None,
) -> Tuple[TauConditionedScoreNet, NoiseSchedule, TauConditionedDecoder]:
    """Build a complete tau-conditioned diffusion decoder.

    Returns
    -------
    model : TauConditionedScoreNet
    noise_schedule : NoiseSchedule
    decoder : TauConditionedDecoder
    """
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    input_dim = primary_dim + output_dim  # concatenated

    model = TauConditionedScoreNet(
        input_dim=input_dim,
        output_dim=output_dim,
        tau_mode=tau_mode,
        d_hidden=d_hidden,
        n_layers=n_layers,
        d_time=d_time,
        d_tau_hidden=d_tau_hidden,
        dropout=dropout,
    ).to(device)

    noise_schedule = NoiseSchedule(T=T, beta_start=beta_start, beta_end=beta_end)
    noise_schedule.to(device)

    decoder = TauConditionedDecoder(
        model, noise_schedule,
        n_sample_steps=n_sample_steps,
        tau_warmup_steps=tau_warmup_steps,
    )

    return model, noise_schedule, decoder


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Tau-Conditioned Denoising: Self-Test")
    print("=" * 70)

    # Device
    if torch.backends.mps.is_available():
        dev = torch.device("mps")
        print(f"Device: MPS (Apple Silicon)")
    else:
        dev = torch.device("cpu")
        print(f"Device: CPU")

    # Surface-17 parameters
    n_syndrome = 8
    n_correction = 9
    print(f"\nCode: surface-17 (syndrome_dim={n_syndrome}, correction_dim={n_correction})")

    # === Test 1: Embedding mode ===
    print(f"\n--- Test 1: tau_mode='embedding' ---")
    model_emb, ns, decoder_emb = build_tcd_decoder(
        primary_dim=n_syndrome,
        output_dim=n_correction,
        tau_mode="embedding",
        d_hidden=128,
        n_layers=3,
        T=50,
        n_sample_steps=10,
        device=dev,
    )
    n_params = sum(p.numel() for p in model_emb.parameters())
    print(f"Parameters: {n_params:,}")
    print(f"Tau gate (init): {torch.sigmoid(model_emb.tau_gate).item():.4f} (should be ~0.05)")

    # Synthetic data
    batch = 16
    syndrome = torch.randint(0, 2, (batch, n_syndrome), dtype=torch.float32, device=dev)
    correction = torch.randint(0, 2, (batch, n_correction), dtype=torch.float32, device=dev)

    # Training with self-tau
    model_emb.train()
    loss, info = compute_tcd_loss(
        model_emb, ns, syndrome, correction,
        use_self_tau=True, tau_dropout=0.1,
    )
    print(f"Training loss (self-tau): {loss.item():.4f}")
    print(f"Tau stats: mean={info.get('tau_mean', 'N/A'):.4f}, "
          f"std={info.get('tau_std', 'N/A'):.4f}")
    print(f"Tau gate: {info['tau_gate']:.4f}")

    # Backward
    loss.backward()
    grad_norm = sum(
        p.grad.norm().item() ** 2
        for p in model_emb.parameters()
        if p.grad is not None
    ) ** 0.5
    print(f"Gradient norm: {grad_norm:.4f}")

    # Check tau_gate has gradient
    if model_emb.tau_gate.grad is not None:
        print(f"Tau gate gradient: {model_emb.tau_gate.grad.item():.6f} (should be nonzero)")
    else:
        print("WARNING: Tau gate has no gradient!")

    # === Test 2: FiLM mode ===
    print(f"\n--- Test 2: tau_mode='film' ---")
    model_film, ns2, decoder_film = build_tcd_decoder(
        primary_dim=n_syndrome,
        output_dim=n_correction,
        tau_mode="film",
        d_hidden=128,
        n_layers=3,
        T=50,
        n_sample_steps=10,
        device=dev,
    )
    n_params_film = sum(p.numel() for p in model_film.parameters())
    print(f"Parameters: {n_params_film:,} (vs embedding: {n_params:,})")

    model_film.train()
    loss_film, info_film = compute_tcd_loss(
        model_film, ns2, syndrome, correction,
        use_self_tau=True,
    )
    print(f"Training loss (FiLM): {loss_film.item():.4f}")

    # === Test 3: Sampling with tau history ===
    print(f"\n--- Test 3: Sampling with amortized tau ---")
    model_emb.eval()
    samples, tau_history = decoder_emb.sample(syndrome)
    print(f"Sample shape: {samples.shape}")
    print(f"All binary: {((samples == 0) | (samples == 1)).all().item()}")
    print(f"Tau history length: {len(tau_history)}")
    print(f"Tau evolution (mean per step):")
    for i, th in enumerate(tau_history):
        print(f"  Step {i}: tau_mean={th.mean().item():.4f}, tau_max={th.max().item():.4f}")

    # === Test 4: No-tau fallback ===
    print(f"\n--- Test 4: Graceful degradation (no tau) ---")
    model_emb.train()
    x_cat = torch.cat([syndrome.float(), correction.float()], dim=-1)
    t = torch.randint(0, 50, (batch,), device=dev)
    logits_no_tau = model_emb(x_cat, t, tau=None)
    logits_with_tau = model_emb(x_cat, t, tau=torch.rand(batch, n_correction, device=dev))
    print(f"Output shape (no tau): {logits_no_tau.shape}")
    print(f"Output shape (with tau): {logits_with_tau.shape}")
    diff = (logits_no_tau - logits_with_tau).abs().mean().item()
    print(f"Mean absolute difference: {diff:.6f} (should be small due to zero-init gate)")

    # === Test 5: Tau computation utilities ===
    print(f"\n--- Test 5: Tau computation ---")
    # Binary logits
    binary_logits = torch.randn(4, 9, device=dev)
    tau_binary = compute_tau_from_logits(binary_logits)
    print(f"Binary tau shape: {tau_binary.shape}, range: [{tau_binary.min():.4f}, {tau_binary.max():.4f}]")

    # Categorical logits (LLM case)
    cat_logits = torch.randn(4, 16, 100, device=dev)  # (batch, seq_len, vocab)
    tau_cat = compute_tau_from_logits(cat_logits)
    print(f"Categorical tau shape: {tau_cat.shape}, range: [{tau_cat.min():.4f}, {tau_cat.max():.4f}]")

    # With mask
    mask = torch.randint(0, 2, (4, 16), device=dev).bool()
    tau_masked = compute_tau_from_logits(cat_logits, mask=mask)
    print(f"Masked tau: nonzero positions = {(tau_masked > 0).sum().item()}/{mask.sum().item()} masked")

    print("\n" + "=" * 70)
    print("All tau-conditioned denoising tests passed.")
    print("=" * 70)
