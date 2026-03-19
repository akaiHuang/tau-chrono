"""
Score network and diffusion decoder for QEC.
=============================================

ScoreNet: A neural network that predicts denoising logits for each bit of
a correction vector, conditioned on:
  - Syndrome measurements (detector outcomes)
  - Current noisy correction c_t
  - Diffusion timestep t
  - Optional retrodiction (tau) features from the detector error model

Architecture:
  - FiLM conditioning: timestep and tau features modulate hidden layers
    via affine transformations (scale + shift), following Perez et al. 2018.
  - Residual MLP blocks with LayerNorm for stable training.
  - Output: logits for each correction bit (probability of being 1).

DiffusionDecoder: End-to-end wrapper that combines the score network with
forward/reverse diffusion processes for inference.

Compatible with CPU and MPS (Apple Silicon). No CUDA dependency.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .noise_schedule import NoiseSchedule
except ImportError:
    from noise_schedule import NoiseSchedule


# ---------------------------------------------------------------------------
# Sinusoidal timestep embedding (standard from DDPM)
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timestep t.

    Maps integer timestep to a d_model-dimensional vector using sin/cos
    frequencies at different scales, following Vaswani et al. 2017 / Ho et al. 2020.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        t : Tensor, shape (batch,)
            Integer timestep indices.

        Returns
        -------
        emb : Tensor, shape (batch, d_model)
        """
        device = t.device
        half_dim = self.d_model // 2
        emb_scale = math.log(10000.0) / (half_dim - 1)
        freqs = torch.exp(-emb_scale * torch.arange(half_dim, device=device, dtype=torch.float32))
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)  # (batch, half_dim)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (batch, d_model)
        if self.d_model % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


# ---------------------------------------------------------------------------
# FiLM conditioning layer
# ---------------------------------------------------------------------------

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation.

    Applies an affine transformation conditioned on an external signal z:
        h' = gamma(z) * h + beta(z)

    This allows the timestep / tau information to modulate each hidden layer.
    """

    def __init__(self, d_hidden: int, d_cond: int):
        super().__init__()
        self.linear = nn.Linear(d_cond, 2 * d_hidden)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        h : Tensor, shape (batch, d_hidden)
            Hidden activations to modulate.
        z : Tensor, shape (batch, d_cond)
            Conditioning signal (timestep + tau features).

        Returns
        -------
        h' : Tensor, shape (batch, d_hidden)
        """
        gamma_beta = self.linear(z)  # (batch, 2 * d_hidden)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return (1.0 + gamma) * h + beta


# ---------------------------------------------------------------------------
# Residual MLP block with FiLM conditioning
# ---------------------------------------------------------------------------

class ResidualFiLMBlock(nn.Module):
    """Residual MLP block with FiLM conditioning and LayerNorm."""

    def __init__(self, d_hidden: int, d_cond: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_hidden)
        self.linear1 = nn.Linear(d_hidden, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_hidden)
        self.film = FiLMLayer(d_hidden, d_cond)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (batch, d_hidden)
        cond : Tensor, shape (batch, d_cond)

        Returns
        -------
        out : Tensor, shape (batch, d_hidden)
        """
        h = self.norm(x)
        h = F.gelu(self.linear1(h))
        h = self.film(h, cond)
        h = self.dropout(h)
        h = self.linear2(h)
        return x + h


# ---------------------------------------------------------------------------
# ScoreNet: the main score network
# ---------------------------------------------------------------------------

class ScoreNet(nn.Module):
    """Score network for discrete diffusion QEC decoding.

    Predicts denoising logits for each correction bit, conditioned on
    syndrome measurements, noisy correction, timestep, and optional tau features.

    Parameters
    ----------
    syndrome_dim : int
        Number of detector outcomes (syndrome bits).
    correction_dim : int
        Number of correction bits (data qubits for the observable).
    tau_dim : int
        Dimension of retrodiction (tau) feature vector. 0 to disable.
    d_hidden : int
        Hidden layer dimension.
    n_layers : int
        Number of residual FiLM blocks.
    d_time : int
        Dimension of timestep embedding.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        syndrome_dim: int,
        correction_dim: int,
        tau_dim: int = 0,
        d_hidden: int = 256,
        n_layers: int = 4,
        d_time: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.syndrome_dim = syndrome_dim
        self.correction_dim = correction_dim
        self.tau_dim = tau_dim

        # Input: syndrome + noisy correction concatenated
        input_dim = syndrome_dim + correction_dim

        # Timestep embedding
        self.time_embed = SinusoidalTimeEmbedding(d_time)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_time, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
        )

        # Tau feature projection (if enabled)
        if tau_dim > 0:
            self.tau_proj = nn.Sequential(
                nn.Linear(tau_dim, d_hidden),
                nn.GELU(),
                nn.Linear(d_hidden, d_hidden),
            )
        else:
            self.tau_proj = None

        # Conditioning dimension: time + optional tau
        d_cond = d_hidden  # time and tau get summed into a single d_hidden vector

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
        )

        # Residual FiLM blocks
        self.blocks = nn.ModuleList([
            ResidualFiLMBlock(d_hidden, d_cond, dropout)
            for _ in range(n_layers)
        ])

        # Output head: logits for each correction bit
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, correction_dim),
        )

    def forward(
        self,
        syndrome: torch.Tensor,
        c_t: torch.Tensor,
        t: torch.Tensor,
        tau: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass: predict denoising logits.

        Parameters
        ----------
        syndrome : Tensor, shape (batch, syndrome_dim)
            Detector outcomes (0/1).
        c_t : Tensor, shape (batch, correction_dim)
            Noisy correction vector at timestep t.
        t : Tensor, shape (batch,)
            Diffusion timestep indices.
        tau : Tensor or None, shape (batch, tau_dim)
            Optional retrodiction features.

        Returns
        -------
        logits : Tensor, shape (batch, correction_dim)
            Logits for each correction bit being 1.
        """
        # Build conditioning vector
        t_emb = self.time_embed(t)          # (batch, d_time)
        cond = self.time_mlp(t_emb)         # (batch, d_hidden)

        if tau is not None and self.tau_proj is not None:
            tau_emb = self.tau_proj(tau)     # (batch, d_hidden)
            cond = cond + tau_emb           # additive combination

        # Input: concatenate syndrome and noisy correction
        x = torch.cat([syndrome.float(), c_t.float()], dim=-1)  # (batch, input_dim)
        h = self.input_proj(x)              # (batch, d_hidden)

        # Residual blocks with FiLM conditioning
        for block in self.blocks:
            h = block(h, cond)

        # Output logits
        logits = self.output_head(h)        # (batch, correction_dim)
        return logits


# ---------------------------------------------------------------------------
# DiffusionDecoder: end-to-end inference wrapper
# ---------------------------------------------------------------------------

class DiffusionDecoder:
    """End-to-end diffusion decoder for QEC.

    Wraps a trained ScoreNet with the reverse diffusion sampling loop.

    Parameters
    ----------
    model : ScoreNet
        Trained score network.
    noise_schedule : NoiseSchedule
        Noise schedule (must match training configuration).
    n_sample_steps : int
        Number of reverse diffusion steps for sampling (can be < T via step skipping).
    """

    def __init__(self, model: ScoreNet, noise_schedule, n_sample_steps: int = 20):
        self.model = model
        self.schedule = noise_schedule
        self.n_sample_steps = n_sample_steps

        # Build skip-step indices: evenly spaced from T-1 down to 0
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
        syndrome: torch.Tensor,
        tau: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample corrections via reverse diffusion.

        Parameters
        ----------
        syndrome : Tensor, shape (batch, syndrome_dim)
            Detector outcomes.
        tau : Tensor or None, shape (batch, tau_dim)
            Optional retrodiction features.

        Returns
        -------
        correction : Tensor, shape (batch, correction_dim)
            Predicted binary correction vectors (0 or 1).
        """
        self.model.eval()
        device = syndrome.device
        batch_size = syndrome.shape[0]
        correction_dim = self.model.correction_dim

        # Start from fully masked (all 0.5)
        c = torch.full(
            (batch_size, correction_dim), 0.5,
            device=device, dtype=torch.float32,
        )

        for step_idx, t_val in enumerate(self.timesteps):
            t = torch.full((batch_size,), t_val, device=device, dtype=torch.long)

            # Score network predicts clean correction logits
            logits = self.model(syndrome, c, t, tau)
            probs = torch.sigmoid(logits)

            # Determine how many bits to "unmask" at this step
            # Progressive unmasking: more bits get decided as t decreases
            if step_idx < len(self.timesteps) - 1:
                # Interpolate between current noisy state and prediction
                # As we progress, trust the prediction more
                progress = (step_idx + 1) / len(self.timesteps)
                c = (1.0 - progress) * c + progress * probs
            else:
                # Final step: commit to the prediction
                c = probs

        # Threshold to binary
        return (c > 0.5).long()

    def decode_batch(
        self,
        syndromes: torch.Tensor,
        tau: Optional[torch.Tensor] = None,
        batch_size: int = 512,
    ) -> torch.Tensor:
        """Decode a large set of syndromes in batches.

        Parameters
        ----------
        syndromes : Tensor, shape (N, syndrome_dim)
        tau : Tensor or None, shape (N, tau_dim)
        batch_size : int
            Processing batch size.

        Returns
        -------
        corrections : Tensor, shape (N,)
            Predicted observable flips (0 or 1).
        """
        N = syndromes.shape[0]
        all_corrections = []

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            syn_batch = syndromes[start:end]
            tau_batch = tau[start:end] if tau is not None else None
            corr = self.sample(syn_batch, tau_batch)
            # Sum correction bits mod 2 to get observable flip
            # (for surface code, the observable is the parity of corrections
            #  along a logical operator path)
            obs_flip = corr.sum(dim=-1) % 2
            all_corrections.append(obs_flip)

        return torch.cat(all_corrections, dim=0)


# ---------------------------------------------------------------------------
# Training loss function
# ---------------------------------------------------------------------------

def compute_diffusion_loss(
    model: ScoreNet,
    noise_schedule: NoiseSchedule,
    syndrome: torch.Tensor,
    correction: torch.Tensor,
    tau: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute discrete diffusion training loss.

    Masks random correction bits according to the noise schedule, then
    trains the score network to predict the original (clean) bits at the
    masked positions via binary cross-entropy.

    Parameters
    ----------
    model : ScoreNet
        The score network.
    noise_schedule : NoiseSchedule
        Forward diffusion schedule.
    syndrome : Tensor, shape (batch, syndrome_dim)
        Detector outcomes.
    correction : Tensor, shape (batch, correction_dim)
        Ground-truth correction bits in {0, 1}.
    tau : Tensor or None, shape (batch, tau_dim)
        Optional retrodiction features.

    Returns
    -------
    loss : scalar Tensor
        Mean BCE over masked positions.
    """
    device = syndrome.device
    batch_size = syndrome.shape[0]

    # Ensure schedule tensors are on the right device
    noise_schedule.to(device)

    # Sample random timesteps
    t_idx = noise_schedule.sample_timesteps(batch_size, device)  # (batch,) long

    # Forward process: mask correction bits
    c_t = noise_schedule.forward_process(correction.float(), t_idx)  # (batch, correction_dim)

    # Score network predicts logits for clean correction
    logits = model(syndrome, c_t, t_idx, tau)  # (batch, correction_dim)

    # Identify masked positions (where c_t == 0.5)
    is_masked = (c_t == NoiseSchedule.MASK_VALUE).float()  # (batch, correction_dim)

    # BCE loss on masked positions only
    bce = F.binary_cross_entropy_with_logits(
        logits, correction.float(), reduction="none"
    )  # (batch, correction_dim)

    masked_bce = bce * is_masked
    n_masked = is_masked.sum().clamp(min=1.0)
    loss = masked_bce.sum() / n_masked

    return loss


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def build_decoder(
    syndrome_dim: int,
    correction_dim: int,
    tau_dim: int = 0,
    d_hidden: int = 256,
    n_layers: int = 4,
    d_time: int = 64,
    dropout: float = 0.1,
    T: int = 100,
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
    n_sample_steps: int = 20,
    device: Optional[torch.device] = None,
) -> tuple[ScoreNet, NoiseSchedule, DiffusionDecoder]:
    """Build a complete diffusion decoder system.

    Returns
    -------
    model : ScoreNet
        The score network (for training).
    noise_schedule : NoiseSchedule
        The noise schedule (for training loss).
    decoder : DiffusionDecoder
        The inference wrapper (for sampling).
    """
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    model = ScoreNet(
        syndrome_dim=syndrome_dim,
        correction_dim=correction_dim,
        tau_dim=tau_dim,
        d_hidden=d_hidden,
        n_layers=n_layers,
        d_time=d_time,
        dropout=dropout,
    ).to(device)

    noise_schedule = NoiseSchedule(T=T, beta_start=beta_start, beta_end=beta_end)
    noise_schedule.to(device)

    decoder = DiffusionDecoder(model, noise_schedule, n_sample_steps=n_sample_steps)

    return model, noise_schedule, decoder


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("ScoreNet + DiffusionDecoder self-test")
    print("=" * 60)

    # Pick device
    if torch.backends.mps.is_available():
        dev = torch.device("mps")
        print(f"Device: MPS (Apple Silicon)")
    else:
        dev = torch.device("cpu")
        print(f"Device: CPU")

    # Surface-17 parameters (example)
    n_det = 8        # syndrome detectors
    n_corr = 9       # data qubits (correction bits)
    n_tau = 8        # tau features (one per detector)

    print(f"\nCode: surface-17  (syndrome_dim={n_det}, correction_dim={n_corr})")

    # Build decoder system
    model, ns, decoder = build_decoder(
        syndrome_dim=n_det,
        correction_dim=n_corr,
        tau_dim=n_tau,
        d_hidden=128,       # smaller for test
        n_layers=3,
        T=50,
        n_sample_steps=10,
        device=dev,
    )

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Synthetic data
    batch = 16
    syndrome = torch.randint(0, 2, (batch, n_det), dtype=torch.float32, device=dev)
    correction = torch.randint(0, 2, (batch, n_corr), dtype=torch.float32, device=dev)
    tau = torch.rand(batch, n_tau, device=dev)

    # --- Training loss ---
    model.train()
    loss = compute_diffusion_loss(model, ns, syndrome, correction, tau)
    print(f"\nTraining loss (random init): {loss.item():.4f}")
    print(f"  Expected ~ln(2) = {math.log(2):.4f} for random predictions")

    # --- Backward pass ---
    loss.backward()
    grad_norm = sum(
        p.grad.norm().item() ** 2
        for p in model.parameters()
        if p.grad is not None
    ) ** 0.5
    print(f"  Gradient norm: {grad_norm:.4f}")

    # --- Training loss without tau ---
    model.zero_grad()
    model_no_tau, ns2, decoder_no_tau = build_decoder(
        syndrome_dim=n_det,
        correction_dim=n_corr,
        tau_dim=0,
        d_hidden=128,
        n_layers=3,
        T=50,
        device=dev,
    )
    loss_no_tau = compute_diffusion_loss(model_no_tau, ns2, syndrome, correction)
    print(f"\nTraining loss (no tau): {loss_no_tau.item():.4f}")

    # --- Sampling ---
    print(f"\nSampling (T={ns.T}, {decoder.n_sample_steps} steps)...")
    model.eval()
    samples = decoder.sample(syndrome, tau)
    print(f"  Sample shape: {samples.shape}")
    print(f"  Sample dtype: {samples.dtype}")
    print(f"  Sample values (first 3):\n{samples[:3]}")
    print(f"  All binary: {((samples == 0) | (samples == 1)).all().item()}")

    # --- Noise schedule info ---
    print(f"\nNoise schedule: {ns}")
    print(f"  alpha_bar[0]  = {ns.alpha_bar[0]:.6f}  (almost no masking)")
    print(f"  alpha_bar[-1] = {ns.alpha_bar[-1]:.6f}  (heavy masking)")
    print(f"  mask_prob[0]  = {ns.mask_prob[0]:.6f}")
    print(f"  mask_prob[-1] = {ns.mask_prob[-1]:.6f}")

    # --- Batch decode ---
    print(f"\nBatch decode test...")
    N = 64
    syns = torch.randint(0, 2, (N, n_det), dtype=torch.float32, device=dev)
    taus = torch.rand(N, n_tau, device=dev)
    obs_flips = decoder.decode_batch(syns, taus, batch_size=32)
    print(f"  decode_batch output shape: {obs_flips.shape}")
    print(f"  All values in {{0,1}}: {((obs_flips == 0) | (obs_flips == 1)).all().item()}")

    print("\n" + "=" * 60)
    print("All tests passed.")
    print("=" * 60)
