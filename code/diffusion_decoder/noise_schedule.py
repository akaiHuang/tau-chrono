"""
Noise schedule for discrete masking diffusion.

Forward process: q(c_t | c_0) masks correction bits with probability beta_t.
    c_t[i] = c_0[i]  with probability (1 - beta_t)
    c_t[i] = MASK     with probability beta_t

The MASK value is represented as 0.5 in continuous space (midpoint between
the two binary states 0 and 1), making the network's job clear: move away
from 0.5 toward the correct binary value.

The cumulative masking probability alpha_bar_t = prod_{s=1}^{t} (1 - beta_s)
gives the probability that a bit has survived unmasked up to step t.
"""

import torch
import numpy as np


class NoiseSchedule:
    """Linear beta schedule for discrete masking diffusion.

    Parameters
    ----------
    T : int
        Number of diffusion timesteps.
    beta_start : float
        Starting noise rate (small, near 0).
    beta_end : float
        Ending noise rate (larger, but << 1 per step).
    """

    MASK_VALUE = 0.5  # continuous representation of the masked state

    def __init__(
        self,
        T: int = 100,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        self.T = T
        self.beta_start = beta_start
        self.beta_end = beta_end

        # Linear schedule: beta_1, beta_2, ..., beta_T
        betas = np.linspace(beta_start, beta_end, T, dtype=np.float64)
        self.betas = torch.tensor(betas, dtype=torch.float32)  # (T,)

        # alpha_t = 1 - beta_t
        alphas = 1.0 - betas
        self.alphas = torch.tensor(alphas, dtype=torch.float32)  # (T,)

        # alpha_bar_t = cumulative product of alphas
        # This is the probability a bit is still unmasked at step t
        alpha_bar = np.cumprod(alphas)
        self.alpha_bar = torch.tensor(alpha_bar, dtype=torch.float32)  # (T,)

        # Precompute mask probability at each step: 1 - alpha_bar_t
        self.mask_prob = 1.0 - self.alpha_bar  # (T,)

    def to(self, device: torch.device) -> "NoiseSchedule":
        """Move all tensors to the specified device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        self.mask_prob = self.mask_prob.to(device)
        return self

    def forward_process(
        self,
        c_0: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Apply forward masking diffusion: q(c_t | c_0).

        Parameters
        ----------
        c_0 : torch.Tensor
            Clean correction bits, shape (batch, correction_dim). Values in {0, 1}.
        t : torch.Tensor
            Timestep indices, shape (batch,). Integer values in [0, T-1].

        Returns
        -------
        c_t : torch.Tensor
            Noisy (masked) corrections, shape (batch, correction_dim).
            Unmasked bits keep their original value; masked bits become MASK_VALUE.
        """
        batch_size = c_0.shape[0]
        device = c_0.device

        # Probability of masking each bit at timestep t
        # mask_prob[t] = 1 - alpha_bar[t]
        p_mask = self.mask_prob[t]  # (batch,)
        p_mask = p_mask.unsqueeze(-1)  # (batch, 1) for broadcasting

        # Sample mask: 1 where we mask, 0 where we keep
        mask = torch.bernoulli(p_mask.expand_as(c_0)).to(device)  # (batch, correction_dim)

        # Apply: masked positions get MASK_VALUE, others keep c_0
        c_t = c_0 * (1.0 - mask) + self.MASK_VALUE * mask

        return c_t

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample random timesteps uniformly from [0, T-1].

        Parameters
        ----------
        batch_size : int
            Number of timesteps to sample.
        device : torch.device
            Device for the output tensor.

        Returns
        -------
        t : torch.Tensor
            Random timestep indices, shape (batch_size,). dtype=torch.long.
        """
        return torch.randint(0, self.T, (batch_size,), device=device)

    def get_beta(self, t: torch.Tensor) -> torch.Tensor:
        """Get beta values for given timesteps."""
        return self.betas[t]

    def get_alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """Get cumulative alpha_bar values for given timesteps."""
        return self.alpha_bar[t]

    def __repr__(self) -> str:
        return (
            f"NoiseSchedule(T={self.T}, "
            f"beta=[{self.beta_start:.4f}, {self.beta_end:.4f}], "
            f"final_mask_prob={self.mask_prob[-1]:.4f})"
        )
