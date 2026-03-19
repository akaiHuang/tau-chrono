"""
Diffusion-based QEC Decoder
============================

A discrete diffusion model for quantum error correction decoding.
The score network predicts denoising directions conditioned on syndrome
measurements and optional retrodiction (tau) priors.

Core components:
    - ScoreNet: Graph-aware score network with FiLM conditioning
    - NoiseSchedule: Linear beta schedule for discrete masking diffusion
    - DiffusionDecoder: End-to-end decoder wrapping forward/reverse processes

Compatible with CPU and MPS (Apple Silicon). No CUDA dependency.
"""

from .score_network import ScoreNet, DiffusionDecoder
from .noise_schedule import NoiseSchedule

__all__ = ["ScoreNet", "DiffusionDecoder", "NoiseSchedule"]
