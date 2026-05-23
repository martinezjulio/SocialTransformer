"""Inference-time utilities for the SocialTransformer family.

Thompson sampling over predicted Beta posteriors lives here so it can be reused
by ``run_inference.py`` and any other inference entry points without depending
on the model module hierarchy.
"""

import numpy as np
import torch


@torch.no_grad()
def thompson_sample_action(
    alpha_star: torch.Tensor,
    beta_star: torch.Tensor,
    np_rng: "np.random.Generator",
) -> int:
    """Draw a TS action from Beta(alpha_star, beta_star) per arm.

    Uses an explicit numpy Generator so callers control reproducibility
    without relying on torch's global RNG state.
    """
    alpha_np = alpha_star.detach().cpu().numpy()
    beta_np = beta_star.detach().cpu().numpy()
    draws = np_rng.beta(alpha_np, beta_np)
    return int(np.argmax(draws))
