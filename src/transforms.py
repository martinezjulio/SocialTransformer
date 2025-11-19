"""
Normalization transforms for belief parameters.
"""
import torch
from typing import Union


class BeliefNormalizer:
    """
    Normalizes and denormalizes demonstrator belief parameters (alpha, beta).
    
    Used when training with log_mse loss to help output layer learn values in [0,1] range
    instead of [1, N_max + alpha0 + beta0] range.
    """
    
    def __init__(self, N_max: float, alpha0: float = 1.0, beta0: float = 1.0):
        """
        Args:
            N_max: Maximum pre-episode experience (from dataset metadata)
            alpha0: Prior alpha parameter
            beta0: Prior beta parameter
        """
        self.N_max = N_max
        self.alpha0 = alpha0
        self.beta0 = beta0
        # Maximum possible value for alpha or beta: prior + N_max pulls all to one outcome
        self.norm_factor = N_max + alpha0 + beta0
    
    def normalize_alpha(self, alpha: Union[torch.Tensor, float]) -> Union[torch.Tensor, float]:
        """Normalize alpha parameter to approximately [0, 1] range."""
        return alpha / self.norm_factor
    
    def normalize_beta(self, beta: Union[torch.Tensor, float]) -> Union[torch.Tensor, float]:
        """Normalize beta parameter to approximately [0, 1] range."""
        return beta / self.norm_factor
    
    def denormalize_alpha(self, alpha_norm: Union[torch.Tensor, float]) -> Union[torch.Tensor, float]:
        """Denormalize alpha parameter back to Beta distribution space."""
        return alpha_norm * self.norm_factor
    
    def denormalize_beta(self, beta_norm: Union[torch.Tensor, float]) -> Union[torch.Tensor, float]:
        """Denormalize beta parameter back to Beta distribution space."""
        return beta_norm * self.norm_factor
    
    def normalize(self, alpha: torch.Tensor, beta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize both alpha and beta together."""
        return self.normalize_alpha(alpha), self.normalize_beta(beta)
    
    def denormalize(self, alpha_norm: torch.Tensor, beta_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Denormalize both alpha and beta together."""
        return self.denormalize_alpha(alpha_norm), self.denormalize_beta(beta_norm)
    
    def __repr__(self):
        return f"BeliefNormalizer(N_max={self.N_max}, alpha0={self.alpha0}, beta0={self.beta0}, norm_factor={self.norm_factor})"

