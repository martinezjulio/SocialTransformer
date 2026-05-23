"""SocialTransformer model package.

Dispatch via ``build_model(profile_method, config)``; import concrete classes
directly when you need them.
"""

from typing import Dict

import torch.nn as nn

from models.base import BaseSocialTransformer
from models.social_model import SocialTransformer
from models.profile_social_model import (
    ProfileSocialTransformer,
    LearnedProfileTransformer,
)

__all__ = [
    "BaseSocialTransformer",
    "SocialTransformer",
    "ProfileSocialTransformer",
    "LearnedProfileTransformer",
    "build_model",
]


def build_model(profile_method: str, config: Dict) -> nn.Module:
    """Construct the right model class for the requested profile method.

    Args:
        profile_method: One of "no_profile", "cross_attention",
            "agent_pooling", "learned_profile".
        config: Dict with model hyperparameters. Required keys: num_arms,
            num_agents, num_bandits, num_demonstrators, d_model, nhead,
            num_layers, dim_feedforward, dropout. Optional keys (with sensible
            defaults): env_theta_loss, experience_head_loss_type,
            theta_head_input, demo_beliefs_head_input, experience_head_input.
            Extra keys are ignored; head-input keys are ignored when
            profile_method='no_profile'.
    """
    common = dict(
        num_arms=config['num_arms'],
        num_agents=config['num_agents'],
        num_bandits=config['num_bandits'],
        num_demonstrators=config['num_demonstrators'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        env_theta_loss=config.get('env_theta_loss', 'bce'),
    )

    if profile_method == "no_profile":
        return SocialTransformer(**common)

    profiled_extras = dict(
        experience_head_loss_type=config.get('experience_head_loss_type', 'mse'),
        theta_head_input=config.get('theta_head_input', 'query_only'),
        demo_beliefs_head_input=config.get('demo_beliefs_head_input', 'query_and_profiles'),
        experience_head_input=config.get('experience_head_input', 'profiles_only'),
    )

    if profile_method == "learned_profile":
        return LearnedProfileTransformer(**common, **profiled_extras)

    if profile_method in ("cross_attention", "agent_pooling"):
        return ProfileSocialTransformer(
            profile_method=profile_method, **common, **profiled_extras
        )

    raise ValueError(
        f"Unknown profile_method '{profile_method}'. Must be one of "
        "'no_profile', 'cross_attention', 'agent_pooling', 'learned_profile'."
    )
