"""SocialTransformer model package.

Dispatch via ``build_model(model_type, config)``; import concrete classes
directly when you need them.

``social`` is the only model type in use. The three ``social_profile_*``
variants are DEPRECATED exploratory paths: no published result used one, and
every sweep config in ``src/Sweeps/`` pins ``model_type: "social"``. They are
kept so old checkpoints still load — prefer ``social`` for new work.
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


# Legacy → current value translation. Keeps pre-rename checkpoints and any
# leftover sweep configs loadable; safe to remove once nothing in the wild
# still uses the old names.
_LEGACY_ALIASES = {
    "no_profile":      "social",
    "learned_profile": "social_profile_learned",
    "agent_pooling":   "social_profile_agent_pooling",
    "cross_attention": "social_profile_cross_attention",
}


def build_model(model_type: str, config: Dict) -> nn.Module:
    """Construct the right model class for the requested model_type.

    Args:
        model_type: One of "social", "social_profile_learned",
            "social_profile_agent_pooling", "social_profile_cross_attention".
            The four pre-rename strings ("no_profile", "learned_profile",
            "agent_pooling", "cross_attention") are also accepted via an
            alias map and resolve to the corresponding new value.

            Use "social": it is what every published result was produced
            with. The three "social_profile_*" variants are DEPRECATED and
            unvalidated. "social_profile_learned" in particular discards the
            segment-cutoff attention mask it is handed (it rebuilds a plain
            causal mask at the spliced sequence length — see
            profile_social_model.py), so it trains on full context while the
            other types train on the truncated context train.py intends.
        config: Dict with model hyperparameters. Required keys: num_arms,
            num_agents, num_bandits, num_demonstrators, d_model, nhead,
            num_layers, dim_feedforward, dropout. Optional keys (with sensible
            defaults): env_theta_loss, experience_head_loss_type,
            theta_head_input, demo_beliefs_head_input, experience_head_input.
            Extra keys are ignored; head-input keys are ignored when
            model_type='social'.
    """
    model_type = _LEGACY_ALIASES.get(model_type, model_type)

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

    if model_type == "social":
        return SocialTransformer(**common)

    profiled_extras = dict(
        experience_head_loss_type=config.get('experience_head_loss_type', 'mse'),
        theta_head_input=config.get('theta_head_input', 'query_only'),
        demo_beliefs_head_input=config.get('demo_beliefs_head_input', 'query_and_profiles'),
        experience_head_input=config.get('experience_head_input', 'profiles_only'),
    )

    if model_type == "social_profile_learned":
        return LearnedProfileTransformer(**common, **profiled_extras)

    if model_type in ("social_profile_cross_attention", "social_profile_agent_pooling"):
        return ProfileSocialTransformer(
            model_type=model_type, **common, **profiled_extras
        )

    raise ValueError(
        f"Unknown model_type {model_type!r}. Must be one of "
        "'social', 'social_profile_learned', 'social_profile_agent_pooling', "
        "'social_profile_cross_attention'."
    )
