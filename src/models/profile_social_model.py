"""
Profile-aware SocialTransformer variants.

``ProfileSocialTransformer`` covers ``cross_attention`` and ``agent_pooling``.
``LearnedProfileTransformer`` inserts learnable profile tokens into the input
sequence pre-backbone.

Both classes are self-contained: each owns its full ``__init__`` and ``forward``
top-to-bottom, with no inheritance between them and no hooks on
``BaseSocialTransformer``.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from models.base import BaseSocialTransformer
from models.components import (
    EventEmbedder,
    CausalTransformerEncoder,
    BanditPayoffsHead,
    DemoBeliefsHead,
    DemoExperienceHead,
)


# ---------------------------
# Profile extractors
# ---------------------------

class ProfileExtractor(nn.Module):
    """Base class for extracting demonstrator profile embeddings."""
    def forward(self, h: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError


class LearnedProfileExtractor(ProfileExtractor):
    """Learnable profile tokens that get inserted into the sequence before each QUERY."""
    def __init__(self, d_model: int, num_demonstrators: int):
        super().__init__()
        self.num_demonstrators = num_demonstrators
        self.profile_tokens = nn.Parameter(torch.randn(num_demonstrators, d_model))

    def get_profile_tokens(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.profile_tokens.unsqueeze(0).expand(batch_size, -1, -1)  # [B, D, d_model]


class AgentPoolingExtractor(ProfileExtractor):
    """Mean-pool transformer hidden states grouped by agent_id."""
    def __init__(self, num_demonstrators: int):
        super().__init__()
        self.num_demonstrators = num_demonstrators

    def forward(self, h: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        agent_ids = batch["agent_ids"]  # [B, T]
        profiles = []
        for demo_id in range(1, self.num_demonstrators + 1):  # demos are 1-indexed (0 is learner)
            mask = (agent_ids == demo_id).unsqueeze(-1)  # [B, T, 1]
            demo_h = h * mask.float()
            demo_count = mask.float().sum(dim=1, keepdim=True)
            demo_count = torch.clamp(demo_count, min=1.0)
            demo_profile = demo_h.sum(dim=1) / demo_count.squeeze(-1)
            profiles.append(demo_profile)
        return torch.stack(profiles, dim=1)  # [B, D, d_model]


class CrossAttentionExtractor(ProfileExtractor):
    """Learned query vectors cross-attend over the hidden states to produce profiles."""
    def __init__(self, d_model: int, num_demonstrators: int, nhead: int = 8):
        super().__init__()
        self.num_demonstrators = num_demonstrators
        self.query_vectors = nn.Parameter(torch.randn(num_demonstrators, d_model))
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)

    def forward(self, h: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        B = h.shape[0]
        queries = self.query_vectors.unsqueeze(0).expand(B, -1, -1)
        key_padding_mask = batch.get("pad_mask", None)
        profiles, _ = self.cross_attn(
            query=queries,
            key=h,
            value=h,
            key_padding_mask=key_padding_mask,
        )
        return profiles  # [B, D, d_model]


# ---------------------------
# ProfileSocialTransformer (cross_attention + agent_pooling)
# ---------------------------

class ProfileSocialTransformer(BaseSocialTransformer):
    """SocialTransformer with a post-backbone profile extractor and experience head.

    Self-contained: own ``__init__``, own ``forward``. Supports
    ``profile_method='cross_attention'`` or ``'agent_pooling'``. For
    ``'learned_profile'`` use ``LearnedProfileTransformer`` (different forward
    because the sequence is mutated pre-backbone).
    """

    def __init__(
        self,
        *,
        profile_method: str,
        num_arms: int,
        num_agents: int,
        num_bandits: int,
        num_demonstrators: int = 2,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        env_theta_loss: str = "bce",
        experience_head_loss_type: str = "mse",
        theta_head_input: str = "query_only",
        demo_beliefs_head_input: str = "query_and_profiles",
        experience_head_input: str = "profiles_only",
    ) -> None:
        super().__init__()
        self.profile_method = profile_method
        self.num_arms = num_arms
        self.num_demonstrators = num_demonstrators
        self.env_theta_loss = env_theta_loss
        self.experience_head_input = experience_head_input

        self.embed = EventEmbedder(d_model, num_bandits, num_agents, num_arms, dropout)
        self.backbone = CausalTransformerEncoder(d_model, nhead, num_layers, dim_feedforward, dropout)

        if profile_method == "agent_pooling":
            self.profile_extractor: ProfileExtractor = AgentPoolingExtractor(num_demonstrators)
        elif profile_method == "cross_attention":
            self.profile_extractor = CrossAttentionExtractor(d_model, num_demonstrators, nhead)
        else:
            raise ValueError(
                f"ProfileSocialTransformer supports 'cross_attention' or 'agent_pooling'; "
                f"got '{profile_method}'. Use LearnedProfileTransformer for 'learned_profile'."
            )

        self.bandit_payoffs_head = BanditPayoffsHead(
            d_model, num_arms,
            input_mode=theta_head_input,
            num_demonstrators=num_demonstrators,
            env_theta_loss=env_theta_loss,
        )
        self.demo_beliefs_head = DemoBeliefsHead(
            d_model, num_arms, num_demonstrators,
            input_mode=demo_beliefs_head_input,
        )
        self.demo_experience_head = DemoExperienceHead(
            d_model, num_demonstrators,
            output_type=experience_head_loss_type,
            input_mode=experience_head_input,
        )

    def forward(self, batch: Dict[str, torch.Tensor],
                custom_attn_mask: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        # 1. Embed structured tokens.
        x = self.embed(
            bandit_ids=batch["bandit_ids"],
            agent_ids=batch["agent_ids"],
            action_ids=batch["action_ids"],
            reward_ids=batch["reward_ids"],
            special_ids=batch["special_ids"],
            pad_mask=batch["pad_mask"],
        )

        # 2. Causal transformer backbone.
        h = self.backbone(x, pad_mask=batch["pad_mask"], custom_attn_mask=custom_attn_mask)

        # 3. Compute per-demonstrator profiles once (used by every QUERY).
        profiles_global = self.profile_extractor(h, batch)  # [B, D, d_model]

        # 4. Find QUERY positions.
        special_ids = batch["special_ids"]
        query_mask = (special_ids == 1)
        B = special_ids.shape[0]
        num_queries = int(query_mask.sum(dim=1).max().item())
        query_positions = torch.full((B, num_queries), -1, dtype=torch.long, device=special_ids.device)
        for b in range(B):
            qp = query_mask[b].nonzero(as_tuple=False).squeeze(-1)
            query_positions[b, :len(qp)] = qp

        # 5. Per-QUERY head dispatch with shared profiles.
        alpha_theta_all, beta_theta_all, theta_logits_all = [], [], []
        alpha_demo_all, beta_demo_all = [], []
        query_emb_last = None  # for experience head
        batch_indices = torch.arange(B, device=h.device)
        for q in range(num_queries):
            query_emb = h[batch_indices, query_positions[:, q]]
            query_emb_last = query_emb

            if self.env_theta_loss == "bce":
                theta_logits_all.append(self.bandit_payoffs_head(query_emb, profiles_global))
            else:
                alpha_theta, beta_theta = self.bandit_payoffs_head(query_emb, profiles_global)
                alpha_theta_all.append(alpha_theta)
                beta_theta_all.append(beta_theta)

            alpha_demo, beta_demo = self.demo_beliefs_head(query_emb, profiles_global)
            alpha_demo_all.append(alpha_demo)
            beta_demo_all.append(beta_demo)

        # 6. Stack outputs.
        out: Dict[str, torch.Tensor] = {
            "alpha_demo": torch.stack(alpha_demo_all, dim=1),
            "beta_demo": torch.stack(beta_demo_all, dim=1),
        }
        if self.env_theta_loss == "bce":
            out["theta_logits"] = torch.stack(theta_logits_all, dim=1)
        else:
            out["alpha_theta"] = torch.stack(alpha_theta_all, dim=1)
            out["beta_theta"] = torch.stack(beta_theta_all, dim=1)

        # 7. Experience head — uses the last QUERY's context.
        if self.experience_head_input == "query_only":
            dummy_profiles = torch.zeros(B, self.num_demonstrators, h.shape[-1], device=h.device)
            out["N_tot_pred"] = self.demo_experience_head(dummy_profiles, query_emb=query_emb_last)
        elif self.experience_head_input == "profiles_only":
            out["N_tot_pred"] = self.demo_experience_head(profiles_global)
        else:  # query_and_profiles
            out["N_tot_pred"] = self.demo_experience_head(profiles_global, query_emb=query_emb_last)

        return out


# ---------------------------
# LearnedProfileTransformer (sequence-mutating variant)
# ---------------------------

class LearnedProfileTransformer(BaseSocialTransformer):
    """Inserts D learnable profile tokens into the token sequence before each QUERY.

    Self-contained: own ``__init__``, own ``forward``. Inherits from
    ``BaseSocialTransformer`` directly, not from ``ProfileSocialTransformer``.

    Because the sequence is mutated pre-backbone, QUERY positions in the
    post-backbone hidden states are shifted; per-QUERY profiles are read
    directly from the inserted slots.
    """

    def __init__(
        self,
        *,
        num_arms: int,
        num_agents: int,
        num_bandits: int,
        num_demonstrators: int = 2,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        env_theta_loss: str = "bce",
        experience_head_loss_type: str = "mse",
        theta_head_input: str = "query_only",
        demo_beliefs_head_input: str = "query_and_profiles",
        experience_head_input: str = "profiles_only",
    ) -> None:
        super().__init__()
        self.profile_method = "learned_profile"
        self.num_arms = num_arms
        self.num_demonstrators = num_demonstrators
        self.env_theta_loss = env_theta_loss
        self.experience_head_input = experience_head_input

        self.embed = EventEmbedder(d_model, num_bandits, num_agents, num_arms, dropout)
        self.backbone = CausalTransformerEncoder(d_model, nhead, num_layers, dim_feedforward, dropout)
        self.profile_extractor = LearnedProfileExtractor(d_model, num_demonstrators)
        self.bandit_payoffs_head = BanditPayoffsHead(
            d_model, num_arms,
            input_mode=theta_head_input,
            num_demonstrators=num_demonstrators,
            env_theta_loss=env_theta_loss,
        )
        self.demo_beliefs_head = DemoBeliefsHead(
            d_model, num_arms, num_demonstrators,
            input_mode=demo_beliefs_head_input,
        )
        self.demo_experience_head = DemoExperienceHead(
            d_model, num_demonstrators,
            output_type=experience_head_loss_type,
            input_mode=experience_head_input,
        )

    def forward(self, batch: Dict[str, torch.Tensor],
                custom_attn_mask: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        D = self.num_demonstrators

        # 1. Embed structured tokens.
        x = self.embed(
            bandit_ids=batch["bandit_ids"],
            agent_ids=batch["agent_ids"],
            action_ids=batch["action_ids"],
            reward_ids=batch["reward_ids"],
            special_ids=batch["special_ids"],
            pad_mask=batch["pad_mask"],
        )

        B = x.shape[0]
        device = x.device

        # 2. Insert D learnable profile tokens before each QUERY.
        # QUERY positions are assumed uniform across the batch.
        query_mask = (batch["special_ids"] == 1)
        query_positions_list = query_mask[0].nonzero(as_tuple=False).squeeze(-1).tolist()
        profile_tokens = self.profile_extractor.get_profile_tokens(B, device)  # [B, D, d_model]

        pieces = []
        last_idx = 0
        for q_pos in query_positions_list:
            if q_pos > last_idx:
                pieces.append(x[:, last_idx:q_pos, :])
            pieces.append(profile_tokens)
            last_idx = q_pos
        if last_idx < x.shape[1]:
            pieces.append(x[:, last_idx:, :])
        x = torch.cat(pieces, dim=1)

        # 3. Extend pad_mask in lockstep (profiles are never padding).
        if batch["pad_mask"] is not None:
            extended_pad_mask = batch["pad_mask"].clone()
            for q_idx, q_pos in enumerate(query_positions_list):
                insert_pos = q_pos + q_idx * D
                profile_mask = torch.zeros(B, D, dtype=torch.bool, device=device)
                extended_pad_mask = torch.cat([
                    extended_pad_mask[:, :insert_pos],
                    profile_mask,
                    extended_pad_mask[:, insert_pos:],
                ], dim=1)
        else:
            extended_pad_mask = None

        # 4. If a custom attention mask was provided, rebuild a plain causal
        #    mask at the new (longer) sequence length.
        if custom_attn_mask is not None:
            T_new = x.shape[1]
            custom_attn_mask = torch.triu(
                torch.ones(T_new, T_new, dtype=torch.bool, device=device), diagonal=1
            )

        # 5. Causal transformer backbone over the extended sequence.
        h = self.backbone(x, pad_mask=extended_pad_mask, custom_attn_mask=custom_attn_mask)

        # 6. Compute shifted QUERY positions: original p is now at p + q_idx*D + D
        #    (q_idx*D from prior insertions; +D from the insertion right before this QUERY).
        shifted = [p + q_idx * D + D for q_idx, p in enumerate(query_positions_list)]

        # 7. Per-QUERY head dispatch with per-query profile slice from inserted slots.
        alpha_theta_all, beta_theta_all, theta_logits_all = [], [], []
        alpha_demo_all, beta_demo_all = [], []
        query_emb_last = None
        profiles_for_exp = None
        batch_indices = torch.arange(B, device=h.device)
        for q_pos_actual in shifted:
            query_emb = h[batch_indices, torch.full((B,), q_pos_actual, dtype=torch.long, device=h.device)]
            profiles_q = h[:, q_pos_actual - D:q_pos_actual, :]  # [B, D, d_model]
            query_emb_last = query_emb
            profiles_for_exp = profiles_q

            if self.env_theta_loss == "bce":
                theta_logits_all.append(self.bandit_payoffs_head(query_emb, profiles_q))
            else:
                alpha_theta, beta_theta = self.bandit_payoffs_head(query_emb, profiles_q)
                alpha_theta_all.append(alpha_theta)
                beta_theta_all.append(beta_theta)

            alpha_demo, beta_demo = self.demo_beliefs_head(query_emb, profiles_q)
            alpha_demo_all.append(alpha_demo)
            beta_demo_all.append(beta_demo)

        # 8. Stack outputs.
        out: Dict[str, torch.Tensor] = {
            "alpha_demo": torch.stack(alpha_demo_all, dim=1),
            "beta_demo": torch.stack(beta_demo_all, dim=1),
        }
        if self.env_theta_loss == "bce":
            out["theta_logits"] = torch.stack(theta_logits_all, dim=1)
        else:
            out["alpha_theta"] = torch.stack(alpha_theta_all, dim=1)
            out["beta_theta"] = torch.stack(beta_theta_all, dim=1)

        # 9. Experience head — uses the last QUERY's context.
        if self.experience_head_input == "query_only":
            dummy_profiles = torch.zeros(B, self.num_demonstrators, h.shape[-1], device=h.device)
            out["N_tot_pred"] = self.demo_experience_head(dummy_profiles, query_emb=query_emb_last)
        elif self.experience_head_input == "profiles_only":
            out["N_tot_pred"] = self.demo_experience_head(profiles_for_exp)
        else:  # query_and_profiles
            out["N_tot_pred"] = self.demo_experience_head(profiles_for_exp, query_emb=query_emb_last)

        return out
