"""
SocialTransformer — the default no_profile model.

Self-contained: this file owns the entire forward pass top-to-bottom. No hooks,
no template-method indirection. Reads inline from embedder → backbone → heads.
"""

from typing import Dict, Optional

import torch

from models.base import BaseSocialTransformer
from models.components import (
    EventEmbedder,
    CausalTransformerEncoder,
    BanditPayoffsHead,
    DemoBeliefsHead,
)


class SocialTransformer(BaseSocialTransformer):
    """Default no_profile model.

    Architecture: ``embed → causal transformer → per-QUERY heads``. Heads read
    the QUERY token only (no demonstrator profiles). No experience head.

    Head input modes are hardcoded to ``query_only``: by construction, this
    model has no profiles to feed any head — anyone wanting profile-aware
    behavior should use ``ProfileSocialTransformer`` or
    ``LearnedProfileTransformer``.
    """

    def __init__(
        self,
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
    ) -> None:
        super().__init__()
        self.num_arms = num_arms
        self.num_demonstrators = num_demonstrators
        self.env_theta_loss = env_theta_loss

        self.embed = EventEmbedder(d_model, num_bandits, num_agents, num_arms, dropout)
        self.backbone = CausalTransformerEncoder(d_model, nhead, num_layers, dim_feedforward, dropout)
        self.bandit_payoffs_head = BanditPayoffsHead(
            d_model, num_arms,
            input_mode="query_only",
            num_demonstrators=num_demonstrators,
            env_theta_loss=env_theta_loss,
        )
        self.demo_beliefs_head = DemoBeliefsHead(
            d_model, num_arms, num_demonstrators,
            input_mode="query_only",
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

        # 3. Find QUERY positions in the original sequence.
        special_ids = batch["special_ids"]
        query_mask = (special_ids == 1)  # [B, T]
        B, T = special_ids.shape
        num_queries = int(query_mask.sum(dim=1).max().item())
        query_positions = torch.full((B, num_queries), -1, dtype=torch.long, device=special_ids.device)
        for b in range(B):
            qp = query_mask[b].nonzero(as_tuple=False).squeeze(-1)
            query_positions[b, :len(qp)] = qp

        # 4. Per-QUERY head dispatch (no profiles).
        alpha_theta_all, beta_theta_all, theta_logits_all = [], [], []
        alpha_demo_all, beta_demo_all = [], []
        batch_indices = torch.arange(B, device=h.device)
        for q in range(num_queries):
            query_emb = h[batch_indices, query_positions[:, q]]  # [B, d_model]

            if self.env_theta_loss == "bce":
                theta_logits_all.append(self.bandit_payoffs_head(query_emb))
            else:
                alpha_theta, beta_theta = self.bandit_payoffs_head(query_emb)
                alpha_theta_all.append(alpha_theta)
                beta_theta_all.append(beta_theta)

            alpha_demo, beta_demo = self.demo_beliefs_head(query_emb)
            alpha_demo_all.append(alpha_demo)
            beta_demo_all.append(beta_demo)

        # 5. Stack outputs.
        out: Dict[str, torch.Tensor] = {
            "alpha_demo": torch.stack(alpha_demo_all, dim=1),  # [B, num_queries, D, K]
            "beta_demo": torch.stack(beta_demo_all, dim=1),
        }
        if self.env_theta_loss == "bce":
            out["theta_logits"] = torch.stack(theta_logits_all, dim=1)  # [B, num_queries, K]
        else:
            out["alpha_theta"] = torch.stack(alpha_theta_all, dim=1)
            out["beta_theta"] = torch.stack(beta_theta_all, dim=1)

        # no_profile has no experience head — no N_tot_pred key emitted.
        return out
