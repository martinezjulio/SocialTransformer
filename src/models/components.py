"""
Shared building blocks for the SocialTransformer family.

Pure library — these are nn.Modules that the concrete models compose. No
transformer class lives here.
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Token / Event embedding
# ---------------------------

class EventEmbedder(nn.Module):
    """
    Build token embeddings from structured fields:
    - bandit segment id (categorical, for "this comes from a different bandit")
    - agent id (0..num_agents-1). Convention: 0 for learner, 1..D for demonstrators
    - action id (0..K-1)
    - reward (0 or 1; use 2 for PAD/NA)
    - special marker (0=NONE, 1=QUERY, 2=BANDIT_SEP)
    We sum the field embeddings + positional encodings to get a token embedding.
    """
    def __init__(
        self,
        d_model: int,
        num_bandits: int,
        num_agents: int,
        num_actions: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.bandit_emb = nn.Embedding(num_bandits, d_model)
        self.agent_emb  = nn.Embedding(num_agents + 1, d_model)  # +1 for structural token padding (uses num_agents as index)
        self.action_emb = nn.Embedding(num_actions + 1, d_model)  # +1 for PAD action
        self.reward_emb = nn.Embedding(3, d_model)  # {0,1,2(PAD)}
        self.special_emb = nn.Embedding(3, d_model)  # NONE, QUERY, BANDIT_SEP
        self.pos_emb = nn.Embedding(4096, d_model)   # sufficient for long sequences
        self.ln = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        bandit_ids: torch.LongTensor,   # [B, T]
        agent_ids: torch.LongTensor,    # [B, T]
        action_ids: torch.LongTensor,   # [B, T]
        reward_ids: torch.LongTensor,   # [B, T]
        special_ids: torch.LongTensor,  # [B, T]
        pad_mask: Optional[torch.BoolTensor] = None,  # [B, T]
    ) -> torch.Tensor:
        B, T = bandit_ids.shape
        pos = torch.arange(T, device=bandit_ids.device)[None, :].expand(B, T)
        x = (
            self.bandit_emb(bandit_ids)
            + self.agent_emb(agent_ids)
            + self.action_emb(action_ids)
            + self.reward_emb(reward_ids)
            + self.special_emb(special_ids)
            + self.pos_emb(pos)
        )
        x = self.ln(x)
        x = self.drop(x)
        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        return x


# ---------------------------
# Transformer backbone
# ---------------------------

class CausalTransformerEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    @staticmethod
    def _causal_mask(T: int, device) -> torch.Tensor:
        mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
        return mask

    def forward(self, x: torch.Tensor,
                pad_mask: Optional[torch.BoolTensor] = None,
                custom_attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        T = x.shape[1]
        if custom_attn_mask is not None:
            attn_mask = custom_attn_mask
        else:
            attn_mask = self._causal_mask(T, x.device)
        return self.encoder(x, mask=attn_mask, src_key_padding_mask=pad_mask)


# ---------------------------
# Prediction heads
# ---------------------------

class BanditPayoffsHead(nn.Module):
    """Predict bandit payoffs (theta) via either BCE logits or Beta params."""
    def __init__(self, d_model: int, num_arms: int, input_mode: str = "query_only",
                 num_demonstrators: int = 2, env_theta_loss: str = "bce"):
        super().__init__()
        self.num_arms = num_arms
        self.input_mode = input_mode
        self.env_theta_loss = env_theta_loss
        out_dim = num_arms if env_theta_loss == "bce" else 2 * num_arms

        if input_mode == "query_only":
            input_size = d_model
        elif input_mode == "profiles_only":
            input_size = d_model * num_demonstrators
        elif input_mode == "query_and_profiles":
            input_size = d_model + d_model * num_demonstrators
        else:
            raise ValueError(f"Unknown input_mode: {input_mode}")

        self.mlp = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.ReLU(),
            nn.Linear(d_model, out_dim),
        )
        self.softplus = nn.Softplus()

    def forward(self, query_emb: torch.Tensor, profiles: Optional[torch.Tensor] = None
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B = query_emb.shape[0]
        K = self.num_arms

        if self.input_mode == "query_only":
            mlp_input = query_emb
        elif self.input_mode == "profiles_only":
            if profiles is None:
                raise ValueError("profiles_only mode requires profiles")
            mlp_input = profiles.reshape(B, -1)
        elif self.input_mode == "query_and_profiles":
            if profiles is None:
                raise ValueError("query_and_profiles mode requires profiles")
            mlp_input = torch.cat([query_emb, profiles.reshape(B, -1)], dim=-1)

        theta_out = self.mlp(mlp_input)
        if self.env_theta_loss == "bce":
            return theta_out
        theta_out = theta_out.view(B, 2, K)
        alpha_theta = self.softplus(theta_out[:, 0, :]) + 1e-3
        beta_theta = self.softplus(theta_out[:, 1, :]) + 1e-3
        return alpha_theta, beta_theta


class DemoBeliefsHead(nn.Module):
    """Predict per-demonstrator beliefs as Beta(α, β) per arm."""
    def __init__(self, d_model: int, num_arms: int, num_demonstrators: int = 2,
                 input_mode: str = "query_and_profiles"):
        super().__init__()
        self.num_arms = num_arms
        self.num_demonstrators = num_demonstrators
        self.input_mode = input_mode

        if input_mode == "query_only":
            input_size = d_model
        elif input_mode == "profiles_only":
            input_size = d_model * num_demonstrators
        elif input_mode == "query_and_profiles":
            input_size = d_model + d_model * num_demonstrators
        else:
            raise ValueError(f"Unknown input_mode: {input_mode}")

        self.mlp = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2 * num_demonstrators * num_arms),
        )
        self.softplus = nn.Softplus()

    def forward(self, query_emb: torch.Tensor, profiles: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = query_emb.shape[0]
        K = self.num_arms
        D = self.num_demonstrators

        if self.input_mode == "query_only":
            mlp_input = query_emb
        elif self.input_mode == "profiles_only":
            if profiles is None:
                raise ValueError("profiles_only mode requires profiles")
            mlp_input = profiles.reshape(B, -1)
        elif self.input_mode == "query_and_profiles":
            if profiles is None:
                raise ValueError("query_and_profiles mode requires profiles")
            mlp_input = torch.cat([query_emb, profiles.reshape(B, -1)], dim=-1)

        demo_out = self.mlp(mlp_input)
        demo_out = demo_out.view(B, D, 2, K)
        alpha_demo = self.softplus(demo_out[:, :, 0, :]) + 1e-3
        beta_demo = self.softplus(demo_out[:, :, 1, :]) + 1e-3
        return alpha_demo, beta_demo


class DemoExperienceHead(nn.Module):
    """Predict per-demonstrator experience (N_tot) — continuous or expert/novice logits.

    Only instantiated by profile-aware models; the no_profile model does not
    use this head.
    """
    def __init__(self, d_model: int, num_demonstrators: int,
                 output_type: str = "mse", input_mode: str = "profiles_only"):
        super().__init__()
        self.output_type = output_type
        self.num_demonstrators = num_demonstrators
        self.input_mode = input_mode

        if input_mode == "query_only":
            input_size = d_model
        elif input_mode == "profiles_only":
            input_size = d_model
        elif input_mode == "query_and_profiles":
            input_size = d_model + d_model
        else:
            raise ValueError(f"Unknown input_mode: {input_mode}")

        self.mlp = nn.Sequential(
            nn.Linear(input_size, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, profiles: torch.Tensor, query_emb: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        D = profiles.shape[1]
        outputs = []
        for demo_id in range(D):
            profile_d = profiles[:, demo_id, :]
            if self.input_mode == "query_only":
                if query_emb is None:
                    raise ValueError("query_only mode requires query_emb")
                mlp_input = query_emb
            elif self.input_mode == "profiles_only":
                mlp_input = profile_d
            elif self.input_mode == "query_and_profiles":
                if query_emb is None:
                    raise ValueError("query_and_profiles mode requires query_emb")
                mlp_input = torch.cat([query_emb, profile_d], dim=-1)
            out_d = self.mlp(mlp_input).squeeze(-1)
            outputs.append(out_d)
        out = torch.stack(outputs, dim=1)
        if self.output_type == "mse":
            return F.softplus(out) + 1e-3
        return out  # bce → raw logits
