
import math
from typing import Optional, Dict, Tuple

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
        self.agent_emb  = nn.Embedding(num_agents, d_model)
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
        # shape [T, T], True where positions should be masked (upper triangle)
        mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
        return mask

    def forward(self, x: torch.Tensor, 
                pad_mask: Optional[torch.BoolTensor] = None,
                custom_attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        
        if custom_attn_mask is not None:
            attn_mask = custom_attn_mask
        else:
            attn_mask = self._causal_mask(T, x.device)  # same for all batch items
        
        h = self.encoder(x, mask=attn_mask, src_key_padding_mask=pad_mask)
        return h


# ---------------------------
# Heads for theta and demo-start Betas
# ---------------------------

class Heads(nn.Module):
    """
    Produce per-arm positive parameters:
    - theta head: (alpha_theta, beta_theta)  (interpreted as a Beta over the true payoff theta_k)
    - demo head: (alpha_demo, beta_demo) per demonstrator (start-of-tail Beta shapes)
    We read out from a pooled representation (use the last token by default or a QUERY token embedding).
    """
    def __init__(self, d_model: int, num_arms: int, num_demonstrators: int = 2):
        super().__init__()
        self.num_arms = num_arms
        self.num_demonstrators = num_demonstrators
        
        # Separate MLPs for theta and demonstrator beliefs
        self.mlp_theta = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2 * num_arms),  # alpha_theta, beta_theta
        )
        
        self.mlp_demo = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2 * num_demonstrators * num_arms),  # alpha/beta per demo
        )
        
        self.softplus = nn.Softplus()

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        # h: [B, T, D]; take last token (assumed to be QUERY)
        z = h[:, -1, :]  # [B, D]
        B = z.shape[0]
        K = self.num_arms
        D = self.num_demonstrators
        
        # Theta prediction
        theta_out = self.mlp_theta(z)  # [B, 2K]
        theta_out = theta_out.view(B, 2, K)
        alpha_theta = self.softplus(theta_out[:, 0, :]) + 1e-3  # [B, K]
        beta_theta = self.softplus(theta_out[:, 1, :]) + 1e-3   # [B, K]
        
        # Per-demonstrator predictions
        demo_out = self.mlp_demo(z)  # [B, 2*D*K]
        demo_out = demo_out.view(B, D, 2, K)
        alpha_demo = self.softplus(demo_out[:, :, 0, :]) + 1e-3  # [B, D, K]
        beta_demo = self.softplus(demo_out[:, :, 1, :]) + 1e-3   # [B, D, K]
        
        return {
            "alpha_theta": alpha_theta,     # [B, K]
            "beta_theta": beta_theta,       # [B, K]
            "alpha_demo": alpha_demo,       # [B, D, K]
            "beta_demo": beta_demo,         # [B, D, K]
        }


class SBIPosteriorTransformer(nn.Module):
    """
    Full model: event embedder -> causal transformer -> readout heads.
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
    ) -> None:
        super().__init__()
        self.embed = EventEmbedder(d_model, num_bandits, num_agents, num_arms, dropout)
        self.backbone = CausalTransformerEncoder(d_model, nhead, num_layers, dim_feedforward, dropout)
        self.heads = Heads(d_model, num_arms, num_demonstrators)

    def forward(self, batch: Dict[str, torch.Tensor], 
                custom_attn_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        x = self.embed(
            bandit_ids=batch["bandit_ids"],
            agent_ids=batch["agent_ids"],
            action_ids=batch["action_ids"],
            reward_ids=batch["reward_ids"],
            special_ids=batch["special_ids"],
            pad_mask=batch["pad_mask"],
        )
        h = self.backbone(x, pad_mask=batch["pad_mask"], custom_attn_mask=custom_attn_mask)
        out = self.heads(h)
        return out


# ---------------------------
# Thompson Sampling helper (inference-time)
# ---------------------------

@torch.no_grad()
def thompson_sample_action(alpha_star: torch.Tensor, beta_star: torch.Tensor) -> int:
    """
    Draw a TS action from Beta(alpha_star, beta_star) per arm.
    alpha_star, beta_star: [K]
    """
    K = alpha_star.shape[0]
    dist = torch.distributions.Beta(alpha_star, beta_star)
    draws = dist.sample()
    return int(torch.argmax(draws).item())