
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
# Profile Extractors
# ---------------------------

class ProfileExtractor(nn.Module):
    """Base class for extracting demonstrator profile embeddings."""
    def forward(self, h: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            h: Transformer hidden states [B, T, d_model]
            batch: Batch dictionary with metadata
        
        Returns:
            profile_embeddings: [B, num_demonstrators, d_model]
        """
        raise NotImplementedError


class LearnedProfileExtractor(ProfileExtractor):
    """
    Learned profile tokens appended to sequence.
    Profile tokens are added before transformer and extracted after.
    """
    def __init__(self, d_model: int, num_demonstrators: int):
        super().__init__()
        self.num_demonstrators = num_demonstrators
        # Learnable profile tokens (one per demonstrator)
        self.profile_tokens = nn.Parameter(torch.randn(num_demonstrators, d_model))
    
    def get_profile_tokens(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Get profile tokens expanded for batch."""
        return self.profile_tokens.unsqueeze(0).expand(batch_size, -1, -1)  # [B, D, d_model]
    
    def forward(self, h: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract profile embeddings from last D positions (before QUERY token)."""
        # Profile tokens are at positions [-num_demonstrators-1:-1] (before QUERY at -1)
        return h[:, -(self.num_demonstrators + 1):-1, :]  # [B, D, d_model]


class AgentPoolingExtractor(ProfileExtractor):
    """
    Pool transformer hidden states by agent_id to create profile embeddings.
    """
    def __init__(self, num_demonstrators: int):
        super().__init__()
        self.num_demonstrators = num_demonstrators
    
    def forward(self, h: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Pool hidden states for each demonstrator.
        
        Args:
            h: [B, T, d_model]
            batch: contains 'agent_ids' [B, T]
        
        Returns:
            [B, num_demonstrators, d_model]
        """
        B, T, d_model = h.shape
        agent_ids = batch["agent_ids"]  # [B, T]
        
        profiles = []
        for demo_id in range(1, self.num_demonstrators + 1):  # demos are 1-indexed (0 is learner)
            # Mask for this demonstrator's tokens
            mask = (agent_ids == demo_id).unsqueeze(-1)  # [B, T, 1]
            
            # Mean pool over this demonstrator's tokens
            demo_h = h * mask.float()  # [B, T, d_model]
            demo_count = mask.float().sum(dim=1, keepdim=True)  # [B, 1, 1]
            demo_count = torch.clamp(demo_count, min=1.0)  # Avoid division by zero
            demo_profile = demo_h.sum(dim=1) / demo_count.squeeze(-1)  # [B, d_model]
            
            profiles.append(demo_profile)
        
        return torch.stack(profiles, dim=1)  # [B, D, d_model]


class CrossAttentionExtractor(ProfileExtractor):
    """
    Use learned query vectors with cross-attention to extract profile embeddings.
    """
    def __init__(self, d_model: int, num_demonstrators: int, nhead: int = 8):
        super().__init__()
        self.num_demonstrators = num_demonstrators
        # Learnable query vectors (one per demonstrator)
        self.query_vectors = nn.Parameter(torch.randn(num_demonstrators, d_model))
        # Cross-attention layer
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
    
    def forward(self, h: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Cross-attend from query vectors to transformer hidden states.
        
        Args:
            h: [B, T, d_model]
            batch: batch dictionary (for pad_mask if needed)
        
        Returns:
            [B, num_demonstrators, d_model]
        """
        B, T, d_model = h.shape
        
        # Expand queries for batch
        queries = self.query_vectors.unsqueeze(0).expand(B, -1, -1)  # [B, D, d_model]
        
        # Cross-attend: queries attend to hidden states
        # key_padding_mask: [B, T] where True means ignore (pad positions)
        key_padding_mask = batch.get("pad_mask", None)
        
        profiles, _ = self.cross_attn(
            query=queries,
            key=h,
            value=h,
            key_padding_mask=key_padding_mask
        )
        
        return profiles  # [B, D, d_model]


# ---------------------------
# Heads for theta and demo-start Betas
# ---------------------------

class DemoExperienceHead(nn.Module):
    """
    Predict demonstrator experience level (N_tot) from profile embeddings.
    Supports two output types:
    - "mse": continuous N_tot prediction
    - "bce": logits for expert vs novice binary classification
    """
    def __init__(self, d_model: int, num_demonstrators: int, output_type: str = "mse"):
        super().__init__()
        self.output_type = output_type
        self.num_demonstrators = num_demonstrators
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, profiles: torch.Tensor) -> torch.Tensor:
        """
        Args:
            profiles: [B, num_demonstrators, d_model]
        
        Returns:
            [B, num_demonstrators] - N_tot predictions (continuous) or logits (for BCE)
        """
        out = self.mlp(profiles).squeeze(-1)  # [B, D]
        if self.output_type == "mse":
            return F.softplus(out) + 1e-3  # Positive values for N_tot
        else:  # bce
            return out  # Return logits


class BanditPayoffsHead(nn.Module):
    """
    Predict bandit payoffs (theta) via Beta distribution parameters.
    """
    def __init__(self, d_model: int, num_arms: int):
        super().__init__()
        self.num_arms = num_arms
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2 * num_arms),  # alpha_theta, beta_theta
        )
        self.softplus = nn.Softplus()
    
    def forward(self, query_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query_emb: [B, d_model] - QUERY token embedding
        
        Returns:
            alpha_theta: [B, K]
            beta_theta: [B, K]
        """
        B = query_emb.shape[0]
        K = self.num_arms
        
        theta_out = self.mlp(query_emb)  # [B, 2K]
        theta_out = theta_out.view(B, 2, K)
        alpha_theta = self.softplus(theta_out[:, 0, :]) + 1e-3  # [B, K]
        beta_theta = self.softplus(theta_out[:, 1, :]) + 1e-3   # [B, K]
        
        return alpha_theta, beta_theta


class DemoBeliefsHead(nn.Module):
    """
    Predict demonstrator beliefs via Beta distribution parameters.
    Optionally conditions on profile embeddings.
    """
    def __init__(self, d_model: int, num_arms: int, num_demonstrators: int = 2, use_profiles: bool = False):
        super().__init__()
        self.num_arms = num_arms
        self.num_demonstrators = num_demonstrators
        self.use_profiles = use_profiles
        
        # Input size: query + optional profiles
        input_size = d_model + (d_model * num_demonstrators if use_profiles else 0)
        
        self.mlp = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2 * num_demonstrators * num_arms),  # alpha/beta per demo
        )
        self.softplus = nn.Softplus()
    
    def forward(self, query_emb: torch.Tensor, profile_embeddings: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query_emb: [B, d_model] - QUERY token embedding
            profile_embeddings: Optional [B, D, d_model] - demonstrator profiles
        
        Returns:
            alpha_demo: [B, D, K]
            beta_demo: [B, D, K]
        """
        B = query_emb.shape[0]
        K = self.num_arms
        D = self.num_demonstrators
        
        # Concatenate query with profile embeddings if provided
        if self.use_profiles and profile_embeddings is not None:
            # Flatten profile embeddings and concatenate
            profiles_flat = profile_embeddings.view(B, -1)  # [B, D*d_model]
            mlp_input = torch.cat([query_emb, profiles_flat], dim=-1)  # [B, d_model + D*d_model]
        else:
            mlp_input = query_emb
        
        demo_out = self.mlp(mlp_input)  # [B, 2*D*K]
        demo_out = demo_out.view(B, D, 2, K)
        alpha_demo = self.softplus(demo_out[:, :, 0, :]) + 1e-3  # [B, D, K]
        beta_demo = self.softplus(demo_out[:, :, 1, :]) + 1e-3   # [B, D, K]
        
        return alpha_demo, beta_demo


class Heads(nn.Module):
    """
    DEPRECATED: Kept for compatibility with old checkpoints.
    Use BanditPayoffsHead + DemoBeliefsHead instead.
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
    Supports multiple profile extraction methods for learning demonstrator skills.
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
        profile_method: str = "no_profile",
        experience_head_loss_type: str = "mse",
    ) -> None:
        super().__init__()
        self.profile_method = profile_method
        self.num_demonstrators = num_demonstrators
        
        self.embed = EventEmbedder(d_model, num_bandits, num_agents, num_arms, dropout)
        self.backbone = CausalTransformerEncoder(d_model, nhead, num_layers, dim_feedforward, dropout)
        
        # Separate heads for predictions
        self.bandit_payoffs_head = BanditPayoffsHead(d_model, num_arms)
        self.demo_beliefs_head = DemoBeliefsHead(
            d_model, num_arms, num_demonstrators, 
            use_profiles=(profile_method != "no_profile")
        )
        
        # Profile extractor and experience head (only for profile methods)
        if profile_method != "no_profile":
            if profile_method == "learned_profile":
                self.profile_extractor = LearnedProfileExtractor(d_model, num_demonstrators)
            elif profile_method == "agent_pooling":
                self.profile_extractor = AgentPoolingExtractor(num_demonstrators)
            elif profile_method == "cross_attention":
                self.profile_extractor = CrossAttentionExtractor(d_model, num_demonstrators, nhead)
            else:
                raise ValueError(f"Unknown profile_method: {profile_method}")
            
            self.demo_experience_head = DemoExperienceHead(
                d_model, num_demonstrators, output_type=experience_head_loss_type
            )
        else:
            self.profile_extractor = None
            self.demo_experience_head = None
        
        # Keep old heads for compatibility with old checkpoints
        self.heads = Heads(d_model, num_arms, num_demonstrators)

    def forward(self, batch: Dict[str, torch.Tensor], 
                custom_attn_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # Embed events
        x = self.embed(
            bandit_ids=batch["bandit_ids"],
            agent_ids=batch["agent_ids"],
            action_ids=batch["action_ids"],
            reward_ids=batch["reward_ids"],
            special_ids=batch["special_ids"],
            pad_mask=batch["pad_mask"],
        )
        
        # For learned_profile: append profile tokens before transformer
        if self.profile_method == "learned_profile":
            B = x.shape[0]
            device = x.device
            profile_tokens = self.profile_extractor.get_profile_tokens(B, device)  # [B, D, d_model]
            x = torch.cat([x, profile_tokens], dim=1)  # [B, T+D, d_model]
            
            # Extend pad_mask to cover profile tokens (they're not padded)
            if batch["pad_mask"] is not None:
                profile_mask = torch.zeros(B, self.num_demonstrators, dtype=torch.bool, device=device)
                extended_pad_mask = torch.cat([batch["pad_mask"], profile_mask], dim=1)
            else:
                extended_pad_mask = None
            
            # Extend attention mask if provided
            if custom_attn_mask is not None:
                T = custom_attn_mask.shape[0]
                D = self.num_demonstrators
                new_size = T + D
                extended_mask = torch.zeros(new_size, new_size, dtype=torch.bool, device=device)
                # Copy existing mask for event tokens
                extended_mask[:T, :T] = custom_attn_mask
                # Profile tokens can attend to all event tokens (but not future)
                extended_mask[T:, T:] = torch.triu(torch.ones(D, D, dtype=torch.bool, device=device), diagonal=1)
                custom_attn_mask = extended_mask
        else:
            extended_pad_mask = batch["pad_mask"]
        
        # Run transformer
        h = self.backbone(x, pad_mask=extended_pad_mask, custom_attn_mask=custom_attn_mask)
        
        # Extract QUERY token embedding
        query_emb = h[:, -1, :]  # [B, d_model]
        
        # Bandit payoffs (always computed)
        out = {}
        alpha_theta, beta_theta = self.bandit_payoffs_head(query_emb)
        out["alpha_theta"] = alpha_theta
        out["beta_theta"] = beta_theta
        
        # Demonstrator beliefs and experience
        if self.profile_method == "no_profile":
            # No profile embeddings
            alpha_demo, beta_demo = self.demo_beliefs_head(query_emb)
            out["alpha_demo"] = alpha_demo
            out["beta_demo"] = beta_demo
        else:
            # Extract profile embeddings
            profiles = self.profile_extractor(h, batch)  # [B, D, d_model]
            
            # Demonstrator beliefs conditioned on profiles
            alpha_demo, beta_demo = self.demo_beliefs_head(query_emb, profiles)
            out["alpha_demo"] = alpha_demo
            out["beta_demo"] = beta_demo
            
            # Direct N_tot prediction
            out["N_tot_pred"] = self.demo_experience_head(profiles)  # [B, D]
        
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