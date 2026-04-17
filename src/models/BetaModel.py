
import math
from typing import Optional, Dict, Tuple

import numpy as np
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
    Configurable to use query embeddings and/or profile embeddings.
    Supports two output types:
    - "mse": continuous N_tot prediction
    - "bce": logits for expert vs novice binary classification
    """
    def __init__(self, d_model: int, num_demonstrators: int, output_type: str = "mse", input_mode: str = "profiles_only"):
        super().__init__()
        self.output_type = output_type
        self.num_demonstrators = num_demonstrators
        self.input_mode = input_mode
        
        # Determine input size (per demonstrator)
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
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, profiles: torch.Tensor, query_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            profiles: [B, num_demonstrators, d_model]
            query_emb: Optional [B, d_model] - QUERY token hidden state
        
        Returns:
            [B, num_demonstrators] - N_tot predictions (continuous) or logits (for BCE)
        """
        B, D, d = profiles.shape
        outputs = []
        
        for demo_id in range(D):
            profile_d = profiles[:, demo_id, :]  # [B, d_model]
            
            # Build input based on mode
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
            
            out_d = self.mlp(mlp_input).squeeze(-1)  # [B]
            outputs.append(out_d)
        
        out = torch.stack(outputs, dim=1)  # [B, D]
        
        if self.output_type == "mse":
            return F.softplus(out) + 1e-3  # Positive values for N_tot
        else:  # bce
            return out  # Return logits


class BanditPayoffsHead(nn.Module):
    """
    Predict bandit payoffs (theta) via Beta distribution parameters.
    Configurable to use query embeddings and/or profile embeddings.
    """
    def __init__(self, d_model: int, num_arms: int, input_mode: str = "query_only", num_demonstrators: int = 2):
        super().__init__()
        self.num_arms = num_arms
        self.input_mode = input_mode
        
        # Determine input size based on mode
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
            nn.Linear(d_model, 2 * num_arms),
        )
        self.softplus = nn.Softplus()
    
    def forward(self, query_emb: torch.Tensor, profiles: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query_emb: [B, d_model] - QUERY token hidden state
            profiles: Optional [B, D, d_model] - profile hidden states
        
        Returns:
            alpha_theta: [B, K]
            beta_theta: [B, K]
        """
        B = query_emb.shape[0]
        K = self.num_arms
        
        # Build input based on mode
        if self.input_mode == "query_only":
            mlp_input = query_emb
        elif self.input_mode == "profiles_only":
            if profiles is None:
                raise ValueError("profiles_only mode requires profiles")
            mlp_input = profiles.view(B, -1)  # [B, D*d_model]
        elif self.input_mode == "query_and_profiles":
            if profiles is None:
                raise ValueError("query_and_profiles mode requires profiles")
            mlp_input = torch.cat([query_emb, profiles.view(B, -1)], dim=-1)
        
        theta_out = self.mlp(mlp_input)
        theta_out = theta_out.view(B, 2, K)
        alpha_theta = self.softplus(theta_out[:, 0, :]) + 1e-3
        beta_theta = self.softplus(theta_out[:, 1, :]) + 1e-3
        
        return alpha_theta, beta_theta


class DemoBeliefsHead(nn.Module):
    """
    Predict demonstrator beliefs via Beta distribution parameters.
    Configurable to use query embeddings and/or profile embeddings.
    """
    def __init__(self, d_model: int, num_arms: int, num_demonstrators: int = 2, input_mode: str = "query_and_profiles"):
        super().__init__()
        self.num_arms = num_arms
        self.num_demonstrators = num_demonstrators
        self.input_mode = input_mode
        
        # Determine input size based on mode
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
    
    def forward(self, query_emb: torch.Tensor, profiles: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query_emb: [B, d_model] - QUERY token hidden state
            profiles: Optional [B, D, d_model] - profile hidden states
        
        Returns:
            alpha_demo: [B, D, K]
            beta_demo: [B, D, K]
        """
        B = query_emb.shape[0]
        K = self.num_arms
        D = self.num_demonstrators
        
        # Build input based on mode
        if self.input_mode == "query_only":
            mlp_input = query_emb
        elif self.input_mode == "profiles_only":
            if profiles is None:
                raise ValueError("profiles_only mode requires profiles")
            mlp_input = profiles.view(B, -1)
        elif self.input_mode == "query_and_profiles":
            if profiles is None:
                raise ValueError("query_and_profiles mode requires profiles")
            mlp_input = torch.cat([query_emb, profiles.view(B, -1)], dim=-1)
        
        demo_out = self.mlp(mlp_input)
        demo_out = demo_out.view(B, D, 2, K)
        alpha_demo = self.softplus(demo_out[:, :, 0, :]) + 1e-3
        beta_demo = self.softplus(demo_out[:, :, 1, :]) + 1e-3
        
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
        theta_head_input: str = "query_only",
        demo_beliefs_head_input: str = "query_and_profiles",
        experience_head_input: str = "profiles_only",
    ) -> None:
        super().__init__()
        self.profile_method = profile_method
        self.num_demonstrators = num_demonstrators
        
        # Validate and override input modes for no_profile method
        if profile_method == "no_profile":
            if theta_head_input != "query_only":
                print(f"Warning: profile_method='no_profile', forcing theta_head_input='query_only'")
                theta_head_input = "query_only"
            if demo_beliefs_head_input != "query_only":
                print(f"Warning: profile_method='no_profile', forcing demo_beliefs_head_input='query_only'")
                demo_beliefs_head_input = "query_only"
            if experience_head_input != "query_only":
                print(f"Warning: profile_method='no_profile', forcing experience_head_input='query_only'")
                experience_head_input = "query_only"
        
        # Store for later use
        self.theta_head_input = theta_head_input
        self.demo_beliefs_head_input = demo_beliefs_head_input
        self.experience_head_input = experience_head_input
        
        self.embed = EventEmbedder(d_model, num_bandits, num_agents, num_arms, dropout)
        self.backbone = CausalTransformerEncoder(d_model, nhead, num_layers, dim_feedforward, dropout)
        
        # Separate heads with configurable input modes
        self.bandit_payoffs_head = BanditPayoffsHead(
            d_model, num_arms,
            input_mode=theta_head_input,
            num_demonstrators=num_demonstrators
        )
        self.demo_beliefs_head = DemoBeliefsHead(
            d_model, num_arms, num_demonstrators,
            input_mode=demo_beliefs_head_input
        )
        
        # Profile extractor and experience head
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
                d_model, num_demonstrators,
                output_type=experience_head_loss_type,
                input_mode=experience_head_input
            )
        else:
            self.profile_extractor = None
            # Still create experience head for no_profile (uses query_only)
            self.demo_experience_head = DemoExperienceHead(
                d_model, num_demonstrators,
                output_type=experience_head_loss_type,
                input_mode="query_only"
            )
        
        # Keep old heads for compatibility with old checkpoints
        self.heads = Heads(d_model, num_arms, num_demonstrators)

    def _find_query_positions(self, special_ids: torch.Tensor) -> torch.Tensor:
        """
        Find positions of QUERY tokens in sequence.
        
        Args:
            special_ids: [B, T] tensor of special IDs
        
        Returns:
            query_positions: [B, num_queries] tensor of QUERY positions (-1 for padding)
        """
        # QUERY has special_id = 1
        query_mask = (special_ids == 1)  # [B, T]
        
        # Get positions for each batch item
        B, T = special_ids.shape
        max_queries = query_mask.sum(dim=1).max().item()
        
        query_positions = torch.full((B, max_queries), -1, dtype=torch.long, device=special_ids.device)
        
        for b in range(B):
            query_pos_b = query_mask[b].nonzero(as_tuple=False).squeeze(-1)
            query_positions[b, :len(query_pos_b)] = query_pos_b
        
        return query_positions

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
        
        # For learned_profile: insert profile tokens before each QUERY
        if self.profile_method == "learned_profile":
            B = x.shape[0]
            device = x.device
            
            # Find QUERY positions in original sequence
            query_mask = (batch["special_ids"] == 1)  # [B, T]
            # Assuming uniform structure across batch
            query_positions_list = query_mask[0].nonzero(as_tuple=False).squeeze(-1).tolist()
            num_queries = len(query_positions_list)
            
            # Get profile tokens (learned parameters)
            profile_tokens = self.profile_extractor.get_profile_tokens(B, device)  # [B, D, d_model]
            
            # Insert profile tokens before each QUERY
            pieces = []
            last_idx = 0
            
            for q_pos in query_positions_list:
                # Add tokens up to (but not including) QUERY
                if q_pos > last_idx:
                    pieces.append(x[:, last_idx:q_pos, :])
                # Insert profiles before QUERY
                pieces.append(profile_tokens)
                last_idx = q_pos
            
            # Add remaining tokens (QUERYs and anything after)
            if last_idx < x.shape[1]:
                pieces.append(x[:, last_idx:, :])
            
            # Concatenate all pieces
            x = torch.cat(pieces, dim=1)  # [B, T + num_queries*D, d_model]
            
            # Extend pad_mask for inserted profile tokens
            if batch["pad_mask"] is not None:
                extended_pad_mask = batch["pad_mask"].clone()
                for q_idx, q_pos in enumerate(query_positions_list):
                    # Calculate insertion position (accounting for previously inserted profiles)
                    insert_pos = q_pos + q_idx * self.num_demonstrators
                    # Insert D False values (profiles are not padding)
                    profile_mask = torch.zeros(B, self.num_demonstrators, dtype=torch.bool, device=device)
                    extended_pad_mask = torch.cat([
                        extended_pad_mask[:, :insert_pos],
                        profile_mask,
                        extended_pad_mask[:, insert_pos:]
                    ], dim=1)
            else:
                extended_pad_mask = None
            
            # Custom attention mask extension (if provided)
            # Profiles use standard causal attention
            if custom_attn_mask is not None:
                # Rebuild mask for new sequence length
                T_new = x.shape[1]
                extended_mask = torch.triu(torch.ones(T_new, T_new, dtype=torch.bool, device=device), diagonal=1)
                # Profiles get standard causal (already set by triu)
                custom_attn_mask = extended_mask
        else:
            extended_pad_mask = batch["pad_mask"]
        
        # Run transformer
        h = self.backbone(x, pad_mask=extended_pad_mask, custom_attn_mask=custom_attn_mask)
        
        # Find QUERY positions (in potentially extended sequence)
        if self.profile_method == "learned_profile":
            # QUERY positions shifted by inserted profiles
            # Original QUERY at pos Q is now at Q + q_idx * D + D (profiles before it)
            query_mask_orig = (batch["special_ids"] == 1)
            query_positions_orig_list = query_mask_orig[0].nonzero(as_tuple=False).squeeze(-1).tolist()
            num_queries = len(query_positions_orig_list)
            B = h.shape[0]
        else:
            # QUERY positions unchanged
            query_positions = self._find_query_positions(batch["special_ids"])  # [B, num_queries]
            B, num_queries = query_positions.shape
        
        # Extract global profiles for non-learned methods
        if self.profile_method != "no_profile" and self.profile_method != "learned_profile":
            profiles_global = self.profile_extractor(h, batch)  # [B, D, d_model]
        
        # Run predictions for each QUERY
        alpha_theta_all = []
        beta_theta_all = []
        alpha_demo_all = []
        beta_demo_all = []
        query_embs_all = []  # Store for experience head
        profiles_for_exp = None  # Will be set to last profiles
        
        for q in range(num_queries):
            if self.profile_method == "learned_profile":
                # Calculate actual QUERY position (shifted by profile insertions)
                query_pos_orig = query_positions_orig_list[q]
                query_pos_actual = query_pos_orig + q * self.num_demonstrators + self.num_demonstrators
                
                # Extract QUERY hidden state
                batch_indices = torch.arange(B, device=h.device)
                query_emb = h[batch_indices, query_pos_actual]  # [B, d_model]
                
                # Extract profiles for this QUERY (D positions before QUERY)
                profile_start = query_pos_actual - self.num_demonstrators
                profiles_q = h[:, profile_start:query_pos_actual, :]  # [B, D, d_model]
                profiles_for_exp = profiles_q  # Update for experience head (use last)
            else:
                # Use original QUERY positions
                query_pos_q = query_positions[:, q]  # [B]
                batch_indices = torch.arange(B, device=h.device)
                query_emb = h[batch_indices, query_pos_q]  # [B, d_model]
                profiles_q = profiles_global if self.profile_method != "no_profile" else None
                if profiles_q is not None:
                    profiles_for_exp = profiles_q
            
            query_embs_all.append(query_emb)
            
            # Bandit payoffs
            alpha_theta, beta_theta = self.bandit_payoffs_head(query_emb, profiles_q)
            alpha_theta_all.append(alpha_theta)
            beta_theta_all.append(beta_theta)
            
            # Demonstrator beliefs
            alpha_demo, beta_demo = self.demo_beliefs_head(query_emb, profiles_q)
            alpha_demo_all.append(alpha_demo)
            beta_demo_all.append(beta_demo)
        
        # Stack predictions
        out = {
            "alpha_theta": torch.stack(alpha_theta_all, dim=1),     # [B, num_queries, K]
            "beta_theta": torch.stack(beta_theta_all, dim=1),       # [B, num_queries, K]
            "alpha_demo": torch.stack(alpha_demo_all, dim=1),       # [B, num_queries, D, K]
            "beta_demo": torch.stack(beta_demo_all, dim=1),         # [B, num_queries, D, K]
        }
        
        # Experience head
        # Use last QUERY's context (most informed)
        query_emb_for_exp = query_embs_all[-1]  # Last QUERY embedding
        
        # Call experience head with appropriate inputs based on input_mode
        if self.experience_head_input == "query_only":
            # Create dummy profiles for compatibility
            dummy_profiles = torch.zeros(B, self.num_demonstrators, h.shape[-1], device=h.device)
            out["N_tot_pred"] = self.demo_experience_head(dummy_profiles, query_emb=query_emb_for_exp)
        elif self.experience_head_input == "profiles_only":
            out["N_tot_pred"] = self.demo_experience_head(profiles_for_exp)
        else:  # query_and_profiles
            out["N_tot_pred"] = self.demo_experience_head(profiles_for_exp, query_emb=query_emb_for_exp)
        
        return out


# ---------------------------
# Thompson Sampling helper (inference-time)
# ---------------------------

@torch.no_grad()
def thompson_sample_action(
    alpha_star: torch.Tensor,
    beta_star: torch.Tensor,
    np_rng: "np.random.Generator",
) -> int:
    """
    Draw a TS action from Beta(alpha_star, beta_star) per arm.
    alpha_star, beta_star: [K]. Uses an explicit numpy Generator so callers
    control reproducibility without relying on torch's global RNG state.
    """
    alpha_np = alpha_star.detach().cpu().numpy()
    beta_np = beta_star.detach().cpu().numpy()
    draws = np_rng.beta(alpha_np, beta_np)
    return int(np.argmax(draws))