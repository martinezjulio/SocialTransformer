"""Load a checkpoint and read the network's beliefs out at several segment cutoffs in one batch.

One session's token tensors ([1, T], from ``sessions.encode_events_to_tensors``) are
expanded to one row per cutoff; the mask comes from ``train.create_segment_mask`` so
inference and training share the cutoff semantics. Demonstrator betas are returned
denormalised when the checkpoint trained with normalised beliefs.
"""

from typing import Dict, List, Optional, Tuple

import torch

from data.transforms import BeliefNormalizer
from models import BaseSocialTransformer
from train import create_segment_mask


def load_model(local_path: Optional[str] = None, wandb_artifact: Optional[str] = None,
               cache_dir: Optional[str] = None, device: str = "cpu") -> Tuple[torch.nn.Module, dict, Optional[BeliefNormalizer]]:
    """Model in eval mode on ``device``, the checkpoint dict, and the belief normaliser (None if unused)."""
    model, checkpoint = BaseSocialTransformer.load_checkpoint(local_path=local_path, wandb_artifact=wandb_artifact, cache_dir=cache_dir)
    norm = checkpoint["normalization"]
    normalizer = None
    if norm.get("normalize_beliefs", False):
        max_exp = norm["expert_N_max"] if "expert_N_max" in norm else norm["N_max"]
        normalizer = BeliefNormalizer(max_experience=max_exp, alpha0=norm["alpha0"], beta0=norm["beta0"])
    return model.to(device).eval(), checkpoint, normalizer


@torch.no_grad()
def readout(model: torch.nn.Module, tensors: Dict[str, torch.Tensor], segment_boundaries: List[int], cutoffs: List[int],
            masking_strategy: str = "all_bandits", segment_round: Optional[List[int]] = None,
            round_bandit_map: Optional[List[int]] = None, normalizer: Optional[BeliefNormalizer] = None) -> Dict[str, torch.Tensor]:
    """Model outputs with a leading cutoff axis: row c sees the first ``cutoffs[c]`` segments.

    Keys follow the model: ``alpha_demo``/``beta_demo`` [C, rounds, D, K] and either
    ``theta_logits`` [C, rounds, K] or ``alpha_theta``/``beta_theta``.
    """
    device = next(model.parameters()).device
    C, T = len(cutoffs), tensors["bandit_ids"].shape[1]
    batch = {k: v.to(device).expand(C, T) for k, v in tensors.items() if k != "pad_mask"}
    batch["pad_mask"] = torch.zeros(C, T, dtype=torch.bool, device=device)
    bounds = torch.tensor([segment_boundaries] * C, device=device)
    rbm = torch.tensor([round_bandit_map] * C, device=device) if round_bandit_map is not None else None
    rounds = [segment_round] * C if segment_round is not None else None
    mask = create_segment_mask(bounds, list(cutoffs), T, device, masking_strategy, rounds, rbm)
    nhead = model.backbone.encoder.layers[0].self_attn.num_heads
    out = model(batch, custom_attn_mask=mask.repeat_interleave(nhead, dim=0))
    if normalizer is not None:
        out["alpha_demo"] = normalizer.denormalize_alpha(out["alpha_demo"])
        out["beta_demo"] = normalizer.denormalize_beta(out["beta_demo"])
    return out
