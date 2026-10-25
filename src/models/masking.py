"""Segment-cutoff attention mask shared by training (`train.py`) and inference (`inference.readout`).

Kept out of `train.py` so inference does not import wandb and the data loaders.
"""

import torch


def create_segment_mask(
    segment_boundaries: torch.Tensor,
    cutoffs: list,
    seq_len: int,
    device: torch.device,
    masking_strategy: str = "all_bandits",
    segment_round: list = None,
    round_bandit_map: torch.Tensor = None
) -> torch.Tensor:
    """
    Create attention mask that makes only visible segments accessible.

    Args:
        segment_boundaries: [B, max_segments] - position of each segment start
        cutoffs: List[int] - how many segments visible per example (1-indexed)
        seq_len: int - total sequence length
        device: torch device
        masking_strategy: "all_bandits" or "current_bandit_only"
        segment_round: List[List[int]] - per-example list of round indices,
            one entry per segment (replaces the old segment_info list-of-dicts;
            only the "round" field was ever read here)
        round_bandit_map: [B, num_rounds] - which bandit for each round

    Returns:
        mask: [B, seq_len, seq_len] - per-example causal mask with segment cutoffs.
            True = masked (not attended to), False = visible. Callers pass this to
            attention as [B*nhead, T, T] (see train_one_epoch).
    """
    B = len(cutoffs)
    # Start with full causal mask for each position
    # mask[i, j] = True means position i cannot attend to position j
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    
    # For batched masking, we need per-example masks
    batch_mask = mask.unsqueeze(0).expand(B, seq_len, seq_len).clone()
    
    for b in range(B):
        # Get cutoff position: everything after this position should be masked
        visible_segs = cutoffs[b]
        if visible_segs < len(segment_boundaries[b]):
            cutoff_pos = int(segment_boundaries[b, visible_segs].item())
        else:
            cutoff_pos = seq_len  # All visible
        
        # Mask out everything from cutoff_pos onward (including the cutoff position itself)
        if cutoff_pos < seq_len:
            batch_mask[b, :, cutoff_pos:] = True  # Can't attend to future
        
        # Additional masking for "current_bandit_only" strategy
        if masking_strategy == "current_bandit_only" and segment_round is not None and round_bandit_map is not None:
            # Determine the target bandit from the cutoff segment
            target_round = segment_round[b][visible_segs - 1]
            target_bandit = int(round_bandit_map[b, target_round].item())

            # Mask out all segments that belong to different bandits
            for seg_idx in range(visible_segs):
                round_i = segment_round[b][seg_idx]
                bandit_i = int(round_bandit_map[b, round_i].item())
                
                if bandit_i != target_bandit:
                    # Mask this entire segment
                    start_pos = int(segment_boundaries[b, seg_idx].item())
                    if seg_idx + 1 < len(segment_boundaries[b]):
                        end_pos = int(segment_boundaries[b, seg_idx + 1].item())
                    else:
                        end_pos = cutoff_pos
                    
                    # Mask all positions from attending to this segment
                    batch_mask[b, :, start_pos:end_pos] = True
    
    # Never leave a row with no visible key: a softmax over an all -inf row is NaN,
    # and under "current_bandit_only" every position in an earlier round loses all of
    # its keys once the cutoff passes that round. The NaN is silent in train mode but
    # poisons the whole sequence in eval mode (fused attention path).
    batch_mask.diagonal(dim1=-2, dim2=-1).fill_(False)

    return batch_mask  # [B, T, T] — caller expands to [B*nhead, T, T]
