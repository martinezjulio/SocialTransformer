"""
Evaluation script for model performance on synthetic test data.

This script:
1. Loads test data from demonstrations_data_02/test
2. Evaluates model at multiple segment cutoffs (varying context lengths)
3. Measures theta (payoff) prediction accuracy
4. Measures demonstrator belief prediction accuracy (expert and novice separately)
5. Outputs comprehensive metrics as CSV
"""

import argparse
import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import load_from_disk

from models import BaseSocialTransformer
from data.transforms import BeliefNormalizer
from data.dataloader import SEGMENT_TYPE_TO_ID, ID_TO_SEGMENT_TYPE
from utils import get_special_characters, setup_eval_directory, get_training_runs_dir, write_run_yaml, write_config_yaml, finalize_run_yaml

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'


SPECIAL = get_special_characters()


def create_segment_mask_single(
    segment_boundaries: List[int],
    cutoff: int,
    seq_len: int,
    device: torch.device,
    masking_strategy: str = "all_bandits",
    segment_round: List[int] = None,
    round_bandit_map: List[int] = None,
) -> torch.Tensor:
    """
    Create attention mask for a single example with a segment cutoff.

    Args:
        segment_boundaries: list of segment start positions
        cutoff:             number of segments to make visible (1-indexed)
        seq_len:            total sequence length
        device:             torch device
        masking_strategy:   "all_bandits" or "current_bandit_only"
        segment_round:      list of round indices, one per segment (replaces
                            the old segment_info list-of-dicts; only the
                            "round" field is needed here)
        round_bandit_map:   list mapping round index → bandit id

    Returns:
        mask: [seq_len, seq_len] — True = masked, False = visible.
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    cutoff_pos = segment_boundaries[cutoff] if cutoff < len(segment_boundaries) else seq_len
    if cutoff_pos < seq_len:
        mask[:, cutoff_pos:] = True

    if masking_strategy == "current_bandit_only" and segment_round is not None and round_bandit_map is not None:
        target_round = int(segment_round[cutoff - 1])
        target_bandit = int(round_bandit_map[target_round])

        for seg_idx in range(cutoff):
            round_i = int(segment_round[seg_idx])
            bandit_i = int(round_bandit_map[round_i])
            if bandit_i != target_bandit:
                start_pos = segment_boundaries[seg_idx]
                if seg_idx + 1 < len(segment_boundaries):
                    end_pos = segment_boundaries[seg_idx + 1]
                else:
                    end_pos = cutoff_pos
                mask[:, start_pos:end_pos] = True

    # Never leave a row with no visible key — softmax over an all -inf row is NaN,
    # which silently poisons every position downstream in eval mode.
    mask.fill_diagonal_(False)

    return mask


def prepare_single_example_batch(example: Dict) -> Dict[str, torch.Tensor]:
    """Wrap a pre-encoded Arrow row as a batch of one for model inference.

    The row already carries the 5 token channels in their final encoded form
    (build_arrow_row did the work at dataset-generation time), so this is just
    add-a-batch-dimension + emit a trivial pad_mask.
    """
    batch: Dict[str, torch.Tensor] = {}
    for k in ("bandit_ids", "agent_ids", "action_ids", "reward_ids", "special_ids"):
        batch[k] = torch.as_tensor(example[k], dtype=torch.long).unsqueeze(0)  # [1, T]
    batch["pad_mask"] = torch.zeros(1, batch["bandit_ids"].shape[1], dtype=torch.bool)
    return batch


def _ensure_tensor(x, dtype=torch.float32) -> torch.Tensor:
    """HF returns torch tensors for set_format columns and Python lists/numpy for
    the rest; this normalizes either input to a tensor of the requested dtype."""
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=dtype)


def _derive_N_tot(alpha: torch.Tensor, beta: torch.Tensor, alpha0: float, beta0: float, K: int) -> float:
    """Recover demonstrator pre-episode experience count from the stored
    (alpha, beta) shapes. Mirrors the formula already used elsewhere in this
    file (line 345-349 of the pre-Arrow version) so we don't need to carry
    N_tot as a separate column in the Arrow schema."""
    return float((alpha.sum() + beta.sum() - alpha0 * K - beta0 * K).item())


def evaluate_single_example(
    example: Dict,
    model: nn.Module,
    device: torch.device,
    num_arms: int,
    num_agents: int,
    num_bandits: int,
    SPECIAL: Dict,
    normalizer=None,
    alpha0: float = 1.0,
    beta0: float = 1.0,
    masking_strategy: str = "all_bandits"
) -> List[Dict]:
    """Evaluate model on a single Arrow row at multiple segment cutoffs.

    Each row arrives pre-encoded (bandit_ids/agent_ids/action_ids/reward_ids/
    special_ids are already the tokenized streams). Labels arrive as
    fixed-shape tensors (y_thetas, y_alpha_demo, y_beta_demo). Segment metadata
    arrives as three parallel lists (segment_round/segment_type_id/segment_agent)
    instead of the old segment_info list-of-dicts.
    """
    # Pull labels straight from the Arrow row — no `labels["bandits"]` lookup.
    y_thetas     = _ensure_tensor(example["y_thetas"])      # [num_bandits, K]
    y_alpha_demo = _ensure_tensor(example["y_alpha_demo"])  # [num_bandits, D, K]
    y_beta_demo  = _ensure_tensor(example["y_beta_demo"])   # [num_bandits, D, K]

    segment_round    = list(example["segment_round"])
    segment_type_id  = list(example["segment_type_id"])
    segment_agent    = list(example["segment_agent"])
    segment_boundaries = list(example["segment_boundaries"])
    round_bandit_map   = list(example["round_bandit_map"])

    # Encoded token streams (used both for the model forward and for
    # reconstructing per-segment reward/payoff sums).
    bandit_ids_list = list(example["bandit_ids"])
    action_ids_list = list(example["action_ids"])
    reward_ids_list = list(example["reward_ids"])
    special_ids_list = list(example["special_ids"])

    num_segments = len(segment_round)
    K = num_arms

    # Determine game_sequence (expert-first vs novice-first) from the first
    # demo segment, exactly as the pre-Arrow code did — segment_agent maps
    # 1-indexed agent ids back to 0-indexed demo ids.
    first_demo_agent = None
    for i in range(num_segments):
        if segment_type_id[i] == SEGMENT_TYPE_TO_ID["demo"]:
            first_demo_agent = int(segment_agent[i]) - 1
            break

    # Identify expert by N_tot from the first bandit (consistent across bandits).
    demo0_N_all = _derive_N_tot(y_alpha_demo[0, 0], y_beta_demo[0, 0], alpha0, beta0, K)
    demo1_N_all = _derive_N_tot(y_alpha_demo[0, 1], y_beta_demo[0, 1], alpha0, beta0, K)
    example_expert_id = 0 if demo0_N_all > demo1_N_all else 1

    game_sequence = (
        "expert-play-novice-play-choice"
        if first_demo_agent == example_expert_id
        else "novice-play-expert-play-choice"
    )

    # Determine segment_order: watch-first (demo before learner) vs play-first.
    first_non_structural = None
    demo_id_v   = SEGMENT_TYPE_TO_ID["demo"]
    learner_id_v = SEGMENT_TYPE_TO_ID["learner"]
    for i in range(num_segments):
        if segment_type_id[i] in (demo_id_v, learner_id_v):
            first_non_structural = segment_type_id[i]
            break
    segment_order = "watch-first" if first_non_structural == demo_id_v else "play-first"

    results = []

    batch = prepare_single_example_batch(example)
    for k in batch:
        batch[k] = batch[k].to(device)
    seq_len = batch["bandit_ids"].shape[1]

    # Evaluate at each possible cutoff
    for cutoff in range(1, num_segments + 1):
        seg_type_id = int(segment_type_id[cutoff - 1])
        seg_type_str = ID_TO_SEGMENT_TYPE[seg_type_id]
        seg_agent = int(segment_agent[cutoff - 1])
        round_idx = int(segment_round[cutoff - 1])
        bandit_id = int(round_bandit_map[round_idx])

        # Per-segment rewards (demo / learner only; bandit_sep and query are
        # structural-only and have no reward):
        #   * segment_total_reward    — observed 0/1 sum over real-event trials
        #   * segment_expected_reward — sum of TRUE payoffs of the chosen arms,
        #     which strips outcome noise across participants picking the same arm
        # We derive these from the encoded token streams (real events are at
        # positions where special_ids == SPECIAL["NONE"]), avoiding the need to
        # carry raw `events` tuples in the Arrow schema.
        if seg_type_id in (demo_id_v, learner_id_v):
            start_pos = segment_boundaries[cutoff - 1]
            end_pos   = segment_boundaries[cutoff] if cutoff < len(segment_boundaries) else seq_len
            tot_r = 0.0
            exp_r = 0.0
            n_real = 0
            for t in range(start_pos, end_pos):
                if special_ids_list[t] == SPECIAL["NONE"]:
                    b_t = int(bandit_ids_list[t])
                    a_t = int(action_ids_list[t])
                    r_t = int(reward_ids_list[t])
                    tot_r += float(r_t)
                    exp_r += float(y_thetas[b_t, a_t].item())
                    n_real += 1
            if n_real > 0:
                segment_total_reward = tot_r
                segment_expected_reward = exp_r
            else:
                segment_total_reward = float("nan")
                segment_expected_reward = float("nan")
        else:
            segment_total_reward = float("nan")
            segment_expected_reward = float("nan")

        # Attention mask for this cutoff
        attn_mask = create_segment_mask_single(
            segment_boundaries,
            cutoff,
            seq_len,
            device,
            masking_strategy=masking_strategy,
            segment_round=segment_round,
            round_bandit_map=round_bandit_map,
        )

        # Run model
        model.eval()
        with torch.no_grad():
            outputs = model(batch, custom_attn_mask=attn_mask)

        # Ground truth for this bandit
        true_theta = y_thetas[bandit_id]

        # Identify expert/novice by N_tot for THIS bandit.
        demo0_N = _derive_N_tot(y_alpha_demo[bandit_id, 0], y_beta_demo[bandit_id, 0], alpha0, beta0, K)
        demo1_N = _derive_N_tot(y_alpha_demo[bandit_id, 1], y_beta_demo[bandit_id, 1], alpha0, beta0, K)
        if demo0_N > demo1_N:
            expert_id, novice_id = 0, 1
        else:
            expert_id, novice_id = 1, 0

        true_alpha_expert = y_alpha_demo[bandit_id, expert_id]
        true_beta_expert  = y_beta_demo[bandit_id, expert_id]
        true_alpha_novice = y_alpha_demo[bandit_id, novice_id]
        true_beta_novice  = y_beta_demo[bandit_id, novice_id]
        
        # Get demo predictions for this bandit
        # outputs are [B, num_queries, K] or [B, num_queries, D, K]
        pred_alpha_demo = outputs["alpha_demo"][0, bandit_id].cpu()    # [num_demos, K]
        pred_beta_demo = outputs["beta_demo"][0, bandit_id].cpu()      # [num_demos, K]
        
        # Denormalize if needed
        if normalizer:
            pred_alpha_demo = normalizer.denormalize_alpha(pred_alpha_demo)
            pred_beta_demo = normalizer.denormalize_beta(pred_beta_demo)
        
        # Compute expected payoffs (branch on environment theta head type)
        if model.env_theta_loss == "bce":
            pred_theta_expected = torch.sigmoid(outputs["theta_logits"][0, bandit_id].cpu())  # [K]
        else:
            pred_alpha_theta = outputs["alpha_theta"][0, bandit_id].cpu()  # [K]
            pred_beta_theta = outputs["beta_theta"][0, bandit_id].cpu()    # [K]
            pred_theta_expected = pred_alpha_theta / (pred_alpha_theta + pred_beta_theta)
        
        # Compute theta metrics
        theta_corr = torch.corrcoef(
            torch.stack([pred_theta_expected, true_theta])
        )[0, 1].item()
        
        # Match predictions to ground truth based on which demo_id is expert/novice
        pred_alpha_expert = pred_alpha_demo[expert_id]
        pred_beta_expert = pred_beta_demo[expert_id]
        pred_alpha_novice = pred_alpha_demo[novice_id]
        pred_beta_novice = pred_beta_demo[novice_id]
        
        # Compute expert metrics
        expert_alpha_corr = torch.corrcoef(
            torch.stack([pred_alpha_expert, true_alpha_expert])
        )[0, 1].item()
        expert_beta_corr = torch.corrcoef(
            torch.stack([pred_beta_expert, true_beta_expert])
        )[0, 1].item()
        
        # Compute novice metrics
        novice_alpha_corr = torch.corrcoef(
            torch.stack([pred_alpha_novice, true_alpha_novice])
        )[0, 1].item()
        novice_beta_corr = torch.corrcoef(
            torch.stack([pred_beta_novice, true_beta_novice])
        )[0, 1].item()
        
        # Compute N_tot errors
        K = num_arms
        pred_expert_N = (pred_alpha_expert.sum() + pred_beta_expert.sum() - alpha0*K - beta0*K).item()
        true_expert_N = (true_alpha_expert.sum() + true_beta_expert.sum() - alpha0*K - beta0*K).item()
        
        pred_novice_N = (pred_alpha_novice.sum() + pred_beta_novice.sum() - alpha0*K - beta0*K).item()
        true_novice_N = (true_alpha_novice.sum() + true_beta_novice.sum() - alpha0*K - beta0*K).item()
        
        # Compile metrics
        metrics = {
            "cutoff": cutoff,
            "game_sequence": game_sequence,
            "segment_order": segment_order,
            "segment_type": seg_type_str,
            "agent": seg_agent,
            "round_index": round_idx,
            "bandit_id": int(bandit_id),
            "expert_demo_id": expert_id,
            "novice_demo_id": novice_id,
            
            # Theta metrics
            "theta_correlation": theta_corr,
            "theta_variance": pred_theta_expected.var().item(),
            "theta_mse": F.mse_loss(pred_theta_expected, true_theta).item(),
            "theta_range": (pred_theta_expected.max() - pred_theta_expected.min()).item(),
            
            # Expert metrics
            "expert_alpha_corr": expert_alpha_corr,
            "expert_beta_corr": expert_beta_corr,
            "expert_alpha_mse": F.mse_loss(pred_alpha_expert, true_alpha_expert).item(),
            "expert_beta_mse": F.mse_loss(pred_beta_expert, true_beta_expert).item(),
            "expert_N_tot_error": abs(pred_expert_N - true_expert_N),
            "expert_N_tot_true": true_expert_N,
            "expert_N_tot_pred": pred_expert_N,
            "expert_N_tot_raw": demo0_N if expert_id == 0 else demo1_N,

            # Total reward summed over the trials of the game ending at this cutoff
            "segment_total_reward": segment_total_reward,
            "segment_expected_reward": segment_expected_reward,
            
            # Novice metrics
            "novice_alpha_corr": novice_alpha_corr,
            "novice_beta_corr": novice_beta_corr,
            "novice_alpha_mse": F.mse_loss(pred_alpha_novice, true_alpha_novice).item(),
            "novice_beta_mse": F.mse_loss(pred_beta_novice, true_beta_novice).item(),
            "novice_N_tot_error": abs(pred_novice_N - true_novice_N),
            "novice_N_tot_true": true_novice_N,
            "novice_N_tot_pred": pred_novice_N,
            "novice_N_tot_raw": demo0_N if novice_id == 0 else demo1_N,
        }
        
        # Add experience head metrics if present
        if 'N_tot_pred' in outputs:
            N_tot_pred = outputs['N_tot_pred'][0].cpu()  # [num_demonstrators]
            
            # Get expert and novice predictions
            pred_expert_N_head = N_tot_pred[expert_id].item()
            pred_novice_N_head = N_tot_pred[novice_id].item()
            
            # For BCE mode, convert logits to probabilities
            # We can detect BCE vs MSE by checking if values are in reasonable N_tot range
            # If values are very negative (logits), convert to probabilities
            if pred_expert_N_head < 0 or pred_novice_N_head < -5:  # Likely logits
                pred_expert_prob = torch.sigmoid(N_tot_pred[expert_id]).item()
                pred_novice_prob = torch.sigmoid(N_tot_pred[novice_id]).item()
                metrics["experience_head_expert_prob"] = pred_expert_prob
                metrics["experience_head_novice_prob"] = pred_novice_prob
                # Binary predictions (threshold 0.5 on probability)
                metrics["experience_head_expert_pred_binary"] = int(pred_expert_prob > 0.5)
                metrics["experience_head_novice_pred_binary"] = int(pred_novice_prob > 0.5)
                # Accuracy (expert should be 1, novice should be 0)
                metrics["experience_head_expert_correct"] = int(pred_expert_prob > 0.5)
                metrics["experience_head_novice_correct"] = int(pred_novice_prob <= 0.5)
            else:
                # MSE mode - continuous N_tot
                metrics["experience_head_expert_N_pred"] = pred_expert_N_head
                metrics["experience_head_novice_N_pred"] = pred_novice_N_head
                metrics["experience_head_expert_error"] = abs(pred_expert_N_head - true_expert_N)
                metrics["experience_head_novice_error"] = abs(pred_novice_N_head - true_novice_N)
        
        results.append(metrics)
    
    return results


def build_trial_rows_for_example(example: Dict, ex_idx: int, alpha0: float = 1.0, beta0: float = 1.0) -> List[Dict]:
    """
    Emit one row per simulator trial for a single synthetic example, matching
    the schema of experiment_trial_predictions.csv (from run_inference.py).
    Synthetic trajectories come from the data generator rather than from a
    trained model running Thompson sampling, so prior-source distinctions
    don't apply: every row is tagged with learner_segment_input='model' and
    beta_method='simulator', and the per-trial posterior columns are empty.

    Operates on the new Arrow row schema: flattened segment_* columns,
    fixed-shape y_thetas / y_alpha_demo / y_beta_demo, and the encoded token
    streams in place of raw `events` tuples.
    """
    segment_round    = list(example["segment_round"])
    segment_type_id  = list(example["segment_type_id"])
    segment_agent    = list(example["segment_agent"])
    boundaries       = list(example["segment_boundaries"])
    round_bandit_map = list(example["round_bandit_map"])

    bandit_ids_list  = list(example["bandit_ids"])
    action_ids_list  = list(example["action_ids"])
    reward_ids_list  = list(example["reward_ids"])
    special_ids_list = list(example["special_ids"])

    y_thetas     = _ensure_tensor(example["y_thetas"])
    y_alpha_demo = _ensure_tensor(example["y_alpha_demo"])
    y_beta_demo  = _ensure_tensor(example["y_beta_demo"])

    demo_id_v   = SEGMENT_TYPE_TO_ID["demo"]
    learner_id_v = SEGMENT_TYPE_TO_ID["learner"]
    K = y_thetas.shape[1]

    # game_sequence + expert/novice mapping
    first_demo_agent = None
    for i, t in enumerate(segment_type_id):
        if t == demo_id_v:
            first_demo_agent = int(segment_agent[i]) - 1
            break
    demo0_N = _derive_N_tot(y_alpha_demo[0, 0], y_beta_demo[0, 0], alpha0, beta0, K)
    demo1_N = _derive_N_tot(y_alpha_demo[0, 1], y_beta_demo[0, 1], alpha0, beta0, K)
    expert_demo_id = 0 if demo0_N > demo1_N else 1
    game_sequence = (
        "expert-play-novice-play-choice"
        if first_demo_agent == expert_demo_id
        else "novice-play-expert-play-choice"
    )

    session_id = f"synth_{ex_idx}"
    session_index = ex_idx
    anonymous_id = session_id

    # Within each round, demo/learner segments alternate watch/play. Track the
    # in-round index so we can assign a contiguous game_index = 4*round + idx.
    round_game_counter: Dict[int, int] = {}
    rows: List[Dict] = []
    n_segments = len(segment_type_id)
    seq_len = len(bandit_ids_list)
    for seg_idx in range(n_segments):
        seg_type_id = int(segment_type_id[seg_idx])
        if seg_type_id not in (demo_id_v, learner_id_v):
            continue

        round_idx = int(segment_round[seg_idx])
        in_round = round_game_counter.get(round_idx, 0)
        game_index = 4 * round_idx + in_round
        round_game_counter[round_idx] = in_round + 1

        start = boundaries[seg_idx]
        end   = boundaries[seg_idx + 1] if (seg_idx + 1) < len(boundaries) else seq_len

        # Filter the encoded slice down to real events (special_ids == NONE).
        seg_real = [
            (int(bandit_ids_list[t]), int(action_ids_list[t]), int(reward_ids_list[t]))
            for t in range(start, end)
            if special_ids_list[t] == SPECIAL["NONE"]
        ]
        if not seg_real:
            continue

        if seg_type_id == demo_id_v:
            game = "watch"
            demo_id = int(segment_agent[seg_idx]) - 1
            demonstrator_type = "expert" if demo_id == expert_demo_id else "novice"
        else:
            game = "play"
            demonstrator_type = float("nan")

        bandit_id = seg_real[0][0]
        zone_bandit = ",".join(f"{p:.6g}" for p in y_thetas[bandit_id].tolist())

        for (_b, a, r) in seg_real:
            rows.append({
                "session_id": session_id,
                "session_index": session_index,
                "anonymous_id": anonymous_id,
                "position": a,
                "action": "shoot",
                "game": game,
                "game_sequence": game_sequence,
                "game_index": game_index,
                "round_index": round_idx,
                "demonstrator_type": demonstrator_type,
                "zone_bandit": zone_bandit,
                "zone_global_index": int(round_bandit_map[round_idx]),
                "reward_generated": bool(r),
                "rollout_id": 0,
                "learner_segment_input": "model",
                "beta_method": "simulator",
                "theta_betas": float("nan"),
                "expert_betas": float("nan"),
                "novice_betas": float("nan"),
            })
    return rows


def plot_evaluation_results(df: pd.DataFrame, output_dir: str = "figures/evaluation"):
    """Generate comprehensive visualization of evaluation results."""
    
    print(f"\n{'='*60}")
    print("Generating Visualization")
    print(f"{'='*60}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Compute ground truth statistics
    true_payoffs = [0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.75, 0.8]
    true_variance = pd.Series(true_payoffs).var()
    true_range = max(true_payoffs) - min(true_payoffs)
    
    print(f"\nGround truth statistics:")
    print(f"  True variance: {true_variance:.6f}")
    print(f"  True range: {true_range:.2f}")
    
    # Aggregate by cutoff
    cutoff_stats = df.groupby('cutoff').agg({
        'theta_correlation': ['mean', 'std', 'count'],
        'theta_variance': ['mean', 'std'],
        'theta_mse': ['mean', 'std'],
        'theta_range': ['mean', 'std'],
        'expert_alpha_corr': ['mean', 'std'],
        'expert_beta_corr': ['mean', 'std'],
        'expert_N_tot_error': ['mean', 'std'],
        'expert_N_tot_raw': ['mean'],
        'expert_N_tot_pred': ['mean', 'std'],
        'novice_alpha_corr': ['mean', 'std'],
        'novice_beta_corr': ['mean', 'std'],
        'novice_N_tot_error': ['mean', 'std'],
        'novice_N_tot_raw': ['mean'],
        'novice_N_tot_pred': ['mean', 'std'],
    }).reset_index()
    
    # Flatten column names
    cutoff_stats.columns = ['_'.join(col).strip('_') for col in cutoff_stats.columns.values]
    
    # Compute standard errors
    n = cutoff_stats['theta_correlation_count']
    cutoff_stats['theta_correlation_se'] = cutoff_stats['theta_correlation_std'] / (n ** 0.5)
    cutoff_stats['theta_variance_se'] = cutoff_stats['theta_variance_std'] / (n ** 0.5)
    cutoff_stats['theta_mse_se'] = cutoff_stats['theta_mse_std'] / (n ** 0.5)
    cutoff_stats['theta_range_se'] = cutoff_stats['theta_range_std'] / (n ** 0.5)
    cutoff_stats['expert_alpha_corr_se'] = cutoff_stats['expert_alpha_corr_std'] / (n ** 0.5)
    cutoff_stats['expert_beta_corr_se'] = cutoff_stats['expert_beta_corr_std'] / (n ** 0.5)
    cutoff_stats['expert_N_tot_error_se'] = cutoff_stats['expert_N_tot_error_std'] / (n ** 0.5)
    cutoff_stats['expert_N_tot_pred_se'] = cutoff_stats['expert_N_tot_pred_std'] / (n ** 0.5)
    cutoff_stats['novice_alpha_corr_se'] = cutoff_stats['novice_alpha_corr_std'] / (n ** 0.5)
    cutoff_stats['novice_beta_corr_se'] = cutoff_stats['novice_beta_corr_std'] / (n ** 0.5)
    cutoff_stats['novice_N_tot_error_se'] = cutoff_stats['novice_N_tot_error_std'] / (n ** 0.5)
    cutoff_stats['novice_N_tot_pred_se'] = cutoff_stats['novice_N_tot_pred_std'] / (n ** 0.5)
    
    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    fig.suptitle('Model Evaluation on Test Set - All Metrics (error bars = ±1 SE)', fontsize=18, fontweight='bold')
    
    # Plot 1: Theta correlation
    ax = axes[0, 0]
    ax.errorbar(cutoff_stats['cutoff'], cutoff_stats['theta_correlation_mean'],
                yerr=cutoff_stats['theta_correlation_se'], 
                marker='o', capsize=5, capthick=2, linewidth=2)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Random (0)')
    ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Good (0.5)')
    ax.set_xlabel('Segment Cutoff (Context Length)', fontsize=12)
    ax.set_ylabel('Correlation', fontsize=12)
    ax.set_title('Theta Prediction Accuracy', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 2: Theta variance
    ax = axes[0, 1]
    ax.errorbar(cutoff_stats['cutoff'], cutoff_stats['theta_variance_mean'],
                yerr=cutoff_stats['theta_variance_se'],
                marker='o', capsize=5, capthick=2, linewidth=2, color='orange', label='Predicted')
    ax.axhline(y=true_variance, color='green', linestyle='--', linewidth=2, alpha=0.7, 
               label=f'True variance ({true_variance:.4f})')
    ax.set_xlabel('Segment Cutoff (Context Length)', fontsize=12)
    ax.set_ylabel('Variance of Expected Payoffs', fontsize=12)
    ax.set_title('Theta Prediction Informativeness', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 3: Theta range
    ax = axes[0, 2]
    ax.errorbar(cutoff_stats['cutoff'], cutoff_stats['theta_range_mean'],
                yerr=cutoff_stats['theta_range_se'],
                marker='o', capsize=5, capthick=2, linewidth=2, color='green', label='Predicted')
    ax.axhline(y=true_range, color='purple', linestyle='--', linewidth=2, alpha=0.7,
               label=f'True range ({true_range:.2f})')
    ax.set_xlabel('Segment Cutoff (Context Length)', fontsize=12)
    ax.set_ylabel('Range (max - min)', fontsize=12)
    ax.set_title('Theta Prediction Range', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 4: Alpha correlation
    ax = axes[1, 0]
    ax.errorbar(cutoff_stats['cutoff'], cutoff_stats['expert_alpha_corr_mean'],
                yerr=cutoff_stats['expert_alpha_corr_se'],
                marker='o', capsize=5, capthick=2, linewidth=2, color='blue', label='Expert', alpha=0.8)
    ax.errorbar(cutoff_stats['cutoff'], cutoff_stats['novice_alpha_corr_mean'],
                yerr=cutoff_stats['novice_alpha_corr_se'],
                marker='s', capsize=5, capthick=2, linewidth=2, color='red', label='Novice', alpha=0.8)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Segment Cutoff (Context Length)', fontsize=12)
    ax.set_ylabel('Correlation', fontsize=12)
    ax.set_title('Alpha Prediction Accuracy', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 5: Beta correlation
    ax = axes[1, 1]
    ax.errorbar(cutoff_stats['cutoff'], cutoff_stats['expert_beta_corr_mean'],
                yerr=cutoff_stats['expert_beta_corr_se'],
                marker='o', capsize=5, capthick=2, linewidth=2, color='blue', label='Expert', alpha=0.8)
    ax.errorbar(cutoff_stats['cutoff'], cutoff_stats['novice_beta_corr_mean'],
                yerr=cutoff_stats['novice_beta_corr_se'],
                marker='s', capsize=5, capthick=2, linewidth=2, color='red', label='Novice', alpha=0.8)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Segment Cutoff (Context Length)', fontsize=12)
    ax.set_ylabel('Correlation', fontsize=12)
    ax.set_title('Beta Prediction Accuracy', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 6: Theta MSE
    ax = axes[1, 2]
    ax.errorbar(cutoff_stats['cutoff'], cutoff_stats['theta_mse_mean'],
                yerr=cutoff_stats['theta_mse_se'],
                marker='o', capsize=5, capthick=2, linewidth=2, color='purple')
    ax.set_xlabel('Segment Cutoff (Context Length)', fontsize=12)
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_title('Theta Prediction MSE', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 7: Combined N_tot predictions
    ax = axes[2, 0]
    true_expert_N_tot = df['expert_N_tot_true'].mean()
    true_novice_N_tot = df['novice_N_tot_true'].mean()
    ax.axhline(y=true_expert_N_tot, color='lightblue', linestyle='--', linewidth=2.5, alpha=0.7,
               label=f'True Expert N_tot ({true_expert_N_tot:.0f})', zorder=1)
    ax.errorbar(cutoff_stats['cutoff'], cutoff_stats['expert_N_tot_pred_mean'],
                yerr=cutoff_stats['expert_N_tot_pred_se'],
                marker='o', linewidth=2, color='blue', label='Predicted Expert N_tot', 
                markersize=6, capsize=4, alpha=0.8, zorder=3)
    ax.axhline(y=true_novice_N_tot, color='lightcoral', linestyle='--', linewidth=2.5, alpha=0.7,
               label=f'True Novice N_tot ({true_novice_N_tot:.0f})', zorder=1)
    ax.errorbar(cutoff_stats['cutoff'], cutoff_stats['novice_N_tot_pred_mean'],
                yerr=cutoff_stats['novice_N_tot_pred_se'],
                marker='s', linewidth=2, color='red', label='Predicted Novice N_tot',
                markersize=6, capsize=4, alpha=0.8, zorder=3)
    ax.set_xlabel('Segment Cutoff (Context Length)', fontsize=12)
    ax.set_ylabel('N_tot (Total Experience)', fontsize=12)
    ax.set_title('N_tot Predictions vs Ground Truth', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_ylim(-5, true_expert_N_tot * 1.15)
    
    # Plot 8: N_tot absolute errors
    ax = axes[2, 1]
    ax.errorbar(cutoff_stats['cutoff'], cutoff_stats['expert_N_tot_error_mean'],
                yerr=cutoff_stats['expert_N_tot_error_se'],
                marker='o', capsize=5, capthick=2, linewidth=2, color='blue', label='Expert', alpha=0.8)
    ax.errorbar(cutoff_stats['cutoff'], cutoff_stats['novice_N_tot_error_mean'],
                yerr=cutoff_stats['novice_N_tot_error_se'],
                marker='s', capsize=5, capthick=2, linewidth=2, color='red', label='Novice', alpha=0.8)
    ax.set_xlabel('Segment Cutoff (Context Length)', fontsize=12)
    ax.set_ylabel('Absolute Error in N_tot', fontsize=12)
    ax.set_title('N_tot Prediction Error', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 9: N_tot relative error
    ax = axes[2, 2]
    expert_rel_error = (cutoff_stats['expert_N_tot_error_mean'] / true_expert_N_tot) * 100
    expert_rel_error_se = (cutoff_stats['expert_N_tot_error_se'] / true_expert_N_tot) * 100
    ax.errorbar(cutoff_stats['cutoff'], expert_rel_error,
                yerr=expert_rel_error_se,
                marker='o', linewidth=2, color='blue', label=f'Expert (% of {true_expert_N_tot:.0f})',
                markersize=6, capsize=4, alpha=0.8)
    ax2 = ax.twinx()
    ax2.errorbar(cutoff_stats['cutoff'], cutoff_stats['novice_N_tot_pred_mean'],
                 yerr=cutoff_stats['novice_N_tot_pred_se'],
                 marker='s', linewidth=2, color='red', label='Novice (abs pred)', 
                 markersize=6, capsize=4, alpha=0.8)
    ax.set_xlabel('Segment Cutoff (Context Length)', fontsize=12)
    ax.set_ylabel('Expert: % Error', fontsize=12, color='blue')
    ax2.set_ylabel('Novice: Absolute Prediction', fontsize=12, color='red')
    ax.set_title('N_tot Prediction Quality', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='red')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir) / 'evaluation_summary.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Plot saved to {output_path}")
    
    # Print detailed N_tot summary
    print("\n" + "="*60)
    print("N_tot Prediction Evolution")
    print("="*60)
    print(f"\nExpert N_tot (True = {true_expert_N_tot:.0f}):")
    print(cutoff_stats[['cutoff', 'expert_N_tot_pred_mean', 'expert_N_tot_error_mean']].to_string(index=False))
    
    print("\nNovice N_tot (True = 0):")
    print(cutoff_stats[['cutoff', 'novice_N_tot_pred_mean', 'novice_N_tot_error_mean']].to_string(index=False))
    
    # Print summary by round
    print("\n" + "="*60)
    print("Summary by Round (approximate)")
    print("="*60)
    
    df['round_approx'] = (df['cutoff'] - 1) // 5
    round_summary = df.groupby('round_approx').agg({
        'theta_correlation': 'mean',
        'theta_variance': 'mean',
        'expert_alpha_corr': 'mean',
        'novice_alpha_corr': 'mean',
        'expert_N_tot_error': 'mean',
        'novice_N_tot_error': 'mean',
    })
    print(round_summary)
    
    return fig


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on test set")
    parser.add_argument("--test_data_dir", type=str, required=True,
        help="Path to the dataset root (parent of train/val/test/); the test "
             "split is loaded from <test_data_dir>/test/ as a HF Arrow dataset.")
    parser.add_argument("--output_path", type=str,
        default="data/evaluation/test_cutoff_model_predictions.csv",
        help="Path to save per-cutoff evaluation results")
    parser.add_argument("--trial_output_path", type=str,
        default="data/evaluation/test_trial_model_predictions.csv",
        help="Path to save per-trial trajectories (mirrors experiment_trial_predictions.csv schema)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--max_examples", type=int, default=None,
        help="Max examples to evaluate (None = all)")
    parser.add_argument("--skip_plot", action="store_true",
        help="Skip generating visualization (only save CSV)")
    parser.add_argument("--plot_output_dir", type=str, default="figures/evaluation",
        help="Directory to save plots")
    parser.add_argument("--masking_strategy_override", type=str, default=None,
        choices=["all_bandits", "current_bandit_only"],
        help="Override masking strategy from checkpoint (optional)")

    # Checkpoint loading options (mutually exclusive)
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument("--checkpoint_path", type=str,
        help="Path to local checkpoint file")
    checkpoint_group.add_argument("--wandb_artifact", type=str,
        help="Load from wandb artifact: 'entity/project/artifact:alias'")
    
    args = parser.parse_args()
    
    # Set up eval directory structure
    eval_dir = setup_eval_directory()
    print(f"Eval directory: {eval_dir}")

    # Write run metadata
    metadata = {
        'script': 'eval.py',
        'command': ' '.join(sys.argv),
        'test_data_dir': args.test_data_dir,
    }
    write_run_yaml(eval_dir, metadata, subdir='metadata')
    write_config_yaml(eval_dir, vars(args), subdir='metadata')

    # Set output paths to eval directory
    if args.output_path == "data/evaluation/test_cutoff_model_predictions.csv":
        args.output_path = str(eval_dir / 'data' / 'test_cutoff_model_predictions.csv')
    if args.trial_output_path == "data/evaluation/test_trial_model_predictions.csv":
        args.trial_output_path = str(eval_dir / 'data' / 'test_trial_model_predictions.csv')
    if args.plot_output_dir == "figures/evaluation":
        args.plot_output_dir = str(eval_dir / 'figures')
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    try:
        # Load checkpoint
        print("\n" + "="*60)
        print("Loading Model Checkpoint")
        print("="*60)
    
        model, checkpoint = BaseSocialTransformer.load_checkpoint(
            local_path=args.checkpoint_path,
            wandb_artifact=args.wandb_artifact,
            cache_dir=str(get_training_runs_dir()),
        )

        model = model.to(device)
        model.eval()
        model_config = checkpoint['model_config']

        print(f"\nModel configuration:")
        print(f"  num_arms: {model_config['num_arms']}")
        print(f"  num_agents: {model_config['num_agents']}")
        print(f"  num_bandits: {model_config['num_bandits']}")
        print(f"  num_demonstrators: {model_config['num_demonstrators']}")
        print(f"  d_model: {model_config['d_model']}")
        print(f"  nhead: {model_config['nhead']}")
        print(f"  num_layers: {model_config['num_layers']}")
        print(f"  model_type: {model_config.get('model_type') or model_config.get('profile_method', 'social')}")
        print(f"  experience_head_loss_type: {model_config.get('experience_head_loss_type', 'mse')}")
        print("✓ Model loaded successfully!")
    
        # Get normalization parameters
        normalizer = None
        if 'normalization' in checkpoint and checkpoint['normalization'].get('normalize_beliefs', False):
            norm_dict = checkpoint['normalization']
            if 'expert_N_max' in norm_dict:
                max_exp = norm_dict['expert_N_max']
            elif 'N_max' in norm_dict:
                print("WARNING: checkpoint uses deprecated 'N_max' key in normalization dict.")
                max_exp = norm_dict['N_max']
            else:
                raise ValueError("Checkpoint normalization missing both 'expert_N_max' and 'N_max'")
            normalizer = BeliefNormalizer(
                max_experience=max_exp,
                alpha0=norm_dict['alpha0'],
                beta0=norm_dict['beta0']
            )
            print("✓ Using belief normalization")
    
        alpha0 = checkpoint['normalization']['alpha0']
        beta0 = checkpoint['normalization']['beta0']
        print(f"  alpha0={alpha0}, beta0={beta0}")
    
        # Get masking strategy (from checkpoint or override)
        if args.masking_strategy_override:
            masking_strategy = args.masking_strategy_override
            print(f"\n✓ Using overridden masking strategy: {masking_strategy}")
        else:
            masking_strategy = checkpoint.get('config', {}).get('masking_strategy', 'all_bandits')
            print(f"\n✓ Using masking strategy from checkpoint: {masking_strategy}")
    
        # Load test data (mmap-backed Arrow — random row access is cheap, no
        # full-dataset materialization).
        print("\n" + "="*60)
        print("Loading Test Data")
        print("="*60)

        test_split_dir = os.path.join(args.test_data_dir, "test")
        print(f"Loading Arrow dataset from {test_split_dir}")
        all_examples = load_from_disk(test_split_dir)
        # Standard torch format on the fixed-shape label columns; variable-length
        # sequence columns stay as Python lists.
        all_examples.set_format(
            type="torch",
            columns=["y_thetas", "y_alpha_demo", "y_beta_demo"],
            output_all_columns=True,
        )
        print(f"✓ Loaded {len(all_examples)} examples")

        # Limit examples if requested
        if args.max_examples:
            all_examples = all_examples.select(range(args.max_examples))
            print(f"  Evaluating on {len(all_examples)} examples (limited)")
    
        # Run evaluation
        print("\n" + "="*60)
        print("Running Evaluation")
        print("="*60)
    
        all_results = []
    
        for ex_idx, example in enumerate(tqdm(all_examples, desc="Evaluating examples")):
            try:
                example_results = evaluate_single_example(
                    example,
                    model,
                    device,
                    model_config['num_arms'],
                    model_config['num_agents'],
                    model_config['num_bandits'],
                    SPECIAL,
                    normalizer=normalizer,
                    alpha0=alpha0,
                    beta0=beta0,
                    masking_strategy=masking_strategy
                )
            
                # Add example_id to each result
                for result in example_results:
                    result['example_id'] = ex_idx
            
                all_results.extend(example_results)
            
            except Exception as e:
                print(f"\n  Warning: Failed on example {ex_idx}: {e}")
                continue
    
        # Convert to DataFrame
        print(f"\n✓ Completed evaluation on {len(all_examples)} examples")
        print(f"  Total metrics computed: {len(all_results)}")
    
        df = pd.DataFrame(all_results)
    
        # Create output directory
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
        # Save per-cutoff results
        print(f"\nSaving per-cutoff results to {args.output_path}...")
        df.to_csv(args.output_path, index=False)

        # Emit per-trial CSV for the hybrid pipeline report (synthetic source).
        print(f"\nBuilding per-trial rows from {len(all_examples)} examples...")
        trial_rows: List[Dict] = []
        for ex_idx, example in enumerate(all_examples):
            try:
                trial_rows.extend(build_trial_rows_for_example(example, ex_idx, alpha0=alpha0, beta0=beta0))
            except Exception as e:
                print(f"  Warning: Failed to build trial rows for example {ex_idx}: {e}")
                continue
        trial_df = pd.DataFrame(trial_rows)
        trial_output_path = Path(args.trial_output_path)
        trial_output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving per-trial results to {trial_output_path} ({len(trial_df)} rows)...")
        trial_df.to_csv(trial_output_path, index=False)
    
        # Print summary statistics
        print("\n" + "="*60)
        print("Summary Statistics")
        print("="*60)
    
        print("\nTheta Correlation by Cutoff:")
        print(df.groupby('cutoff')['theta_correlation'].agg(['mean', 'std', 'count']))
    
        print("\nTheta Variance by Cutoff:")
        print(df.groupby('cutoff')['theta_variance'].agg(['mean', 'std']))
    
        print("\nExpert Prediction Accuracy:")
        print(f"  Alpha correlation: {df['expert_alpha_corr'].mean():.4f} ± {df['expert_alpha_corr'].std():.4f}")
        print(f"  Beta correlation: {df['expert_beta_corr'].mean():.4f} ± {df['expert_beta_corr'].std():.4f}")
        print(f"  N_tot error: {df['expert_N_tot_error'].mean():.2f} ± {df['expert_N_tot_error'].std():.2f}")
        print(f"  N_tot raw (from data): {df['expert_N_tot_raw'].mean():.1f}")
        print(f"  N_tot true (computed): {df['expert_N_tot_true'].mean():.1f}")
        print(f"  N_tot pred: {df['expert_N_tot_pred'].mean():.1f}")
    
        print("\nNovice Prediction Accuracy:")
        print(f"  Alpha correlation: {df['novice_alpha_corr'].mean():.4f} ± {df['novice_alpha_corr'].std():.4f}")
        print(f"  Beta correlation: {df['novice_beta_corr'].mean():.4f} ± {df['novice_beta_corr'].std():.4f}")
        print(f"  N_tot error: {df['novice_N_tot_error'].mean():.2f} ± {df['novice_N_tot_error'].std():.2f}")
        print(f"  N_tot raw (from data): {df['novice_N_tot_raw'].mean():.1f}")
        print(f"  N_tot true (computed): {df['novice_N_tot_true'].mean():.1f}")
        print(f"  N_tot pred: {df['novice_N_tot_pred'].mean():.1f}")
    
        print(f"\n✓ Evaluation complete! Results saved to {args.output_path}")
        
        # Generate visualization if requested
        if not args.skip_plot:
            plot_evaluation_results(df, args.plot_output_dir)
        else:
            print("\nSkipping visualization (--skip_plot flag set)")
        
        # Evaluation completed successfully
        finalize_run_yaml(eval_dir, status='completed', subdir='metadata')
        print(f"\n{'='*60}")
        print("Evaluation completed successfully!")
        print(f"{'='*60}")
        
    except KeyboardInterrupt:
        print(f"\n{'='*60}")
        print("Evaluation interrupted by user")
        print(f"{'='*60}")
        finalize_run_yaml(eval_dir, status='interrupted', subdir='metadata')
        raise
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"Evaluation failed with error: {e}")
        print(f"{'='*60}")
        finalize_run_yaml(eval_dir, status='failed', subdir='metadata', error=str(e))
        raise


if __name__ == "__main__":
    main()


