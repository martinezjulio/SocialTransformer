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

from models.BetaModel import SBIPosteriorTransformer
from models.load_checkpoint import load_checkpoint_from_local, load_checkpoint_from_wandb_artifact, load_checkpoint_from_wandb_sweep
from data.transforms import BeliefNormalizer
from data.dataloader import BanditSequenceDataset
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
    segment_info: List[Dict] = None,
    round_bandit_map: List[int] = None
) -> torch.Tensor:
    """
    Create attention mask for single example with segment cutoff.
    
    Args:
        segment_boundaries: List of segment start positions
        cutoff: Number of segments to make visible (1-indexed)
        seq_len: Total sequence length
        device: torch device
        masking_strategy: "all_bandits" or "current_bandit_only"
        segment_info: List of segment metadata dicts
        round_bandit_map: List mapping round index to bandit ID
        
    Returns:
        mask: [seq_len, seq_len] - True = masked (not attended to), False = visible
    """
    # Start with causal mask
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    
    # Get cutoff position
    if cutoff < len(segment_boundaries):
        cutoff_pos = segment_boundaries[cutoff]
    else:
        cutoff_pos = seq_len  # All visible
    
    # Mask out everything from cutoff_pos onward
    if cutoff_pos < seq_len:
        mask[:, cutoff_pos:] = True
    
    # Additional masking for "current_bandit_only" strategy
    if masking_strategy == "current_bandit_only" and segment_info is not None and round_bandit_map is not None:
        # Determine the target bandit from the cutoff segment
        seg_info_target = segment_info[cutoff - 1]
        target_round = seg_info_target["round"]
        target_bandit = int(round_bandit_map[target_round])
        
        # Mask out all segments that belong to different bandits
        for seg_idx in range(cutoff):
            seg_info_i = segment_info[seg_idx]
            round_i = seg_info_i["round"]
            bandit_i = int(round_bandit_map[round_i])
            
            if bandit_i != target_bandit:
                # Mask this entire segment
                start_pos = segment_boundaries[seg_idx]
                if seg_idx + 1 < len(segment_boundaries):
                    end_pos = segment_boundaries[seg_idx + 1]
                else:
                    end_pos = cutoff_pos
                
                # Mask all positions from attending to this segment
                mask[:, start_pos:end_pos] = True
    
    return mask


def prepare_single_example_batch(
    example: Dict,
    num_arms: int,
    num_agents: int,
    num_bandits: int,
    SPECIAL: Dict[str, int]
) -> Dict[str, torch.Tensor]:
    """
    Prepare a single example as a batch for model inference.
    Reuses logic from BanditSequenceDataset.__getitem__.
    """
    from data.dataloader import _remap_agent_ids_local
    
    # Remap agent ids
    events = example["events"]
    remapped_events, _ = _remap_agent_ids_local(events, num_agents=num_agents)
    
    # Encode events (includes QUERY tokens from raw data)
    bandit_ids = []
    agent_ids = []
    action_ids = []
    reward_ids = []
    special_ids = []
    
    for (b, ag, a, r) in remapped_events:
        assert 0 <= b < num_bandits, f"Invalid bandit_id {b}, must be in [0, {num_bandits})"
        # Check if this is structural token (agent_id was set to num_agents in remapping)
        if ag == num_agents:
            # Distinguish BANDIT_SEP (a=-1) from QUERY (a=-2)
            if a == -2:
                # QUERY token
                bandit_ids.append(b)
                agent_ids.append(num_agents)  # padding
                action_ids.append(num_arms)   # padding
                reward_ids.append(2)          # padding
                special_ids.append(SPECIAL["QUERY"])
            else:
                # BANDIT_SEP token
                bandit_ids.append(b)
                agent_ids.append(num_agents)  # padding
                action_ids.append(num_arms)   # padding
                reward_ids.append(2)          # padding
                special_ids.append(SPECIAL["BANDIT_SEP"])
        else:
            # Regular event
            bandit_ids.append(b)
            agent_ids.append(ag)
            action_ids.append(a if (0 <= a < num_arms) else num_arms)
            reward_ids.append(r if r in (0, 1) else 2)
            special_ids.append(SPECIAL["NONE"])
    
    # Convert to tensors with batch dimension
    batch = {
        "bandit_ids": torch.tensor(bandit_ids, dtype=torch.long).unsqueeze(0),  # [1, T]
        "agent_ids": torch.tensor(agent_ids, dtype=torch.long).unsqueeze(0),
        "action_ids": torch.tensor(action_ids, dtype=torch.long).unsqueeze(0),
        "reward_ids": torch.tensor(reward_ids, dtype=torch.long).unsqueeze(0),
        "special_ids": torch.tensor(special_ids, dtype=torch.long).unsqueeze(0),
        "pad_mask": torch.zeros(1, len(bandit_ids), dtype=torch.bool),
    }
    
    return batch


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
    """
    Evaluate model on single example at multiple segment cutoffs.
    Returns list of metric dicts, one per cutoff position.
    """
    # Get ground truth
    labels = example["labels"]
    bandits = labels["bandits"]
    num_segments = len(example["segment_info"])  # segment_info, not segment_boundaries!

    # Determine game_sequence (condition) from first demo segment ordering
    first_demo_id = None
    for seg in example["segment_info"]:
        if seg["type"] == "demo":
            first_demo_id = seg["agent"] - 1  # convert agent_id (1,2) to demo_id (0,1)
            break

    # Identify expert by N_tot from first bandit (consistent across bandits)
    demo0_N = bandits[0]["demonstrators"][0]["N_tot"]
    demo1_N = bandits[0]["demonstrators"][1]["N_tot"]
    example_expert_id = 0 if demo0_N > demo1_N else 1

    if first_demo_id == example_expert_id:
        game_sequence = "expert-play-novice-play-choice"
    else:
        game_sequence = "novice-play-expert-play-choice"

    # Determine segment_order: watch-first (demo before learner) vs play-first (learner before demo)
    first_non_structural = None
    for seg in example["segment_info"]:
        if seg["type"] in ("demo", "learner"):
            first_non_structural = seg["type"]
            break
    segment_order = "watch-first" if first_non_structural == "demo" else "play-first"

    results = []
    
    # Prepare batch once
    batch = prepare_single_example_batch(
        example, num_arms, num_agents, num_bandits, SPECIAL
    )
    
    # Move to device
    for k in batch:
        batch[k] = batch[k].to(device)
    
    seq_len = batch["bandit_ids"].shape[1]
    
    # Evaluate at each possible cutoff
    for cutoff in range(1, num_segments + 1):
        seg_info = example["segment_info"][cutoff - 1]
        round_idx = seg_info["round"]
        bandit_id = int(example["round_bandit_map"][round_idx])
        
        # Create attention mask for this cutoff
        attn_mask = create_segment_mask_single(
            example["segment_boundaries"],
            cutoff,
            seq_len,
            device,
            masking_strategy=masking_strategy,
            segment_info=example["segment_info"],
            round_bandit_map=example["round_bandit_map"]
        )
        
        # Run model
        model.eval()
        with torch.no_grad():
            outputs = model(batch, custom_attn_mask=attn_mask)
        
        # Get ground truth for this bandit (bandits is a list indexed by bandit_id)
        bandit_info = bandits[bandit_id]
        true_theta = torch.tensor(bandit_info["theta"], dtype=torch.float32)
        
        # Identify which demo_id is expert vs novice by checking N_tot
        # Expert has high N_tot, novice has low N_tot
        demo0_N = bandit_info["demonstrators"][0]["N_tot"]
        demo1_N = bandit_info["demonstrators"][1]["N_tot"]
        
        if demo0_N > demo1_N:
            expert_id, novice_id = 0, 1
        else:
            expert_id, novice_id = 1, 0
        
        true_alpha_expert = torch.tensor(bandit_info["demonstrators"][expert_id]["alpha0"], dtype=torch.float32)
        true_beta_expert = torch.tensor(bandit_info["demonstrators"][expert_id]["beta0"], dtype=torch.float32)
        true_alpha_novice = torch.tensor(bandit_info["demonstrators"][novice_id]["alpha0"], dtype=torch.float32)
        true_beta_novice = torch.tensor(bandit_info["demonstrators"][novice_id]["beta0"], dtype=torch.float32)
        
        # Get predictions for this bandit
        # outputs are [B, num_queries, K] or [B, num_queries, D, K]
        pred_alpha_theta = outputs["alpha_theta"][0, bandit_id].cpu()  # [K]
        pred_beta_theta = outputs["beta_theta"][0, bandit_id].cpu()    # [K]
        pred_alpha_demo = outputs["alpha_demo"][0, bandit_id].cpu()    # [num_demos, K]
        pred_beta_demo = outputs["beta_demo"][0, bandit_id].cpu()      # [num_demos, K]
        
        # Denormalize if needed
        if normalizer:
            pred_alpha_demo = normalizer.denormalize_alpha(pred_alpha_demo)
            pred_beta_demo = normalizer.denormalize_beta(pred_beta_demo)
        
        # Compute expected payoffs
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
            "segment_type": seg_info["type"],
            "agent": seg_info.get("agent", -1),
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
    parser.add_argument("--test_manifest", type=str, required=True,
        help="Path to test manifest.json")
    parser.add_argument("--output_path", type=str, 
        default="data/evaluation/test_evaluation.csv",
        help="Path to save evaluation results")
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
    parser.add_argument("--profile_method_override", type=str, default=None,
        choices=["no_profile", "learned_profile", "agent_pooling", "cross_attention"],
        help="Override profile method from checkpoint (optional)")
    
    # Checkpoint loading options (mutually exclusive)
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument("--checkpoint_path", type=str,
        help="Path to local checkpoint file")
    checkpoint_group.add_argument("--wandb_artifact", type=str,
        help="Load from wandb artifact: 'entity/project/artifact:alias'")
    checkpoint_group.add_argument("--wandb_sweep", type=str,
        help="Load best model from sweep: 'entity/project/sweep_id'")
    
    args = parser.parse_args()
    
    # Set up eval directory structure
    eval_dir = setup_eval_directory()
    print(f"Eval directory: {eval_dir}")

    # Write run metadata
    metadata = {
        'script': 'eval.py',
        'command': ' '.join(sys.argv),
        'test_manifest': args.test_manifest,
    }
    write_run_yaml(eval_dir, metadata, subdir='metadata')
    write_config_yaml(eval_dir, vars(args), subdir='metadata')

    # Set output paths to eval directory
    if args.output_path == "data/evaluation/test_evaluation.csv":
        args.output_path = str(eval_dir / 'data' / 'test_evaluation.csv')
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
    
        if args.checkpoint_path:
            checkpoint_path, checkpoint = load_checkpoint_from_local(args.checkpoint_path)
        elif args.wandb_artifact:
            checkpoint_path, checkpoint = load_checkpoint_from_wandb_artifact(
                args.wandb_artifact, str(get_training_runs_dir())
            )
        elif args.wandb_sweep:
            checkpoint_path, checkpoint = load_checkpoint_from_wandb_sweep(
                args.wandb_sweep, str(get_training_runs_dir())
            )
        else:
            raise ValueError("No checkpoint source specified")
    
        # Extract model configuration
        if 'model_config' not in checkpoint:
            raise ValueError("Checkpoint missing 'model_config'")
    
        model_config = checkpoint['model_config']
    
        print(f"\nModel configuration:")
        print(f"  num_arms: {model_config['num_arms']}")
        print(f"  num_agents: {model_config['num_agents']}")
        print(f"  num_bandits: {model_config['num_bandits']}")
        print(f"  num_demonstrators: {model_config['num_demonstrators']}")
        print(f"  d_model: {model_config['d_model']}")
        print(f"  nhead: {model_config['nhead']}")
        print(f"  num_layers: {model_config['num_layers']}")
    
        # Get profile method (from checkpoint or override)
        if args.profile_method_override:
            profile_method = args.profile_method_override
            print(f"✓ Using overridden profile method: {profile_method}")
        else:
            profile_method = checkpoint['model_config'].get('profile_method', 'no_profile')
            print(f"✓ Using profile method from checkpoint: {profile_method}")

        experience_head_loss_type = checkpoint['model_config'].get('experience_head_loss_type', 'mse')
        print(f"✓ Experience head loss type: {experience_head_loss_type}")

        # Create model
        model = SBIPosteriorTransformer(
            num_arms=model_config['num_arms'],
            num_agents=model_config['num_agents'],
            num_bandits=model_config['num_bandits'],
            num_demonstrators=model_config['num_demonstrators'],
            d_model=model_config['d_model'],
            nhead=model_config['nhead'],
            num_layers=model_config['num_layers'],
            dim_feedforward=model_config['dim_feedforward'],
            dropout=model_config['dropout'],
            profile_method=profile_method,
            experience_head_loss_type=experience_head_loss_type,
            theta_head_input=model_config.get('theta_head_input', 'query_only'),
            demo_beliefs_head_input=model_config.get('demo_beliefs_head_input', 'query_and_profiles'),
            experience_head_input=model_config.get('experience_head_input', 'profiles_only'),
        ).to(device)
    
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
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
    
        # Load test data
        print("\n" + "="*60)
        print("Loading Test Data")
        print("="*60)
    
        manifest_path = args.test_manifest
        print(f"Reading manifest: {manifest_path}")
    
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
    
        test_dir = os.path.dirname(manifest_path)
        shards = manifest['shards']
    
        print(f"Found {len(shards)} shards")
        print(f"Total examples: {sum(s['num_examples'] for s in shards)}")
    
        # Load examples from shards
        all_examples = []
        for shard_info in shards:
            shard_path = os.path.join(test_dir, shard_info['path'])
            print(f"  Loading {shard_path}...")
            examples = torch.load(shard_path, map_location='cpu')
            all_examples.extend(examples)
    
        print(f"✓ Loaded {len(all_examples)} examples")
    
        # Limit examples if requested
        if args.max_examples:
            all_examples = all_examples[:args.max_examples]
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
    
        # Save results
        print(f"\nSaving results to {args.output_path}...")
        df.to_csv(args.output_path, index=False)
    
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


