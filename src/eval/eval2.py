"""
Evaluation script for model performance on simulated test data.

This script:
1. Loads test data from same source as trained model 
2. Loads model from wandb artifact
3. Performs rollouts at multiple segment cutoffs 
4. Measures theta prediction accuracy 
5. Measures each demonstrators prediction accuracy 
6. Outputs evaluations csv 
7. Plots evaluations 
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
import numpy as np

from models.BetaModel import SBIPosteriorTransformer
from models.load_checkpoint import load_checkpoint_from_local, load_checkpoint_from_wandb_artifact, load_checkpoint_from_wandb_sweep
from data.transforms import BeliefNormalizer
from data.dataloader import BanditSequenceDataset
from utils import get_special_characters, setup_eval_directory, get_training_runs_dir, write_run_yaml, write_config_yaml, finalize_run_yaml

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
    # generally triu creates a mask to return the upper triangular matrix 
    # diagonal=0 says return the diagonals too, diagonal>0 says how many of the diagonal levels to remove 
    # so diagonal=1 remove the main diagonals (so effectively returns a mask for upper triangular matrix exlusive of the diagonals)
    # e.g. tensor([[False,  True,  True,  True],
    #              [False, False,  True,  True],
    #              [False, False, False,  True],
    #              [False, False, False, False]])

    # But for additive masking as we are doing here it does the following: 
    # True = masked out (NOT visible)
    # False = visible (can be attended to)
    
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
        # loop is necessary since bandits may repeat in data where change_bandits_with_rounds: false
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
    num_segments = len(example["segment_info"])
    
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
        
        # Get ground truth for this bandit
        bandit_info = bandits[bandit_id]
        true_theta = torch.tensor(bandit_info["theta"], dtype=torch.float32)
        
        # Identify which demo_id is expert vs novice by checking N_tot
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
        
        # Get predictions for this round's QUERY
        pred_alpha_theta = outputs["alpha_theta"][0, round_idx].cpu()  # [K]
        pred_beta_theta = outputs["beta_theta"][0, round_idx].cpu()    # [K]
        pred_alpha_demo = outputs["alpha_demo"][0, round_idx].cpu()    # [D, K]
        pred_beta_demo = outputs["beta_demo"][0, round_idx].cpu()      # [D, K]
        
        # Denormalize demo predictions if needed
        if normalizer:
            pred_alpha_demo = normalizer.denormalize_alpha(pred_alpha_demo)
            pred_beta_demo = normalizer.denormalize_beta(pred_beta_demo)
        
        # Compute expected payoffs
        pred_theta_expected = pred_alpha_theta / (pred_alpha_theta + pred_beta_theta)
        
        # Compute theta correlation
        theta_corr = torch.corrcoef(
            torch.stack([pred_theta_expected, true_theta])
        )[0, 1].item()
        
        # Compute additional metrics
        theta_mse = F.mse_loss(pred_theta_expected, true_theta).item()
        theta_variance = pred_theta_expected.var().item()
        theta_range = (pred_theta_expected.max() - pred_theta_expected.min()).item()
        theta_tot = (pred_alpha_theta.sum() + pred_beta_theta.sum()).item()
        
        # Match predictions to ground truth based on expert/novice identification
        pred_alpha_expert = pred_alpha_demo[expert_id]
        pred_beta_expert = pred_beta_demo[expert_id]
        pred_alpha_novice = pred_alpha_demo[novice_id]
        pred_beta_novice = pred_beta_demo[novice_id]
        
        # Compute demonstrator alpha/beta correlations
        expert_alpha_corr = torch.corrcoef(
            torch.stack([pred_alpha_expert, true_alpha_expert])
        )[0, 1].item()
        expert_beta_corr = torch.corrcoef(
            torch.stack([pred_beta_expert, true_beta_expert])
        )[0, 1].item()
        novice_alpha_corr = torch.corrcoef(
            torch.stack([pred_alpha_novice, true_alpha_novice])
        )[0, 1].item()
        novice_beta_corr = torch.corrcoef(
            torch.stack([pred_beta_novice, true_beta_novice])
        )[0, 1].item()
        
        # Compute N_tot (effective experience count, subtracting prior)
        K = num_arms
        pred_expert_N = (pred_alpha_expert.sum() + pred_beta_expert.sum() - alpha0*K - beta0*K).item()
        true_expert_N = (true_alpha_expert.sum() + true_beta_expert.sum() - alpha0*K - beta0*K).item()
        pred_novice_N = (pred_alpha_novice.sum() + pred_beta_novice.sum() - alpha0*K - beta0*K).item()
        true_novice_N = (true_alpha_novice.sum() + true_beta_novice.sum() - alpha0*K - beta0*K).item()
        
        # Compile metrics
        metrics = {
            "cutoff": cutoff,
            "segment_type": seg_info["type"],
            "agent": seg_info.get("agent", -1),
            "round_index": round_idx,
            "bandit_id": int(bandit_id),
            
            # Theta metrics
            "theta_correlation": theta_corr,
            "theta_mse": theta_mse,
            "theta_variance": theta_variance,
            "theta_range": theta_range,
            "theta_tot": theta_tot,
            
            # Demonstrator metrics
            "expert_alpha_corr": expert_alpha_corr,
            "expert_beta_corr": expert_beta_corr,
            "novice_alpha_corr": novice_alpha_corr,
            "novice_beta_corr": novice_beta_corr,
            "expert_N_tot_pred": pred_expert_N,
            "expert_N_tot_true": true_expert_N,
            "expert_N_tot_error": abs(pred_expert_N - true_expert_N),
            "novice_N_tot_pred": pred_novice_N,
            "novice_N_tot_true": true_novice_N,
            "novice_N_tot_error": abs(pred_novice_N - true_novice_N),
        }
        
        results.append(metrics)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on test set")
    parser.add_argument("--test_manifest", type=str, required=True,
        help="Path to test manifest.json")

    # Checkpoint loading options (mutually exclusive)
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument("--checkpoint_path", type=str,
        help="Path to local checkpoint file")
    checkpoint_group.add_argument("--wandb_artifact", type=str,
        help="Load from wandb artifact: 'entity/project/artifact:alias'")

    args = parser.parse_args()

    eval_dir = setup_eval_directory()
    print(f"Eval directory: {eval_dir}")
    metadata = {
        'script': 'eval2.py',
        'command': ' '.join(sys.argv),
        'test_manifest': args.test_manifest,
    }
    write_run_yaml(eval_dir, metadata, subdir='metadata')
    write_config_yaml(eval_dir, vars(args), subdir='metadata')
    output_path = str(eval_dir / 'data' / 'test_evaluation.csv')
    plot_output_dir = str(eval_dir / 'figures')

    device = torch.device("cuda")

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
    
    # Extract masking strategy
    masking_strategy = model_config.get('masking_strategy', 'all_bandits')
    print(f"  masking_strategy: {masking_strategy}")
    
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
        profile_method=model_config.get('profile_method', 'learned_profile') ,
        experience_head_loss_type=model_config.get('profile_method', 'mse'),
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

    print("\n" + "="*60)
    print("Running Evaluation")
    print("="*60)
    
    all_results = []
    
    for ex_idx, example in enumerate(tqdm(all_examples, desc="Evaluating examples")):
        # Evaluate this example at all segment cutoffs
        example_results = evaluate_single_example(
            example=example,
            model=model,
            device=device,
            num_arms=model_config['num_arms'],
            num_agents=model_config['num_agents'],
            num_bandits=model_config['num_bandits'],
            SPECIAL=SPECIAL,
            normalizer=normalizer,
            alpha0=alpha0,
            beta0=beta0,
            masking_strategy=masking_strategy
        )
        all_results.extend(example_results)
    
    # Convert to DataFrame
    df_results = pd.DataFrame(all_results)
    
    # Save full results CSV
    print("\n" + "="*60)
    print("Saving Results")
    print("="*60)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_path, index=False)
    print(f"\n✓ Saved results CSV: {output_path}")
    print(f"  Total evaluations: {len(df_results)}")
    print(f"  Unique cutoffs: {df_results['cutoff'].nunique()}")
    
    # Create theta correlation figure
    print("\n" + "="*60)
    print("Generating Visualization")
    print("="*60)
    
    # Compute ground truth statistics
    true_payoffs = [0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.75, 0.8]
    true_variance = pd.Series(true_payoffs).var()
    true_range = max(true_payoffs) - min(true_payoffs)
    
    # Aggregate by cutoff
    cutoff_stats = df_results.groupby('cutoff').agg({
        'theta_correlation': ['mean', 'std', 'count'],
        'theta_variance': ['mean', 'std'],
        'theta_range': ['mean', 'std'],
        'theta_tot': ['mean', 'std'],
        'expert_alpha_corr': ['mean', 'std'],
        'expert_beta_corr': ['mean', 'std'],
        'novice_alpha_corr': ['mean', 'std'],
        'novice_beta_corr': ['mean', 'std'],
        'expert_N_tot_pred': ['mean', 'std'],
        'novice_N_tot_pred': ['mean', 'std'],
        'expert_N_tot_error': ['mean', 'std'],
        'novice_N_tot_error': ['mean', 'std'],
    }).reset_index()
    
    # Flatten column names
    cutoff_stats.columns = ['_'.join(col).strip('_') for col in cutoff_stats.columns.values]
    
    # Compute standard errors
    n = cutoff_stats['theta_correlation_count']
    cutoff_stats['theta_correlation_se'] = cutoff_stats['theta_correlation_std'] / (n ** 0.5)
    cutoff_stats['theta_variance_se'] = cutoff_stats['theta_variance_std'] / (n ** 0.5)
    cutoff_stats['theta_range_se'] = cutoff_stats['theta_range_std'] / (n ** 0.5)
    cutoff_stats['theta_tot_se'] = cutoff_stats['theta_tot_std'] / (n ** 0.5)
    cutoff_stats['expert_alpha_corr_se'] = cutoff_stats['expert_alpha_corr_std'] / (n ** 0.5)
    cutoff_stats['expert_beta_corr_se'] = cutoff_stats['expert_beta_corr_std'] / (n ** 0.5)
    cutoff_stats['novice_alpha_corr_se'] = cutoff_stats['novice_alpha_corr_std'] / (n ** 0.5)
    cutoff_stats['novice_beta_corr_se'] = cutoff_stats['novice_beta_corr_std'] / (n ** 0.5)
    cutoff_stats['expert_N_tot_pred_se'] = cutoff_stats['expert_N_tot_pred_std'] / (n ** 0.5)
    cutoff_stats['novice_N_tot_pred_se'] = cutoff_stats['novice_N_tot_pred_std'] / (n ** 0.5)
    cutoff_stats['expert_N_tot_error_se'] = cutoff_stats['expert_N_tot_error_std'] / (n ** 0.5)
    cutoff_stats['novice_N_tot_error_se'] = cutoff_stats['novice_N_tot_error_std'] / (n ** 0.5)
    
    # Create combined 2x4 figure
    num_arms_K = model_config['num_arms']
    prior_tot = num_arms_K * (alpha0 + beta0)
    true_expert_N_ref = df_results['expert_N_tot_true'].mean()
    true_novice_N_ref = df_results['novice_N_tot_true'].mean()
    
    fig, axes = plt.subplots(2, 4, figsize=(28, 12))
    fig.suptitle('Model Evaluation Summary (error bars = ±1 SE)', fontsize=18, fontweight='bold')
    
    # Row 1, Col 0: Theta Prediction Accuracy
    ax = axes[0, 0]
    ax.errorbar(cutoff_stats['cutoff'], cutoff_stats['theta_correlation_mean'],
                yerr=cutoff_stats['theta_correlation_se'],
                marker='o', capsize=4, capthick=1.5, linewidth=1.5)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Random (0)')
    ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Good (0.5)')
    ax.set_xlabel('Segment Cutoff', fontsize=10)
    ax.set_ylabel('Correlation', fontsize=10)
    ax.set_title('Theta Prediction Accuracy', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    
    # Row 1, Col 1: Theta Variance
    ax = axes[0, 1]
    ax.errorbar(cutoff_stats['cutoff'], cutoff_stats['theta_variance_mean'],
                yerr=cutoff_stats['theta_variance_se'],
                marker='o', capsize=4, capthick=1.5, linewidth=1.5, color='orange', label='Predicted')
    ax.axhline(y=true_variance, color='green', linestyle='--', linewidth=2, alpha=0.7,
               label=f'True variance ({true_variance:.4f})')
    ax.set_xlabel('Segment Cutoff', fontsize=10)
    ax.set_ylabel('Variance of Expected Payoffs', fontsize=10)
    ax.set_title('Theta Prediction Informativeness', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    
    # Row 1, Col 2: Theta Range
    ax = axes[0, 2]
    ax.errorbar(cutoff_stats['cutoff'], cutoff_stats['theta_range_mean'],
                yerr=cutoff_stats['theta_range_se'],
                marker='o', capsize=4, capthick=1.5, linewidth=1.5, color='green', label='Predicted')
    ax.axhline(y=true_range, color='purple', linestyle='--', linewidth=2, alpha=0.7,
               label=f'True range ({true_range:.2f})')
    ax.set_xlabel('Segment Cutoff', fontsize=10)
    ax.set_ylabel('Range (max - min)', fontsize=10)
    ax.set_title('Theta Prediction Range', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    
    # Row 1, Col 3: Theta Tot
    ax = axes[0, 3]
    ax.errorbar(cutoff_stats['cutoff'], cutoff_stats['theta_tot_mean'],
                yerr=cutoff_stats['theta_tot_se'],
                marker='o', capsize=4, capthick=1.5, linewidth=1.5, color='blue', label='Predicted')
    ax.axhline(y=prior_tot, color='red', linestyle='--', linewidth=2, alpha=0.7,
               label=f'Prior K*(a0+b0) = {prior_tot:.1f}')
    ax.set_xlabel('Segment Cutoff', fontsize=10)
    ax.set_ylabel('Theta Tot (sum of all a + b)', fontsize=10)
    ax.set_title('Theta Head Parameter Magnitude', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    
    # Row 2, Col 0: Alpha Correlation
    ax = axes[1, 0]
    ax.errorbar(cutoff_stats['cutoff'], cutoff_stats['expert_alpha_corr_mean'],
                yerr=cutoff_stats['expert_alpha_corr_se'],
                marker='o', capsize=4, capthick=1.5, linewidth=1.5, color='blue', label='Expert', alpha=0.8)
    ax.errorbar(cutoff_stats['cutoff'], cutoff_stats['novice_alpha_corr_mean'],
                yerr=cutoff_stats['novice_alpha_corr_se'],
                marker='s', capsize=4, capthick=1.5, linewidth=1.5, color='red', label='Novice', alpha=0.8)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Segment Cutoff', fontsize=10)
    ax.set_ylabel('Correlation', fontsize=10)
    ax.set_title('Alpha Prediction Accuracy', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    
    # Row 2, Col 1: Beta Correlation
    ax = axes[1, 1]
    ax.errorbar(cutoff_stats['cutoff'], cutoff_stats['expert_beta_corr_mean'],
                yerr=cutoff_stats['expert_beta_corr_se'],
                marker='o', capsize=4, capthick=1.5, linewidth=1.5, color='blue', label='Expert', alpha=0.8)
    ax.errorbar(cutoff_stats['cutoff'], cutoff_stats['novice_beta_corr_mean'],
                yerr=cutoff_stats['novice_beta_corr_se'],
                marker='s', capsize=4, capthick=1.5, linewidth=1.5, color='red', label='Novice', alpha=0.8)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Segment Cutoff', fontsize=10)
    ax.set_ylabel('Correlation', fontsize=10)
    ax.set_title('Beta Prediction Accuracy', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    
    # Row 2, Col 2: N_tot Predictions vs Ground Truth
    ax = axes[1, 2]
    ax.axhline(y=true_expert_N_ref, color='lightblue', linestyle='--', linewidth=2, alpha=0.7,
               label=f'True Expert ({true_expert_N_ref:.0f})', zorder=1)
    ax.errorbar(cutoff_stats['cutoff'], cutoff_stats['expert_N_tot_pred_mean'],
                yerr=cutoff_stats['expert_N_tot_pred_se'],
                marker='o', linewidth=1.5, color='blue', label='Pred Expert',
                markersize=5, capsize=3, alpha=0.8, zorder=3)
    ax.axhline(y=true_novice_N_ref, color='lightcoral', linestyle='--', linewidth=2, alpha=0.7,
               label=f'True Novice ({true_novice_N_ref:.0f})', zorder=1)
    ax.errorbar(cutoff_stats['cutoff'], cutoff_stats['novice_N_tot_pred_mean'],
                yerr=cutoff_stats['novice_N_tot_pred_se'],
                marker='s', linewidth=1.5, color='red', label='Pred Novice',
                markersize=5, capsize=3, alpha=0.8, zorder=3)
    ax.set_xlabel('Segment Cutoff', fontsize=10)
    ax.set_ylabel('N_tot (Total Experience)', fontsize=10)
    ax.set_title('N_tot Predictions vs Ground Truth', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    
    # Row 2, Col 3: N_tot Absolute Error
    ax = axes[1, 3]
    ax.errorbar(cutoff_stats['cutoff'], cutoff_stats['expert_N_tot_error_mean'],
                yerr=cutoff_stats['expert_N_tot_error_se'],
                marker='o', capsize=4, capthick=1.5, linewidth=1.5, color='blue', label='Expert', alpha=0.8)
    ax.errorbar(cutoff_stats['cutoff'], cutoff_stats['novice_N_tot_error_mean'],
                yerr=cutoff_stats['novice_N_tot_error_se'],
                marker='s', capsize=4, capthick=1.5, linewidth=1.5, color='red', label='Novice', alpha=0.8)
    ax.set_xlabel('Segment Cutoff', fontsize=10)
    ax.set_ylabel('Absolute Error in N_tot', fontsize=10)
    ax.set_title('N_tot Prediction Error', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    
    plt.tight_layout()
    
    plot_path = Path(plot_output_dir) / 'evaluation_summary.png'
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n✓ Saved plot: {plot_path}")
    
    # Finalize run metadata
    finalize_run_yaml(eval_dir, status='completed', subdir='metadata')
    print(f"\n{'='*60}")
    print("Evaluation completed successfully!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()