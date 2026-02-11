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

from models.BetaModel import SBIPosteriorTransformer
from load_checkpoint import load_checkpoint_from_local, load_checkpoint_from_wandb_artifact, load_checkpoint_from_wandb_sweep
from transforms import BeliefNormalizer
from dataloader import BanditSequenceDataset
from utils import get_special_characters, get_run_id, setup_run_directory, write_run_yaml, write_config_yaml, finalize_run_yaml

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
    from dataloader import _remap_agent_ids_local
    
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

    run_id = get_run_id(use_wandb=False)
    run_dir = setup_run_directory(run_id)
    print(f"Run directory: {run_dir}")
    print(f"Run ID: {run_id}")
    metadata = {
        'run_id': run_id,
        'script': 'eval.py',
        'command': ' '.join(sys.argv),
        'test_manifest': args.test_manifest,
    }
    write_run_yaml(run_dir, metadata)
    write_config_yaml(run_dir, vars(args))
    output_path = str(run_dir / 'results' / 'evaluation' / 'test_evaluation.csv')
    plot_output_dir = str(run_dir / 'results' / 'figures' / 'evaluation')

    device = torch.device("cuda")

    # Load checkpoint
    print("\n" + "="*60)
    print("Loading Model Checkpoint")
    print("="*60)
    
    if args.checkpoint_path:
        checkpoint_path, checkpoint = load_checkpoint_from_local(args.checkpoint_path)
    elif args.wandb_artifact:
        download_dir = str(run_dir / 'checkpoints' / 'wandb_downloads')
        checkpoint_path, checkpoint = load_checkpoint_from_wandb_artifact(
            args.wandb_artifact, download_dir
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
        normalizer = BeliefNormalizer(
            N_max=checkpoint['normalization']['N_max'],
            alpha0=checkpoint['normalization']['alpha0'],
            beta0=checkpoint['normalization']['beta0']
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
        
        batch = prepare_single_example_batch(example, model_config['num_arms'], model_config['num_agents'], model_config['num_bandits'], SPECIAL)
        print(batch)
        break

if __name__ == "__main__":
    main()