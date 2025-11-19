"""
Inference script for comparing model predictions with human experimental data.

This script:
1. Loads human experiment data (CSV)
2. Extracts demonstration segments
3. Uses the model to predict demonstrator beliefs
4. Simulates learner behavior using Thompson sampling with predicted beliefs
5. Outputs hybrid data (real demos + simulated learner segments)
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random

import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from models.BetaModel import SBIPosteriorTransformer, thompson_sample_action
from load_checkpoint import load_checkpoint_from_file
from transforms import BeliefNormalizer


# Constants
LEARNER_RAW_ID = 99
SPECIAL = {
    "NONE": 0,
    "QUERY": 1,
    "BANDIT_SEP": 2,
}


def load_and_preprocess_data(csv_path: str) -> pd.DataFrame:
    """
    Load experiment data and filter for relevant rows.
    
    Args:
        csv_path: Path to the experiment CSV file
        
    Returns:
        Filtered DataFrame with only shot events from watch/play games
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} total rows")
    
    # Convert boolean strings to actual booleans - handle various formats
    # The CSV has "FALSE" and "TRUE" as strings
    df['shot'] = df['shot'].astype(str).str.upper().isin(['TRUE', 'T', '1', 'YES'])
    df['reward_generated'] = df['reward_generated'].astype(str).str.upper().isin(['TRUE', 'T', '1', 'YES'])
    
    print(f"Rows with shot==True: {df['shot'].sum()}")
    print(f"Games: {df['game'].value_counts().to_dict()}")
    
    # Filter: only keep rows where shot==True and game is not 'choice'
    df_filtered = df[(df['shot'] == True) & (df['game'].isin(['watch', 'play']))].copy()
    
    print(f"After filtering for shot==True and game in ['watch', 'play']: {len(df_filtered)} rows")
    
    # Keep only relevant columns
    cols_to_keep = [
        'session_id', 'session_index', 'anonymous_id', 'position', 'action',
        'game', 'game_sequence', 'game_index', 'round_index', 'demonstrator_type', 
        'demonstrator_trajectory', 'zone_bandit', 'zone_global_index', 
        'reward_generated'
    ]
    df_filtered = df_filtered[cols_to_keep]
    
    # Sort by session_id and game_index to maintain temporal order
    df_filtered = df_filtered.sort_values(['session_id', 'round_index', 'game_index']).reset_index(drop=True)
    
    print(f"Filtered to {len(df_filtered)} shot events from {df_filtered['session_id'].nunique()} sessions")
    
    return df_filtered


def parse_game_sequence(game_seq: str) -> List[str]:
    """
    Parse game sequence string to extract demonstrator order.
    
    Args:
        game_seq: e.g., "novice-play-expert-play-choice" or "expert-play-novice-play-choice"
        
    Returns:
        List of segment types in order, e.g., ['novice', 'play', 'expert', 'play', 'choice']
    """
    return game_seq.split('-')


def parse_bandit_payoffs(zone_bandit_str: str) -> List[float]:
    """
    Parse zone_bandit string to list of float payoffs.
    
    Args:
        zone_bandit_str: e.g., "0.55,0,0.8,0.5,0.4,0.3,0.6,0.75,0.1,0.35,0.45,0.2"
        
    Returns:
        List of payoff probabilities
    """
    return [float(x) for x in zone_bandit_str.split(',')]


def thompson_sampling_step(
    alpha: torch.Tensor, 
    beta: torch.Tensor, 
    bandit_payoffs: List[float],
    num_steps: int = 6
) -> List[Tuple[int, int]]:
    """
    Run Thompson sampling for a learner segment.
    
    Args:
        alpha: Initial alpha parameters [K]
        beta: Initial beta parameters [K]
        bandit_payoffs: True payoff probabilities for each arm
        num_steps: Number of actions to take (default 6)
        
    Returns:
        List of (action, reward) tuples
    """
    alpha = alpha.clone()
    beta = beta.clone()
    trajectory = []
    
    for _ in range(num_steps):
        # Sample action using Thompson sampling
        action = thompson_sample_action(alpha, beta)
        
        # Generate reward from true bandit
        reward = 1 if random.random() < bandit_payoffs[action] else 0
        
        trajectory.append((action, reward))
        
        # Update beliefs
        alpha[action] += reward
        beta[action] += (1 - reward)
    
    return trajectory


def convert_session_to_model_format(
    session_df: pd.DataFrame,
    visible_game_indices: List[int],
    num_arms: int = 12
) -> Dict:
    """
    Convert session data to model input format.
    
    Args:
        session_df: DataFrame for a single session
        visible_game_indices: Which game_index values to include
        num_arms: Number of bandit arms
        
    Returns:
        Dictionary with model input tensors and metadata
    """
    # Filter to visible games
    visible_df = session_df[session_df['game_index'].isin(visible_game_indices)].copy()
    
    # Map zone_global_index to local bandit IDs (0-indexed per session)
    zone_to_bandit = {zone: idx for idx, zone in enumerate(sorted(visible_df['zone_global_index'].unique()))}
    visible_df['bandit_id'] = visible_df['zone_global_index'].map(zone_to_bandit)
    
    # Map demonstrator types to agent IDs
    # Learner = 0, then assign IDs to demonstrators in order they appear
    demo_types = []
    for _, row in visible_df.iterrows():
        if row['game'] == 'watch' and pd.notna(row['demonstrator_type']):
            if row['demonstrator_type'] not in demo_types:
                demo_types.append(row['demonstrator_type'])
    
    demo_to_agent = {demo_type: idx + 1 for idx, demo_type in enumerate(demo_types)}
    
    # Build events list: (bandit_id, agent_id, action, reward)
    events = []
    segment_boundaries = [0]  # Start positions of each segment
    segment_info = []
    round_boundaries = [0]
    current_round = None
    
    for idx, (_, row) in enumerate(visible_df.iterrows()):
        bandit_id = int(row['bandit_id'])
        action = int(row['position'])
        reward = 1 if row['reward_generated'] else 0
        
        # Determine agent_id
        if row['game'] == 'watch':
            agent_id = demo_to_agent[row['demonstrator_type']]
        else:  # play
            agent_id = 0  # learner
        
        events.append((bandit_id, agent_id, action, reward))
        
        # Track round boundaries
        if current_round is None:
            current_round = row['round_index']
        elif row['round_index'] != current_round:
            round_boundaries.append(idx)
            current_round = row['round_index']
        
        # Track segment boundaries (when game_index changes)
        if idx > 0:
            prev_game_idx = visible_df.iloc[idx - 1]['game_index']
            if row['game_index'] != prev_game_idx:
                segment_boundaries.append(idx)
                segment_info.append({
                    'round': int(visible_df.iloc[idx - 1]['round_index']),
                    'game_index': int(prev_game_idx)
                })
    
    # Add final segment info
    segment_info.append({
        'round': int(visible_df.iloc[-1]['round_index']),
        'game_index': int(visible_df.iloc[-1]['game_index'])
    })
    
    round_boundaries.append(len(events))
    
    # Get round-bandit mapping
    round_bandit_map = []
    for round_idx in sorted(visible_df['round_index'].unique()):
        round_df = visible_df[visible_df['round_index'] == round_idx]
        bandit_id = int(round_df.iloc[0]['bandit_id'])
        round_bandit_map.append(bandit_id)
    
    # Get bandit payoffs for each unique bandit
    bandits_info = []
    for zone_idx in sorted(zone_to_bandit.keys()):
        zone_df = visible_df[visible_df['zone_global_index'] == zone_idx]
        payoffs = parse_bandit_payoffs(zone_df.iloc[0]['zone_bandit'])
        bandits_info.append({
            'zone_global_index': int(zone_idx),
            'payoffs': payoffs
        })
    
    return {
        'events': events,
        'segment_boundaries': segment_boundaries,
        'segment_info': segment_info,
        'round_boundaries': round_boundaries,
        'round_bandit_map': round_bandit_map,
        'bandits_info': bandits_info,
        'zone_to_bandit': zone_to_bandit,
        'demo_to_agent': demo_to_agent,
        'num_demonstrators': len(demo_types)
    }


def encode_events_to_tensors(
    events: List[Tuple[int, int, int, int]],
    num_arms: int,
    num_bandits: int
) -> Dict[str, torch.Tensor]:
    """
    Encode events to tensor format for model input.
    
    Args:
        events: List of (bandit_id, agent_id, action, reward) tuples
        num_arms: Number of bandit arms
        num_bandits: Number of unique bandits
        
    Returns:
        Dictionary of tensors
    """
    bandit_ids = []
    agent_ids = []
    action_ids = []
    reward_ids = []
    special_ids = []
    
    for (b, ag, a, r) in events:
        bandit_ids.append(min(max(b, 0), num_bandits - 1))
        agent_ids.append(ag)
        action_ids.append(a if (0 <= a < num_arms) else num_arms)
        reward_ids.append(r if r in (0, 1) else 2)
        special_ids.append(SPECIAL["NONE"])
    
    # Add QUERY token at the end
    bandit_ids.append(0)
    agent_ids.append(0)
    action_ids.append(num_arms)  # PAD
    reward_ids.append(2)  # PAD
    special_ids.append(SPECIAL["QUERY"])
    
    return {
        'bandit_ids': torch.tensor(bandit_ids, dtype=torch.long).unsqueeze(0),  # [1, T]
        'agent_ids': torch.tensor(agent_ids, dtype=torch.long).unsqueeze(0),
        'action_ids': torch.tensor(action_ids, dtype=torch.long).unsqueeze(0),
        'reward_ids': torch.tensor(reward_ids, dtype=torch.long).unsqueeze(0),
        'special_ids': torch.tensor(special_ids, dtype=torch.long).unsqueeze(0),
    }


def create_segment_mask(segment_boundaries: List[int], cutoff: int, seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create attention mask that makes only the first 'cutoff' segments visible.
    
    Args:
        segment_boundaries: List of segment start positions
        cutoff: Number of segments to make visible (1-indexed)
        seq_len: Total sequence length
        device: torch device
        
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
    
    return mask


def run_inference_on_session(
    session_df: pd.DataFrame,
    model: nn.Module,
    device: torch.device,
    alpha0: float,
    beta0: float,
    num_arms: int = 12,
    learner_steps: int = 6
) -> pd.DataFrame:
    """
    Run inference on a single session, generating hybrid data.
    
    Args:
        session_df: DataFrame for one session
        model: Trained model
        device: torch device
        alpha0: Prior alpha parameter
        beta0: Prior beta parameter
        num_arms: Number of bandit arms
        learner_steps: Number of steps per learner segment
        
    Returns:
        DataFrame with original demos + simulated learner segments
    """
    # Parse game sequence to understand structure
    game_seq = session_df.iloc[0]['game_sequence']
    seq_parts = parse_game_sequence(game_seq)
    
    # Identify which game_indices are demos vs learners
    game_indices = sorted(session_df['game_index'].unique())
    demo_games = session_df[session_df['game'] == 'watch']['game_index'].unique()
    learner_games = session_df[session_df['game'] == 'play']['game_index'].unique()
    
    # Build result incrementally
    result_rows = []
    accumulated_events = []
    current_segment_idx = 0
    
    # Process games in order
    for game_idx in game_indices:
        game_df = session_df[session_df['game_index'] == game_idx].copy()
        
        if game_idx in demo_games:
            # This is a demonstration segment - keep as is
            for _, row in game_df.iterrows():
                result_rows.append(row.to_dict())
                
                # Add to accumulated events for model context
                bandit_id = 0  # Will be properly mapped later
                demo_type = row['demonstrator_type']
                action = int(row['position'])
                reward = 1 if row['reward_generated'] else 0
                # Events will be properly formatted when we do inference
            
            current_segment_idx += 1
            
        elif game_idx in learner_games:
            # This is a learner segment - simulate it using model predictions
            
            # Convert accumulated context to model format
            visible_games = [g for g in game_indices if g < game_idx]
            
            if len(visible_games) == 0:
                # No context yet, shouldn't happen in proper game sequences
                print(f"Warning: Learner segment before any demo in session {session_df.iloc[0]['session_id']}")
                continue
            
            model_input = convert_session_to_model_format(
                session_df, 
                visible_games,
                num_arms
            )
            
            # Encode to tensors
            tensors = encode_events_to_tensors(
                model_input['events'],
                num_arms,
                len(model_input['bandits_info'])
            )
            
            # Add padding mask (no padding for single sequence)
            seq_len = tensors['bandit_ids'].shape[1]
            tensors['pad_mask'] = torch.zeros(1, seq_len, dtype=torch.bool)
            
            # Move to device
            for k in tensors:
                tensors[k] = tensors[k].to(device)
            
            # Create segment mask to only see visible segments
            num_visible_segments = len(visible_games)
            attn_mask = create_segment_mask(
                model_input['segment_boundaries'] + [seq_len],  # Add end position
                num_visible_segments,
                seq_len,
                device
            )
            
            # Run model inference
            with torch.no_grad():
                model.eval()
                outputs = model(tensors, custom_attn_mask=attn_mask)
            
            # Get predicted beliefs for the demonstrator from the last demo segment
            last_demo_game_idx = visible_games[-1]
            last_demo_df = session_df[session_df['game_index'] == last_demo_game_idx]
            last_demo_type = last_demo_df.iloc[0]['demonstrator_type']
            
            # Map demo type to agent ID
            demo_agent_id = model_input['demo_to_agent'][last_demo_type]
            demo_id = demo_agent_id - 1  # Convert to 0-indexed for predictions
            
            # Get predictions (alpha_demo, beta_demo are [1, num_demos, K])
            pred_alpha = outputs['alpha_demo'][0, demo_id, :]  # [K]
            pred_beta = outputs['beta_demo'][0, demo_id, :]    # [K]
            
            # Get current bandit payoffs
            current_round = game_df.iloc[0]['round_index']
            current_zone = game_df.iloc[0]['zone_global_index']
            bandit_payoffs = parse_bandit_payoffs(game_df.iloc[0]['zone_bandit'])
            
            # Run Thompson sampling
            trajectory = thompson_sampling_step(
                pred_alpha.cpu(),
                pred_beta.cpu(),
                bandit_payoffs,
                learner_steps
            )
            
            # Create rows for simulated learner segment
            for step_idx, (action_idx, reward) in enumerate(trajectory):
                row = {
                    'session_id': session_df.iloc[0]['session_id'],
                    'session_index': session_df.iloc[0]['session_index'],
                    'anonymous_id': session_df.iloc[0]['anonymous_id'],
                    'position': action_idx,
                    'action': 'shoot',
                    'game': 'play',
                    'game_sequence': game_seq,
                    'game_index': game_idx,
                    'round_index': current_round,
                    'demonstrator_type': np.nan,
                    'demonstrator_trajectory': np.nan,
                    'zone_bandit': game_df.iloc[0]['zone_bandit'],
                    'zone_global_index': current_zone,
                    'reward_generated': bool(reward)
                }
                result_rows.append(row)
            
            current_segment_idx += 1
    
    return pd.DataFrame(result_rows)


def main():
    parser = argparse.ArgumentParser(
        description="Run model inference on human experiment data to generate hybrid predictions"
    )
    parser.add_argument(
        "--experiment_data_path",
        type=str,
        required=True,
        help="Path to experiment CSV file"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to model checkpoint file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/model_predictions/hybrid_model_predictions.csv",
        help="Path to output CSV file (default: data/model_predictions/hybrid_model_predictions.csv)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run inference on (default: cpu)"
    )
    parser.add_argument(
        "--num_arms",
        type=int,
        default=12,
        help="Number of bandit arms (default: 12)"
    )
    parser.add_argument(
        "--num_agents",
        type=int,
        default=4,
        help="Maximum number of agents (learner + demonstrators) (default: 4)"
    )
    parser.add_argument(
        "--learner_steps",
        type=int,
        default=6,
        help="Number of steps per learner segment (default: 6)"
    )
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load experiment data
    df = load_and_preprocess_data(args.experiment_data_path)
    
    # Check if we have data
    if len(df) == 0:
        print("\nERROR: No data after filtering!")
        print("Please check that:")
        print("  1. The CSV file has rows where shot==True")
        print("  2. The game column contains 'watch' or 'play' values")
        return
    
    # Load model checkpoint first to get the correct configuration
    print(f"\nLoading model checkpoint from {args.checkpoint_path}...")
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    
    # Determine number of demonstrators and bandits from data
    num_demonstrators = 2  # Typically novice + expert
    num_bandits = df.groupby('session_id')['zone_global_index'].nunique().max()
    
    print(f"Detected {num_bandits} unique bandits across all sessions")
    
    # Try to load model configuration from checkpoint
    if 'model_config' in checkpoint:
        print("Found model_config in checkpoint, using saved hyperparameters")
        model_config = checkpoint['model_config']
        
        # Override with detected values from data where necessary
        model_config['num_bandits'] = num_bandits
        model_config['num_demonstrators'] = num_demonstrators
        
        print(f"Model configuration from checkpoint:")
        print(f"  d_model: {model_config['d_model']}")
        print(f"  num_layers: {model_config['num_layers']}")
        print(f"  dim_feedforward: {model_config['dim_feedforward']}")
        print(f"  nhead: {model_config['nhead']}")
        print(f"  num_agents: {model_config['num_agents']}")
        print(f"  dropout: {model_config['dropout']}")
    else:
        # Fallback: infer from state_dict for older checkpoints
        print("Warning: model_config not found in checkpoint, inferring from state_dict")
        state_dict = checkpoint['model_state_dict']
        
        # Infer dimensions from state dict
        agent_emb_shape = state_dict['embed.agent_emb.weight'].shape
        num_agents_inferred = agent_emb_shape[0]
        d_model_inferred = agent_emb_shape[1]
        
        # Infer num_layers by counting encoder layers
        num_layers_inferred = 0
        for key in state_dict.keys():
            if 'backbone.encoder.layers.' in key:
                layer_num = int(key.split('backbone.encoder.layers.')[1].split('.')[0])
                num_layers_inferred = max(num_layers_inferred, layer_num + 1)
        
        # Infer dim_feedforward from linear1 weight
        if 'backbone.encoder.layers.0.linear1.weight' in state_dict:
            dim_feedforward_inferred = state_dict['backbone.encoder.layers.0.linear1.weight'].shape[0]
        else:
            dim_feedforward_inferred = 2048  # default
        
        # Infer nhead from attention weights
        if 'backbone.encoder.layers.0.self_attn.in_proj_weight' in state_dict:
            in_proj_weight_shape = state_dict['backbone.encoder.layers.0.self_attn.in_proj_weight'].shape
            # in_proj_weight has shape [3*d_model, d_model] for Q, K, V
            nhead_inferred = 8  # default, hard to infer accurately
        else:
            nhead_inferred = 8
        
        model_config = {
            'num_arms': args.num_arms,
            'num_agents': num_agents_inferred,
            'num_bandits': num_bandits,
            'num_demonstrators': num_demonstrators,
            'd_model': d_model_inferred,
            'nhead': nhead_inferred,
            'num_layers': num_layers_inferred,
            'dim_feedforward': dim_feedforward_inferred,
            'dropout': 0.1,  # Can't infer dropout, use default
        }
        
        print(f"Inferred model configuration:")
        print(f"  d_model: {d_model_inferred}")
        print(f"  num_layers: {num_layers_inferred}")
        print(f"  dim_feedforward: {dim_feedforward_inferred}")
        print(f"  nhead: {nhead_inferred} (using default)")
        print(f"  num_agents: {num_agents_inferred}")
    
    # Create model with configuration
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
    ).to(device)
    
    # Load checkpoint state
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Checkpoint loaded successfully!")
    
    # Get normalization parameters
    alpha0 = checkpoint.get('normalization', {}).get('alpha0', 1.0)
    beta0 = checkpoint.get('normalization', {}).get('beta0', 1.0)
    print(f"Using alpha0={alpha0}, beta0={beta0}")
    
    # Run inference on each session
    print(f"\nProcessing {df['session_id'].nunique()} sessions...")
    all_results = []
    
    for session_id in df['session_id'].unique():
        print(f"Processing session: {session_id}")
        session_df = df[df['session_id'] == session_id].copy()
        
        result_df = run_inference_on_session(
            session_df,
            model,
            device,
            alpha0,
            beta0,
            args.num_arms,
            args.learner_steps
        )
        
        all_results.append(result_df)
    
    # Combine all results
    final_df = pd.concat(all_results, ignore_index=True)
    
    # Create output directory if needed
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    print(f"\nSaving results to {args.output_path}...")
    final_df.to_csv(args.output_path, index=False)
    
    print(f"\nDone! Generated {len(final_df)} rows for {final_df['session_id'].nunique()} sessions")
    print(f"Output saved to: {args.output_path}")


if __name__ == "__main__":
    main()

