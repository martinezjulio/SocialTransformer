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
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from models.BetaModel import SBIPosteriorTransformer, thompson_sample_action
from load_checkpoint import load_checkpoint_from_local, load_checkpoint_from_wandb_artifact, load_checkpoint_from_wandb_sweep
from transforms import BeliefNormalizer
from utils import setup_inference_directory, get_training_runs_dir, write_run_yaml, write_config_yaml, finalize_run_yaml, get_special_characters


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
        'zone_bandit', 'zone_global_index', 'reward_generated'
    ]
    df_filtered = df_filtered[cols_to_keep]
    
    # Sort by session_id and game_index to maintain temporal order
    df_filtered = df_filtered.sort_values(['session_id', 'round_index', 'game_index']).reset_index(drop=True)
    
    # Remove sessions with incomplete round structure
    expected_rounds = {0, 1, 2}
    expected_games_per_round = 4  # watch, play, watch, play
    incomplete_sessions = []
    for sid in df_filtered['session_id'].unique():
        sdf = df_filtered[df_filtered['session_id'] == sid]
        rounds = set(int(r) for r in sdf['round_index'].unique())
        n_games = sdf['game_index'].nunique()
        if rounds != expected_rounds or n_games != expected_games_per_round * len(expected_rounds):
            incomplete_sessions.append(sid)
    
    if incomplete_sessions:
        print(f"\nRemoving {len(incomplete_sessions)} session(s) with incomplete data:")
        for sid in incomplete_sessions:
            sdf = df_filtered[df_filtered['session_id'] == sid]
            rounds = sorted(int(r) for r in sdf['round_index'].unique())
            n_games = sdf['game_index'].nunique()
            print(f"  {sid}: rounds={rounds}, games={n_games}")
        df_filtered = df_filtered[~df_filtered['session_id'].isin(incomplete_sessions)].reset_index(drop=True)
    
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
    cleaned = zone_bandit_str.strip().strip('[]')
    return [float(x) for x in cleaned.split(',')]


def parse_demonstrator_order(game_sequence: str) -> Dict[str, int]:
    """
    Parse game sequence to determine which demonstrator type maps to which model output index.
    
    Args:
        game_sequence: e.g., "novice-play-expert-play-choice" or "expert-play-novice-play-choice"
        
    Returns:
        Dictionary mapping demonstrator type to model demo index (0 or 1)
        e.g., {'novice': 0, 'expert': 1} or {'expert': 0, 'novice': 1}
    """
    parts = game_sequence.split('-')
    demo_order = []
    for part in parts:
        if part in ['novice', 'expert']:
            demo_order.append(part)
    
    # Map first demonstrator to index 0, second to index 1
    result = {}
    for idx, demo_type in enumerate(demo_order):
        result[demo_type] = idx
    
    return result


def serialize_betas(alpha: torch.Tensor, beta: torch.Tensor) -> str:
    """
    Serialize alpha and beta tensors to string of tuples.
    
    Args:
        alpha: [K] tensor of alpha parameters
        beta: [K] tensor of beta parameters
        
    Returns:
        String like "[(a1, b1), (a2, b2), ...]"
    """
    pairs = [(float(a), float(b)) for a, b in zip(alpha.cpu(), beta.cpu())]
    return str(pairs)


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
    
    # Build events list with segment boundaries matching training data format:
    # Each segment (watch/play game) gets a boundary, plus QUERY tokens at end of each round
    events = []
    segment_boundaries = [0]
    segment_info = []
    round_boundaries = [0]
    current_round = None
    current_game_idx = None
    rounds_seen = []
    
    for idx, (_, row) in enumerate(visible_df.iterrows()):
        bandit_id = int(row['bandit_id'])
        action = int(row['position'])
        reward = 1 if row['reward_generated'] else 0

        if row['game'] == 'watch':
            agent_id = demo_to_agent[row['demonstrator_type']]
            seg_type = "demo"
        else:
            agent_id = 0
            seg_type = "learner"

        if current_round is None:
            current_round = row['round_index']
            current_game_idx = row['game_index']
            rounds_seen.append(current_round)

            # BANDIT_SEP token at start of first round
            events.append((bandit_id, -1, -1, -1))
            segment_boundaries.append(len(events))
            segment_info.append({"round": int(current_round), "type": "bandit_sep", "agent": -1})
        elif row['round_index'] != current_round:
            # End of previous segment
            segment_boundaries.append(len(events))
            prev_row = visible_df.iloc[idx - 1]
            prev_agent = demo_to_agent[prev_row['demonstrator_type']] if prev_row['game'] == 'watch' else 0
            segment_info.append({
                "round": int(current_round),
                "type": "demo" if prev_row['game'] == 'watch' else "learner",
                "agent": prev_agent
            })

            # QUERY token at end of previous round
            prev_bandit = int(visible_df.iloc[idx - 1]['bandit_id'])
            events.append((prev_bandit, -1, -2, -1))
            segment_boundaries.append(len(events))
            segment_info.append({
                "round": int(current_round),
                "type": "query",
                "agent": -1,
                "bandit": prev_bandit
            })

            round_boundaries.append(len(events))
            current_round = row['round_index']
            current_game_idx = row['game_index']
            rounds_seen.append(current_round)

            # BANDIT_SEP token at start of new round
            events.append((bandit_id, -1, -1, -1))
            segment_boundaries.append(len(events))
            segment_info.append({"round": int(current_round), "type": "bandit_sep", "agent": -1})
        elif row['game_index'] != current_game_idx:
            # New segment within same round
            segment_boundaries.append(len(events))
            prev_row = visible_df.iloc[idx - 1]
            prev_agent = demo_to_agent[prev_row['demonstrator_type']] if prev_row['game'] == 'watch' else 0
            segment_info.append({
                "round": int(current_round),
                "type": "demo" if prev_row['game'] == 'watch' else "learner",
                "agent": prev_agent
            })
            current_game_idx = row['game_index']

        events.append((bandit_id, agent_id, action, reward))
    
    # Final segment info for last segment
    last_row = visible_df.iloc[-1]
    last_agent = demo_to_agent[last_row['demonstrator_type']] if last_row['game'] == 'watch' else 0
    segment_boundaries.append(len(events))
    segment_info.append({
        "round": int(last_row['round_index']),
        "type": "demo" if last_row['game'] == 'watch' else "learner",
        "agent": last_agent
    })
    
    # QUERY token at end of last round
    last_bandit = int(last_row['bandit_id'])
    events.append((last_bandit, -1, -2, -1))
    segment_boundaries.append(len(events))
    segment_info.append({
        "round": int(last_row['round_index']),
        "type": "query",
        "agent": -1,
        "bandit": last_bandit
    })
    
    round_boundaries.append(len(events))
    
    # Round-to-query-index mapping (each round produces one QUERY, indexed 0, 1, 2...)
    sorted_rounds = sorted(visible_df['round_index'].unique())
    round_to_index = {r: i for i, r in enumerate(sorted_rounds)}
    
    round_bandit_map = []
    for round_idx in sorted_rounds:
        round_df = visible_df[visible_df['round_index'] == round_idx]
        round_bandit_map.append(int(round_df.iloc[0]['bandit_id']))
    
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
        'round_to_index': round_to_index,
        'bandits_info': bandits_info,
        'zone_to_bandit': zone_to_bandit,
        'demo_to_agent': demo_to_agent,
        'num_demonstrators': len(demo_types)
    }


def encode_events_to_tensors(
    events: List[Tuple[int, int, int, int]],
    num_arms: int,
    num_bandits: int,
    num_agents: int,
    SPECIAL: Dict[str, int]
) -> Dict[str, torch.Tensor]:
    """
    Encode events to tensor format for model input.
    Handles structural tokens (BANDIT_SEP and QUERY) which use ag=-1 sentinel.
    
    Args:
        events: List of (bandit_id, agent_id, action, reward) tuples
                QUERY tokens: (bandit_id, -1, -2, -1)
                BANDIT_SEP tokens: (bandit_id, -1, -1, -1) 
        num_arms: Number of bandit arms
        num_bandits: Number of unique bandits
        num_agents: Number of agents (for padding)
        SPECIAL: Dictionary of special token IDs
        
    Returns:
        Dictionary of tensors
    """
    bandit_ids = []
    agent_ids = []
    action_ids = []
    reward_ids = []
    special_ids = []
    
    for (b, ag, a, r) in events:
        # Check if structural token (uses -1 sentinel)
        if ag == -1:
            # Distinguish BANDIT_SEP (a=-1) from QUERY (a=-2)
            if a == -2:
                # QUERY token
                bandit_ids.append(min(max(b, 0), num_bandits - 1))
                agent_ids.append(num_agents)  # padding
                action_ids.append(num_arms)   # padding
                reward_ids.append(2)          # padding
                special_ids.append(SPECIAL["QUERY"])
            else:
                # BANDIT_SEP token
                bandit_ids.append(min(max(b, 0), num_bandits - 1))
                agent_ids.append(num_agents)  # padding
                action_ids.append(num_arms)   # padding
                reward_ids.append(2)          # padding
                special_ids.append(SPECIAL["BANDIT_SEP"])
        else:
            # Regular event
            bandit_ids.append(min(max(b, 0), num_bandits - 1))
            agent_ids.append(ag)
            action_ids.append(a if (0 <= a < num_arms) else num_arms)
            reward_ids.append(r if r in (0, 1) else 2)
            special_ids.append(SPECIAL["NONE"])
    
    return {
        'bandit_ids': torch.tensor(bandit_ids, dtype=torch.long).unsqueeze(0),  # [1, T]
        'agent_ids': torch.tensor(agent_ids, dtype=torch.long).unsqueeze(0),
        'action_ids': torch.tensor(action_ids, dtype=torch.long).unsqueeze(0),
        'reward_ids': torch.tensor(reward_ids, dtype=torch.long).unsqueeze(0),
        'special_ids': torch.tensor(special_ids, dtype=torch.long).unsqueeze(0),
    }


def create_segment_mask(
    segment_boundaries: List[int],
    cutoff: int,
    seq_len: int,
    device: torch.device,
    masking_strategy: str = "all_bandits",
    segment_info: List[Dict] = None,
    round_bandit_map: List[int] = None
) -> torch.Tensor:
    """
    Create attention mask that makes only the first 'cutoff' segments visible.
    
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


def run_inference_on_session(
    session_df: pd.DataFrame,
    model: nn.Module,
    device: torch.device,
    alpha0: float,
    beta0: float,
    num_arms: int = 12,
    num_agents: int = 3,
    learner_steps: int = 6,
    SPECIAL: Dict[str, int] = None,
    learner_segment_input: str = "human",
    beta_method: str = "theta_betas",
    num_rollouts: int = 1,
    seed: int = 42,
    masking_strategy: str = "all_bandits"
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
        num_agents: Number of agents (for padding)
        learner_steps: Number of steps per learner segment
        SPECIAL: Dictionary of special token IDs
        learner_segment_input: "human" to use original learner data as context, 
                              "model" to use simulated learner data as context
        beta_method: Which beta parameters to use for Thompson sampling:
                    "theta_betas" (bandit payoff beliefs),
                    "novice_betas" (novice demonstrator beliefs),
                    "expert_betas" (expert demonstrator beliefs), or
                    "recent_betas" (most recent demonstrator's beliefs)
        num_rollouts: Number of independent rollouts per session
        seed: Base random seed for reproducibility
        
    Returns:
        DataFrame with original demos + simulated learner segments,
        plus learner_segment_input, beta_method, and rollout_id columns
    """
    # Parse game sequence to understand structure
    game_seq = session_df.iloc[0]['game_sequence']
    seq_parts = parse_game_sequence(game_seq)
    
    # Parse demonstrator order (which demo type maps to which model index)
    demo_order_map = parse_demonstrator_order(game_seq)
    
    # Identify which game_indices are demos vs learners
    game_indices = sorted(session_df['game_index'].unique())
    demo_games = session_df[session_df['game'] == 'watch']['game_index'].unique()
    learner_games = session_df[session_df['game'] == 'play']['game_index'].unique()
    
    # Collect results from all rollouts
    all_rollout_rows = []
    
    # Run multiple independent rollouts
    for rollout_id in range(num_rollouts):
        # Set reproducible seed for this rollout
        rollout_seed = seed + rollout_id
        random.seed(rollout_seed)
        np.random.seed(rollout_seed)
        
        # Build result incrementally
        result_rows = []
        accumulated_events = []
        current_segment_idx = 0
        
        # Track simulated learner data for "model" mode
        simulated_data_rows = []
        
        # Process games in order
        for game_idx in game_indices:
            game_df = session_df[session_df['game_index'] == game_idx].copy()
            
            if game_idx in demo_games:
                for _, row in game_df.iterrows():
                    row_dict = row.to_dict()
                    row_dict['rollout_id'] = rollout_id
                    row_dict['learner_segment_input'] = learner_segment_input
                    row_dict['beta_method'] = beta_method
                    row_dict['theta_betas'] = np.nan
                    row_dict['expert_betas'] = np.nan
                    row_dict['novice_betas'] = np.nan
                    result_rows.append(row_dict)
                
                current_segment_idx += 1
                
            elif game_idx in learner_games:
                visible_games = [g for g in game_indices if g < game_idx]
                
                if len(visible_games) == 0:
                    print(f"Warning: Learner segment before any demo in session {session_df.iloc[0]['session_id']}")
                    continue
                
                if learner_segment_input == "model" and len(simulated_data_rows) > 0:
                    simulated_df = pd.DataFrame(simulated_data_rows)
                    demos_df = session_df[
                        (session_df['game'] == 'watch') & 
                        (session_df['game_index'].isin(visible_games))
                    ].copy()
                    context_df = pd.concat([demos_df, simulated_df], ignore_index=True)
                    context_df = context_df.sort_values(['round_index', 'game_index']).reset_index(drop=True)
                else:
                    context_df = session_df
                
                model_input = convert_session_to_model_format(
                    context_df, 
                    visible_games,
                    num_arms
                )
                
                tensors = encode_events_to_tensors(
                    model_input['events'],
                    num_arms,
                    len(model_input['bandits_info']),
                    num_agents,
                    SPECIAL
                )
                
                seq_len = tensors['bandit_ids'].shape[1]
                tensors['pad_mask'] = torch.zeros(1, seq_len, dtype=torch.bool)
                
                for k in tensors:
                    tensors[k] = tensors[k].to(device)
                
                num_visible_segments = len(model_input['segment_info'])
                attn_mask = create_segment_mask(
                    model_input['segment_boundaries'],
                    num_visible_segments,
                    seq_len,
                    device,
                    masking_strategy=masking_strategy,
                    segment_info=model_input['segment_info'],
                    round_bandit_map=model_input['round_bandit_map']
                )
                
                with torch.no_grad():
                    model.eval()
                    outputs = model(tensors, custom_attn_mask=attn_mask)
                
                current_round = int(game_df.iloc[0]['round_index'])
                current_zone = int(game_df.iloc[0]['zone_global_index'])
                
                # Use round index to index model outputs (one prediction per QUERY per round)
                if current_round in model_input['round_to_index']:
                    current_round_idx = model_input['round_to_index'][current_round]
                else:
                    # Incomplete session: current round not in visible context, use last QUERY
                    print(f"  Warning: Round {current_round} not in visible context for session "
                          f"{session_df.iloc[0]['session_id']}, game_idx={game_idx}. Using last available QUERY.")
                    current_round_idx = len(model_input['round_to_index']) - 1
                
                pred_alpha_theta_cur = outputs['alpha_theta'][0, current_round_idx]  # [K]
                pred_beta_theta_cur = outputs['beta_theta'][0, current_round_idx]    # [K]
                pred_alpha_demo_cur = outputs['alpha_demo'][0, current_round_idx]    # [D, K]
                pred_beta_demo_cur = outputs['beta_demo'][0, current_round_idx]      # [D, K]
                
                theta_betas_str = serialize_betas(pred_alpha_theta_cur, pred_beta_theta_cur)
                
                novice_demo_idx = demo_order_map['novice']
                novice_betas_str = serialize_betas(
                    pred_alpha_demo_cur[novice_demo_idx, :],
                    pred_beta_demo_cur[novice_demo_idx, :]
                )
                
                expert_demo_idx = demo_order_map['expert']
                expert_betas_str = serialize_betas(
                    pred_alpha_demo_cur[expert_demo_idx, :],
                    pred_beta_demo_cur[expert_demo_idx, :]
                )
                
                if beta_method == "theta_betas":
                    pred_alpha = pred_alpha_theta_cur
                    pred_beta = pred_beta_theta_cur
                elif beta_method == "novice_betas":
                    pred_alpha = pred_alpha_demo_cur[demo_order_map['novice'], :]
                    pred_beta = pred_beta_demo_cur[demo_order_map['novice'], :]
                elif beta_method == "expert_betas":
                    pred_alpha = pred_alpha_demo_cur[demo_order_map['expert'], :]
                    pred_beta = pred_beta_demo_cur[demo_order_map['expert'], :]
                elif beta_method == "recent_betas":
                    last_demo_game_idx = max(g for g in visible_games if g in demo_games)
                    last_demo_df = session_df[session_df['game_index'] == last_demo_game_idx]
                    last_demo_type = last_demo_df.iloc[0]['demonstrator_type']
                    recent_demo_idx = demo_order_map[last_demo_type]
                    pred_alpha = pred_alpha_demo_cur[recent_demo_idx, :]
                    pred_beta = pred_beta_demo_cur[recent_demo_idx, :]
                else:
                    raise ValueError(f"Unknown beta_method: {beta_method}")
                
                bandit_payoffs = parse_bandit_payoffs(game_df.iloc[0]['zone_bandit'])
                
                trajectory = thompson_sampling_step(
                    pred_alpha.cpu(),
                    pred_beta.cpu(),
                    bandit_payoffs,
                    learner_steps
                )
                
                for step_idx, (action_idx, reward) in enumerate(trajectory):
                    row = {
                        'rollout_id': rollout_id,
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
                        'zone_bandit': game_df.iloc[0]['zone_bandit'],
                        'zone_global_index': current_zone,
                        'reward_generated': bool(reward),
                        'learner_segment_input': learner_segment_input,
                        'beta_method': beta_method,
                        'theta_betas': theta_betas_str,
                        'expert_betas': expert_betas_str,
                        'novice_betas': novice_betas_str
                    }
                    result_rows.append(row)
                    simulated_data_rows.append(row)
                
                current_segment_idx += 1
        
        all_rollout_rows.extend(result_rows)
    
    return pd.DataFrame(all_rollout_rows)


def main():
    parser = argparse.ArgumentParser(description="Run model inference on human experiment data to generate hybrid predictions")
    parser.add_argument("--experiment_data_path",type=str,required=True,help="Path to experiment CSV file")
    parser.add_argument("--output_path",type=str,default="data/model_predictions/hybrid_model_predictions.csv",help="Path to output CSV file")
    parser.add_argument("--device",type=str,default="cpu",choices=["cpu", "cuda"],help="Device to run inference on")
    parser.add_argument("--num_rollouts", type=int, default=1,
        help="Number of independent rollouts per session (default: 1)")
    parser.add_argument("--seed", type=int, default=42,
        help="Base random seed for reproducibility (default: 42)")
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
        help="Load from wandb artifact: 'entity/project/artifact:alias' (e.g., 'user/project/model-checkpoint:best')")
    checkpoint_group.add_argument("--wandb_sweep", type=str,
        help="Load best model from sweep: 'entity/project/sweep_id' (e.g., 'user/project/abc123xyz')")
    
    args = parser.parse_args()
    
    # Set up inference directory structure
    inference_dir = setup_inference_directory()
    print(f"Inference directory: {inference_dir}")

    # Write run metadata
    metadata = {
        'script': 'inference.py',
        'command': ' '.join(sys.argv),
        'experiment_data_path': args.experiment_data_path,
        'num_rollouts': args.num_rollouts,
        'seed': args.seed,
    }
    write_run_yaml(inference_dir, metadata, subdir='metadata')
    write_config_yaml(inference_dir, vars(args), subdir='metadata')

    # Update output path to use analysis directory if it's still default
    if args.output_path == "data/model_predictions/hybrid_model_predictions.csv":
        args.output_path = str(inference_dir / 'data' / 'hybrid_model_predictions.csv')
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    try:
        # Load experiment data
        df = load_and_preprocess_data(args.experiment_data_path)
        
        if len(df) == 0:
            print("\nERROR: No data after filtering!")
            print("Please check that:")
            print("  1. The CSV file has rows where shot==True")
            print("  2. The game column contains 'watch' or 'play' values")
            return
        
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
        
        SPECIAL = get_special_characters()
        
        # Extract config BEFORE model creation to avoid forward-reference
        masking_strategy = model_config.get('masking_strategy', 'all_bandits')
        profile_method = model_config.get('profile_method', 'learned_profile')
        experience_head_loss_type = model_config.get('experience_head_loss_type', 'mse')
        
        # Apply overrides if specified
        if args.masking_strategy_override:
            masking_strategy = args.masking_strategy_override
            print(f"  masking_strategy: {masking_strategy} (override)")
        else:
            print(f"  masking_strategy: {masking_strategy}")
        
        if args.profile_method_override:
            profile_method = args.profile_method_override
            print(f"  profile_method: {profile_method} (override)")
        else:
            print(f"  profile_method: {profile_method}")
        
        print(f"  experience_head_loss_type: {experience_head_loss_type}")
        
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
        print("Model loaded successfully!")
        
        if 'normalization' not in checkpoint:
            raise ValueError("Checkpoint missing 'normalization' parameters")
        
        alpha0 = checkpoint['normalization']['alpha0']
        beta0 = checkpoint['normalization']['beta0']
        print(f"  alpha0={alpha0}, beta0={beta0}")
        
        # Run inference on each session for all 8 combinations
        print(f"\nProcessing {df['session_id'].nunique()} sessions with {args.num_rollouts} rollout(s) x 8 combinations...")
        
        all_results = []
        
        for learner_input in tqdm(["human", "model"], desc="Learner input mode", position=0):
            for beta_method in tqdm(["theta_betas", "novice_betas", "expert_betas", "recent_betas"], 
                                    desc=f"Beta method ({learner_input})", position=1, leave=False):
                
                session_ids = df['session_id'].unique()
                for session_id in tqdm(session_ids, 
                                      desc=f"Sessions ({learner_input}, {beta_method})", 
                                      position=2, leave=False):
                    session_df = df[df['session_id'] == session_id].copy()
                    
                    result_df = run_inference_on_session(
                        session_df,
                        model,
                        device,
                        alpha0,
                        beta0,
                        model_config['num_arms'],
                        model_config['num_agents'],
                        model_config['T_learner'],
                        SPECIAL,
                        learner_segment_input=learner_input,
                        beta_method=beta_method,
                        num_rollouts=args.num_rollouts,
                        seed=args.seed,
                        masking_strategy=masking_strategy
                    )
                    
                    all_results.append(result_df)
        
        print("\nCombining all results...")
        final_df = pd.concat(all_results, ignore_index=True)
        
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving results to {args.output_path}...")
        final_df.to_csv(args.output_path, index=False)
        
        print(f"\nDone! Generated {len(final_df)} rows for {final_df['session_id'].nunique()} sessions")
        print(f"  - 8 combinations x {args.num_rollouts} rollouts = {8 * args.num_rollouts} runs per session")
        print(f"Output saved to: {args.output_path}")
        
        finalize_run_yaml(inference_dir, status='completed', subdir='metadata')
        print(f"\n{'='*60}")
        print("Inference completed successfully!")
        print(f"{'='*60}")
        
    except KeyboardInterrupt:
        print(f"\n{'='*60}")
        print("Inference interrupted by user")
        print(f"{'='*60}")
        finalize_run_yaml(inference_dir, status='interrupted', subdir='metadata')
        raise
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"Inference failed with error: {e}")
        print(f"{'='*60}")
        finalize_run_yaml(inference_dir, status='failed', subdir='metadata', error=str(e))
        raise


if __name__ == "__main__":
    main()

