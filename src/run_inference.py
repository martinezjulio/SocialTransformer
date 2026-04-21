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
import hashlib
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
from models.load_checkpoint import load_checkpoint_from_local, load_checkpoint_from_wandb_artifact, load_checkpoint_from_wandb_sweep
from data.transforms import BeliefNormalizer
from utils import setup_eval_directory, get_training_runs_dir, write_run_yaml, write_config_yaml, finalize_run_yaml, get_special_characters


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
    
    # Validate session structure (expected to pass — preprocess.R drops bad sessions upstream).
    expected_rounds = {0, 1, 2}
    expected_games_per_round = 4  # watch, play, watch, play
    expected_shots_per_play = 6   # matches training T_learner; enforced strictly

    bad_sessions = {}
    for sid in df_filtered['session_id'].unique():
        sdf = df_filtered[df_filtered['session_id'] == sid]
        rounds = set(int(r) for r in sdf['round_index'].unique())
        n_games = sdf['game_index'].nunique()
        reasons = []
        if rounds != expected_rounds or n_games != expected_games_per_round * len(expected_rounds):
            reasons.append(f"incomplete-rounds (rounds={sorted(rounds)}, n_games={n_games})")
        play_counts = sdf[sdf['game'] == 'play'].groupby('game_index').size()
        if (play_counts != expected_shots_per_play).any():
            bad = play_counts[play_counts != expected_shots_per_play].to_dict()
            reasons.append(f"short-play {bad}")
        if reasons:
            bad_sessions[sid] = reasons

    if bad_sessions:
        lines = [f"  {sid}: {'; '.join(rs)}" for sid, rs in bad_sessions.items()]
        raise ValueError(
            f"Found {len(bad_sessions)} session(s) with invalid structure in "
            f"preprocessed_data.csv. Re-run reports/r/preprocess.R to regenerate.\n"
            + "\n".join(lines)
        )

    print(f"Validated {len(df_filtered)} shot events from {df_filtered['session_id'].nunique()} sessions")

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


def _rollout_seed(base_seed: int, session_id: str, rollout_id: int) -> int:
    # Stable hash (SHA-256) avoids PYTHONHASHSEED non-determinism across processes.
    digest = hashlib.sha256(f"{session_id}|{rollout_id}".encode()).digest()
    return (base_seed + int.from_bytes(digest[:4], "big")) % (2**31)


def thompson_sampling_step(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    bandit_payoffs: List[float],
    num_steps: int = 6,
    *,
    rng: random.Random,
    np_rng: np.random.Generator,
) -> List[Tuple[int, int]]:
    """
    Run Thompson sampling for a learner segment.

    Args:
        alpha: Initial alpha parameters [K]
        beta: Initial beta parameters [K]
        bandit_payoffs: True payoff probabilities for each arm
        num_steps: Number of actions to take (default 6)
        rng: Python RNG for reward Bernoulli draws
        np_rng: Numpy generator for Thompson action sampling

    Returns:
        List of (action, reward) tuples
    """
    alpha = alpha.clone()
    beta = beta.clone()
    trajectory = []

    for _ in range(num_steps):
        action = thompson_sample_action(alpha, beta, np_rng)
        reward = 1 if rng.random() < bandit_payoffs[action] else 0
        trajectory.append((action, reward))
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
    masking_strategy: str = "all_bandits",
    normalizer: Optional[BeliefNormalizer] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run inference on a single session, producing both per-trial rows
    (for experiment_trial_predictions.csv) and per-cutoff rows (for the N_total
    experience plots).

    Unified cutoff sweep: the full session is tokenized once via
    convert_session_to_model_format/encode_events_to_tensors. For each rollout
    we clone the tensors and iterate cutoff ∈ 1..num_segments, running one
    forward pass per cutoff on the shared tensors with a segment-level
    attention mask. For model-mode learner segments we additionally run a
    pre-segment forward pass (cutoff - 1) whose output seeds Thompson sampling;
    the sampled (action, reward) trajectory is spliced into full_tensors'
    action_ids/reward_ids before the recorded post-segment pass runs. This
    guarantees bit-identical outputs to human mode when Thompson sampling is
    replaced with the real human actions.

    Returns:
        per_game_df: per-trial / per-Thompson-step rows with existing schema
            (experiment_trial_predictions.csv consumer).
        per_cutoff_df: one row per (cutoff, rollout) tagged with
            segment_type / segment_order / game_sequence / round_index /
            expert_N_tot_pred / novice_N_tot_pred / beta strings.
    """
    # Parse game sequence to understand structure
    game_seq = session_df.iloc[0]['game_sequence']
    demo_order_map = parse_demonstrator_order(game_seq)

    # Identify which game_indices are demos vs learners
    game_indices = sorted(session_df['game_index'].unique())
    demo_games = set(session_df[session_df['game'] == 'watch']['game_index'].unique().tolist())
    learner_games = set(session_df[session_df['game'] == 'play']['game_index'].unique().tolist())

    expert_demo_idx = demo_order_map['expert']
    novice_demo_idx = demo_order_map['novice']

    session_id_val = session_df.iloc[0]['session_id']
    session_index_val = session_df.iloc[0]['session_index']
    anonymous_id_val = session_df.iloc[0]['anonymous_id']

    K = num_arms

    # Tokenize the full session once.
    full_model_input = convert_session_to_model_format(
        session_df, game_indices, num_arms
    )
    num_segments = len(full_model_input['segment_info'])
    segment_boundaries = full_model_input['segment_boundaries']
    segment_info = full_model_input['segment_info']

    original_tensors = encode_events_to_tensors(
        full_model_input['events'], num_arms,
        len(full_model_input['bandits_info']), num_agents, SPECIAL
    )
    full_seq_len = original_tensors['bandit_ids'].shape[1]
    original_tensors['pad_mask'] = torch.zeros(1, full_seq_len, dtype=torch.bool)
    for k in original_tensors:
        original_tensors[k] = original_tensors[k].to(device)

    # Derive segment_order ("watch-first" vs "play-first") from the first
    # non-structural segment; for human data this is always "watch-first".
    first_non_structural = None
    for seg in segment_info:
        if seg['type'] in ('demo', 'learner'):
            first_non_structural = seg['type']
            break
    segment_order = "watch-first" if first_non_structural == "demo" else "play-first"

    # Map each demo/learner segment index to the game_index it represents,
    # so we can route per-segment predictions back to per-game rows, and
    # record the (start_pos, end_pos) slice of each learner segment for
    # Thompson splicing.
    seg_idx_to_game_idx = {}
    learner_seg_slices = {}  # seg_idx -> (start_pos, end_pos)
    demo_learner_idx = 0
    for seg_idx, seg in enumerate(segment_info):
        if seg['type'] in ('demo', 'learner'):
            g_idx = game_indices[demo_learner_idx]
            seg_idx_to_game_idx[seg_idx] = g_idx
            if seg['type'] == 'learner':
                start_pos = segment_boundaries[seg_idx]
                end_pos = segment_boundaries[seg_idx + 1]
                learner_seg_slices[seg_idx] = (start_pos, end_pos)
            demo_learner_idx += 1

    # Precompute each learner segment's true bandit payoffs (from session_df)
    # so Thompson sampling can generate rewards against the real task.
    learner_seg_bandit_payoffs = {}
    for seg_idx, g_idx in seg_idx_to_game_idx.items():
        if seg_idx in learner_seg_slices:
            g_df = session_df[session_df['game_index'] == g_idx]
            learner_seg_bandit_payoffs[seg_idx] = parse_bandit_payoffs(
                g_df.iloc[0]['zone_bandit']
            )

    # Per-game reward summaries observed in the experiment CSV. Two parallel
    # columns are maintained:
    #   * human_segment_total_reward    — observed 0/1 sum (luck-dependent)
    #   * human_segment_expected_reward — sum of the TRUE payoff of the arm
    #     chosen on each trial (decision quality without outcome noise)
    # Used on demo cutoffs (always) and learner cutoffs (human mode).
    # Model-mode learner cutoffs override these with the spliced Thompson
    # trajectory values.
    human_segment_total_reward: Dict[int, float] = {}
    human_segment_expected_reward: Dict[int, float] = {}
    seg_true_payoffs: Dict[int, List[float]] = {}
    for seg_idx, g_idx in seg_idx_to_game_idx.items():
        g_df = session_df[session_df['game_index'] == g_idx]
        human_segment_total_reward[seg_idx] = float(g_df['reward_generated'].sum())
        payoffs = parse_bandit_payoffs(g_df.iloc[0]['zone_bandit'])
        seg_true_payoffs[seg_idx] = payoffs
        human_segment_expected_reward[seg_idx] = float(
            sum(payoffs[int(pos)] for pos in g_df['position'])
        )

    def _denorm_demo(alpha_demo: torch.Tensor, beta_demo: torch.Tensor):
        if normalizer is not None:
            alpha_demo = normalizer.denormalize_alpha(alpha_demo)
            beta_demo = normalizer.denormalize_beta(beta_demo)
        return alpha_demo, beta_demo

    def _n_tot(alpha_row: torch.Tensor, beta_row: torch.Tensor) -> float:
        return float(
            (alpha_row.sum() + beta_row.sum() - alpha0 * K - beta0 * K).item()
        )

    def _run_forward(tensors_state, mask_cutoff: int) -> Dict[str, torch.Tensor]:
        mask = create_segment_mask(
            segment_boundaries,
            mask_cutoff,
            full_seq_len,
            device,
            masking_strategy=masking_strategy,
            segment_info=segment_info,
            round_bandit_map=full_model_input['round_bandit_map'],
        )
        with torch.no_grad():
            model.eval()
            return model(tensors_state, custom_attn_mask=mask)

    def _most_recent_demo_agent_idx(upper_inclusive: int) -> Optional[int]:
        for idx in range(upper_inclusive, -1, -1):
            prior = segment_info[idx]
            if prior['type'] == 'demo':
                return prior['agent'] - 1
        return None

    def _choose_thompson_betas(pre_alpha_theta, pre_beta_theta,
                               pre_alpha_demo, pre_beta_demo, seg_idx):
        """Pick (alpha, beta) the Thompson sampler should draw from."""
        if beta_method == "theta_betas":
            return pre_alpha_theta, pre_beta_theta
        if beta_method == "novice_betas":
            return pre_alpha_demo[novice_demo_idx, :], pre_beta_demo[novice_demo_idx, :]
        if beta_method == "expert_betas":
            return pre_alpha_demo[expert_demo_idx, :], pre_beta_demo[expert_demo_idx, :]
        if beta_method == "recent_betas":
            recent_idx = _most_recent_demo_agent_idx(seg_idx - 1)
            if recent_idx is not None:
                return (pre_alpha_demo[recent_idx, :], pre_beta_demo[recent_idx, :])
            return pre_alpha_theta, pre_beta_theta
        raise ValueError(f"Unknown beta_method: {beta_method}")

    per_game_rows = []
    per_cutoff_rows = []

    for rollout_id in range(num_rollouts):
        # Per-(session, rollout) RNGs so sessions are statistically independent
        # while the same --seed still produces byte-identical CSVs across runs.
        rollout_seed = _rollout_seed(seed, str(session_id_val), rollout_id)
        rng = random.Random(rollout_seed)
        np_rng = np.random.default_rng(rollout_seed)

        # Fresh tensors per rollout so Thompson splices never leak across rollouts.
        full_tensors = {k: v.clone() for k, v in original_tensors.items()}

        # Per-game bookkeeping populated during the cutoff sweep.
        game_pred_cache = {}  # game_idx -> dict(theta/expert/novice strings, round_idx)
        learner_trajectories = {}  # game_idx -> list of (action, reward)

        for cutoff in range(1, num_segments + 1):
            seg_idx = cutoff - 1
            seg = segment_info[seg_idx]
            seg_type = seg['type']
            round_idx = seg['round']

            # Model-mode learner segment: pre-pass → Thompson → splice,
            # BEFORE the recorded forward pass runs on the updated tensors.
            if learner_segment_input == "model" and seg_type == "learner":
                pre_mask_cutoff = cutoff - 1
                if pre_mask_cutoff >= 1:
                    pre_outputs = _run_forward(full_tensors, pre_mask_cutoff)
                    current_round_idx = full_model_input['round_to_index'][round_idx]
                    pre_alpha_theta = pre_outputs['alpha_theta'][0, current_round_idx]
                    pre_beta_theta = pre_outputs['beta_theta'][0, current_round_idx]
                    pre_alpha_demo_raw = pre_outputs['alpha_demo'][0, current_round_idx]
                    pre_beta_demo_raw = pre_outputs['beta_demo'][0, current_round_idx]
                    pre_alpha_demo, pre_beta_demo = _denorm_demo(
                        pre_alpha_demo_raw, pre_beta_demo_raw
                    )

                    pred_alpha, pred_beta = _choose_thompson_betas(
                        pre_alpha_theta, pre_beta_theta,
                        pre_alpha_demo, pre_beta_demo,
                        seg_idx,
                    )

                    # Match the actual number of events the human played in this
                    # segment rather than the model-config default — human play
                    # segments vary in length (1..T_learner) and the spliced
                    # trajectory must fit the tensor slots exactly.
                    start_pos, end_pos = learner_seg_slices[seg_idx]
                    seg_len = end_pos - start_pos

                    trajectory = thompson_sampling_step(
                        pred_alpha.detach().cpu(),
                        pred_beta.detach().cpu(),
                        learner_seg_bandit_payoffs[seg_idx],
                        seg_len,
                        rng=rng,
                        np_rng=np_rng,
                    )
                    for step_idx, (action_idx, reward) in enumerate(trajectory):
                        pos = start_pos + step_idx
                        action_val = action_idx if (0 <= action_idx < num_arms) else num_arms
                        reward_val = int(reward) if int(reward) in (0, 1) else 2
                        full_tensors['action_ids'][0, pos] = action_val
                        full_tensors['reward_ids'][0, pos] = reward_val

                    learner_trajectories[seg_idx_to_game_idx[seg_idx]] = trajectory

            # Recorded (post-segment) forward pass on the current tensor state.
            outputs = _run_forward(full_tensors, cutoff)
            current_round_idx = full_model_input['round_to_index'][round_idx]

            post_alpha_theta = outputs['alpha_theta'][0, current_round_idx]
            post_beta_theta = outputs['beta_theta'][0, current_round_idx]
            post_alpha_demo_raw = outputs['alpha_demo'][0, current_round_idx]
            post_beta_demo_raw = outputs['beta_demo'][0, current_round_idx]
            post_alpha_demo, post_beta_demo = _denorm_demo(
                post_alpha_demo_raw, post_beta_demo_raw
            )

            theta_betas_str = serialize_betas(post_alpha_theta, post_beta_theta)
            expert_betas_str = serialize_betas(
                post_alpha_demo[expert_demo_idx, :], post_beta_demo[expert_demo_idx, :]
            )
            novice_betas_str = serialize_betas(
                post_alpha_demo[novice_demo_idx, :], post_beta_demo[novice_demo_idx, :]
            )
            recent_demo_idx = _most_recent_demo_agent_idx(seg_idx)
            if recent_demo_idx is not None:
                recent_betas_str = serialize_betas(
                    post_alpha_demo[recent_demo_idx, :],
                    post_beta_demo[recent_demo_idx, :],
                )
            else:
                recent_betas_str = theta_betas_str

            true_payoffs_list = seg_true_payoffs.get(seg_idx)
            true_payoffs_str = (
                ','.join(f"{p:.4f}" for p in true_payoffs_list)
                if true_payoffs_list is not None
                else ''
            )

            expert_n_tot = _n_tot(
                post_alpha_demo[expert_demo_idx, :].detach().cpu(),
                post_beta_demo[expert_demo_idx, :].detach().cpu(),
            )
            novice_n_tot = _n_tot(
                post_alpha_demo[novice_demo_idx, :].detach().cpu(),
                post_beta_demo[novice_demo_idx, :].detach().cpu(),
            )

            # Per-game reward summaries for the game ending at this cutoff.
            # Two parallel columns: segment_total_reward (observed 0/1 sum)
            # and segment_expected_reward (sum of true payoffs of chosen
            # arms). Demo cutoffs always come from session_df. Learner
            # cutoffs come from session_df in human mode and from the
            # spliced Thompson trajectory in model mode. Non-game cutoffs
            # (bandit_sep, query) get NaN.
            if seg_type == "demo":
                segment_total_reward = human_segment_total_reward[seg_idx]
                segment_expected_reward = human_segment_expected_reward[seg_idx]
            elif seg_type == "learner":
                if learner_segment_input == "human":
                    segment_total_reward = human_segment_total_reward[seg_idx]
                    segment_expected_reward = human_segment_expected_reward[seg_idx]
                else:
                    g_idx = seg_idx_to_game_idx[seg_idx]
                    traj = learner_trajectories.get(g_idx)
                    if traj:
                        segment_total_reward = float(
                            sum(int(r) for _, r in traj)
                        )
                        seg_payoffs = learner_seg_bandit_payoffs[seg_idx]
                        segment_expected_reward = float(
                            sum(float(seg_payoffs[a]) for a, _ in traj)
                        )
                    else:
                        segment_total_reward = float('nan')
                        segment_expected_reward = float('nan')
            else:
                segment_total_reward = float('nan')
                segment_expected_reward = float('nan')

            per_cutoff_rows.append({
                'session_id': session_id_val,
                'session_index': session_index_val,
                'anonymous_id': anonymous_id_val,
                'rollout_id': rollout_id,
                'learner_segment_input': learner_segment_input,
                'beta_method': beta_method,
                'cutoff': cutoff,
                'segment_type': seg_type,
                'segment_order': segment_order,
                'game_sequence': game_seq,
                'round_index': int(round_idx),
                'expert_N_tot_pred': expert_n_tot,
                'novice_N_tot_pred': novice_n_tot,
                'segment_total_reward': segment_total_reward,
                'segment_expected_reward': segment_expected_reward,
                'theta_betas': theta_betas_str,
                'expert_betas': expert_betas_str,
                'novice_betas': novice_betas_str,
                'recent_betas': recent_betas_str,
                'true_payoffs': true_payoffs_str,
            })

            if seg_idx in seg_idx_to_game_idx:
                game_pred_cache[seg_idx_to_game_idx[seg_idx]] = {
                    'theta_betas': theta_betas_str,
                    'expert_betas': expert_betas_str,
                    'novice_betas': novice_betas_str,
                    'round_idx': int(round_idx),
                }

        # Emit per-game / per-step rows using the cached per-segment predictions.
        for game_idx in game_indices:
            game_df = session_df[session_df['game_index'] == game_idx]
            preds = game_pred_cache.get(game_idx)
            theta_str = preds['theta_betas'] if preds else np.nan
            expert_str = preds['expert_betas'] if preds else np.nan
            novice_str = preds['novice_betas'] if preds else np.nan
            round_idx_for_game = preds['round_idx'] if preds else int(game_df.iloc[0]['round_index'])

            if game_idx in demo_games:
                for _, row in game_df.iterrows():
                    row_dict = row.to_dict()
                    row_dict['rollout_id'] = rollout_id
                    row_dict['learner_segment_input'] = learner_segment_input
                    row_dict['beta_method'] = beta_method
                    row_dict['theta_betas'] = theta_str
                    row_dict['expert_betas'] = expert_str
                    row_dict['novice_betas'] = novice_str
                    per_game_rows.append(row_dict)

            elif game_idx in learner_games:
                if learner_segment_input == "human":
                    for _, row in game_df.iterrows():
                        row_dict = row.to_dict()
                        row_dict['rollout_id'] = rollout_id
                        row_dict['learner_segment_input'] = learner_segment_input
                        row_dict['beta_method'] = beta_method
                        row_dict['theta_betas'] = theta_str
                        row_dict['expert_betas'] = expert_str
                        row_dict['novice_betas'] = novice_str
                        per_game_rows.append(row_dict)
                else:
                    trajectory = learner_trajectories.get(game_idx, [])
                    current_zone = int(game_df.iloc[0]['zone_global_index'])
                    for step_idx, (action_idx, reward) in enumerate(trajectory):
                        per_game_rows.append({
                            'rollout_id': rollout_id,
                            'session_id': session_id_val,
                            'session_index': session_index_val,
                            'anonymous_id': anonymous_id_val,
                            'position': action_idx,
                            'action': 'shoot',
                            'game': 'play',
                            'game_sequence': game_seq,
                            'game_index': game_idx,
                            'round_index': round_idx_for_game,
                            'demonstrator_type': np.nan,
                            'zone_bandit': game_df.iloc[0]['zone_bandit'],
                            'zone_global_index': current_zone,
                            'reward_generated': bool(reward),
                            'learner_segment_input': learner_segment_input,
                            'beta_method': beta_method,
                            'theta_betas': theta_str,
                            'expert_betas': expert_str,
                            'novice_betas': novice_str,
                        })

    return pd.DataFrame(per_game_rows), pd.DataFrame(per_cutoff_rows)



def main():
    parser = argparse.ArgumentParser(description="Run model inference on human experiment data to generate hybrid predictions")
    parser.add_argument("--experiment_data_path",type=str,required=True,help="Path to experiment CSV file")
    parser.add_argument("--output_path",type=str,default="data/model_predictions/experiment_trial_predictions.csv",help="Path to output CSV file")
    parser.add_argument("--device",type=str,default="cpu",choices=["cpu", "cuda"],help="Device to run inference on")
    parser.add_argument("--num_rollouts", type=int, default=5,
        help="Number of independent rollouts per session (default: 5)")
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
    
    # Set up eval directory structure (all inference artifacts now live under eval/)
    eval_dir = setup_eval_directory()
    print(f"Output directory: {eval_dir}")

    # Write run metadata under eval/metadata/ with an inference_ prefix so it
    # doesn't collide with eval.py's run.yaml / config.yaml.
    metadata = {
        'script': 'run_inference.py',
        'command': ' '.join(sys.argv),
        'experiment_data_path': args.experiment_data_path,
        'num_rollouts': args.num_rollouts,
        'seed': args.seed,
    }
    write_run_yaml(eval_dir, metadata, subdir='metadata', filename='inference_run.yaml')
    write_config_yaml(eval_dir, vars(args), subdir='metadata', filename='inference_config.yaml')

    # Update output path to write the per-trial CSV into eval/data/ if still default
    if args.output_path == "data/model_predictions/experiment_trial_predictions.csv":
        args.output_path = str(eval_dir / 'data' / 'experiment_trial_predictions.csv')
    
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

        # Mirror eval.py: if the checkpoint normalized beliefs during training,
        # wire up a BeliefNormalizer so inference denormalizes before computing
        # N_tot. Without this the per-cutoff experience numbers would be in
        # normalized space and meaningless.
        normalizer = None
        if checkpoint['normalization'].get('normalize_beliefs', False):
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
                beta0=norm_dict['beta0'],
            )
            print("✓ Using belief normalization")

        # Run inference on each session for all 8 combinations
        print(f"\nProcessing {df['session_id'].nunique()} sessions with {args.num_rollouts} rollout(s) x 8 combinations...")

        per_game_results = []
        per_cutoff_results = []

        for learner_input in tqdm(["human", "model"], desc="Learner input mode", position=0):
            for beta_method in tqdm(["theta_betas", "novice_betas", "expert_betas", "recent_betas"],
                                    desc=f"Beta method ({learner_input})", position=1, leave=False):

                session_ids = df['session_id'].unique()
                for session_id in tqdm(session_ids,
                                      desc=f"Sessions ({learner_input}, {beta_method})",
                                      position=2, leave=False):
                    session_df = df[df['session_id'] == session_id].copy()

                    per_game_df, per_cutoff_df = run_inference_on_session(
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
                        masking_strategy=masking_strategy,
                        normalizer=normalizer,
                    )

                    per_game_results.append(per_game_df)
                    per_cutoff_results.append(per_cutoff_df)

        print("\nCombining all results...")
        final_per_game_df = pd.concat(per_game_results, ignore_index=True)
        final_per_cutoff_df = pd.concat(per_cutoff_results, ignore_index=True)

        # experiment_trial_predictions.csv — per-game / per-step rows
        # (consumer: analysis_report.qmd, hybrid_model_pipeline_eval.qmd).
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving per-trial predictions to {output_path}...")
        final_per_game_df.to_csv(output_path, index=False)

        # Per-cutoff predictions split by learner mode, also under eval/data/
        # (consumer: eval_report.qmd N_total comparison plot).
        eval_data_dir = eval_dir / 'data'
        eval_data_dir.mkdir(parents=True, exist_ok=True)

        human_cutoff_df = final_per_cutoff_df[final_per_cutoff_df['learner_segment_input'] == 'human']
        model_cutoff_df = final_per_cutoff_df[final_per_cutoff_df['learner_segment_input'] == 'model']

        human_cutoff_path = eval_data_dir / 'experiment_cutoff_human_predictions.csv'
        model_cutoff_path = eval_data_dir / 'experiment_cutoff_model_predictions.csv'
        print(f"Saving human-mode per-cutoff predictions to {human_cutoff_path}...")
        human_cutoff_df.to_csv(human_cutoff_path, index=False)
        print(f"Saving model-mode per-cutoff predictions to {model_cutoff_path}...")
        model_cutoff_df.to_csv(model_cutoff_path, index=False)

        print(f"\nDone! Per-trial rows: {len(final_per_game_df)}, per-cutoff rows: {len(final_per_cutoff_df)}")
        print(f"  Sessions: {final_per_game_df['session_id'].nunique()}")
        print(f"  - 8 combinations x {args.num_rollouts} rollouts = {8 * args.num_rollouts} runs per session")
        print(f"Per-trial output: {output_path}")
        print(f"Per-cutoff outputs: {human_cutoff_path}, {model_cutoff_path}")

        finalize_run_yaml(eval_dir, status='completed', subdir='metadata', filename='inference_run.yaml')
        print(f"\n{'='*60}")
        print("Inference completed successfully!")
        print(f"{'='*60}")
        
    except KeyboardInterrupt:
        print(f"\n{'='*60}")
        print("Inference interrupted by user")
        print(f"{'='*60}")
        finalize_run_yaml(eval_dir, status='interrupted', subdir='metadata', filename='inference_run.yaml')
        raise
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"Inference failed with error: {e}")
        print(f"{'='*60}")
        finalize_run_yaml(eval_dir, status='failed', subdir='metadata', filename='inference_run.yaml', error=str(e))
        raise


if __name__ == "__main__":
    main()

