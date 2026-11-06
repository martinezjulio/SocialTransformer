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

from data.transforms import BeliefNormalizer
from inference.learners import ThompsonLearner, TrialState
from inference.readout import load_model, readout
from inference.sessions import (convert_session_to_model_format, encode_events_to_tensors,
                                load_shots, parse_bandit_payoffs, validate_sessions)
from inference.simulate import play_segment
from utils import setup_eval_directory, get_training_runs_dir, write_run_yaml, write_config_yaml, finalize_run_yaml, get_special_characters




def parse_game_sequence(game_seq: str) -> List[str]:
    """
    Parse game sequence string to extract demonstrator order.
    
    Args:
        game_seq: e.g., "novice-play-expert-play-choice" or "expert-play-novice-play-choice"
        
    Returns:
        List of segment types in order, e.g., ['novice', 'play', 'expert', 'play', 'choice']
    """
    return game_seq.split('-')




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


def serialize_thetas(theta: torch.Tensor) -> str:
    """
    Serialize a [K] tensor of per-arm theta point estimates to a string list.

    Args:
        theta: [K] tensor of expected payoffs (one per arm)

    Returns:
        String like "[t1, t2, ...]"
    """
    return str([float(t) for t in theta.cpu()])


def _rollout_seed(base_seed: int, session_id: str, rollout_id: int) -> int:
    # Stable hash (SHA-256) avoids PYTHONHASHSEED non-determinism across processes.
    digest = hashlib.sha256(f"{session_id}|{rollout_id}".encode()).digest()
    return (base_seed + int.from_bytes(digest[:4], "big")) % (2**31)




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
    own_experience: str = "carry",
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

    segment_round = [seg['round'] for seg in segment_info]

    def _run_forward(tensors_state, mask_cutoff: int) -> Dict[str, torch.Tensor]:
        return readout(model, tensors_state, segment_boundaries, [mask_cutoff], masking_strategy,
                       segment_round, full_model_input['round_bandit_map'])

    def _most_recent_demo_agent_idx(upper_inclusive: int) -> Optional[int]:
        for idx in range(upper_inclusive, -1, -1):
            prior = segment_info[idx]
            if prior['type'] == 'demo':
                return prior['agent'] - 1
        return None

    per_game_rows = []
    per_cutoff_rows = []

    # The readout is a belief about the DEMONSTRATOR, so it carries no record of what the
    # learner itself earned. Without an explicit tally the learner would forget its own
    # outcomes at every segment boundary. "carry" accumulates them across a round's two play
    # segments; "reset" clears at each segment start, reproducing the pre-fix behaviour.
    # Counts are always keyed by round: payoffs are redrawn per round, so nothing carries
    # between rounds.
    own_experience_label = (
        own_experience if learner_segment_input == "model" else "not_applicable"
    )

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
        own_counts = {}  # round_idx -> [successes, failures] over arms, this rollout

        for cutoff in range(1, num_segments + 1):
            seg_idx = cutoff - 1
            seg = segment_info[seg_idx]
            seg_type = seg['type']
            round_idx = seg['round']
            thompson_betas_str = ''   # set below on model-mode learner cutoffs

            # Model-mode learner segment: pre-pass → Thompson → splice,
            # BEFORE the recorded forward pass runs on the updated tensors.
            if learner_segment_input == "model" and seg_type == "learner":
                pre_mask_cutoff = cutoff - 1
                if pre_mask_cutoff >= 1:
                    pre_outputs = _run_forward(full_tensors, pre_mask_cutoff)
                    current_round_idx = full_model_input['round_to_index'][round_idx]
                    if model.env_theta_loss == "bce":
                        pre_alpha_theta = pre_beta_theta = None
                    else:
                        pre_alpha_theta = pre_outputs['alpha_theta'][0, current_round_idx]
                        pre_beta_theta = pre_outputs['beta_theta'][0, current_round_idx]
                    pre_alpha_demo_raw = pre_outputs['alpha_demo'][0, current_round_idx]
                    pre_beta_demo_raw = pre_outputs['beta_demo'][0, current_round_idx]
                    pre_alpha_demo, pre_beta_demo = _denorm_demo(
                        pre_alpha_demo_raw, pre_beta_demo_raw
                    )

                    # Own experience on this round's bandit: "carry" keeps both play games, "reset" this one.
                    acc = own_counts.setdefault(int(round_idx), [np.zeros(num_arms), np.zeros(num_arms)])
                    if own_experience == "reset":
                        acc[0][:] = 0.0
                        acc[1][:] = 0.0
                    learner = ThompsonLearner(beta_method, own_experience, np_rng)
                    state = TrialState(pre_alpha_demo, pre_beta_demo, pre_alpha_theta, pre_beta_theta,
                                       expert_demo_idx, novice_demo_idx, _most_recent_demo_agent_idx(seg_idx - 1),
                                       torch.zeros(2, num_arms), torch.as_tensor(np.stack(acc)), round_index=int(round_idx))
                    thompson_betas_str = serialize_betas(*learner.belief(state))
                    # The human's slots are refilled exactly: play segments vary in length (1..T_learner).
                    start_pos, end_pos = learner_seg_slices[seg_idx]
                    trajectory = play_segment(learner, state, full_tensors, start_pos, end_pos,
                                              learner_seg_bandit_payoffs[seg_idx], rng, np_rng)
                    acc[0] += state.own_within[0].numpy()
                    acc[1] += state.own_within[1].numpy()
                    learner_trajectories[seg_idx_to_game_idx[seg_idx]] = trajectory

            # Recorded (post-segment) forward pass on the current tensor state.
            outputs = _run_forward(full_tensors, cutoff)
            current_round_idx = full_model_input['round_to_index'][round_idx]

            post_alpha_demo_raw = outputs['alpha_demo'][0, current_round_idx]
            post_beta_demo_raw = outputs['beta_demo'][0, current_round_idx]
            post_alpha_demo, post_beta_demo = _denorm_demo(
                post_alpha_demo_raw, post_beta_demo_raw
            )

            # BCE env head: point estimate only (no Beta posterior → theta_betas empty).
            # beta_nll: full posterior in theta_betas, plus its expected value.
            if model.env_theta_loss == "bce":
                theta_betas_str = ''
                theta_predicted = torch.sigmoid(outputs['theta_logits'][0, current_round_idx])
            else:
                post_alpha_theta = outputs['alpha_theta'][0, current_round_idx]
                post_beta_theta = outputs['beta_theta'][0, current_round_idx]
                theta_betas_str = serialize_betas(post_alpha_theta, post_beta_theta)
                theta_predicted = post_alpha_theta / (post_alpha_theta + post_beta_theta)
            theta_predicted_str = serialize_thetas(theta_predicted)
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
                'own_experience': own_experience_label,
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
                'theta_predicted': theta_predicted_str,
                'expert_betas': expert_betas_str,
                'novice_betas': novice_betas_str,
                'recent_betas': recent_betas_str,
                'thompson_betas': thompson_betas_str,
                'true_payoffs': true_payoffs_str,
            })

            if seg_idx in seg_idx_to_game_idx:
                game_pred_cache[seg_idx_to_game_idx[seg_idx]] = {
                    'theta_betas': theta_betas_str,
                    'theta_predicted': theta_predicted_str,
                    'expert_betas': expert_betas_str,
                    'novice_betas': novice_betas_str,
                    'round_idx': int(round_idx),
                }

        # Emit per-game / per-step rows using the cached per-segment predictions.
        for game_idx in game_indices:
            game_df = session_df[session_df['game_index'] == game_idx]
            preds = game_pred_cache.get(game_idx)
            theta_str = preds['theta_betas'] if preds else np.nan
            theta_pred_str = preds['theta_predicted'] if preds else np.nan
            expert_str = preds['expert_betas'] if preds else np.nan
            novice_str = preds['novice_betas'] if preds else np.nan
            round_idx_for_game = preds['round_idx'] if preds else int(game_df.iloc[0]['round_index'])

            if game_idx in demo_games:
                for _, row in game_df.iterrows():
                    row_dict = row.to_dict()
                    row_dict['rollout_id'] = rollout_id
                    row_dict['learner_segment_input'] = learner_segment_input
                    row_dict['beta_method'] = beta_method
                    row_dict['own_experience'] = own_experience_label
                    row_dict['theta_betas'] = theta_str
                    row_dict['theta_predicted'] = theta_pred_str
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
                        row_dict['own_experience'] = own_experience_label
                        row_dict['theta_betas'] = theta_str
                        row_dict['theta_predicted'] = theta_pred_str
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
                            'own_experience': own_experience_label,
                            'theta_betas': theta_str,
                            'theta_predicted': theta_pred_str,
                            'expert_betas': expert_str,
                            'novice_betas': novice_str,
                        })

    return pd.DataFrame(per_game_rows), pd.DataFrame(per_cutoff_rows)



def main():
    parser = argparse.ArgumentParser(description="Run model inference on human experiment data to generate hybrid predictions")
    parser.add_argument("--experiment_data_path",type=str,required=True,help="Path to experiment CSV file")
    parser.add_argument("--output_path",type=str,default="data/model_predictions/experiment_trial_predictions.csv",help="Path to output CSV file")
    parser.add_argument("--eval_dir", type=str, default=None,
        help="Directory for the CSVs (data/) and metadata yamls (metadata/); default "
             "$OUTPUT_DATA_DIR/eval/. Give a per-pilot directory, e.g. $OUTPUT_DATA_DIR/eval/pilot_E, "
             "so pilots do not overwrite each other")
    parser.add_argument("--device",type=str,default="cpu",choices=["cpu", "cuda"],help="Device to run inference on")
    parser.add_argument("--num_rollouts", type=int, default=5,
        help="Number of independent rollouts per session (default: 5)")
    parser.add_argument("--seed", type=int, default=42,
        help="Base random seed for reproducibility (default: 42)")
    parser.add_argument("--own_experience", type=str, default="both",
        choices=["carry", "reset", "both"],
        help="How the simulated learner treats its own successes/failures on a round's bandit: "
             "'carry' accumulates them across the round's two play segments (plain Bayes), "
             "'reset' clears them at each segment boundary, 'both' runs each as a separate "
             "pass tagged in the own_experience column (default: both)")
    parser.add_argument("--masking_strategy_override", type=str, default=None,
        choices=["all_bandits", "current_bandit_only"],
        help="Override masking strategy from checkpoint (optional)")

    # Checkpoint loading options (mutually exclusive)
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument("--checkpoint_path", type=str,
        help="Path to local checkpoint file")
    checkpoint_group.add_argument("--wandb_artifact", type=str,
        help="Load from wandb artifact: 'entity/project/artifact:alias' (e.g., 'user/project/model-checkpoint:best')")
    
    args = parser.parse_args()
    
    # Set up eval directory structure (all inference artifacts now live under eval/)
    if args.eval_dir is None:
        eval_dir = setup_eval_directory()
    else:
        eval_dir = Path(args.eval_dir)
        for subdir in ['data', 'metadata']:
            (eval_dir / subdir).mkdir(parents=True, exist_ok=True)
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
        df = load_shots(args.experiment_data_path)
        validate_sessions(df)
        df = df.drop(columns=['step', 'timestamp'], errors='ignore')   # ordering columns only (no timestamp in the public release)
        
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
        
        model, checkpoint, normalizer = load_model(args.checkpoint_path, args.wandb_artifact,
                                                   str(get_training_runs_dir()), args.device)
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

        # Pull masking strategy + profile method from the checkpoint.
        masking_strategy = model_config.get('masking_strategy', 'all_bandits')
        model_type = model_config.get('model_type') or model_config.get('profile_method', 'social')

        if args.masking_strategy_override:
            masking_strategy = args.masking_strategy_override
            print(f"  masking_strategy: {masking_strategy} (override)")
        else:
            print(f"  masking_strategy: {masking_strategy}")

        print(f"  model_type: {model_type}")
        print(f"  experience_head_loss_type: {model_config.get('experience_head_loss_type', 'mse')}")
        print("Model loaded successfully!")
        
        if 'normalization' not in checkpoint:
            raise ValueError("Checkpoint missing 'normalization' parameters")

        alpha0 = checkpoint['normalization']['alpha0']
        beta0 = checkpoint['normalization']['beta0']
        print(f"  alpha0={alpha0}, beta0={beta0}")


        # Thompson beta methods. theta_betas needs a Beta posterior over theta,
        # which a BCE environment head does not produce — drop it for those.
        beta_methods = ["theta_betas", "novice_betas", "expert_betas", "recent_betas"]
        if model.env_theta_loss == "bce":
            beta_methods = [m for m in beta_methods if m != "theta_betas"]

        # Own-experience variants to run. "both" keeps them in one flat output directory,
        # separated by the own_experience column, so neither overwrites the other.
        own_variants = (["carry", "reset"] if args.own_experience == "both"
                        else [args.own_experience])

        # Run inference on each session for all combinations
        print(f"\nProcessing {df['session_id'].nunique()} sessions — "
              f"human: 1 pass; model: {len(beta_methods)} methods x {len(own_variants)} "
              f"own-experience variant(s) x {args.num_rollouts} rollouts")

        per_game_results = []
        per_cutoff_results = []

        for learner_input in tqdm(["human", "model"], desc="Learner input mode", position=0):
            # Human mode replays the participant's recorded actions — no Thompson
            # sampling, so beta_method is never consulted and every rollout is
            # identical. Run it once and label it. The label is spelled out rather
            # than "n/a" because pandas.read_csv parses "n/a" back as NaN.
            combos = ([(m, v, args.num_rollouts) for m in beta_methods for v in own_variants]
                      if learner_input == "model"
                      else [("not_applicable", "not_applicable", 1)])
            for beta_method, own_exp, n_rollouts in tqdm(combos,
                                    desc=f"Beta method ({learner_input})", position=1, leave=False):

                session_ids = df['session_id'].unique()
                for session_id in tqdm(session_ids,
                                      desc=f"Sessions ({learner_input}, {beta_method}, {own_exp})",
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
                        own_experience=own_exp,
                        num_rollouts=n_rollouts,
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
        print(f"  - per session: 1 human pass + {len(beta_methods) * args.num_rollouts} model passes")
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

