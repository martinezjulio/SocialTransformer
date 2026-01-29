#!/usr/bin/env python3
"""
Prior Experience Analysis for Social Learning in Thompson Sampling

This script analyzes whether observing an expert Thompson sampler with prior 
experience yields better learning outcomes than just playing the bandit for 
the same total number of steps.

Compares two conditions:
1. Social Learning: Observe expert for T_d steps → play for T_p steps
2. Individual Learning: Play bandit for (T_d + T_p) steps
"""

import numpy as np
import random
import pickle
import json
import sys
from collections import defaultdict
from scipy.stats import beta as beta_dist
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import argparse
from pathlib import Path
from utils import get_run_id, setup_run_directory, write_run_yaml, write_config_yaml, finalize_run_yaml


# ============================================================================
# Thompson Sampling Primitives
# ============================================================================

def thompson_sample(alphas: np.ndarray, betas: np.ndarray) -> int:
    """
    Sample from Beta(alpha, beta) for each arm and return argmax.
    
    Args:
        alphas: Array of shape [K] with alpha parameters
        betas: Array of shape [K] with beta parameters
    
    Returns:
        Arm index with highest sampled value
    """
    K = len(alphas)
    draws = np.array([beta_dist.rvs(alphas[k], betas[k]) for k in range(K)])
    return int(np.argmax(draws))


def greedy_select(alphas: np.ndarray, betas: np.ndarray) -> int:
    """
    Greedy arm selection: pick arm with highest expected payoff.
    
    Args:
        alphas: Array of shape [K] with alpha parameters
        betas: Array of shape [K] with beta parameters
    
    Returns:
        Arm index with highest expected payoff (alpha / (alpha + beta))
    """
    expected_payoffs = alphas / (alphas + betas)
    return int(np.argmax(expected_payoffs))


def pull_arm(arm: int, thetas: List[float]) -> int:
    """
    Pull an arm and get reward from Bernoulli(theta[arm]).
    
    Args:
        arm: Arm index to pull
        thetas: True payoff probabilities for each arm
    
    Returns:
        Reward (0 or 1)
    """
    return 1 if random.random() < thetas[arm] else 0


def generate_bandit(payoffs: List[float] = None) -> List[float]:
    """
    Generate a bandit by randomly permuting fixed payoff values.
    
    Args:
        payoffs: Fixed list of payoff values to permute
    
    Returns:
        Random permutation of payoff values
    """
    if payoffs is None:
        payoffs = [0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.75, 0.8]
    
    thetas = payoffs.copy()
    random.shuffle(thetas)
    return thetas


def run_thompson_sampling(
    thetas: List[float],
    alpha_init: np.ndarray,
    beta_init: np.ndarray,
    steps: int,
    record_at: List[int] = None
) -> Tuple[List[int], List[int], np.ndarray, np.ndarray, Dict[int, float]]:
    """
    Run Thompson sampling for a given number of steps.
    
    Args:
        thetas: True payoff probabilities for each arm
        alpha_init: Initial alpha parameters [K]
        beta_init: Initial beta parameters [K]
        steps: Number of steps to run
        record_at: List of step numbers to record cumulative rewards at
    
    Returns:
        actions: List of actions taken
        rewards: List of rewards received
        alphas: Final alpha parameters
        betas: Final beta parameters
        recorded_rewards: Dict mapping step numbers to cumulative rewards
    """
    K = len(thetas)
    alphas = alpha_init.copy()
    betas = beta_init.copy()
    
    actions = []
    rewards = []
    cumulative_reward = 0
    recorded_rewards = {}
    
    if record_at is None:
        record_at = []
    
    for step in range(1, steps + 1):
        # Thompson sampling: sample from posterior and pick best
        arm = thompson_sample(alphas, betas)
        reward = pull_arm(arm, thetas)
        
        # Record
        actions.append(arm)
        rewards.append(reward)
        cumulative_reward += reward
        
        # Update beliefs
        alphas[arm] += reward
        betas[arm] += (1 - reward)
        
        # Record cumulative reward at specific steps
        if step in record_at:
            recorded_rewards[step] = cumulative_reward
    
    return actions, rewards, alphas, betas, recorded_rewards


# ============================================================================
# Expert Simulation
# ============================================================================

def simulate_expert(
    thetas: List[float],
    N: int,
    T_d: int,
    K: int = 12
) -> Tuple[List[int], List[int], np.ndarray, np.ndarray]:
    """
    Simulate an expert Thompson sampler with N latent steps + T_d observed steps.
    
    Args:
        thetas: True payoff probabilities
        N: Number of latent (unobserved) steps
        T_d: Number of observed demonstration steps
        K: Number of arms
    
    Returns:
        observed_actions: Actions from T_d observed steps
        observed_rewards: Rewards from T_d observed steps
        final_alphas: Expert's final beliefs after N + T_d steps
        final_betas: Expert's final beliefs after N + T_d steps
    """
    # Initialize expert with uniform prior
    alphas = np.ones(K)
    betas = np.ones(K)
    
    # Phase 1: N latent steps (unobserved)
    if N > 0:
        _, _, alphas, betas, _ = run_thompson_sampling(
            thetas, alphas, betas, N
        )
    
    # Phase 2: T_d observed steps
    observed_actions, observed_rewards, alphas, betas, _ = run_thompson_sampling(
        thetas, alphas, betas, T_d
    )
    
    return observed_actions, observed_rewards, alphas, betas


def estimate_priors_from_observations(
    actions: List[int],
    rewards: List[int],
    K: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute naive prior estimates from observed actions and rewards.
    
    Args:
        actions: List of observed actions
        rewards: List of observed rewards
        K: Number of arms
    
    Returns:
        alphas: Alpha parameters (1 + successes per arm)
        betas: Beta parameters (1 + failures per arm)
    """
    alphas = np.ones(K)
    betas = np.ones(K)
    
    for action, reward in zip(actions, rewards):
        alphas[action] += reward
        betas[action] += (1 - reward)
    
    return alphas, betas


# ============================================================================
# Experimental Conditions
# ============================================================================

def run_social_condition(
    thetas: List[float],
    N: int,
    T_d: int,
    T_p_values: List[int],
    K: int = 12
) -> Tuple[Dict[int, float], Dict[int, List[int]], Dict[int, float]]:
    """
    Run the social learning condition: observe expert → play bandit.
    
    Args:
        thetas: True payoff probabilities
        N: Expert's latent experience
        T_d: Number of observed demonstration steps
        T_p_values: List of play lengths to record (e.g., [2, 4, 8, 16])
        K: Number of arms
    
    Returns:
        recorded_rewards: Dict mapping T_p to cumulative rewards
        recorded_actions: Dict mapping T_p to list of actions taken
        recorded_step_rewards: Dict mapping T_p to reward at that specific step
    """
    # Step 1: Expert plays with N latent + T_d observed steps
    observed_actions, observed_rewards, _, _ = simulate_expert(thetas, N, T_d, K)
    
    # Step 2: Learner estimates priors from observations
    learner_alphas, learner_betas = estimate_priors_from_observations(
        observed_actions, observed_rewards, K
    )
    
    # Step 3: Learner plays for max(T_p_values) steps, recording at each T_p
    max_T_p = max(T_p_values)
    actions, rewards, _, _, recorded_rewards = run_thompson_sampling(
        thetas, learner_alphas, learner_betas, max_T_p, record_at=T_p_values
    )
    
    # Extract action sequences and step rewards at each T_p checkpoint
    recorded_actions = {}
    recorded_step_rewards = {}
    for T_p in T_p_values:
        recorded_actions[T_p] = actions[:T_p]  # First T_p actions
        recorded_step_rewards[T_p] = rewards[T_p - 1]  # Reward at step T_p (0-indexed)
    
    return recorded_rewards, recorded_actions, recorded_step_rewards


def run_individual_condition(
    thetas: List[float],
    T_d: int,
    T_p_values: List[int],
    K: int = 12
) -> Tuple[Dict[int, float], Dict[int, List[int]], Dict[int, float]]:
    """
    Run the individual learning condition: play bandit for T_d + T_p steps.
    
    Plays for T_d steps (learning phase, like observation) + T_p steps (performance phase).
    Only measures rewards/actions/entropy from the final T_p steps for fair comparison.
    
    Args:
        thetas: True payoff probabilities
        T_d: Learning phase length (to match demonstration length in social condition)
        T_p_values: List of play lengths to record
        K: Number of arms
    
    Returns:
        result: Dict mapping T_p values to cumulative rewards from final T_p steps only
        recorded_actions: Dict mapping T_p to list of actions from final T_p steps only
        recorded_step_rewards: Dict mapping T_p to reward at that specific step
    """
    # Initialize with uniform prior
    alphas = np.ones(K)
    betas = np.ones(K)
    
    # Play for T_d + max(T_p_values) steps
    max_T_p = max(T_p_values)
    total_steps = T_d + max_T_p
    
    actions, rewards, _, _, _ = run_thompson_sampling(
        thetas, alphas, betas, total_steps, record_at=[]
    )
    
    # Only measure the final T_p steps (after T_d learning phase)
    result = {}
    recorded_actions = {}
    recorded_step_rewards = {}
    for t_p in T_p_values:
        # Extract actions and rewards from the final T_p steps only
        start_idx = T_d
        end_idx = T_d + t_p
        final_actions = actions[start_idx:end_idx]
        final_rewards = rewards[start_idx:end_idx]
        
        # Cumulative reward over final T_p steps only
        result[t_p] = sum(final_rewards)
        
        # Actions from final T_p steps
        recorded_actions[t_p] = final_actions
        
        # Reward at timestep T_d + T_p (0-indexed, so index is T_d + T_p - 1)
        recorded_step_rewards[t_p] = rewards[T_d + t_p - 1]
    
    return result, recorded_actions, recorded_step_rewards


def run_greedy_social_condition(
    thetas: List[float],
    N: int,
    T_d: int,
    T_p_values: List[int],
    K: int = 12
) -> Tuple[Dict[int, float], Dict[int, List[int]], Dict[int, float]]:
    """
    Run greedy social learning: observe expert → play greedily.
    
    Uses naive priors from observations but selects arms greedily
    (highest expected payoff) instead of Thompson sampling.
    
    Args:
        thetas: True payoff probabilities
        N: Expert's latent experience
        T_d: Number of observed demonstration steps
        T_p_values: List of play lengths to record
        K: Number of arms
    
    Returns:
        recorded_rewards: Dict mapping T_p to cumulative rewards
        recorded_actions: Dict mapping T_p to list of actions taken
        recorded_step_rewards: Dict mapping T_p to reward at that specific step
    """
    # Step 1: Expert plays with N latent + T_d observed steps
    observed_actions, observed_rewards, _, _ = simulate_expert(thetas, N, T_d, K)
    
    # Step 2: Learner estimates priors from observations (naive)
    learner_alphas, learner_betas = estimate_priors_from_observations(
        observed_actions, observed_rewards, K
    )
    
    # Step 3: Learner plays greedily for max(T_p_values) steps
    max_T_p = max(T_p_values)
    actions = []
    rewards = []
    
    alphas = learner_alphas.copy()
    betas = learner_betas.copy()
    
    for step in range(1, max_T_p + 1):
        # Greedy selection
        arm = greedy_select(alphas, betas)
        reward = pull_arm(arm, thetas)
        
        actions.append(arm)
        rewards.append(reward)
        
        # Update beliefs
        alphas[arm] += reward
        betas[arm] += (1 - reward)
    
    # Record at checkpoints
    recorded_rewards = {}
    recorded_actions = {}
    recorded_step_rewards = {}
    
    for T_p in T_p_values:
        cumulative = sum(rewards[:T_p])
        recorded_rewards[T_p] = cumulative
        recorded_actions[T_p] = actions[:T_p]
        recorded_step_rewards[T_p] = rewards[T_p - 1]
    
    return recorded_rewards, recorded_actions, recorded_step_rewards


def run_oracle_social_condition(
    thetas: List[float],
    N: int,
    T_d: int,
    T_p_values: List[int],
    K: int = 12
) -> Tuple[Dict[int, float], Dict[int, List[int]], Dict[int, float]]:
    """
    Run oracle social learning: use expert's true final beliefs.
    
    Gets the expert's actual beliefs after N + T_d steps (including
    latent experience), then Thompson samples from those beliefs.
    
    Args:
        thetas: True payoff probabilities
        N: Expert's latent experience
        T_d: Number of observed demonstration steps
        T_p_values: List of play lengths to record
        K: Number of arms
    
    Returns:
        recorded_rewards: Dict mapping T_p to cumulative rewards
        recorded_actions: Dict mapping T_p to list of actions taken
        recorded_step_rewards: Dict mapping T_p to reward at that specific step
    """
    # Step 1: Expert plays with N latent + T_d observed steps
    # Get the expert's FINAL beliefs (oracle knowledge)
    _, _, expert_final_alphas, expert_final_betas = simulate_expert(thetas, N, T_d, K)
    
    # Step 2: Learner uses oracle beliefs (expert's true final beliefs)
    learner_alphas = expert_final_alphas.copy()
    learner_betas = expert_final_betas.copy()
    
    # Step 3: Learner plays with Thompson sampling for max(T_p_values) steps
    max_T_p = max(T_p_values)
    actions, rewards, _, _, recorded_rewards = run_thompson_sampling(
        thetas, learner_alphas, learner_betas, max_T_p, record_at=T_p_values
    )
    
    # Extract action sequences and step rewards
    recorded_actions = {}
    recorded_step_rewards = {}
    for T_p in T_p_values:
        recorded_actions[T_p] = actions[:T_p]
        recorded_step_rewards[T_p] = rewards[T_p - 1]
    
    return recorded_rewards, recorded_actions, recorded_step_rewards


def run_oracle_greedy_social_condition(
    thetas: List[float],
    N: int,
    T_d: int,
    T_p_values: List[int],
    K: int = 12
) -> Tuple[Dict[int, float], Dict[int, List[int]], Dict[int, float]]:
    """
    Run oracle greedy social learning: use expert's true final beliefs with greedy selection.
    
    Gets the expert's actual beliefs after N + T_d steps (including
    latent experience), then greedily selects arms.
    
    Args:
        thetas: True payoff probabilities
        N: Expert's latent experience
        T_d: Number of observed demonstration steps
        T_p_values: List of play lengths to record
        K: Number of arms
    
    Returns:
        recorded_rewards: Dict mapping T_p to cumulative rewards
        recorded_actions: Dict mapping T_p to list of actions taken
        recorded_step_rewards: Dict mapping T_p to reward at that specific step
    """
    # Step 1: Expert plays with N latent + T_d observed steps
    # Get the expert's FINAL beliefs (oracle knowledge)
    _, _, expert_final_alphas, expert_final_betas = simulate_expert(thetas, N, T_d, K)
    
    # Step 2: Learner uses oracle beliefs (expert's true final beliefs)
    learner_alphas = expert_final_alphas.copy()
    learner_betas = expert_final_betas.copy()
    
    # Step 3: Learner plays greedily for max(T_p_values) steps
    max_T_p = max(T_p_values)
    actions = []
    rewards = []
    
    alphas = learner_alphas.copy()
    betas = learner_betas.copy()
    
    for step in range(1, max_T_p + 1):
        # Greedy selection
        arm = greedy_select(alphas, betas)
        reward = pull_arm(arm, thetas)
        
        actions.append(arm)
        rewards.append(reward)
        
        # Update beliefs
        alphas[arm] += reward
        betas[arm] += (1 - reward)
    
    # Record at checkpoints
    recorded_rewards = {}
    recorded_actions = {}
    recorded_step_rewards = {}
    
    for T_p in T_p_values:
        cumulative = sum(rewards[:T_p])
        recorded_rewards[T_p] = cumulative
        recorded_actions[T_p] = actions[:T_p]
        recorded_step_rewards[T_p] = rewards[T_p - 1]
    
    return recorded_rewards, recorded_actions, recorded_step_rewards


# ============================================================================
# Main Experiment
# ============================================================================

def run_experiment(
    num_bandits: int = 1000,
    N: int = 128,
    T_d_values: List[int] = None,
    T_p_values: List[int] = None,
    K: int = 12,
    payoffs: List[float] = None,
    seed: int = 42,
    verbose: bool = True
) -> Dict:
    """
    Run the full experiment comparing social vs individual learning.
    
    Args:
        num_bandits: Number of random bandits to test
        N: Expert's latent prior experience
        T_d_values: List of demonstration lengths to test
        T_p_values: List of play lengths to test
        K: Number of arms per bandit
        payoffs: Fixed payoff values to permute
        seed: Random seed
        verbose: Whether to print progress
    
    Returns:
        Dictionary with results for both conditions
    """
    if T_d_values is None:
        T_d_values = list(range(1, 101))
    
    if T_p_values is None:
        T_p_values = [2, 4, 8, 16]
    
    if payoffs is None:
        payoffs = [0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.75, 0.8]
    
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    
    # Initialize result storage
    # Structure: results[condition][T_d][T_p] = list of values
    results = {
        'individual': defaultdict(lambda: defaultdict(list)),
        'naive_thompson_social': defaultdict(lambda: defaultdict(list)),
        'naive_greedy_social': defaultdict(lambda: defaultdict(list)),
        'oracle_thompson_social': defaultdict(lambda: defaultdict(list)),
        'oracle_greedy_social': defaultdict(lambda: defaultdict(list)),
        # Actions
        'individual_actions': defaultdict(lambda: defaultdict(list)),
        'naive_thompson_social_actions': defaultdict(lambda: defaultdict(list)),
        'naive_greedy_social_actions': defaultdict(lambda: defaultdict(list)),
        'oracle_thompson_social_actions': defaultdict(lambda: defaultdict(list)),
        'oracle_greedy_social_actions': defaultdict(lambda: defaultdict(list)),
        # Step rewards
        'individual_step_rewards': defaultdict(lambda: defaultdict(list)),
        'naive_thompson_social_step_rewards': defaultdict(lambda: defaultdict(list)),
        'naive_greedy_social_step_rewards': defaultdict(lambda: defaultdict(list)),
        'oracle_thompson_social_step_rewards': defaultdict(lambda: defaultdict(list)),
        'oracle_greedy_social_step_rewards': defaultdict(lambda: defaultdict(list))
    }
    
    if verbose:
        print(f"Running experiment with {num_bandits} bandits...")
        print(f"Parameters: N={N}, K={K}")
        print(f"T_d range: {min(T_d_values)} to {max(T_d_values)}")
        print(f"T_p values: {T_p_values}")
        print()
    
    # Main experiment loop
    for bandit_idx in range(num_bandits):
        if verbose and (bandit_idx + 1) % 100 == 0:
            print(f"Progress: {bandit_idx + 1}/{num_bandits} bandits completed")
        
        # Generate a random bandit
        thetas = generate_bandit(payoffs)
        
        # Test each T_d value
        for T_d in T_d_values:
            # Individual learning condition
            individual_rewards, individual_actions, individual_step_rewards = run_individual_condition(thetas, T_d, T_p_values, K)
            for T_p in T_p_values:
                results['individual'][T_d][T_p].append(individual_rewards[T_p])
                results['individual_actions'][T_d][T_p].append(individual_actions[T_p])
                results['individual_step_rewards'][T_d][T_p].append(individual_step_rewards[T_p])
            
            # Naive Thompson social learning condition
            naive_thompson_rewards, naive_thompson_actions, naive_thompson_step_rewards = run_social_condition(thetas, N, T_d, T_p_values, K)
            for T_p in T_p_values:
                results['naive_thompson_social'][T_d][T_p].append(naive_thompson_rewards[T_p])
                results['naive_thompson_social_actions'][T_d][T_p].append(naive_thompson_actions[T_p])
                results['naive_thompson_social_step_rewards'][T_d][T_p].append(naive_thompson_step_rewards[T_p])
            
            # Naive Greedy social learning condition
            naive_greedy_rewards, naive_greedy_actions, naive_greedy_step_rewards = run_greedy_social_condition(thetas, N, T_d, T_p_values, K)
            for T_p in T_p_values:
                results['naive_greedy_social'][T_d][T_p].append(naive_greedy_rewards[T_p])
                results['naive_greedy_social_actions'][T_d][T_p].append(naive_greedy_actions[T_p])
                results['naive_greedy_social_step_rewards'][T_d][T_p].append(naive_greedy_step_rewards[T_p])
            
            # Oracle Thompson social learning condition
            oracle_thompson_rewards, oracle_thompson_actions, oracle_thompson_step_rewards = run_oracle_social_condition(thetas, N, T_d, T_p_values, K)
            for T_p in T_p_values:
                results['oracle_thompson_social'][T_d][T_p].append(oracle_thompson_rewards[T_p])
                results['oracle_thompson_social_actions'][T_d][T_p].append(oracle_thompson_actions[T_p])
                results['oracle_thompson_social_step_rewards'][T_d][T_p].append(oracle_thompson_step_rewards[T_p])
            
            # Oracle Greedy social learning condition (NEW)
            oracle_greedy_rewards, oracle_greedy_actions, oracle_greedy_step_rewards = run_oracle_greedy_social_condition(thetas, N, T_d, T_p_values, K)
            for T_p in T_p_values:
                results['oracle_greedy_social'][T_d][T_p].append(oracle_greedy_rewards[T_p])
                results['oracle_greedy_social_actions'][T_d][T_p].append(oracle_greedy_actions[T_p])
                results['oracle_greedy_social_step_rewards'][T_d][T_p].append(oracle_greedy_step_rewards[T_p])
    
    if verbose:
        print(f"Completed all {num_bandits} bandits!")
        print()
    
    # Convert to regular dicts for serialization
    results_dict = {
        'individual': {T_d: dict(results['individual'][T_d]) for T_d in T_d_values},
        'naive_thompson_social': {T_d: dict(results['naive_thompson_social'][T_d]) for T_d in T_d_values},
        'naive_greedy_social': {T_d: dict(results['naive_greedy_social'][T_d]) for T_d in T_d_values},
        'oracle_thompson_social': {T_d: dict(results['oracle_thompson_social'][T_d]) for T_d in T_d_values},
        'oracle_greedy_social': {T_d: dict(results['oracle_greedy_social'][T_d]) for T_d in T_d_values},
        'individual_actions': {T_d: dict(results['individual_actions'][T_d]) for T_d in T_d_values},
        'naive_thompson_social_actions': {T_d: dict(results['naive_thompson_social_actions'][T_d]) for T_d in T_d_values},
        'naive_greedy_social_actions': {T_d: dict(results['naive_greedy_social_actions'][T_d]) for T_d in T_d_values},
        'oracle_thompson_social_actions': {T_d: dict(results['oracle_thompson_social_actions'][T_d]) for T_d in T_d_values},
        'oracle_greedy_social_actions': {T_d: dict(results['oracle_greedy_social_actions'][T_d]) for T_d in T_d_values},
        'individual_step_rewards': {T_d: dict(results['individual_step_rewards'][T_d]) for T_d in T_d_values},
        'naive_thompson_social_step_rewards': {T_d: dict(results['naive_thompson_social_step_rewards'][T_d]) for T_d in T_d_values},
        'naive_greedy_social_step_rewards': {T_d: dict(results['naive_greedy_social_step_rewards'][T_d]) for T_d in T_d_values},
        'oracle_thompson_social_step_rewards': {T_d: dict(results['oracle_thompson_social_step_rewards'][T_d]) for T_d in T_d_values},
        'oracle_greedy_social_step_rewards': {T_d: dict(results['oracle_greedy_social_step_rewards'][T_d]) for T_d in T_d_values}
    }
    
    # Add metadata
    results_dict['metadata'] = {
        'num_bandits': num_bandits,
        'N': N,
        'T_d_values': T_d_values,
        'T_p_values': T_p_values,
        'K': K,
        'payoffs': payoffs,
        'seed': seed
    }
    
    return results_dict


# ============================================================================
# Analysis and Visualization
# ============================================================================

def compute_entropy(action_sequence: List[int], K: int = 12) -> float:
    """
    Compute entropy of arm selection distribution.
    
    Args:
        action_sequence: List of arm indices selected
        K: Number of arms
    
    Returns:
        Entropy in nats (use base e)
    """
    if len(action_sequence) == 0:
        return 0.0
    
    # Count frequency of each arm
    counts = np.zeros(K)
    for action in action_sequence:
        counts[action] += 1
    
    # Compute probability distribution
    probs = counts / len(action_sequence)
    
    # Compute entropy (filter out zero probabilities)
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log(probs))
    
    return entropy


def compute_summary_statistics(results: Dict) -> Dict:
    """
    Compute summary statistics from experimental results.
    
    Args:
        results: Results dictionary from run_experiment
    
    Returns:
        Dictionary with summary statistics
    """
    metadata = results['metadata']
    T_d_values = metadata['T_d_values']
    T_p_values = metadata['T_p_values']
    
    conditions = ['individual', 'naive_thompson_social', 'naive_greedy_social', 
                  'oracle_thompson_social', 'oracle_greedy_social']
    
    summary = {
        'mean_rewards': {cond: defaultdict(dict) for cond in conditions},
        'std_rewards': {cond: defaultdict(dict) for cond in conditions},
        'social_advantage': defaultdict(dict),
        'crossover_T_d': {},
        'mean_step_rewards': {cond: defaultdict(dict) for cond in conditions},
        'std_step_rewards': {cond: defaultdict(dict) for cond in conditions},
        'mean_entropy': {cond: defaultdict(dict) for cond in conditions},
        'std_entropy': {cond: defaultdict(dict) for cond in conditions}
    }
    
    # Compute means and standard deviations for all conditions
    for condition in conditions:
        for T_d in T_d_values:
            for T_p in T_p_values:
                # Cumulative rewards
                rewards = results[condition][T_d][T_p]
                summary['mean_rewards'][condition][T_d][T_p] = np.mean(rewards)
                summary['std_rewards'][condition][T_d][T_p] = np.std(rewards)
    
    # Compute social advantage (naive_thompson_social - individual)
    for T_d in T_d_values:
        for T_p in T_p_values:
            naive_thompson_mean = summary['mean_rewards']['naive_thompson_social'][T_d][T_p]
            individual_mean = summary['mean_rewards']['individual'][T_d][T_p]
            summary['social_advantage'][T_d][T_p] = naive_thompson_mean - individual_mean
    
    # Find crossover point for each T_p (where individual catches up to naive social)
    for T_p in T_p_values:
        crossover = None
        for T_d in T_d_values:
            if summary['social_advantage'][T_d][T_p] <= 0:
                crossover = T_d
                break
        summary['crossover_T_d'][T_p] = crossover
    
    # Compute step reward statistics (reward at specific timestep T_p)
    for condition in conditions:
        step_reward_key = f'{condition}_step_rewards'
        for T_d in T_d_values:
            for T_p in T_p_values:
                step_rewards = results[step_reward_key][T_d][T_p]
                summary['mean_step_rewards'][condition][T_d][T_p] = np.mean(step_rewards)
                summary['std_step_rewards'][condition][T_d][T_p] = np.std(step_rewards)
    
    # Compute entropy statistics
    K = metadata['K']
    for condition in conditions:
        action_key = f'{condition}_actions'
        for T_d in T_d_values:
            for T_p in T_p_values:
                action_sequences = results[action_key][T_d][T_p]
                entropies = [compute_entropy(seq, K) for seq in action_sequences]
                summary['mean_entropy'][condition][T_d][T_p] = np.mean(entropies)
                summary['std_entropy'][condition][T_d][T_p] = np.std(entropies)
    
    return summary


def print_summary(results: Dict, summary: Dict):
    """Print summary statistics to console."""
    metadata = results['metadata']
    T_p_values = metadata['T_p_values']
    
    print("=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print()
    
    print(f"Total bandits tested: {metadata['num_bandits']}")
    print(f"Expert latent experience (N): {metadata['N']}")
    print()
    
    # Social advantage at different T_d values
    print("Social Learning Advantage (Social - Individual):")
    print("-" * 70)
    for T_p in T_p_values:
        print(f"\nT_p = {T_p}:")
        # Show advantage at key T_d values
        for T_d in [1, 10, 25, 50, 75, 100]:
            if T_d in summary['social_advantage']:
                adv = summary['social_advantage'][T_d][T_p]
                social_mean = summary['mean_rewards']['social'][T_d][T_p]
                individual_mean = summary['mean_rewards']['individual'][T_d][T_p]
                print(f"  T_d={T_d:3d}: {adv:+.4f} (Social: {social_mean:.4f}, Individual: {individual_mean:.4f})")
        
        # Crossover point
        crossover = summary['crossover_T_d'][T_p]
        if crossover is not None:
            print(f"  → Social advantage lost at T_d = {crossover}")
        else:
            print(f"  → Social maintains advantage for all T_d tested")
    
    print()
    print("=" * 70)


def plot_results(results: Dict, summary: Dict, output_path: str = "prior_experience_analysis.png"):
    """
    Create visualization with 4 subplots comparing conditions.
    
    Args:
        results: Results dictionary from run_experiment
        summary: Summary statistics from compute_summary_statistics
        output_path: Path to save figure
    """
    metadata = results['metadata']
    T_d_values = metadata['T_d_values']
    T_p_values = metadata['T_p_values']
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(1, 4, figsize=(20, 4), sharey=True)
    fig.suptitle('Learning Conditions Comparison: Cumulative Rewards', 
                 fontsize=16, fontweight='bold')
    
    for idx, T_p in enumerate(T_p_values):
        ax = axes[idx]
        
        # Extract data for all 5 conditions
        individual_means = [summary['mean_rewards']['individual'][T_d][T_p] for T_d in T_d_values]
        naive_thompson_means = [summary['mean_rewards']['naive_thompson_social'][T_d][T_p] for T_d in T_d_values]
        naive_greedy_means = [summary['mean_rewards']['naive_greedy_social'][T_d][T_p] for T_d in T_d_values]
        oracle_thompson_means = [summary['mean_rewards']['oracle_thompson_social'][T_d][T_p] for T_d in T_d_values]
        oracle_greedy_means = [summary['mean_rewards']['oracle_greedy_social'][T_d][T_p] for T_d in T_d_values]
        
        individual_stds = [summary['std_rewards']['individual'][T_d][T_p] for T_d in T_d_values]
        naive_thompson_stds = [summary['std_rewards']['naive_thompson_social'][T_d][T_p] for T_d in T_d_values]
        naive_greedy_stds = [summary['std_rewards']['naive_greedy_social'][T_d][T_p] for T_d in T_d_values]
        oracle_thompson_stds = [summary['std_rewards']['oracle_thompson_social'][T_d][T_p] for T_d in T_d_values]
        oracle_greedy_stds = [summary['std_rewards']['oracle_greedy_social'][T_d][T_p] for T_d in T_d_values]
        
        # Compute standard error
        n = metadata['num_bandits']
        individual_se = np.array(individual_stds) / np.sqrt(n)
        naive_thompson_se = np.array(naive_thompson_stds) / np.sqrt(n)
        naive_greedy_se = np.array(naive_greedy_stds) / np.sqrt(n)
        oracle_thompson_se = np.array(oracle_thompson_stds) / np.sqrt(n)
        oracle_greedy_se = np.array(oracle_greedy_stds) / np.sqrt(n)
        
        # Plot all 5 curves with error shading
        ax.plot(T_d_values, individual_means, color='#ff7f0e', linewidth=2, label='Individual')
        ax.fill_between(T_d_values,
                        np.array(individual_means) - individual_se,
                        np.array(individual_means) + individual_se,
                        color='#ff7f0e', alpha=0.2)
        
        ax.plot(T_d_values, naive_thompson_means, color='#1f77b4', linewidth=2, label='Naive Thompson Social')
        ax.fill_between(T_d_values, 
                        np.array(naive_thompson_means) - naive_thompson_se,
                        np.array(naive_thompson_means) + naive_thompson_se,
                        color='#1f77b4', alpha=0.2)
        
        ax.plot(T_d_values, naive_greedy_means, color='#2ca02c', linewidth=2, label='Naive Greedy Social')
        ax.fill_between(T_d_values,
                        np.array(naive_greedy_means) - naive_greedy_se,
                        np.array(naive_greedy_means) + naive_greedy_se,
                        color='#2ca02c', alpha=0.2)
        
        ax.plot(T_d_values, oracle_thompson_means, color='#d62728', linewidth=2, label='Oracle Thompson Social')
        ax.fill_between(T_d_values,
                        np.array(oracle_thompson_means) - oracle_thompson_se,
                        np.array(oracle_thompson_means) + oracle_thompson_se,
                        color='#d62728', alpha=0.2)
        
        ax.plot(T_d_values, oracle_greedy_means, color='#9467bd', linewidth=2, label='Oracle Greedy Social')
        ax.fill_between(T_d_values,
                        np.array(oracle_greedy_means) - oracle_greedy_se,
                        np.array(oracle_greedy_means) + oracle_greedy_se,
                        color='#9467bd', alpha=0.2)
        
        # Formatting
        ax.set_xlabel('Demonstration Length (T_d)', fontsize=12)
        if idx == 0:
            ax.set_ylabel(f'Average Cumulative Reward', fontsize=12)
        ax.set_title(f'T_p = {T_p}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add legend only to first subplot
        if idx == 0:
            ax.legend(loc='best', fontsize=10)
        
        # Mark crossover point if exists
        crossover = summary['crossover_T_d'][T_p]
        if crossover is not None:
            ax.axvline(x=crossover, color='red', linestyle='--', alpha=0.5, linewidth=1)
            ax.text(crossover, ax.get_ylim()[0] + 0.1, f'T_d={crossover}', 
                   rotation=90, va='bottom', ha='right', fontsize=8, color='red')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_path}")
    plt.close()


def plot_step_rewards(results: Dict, summary: Dict, output_path: str):
    """
    Create visualization of average reward at specific timestep T_p.
    
    Shows the reward received at step T_p (not cumulative), averaged across all bandits.
    
    Structure: 4 subplots (one per T_p value)
    X-axis: T_d (demonstration length)
    Y-axis: Average reward at timestep T_p
    """
    metadata = results['metadata']
    T_d_values = metadata['T_d_values']
    T_p_values = metadata['T_p_values']
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 4), sharey=True)
    fig.suptitle('Learning Conditions Comparison: Average Reward at Timestep T_p', 
                 fontsize=16, fontweight='bold')
    
    for idx, T_p in enumerate(T_p_values):
        ax = axes[idx]
        
        # Extract average reward at timestep T_p for all 5 conditions
        individual_means = [summary['mean_step_rewards']['individual'][T_d][T_p] 
                           for T_d in T_d_values]
        naive_thompson_means = [summary['mean_step_rewards']['naive_thompson_social'][T_d][T_p] 
                                for T_d in T_d_values]
        naive_greedy_means = [summary['mean_step_rewards']['naive_greedy_social'][T_d][T_p] 
                              for T_d in T_d_values]
        oracle_thompson_means = [summary['mean_step_rewards']['oracle_thompson_social'][T_d][T_p] 
                                 for T_d in T_d_values]
        oracle_greedy_means = [summary['mean_step_rewards']['oracle_greedy_social'][T_d][T_p] 
                               for T_d in T_d_values]
        
        # Standard error
        n = metadata['num_bandits']
        individual_se = [summary['std_step_rewards']['individual'][T_d][T_p] / np.sqrt(n)
                        for T_d in T_d_values]
        naive_thompson_se = [summary['std_step_rewards']['naive_thompson_social'][T_d][T_p] / np.sqrt(n)
                             for T_d in T_d_values]
        naive_greedy_se = [summary['std_step_rewards']['naive_greedy_social'][T_d][T_p] / np.sqrt(n)
                           for T_d in T_d_values]
        oracle_thompson_se = [summary['std_step_rewards']['oracle_thompson_social'][T_d][T_p] / np.sqrt(n)
                              for T_d in T_d_values]
        oracle_greedy_se = [summary['std_step_rewards']['oracle_greedy_social'][T_d][T_p] / np.sqrt(n)
                            for T_d in T_d_values]
        
        # Plot all 5 curves with error shading
        ax.plot(T_d_values, individual_means, color='#ff7f0e', linewidth=2,
                label='Individual')
        ax.fill_between(T_d_values,
                        np.array(individual_means) - individual_se,
                        np.array(individual_means) + individual_se,
                        color='#ff7f0e', alpha=0.2)
        
        ax.plot(T_d_values, naive_thompson_means, color='#1f77b4', linewidth=2, 
                label='Naive Thompson Social')
        ax.fill_between(T_d_values, 
                        np.array(naive_thompson_means) - naive_thompson_se,
                        np.array(naive_thompson_means) + naive_thompson_se,
                        color='#1f77b4', alpha=0.2)
        
        ax.plot(T_d_values, naive_greedy_means, color='#2ca02c', linewidth=2,
                label='Naive Greedy Social')
        ax.fill_between(T_d_values,
                        np.array(naive_greedy_means) - naive_greedy_se,
                        np.array(naive_greedy_means) + naive_greedy_se,
                        color='#2ca02c', alpha=0.2)
        
        ax.plot(T_d_values, oracle_thompson_means, color='#d62728', linewidth=2,
                label='Oracle Thompson Social')
        ax.fill_between(T_d_values,
                        np.array(oracle_thompson_means) - oracle_thompson_se,
                        np.array(oracle_thompson_means) + oracle_thompson_se,
                        color='#d62728', alpha=0.2)
        
        ax.plot(T_d_values, oracle_greedy_means, color='#9467bd', linewidth=2,
                label='Oracle Greedy Social')
        ax.fill_between(T_d_values,
                        np.array(oracle_greedy_means) - oracle_greedy_se,
                        np.array(oracle_greedy_means) + oracle_greedy_se,
                        color='#9467bd', alpha=0.2)
        
        # Formatting
        ax.set_xlabel('Demonstration Length (T_d)', fontsize=12)
        if idx == 0:
            ax.set_ylabel('Average Reward at Step T_p', fontsize=12)
        ax.set_title(f'T_p = {T_p}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])  # Reward is binary (0 or 1)
        
        if idx == 0:
            ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Step reward figure saved to: {output_path}")
    plt.close()


def plot_entropy(results: Dict, summary: Dict, output_path: str):
    """
    Create visualization of entropy of arm selection.
    
    Structure: 4 subplots (one per T_p value)
    X-axis: T_d (demonstration length)
    Y-axis: Entropy (nats)
    
    Higher entropy = more exploration
    Lower entropy = more exploitation/confidence
    """
    metadata = results['metadata']
    T_d_values = metadata['T_d_values']
    T_p_values = metadata['T_p_values']
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 4), sharey=True)
    fig.suptitle('Learning Conditions Comparison: Entropy of Arm Selection', 
                 fontsize=16, fontweight='bold')
    
    for idx, T_p in enumerate(T_p_values):
        ax = axes[idx]
        
        # Extract entropy data for all 5 conditions
        individual_entropies = [summary['mean_entropy']['individual'][T_d][T_p]
                                for T_d in T_d_values]
        naive_thompson_entropies = [summary['mean_entropy']['naive_thompson_social'][T_d][T_p] 
                                    for T_d in T_d_values]
        naive_greedy_entropies = [summary['mean_entropy']['naive_greedy_social'][T_d][T_p]
                                  for T_d in T_d_values]
        oracle_thompson_entropies = [summary['mean_entropy']['oracle_thompson_social'][T_d][T_p]
                                     for T_d in T_d_values]
        oracle_greedy_entropies = [summary['mean_entropy']['oracle_greedy_social'][T_d][T_p]
                                   for T_d in T_d_values]
        
        # Standard error
        n = metadata['num_bandits']
        individual_se = [summary['std_entropy']['individual'][T_d][T_p] / np.sqrt(n)
                         for T_d in T_d_values]
        naive_thompson_se = [summary['std_entropy']['naive_thompson_social'][T_d][T_p] / np.sqrt(n)
                             for T_d in T_d_values]
        naive_greedy_se = [summary['std_entropy']['naive_greedy_social'][T_d][T_p] / np.sqrt(n)
                           for T_d in T_d_values]
        oracle_thompson_se = [summary['std_entropy']['oracle_thompson_social'][T_d][T_p] / np.sqrt(n)
                              for T_d in T_d_values]
        oracle_greedy_se = [summary['std_entropy']['oracle_greedy_social'][T_d][T_p] / np.sqrt(n)
                            for T_d in T_d_values]
        
        # Plot all 5 curves
        ax.plot(T_d_values, individual_entropies, color='#ff7f0e', linewidth=2,
                label='Individual')
        ax.fill_between(T_d_values,
                        np.array(individual_entropies) - individual_se,
                        np.array(individual_entropies) + individual_se,
                        color='#ff7f0e', alpha=0.2)
        
        ax.plot(T_d_values, naive_thompson_entropies, color='#1f77b4', linewidth=2,
                label='Naive Thompson Social')
        ax.fill_between(T_d_values,
                        np.array(naive_thompson_entropies) - naive_thompson_se,
                        np.array(naive_thompson_entropies) + naive_thompson_se,
                        color='#1f77b4', alpha=0.2)
        
        ax.plot(T_d_values, naive_greedy_entropies, color='#2ca02c', linewidth=2,
                label='Naive Greedy Social')
        ax.fill_between(T_d_values,
                        np.array(naive_greedy_entropies) - naive_greedy_se,
                        np.array(naive_greedy_entropies) + naive_greedy_se,
                        color='#2ca02c', alpha=0.2)
        
        ax.plot(T_d_values, oracle_thompson_entropies, color='#d62728', linewidth=2,
                label='Oracle Thompson Social')
        ax.fill_between(T_d_values,
                        np.array(oracle_thompson_entropies) - oracle_thompson_se,
                        np.array(oracle_thompson_entropies) + oracle_thompson_se,
                        color='#d62728', alpha=0.2)
        
        ax.plot(T_d_values, oracle_greedy_entropies, color='#9467bd', linewidth=2,
                label='Oracle Greedy Social')
        ax.fill_between(T_d_values,
                        np.array(oracle_greedy_entropies) - oracle_greedy_se,
                        np.array(oracle_greedy_entropies) + oracle_greedy_se,
                        color='#9467bd', alpha=0.2)
        
        # Add reference line for maximum entropy (uniform distribution)
        K = metadata['K']
        max_entropy = np.log(K)  # Uniform distribution entropy
        ax.axhline(y=max_entropy, color='gray', linestyle='--', alpha=0.5,
                   label=f'Max (uniform): {max_entropy:.2f}')
        
        # Formatting
        ax.set_xlabel('Demonstration Length (T_d)', fontsize=12)
        if idx == 0:
            ax.set_ylabel('Entropy (nats)', fontsize=12)
        ax.set_title(f'T_p = {T_p}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        if idx == 0:
            ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Entropy figure saved to: {output_path}")
    plt.close()


def save_results(results: Dict, output_path: str = "prior_experience_results.pkl"):
    """Save results to pickle file."""
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to: {output_path}")


def save_results_json(results: Dict, summary: Dict, output_path: str = "prior_experience_results.json"):
    """Save results and summary to JSON file (without raw data for size)."""
    conditions = ['individual', 'naive_thompson_social', 'naive_greedy_social', 
                  'oracle_thompson_social', 'oracle_greedy_social']
    
    output = {
        'metadata': results['metadata'],
        'summary': {
            'mean_rewards': {
                cond: {str(k): v for k, v in summary['mean_rewards'][cond].items()}
                for cond in conditions
            },
            'mean_step_rewards': {
                cond: {str(k): v for k, v in summary['mean_step_rewards'][cond].items()}
                for cond in conditions
            },
            'mean_entropy': {
                cond: {str(k): v for k, v in summary['mean_entropy'][cond].items()}
                for cond in conditions
            },
            'social_advantage': {str(k): v for k, v in summary['social_advantage'].items()},
            'crossover_T_d': summary['crossover_T_d']
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Summary saved to: {output_path}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Analyze social vs individual learning in Thompson sampling'
    )
    parser.add_argument('--num_bandits', type=int, default=256,
                       help='Number of random bandits to test (default: 1000)')
    parser.add_argument('--N', type=int, default=128,
                       help='Expert latent prior experience (default: 128)')
    parser.add_argument('--K', type=int, default=12,
                       help='Number of arms per bandit (default: 12)')
    parser.add_argument('--T_d_min', type=int, default=1,
                       help='Minimum demonstration length (default: 1)')
    parser.add_argument('--T_d_max', type=int, default=20,
                       help='Maximum demonstration length (default: 100)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--figures_dir', type=str, default=None,
                       help='Output directory for figures (default: figures/prior_experience_analysis_results/)')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Output directory for data files (default: data/prior_experience_analysis_results/)')
    parser.add_argument('--quick_test', action='store_true',
                       help='Run quick test with 10 bandits')
    
    args = parser.parse_args()
    
    # Set up run directory structure (no wandb for this script)
    run_id = get_run_id(use_wandb=False)
    run_dir = setup_run_directory(run_id)
    print(f"Run directory: {run_dir}")
    print(f"Run ID: {run_id}")
    
    # Write run metadata
    metadata = {
        'run_id': run_id,
        'script': 'prior_experience_analysis.py',
        'command': ' '.join(sys.argv),
        'num_bandits': args.num_bandits,
        'N': args.N,
        'K': args.K,
        'T_d_min': args.T_d_min,
        'T_d_max': args.T_d_max,
        'seed': args.seed,
        'quick_test': args.quick_test,
    }
    write_run_yaml(run_dir, metadata)
    write_config_yaml(run_dir, vars(args))
    
    # Quick test mode
    if args.quick_test:
        print("Running in QUICK TEST mode (10 bandits, T_d up to 20)")
        args.num_bandits = 10
        args.T_d_max = 20
    
    # Setup output directories
    import os
    
    # Set default paths to use run directory
    if args.figures_dir is None:
        figures_dir = str(run_dir / 'results' / 'figures')
    else:
        figures_dir = args.figures_dir
    
    if args.data_dir is None:
        data_dir = str(run_dir / 'results')
    else:
        data_dir = args.data_dir
    
    # Create directories if they don't exist
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Output directories:")
    print(f"  Figures: {figures_dir}")
    print(f"  Data: {data_dir}")
    print()
    
    try:
        # Setup
        T_d_values = list(range(args.T_d_min, args.T_d_max + 1))
        T_p_values = [2, 4, 8, 16]
        payoffs = [0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.75, 0.8]
        
        # Run experiment
        results = run_experiment(
            num_bandits=args.num_bandits,
            N=args.N,
            T_d_values=T_d_values,
            T_p_values=T_p_values,
            K=args.K,
            payoffs=payoffs,
            seed=args.seed,
            verbose=True
        )
        
        # Compute summary statistics
        summary = compute_summary_statistics(results)
        
        # Generate plots
        # 1. Cumulative rewards
        figure_path_cumulative = os.path.join(figures_dir, 'prior_experience_analysis_cumulative.png')
        plot_results(results, summary, figure_path_cumulative)
        
        # 2. Step rewards
        figure_path_step = os.path.join(figures_dir, 'prior_experience_analysis_step_reward.png')
        plot_step_rewards(results, summary, figure_path_step)
        
        # 3. Entropy
        figure_path_entropy = os.path.join(figures_dir, 'prior_experience_analysis_entropy.png')
        plot_entropy(results, summary, figure_path_entropy)
        
        # Save results
        pkl_path = os.path.join(data_dir, 'prior_experience_results.pkl')
        save_results(results, pkl_path)
        
        json_path = os.path.join(data_dir, 'prior_experience_results.json')
        save_results_json(results, summary, json_path)
        
        print("\nAnalysis complete!")
        
        # Analysis completed successfully
        finalize_run_yaml(run_dir, status='completed')
        print(f"\n{'='*60}")
        print("Analysis completed successfully!")
        print(f"{'='*60}")
        
    except KeyboardInterrupt:
        print(f"\n{'='*60}")
        print("Analysis interrupted by user")
        print(f"{'='*60}")
        finalize_run_yaml(run_dir, status='interrupted')
        raise
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"Analysis failed with error: {e}")
        print(f"{'='*60}")
        finalize_run_yaml(run_dir, status='failed', error=str(e))
        raise


if __name__ == '__main__':
    main()
