"""Social Learning Analysis: Multi-agent bandit experiments."""
import numpy as np
import random
import sys
import os
import pandas as pd
from scipy.stats import entropy
import matplotlib.pyplot as plt
from typing import List
from pathlib import Path
from utils import get_run_id, setup_run_directory, write_run_yaml, write_config_yaml, finalize_run_yaml


class Bandit: 
    def __init__(self, payoffs: List[float]):
        self.payoffs = payoffs

    def pull(self, action: int) -> int:
        if random.random() < self.payoffs[action]:
            return 1
        return 0

    @property
    def K(self):
        return len(self.payoffs)

class GreedySampler():
    def __init__(self, alphas: np.ndarray, betas: np.ndarray, bandit: Bandit, T: int):
        self._alphas = alphas
        self._betas = betas
        self._bandit = bandit 
        self._T = T
        self._actions = np.zeros(T, dtype=int)
        self._rewards = np.zeros(T, dtype=int)
        self.q_values = self._alphas / (self._alphas + self._betas)

    def _sample_action(self) -> int: 
        return np.argmax(self.q_values)

    def run(self):
        for t in range(self._T):
            action = self._sample_action()
            reward = self._bandit.pull(action)
            self._actions[t] = action
            self._rewards[t] = reward

    @property
    def actions(self) -> np.ndarray:
        return self._actions

    @property
    def rewards(self) -> np.ndarray:
        return self._rewards

class ThompsonSampler():
    def __init__(self, alphas: np.ndarray, betas: np.ndarray, bandit: Bandit, T: int):
        self._alphas = alphas
        self._betas = betas
        self._bandit = bandit
        self._t_alphas = np.zeros((T, bandit.K), dtype=int)
        self._t_betas = np.zeros((T, bandit.K), dtype=int)
        self._T = T
        self._actions = np.zeros(T, dtype=int)
        self._rewards = np.zeros(T, dtype=int)

    def _sample_action(self) -> int:
        return np.argmax(np.random.beta(self._alphas, self._betas))

    def run(self):
        for t in range(self._T):
            action = self._sample_action()
            reward = self._bandit.pull(action)
            self._alphas[action] += reward
            self._betas[action] += 1 - reward
            self._t_alphas[t] = self._alphas
            self._t_betas[t] = self._betas
            self._actions[t] = action
            self._rewards[t] = reward

    @property
    def alphas(self) -> np.ndarray:
        return self._alphas

    @property
    def betas(self) -> np.ndarray:
        return self._betas

    @property
    def bandit(self) -> Bandit:
        return self._bandit

    @property
    def T(self) -> int:
        return self._T

    @property
    def t_alphas(self) -> np.ndarray:
        return self._t_alphas

    @property
    def t_betas(self) -> np.ndarray:
        return self._t_betas
    
    @property
    def actions(self) -> np.ndarray:
        return self._actions

    @property
    def rewards(self) -> np.ndarray:
        return self._rewards

class Agent():
    def __init__(self, agent_type, actions, rewards):
        self.agent_type = agent_type
        self.rewards = rewards
        self.cumulative_rewards = np.cumsum(rewards)
        self.entropy = compute_entropy(actions)

def compute_entropy(actions):
    counts = np.bincount(actions, minlength=12)
    return entropy(actions)

def append_to_csv(agent, T_E, T_D, T_P, trial_id, filepath):
    """Append agent data to CSV file."""
    n = len(agent.rewards)
    new_data = pd.DataFrame({
        'trial_id': np.repeat(trial_id, n),
        'agent_type': np.repeat(agent.agent_type, n),
        'T_E': np.repeat(T_E, n),
        'T_D': np.repeat(T_D, n),
        'T_P': np.repeat(T_P, n),
        'rewards': agent.rewards,
        'cumulative_rewards': agent.cumulative_rewards,
        'entropy': agent.entropy
    })  
    file_exists = os.path.isfile(filepath)
    new_data.to_csv(filepath, mode='a', header=not file_exists, index=False)


def generate_plot_data(input_csv, output_csv):
    """
    Pre-compute aggregated statistics for fast plotting.
    
    Reads the raw data CSV and generates a summary CSV with means and standard errors
    for each combination of agent_type, T_E, T_D, and T_P.
    
    Args:
        input_csv: Path to raw data CSV file
        output_csv: Path to output aggregated CSV file
    """
    print(f"Generating plot data from {input_csv}...")
    
    # Read raw data
    df = pd.read_csv(input_csv)
    
    # First, aggregate per trial (average across timesteps within each trial)
    df_per_trial = df.groupby(['trial_id', 'agent_type', 'T_E', 'T_D', 'T_P']).agg({
        'entropy': 'mean',
        'rewards': 'mean',
        'cumulative_rewards': 'last'  # Take final cumulative value
    }).reset_index()
    
    # Now aggregate across trials for each combination to get statistics
    grouped = df_per_trial.groupby(['agent_type', 'T_E', 'T_D', 'T_P'])
    
    # Compute mean, std, and count for each metric
    stats_list = []
    for name, group in grouped:
        agent_type, t_e, t_d, t_p = name
        
        row = {
            'agent_type': agent_type,
            'T_E': t_e,
            'T_D': t_d,
            'T_P': t_p
        }
        
        # For each metric, compute mean, std, count, and SE
        for metric in ['entropy', 'rewards', 'cumulative_rewards']:
            values = group[metric]
            mean_val = values.mean()
            std_val = values.std()
            count_val = len(values)
            # Handle case where std is NaN (single sample)
            if pd.isna(std_val) or count_val <= 1:
                se_val = 0.0
            else:
                se_val = std_val / np.sqrt(count_val)
            
            row[f'{metric}_mean'] = mean_val
            row[f'{metric}_std'] = std_val if not pd.isna(std_val) else 0.0
            row[f'{metric}_se'] = se_val
            row[f'{metric}_n'] = count_val
        
        stats_list.append(row)
    
    # Create DataFrame from list
    stats_df = pd.DataFrame(stats_list)
    
    # Sort for readability
    stats_df = stats_df.sort_values(['agent_type', 'T_E', 'T_D', 'T_P'])
    
    # Save to CSV
    stats_df.to_csv(output_csv, index=False)
    
    print(f"Plot data saved to {output_csv}")
    print(f"Reduced from {len(df)} rows to {len(stats_df)} rows")
    
    return stats_df


def plot_reward_by_td_and_tp(csv_file, output_file):
    """
    Create a figure with 3x4 subplots showing reward, cumulative reward, and entropy vs T_D.
    
    Args:
        csv_file: Path to the aggregated plot data CSV
        output_file: Path to save the output figure
    """
    # Read the data
    df = pd.read_csv(csv_file)
    
    # Define the T_P values for each subplot
    t_p_values = [2, 4, 8, 16]
    
    # Create figure with 3 rows and 4 columns
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    
    # Get unique agent types and sort them for consistent ordering
    agent_types = sorted(df['agent_type'].unique())
    
    # Color map for agent types with automatic fallback colors
    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    colors = {
        'individual learner': '#1f77b4',  # blue
        'naive thompson social learner': '#ff7f0e',  # orange
        'naive greedy social learner': '#2ca02c',  # green
    }
    
    # Assign colors to any additional agent types not in the predefined map
    for i, agent_type in enumerate(agent_types):
        if agent_type not in colors:
            colors[agent_type] = color_palette[i % len(color_palette)]
    
    # Marker styles for better distinction
    markers = ['o', 's', '^', 'D', 'v', '<']
    marker_map = {agent_type: markers[i % len(markers)] for i, agent_type in enumerate(agent_types)}
    
    # Define metrics for each row
    metrics = [
        ('rewards_mean', 'rewards_se', 'Average Reward'),
        ('cumulative_rewards_mean', 'cumulative_rewards_se', 'Cumulative Reward'),
        ('entropy_mean', 'entropy_se', 'Entropy')
    ]
    
    # First pass: compute y-axis limits for each row
    row_ylims = []
    for row_idx, (metric_mean, metric_se, ylabel) in enumerate(metrics):
        y_min = float('inf')
        y_max = float('-inf')
        
        for t_p in t_p_values:
            df_tp = df[df['T_P'] == t_p]
            # Get min and max across all agent types for this T_P
            y_min = min(y_min, (df_tp[metric_mean] - df_tp[metric_se]).min())
            y_max = max(y_max, (df_tp[metric_mean] + df_tp[metric_se]).max())
        
        # Add some padding (5% on each side)
        y_range = y_max - y_min
        y_padding = y_range * 0.05
        row_ylims.append((y_min - y_padding, y_max + y_padding))
    
    # Second pass: plot each metric (row) and T_P value (column)
    for row_idx, (metric_mean, metric_se, ylabel) in enumerate(metrics):
        for col_idx, t_p in enumerate(t_p_values):
            ax = axes[row_idx, col_idx]
            
            # Filter data for this T_P value
            df_tp = df[df['T_P'] == t_p]
            
            # Plot each agent type
            for agent_type in agent_types:
                df_agent = df_tp[df_tp['agent_type'] == agent_type]
                
                # Sort by T_D for proper line plotting
                df_agent = df_agent.sort_values('T_D')
                
                # Plot with error bars
                ax.errorbar(df_agent['T_D'], 
                           df_agent[metric_mean],
                           yerr=df_agent[metric_se],
                           label=agent_type,
                           color=colors[agent_type],
                           marker=marker_map[agent_type],
                           capsize=3,
                           capthick=1,
                           linewidth=2,
                           markersize=4,
                           alpha=0.8)
            
            # Configure subplot
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 21)
            ax.set_ylim(row_ylims[row_idx])  # Set consistent y-axis for this row
            
            # Column titles (only on top row)
            if row_idx == 0:
                ax.set_title(f'T_P = {t_p}', fontsize=12, fontweight='bold')
            
            # X-axis label (only on bottom row)
            if row_idx == 2:
                ax.set_xlabel('T_D (Demonstration Steps)', fontsize=11)
            
            # Y-axis label (only on leftmost column)
            if col_idx == 0:
                ax.set_ylabel(ylabel, fontsize=11)
            
            # Legend (only on first subplot)
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=8, loc='best')
    
    # Overall title
    fig.suptitle('Social Learning Analysis: Reward, Cumulative Reward, and Entropy vs Demonstration Steps', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_file}")
    plt.close(fig)
    
    return fig


def run_social_learning_analysis(run_dir, config):
    """
    Run social learning analysis experiments.
    
    Args:
        run_dir: Path to run directory
        config: Dictionary with experiment parameters (K, T_E, T_D, T_P, n_trials, seed)
    """
    # Extract parameters from config
    K = config['K']
    T_E = config['T_E']
    T_D = config['T_D']
    T_P = config['T_P']
    n_trials = config['n_trials']
    
    sorted_payoffs = [0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.75, 0.8]
    prior_alphas = np.ones(K)
    prior_betas = np.ones(K)
    
    # Setup output paths
    results_dir = run_dir / 'results'
    figures_dir = results_dir / 'figures'
    raw_csv = results_dir / 'social_learning_analysis.csv'
    plot_csv = results_dir / 'social_learning_analysis_plot_data.csv'
    
    print(f"Starting social learning analysis...")
    print(f"Parameters: K={K}, T_E={T_E}, T_D={T_D}, T_P={T_P}, n_trials={n_trials}")
    print(f"Output directory: {run_dir}")
    
    # Run experiments
    for i in range(n_trials):
        if (i + 1) % 50 == 0:
            print(f"  Trial {i + 1}/{n_trials}")
        
        payoffs = random.sample(sorted_payoffs, K)
        bandit = Bandit(payoffs)

        demonstrator = ThompsonSampler(prior_alphas.copy(), prior_betas.copy(), bandit, T_E + T_D + max(T_P))
        demonstrator.run()

        for t_d in range(1, T_D + 1):
            for t_p in T_P:
                # Individual learner
                individual_learner_agent = Agent(
                    agent_type='individual learner',
                    actions=demonstrator.actions[t_d:t_d+t_p],
                    rewards=demonstrator.rewards[t_d:t_d+t_p]
                )
                append_to_csv(
                    agent=individual_learner_agent,
                    T_E=T_E, T_D=t_d, T_P=t_p,
                    trial_id=i,
                    filepath=raw_csv
                )

                # Naive thompson social learner
                naive_prior_alphas = demonstrator.t_alphas[T_E + t_d - 1, :] - demonstrator.t_alphas[T_E - 1, :] + prior_alphas
                naive_prior_betas = demonstrator.t_betas[T_E + t_d - 1, :] - demonstrator.t_betas[T_E - 1, :] + prior_betas
                
                naive_thompson_social_learner = ThompsonSampler(
                    naive_prior_alphas.copy(),
                    naive_prior_betas.copy(),
                    bandit, t_p
                )
                naive_thompson_social_learner.run()
                naive_thompson_social_learner_agent = Agent(
                    agent_type='naive thompson social learner',
                    actions=naive_thompson_social_learner.actions,
                    rewards=naive_thompson_social_learner.rewards
                )
                append_to_csv(
                    agent=naive_thompson_social_learner_agent,
                    T_E=T_E, T_D=t_d, T_P=t_p,
                    trial_id=i,
                    filepath=raw_csv
                )

                # Naive greedy social learner
                naive_greedy_social_learner = GreedySampler(
                    naive_prior_alphas.copy(),
                    naive_prior_betas.copy(),
                    bandit, t_p
                )
                naive_greedy_social_learner.run()
                naive_greedy_social_learner_agent = Agent(
                    agent_type='naive greedy social learner',
                    actions=naive_greedy_social_learner.actions,
                    rewards=naive_greedy_social_learner.rewards
                )
                append_to_csv(
                    agent=naive_greedy_social_learner_agent,
                    T_E=T_E, T_D=t_d, T_P=t_p,
                    trial_id=i,
                    filepath=raw_csv
                )

    print(f"Completed {n_trials} trials")
    
    # Generate aggregated plot data
    print("\nGenerating aggregated statistics...")
    generate_plot_data(input_csv=raw_csv, output_csv=plot_csv)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_output = figures_dir / 'reward_by_td_tp.png'
    plot_reward_by_td_and_tp(csv_file=plot_csv, output_file=plot_output)
    
    print(f"\nAnalysis complete!")
    print(f"Raw data: {raw_csv}")
    print(f"Aggregated data: {plot_csv}")
    print(f"Figure: {plot_output}")


def main():
    """Main entry point for social learning analysis."""
    # Configuration - edit these parameters as needed
    config = {
        'K': 12,
        'T_E': 128,
        'T_D': 20,
        'T_P': [2, 4, 8, 16],
        'n_trials': 256,
        'seed': 1234,  # Change this value to get different random results, or set to None
    }
    
    # Set random seed if provided
    if config['seed'] is not None:
        random.seed(config['seed'])
        np.random.seed(config['seed'])
        print(f"Random seed set to {config['seed']}")
    
    # Setup run directory
    run_id = get_run_id(use_wandb=False)
    run_dir = setup_run_directory(run_id)
    
    print(f"Run ID: {run_id}")
    print(f"Run directory: {run_dir}")
    
    # Write config and run metadata
    write_config_yaml(run_dir, config)
    
    metadata = {
        'run_id': run_id,
        'script': 'social_learning_analysis.py',
        'command': ' '.join(sys.argv),
    }
    write_run_yaml(run_dir, metadata)
    
    try:
        # Run analysis
        run_social_learning_analysis(run_dir, config)
        
        # Mark as completed
        finalize_run_yaml(run_dir, status='completed')
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        finalize_run_yaml(run_dir, status='interrupted')
        sys.exit(1)
        
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        finalize_run_yaml(run_dir, status='failed', error=str(e))
        sys.exit(1)


if __name__ == '__main__':
    main()
