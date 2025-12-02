#!/usr/bin/env python3
"""
Analyze W&B sweep results to find best hyperparameters for different objectives.

Usage:
    python src/analyze_sweep.py --sweep_id <sweep-id>
    
Example:
    python src/analyze_sweep.py --sweep_id juliomartinez-individual/social-belief-modeling/abc123xyz
"""

import argparse
import wandb
import pandas as pd


def analyze_sweep(sweep_id: str, top_k: int = 10):
    """
    Analyze a W&B sweep and print best configurations for different metrics.
    
    Args:
        sweep_id: Full sweep ID in format "entity/project/sweep_id"
        top_k: Number of top runs to display for each metric
    """
    print("="*80)
    print(f"Analyzing Sweep: {sweep_id}")
    print("="*80)
    
    # Connect to W&B
    api = wandb.Api()
    
    try:
        sweep = api.sweep(sweep_id)
    except Exception as e:
        print(f"Error loading sweep: {e}")
        print("\nMake sure the sweep ID is in format: entity/project/sweep_id")
        return
    
    print(f"\nSweep Name: {sweep.name}")
    print(f"Total Runs: {len(sweep.runs)}")
    print(f"Sweep Method: {sweep.config.get('method', 'unknown')}")
    print(f"Optimizing Metric: {sweep.config.get('metric', {}).get('name', 'unknown')}")
    print("\n" + "="*80)
    
    # Get all runs sorted by different metrics
    runs_by_combined = sorted(sweep.runs, key=lambda r: r.summary.get('val/loss', float('inf')))
    runs_by_theta = sorted(sweep.runs, key=lambda r: r.summary.get('val/loss_theta', float('inf')))
    runs_by_demo = sorted(sweep.runs, key=lambda r: r.summary.get('val/loss_demo', float('inf')))
    
    # Print best runs for each metric
    print("\n🏆 TOP RUNS BY COMBINED LOSS (val/loss)")
    print("-"*80)
    print_top_runs(runs_by_combined, 'val/loss', top_k)
    
    print("\n🎯 TOP RUNS BY THETA LOSS (val/loss_theta)")
    print("-"*80)
    print_top_runs(runs_by_theta, 'val/loss_theta', top_k)
    
    print("\n🧠 TOP RUNS BY DEMO LOSS (val/loss_demo)")
    print("-"*80)
    print_top_runs(runs_by_demo, 'val/loss_demo', top_k)
    
    # Compare best configs
    print("\n" + "="*80)
    print("📊 BEST CONFIGURATION COMPARISON")
    print("="*80)
    
    print("\n✨ Best for Combined Loss:")
    print_config(runs_by_combined[0])
    
    print("\n🎯 Best for Theta Loss:")
    print_config(runs_by_theta[0])
    
    print("\n🧠 Best for Demo Loss:")
    print_config(runs_by_demo[0])
    
    # Create comparison dataframe
    print("\n" + "="*80)
    print("📈 PERFORMANCE COMPARISON OF BEST RUNS")
    print("="*80)
    
    comparison_data = []
    for name, runs in [("Combined", runs_by_combined), ("Theta", runs_by_theta), ("Demo", runs_by_demo)]:
        run = runs[0]
        comparison_data.append({
            'Optimized For': name,
            'Run Name': run.name,
            'val/loss': f"{run.summary.get('val/loss', float('nan')):.4f}",
            'val/loss_theta': f"{run.summary.get('val/loss_theta', float('nan')):.4f}",
            'val/loss_demo': f"{run.summary.get('val/loss_demo', float('nan')):.4f}",
            'Batch Size': run.config.get('batch_size', 'N/A'),
            'LR': run.config.get('lr', 'N/A'),
            'Loss Type': run.config.get('demo_belief_loss', 'N/A'),
        })
    
    df = pd.DataFrame(comparison_data)
    print(df.to_string(index=False))
    
    print("\n" + "="*80)
    print("✅ Analysis Complete!")
    print("="*80)
    print(f"\nView full results at: {sweep.url}")


def print_top_runs(runs, metric_name, top_k):
    """Print top K runs for a given metric."""
    print(f"\n{'Rank':<6} {'Run Name':<30} {metric_name:<18} {'Batch':<8} {'LR':<10} {'Loss Type':<20}")
    print("-"*100)
    
    for i, run in enumerate(runs[:top_k], 1):
        metric_val = run.summary.get(metric_name, float('nan'))
        batch_size = run.config.get('batch_size', 'N/A')
        lr = run.config.get('lr', 'N/A')
        loss_type = run.config.get('demo_belief_loss', 'N/A')
        
        print(f"{i:<6} {run.name:<30} {metric_val:<18.6f} {batch_size:<8} {lr:<10.6f} {loss_type:<20}")


def print_config(run):
    """Print detailed configuration of a run."""
    print(f"  Run: {run.name} ({run.id})")
    print(f"  URL: {run.url}")
    print(f"\n  Performance:")
    print(f"    val/loss:       {run.summary.get('val/loss', 'N/A'):.6f}")
    print(f"    val/loss_theta: {run.summary.get('val/loss_theta', 'N/A'):.6f}")
    print(f"    val/loss_demo:  {run.summary.get('val/loss_demo', 'N/A'):.6f}")
    
    print(f"\n  Hyperparameters:")
    important_params = [
        'batch_size', 'lr', 'optimizer', 'dropout', 'num_layers', 
        'd_model', 'nhead', 'dim_feedforward', 'demo_loss_weight', 'demo_belief_loss'
    ]
    
    for param in important_params:
        value = run.config.get(param, 'N/A')
        print(f"    {param:<20}: {value}")


def main():
    parser = argparse.ArgumentParser(description="Analyze W&B sweep results")
    parser.add_argument(
        "--sweep_id", 
        type=str, 
        required=True,
        help="Full sweep ID in format: entity/project/sweep_id"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top runs to display for each metric (default: 10)"
    )
    
    args = parser.parse_args()
    
    # Validate sweep_id format
    if args.sweep_id.count('/') != 2:
        print("Error: sweep_id must be in format 'entity/project/sweep_id'")
        print("Example: juliomartinez-individual/social-belief-modeling/abc123xyz")
        return
    
    analyze_sweep(args.sweep_id, args.top_k)


if __name__ == "__main__":
    main()

