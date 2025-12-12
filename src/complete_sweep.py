"""
Script to resume incomplete runs from a wandb sweep.

This script queries a wandb sweep for runs that didn't complete successfully
(crashed, failed, or killed) and provides an interactive interface to resume them.
"""

import argparse
import subprocess
import sys
import wandb


def get_incomplete_runs(sweep_id, entity, project):
    """
    Query wandb API for incomplete runs in a sweep.
    
    Args:
        sweep_id: Wandb sweep ID
        entity: Wandb entity
        project: Wandb project
        
    Returns:
        List of dicts with run information
    """
    api = wandb.Api()
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    
    incomplete = []
    for run in sweep.runs:
        # Only include runs that didn't finish cleanly
        # Trust wandb state: 'finished' means complete (even if early stopped)
        if run.state in ['crashed', 'failed', 'killed']:
            incomplete.append({
                'id': run.id,
                'name': run.name,
                'state': run.state,
                'config': run.config,
                'epoch': run.summary.get('epoch', 0),
                'val_loss': run.summary.get('val/loss', 'N/A')
            })
    
    return incomplete


def display_and_select(runs):
    """
    Display incomplete runs and get user selection.
    
    Args:
        runs: List of run info dicts
        
    Returns:
        List of selected run info dicts
    """
    print(f"\nFound {len(runs)} incomplete runs:\n")
    for i, run in enumerate(runs, 1):
        print(f"[{i}] {run['name']} (ID: {run['id']})")
        print(f"    State: {run['state']}")
        print(f"    Epoch: {run['epoch']}/{run['config'].get('epochs', '?')}")
        print(f"    Val loss: {run['val_loss']}\n")
    
    while True:
        selection = input("Select runs to resume (e.g., '1,3,5' or 'all' or 'q' to quit): ").strip()
        
        if selection.lower() == 'q':
            return []
        
        if selection.lower() == 'all':
            return runs
        
        try:
            # Parse comma-separated indices
            indices = [int(x.strip()) - 1 for x in selection.split(',')]
            
            # Validate indices
            if all(0 <= i < len(runs) for i in indices):
                return [runs[i] for i in indices]
            else:
                print(f"ERROR: Invalid selection. Please choose numbers between 1 and {len(runs)}")
        except (ValueError, IndexError):
            print("ERROR: Invalid format. Use comma-separated numbers (e.g., '1,3,5') or 'all'")


def resume_run(run_info, entity, project):
    """
    Resume a single run by calling train.py with all config parameters.
    
    Args:
        run_info: Dict with run information
        entity: Wandb entity
        project: Wandb project
    """
    config = run_info['config']
    
    # Map config keys to train.py argument names
    # wandb config uses 'learning_rate', train.py uses '--lr'
    cmd = [
        'python', 'src/train.py',
        '--resume_from_checkpoint', run_info['id'],
        '--checkpoint_source', 'wandb',
        '--resume_run',
        '--use_wandb',
        '--wandb_entity', entity,
        '--wandb_project', project,
        '--dataset_path', config.get('dataset_path', 'data/demonstrations_data_00'),
        '--epochs', str(config['epochs']),
        '--lr', str(config['learning_rate']),
        '--batch_size', str(config['batch_size']),
        '--optimizer', config['optimizer'],
        '--dropout', str(config['dropout']),
        '--nhead', str(config['nhead']),
        '--num_layers', str(config['num_layers']),
        '--dim_feedforward', str(config['dim_feedforward']),
        '--d_model', str(config['d_model']),
        '--demo_loss_weight', str(config['demo_loss_weight']),
        '--demo_belief_loss', config['demo_belief_loss'],
        '--weight_decay', str(config.get('weight_decay', 0.01)),
        '--device', config.get('device', 'cuda'),
    ]
    
    # Add scheduler flags if used
    if config.get('scheduler'):
        cmd.extend(['--use_scheduler', '--scheduler', config['scheduler']])
        if 'warmup_epochs' in config:
            cmd.extend(['--warmup_epochs', str(config['warmup_epochs'])])
        if 'min_lr_ratio' in config:
            cmd.extend(['--min_lr_ratio', str(config['min_lr_ratio'])])
        if 'patience' in config:
            cmd.extend(['--patience', str(config['patience'])])
        if 'factor' in config:
            cmd.extend(['--factor', str(config['factor'])])
    
    if config.get('early_stopping_patience', 0) > 0:
        cmd.extend(['--early_stopping_patience', str(config['early_stopping_patience'])])
    
    print(f"\nExecuting: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Resume incomplete runs from a wandb sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Resume incomplete runs from a sweep
  python src/complete_sweep.py abc123xyz --entity myuser --project myproject
  
  # The script will:
  1. Query the sweep for crashed/failed/killed runs
  2. Display them in an interactive list
  3. Let you select which ones to resume
  4. Resume each selected run with its original configuration
        """
    )
    parser.add_argument('sweep_id', help='Wandb sweep ID')
    parser.add_argument('--entity', required=True, help='Wandb entity')
    parser.add_argument('--project', required=True, help='Wandb project')
    args = parser.parse_args()
    
    print(f"{'='*50}")
    print(f"Sweep Recovery Tool")
    print(f"{'='*50}")
    print(f"Querying sweep: {args.entity}/{args.project}/{args.sweep_id}")
    print(f"{'='*50}\n")
    
    # Get incomplete runs
    try:
        incomplete = get_incomplete_runs(args.sweep_id, args.entity, args.project)
    except Exception as e:
        print(f"ERROR: Failed to query sweep: {e}")
        sys.exit(1)
    
    if not incomplete:
        print("✓ No incomplete runs found! All runs finished successfully.")
        return
    
    # Display and select
    selected = display_and_select(incomplete)
    
    if not selected:
        print("\nNo runs selected. Exiting.")
        return
    
    print(f"\n{'='*50}")
    print(f"Will resume {len(selected)} run(s)")
    print(f"{'='*50}\n")
    
    # Confirm before starting
    confirm = input("Proceed with resuming selected runs? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return
    
    # Resume each selected run
    success_count = 0
    fail_count = 0
    
    for i, run_info in enumerate(selected, 1):
        print(f"\n{'='*50}")
        print(f"[{i}/{len(selected)}] Resuming run {run_info['id']} ({run_info['name']})")
        print(f"{'='*50}")
        
        try:
            resume_run(run_info, args.entity, args.project)
            print(f"✓ Successfully completed run {run_info['id']}")
            success_count += 1
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to resume run {run_info['id']}: Process exited with code {e.returncode}")
            fail_count += 1
        except Exception as e:
            print(f"✗ Failed to resume run {run_info['id']}: {e}")
            fail_count += 1
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Sweep Recovery Summary")
    print(f"{'='*50}")
    print(f"Total runs attempted: {len(selected)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"{'='*50}\n")
    
    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
