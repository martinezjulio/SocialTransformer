"""Utility functions for run management."""
import os
import sys
import uuid
from pathlib import Path
from datetime import datetime
import yaml

def get_special_characters():
    SPECIAL = {"NONE": 0, "QUERY": 1, "BANDIT_SEP": 2, "LEARNER_RAW_ID": 99}
    return SPECIAL

def generate_run_id():
    """Generate a 10-character UUID for runs without wandb."""
    return str(uuid.uuid4()).replace('-', '')[:10]


def get_run_id(use_wandb=False, wandb_run=None):
    """
    Get run ID based on context.
    
    Args:
        use_wandb: Whether wandb is being used
        wandb_run: wandb.run object if using wandb
        
    Returns:
        run_id string: "wandb_<id>" or "uuid_<id>"
    """
    if use_wandb and wandb_run:
        return f"wandb_{wandb_run.id}"
    else:
        return f"uuid_{generate_run_id()}"


def setup_run_directory(run_id, subdirs=None):
    """
    Create run directory structure under current_run/runs/<run_id>/.
    
    Args:
        run_id: Run identifier (e.g., "wandb_abc123" or "uuid_xyz789")
        subdirs: List of subdirectories to create (default: standard structure)
        
    Returns:
        Path to run directory
    """
    if subdirs is None:
        subdirs = ['checkpoints', 'logs', 'results', 'results/figures', 'derived']
    
    # Run directory is under current_run/runs/<run_id>/
    run_dir = Path('current_run') / 'runs' / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    for subdir in subdirs:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    # Create wandb directory if this is a wandb run
    if run_id.startswith('wandb_'):
        (run_dir / 'wandb').mkdir(exist_ok=True)
    
    return run_dir


def write_run_yaml(run_dir, metadata):
    """
    Write run.yaml with run metadata.
    
    Args:
        run_dir: Path to run directory
        metadata: Dictionary of metadata to write
    """
    run_yaml_path = Path(run_dir) / 'run.yaml'
    
    # Add timestamp if not present
    if 'start_time' not in metadata:
        metadata['start_time'] = datetime.now().isoformat()
    if 'status' not in metadata:
        metadata['status'] = 'running'
    
    with open(run_yaml_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)


def write_config_yaml(run_dir, config):
    """
    Write config.yaml with experiment configuration.
    
    Args:
        run_dir: Path to run directory
        config: Dictionary of configuration (e.g., argparse args as dict)
    """
    config_yaml_path = Path(run_dir) / 'config.yaml'
    with open(config_yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def finalize_run_yaml(run_dir, status='completed', **kwargs):
    """
    Update run.yaml with completion information.
    
    Args:
        run_dir: Path to run directory
        status: Final status ('completed', 'failed', 'interrupted')
        **kwargs: Additional fields to add (e.g., notes, best_checkpoint)
    """
    run_yaml_path = Path(run_dir) / 'run.yaml'
    
    if run_yaml_path.exists():
        with open(run_yaml_path, 'r') as f:
            metadata = yaml.safe_load(f) or {}
    else:
        metadata = {}
    
    metadata['end_time'] = datetime.now().isoformat()
    metadata['status'] = status
    metadata.update(kwargs)
    
    with open(run_yaml_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)
