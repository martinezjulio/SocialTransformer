"""Utility functions for run management."""
import os
import sys
import uuid
import subprocess
from pathlib import Path
from datetime import datetime
import yaml

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def get_special_characters():
    SPECIAL = {"NONE": 0, "QUERY": 1, "BANDIT_SEP": 2, "LEARNER_RAW_ID": 99}
    return SPECIAL

def get_modeling_data_dir():
    """Return the modeling data directory from MODELING_DATA_DIR env var."""
    modeling_data_dir = os.environ.get('MODELING_DATA_DIR')
    if not modeling_data_dir:
        raise RuntimeError(
            "MODELING_DATA_DIR environment variable is not set. "
            "Source the .env file or set MODELING_DATA_DIR before running."
        )
    return Path(modeling_data_dir)

def get_training_runs_dir():
    """Return the training runs directory: $MODELING_DATA_DIR/training_runs/"""
    return get_modeling_data_dir() / 'training_runs'

def get_git_commit():
    """Return the current git commit hash, or None on failure."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None

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


def setup_training_run_directory(run_id, subdirs=None):
    """
    Create training run directory structure under $MODELING_DATA_DIR/training_runs/<run_id>/.

    Args:
        run_id: Run identifier (e.g., "wandb_abc123" or "uuid_xyz789")
        subdirs: List of subdirectories to create (default: standard structure)

    Returns:
        Path to run directory
    """
    if subdirs is None:
        subdirs = ['checkpoints', 'logs', 'results', 'results/figures', 'derived']

    run_dir = get_training_runs_dir() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    for subdir in subdirs:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)

    return run_dir


# Keep old name as alias for backward compatibility during transition
setup_run_directory = setup_training_run_directory


def setup_eval_directory():
    """
    Create eval directory structure under $MODELING_DATA_DIR/eval/.

    Returns:
        Path to eval directory
    """
    eval_dir = get_modeling_data_dir() / 'eval'
    for subdir in ['data', 'figures', 'metadata']:
        (eval_dir / subdir).mkdir(parents=True, exist_ok=True)
    return eval_dir


def setup_analysis_directory():
    """
    Create analysis directory structure under $MODELING_DATA_DIR/analysis/.

    Returns:
        Path to analysis directory
    """
    analysis_dir = get_modeling_data_dir() / 'analysis'
    for subdir in ['data', 'figures', 'metadata']:
        (analysis_dir / subdir).mkdir(parents=True, exist_ok=True)
    return analysis_dir


def write_run_yaml(run_dir, metadata, subdir=None):
    """
    Write run.yaml with run metadata.

    Args:
        run_dir: Path to run directory
        metadata: Dictionary of metadata to write
        subdir: Optional subdirectory within run_dir (e.g., 'metadata')
    """
    target_dir = Path(run_dir) / subdir if subdir else Path(run_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    run_yaml_path = target_dir / 'run.yaml'

    # Add timestamp if not present
    if 'start_time' not in metadata:
        metadata['start_time'] = datetime.now().isoformat()
    if 'status' not in metadata:
        metadata['status'] = 'running'

    # Add git commit
    git_commit = get_git_commit()
    if git_commit and 'git_commit' not in metadata:
        metadata['git_commit'] = git_commit

    with open(run_yaml_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)


def write_config_yaml(run_dir, config, subdir=None):
    """
    Write config.yaml with experiment configuration.

    Args:
        run_dir: Path to run directory
        config: Dictionary of configuration (e.g., argparse args as dict)
        subdir: Optional subdirectory within run_dir (e.g., 'metadata')
    """
    target_dir = Path(run_dir) / subdir if subdir else Path(run_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    config_yaml_path = target_dir / 'config.yaml'
    with open(config_yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def finalize_run_yaml(run_dir, status='completed', subdir=None, **kwargs):
    """
    Update run.yaml with completion information.

    Args:
        run_dir: Path to run directory
        status: Final status ('completed', 'failed', 'interrupted')
        subdir: Optional subdirectory within run_dir (e.g., 'metadata')
        **kwargs: Additional fields to add (e.g., notes, best_checkpoint)
    """
    target_dir = Path(run_dir) / subdir if subdir else Path(run_dir)
    run_yaml_path = target_dir / 'run.yaml'

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
