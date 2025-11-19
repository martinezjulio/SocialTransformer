"""
Utility script to load model checkpoints from local files or wandb artifacts.
"""

import argparse
import os
from typing import Dict, Tuple
import torch
import wandb
from models.BetaModel import SBIPosteriorTransformer


def load_checkpoint_from_local(checkpoint_path: str) -> Tuple[str, Dict]:
    """
    Load checkpoint from local file.
    
    Args:
        checkpoint_path: Path to local checkpoint file
        
    Returns:
        Tuple of (checkpoint_path, checkpoint_dict)
    """
    print(f"Loading checkpoint from local file: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint_path, checkpoint


def load_checkpoint_from_wandb_artifact(artifact_string: str, download_dir: str) -> Tuple[str, Dict]:
    """
    Load checkpoint from wandb artifact.
    
    Args:
        artifact_string: Wandb artifact path in format 'entity/project/artifact:alias'
        download_dir: Base directory for downloads
        
    Returns:
        Tuple of (local_checkpoint_path, checkpoint_dict)
    """
    print(f"Loading checkpoint from wandb artifact: {artifact_string}")
    
    # Parse artifact string
    if artifact_string.count('/') != 2 or ':' not in artifact_string:
        raise ValueError(
            f"Invalid artifact string format: {artifact_string}\n"
            "Expected format: 'entity/project/artifact:alias'"
        )
    
    # Download artifact using wandb API
    api = wandb.Api()
    artifact = api.artifact(artifact_string, type='model')
    
    # Extract project name and artifact info for directory structure
    parts = artifact_string.split('/')
    project_name = parts[1]
    artifact_with_alias = parts[2].replace(':', '_')
    
    # Create download directory
    local_dir = os.path.join(download_dir, project_name, artifact_with_alias)
    os.makedirs(local_dir, exist_ok=True)
    
    print(f"Downloading to: {local_dir}")
    artifact_dir = artifact.download(root=local_dir)
    
    # Find checkpoint file
    checkpoint_files = [f for f in os.listdir(artifact_dir) if f.endswith('.pt')]
    if not checkpoint_files:
        raise FileNotFoundError(f"No .pt checkpoint file found in {artifact_dir}")
    
    checkpoint_path = os.path.join(artifact_dir, checkpoint_files[0])
    print(f"Loaded checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint_path, checkpoint


def load_checkpoint_from_wandb_sweep(sweep_string: str, download_dir: str) -> Tuple[str, Dict]:
    """
    Load best checkpoint from wandb sweep.
    
    Args:
        sweep_string: Sweep path in format 'entity/project/sweep_id'
        download_dir: Base directory for downloads
        
    Returns:
        Tuple of (local_checkpoint_path, checkpoint_dict)
    """
    print(f"Finding best model from wandb sweep: {sweep_string}")
    
    # Parse sweep string
    if sweep_string.count('/') != 2:
        raise ValueError(
            f"Invalid sweep string format: {sweep_string}\n"
            "Expected format: 'entity/project/sweep_id'"
        )
    
    # Get sweep from wandb API
    api = wandb.Api()
    sweep = api.sweep(sweep_string)
    
    print(f"Sweep: {sweep.name}")
    print(f"Analyzing {len(sweep.runs)} runs to find best model...")
    
    # Find best run based on val/loss
    best_run = None
    best_loss = float('inf')
    
    for run in sweep.runs:
        # Check if run has val/loss metric
        if 'val/loss' in run.summary:
            val_loss = run.summary['val/loss']
            if val_loss < best_loss:
                best_loss = val_loss
                best_run = run
    
    if best_run is None:
        raise ValueError("No runs found with 'val/loss' metric in sweep")
    
    print(f"\nBest run found:")
    print(f"  Run ID: {best_run.id}")
    print(f"  Run name: {best_run.name}")
    print(f"  Val loss: {best_loss:.6f}")
    
    # Get artifacts from best run
    artifacts = list(best_run.logged_artifacts())
    model_artifacts = [a for a in artifacts if a.type == 'model']
    
    if not model_artifacts:
        raise ValueError(f"No model artifacts found for run {best_run.id}")
    
    # Find artifact with 'best' alias or use most recent
    best_artifact = None
    for artifact in model_artifacts:
        if 'best' in [alias for alias in artifact.aliases]:
            best_artifact = artifact
            break
    
    if best_artifact is None:
        # Use the most recent artifact
        best_artifact = model_artifacts[-1]
        print(f"  Note: No 'best' alias found, using most recent artifact")
    
    print(f"  Artifact: {best_artifact.name}")
    
    # Extract project name for directory structure
    parts = sweep_string.split('/')
    project_name = parts[1]
    
    # Create download directory using run ID
    local_dir = os.path.join(download_dir, project_name, best_run.id)
    os.makedirs(local_dir, exist_ok=True)
    
    print(f"Downloading to: {local_dir}")
    artifact_dir = best_artifact.download(root=local_dir)
    
    # Find checkpoint file
    checkpoint_files = [f for f in os.listdir(artifact_dir) if f.endswith('.pt')]
    if not checkpoint_files:
        raise FileNotFoundError(f"No .pt checkpoint file found in {artifact_dir}")
    
    checkpoint_path = os.path.join(artifact_dir, checkpoint_files[0])
    print(f"Loaded checkpoint: {checkpoint_path}\n")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint_path, checkpoint


def main():
    parser = argparse.ArgumentParser(description="Load a model checkpoint from file or wandb")
    parser.add_argument("--source", type=str, choices=["file", "wandb"], required=True,
                        help="Where to load checkpoint from")
    parser.add_argument("--checkpoint_path", type=str, help="Path to local checkpoint file (if source=file)")
    parser.add_argument("--wandb_entity", type=str, help="wandb entity (if source=wandb)")
    parser.add_argument("--wandb_project", type=str, help="wandb project (if source=wandb)")
    parser.add_argument("--artifact_name", type=str, default="sbi-transformer-checkpoint",
                        help="wandb artifact name (if source=wandb)")
    parser.add_argument("--alias", type=str, default="best",
                        help="wandb artifact alias: 'best', 'latest', or 'epoch_N' (if source=wandb)")
    args = parser.parse_args()
    
    # First, load the checkpoint to get model_config
    checkpoint = None
    checkpoint_path = None
    
    if args.source == "file":
        if not args.checkpoint_path:
            parser.error("--checkpoint_path is required when source=file")
        checkpoint_path = args.checkpoint_path
        print(f"Loading checkpoint from file: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
    elif args.source == "wandb":
        if not args.wandb_entity or not args.wandb_project:
            parser.error("--wandb_entity and --wandb_project are required when source=wandb")
        
        # Download artifact and get checkpoint
        api = wandb.Api()
        artifact_path = f"{args.wandb_entity}/{args.wandb_project}/{args.artifact_name}:{args.alias}"
        print(f"Downloading artifact: {artifact_path}")
        
        artifact = api.artifact(artifact_path, type='model')
        artifact_dir = artifact.download()
        
        # Get the checkpoint file
        import os
        checkpoint_files = [f for f in os.listdir(artifact_dir) if f.endswith('.pt')]
        if not checkpoint_files:
            raise FileNotFoundError(f"No .pt checkpoint file found in artifact directory {artifact_dir}")
        
        checkpoint_path = os.path.join(artifact_dir, checkpoint_files[0])
        print(f"Downloaded checkpoint to: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract model configuration from checkpoint
    if 'model_config' not in checkpoint:
        raise ValueError(
            f"Checkpoint does not contain 'model_config'. "
            "This checkpoint was saved with an older version of the training script. "
            "Please retrain your model to generate a checkpoint with the required configuration."
        )
    
    model_config = checkpoint['model_config']
    
    print(f"\nModel configuration from checkpoint:")
    print(f"  num_arms: {model_config['num_arms']}")
    print(f"  num_agents: {model_config['num_agents']}")
    print(f"  num_bandits: {model_config['num_bandits']}")
    print(f"  num_demonstrators: {model_config['num_demonstrators']}")
    print(f"  d_model: {model_config['d_model']}")
    print(f"  nhead: {model_config['nhead']}")
    print(f"  num_layers: {model_config['num_layers']}")
    print(f"  dim_feedforward: {model_config['dim_feedforward']}")
    print(f"  dropout: {model_config['dropout']}")
    
    # Create model with configuration from checkpoint
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
    )
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"\nLoaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"  Train loss: {checkpoint['train_stats']['loss']:.4f}")
    print(f"  Val loss: {checkpoint['val_stats']['loss']:.4f}")
    
    print("\n✓ Checkpoint loaded successfully!")
    print("Model is ready for inference or fine-tuning.")


if __name__ == "__main__":
    main()

