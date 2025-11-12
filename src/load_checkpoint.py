"""
Utility script to load model checkpoints from local files or wandb artifacts.
"""

import argparse
import torch
import wandb
from models.BetaModel import SBIPosteriorTransformer


def load_checkpoint_from_file(checkpoint_path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None):
    """
    Load a checkpoint from a local file.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        
    Returns:
        Dictionary containing checkpoint metadata (epoch, stats, etc.)
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"  Train loss: {checkpoint['train_stats']['loss']:.4f}")
    print(f"  Val loss: {checkpoint['val_stats']['loss']:.4f}")
    
    return checkpoint


def load_checkpoint_from_wandb(
    entity: str,
    project: str,
    artifact_name: str,
    alias: str = "best",
    model: torch.nn.Module = None,
    optimizer: torch.optim.Optimizer = None,
):
    """
    Load a checkpoint from wandb artifacts.
    
    Args:
        entity: wandb entity (username or team)
        project: wandb project name
        artifact_name: Name of the artifact
        alias: Artifact alias (e.g., "best", "latest", "epoch_5")
        model: Optional model to load state into
        optimizer: Optional optimizer to load state into
        
    Returns:
        Path to downloaded checkpoint and checkpoint metadata
    """
    # Initialize wandb in dryrun mode (no new run created)
    api = wandb.Api()
    
    # Construct artifact path
    artifact_path = f"{entity}/{project}/{artifact_name}:{alias}"
    print(f"Downloading artifact: {artifact_path}")
    
    # Download the artifact
    artifact = api.artifact(artifact_path, type='model')
    artifact_dir = artifact.download()
    
    # Get the checkpoint file (should be the only .pt file in the directory)
    import os
    checkpoint_files = [f for f in os.listdir(artifact_dir) if f.endswith('.pt')]
    if not checkpoint_files:
        raise FileNotFoundError(f"No .pt checkpoint file found in artifact directory {artifact_dir}")
    
    checkpoint_path = os.path.join(artifact_dir, checkpoint_files[0])
    print(f"Downloaded checkpoint to: {checkpoint_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if model is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded model state")
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Loaded optimizer state")
    
    print(f"\nCheckpoint metadata:")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Train loss: {checkpoint['train_stats']['loss']:.4f}")
    print(f"  Val loss: {checkpoint['val_stats']['loss']:.4f}")
    
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
    parser.add_argument("--num_arms", type=int, default=12, help="Number of arms in bandit")
    parser.add_argument("--max_bandits", type=int, default=3, help="Max number of bandits")
    args = parser.parse_args()
    
    # Create model instance
    model = SBIPosteriorTransformer(
        num_arms=args.num_arms,
        num_agents=4,
        max_bandits=args.max_bandits,
        d_model=256, nhead=8, num_layers=4, dim_feedforward=768, dropout=0.1,
    )
    
    if args.source == "file":
        if not args.checkpoint_path:
            parser.error("--checkpoint_path is required when source=file")
        load_checkpoint_from_file(args.checkpoint_path, model)
        
    elif args.source == "wandb":
        if not args.wandb_entity or not args.wandb_project:
            parser.error("--wandb_entity and --wandb_project are required when source=wandb")
        load_checkpoint_from_wandb(
            entity=args.wandb_entity,
            project=args.wandb_project,
            artifact_name=args.artifact_name,
            alias=args.alias,
            model=model,
        )
    
    print("\n✓ Checkpoint loaded successfully!")
    print("Model is ready for inference or fine-tuning.")


if __name__ == "__main__":
    main()

