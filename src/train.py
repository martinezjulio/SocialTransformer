
import argparse
from typing import Dict
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataloader import ShardedExamplesDataset, BanditSequenceDataset, collate_batch
from models.BetaModel import SBIPosteriorTransformer

import wandb


def beta_nll_of_true_theta(alpha_pred: torch.Tensor, beta_pred: torch.Tensor, theta_true: torch.Tensor) -> torch.Tensor:
    """
    Negative log-likelihood of the true theta under the predicted Beta(alpha_pred, beta_pred).
    Shapes: [B, K] each.
    """
    eps = 1e-6
    theta_true = theta_true.clamp(eps, 1 - eps)
    logB = torch.lgamma(alpha_pred) + torch.lgamma(beta_pred) - torch.lgamma(alpha_pred + beta_pred)
    nll = -((alpha_pred - 1) * torch.log(theta_true) + (beta_pred - 1) * torch.log(1 - theta_true) - logB)
    return nll.mean()


def log_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    MSE in log-space to compare positive parameters across scales.
    """
    return ((torch.log(pred + 1e-8) - torch.log(target + 1e-8)) ** 2).mean()


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    train_stats: Dict[str, float],
    val_stats: Dict[str, float],
    checkpoint_dir: str,
    checkpoint_name: str,
    wandb_run=None,
    artifact_name: str = "model-checkpoint",
    is_best: bool = False,
) -> str:
    """
    Save a checkpoint and optionally log it as a wandb artifact.
    
    Args:
        model: The model to save
        optimizer: The optimizer to save
        epoch: Current epoch number
        train_stats: Training statistics dict
        val_stats: Validation statistics dict
        checkpoint_dir: Directory to save checkpoints
        checkpoint_name: Name of the checkpoint file (e.g., 'checkpoint_epoch_5.pt')
        wandb_run: wandb run object (if None, only saves locally)
        artifact_name: Name for the wandb artifact
        is_best: Whether this is the best model so far
        
    Returns:
        Path to the saved checkpoint
    """
    # Create checkpoint directory if it doesn't exist
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    
    # Save checkpoint with all relevant information
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_stats': train_stats,
        'val_stats': val_stats,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")
    
    # Log to wandb artifacts if run is provided
    if wandb_run is not None:
        artifact = wandb.Artifact(
            name=artifact_name,
            type='model',
            description=f'Model checkpoint at epoch {epoch}',
            metadata={
                'epoch': epoch,
                'train_loss': train_stats['loss'],
                'val_loss': val_stats['loss'],
                'is_best': is_best,
            }
        )
        artifact.add_file(checkpoint_path)
        
        # Add aliases for versioning
        aliases = [f"epoch_{epoch}"]
        if is_best:
            aliases.append("best")
        aliases.append("latest")
        
        wandb_run.log_artifact(artifact, aliases=aliases)
        print(f"Logged checkpoint to wandb with aliases: {aliases}")
    
    return checkpoint_path


def train_one_epoch(
    model: SBIPosteriorTransformer,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    alpha0: float = 1.0,
    beta0: float = 1.0,
) -> Dict[str, float]:
    model.train()
    total_loss, total_theta, total_demo = 0.0, 0.0, 0.0

    for batch in loader:
        for k in ["bandit_ids", "agent_ids", "action_ids", "reward_ids", "special_ids", "pad_mask",
                  "y_theta", "y_alpha_demo0", "y_beta_demo0"]:
            batch[k] = batch[k].to(device)

        optimizer.zero_grad()
        out = model(batch)

        # Losses
        loss_theta = beta_nll_of_true_theta(out["alpha_theta"], out["beta_theta"], batch["y_theta"])
        # Supervise start-of-tail demo shapes (alpha0+S0, beta0+F0)
        loss_demo = log_mse_loss(out["alpha_demo0"], batch["y_alpha_demo0"]) +                     log_mse_loss(out["beta_demo0"],  batch["y_beta_demo0"])

        loss = loss_theta + 0.5 * loss_demo  # weight demo loss mildly; tune as needed
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_theta += loss_theta.item()
        total_demo += loss_demo.item()

    n = len(loader)
    return {
        "loss": total_loss / n,
        "loss_theta": total_theta / n,
        "loss_demo": total_demo / n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/demonstrations_data_00/train/manifest.json")
    parser.add_argument("--val_path", type=str, default="data/demonstrations_data_00/val/manifest.json")
    parser.add_argument("--num_arms", type=int, default=12)
    parser.add_argument("--max_bandits", type=int, default=3) # number of local bandits per context bandit, used for embedding range
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="social-belief-modeling", help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default="juliomartinez-individual", help="W&B user name or team name")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--save_every", type=int, default=0, help="Save checkpoint every N epochs (0 = only save best and final)")
    parser.add_argument("--artifact_name", type=str, default="sbi-transformer-checkpoint", help="Name for wandb artifact")
    args = parser.parse_args()

    device = torch.device(args.device)
    
    # Initialize wandb if requested
    if args.use_wandb:
        run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            config={
                "learning_rate": args.lr,
                "architecture": "SBIPosteriorTransformer",
                "dataset": args.train_path,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "num_arms": args.num_arms,
                "max_bandits": args.max_bandits 
            },
        )
        # Append run ID to checkpoint directory for organization
        args.checkpoint_dir = os.path.join(args.checkpoint_dir, run.id)
        print(f"Checkpoints will be saved to: {args.checkpoint_dir}")
    else:
        run = None

    # Load training data
    raw_train = ShardedExamplesDataset(args.train_path, cache_shards=2)
    train_ds = BanditSequenceDataset(raw_train, num_arms=args.num_arms, num_agents_max=4)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)

    # Load validation data
    raw_val = ShardedExamplesDataset(args.val_path, cache_shards=2)
    val_ds = BanditSequenceDataset(raw_val, num_arms=args.num_arms, num_agents_max=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    model = SBIPosteriorTransformer(
        num_arms=args.num_arms,
        num_agents=4,       # 1 learner (ID 0) + up to 3 demonstrators (IDs 1-3)
        max_bandits=args.max_bandits,
        d_model=256, nhead=8, num_layers=4, dim_feedforward=768, dropout=0.1,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    # Track best validation loss for checkpoint saving
    best_val_loss = float('inf')
    
    # Track previous checkpoint paths for deletion
    prev_best_checkpoint = None
    prev_latest_checkpoint = None

    for epoch in range(1, args.epochs + 1):
        stats = train_one_epoch(model, train_loader, optimizer, device)
        print(f"Epoch {epoch:02d} | loss={stats['loss']:.4f}  theta={stats['loss_theta']:.4f}  demo={stats['loss_demo']:.4f}")
        if run is not None:
            run.log({"train/loss_theta": stats['loss_theta'], "train/loss_demo": stats['loss_demo'], "train/loss": stats['loss'], "epoch": epoch}) # logging to wandb

        # Simple eval (same losses on val set)
        model.eval()
        with torch.no_grad():
            val_tot = {"loss":0.0,"theta":0.0,"demo":0.0}
            for batch in val_loader:
                for k in ["bandit_ids","agent_ids","action_ids","reward_ids","special_ids","pad_mask",
                          "y_theta","y_alpha_demo0","y_beta_demo0"]:
                    batch[k] = batch[k].to(device)
                out = model(batch)
                lt = beta_nll_of_true_theta(out["alpha_theta"], out["beta_theta"], batch["y_theta"])
                ld = log_mse_loss(out["alpha_demo0"], batch["y_alpha_demo0"]) +                      log_mse_loss(out["beta_demo0"],  batch["y_beta_demo0"])
                val_tot["loss"] += (lt + 0.5*ld).item()
                val_tot["theta"]+= lt.item()
                val_tot["demo"] += ld.item()
            n = len(val_loader)
            val_stats = {
                "loss": val_tot['loss']/n,
                "loss_theta": val_tot['theta']/n,
                "loss_demo": val_tot['demo']/n,
            }
            print(f"     val | loss={val_stats['loss']:.4f}  theta={val_stats['loss_theta']:.4f}  demo={val_stats['loss_demo']:.4f}")
            if run is not None:
                run.log({"val/loss_theta": val_stats['loss_theta'], "val/loss_demo": val_stats['loss_demo'], "val/loss": val_stats['loss'], "epoch": epoch})
        
        # Check if this is the best model so far
        is_best = val_stats['loss'] < best_val_loss
        if is_best:
            best_val_loss = val_stats['loss']
            print(f"     ✓ New best model! Val loss: {best_val_loss:.4f}")
            
            # Delete previous best checkpoint if it exists
            if prev_best_checkpoint is not None and os.path.exists(prev_best_checkpoint):
                os.remove(prev_best_checkpoint)
                print(f"     Deleted previous best checkpoint: {prev_best_checkpoint}")
            
            # Save best model checkpoint
            checkpoint_name = f"epoch_{epoch:03d}_best.pt"
            checkpoint_path = save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_stats=stats,
                val_stats=val_stats,
                checkpoint_dir=args.checkpoint_dir,
                checkpoint_name=checkpoint_name,
                wandb_run=run,
                artifact_name=args.artifact_name,
                is_best=True,
            )
            prev_best_checkpoint = checkpoint_path
        
        # Save periodic checkpoints if requested
        if args.save_every > 0 and epoch % args.save_every == 0:
            # Delete previous latest checkpoint if it exists
            if prev_latest_checkpoint is not None and os.path.exists(prev_latest_checkpoint):
                os.remove(prev_latest_checkpoint)
                print(f"     Deleted previous latest checkpoint: {prev_latest_checkpoint}")
            
            checkpoint_name = f"epoch_{epoch:03d}_latest.pt"
            checkpoint_path = save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_stats=stats,
                val_stats=val_stats,
                checkpoint_dir=args.checkpoint_dir,
                checkpoint_name=checkpoint_name,
                wandb_run=run,
                artifact_name=args.artifact_name,
                is_best=False,
            )
            prev_latest_checkpoint = checkpoint_path
    
    # Save final model as latest checkpoint
    # Delete previous latest checkpoint if it exists
    if prev_latest_checkpoint is not None and os.path.exists(prev_latest_checkpoint):
        os.remove(prev_latest_checkpoint)
        print(f"Deleted previous latest checkpoint: {prev_latest_checkpoint}")
    
    checkpoint_name = f"epoch_{args.epochs:03d}_latest.pt"
    final_checkpoint_path = save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=args.epochs,
        train_stats=stats,
        val_stats=val_stats,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name=checkpoint_name,
        wandb_run=run,
        artifact_name=args.artifact_name,
        is_best=False,
    )


if __name__ == "__main__":
    main()
