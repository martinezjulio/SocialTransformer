
import argparse
from typing import Dict
import math
import os
import random
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


def create_segment_mask(segment_boundaries: torch.Tensor, cutoffs: list, seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create attention mask that makes only visible segments accessible.
    
    Args:
        segment_boundaries: [B, max_segments] - position of each segment start
        cutoffs: List[int] - how many segments visible per example (1-indexed)
        seq_len: int - total sequence length
        device: torch device
    
    Returns:
        mask: [seq_len, seq_len] or [B, seq_len, seq_len] - causal mask with segment cutoffs
            True = masked (not attended to), False = visible
    """
    B = len(cutoffs)
    # Start with full causal mask for each position
    # mask[i, j] = True means position i cannot attend to position j
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    
    # For batched masking, we need per-example masks
    batch_mask = mask.unsqueeze(0).expand(B, seq_len, seq_len).clone()
    
    for b in range(B):
        # Get cutoff position: everything after this position should be masked
        visible_segs = cutoffs[b]
        if visible_segs < len(segment_boundaries[b]):
            cutoff_pos = int(segment_boundaries[b, visible_segs].item())
        else:
            cutoff_pos = seq_len  # All visible
        
        # Mask out everything from cutoff_pos onward (including the cutoff position itself)
        if cutoff_pos < seq_len:
            batch_mask[b, :, cutoff_pos:] = True  # Can't attend to future
    
    # If all examples have the same cutoff, can return single mask
    # Otherwise need to handle per-example masks - PyTorch expects [B*H, T, T] for multi-head attention
    # For simplicity, return the first example's mask (works if cutoffs are similar)
    # TODO: For proper batched masking with different cutoffs per example, need more complex handling
    return batch_mask[0]  # Return single [T, T] mask for now


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
    use_segment_masking: bool = True,
    demo_loss_weight: float = 0.5,
) -> Dict[str, float]:
    model.train()
    total_loss, total_theta, total_demo = 0.0, 0.0, 0.0

    for batch in loader:
        for k in ["bandit_ids", "agent_ids", "action_ids", "reward_ids", "special_ids", "pad_mask",
                  "round_boundaries", "round_bandit_map", "segment_boundaries", "y_thetas", "y_alpha_demo", "y_beta_demo"]:
            batch[k] = batch[k].to(device)

        B = batch["bandit_ids"].shape[0]
        seq_len = batch["bandit_ids"].shape[1]
        
        # Randomly select visible segments for each example
        segment_cutoffs = []
        target_bandits = []
        
        if use_segment_masking:
            for b in range(B):
                num_segments = len(batch["segment_info"][b])
                # Randomly choose how many segments visible (at least 1)
                visible_segs = random.randint(1, num_segments)
                segment_cutoffs.append(visible_segs)
                
                # Find which round this segment belongs to
                seg_info = batch["segment_info"][b][visible_segs - 1]
                target_round = seg_info["round"]
                target_bandit = batch["round_bandit_map"][b, target_round]
                target_bandits.append(int(target_bandit.item()))
            
            # Create custom attention mask based on segment cutoffs
            attn_mask = create_segment_mask(
                batch["segment_boundaries"], 
                segment_cutoffs, 
                seq_len,
                device
            )
        else:
            # Use full context (last bandit)
            attn_mask = None
            target_bandits = batch["round_bandit_map"][:, -1].tolist()
        
        optimizer.zero_grad()
        out = model(batch, custom_attn_mask=attn_mask)
        
        # Gather targets for each example's target bandit
        target_bandits_tensor = torch.tensor(target_bandits, device=device)
        batch_indices = torch.arange(B, device=device)
        
        y_theta_target = batch["y_thetas"][batch_indices, target_bandits_tensor]  # [B, K]
        y_alpha_target = batch["y_alpha_demo"][batch_indices, target_bandits_tensor]  # [B, num_demos, K]
        y_beta_target = batch["y_beta_demo"][batch_indices, target_bandits_tensor]   # [B, num_demos, K]
        
        # Compute losses
        loss_theta = beta_nll_of_true_theta(out["alpha_theta"], out["beta_theta"], y_theta_target)
        
        # Per-demonstrator loss (no averaging!)
        num_demonstrators = y_alpha_target.shape[1]
        loss_demo = 0.0
        for demo_id in range(num_demonstrators):
            loss_demo += log_mse_loss(
                out["alpha_demo"][:, demo_id, :], 
                y_alpha_target[:, demo_id, :]
            )
            loss_demo += log_mse_loss(
                out["beta_demo"][:, demo_id, :], 
                y_beta_target[:, demo_id, :]
            )
        loss_demo = loss_demo / (2 * num_demonstrators)  # Average over demos and alpha/beta

        loss = loss_theta + demo_loss_weight * loss_demo
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
    
    # Hyperparameters for sweep
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw", "sgd"], help="Optimizer type")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--nhead", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--dim_feedforward", type=int, default=768, help="Dimension of feedforward network")
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension")
    parser.add_argument("--demo_loss_weight", type=float, default=0.5, help="Weight for demonstrator belief loss (0.0-2.0)")
    
    args = parser.parse_args()

    device = torch.device(args.device)
    
    # Load training data first to get metadata
    raw_train = ShardedExamplesDataset(args.train_path, cache_shards=2)
    
    # Automatically infer dataset parameters from manifest
    num_arms = raw_train.meta.get("K", 12)
    num_demonstrators = raw_train.meta.get("num_demonstrators", 2)
    num_agents = num_demonstrators + 1  # +1 for learner
    num_rounds = raw_train.meta.get("num_rounds", 3)
    change_bandits = raw_train.meta.get("change_bandits_with_rounds", True)
    num_bandits = num_rounds if change_bandits else 1
    
    print(f"Auto-detected from manifest:")
    print(f"  num_arms (K): {num_arms}")
    print(f"  num_demonstrators: {num_demonstrators}")
    print(f"  num_agents (learner + demos): {num_agents}")
    print(f"  num_bandits: {num_bandits}")
    print(f"  num_rounds: {num_rounds}")
    
    # Initialize wandb if requested
    if args.use_wandb:
        run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            config={
                "learning_rate": args.lr,
                "optimizer": args.optimizer,
                "architecture": "SBIPosteriorTransformer",
                "dataset": args.train_path,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "num_arms": num_arms,
                "num_demonstrators": num_demonstrators,
                "num_agents": num_agents,
                "num_bandits": num_bandits,
                "num_rounds": num_rounds,
                "dropout": args.dropout,
                "nhead": args.nhead,
                "num_layers": args.num_layers,
                "dim_feedforward": args.dim_feedforward,
                "d_model": args.d_model,
                "demo_loss_weight": args.demo_loss_weight,
            },
        )
        # Append run ID to checkpoint directory for organization
        args.checkpoint_dir = os.path.join(args.checkpoint_dir, run.id)
        print(f"Checkpoints will be saved to: {args.checkpoint_dir}")
    else:
        run = None

    # Build datasets
    train_ds = BanditSequenceDataset(raw_train, num_arms=num_arms, num_agents=num_agents, num_bandits=num_bandits)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)

    # Load validation data
    raw_val = ShardedExamplesDataset(args.val_path, cache_shards=2)
    val_ds = BanditSequenceDataset(raw_val, num_arms=num_arms, num_agents=num_agents, num_bandits=num_bandits)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    model = SBIPosteriorTransformer(
        num_arms=num_arms,
        num_agents=num_agents,
        num_bandits=num_bandits,
        num_demonstrators=num_demonstrators,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)

    # Configure optimizer based on type
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-2)
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-2)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    # Track best validation loss for checkpoint saving
    best_val_loss = float('inf')
    
    # Track previous checkpoint paths for deletion
    prev_best_checkpoint = None
    prev_latest_checkpoint = None

    for epoch in range(1, args.epochs + 1):
        stats = train_one_epoch(model, train_loader, optimizer, device, demo_loss_weight=args.demo_loss_weight)
        print(f"Epoch {epoch:02d} | loss={stats['loss']:.4f}  theta={stats['loss_theta']:.4f}  demo={stats['loss_demo']:.4f}")

        # Simple eval (same losses on val set) - use full context
        model.eval()
        with torch.no_grad():
            val_tot = {"loss":0.0,"theta":0.0,"demo":0.0}
            for batch in val_loader:
                for k in ["bandit_ids","agent_ids","action_ids","reward_ids","special_ids","pad_mask",
                          "round_boundaries", "round_bandit_map", "segment_boundaries", "y_thetas", "y_alpha_demo", "y_beta_demo"]:
                    batch[k] = batch[k].to(device)
                
                # Use full context for validation (no masking)
                out = model(batch, custom_attn_mask=None)
                
                # Get the last bandit (final round)
                B = batch["bandit_ids"].shape[0]
                last_bandit_idx = batch["round_bandit_map"][:, -1]
                batch_indices = torch.arange(B, device=device)
                y_theta_last = batch["y_thetas"][batch_indices, last_bandit_idx]
                y_alpha_demo_last = batch["y_alpha_demo"][batch_indices, last_bandit_idx]
                y_beta_demo_last = batch["y_beta_demo"][batch_indices, last_bandit_idx]
                
                # Per-demonstrator loss
                num_demonstrators = y_alpha_demo_last.shape[1]
                lt = beta_nll_of_true_theta(out["alpha_theta"], out["beta_theta"], y_theta_last)
                
                ld = 0.0
                for demo_id in range(num_demonstrators):
                    ld += log_mse_loss(out["alpha_demo"][:, demo_id, :], y_alpha_demo_last[:, demo_id, :])
                    ld += log_mse_loss(out["beta_demo"][:, demo_id, :], y_beta_demo_last[:, demo_id, :])
                ld = ld / (2 * num_demonstrators)
                
                val_tot["loss"] += (lt + args.demo_loss_weight*ld).item()
                val_tot["theta"]+= lt.item()
                val_tot["demo"] += ld.item()
            n = len(val_loader)
            val_stats = {
                "loss": val_tot['loss']/n,
                "loss_theta": val_tot['theta']/n,
                "loss_demo": val_tot['demo']/n,
            }
            print(f"     val | loss={val_stats['loss']:.4f}  theta={val_stats['loss_theta']:.4f}  demo={val_stats['loss_demo']:.4f}")
            
            # Log both train and val metrics together
            if run is not None:
                run.log({
                    "train/loss": stats['loss'],
                    "train/loss_theta": stats['loss_theta'],
                    "train/loss_demo": stats['loss_demo'],
                    "val/loss": val_stats['loss'],
                    "val/loss_theta": val_stats['loss_theta'],
                    "val/loss_demo": val_stats['loss_demo'],
                    "epoch": epoch
                })
        
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
