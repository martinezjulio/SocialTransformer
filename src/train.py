
import argparse
from typing import Dict
import math
import os
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataloader import ShardedExamplesDataset, BanditSequenceDataset, collate_batch, SPECIAL, LEARNER_RAW_ID
from models.BetaModel import SBIPosteriorTransformer
from transforms import BeliefNormalizer
from losses import get_demo_belief_loss, requires_normalization, beta_nll_of_true_theta, experience_loss_log, get_experience_head_loss

import wandb


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.0):
    """
    Create a learning rate scheduler with linear warmup followed by cosine decay.
    
    Args:
        optimizer: The optimizer to schedule
        num_warmup_steps: Number of steps for linear warmup
        num_training_steps: Total number of training steps
        min_lr_ratio: Minimum learning rate as ratio of initial LR (default: 0.0)
    
    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def load_checkpoint_for_resume(checkpoint_ref, source, entity=None, project=None, download_dir="checkpoints"):
    """
    Load checkpoint from local path or wandb artifact.
    
    Args:
        checkpoint_ref: Local file path or wandb run ID
        source: 'local' or 'wandb'
        entity: Wandb entity (required if source='wandb')
        project: Wandb project (required if source='wandb')
        download_dir: Directory for downloads
        
    Returns:
        checkpoint dict
    """
    if source == "local":
        if not os.path.exists(checkpoint_ref):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_ref}")
        print(f"Loading checkpoint from local file: {checkpoint_ref}")
        checkpoint = torch.load(checkpoint_ref, map_location='cpu')
        return checkpoint
        
    elif source == "wandb":
        if not entity or not project:
            raise ValueError("entity and project required for wandb source")
        
        print(f"Loading checkpoint from wandb run: {checkpoint_ref}")
        api = wandb.Api()
        run = api.run(f"{entity}/{project}/{checkpoint_ref}")
        
        # Get model artifacts
        artifacts = [a for a in run.logged_artifacts() if a.type == 'model']
        if not artifacts:
            raise ValueError(f"No model artifacts found for run {checkpoint_ref}")
        
        # Download latest artifact (or 'best' if available)
        best_artifact = None
        for artifact in artifacts:
            if 'best' in artifact.aliases:
                best_artifact = artifact
                break
        
        artifact_to_use = best_artifact if best_artifact else artifacts[-1]
        print(f"Downloading artifact: {artifact_to_use.name}")
        
        local_dir = os.path.join(download_dir, checkpoint_ref)
        artifact_dir = artifact_to_use.download(root=local_dir)
        
        # Find checkpoint file
        checkpoint_files = [f for f in os.listdir(artifact_dir) if f.endswith('.pt')]
        if not checkpoint_files:
            raise FileNotFoundError(f"No .pt file in artifact directory")
        
        checkpoint_path = os.path.join(artifact_dir, checkpoint_files[0])
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint


def create_segment_mask(
    segment_boundaries: torch.Tensor,
    cutoffs: list,
    seq_len: int,
    device: torch.device,
    masking_strategy: str = "all_bandits",
    segment_info: list = None,
    round_bandit_map: torch.Tensor = None
) -> torch.Tensor:
    """
    Create attention mask that makes only visible segments accessible.
    
    Args:
        segment_boundaries: [B, max_segments] - position of each segment start
        cutoffs: List[int] - how many segments visible per example (1-indexed)
        seq_len: int - total sequence length
        device: torch device
        masking_strategy: "all_bandits" or "current_bandit_only"
        segment_info: List[List[Dict]] - segment metadata (one list per batch item)
        round_bandit_map: [B, num_rounds] - which bandit for each round
    
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
        
        # Additional masking for "current_bandit_only" strategy
        if masking_strategy == "current_bandit_only" and segment_info is not None and round_bandit_map is not None:
            # Determine the target bandit from the cutoff segment
            seg_info_target = segment_info[b][visible_segs - 1]
            target_round = seg_info_target["round"]
            target_bandit = int(round_bandit_map[b, target_round].item())
            
            # Mask out all segments that belong to different bandits
            for seg_idx in range(visible_segs):
                seg_info_i = segment_info[b][seg_idx]
                round_i = seg_info_i["round"]
                bandit_i = int(round_bandit_map[b, round_i].item())
                
                if bandit_i != target_bandit:
                    # Mask this entire segment
                    start_pos = int(segment_boundaries[b, seg_idx].item())
                    if seg_idx + 1 < len(segment_boundaries[b]):
                        end_pos = int(segment_boundaries[b, seg_idx + 1].item())
                    else:
                        end_pos = cutoff_pos
                    
                    # Mask all positions from attending to this segment
                    batch_mask[b, :, start_pos:end_pos] = True
    
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
    normalizer=None,
    N_max: float = None,
    alpha0: float = None,
    beta0: float = None,
    demo_belief_loss: str = None,
    scheduler=None,
    model_config: dict = None,
    experience_loss_weight: float = None,
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
        scheduler: Optional learning rate scheduler to save
        
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
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'train_stats': train_stats,
        'val_stats': val_stats,
        'normalization': {
            'belief_norm_factor': normalizer.norm_factor if normalizer else None,
            'N_max': N_max,
            'alpha0': alpha0,
            'beta0': beta0,
            'normalize_beliefs': normalizer is not None,
        },
        'config': {
            'demo_belief_loss': demo_belief_loss,
            'experience_loss_weight': experience_loss_weight,
            'masking_strategy': model_config['masking_strategy'] if model_config else 'all_bandits',
            'experience_head_loss_weight': model_config.get('experience_head_loss_weight') if model_config else None,
        }
    }
    
    # Add model config if provided
    if model_config is not None:
        checkpoint['model_config'] = model_config
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
        
        # Try to log artifact to wandb, but don't crash if it fails
        try:
            wandb_run.log_artifact(artifact, aliases=aliases)
            print(f"Logged checkpoint to wandb with aliases: {aliases}")
        except Exception as e:
            print(f"Warning: Failed to log artifact to wandb: {e}")
            print(f"Checkpoint still saved locally at: {checkpoint_path}")
    
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
    demo_loss_fn=None,
    normalizer=None,
    scheduler=None,
    scheduler_type: str = "cosine",
    experience_loss_weight: float = 0.0,
    masking_strategy: str = "all_bandits",
    profile_method: str = "no_profile",
    experience_head_loss_weight: float = 1.0,
    experience_head_loss_fn=None,
    num_arms: int = 12,
) -> Dict[str, float]:
    model.train()
    total_loss, total_theta, total_demo, total_experience, total_experience_head = 0.0, 0.0, 0.0, 0.0, 0.0
    print(f"  Loading first batch from dataloader...")

    for batch_idx, batch in enumerate(loader):
        if batch_idx == 0:
            print(f"  ✓ First batch loaded! Processing...")
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
                device,
                masking_strategy=masking_strategy,
                segment_info=batch["segment_info"],
                round_bandit_map=batch["round_bandit_map"]
            )
        else:
            # Use full context (last bandit)
            attn_mask = None
            target_bandits = batch["round_bandit_map"][:, -1].tolist()
        
        optimizer.zero_grad()
        out = model(batch, custom_attn_mask=attn_mask)
        if batch_idx == 0:
            print(f"  ✓ First forward pass complete! Computing loss...")
        
        # Gather targets for each example's target bandit
        target_bandits_tensor = torch.tensor(target_bandits, device=device)
        batch_indices = torch.arange(B, device=device)
        
        y_theta_target = batch["y_thetas"][batch_indices, target_bandits_tensor]  # [B, K]
        y_alpha_target = batch["y_alpha_demo"][batch_indices, target_bandits_tensor]  # [B, num_demos, K]
        y_beta_target = batch["y_beta_demo"][batch_indices, target_bandits_tensor]   # [B, num_demos, K]
        
        # Compute losses
        loss_theta = beta_nll_of_true_theta(out["alpha_theta"], out["beta_theta"], y_theta_target)
        
        # Denormalize predictions if normalization was used
        pred_alpha = out["alpha_demo"]
        pred_beta = out["beta_demo"]
        if normalizer is not None:
            pred_alpha = normalizer.denormalize_alpha(pred_alpha)
            pred_beta = normalizer.denormalize_beta(pred_beta)
            target_alpha_denorm = normalizer.denormalize_alpha(y_alpha_target)
            target_beta_denorm = normalizer.denormalize_beta(y_beta_target)
        else:
            target_alpha_denorm = y_alpha_target
            target_beta_denorm = y_beta_target
        
        # Per-demonstrator loss using the selected loss function
        num_demonstrators = y_alpha_target.shape[1]
        loss_demo = 0.0
        for demo_id in range(num_demonstrators):
            loss_demo += demo_loss_fn(
                pred_alpha[:, demo_id, :],
                pred_beta[:, demo_id, :],
                target_alpha_denorm[:, demo_id, :],
                target_beta_denorm[:, demo_id, :]
            )
        loss_demo = loss_demo / num_demonstrators  # Average over demonstrators

        # Per-demonstrator experience loss
        loss_experience = 0.0
        for demo_id in range(num_demonstrators):
            loss_experience += experience_loss_log(
                pred_alpha[:, demo_id, :],
                pred_beta[:, demo_id, :],
                target_alpha_denorm[:, demo_id, :],
                target_beta_denorm[:, demo_id, :],
                alpha0=alpha0,
                beta0=beta0
            )
        loss_experience = loss_experience / num_demonstrators

        # Compute experience head loss if present (for profile methods)
        loss_experience_head = 0.0
        if 'N_tot_pred' in out and experience_head_loss_fn is not None:
            loss_experience_head = experience_head_loss_fn(
                out['N_tot_pred'],
                target_alpha_denorm,
                target_beta_denorm,
                alpha0=alpha0,
                beta0=beta0,
                K=num_arms
            )

        loss = (loss_theta + 
                demo_loss_weight * loss_demo + 
                experience_loss_weight * loss_experience + 
                experience_head_loss_weight * loss_experience_head)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Step scheduler after each batch only for cosine scheduler
        if scheduler is not None and scheduler_type == "cosine":
            scheduler.step()

        total_loss += loss.item()
        total_theta += loss_theta.item()
        total_demo += loss_demo.item()
        total_experience += loss_experience.item()
        if isinstance(loss_experience_head, torch.Tensor):
            total_experience_head += loss_experience_head.item()
        else:
            total_experience_head += loss_experience_head
        
        if batch_idx == 0:
            print(f"  ✓ First batch complete! loss={loss.item():.4f}")
        if (batch_idx + 1) % 50 == 0:
            print(f"    Progress: {batch_idx + 1}/{len(loader)} batches")

    n = len(loader)
    return {
        "loss": total_loss / n,
        "loss_theta": total_theta / n,
        "loss_demo": total_demo / n,
        "loss_experience": total_experience / n,
        "loss_experience_head": total_experience_head / n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="data/demonstrations_data_00",
                        help="Path to dataset directory containing dataset_metadata.json and train/val subdirectories")
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
    parser.add_argument("--demo_belief_loss", type=str, default="log_mse",
                        choices=["log_mse", "kl_divergence", "expected_value_mse", "cross_entropy"],
                        help="Loss function for demonstrator belief predictions")
    parser.add_argument("--experience_loss_weight", type=float, default=0.0, 
                        help="Weight for experience (N_tot) loss (0.0-1.0)")
    parser.add_argument("--masking_strategy", type=str, default="all_bandits",
                        choices=["all_bandits", "current_bandit_only"],
                        help="Attention masking strategy: 'all_bandits' (see all prior context) or 'current_bandit_only' (see only current bandit's context)")
    parser.add_argument("--profile_method", type=str, default="no_profile",
                        choices=["no_profile", "learned_profile", "agent_pooling", "cross_attention"],
                        help="Demonstrator profile extraction method")
    parser.add_argument("--experience_head_loss_weight", type=float, default=1.0,
                        help="Weight for direct experience head loss (only used for profile methods)")
    parser.add_argument("--experience_head_loss_type", type=str, default="mse",
                        choices=["mse", "bce"],
                        help="Loss type for experience head: 'mse' (continuous N_tot) or 'bce' (expert vs novice)")
    parser.add_argument("--expert_threshold", type=float, default=30.0,
                        help="N_tot threshold for expert classification (only used for BCE loss)")
    
    # Learning rate scheduler
    parser.add_argument("--use_scheduler", action="store_true", help="Use cosine annealing LR schedule with warmup")
    parser.add_argument("--warmup_epochs", type=int, default=2, help="Number of epochs for LR warmup")
    parser.add_argument("--min_lr_ratio", type=float, default=0.0, help="Minimum LR as ratio of initial LR")
    parser.add_argument("--scheduler", type=str, default="cosine", 
                        choices=["cosine", "plateau"],
                        help="Type of LR scheduler")
    parser.add_argument("--patience", type=int, default=5,
                        help="Patience for ReduceLROnPlateau")
    parser.add_argument("--factor", type=float, default=0.5,
                        help="Factor for ReduceLROnPlateau")
    parser.add_argument("--early_stopping_patience", type=int, default=0,
                        help="Early stopping patience (0=disabled)")
    
    # Checkpoint resumption
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Checkpoint path (local) or run ID (wandb) to resume from")
    parser.add_argument("--checkpoint_source", type=str, choices=["local", "wandb"],
                        help="Source of checkpoint: 'local' file or 'wandb' artifact")
    parser.add_argument("--resume_run", action="store_true",
                        help="Resume same wandb run ID (vs creating new run)")
    parser.add_argument("--weight_decay", type=float, default=1e-2,
                        help="Weight decay for optimizer (default: 0.01)")
    
    args = parser.parse_args()

    device = torch.device(args.device)
    
    # Load dataset metadata
    metadata_path = os.path.join(args.dataset_path, "dataset_metadata.json")
    print(f"Loading dataset metadata from {metadata_path}")
    with open(metadata_path, "r") as f:
        dataset_meta = json.load(f)
    
    # Extract normalization parameters
    N_max = dataset_meta["parameters"]["N_max"]
    alpha0 = dataset_meta["parameters"].get("alpha0", 1.0)
    beta0 = dataset_meta["parameters"].get("beta0", 1.0)
    
    print(f"Dataset parameters: N_max={N_max}, alpha0={alpha0}, beta0={beta0}")
    
    # Create normalizer if needed for the chosen loss
    if requires_normalization(args.demo_belief_loss):
        normalizer = BeliefNormalizer(N_max, alpha0, beta0)
        normalize_beliefs = True
        print(f"Using normalization with factor={normalizer.norm_factor} for loss={args.demo_belief_loss}")
    else:
        normalizer = None
        normalize_beliefs = False
        print(f"No normalization needed for loss={args.demo_belief_loss}")
    
    # Build paths to train and val manifests
    train_path = os.path.join(args.dataset_path, "train", "manifest.json")
    val_path = os.path.join(args.dataset_path, "val", "manifest.json")
    
    # Load training data first to get metadata
    raw_train = ShardedExamplesDataset(train_path, cache_shards=2)
    
    # Automatically infer dataset parameters from manifest
    num_arms = raw_train.meta.get("K", 12)
    num_demonstrators = raw_train.meta.get("num_demonstrators", 2)
    num_agents = num_demonstrators + 1  # +1 for learner
    num_rounds = raw_train.meta.get("num_rounds", 3)
    change_bandits = raw_train.meta.get("change_bandits_with_rounds", True)
    num_bandits = num_rounds if change_bandits else 1
    T_demo = raw_train.meta.get("T_demo", 6)
    T_learner = raw_train.meta.get("T_learner", 6)
    
    print(f"Auto-detected from manifest:")
    print(f"  num_arms (K): {num_arms}")
    print(f"  num_demonstrators: {num_demonstrators}")
    print(f"  num_agents (learner + demos): {num_agents}")
    print(f"  num_bandits: {num_bandits}")
    print(f"  num_rounds: {num_rounds}")
    print(f"  T_demo: {T_demo}")
    print(f"  T_learner: {T_learner}")
    
    # Initialize wandb if requested
    if args.use_wandb:
        # Determine run ID and resume mode for checkpoint resumption
        run_id = None
        resume_mode = "never"
        
        if args.resume_from_checkpoint and args.resume_run:
            # When resuming same run, use the checkpoint reference as run ID (if from wandb)
            if args.checkpoint_source == "wandb":
                run_id = args.resume_from_checkpoint
                resume_mode = "allow"
                print(f"Will resume wandb run: {run_id}")
            else:
                # For local checkpoints with --resume_run, warn user
                print("Warning: --resume_run with local checkpoint creates new run (cannot extract run ID from local file)")
        
        # When running in a sweep, wandb.init() is already called by the sweep agent
        # We pass entity and project only if not already in a sweep context
        run = wandb.init(
            id=run_id,
            resume=resume_mode,
            entity=args.wandb_entity if not wandb.run else None,
            project=args.wandb_project if not wandb.run else None,
            config={
                "learning_rate": args.lr,
                "optimizer": args.optimizer,
                "weight_decay": args.weight_decay,
                "architecture": "SBIPosteriorTransformer",
                "dataset_path": args.dataset_path,
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
                "demo_belief_loss": args.demo_belief_loss,
                "experience_loss_weight": args.experience_loss_weight,
                "masking_strategy": args.masking_strategy,
                "profile_method": args.profile_method,
                "experience_head_loss_weight": args.experience_head_loss_weight,
                "experience_head_loss_type": args.experience_head_loss_type,
                "expert_threshold": args.expert_threshold,
                "normalize_beliefs": normalize_beliefs,
                "N_max": N_max,
                "use_scheduler": args.use_scheduler,
                "warmup_epochs": args.warmup_epochs if args.use_scheduler else None,
                "min_lr_ratio": args.min_lr_ratio if args.use_scheduler else None,
                "scheduler_type": args.scheduler if args.use_scheduler else None,
                "patience": args.patience if (args.use_scheduler and args.scheduler == "plateau") else None,
                "factor": args.factor if (args.use_scheduler and args.scheduler == "plateau") else None,
                "early_stopping_patience": args.early_stopping_patience if args.early_stopping_patience > 0 else None,
            },
        )
        # Append run ID to checkpoint directory for organization
        args.checkpoint_dir = os.path.join(args.checkpoint_dir, run.id)
        print(f"Checkpoints will be saved to: {args.checkpoint_dir}")
    else:
        run = None

    # Build datasets with normalization
    train_ds = BanditSequenceDataset(
        raw_train, 
        num_arms=num_arms, 
        num_agents=num_agents, 
        num_bandits=num_bandits,
        normalize_beliefs=normalize_beliefs,
        normalizer=normalizer
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)  # shuffle=False: rely on epoch shard shuffling + pre-randomized data

    # Load validation data
    raw_val = ShardedExamplesDataset(val_path, cache_shards=2)
    val_ds = BanditSequenceDataset(
        raw_val, 
        num_arms=num_arms, 
        num_agents=num_agents, 
        num_bandits=num_bandits,
        normalize_beliefs=normalize_beliefs,
        normalizer=normalizer
    )
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
        profile_method=args.profile_method,
        experience_head_loss_type=args.experience_head_loss_type,
    ).to(device)
    
    # Store model configuration for saving in checkpoint
    model_config = {
        'num_arms': num_arms,
        'num_agents': num_agents,
        'num_bandits': num_bandits,
        'num_demonstrators': num_demonstrators,
        'd_model': args.d_model,
        'nhead': args.nhead,
        'num_layers': args.num_layers,
        'dim_feedforward': args.dim_feedforward,
        'dropout': args.dropout,
        'T_demo': T_demo,
        'T_learner': T_learner,
        'SPECIAL': SPECIAL,
        'LEARNER_RAW_ID': LEARNER_RAW_ID,
        'masking_strategy': args.masking_strategy,
        'profile_method': args.profile_method,
        'experience_head_loss_type': args.experience_head_loss_type,
    }

    # Configure optimizer based on type
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    # Configure learning rate scheduler with warmup
    scheduler = None
    if args.use_scheduler:
        if args.scheduler == "cosine":
            num_training_steps = len(train_loader) * args.epochs
            num_warmup_steps = len(train_loader) * args.warmup_epochs
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                min_lr_ratio=args.min_lr_ratio
            )
            print(f"Using cosine LR schedule with {args.warmup_epochs} epoch warmup")
            print(f"  Total training steps: {num_training_steps}")
            print(f"  Warmup steps: {num_warmup_steps}")
            print(f"  Min LR ratio: {args.min_lr_ratio}")
        elif args.scheduler == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=args.factor,
                patience=args.patience,
                min_lr=args.lr * args.min_lr_ratio
            )
            print(f"Using ReduceLROnPlateau scheduler")
            print(f"  Patience: {args.patience}, Factor: {args.factor}")
            print(f"  Min LR: {args.lr * args.min_lr_ratio:.2e}")

    # Initialize training state
    start_epoch = 1
    best_val_loss = float('inf')
    
    # Load checkpoint if resuming
    if args.resume_from_checkpoint:
        print(f"\n{'='*50}")
        print("Loading checkpoint for resume...")
        print(f"{'='*50}")
        
        checkpoint = load_checkpoint_for_resume(
            args.resume_from_checkpoint,
            args.checkpoint_source,
            entity=args.wandb_entity if args.use_wandb else None,
            project=args.wandb_project if args.use_wandb else None
        )
        
        # Check if training already complete
        checkpoint_epoch = checkpoint['epoch']
        if checkpoint_epoch >= args.epochs:
            print(f"\n{'='*50}")
            print("Training already complete!")
            print(f"{'='*50}")
            print(f"Checkpoint at epoch {checkpoint_epoch}")
            print(f"Target epochs: {args.epochs}")
            print(f"\nTo continue training, specify more epochs:")
            print(f"  --epochs {checkpoint_epoch + 50}")
            print(f"{'='*50}\n")
            exit(0)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Loaded model weights")
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("✓ Loaded optimizer state")
        
        # Load scheduler if exists and we're using one
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("✓ Loaded scheduler state")
        
        # Set starting epoch and best val loss
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['val_stats']['loss']
        
        print(f"\nResuming from epoch {checkpoint['epoch']}")
        print(f"Will train from epoch {start_epoch} to {args.epochs}")
        print(f"Best val loss so far: {best_val_loss:.4f}")
        print(f"{'='*50}\n")
    
    # Early stopping tracking
    early_stopping_counter = 0
    
    # Track previous checkpoint paths for deletion
    prev_best_checkpoint = None
    prev_latest_checkpoint = None

    # Get the demo belief loss function
    demo_loss_fn = get_demo_belief_loss(args.demo_belief_loss)
    print(f"Using {args.demo_belief_loss} loss for demonstrator beliefs")
    
    # Get the experience head loss function (only for profile methods)
    if args.profile_method != "no_profile":
        experience_head_loss_fn = get_experience_head_loss(args.experience_head_loss_type)
        print(f"Using {args.experience_head_loss_type} loss for experience head")
        print(f"Experience head loss weight: {args.experience_head_loss_weight}")
    else:
        experience_head_loss_fn = None
    
    print(f"\nStarting training loop for {args.epochs} epochs...")
    print(f"Training batches per epoch: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"\nNote: First batch load may take 2-3 seconds (loading shard from disk)...")
    
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'='*50}")
        print(f"Starting Epoch {epoch}/{args.epochs}")
        print(f"{'='*50}")
        
        # Shuffle shard order for better randomization
        if hasattr(raw_train, 'shuffle_shards'):
            raw_train.shuffle_shards()
        
        stats = train_one_epoch(
            model, train_loader, optimizer, device, 
            alpha0=alpha0,
            beta0=beta0,
            demo_loss_weight=args.demo_loss_weight,
            demo_loss_fn=demo_loss_fn,
            normalizer=normalizer,
            scheduler=scheduler,
            scheduler_type=args.scheduler if args.use_scheduler else "cosine",
            experience_loss_weight=args.experience_loss_weight,
            masking_strategy=args.masking_strategy,
            profile_method=args.profile_method,
            experience_head_loss_weight=args.experience_head_loss_weight,
            experience_head_loss_fn=experience_head_loss_fn,
            num_arms=num_arms
        )
        
        # Log current learning rate if scheduler is used
        current_lr = optimizer.param_groups[0]['lr']
        log_str = f"Epoch {epoch:02d} | loss={stats['loss']:.4f}  theta={stats['loss_theta']:.4f}  demo={stats['loss_demo']:.4f}  exp={stats['loss_experience']:.4f}"
        if 'loss_experience_head' in stats and stats['loss_experience_head'] > 0:
            log_str += f"  exp_head={stats['loss_experience_head']:.4f}"
        log_str += f"  lr={current_lr:.2e}"
        print(log_str)

        # Simple eval (same losses on val set) - use full context
        model.eval()
        with torch.no_grad():
            val_tot = {"loss":0.0,"theta":0.0,"demo":0.0,"experience":0.0,"experience_head":0.0}
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
                
                # Denormalize predictions if needed
                pred_alpha_val = out["alpha_demo"]
                pred_beta_val = out["beta_demo"]
                if normalizer is not None:
                    pred_alpha_val = normalizer.denormalize_alpha(pred_alpha_val)
                    pred_beta_val = normalizer.denormalize_beta(pred_beta_val)
                    target_alpha_val = normalizer.denormalize_alpha(y_alpha_demo_last)
                    target_beta_val = normalizer.denormalize_beta(y_beta_demo_last)
                else:
                    target_alpha_val = y_alpha_demo_last
                    target_beta_val = y_beta_demo_last
                
                # Per-demonstrator loss
                num_demonstrators = y_alpha_demo_last.shape[1]
                lt = beta_nll_of_true_theta(out["alpha_theta"], out["beta_theta"], y_theta_last)
                
                ld = 0.0
                for demo_id in range(num_demonstrators):
                    ld += demo_loss_fn(
                        pred_alpha_val[:, demo_id, :],
                        pred_beta_val[:, demo_id, :],
                        target_alpha_val[:, demo_id, :],
                        target_beta_val[:, demo_id, :]
                    )
                ld = ld / num_demonstrators
                
                # Per-demonstrator experience loss
                le = 0.0
                for demo_id in range(num_demonstrators):
                    le += experience_loss_log(
                        pred_alpha_val[:, demo_id, :],
                        pred_beta_val[:, demo_id, :],
                        target_alpha_val[:, demo_id, :],
                        target_beta_val[:, demo_id, :],
                        alpha0=alpha0,
                        beta0=beta0
                    )
                le = le / num_demonstrators
                
                # Compute experience head loss if present
                le_head = 0.0
                if 'N_tot_pred' in out and experience_head_loss_fn is not None:
                    le_head = experience_head_loss_fn(
                        out['N_tot_pred'],
                        target_alpha_val,
                        target_beta_val,
                        alpha0=alpha0,
                        beta0=beta0,
                        K=num_arms
                    )
                
                val_tot["loss"] += (lt + args.demo_loss_weight*ld + args.experience_loss_weight*le + args.experience_head_loss_weight*le_head).item()
                val_tot["theta"]+= lt.item()
                val_tot["demo"] += ld.item()
                val_tot["experience"] += le.item()
                if isinstance(le_head, torch.Tensor):
                    val_tot["experience_head"] += le_head.item()
                else:
                    val_tot["experience_head"] += le_head
            n = len(val_loader)
            val_stats = {
                "loss": val_tot['loss']/n,
                "loss_theta": val_tot['theta']/n,
                "loss_demo": val_tot['demo']/n,
                "loss_experience": val_tot['experience']/n,
                "loss_experience_head": val_tot['experience_head']/n,
            }
            val_log_str = f"     val | loss={val_stats['loss']:.4f}  theta={val_stats['loss_theta']:.4f}  demo={val_stats['loss_demo']:.4f}  exp={val_stats['loss_experience']:.4f}"
            if val_stats['loss_experience_head'] > 0:
                val_log_str += f"  exp_head={val_stats['loss_experience_head']:.4f}"
            print(val_log_str)
            
            # Step plateau scheduler if using it
            if scheduler is not None and args.scheduler == "plateau":
                scheduler.step(val_stats['loss'])
            
            # Log both train and val metrics together
            if run is not None:
                log_dict = {
                    "train/loss": stats['loss'],
                    "train/loss_theta": stats['loss_theta'],
                    "train/loss_demo": stats['loss_demo'],
                    "train/loss_experience": stats['loss_experience'],
                    "val/loss": val_stats['loss'],
                    "val/loss_theta": val_stats['loss_theta'],
                    "val/loss_demo": val_stats['loss_demo'],
                    "val/loss_experience": val_stats['loss_experience'],
                    "learning_rate": current_lr,
                    "epoch": epoch
                }
                # Add experience head loss if present
                if 'loss_experience_head' in stats and stats['loss_experience_head'] > 0:
                    log_dict["train/loss_experience_head"] = stats['loss_experience_head']
                if 'loss_experience_head' in val_stats and val_stats['loss_experience_head'] > 0:
                    log_dict["val/loss_experience_head"] = val_stats['loss_experience_head']
                run.log(log_dict)
        
        # Check if this is the best model so far
        is_best = val_stats['loss'] < best_val_loss
        if is_best:
            best_val_loss = val_stats['loss']
            early_stopping_counter = 0
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
                normalizer=normalizer,
                N_max=N_max,
                alpha0=alpha0,
                beta0=beta0,
                demo_belief_loss=args.demo_belief_loss,
                scheduler=scheduler,
                model_config=model_config,
                experience_loss_weight=args.experience_loss_weight,
            )
            prev_best_checkpoint = checkpoint_path
        else:
            # No improvement
            early_stopping_counter += 1
            if args.early_stopping_patience > 0 and early_stopping_counter >= args.early_stopping_patience:
                print(f"\n{'='*50}")
                print(f"Early stopping triggered after {epoch} epochs")
                print(f"No improvement for {args.early_stopping_patience} epochs")
                print(f"Best val loss: {best_val_loss:.4f}")
                print(f"{'='*50}")
                break
        
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
                normalizer=normalizer,
                N_max=N_max,
                alpha0=alpha0,
                beta0=beta0,
                demo_belief_loss=args.demo_belief_loss,
                scheduler=scheduler,
                model_config=model_config,
                experience_loss_weight=args.experience_loss_weight,
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
        normalizer=normalizer,
        N_max=N_max,
        alpha0=alpha0,
        beta0=beta0,
        demo_belief_loss=args.demo_belief_loss,
        scheduler=scheduler,
        model_config=model_config,
        experience_loss_weight=args.experience_loss_weight,
    )


if __name__ == "__main__":
    main()
