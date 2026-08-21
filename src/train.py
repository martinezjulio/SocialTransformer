
import argparse
from typing import Dict
import math
import os
import sys
import json
import random
import re
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data.dataloader import (
    load_arrow_split,
    make_arrow_loader,
    SPECIAL,
    LEARNER_RAW_ID,
    SEGMENT_TYPE_TO_ID,
)
from models import build_model, BaseSocialTransformer
from data.transforms import BeliefNormalizer
from models.losses import get_env_theta_loss, get_demo_belief_loss, requires_normalization, experience_loss_log, get_experience_head_loss
from utils import get_run_id, setup_training_run_directory, write_run_yaml, write_config_yaml, finalize_run_yaml, get_training_runs_dir

from tqdm import tqdm

import wandb


def setup_distributed(fallback_device: str):
    """Detect torchrun env vars and initialize the DDP process group.

    Returns a tuple ``(is_distributed, rank, world_size, local_rank, device, is_rank_0)``.
    When the script is launched without torchrun (no RANK/WORLD_SIZE in env), this
    is a no-op that returns single-process defaults.
    """
    is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if is_distributed:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        # Device-agnostic backend selection (CUDA -> nccl, ROCm -> rccl,
        # XPU -> xccl, etc.). Requires PyTorch >= 2.6.
        acc = torch.accelerator.current_accelerator()
        backend = torch.distributed.get_default_backend_for_device(acc)
        torch.accelerator.set_device_index(local_rank)
        dist.init_process_group(backend=backend)
        device = torch.device(f"{acc.type}:{local_rank}")
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device(fallback_device)
    is_rank_0 = (rank == 0)
    return is_distributed, rank, world_size, local_rank, device, is_rank_0


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


def create_segment_mask(
    segment_boundaries: torch.Tensor,
    cutoffs: list,
    seq_len: int,
    device: torch.device,
    masking_strategy: str = "all_bandits",
    segment_round: list = None,
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
        segment_round: List[List[int]] - per-example list of round indices,
            one entry per segment (replaces the old segment_info list-of-dicts;
            only the "round" field was ever read here)
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
        if masking_strategy == "current_bandit_only" and segment_round is not None and round_bandit_map is not None:
            # Determine the target bandit from the cutoff segment
            target_round = segment_round[b][visible_segs - 1]
            target_bandit = int(round_bandit_map[b, target_round].item())

            # Mask out all segments that belong to different bandits
            for seg_idx in range(visible_segs):
                round_i = segment_round[b][seg_idx]
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
    max_experience: float = None,
    alpha0: float = None,
    beta0: float = None,
    env_theta_loss: str = None, 
    demo_belief_loss: str = None,
    scheduler=None,
    model_config: dict = None,
    experience_loss_weight: float = None,
    log_to_wandb: bool = True,
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
        log_to_wandb: Whether to log the checkpoint to wandb (default: True)
        
    Returns:
        Path to the saved checkpoint
    """
    # Create checkpoint directory if it doesn't exist
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    
    # Extras are merged with model_state_dict inside model.save_checkpoint.
    extras = {
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'train_stats': train_stats,
        'val_stats': val_stats,
        'normalization': {
            'belief_norm_factor': normalizer.norm_factor if normalizer else None,
            'expert_N_max': max_experience,
            'alpha0': alpha0,
            'beta0': beta0,
            'normalize_beliefs': normalizer is not None,
        },
        'config': {
            'env_theta_loss': env_theta_loss,
            'demo_belief_loss': demo_belief_loss,
            'experience_loss_weight': experience_loss_weight,
            'masking_strategy': model_config['masking_strategy'] if model_config else 'all_bandits',
            'experience_head_loss_weight': model_config.get('experience_head_loss_weight') if model_config else None,
        }
    }
    if model_config is not None:
        extras['model_config'] = model_config
    model.save_checkpoint(checkpoint_path, **extras)
    print(f"Saved checkpoint to {checkpoint_path}")
    
    # Log to wandb artifacts if run is provided and logging is enabled
    if wandb_run is not None and log_to_wandb:
        # Include run ID in artifact name to separate runs in sweeps
        artifact_name_with_run = f"{artifact_name}-run-{wandb_run.id}"
        
        artifact = wandb.Artifact(
            name=artifact_name_with_run,
            type='model',
            description=f'Model checkpoint for run {wandb_run.id} at epoch {epoch}',
            metadata={
                'epoch': epoch,
                'train_loss': train_stats['loss'],
                'val_loss': val_stats['loss'],
                'is_best': is_best,
                'run_id': wandb_run.id,
            }
        )
        # Add file with clean epoch-only name in artifact (local file keeps descriptive name)
        artifact.add_file(checkpoint_path, name=f"epoch_{epoch:03d}.pt")
        
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


def log_dataset_artifact(wandb_run, dataset_path: str, dataset_meta: dict) -> None:
    """Log the training dataset's metadata as a wandb artifact (type='dataset').

    Attaches the root metadata.json only — never the Arrow data files themselves
    (they live on the workstation's TRAINING_DATA_ROOT_DIR; the artifact tracks
    *which* dataset was used, not the rows). wandb content-hash dedup avoids
    spurious new versions when metadata.json is unchanged across re-runs.
    """
    if wandb_run is None:
        return
    try:
        params = dataset_meta.get("parameters", {})
        ds_basename = os.path.basename(os.path.normpath(dataset_path))
        # wandb artifact names allow [A-Za-z0-9._-]; sanitize defensively
        artifact_name = "dataset-" + re.sub(r"[^A-Za-z0-9._-]", "_", ds_basename)

        artifact = wandb.Artifact(
            name=artifact_name,
            type="dataset",
            description=f"Training dataset '{ds_basename}' (mmap-backed Arrow)",
            metadata={
                "dataset_path": dataset_path,
                "dataset_name": dataset_meta.get("dataset_name", ds_basename),
                "generated_at": dataset_meta.get("generated_at"),
                "format": dataset_meta.get("format", "huggingface_arrow"),
                "K": params.get("K"),
                "num_demonstrators": params.get("num_demonstrators"),
                "num_rounds": params.get("num_rounds"),
                "num_train_runs": params.get("num_train_runs"),
                "num_valid_runs": params.get("num_valid_runs"),
                "num_test_runs": params.get("num_test_runs"),
                "expert_N_max": params.get("expert_N_max"),
            },
        )

        # Root metadata.json (the only on-disk file we attach; Arrow data
        # is large and machine-local, tracked by path reference below).
        meta_file = os.path.join(dataset_path, "metadata.json")
        if os.path.isfile(meta_file):
            artifact.add_file(meta_file, name="metadata.json")

        # Track the dataset directory path itself without copying its contents
        artifact.add_reference(f"file://{dataset_path}", name="dataset_dir", checksum=False)

        wandb_run.log_artifact(artifact)
        print(f"Logged dataset artifact '{artifact_name}' to wandb")
    except Exception as e:
        print(f"Warning: Failed to log dataset artifact to wandb: {e}")


def train_one_epoch(
    model: BaseSocialTransformer,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    alpha0: float = 1.0,
    beta0: float = 1.0,
    use_segment_masking: bool = True,
    env_loss_fn=None,
    env_theta_loss: str = "bce",
    demo_loss_weight: float = 0.5,
    demo_loss_fn=None,
    normalizer=None,
    scheduler=None,
    scheduler_type: str = "cosine",
    experience_loss_weight: float = 0.0,
    masking_strategy: str = "all_bandits",
    model_type: str = "social",
    experience_head_loss_weight: float = 1.0,
    experience_head_loss_fn=None,
    num_arms: int = 12,
    is_rank_0: bool = True,
) -> Dict[str, float]:
    model.train()
    total_loss, total_theta, total_demo, total_experience, total_experience_head = 0.0, 0.0, 0.0, 0.0, 0.0
    if is_rank_0:
        print(f"  Loading first batch from dataloader...")

    pbar = tqdm(loader, desc="train", unit="b", disable=not is_rank_0, leave=False)
    for batch_idx, batch in enumerate(pbar):
        if batch_idx == 0 and is_rank_0:
            tqdm.write(f"  ✓ First batch loaded! Processing...")
            tqdm.write(f"  Debug: bandit_ids range = [{batch['bandit_ids'].min().item()}, {batch['bandit_ids'].max().item()}]")
            tqdm.write(f"  Debug: agent_ids range = [{batch['agent_ids'].min().item()}, {batch['agent_ids'].max().item()}]")
            tqdm.write(f"  Debug: special_ids range = [{batch['special_ids'].min().item()}, {batch['special_ids'].max().item()}]")
            tqdm.write(f"  Debug: special_ids unique values = {batch['special_ids'].unique().tolist()}")
            tqdm.write(f"  Debug: num QUERY tokens (special_id=1) = {(batch['special_ids'] == 1).sum().item()}")
            tqdm.write(f"  Debug: round_bandit_map range = [{batch['round_bandit_map'].min().item()}, {batch['round_bandit_map'].max().item()}]")
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
                num_segments = len(batch["segment_round"][b])
                # Randomly choose how many segments visible (at least 1)
                visible_segs = random.randint(1, num_segments)
                segment_cutoffs.append(visible_segs)

                # Find which round this segment belongs to
                target_round = batch["segment_round"][b][visible_segs - 1]
                target_bandit = batch["round_bandit_map"][b, target_round]
                target_bandits.append(int(target_bandit.item()))

            # Create custom attention mask based on segment cutoffs
            attn_mask = create_segment_mask(
                batch["segment_boundaries"],
                segment_cutoffs,
                seq_len,
                device,
                masking_strategy=masking_strategy,
                segment_round=batch["segment_round"],
                round_bandit_map=batch["round_bandit_map"],
            )
        else:
            # Use full context (last bandit)
            attn_mask = None
            target_bandits = batch["round_bandit_map"][:, -1].tolist()
        
        optimizer.zero_grad()
        out = model(batch, custom_attn_mask=attn_mask)
        if batch_idx == 0 and is_rank_0:
            tqdm.write(f"  ✓ First forward pass complete! Computing loss...")
        
        # Gather targets for each example's target bandit
        target_bandits_tensor = torch.tensor(target_bandits, device=device)
        batch_indices = torch.arange(B, device=device)
        
        y_theta_target = batch["y_thetas"][batch_indices, target_bandits_tensor]  # [B, K]
        y_alpha_target = batch["y_alpha_demo"][batch_indices, target_bandits_tensor]  # [B, num_demos, K]
        y_beta_target = batch["y_beta_demo"][batch_indices, target_bandits_tensor]   # [B, num_demos, K]
        
        # Extract demo predictions for target bandits
        # out["alpha_demo"] is [B, num_queries, D, K]; target_bandits_tensor is [B] with bandit indices
        pred_alpha_demo = out["alpha_demo"][batch_indices, target_bandits_tensor]    # [B, D, K]
        pred_beta_demo = out["beta_demo"][batch_indices, target_bandits_tensor]      # [B, D, K]

        # Environment theta loss: branch on the configured loss function
        if env_theta_loss == "bce":
            pred_theta_logits = out["theta_logits"][batch_indices, target_bandits_tensor]  # [B, K]
            loss_theta = env_loss_fn(pred_theta_logits, y_theta_target)
        else:
            pred_alpha_theta = out["alpha_theta"][batch_indices, target_bandits_tensor]  # [B, K]
            pred_beta_theta = out["beta_theta"][batch_indices, target_bandits_tensor]    # [B, K]
            loss_theta = env_loss_fn(pred_alpha_theta, pred_beta_theta, y_theta_target)
        
        # Compute remaining losses (demonstrator beliefs, experience)
        
        # Denormalize predictions if normalization was used
        if normalizer is not None:
            pred_alpha = normalizer.denormalize_alpha(pred_alpha_demo)
            pred_beta = normalizer.denormalize_beta(pred_beta_demo)
            target_alpha_denorm = normalizer.denormalize_alpha(y_alpha_target)
            target_beta_denorm = normalizer.denormalize_beta(y_beta_target)
        else:
            pred_alpha = pred_alpha_demo
            pred_beta = pred_beta_demo
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
        
        if batch_idx == 0 and is_rank_0:
            tqdm.write(f"  ✓ First batch complete! loss={loss.item():.4f}")
        if (batch_idx + 1) % 50 == 0 and is_rank_0:
            seen = batch_idx + 1
            pbar.set_postfix(loss=f"{total_loss / seen:.4f}",
                             theta=f"{total_theta / seen:.4f}",
                             demo=f"{total_demo / seen:.4f}",
                             exp=f"{total_experience / seen:.4f}")

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
    parser.add_argument("--dataset_name", type=str,
                        default="demonstrations_data_02",
                        help="Dataset directory name (resolved against TRAINING_DATA_ROOT_DIR)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for weight init, shuffling, and the per-example "
                             "segment-cutoff draws. Matches the generation/inference default.")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader worker processes (0 = load in the main process). "
                             "Workers prefetch while the GPU computes; 4 is enough to hide "
                             "data time at batch_size=128.")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default=os.environ.get("WANDB_PROJECT", "social-belief-modeling"), help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=os.environ.get("WANDB_ENTITY", "juliomartinez-individual"), help="W&B user name or team name")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--save_every", type=int, default=0, help="Save checkpoint every N epochs (0 = only save best and final)")
    parser.add_argument("--artifact_name", type=str, default="sbi-transformer", help="Name for wandb artifact")
    
    # Hyperparameters for sweep
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw", "sgd"], help="Optimizer type")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--nhead", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--dim_feedforward", type=int, default=768, help="Dimension of feedforward network")
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension")
    parser.add_argument("--env_theta_loss", type=str, default="bce", 
                        choices=["bce", "beta_nll"], 
                        help="Loss function for the environment payoffs")
    parser.add_argument("--demo_loss_weight", type=float, default=0.5, help="Weight for demonstrator belief loss (0.0-2.0)")
    parser.add_argument("--demo_belief_loss", type=str, default="log_mse",
                        choices=["log_mse", "kl_divergence", "expected_value_mse", "cross_entropy"],
                        help="Loss function for demonstrator belief predictions")
    parser.add_argument("--experience_loss_weight", type=float, default=0.0, 
                        help="Weight for experience (N_tot) loss (0.0-1.0)")
    parser.add_argument("--masking_strategy", type=str, default="all_bandits",
                        choices=["all_bandits", "current_bandit_only"],
                        help="Attention masking strategy: 'all_bandits' (see all prior context) or 'current_bandit_only' (see only current bandit's context)")
    parser.add_argument("--model_type", type=str, default="social",
                        choices=[
                            "social",
                            "social_profile_learned",
                            "social_profile_agent_pooling",
                            "social_profile_cross_attention",
                        ],
                        help="SocialTransformer variant to instantiate")
    parser.add_argument("--experience_head_loss_weight", type=float, default=1.0,
                        help="Weight for direct experience head loss (only used for profile model_types)")
    parser.add_argument("--experience_head_loss_type", type=str, default="mse",
                        choices=["mse", "bce"],
                        help="Loss type for experience head: 'mse' (continuous N_tot) or 'bce' (expert vs novice)")
    parser.add_argument("--expert_threshold", type=float, default=30.0,
                        help="N_tot threshold for expert classification (only used for BCE loss)")
    
    # Head input configurations (Phase 3)
    parser.add_argument("--theta_head_input", type=str, default="query_only",
                        choices=["query_only", "query_and_profiles"],
                        help="Input mode for theta head")
    parser.add_argument("--demo_beliefs_head_input", type=str, default="query_and_profiles",
                        choices=["query_only", "profiles_only", "query_and_profiles"],
                        help="Input mode for demo beliefs head (ignored when model_type='social')")
    parser.add_argument("--experience_head_input", type=str, default="profiles_only",
                        choices=["query_only", "profiles_only", "query_and_profiles"],
                        help="Input mode for experience head (ignored when model_type='social')")
    
    # Learning rate scheduler
    parser.add_argument("--use_scheduler", action="store_true", help="Use cosine annealing LR schedule with warmup")
    parser.add_argument("--warmup_epochs", type=int, default=2, help="Number of epochs for LR warmup")
    parser.add_argument("--min_lr", type=float, default=1e-8,
                        help="Minimum learning rate (absolute value, not ratio)")
    parser.add_argument("--scheduler", type=str, default="cosine", 
                        choices=["cosine", "plateau"],
                        help="Type of LR scheduler")
    parser.add_argument("--patience", type=int, default=5,
                        help="Patience for ReduceLROnPlateau")
    parser.add_argument("--factor", type=float, default=0.5,
                        help="Factor for ReduceLROnPlateau")
    
    # Overfitting-based early stopping
    parser.add_argument("--overfitting_threshold", type=float, default=10.0,
                        help="Percentage threshold above best val loss to detect overfitting (0=disabled)")
    parser.add_argument("--overfitting_patience", type=int, default=30,
                        help="Consecutive epochs exceeding threshold to trigger early stopping")
    parser.add_argument("--overfitting_window", type=int, default=15,
                        help="Window size for moving average trend detection")
    
    # Checkpoint resumption
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Checkpoint path (local) or run ID (wandb) to resume from")
    parser.add_argument("--checkpoint_source", type=str, choices=["local", "wandb"],
                        help="Source of checkpoint: 'local' file or 'wandb' artifact")
    parser.add_argument("--resume_run", action="store_true",
                        help="Resume same wandb run ID (vs creating new run)")
    parser.add_argument("--run_id", type=str, default=None,
                        help="Stable run id for resume across SLURM requeues (e.g. episode-$SLURM_JOB_ID). "
                             "Reuses the same wandb run + wandb_<id> dir and auto-resumes from its latest checkpoint.")
    parser.add_argument("--weight_decay", type=float, default=1e-2,
                        help="Weight decay for optimizer (default: 0.01)")
    
    args = parser.parse_args()

    # Seed every RNG the training path touches: torch (weight init), CUDA, and
    # Python's `random` (the per-example segment cutoff in train_one_epoch).
    # Set before the model and loaders are built so init and shuffle order are
    # both covered. Note this makes runs comparable and re-runnable, not
    # bitwise-identical -- cuDNN/cuBLAS kernels remain nondeterministic unless
    # torch.use_deterministic_algorithms(True) is also set, which costs speed.
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Resolve dataset_name to full path
    if not os.path.isabs(args.dataset_name):
        root = os.environ.get("TRAINING_DATA_ROOT_DIR")
        if root:
            args.dataset_name = os.path.join(root, args.dataset_name)

    # DDP setup: detects torchrun and initializes the process group.
    # When launched without torchrun, all of these collapse to single-process defaults.
    is_distributed, rank, world_size, local_rank, device, is_rank_0 = setup_distributed(args.device)

    def unwrap(m):
        """Return the underlying model, stripping the DDP wrapper if present.

        Why: keeps checkpoint state_dict keys free of the `module.` prefix so
        BaseSocialTransformer._build_and_load can strict-load them later.
        """
        return m.module if is_distributed else m

    if is_rank_0 and is_distributed:
        print(f"[DDP] world_size={world_size}, backend={dist.get_backend()}, device={device}")

    # Load dataset metadata. The new Arrow-direct generator writes a single
    # metadata.json at the dataset root (replaces the .pt-era dataset_metadata.json
    # + per-split manifest.json pair).
    metadata_path = os.path.join(args.dataset_name, "metadata.json")
    if is_rank_0:
        print(f"Loading dataset metadata from {metadata_path}")
    with open(metadata_path, "r") as f:
        dataset_meta = json.load(f)

    params = dataset_meta["parameters"]
    max_experience = params["expert_N_max"]
    alpha0 = params.get("alpha0", 1.0)
    beta0 = params.get("beta0", 1.0)

    if is_rank_0:
        print(f"Dataset parameters: max_experience(expert_N_max)={max_experience}, alpha0={alpha0}, beta0={beta0}")

    # Create normalizer if needed for the chosen loss
    if requires_normalization(args.demo_belief_loss):
        normalizer = BeliefNormalizer(max_experience, alpha0, beta0)
        normalize_beliefs = True
        if is_rank_0:
            print(f"Using normalization with factor={normalizer.norm_factor} for loss={args.demo_belief_loss}")
    else:
        normalizer = None
        normalize_beliefs = False
        if is_rank_0:
            print(f"No normalization needed for loss={args.demo_belief_loss}")
    
    # Dataset shape parameters come straight from the root metadata.json
    # (the per-shard manifest.json is gone — there are no shards).
    num_arms = params["K"]
    num_demonstrators = params["num_demonstrators"]
    num_agents = num_demonstrators + 1  # +1 for learner
    num_rounds = params["num_rounds"]
    change_bandits = params.get("change_bandits_with_rounds", True)
    num_bandits = num_rounds if change_bandits else 1
    T_demo = params.get("T_demo", 6)
    T_learner = params.get("T_learner", 6)

    if is_rank_0:
        print(f"Dataset shape (from metadata.json):")
        print(f"  num_arms (K): {num_arms}")
        print(f"  num_demonstrators: {num_demonstrators}")
        print(f"  num_agents (learner + demos): {num_agents}")
        print(f"  num_bandits: {num_bandits}")
        print(f"  num_rounds: {num_rounds}")
        print(f"  T_demo: {T_demo}")
        print(f"  T_learner: {T_learner}")
    
    # Initialize wandb if requested.
    # Only rank 0 manages run directories, wandb init, artifact uploads, and YAML metadata.
    # Other ranks leave `run = None` and never touch the filesystem for run state.
    run = None
    run_dir = None
    # Offline (e.g. Marlowe compute nodes have no outbound internet): defer all artifact
    # uploads to scripts/wandb_sync.sh on a login node. Logging them offline is unreliable
    # because the rolling-checkpoint deletion removes the files the artifact versions point to.
    offline = os.environ.get("WANDB_MODE", "online").lower() in ("offline", "dryrun")
    if is_rank_0 and args.use_wandb:
        # Pre-determine wandb run ID so we can create run_dir before wandb.init()
        in_sweep = wandb.run is not None
        resume_mode = "never"

        if in_sweep:
            # Sweep agent already called wandb.init()
            wandb_run_id = wandb.run.id
        elif args.resume_from_checkpoint and args.resume_run:
            if args.checkpoint_source == "wandb":
                wandb_run_id = args.resume_from_checkpoint
                resume_mode = "allow"
                print(f"Will resume wandb run: {wandb_run_id}")
            else:
                print("Warning: --resume_run with local checkpoint creates new run (cannot extract run ID from local file)")
                wandb_run_id = wandb.util.generate_id()
        elif args.run_id:
            # Stable id (e.g. episode-$SLURM_JOB_ID): reuse the same wandb run across
            # SLURM requeues so offline shards coalesce into one run on sync.
            wandb_run_id = args.run_id
            resume_mode = "allow"
            print(f"Will resume/create stable wandb run: {wandb_run_id}")
        else:
            wandb_run_id = wandb.util.generate_id()

        # Create run directory with minimal structure before wandb.init()
        run_id = f"wandb_{wandb_run_id}"
        run_dir = setup_training_run_directory(run_id, subdirs=['checkpoints', 'wandb'])
        print(f"Run directory: {run_dir}")
        print(f"Run ID: {run_id}")

        # Build wandb config dict
        wandb_config = {
            "learning_rate": args.lr,
            "optimizer": args.optimizer,
            "weight_decay": args.weight_decay,
            "architecture": {
                "social": "SocialTransformer",
                "social_profile_learned": "LearnedProfileTransformer",
            }.get(args.model_type, "ProfileSocialTransformer"),
            "dataset_name": args.dataset_name,
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
            "env_theta_loss": args.env_theta_loss,
            "demo_loss_weight": args.demo_loss_weight,
            "demo_belief_loss": args.demo_belief_loss,
            "experience_loss_weight": args.experience_loss_weight,
            "masking_strategy": args.masking_strategy,
            "model_type": args.model_type,
            "experience_head_loss_weight": args.experience_head_loss_weight,
            "experience_head_loss_type": args.experience_head_loss_type,
            "expert_threshold": args.expert_threshold,
            "theta_head_input": args.theta_head_input,
            "demo_beliefs_head_input": args.demo_beliefs_head_input,
            "experience_head_input": args.experience_head_input,
            "normalize_beliefs": normalize_beliefs,
            "expert_N_max": max_experience,
            "use_scheduler": args.use_scheduler,
            "warmup_epochs": args.warmup_epochs if args.use_scheduler else None,
            "min_lr": args.min_lr if args.use_scheduler else None,
            "scheduler_type": args.scheduler if args.use_scheduler else None,
            "patience": args.patience if (args.use_scheduler and args.scheduler == "plateau") else None,
            "factor": args.factor if (args.use_scheduler and args.scheduler == "plateau") else None,
            "overfitting_threshold": args.overfitting_threshold if args.overfitting_threshold > 0 else None,
            "overfitting_patience": args.overfitting_patience,
            "overfitting_window": args.overfitting_window,
        }

        if in_sweep:
            # Sweep already initialized wandb; just update config
            wandb.config.update(wandb_config, allow_val_change=True)
            run = wandb.run
        else:
            run = wandb.init(
                id=wandb_run_id,
                resume=resume_mode,
                entity=args.wandb_entity,
                project=args.wandb_project,
                dir=str(run_dir),
                config=wandb_config,
            )

        # Define metric axes: everything plotted against epoch.
        # For a wall-clock view, switch any panel's x-axis to "Wall Time"
        # (wandb auto-logs _timestamp on every log call).
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*", step_metric="epoch")
        wandb.define_metric("learning_rate", step_metric="epoch")

        # Log the dataset (metadata files + path) as a wandb artifact.
        # Deferred to wandb_sync.sh when offline (no outbound internet on compute nodes).
        if not offline:
            log_dataset_artifact(run, args.dataset_name, dataset_meta)
    elif is_rank_0:
        # Non-wandb: full directory structure with metadata files
        run_id = get_run_id(use_wandb=False)
        run_dir = setup_training_run_directory(run_id)
        print(f"Run directory: {run_dir}")
        print(f"Run ID: {run_id}")

        metadata = {
            'run_id': run_id,
            'script': 'train.py',
            'command': ' '.join(sys.argv),
            'dataset_name': args.dataset_name,
        }
        write_run_yaml(run_dir, metadata)
        write_config_yaml(run_dir, vars(args))

    # Update checkpoint directory to use run directory (rank 0 only — non-rank-0
    # never writes checkpoints, so the path doesn't need to exist there).
    if is_rank_0:
        args.checkpoint_dir = str(run_dir / 'checkpoints')
        print(f"Checkpoints will be saved to: {args.checkpoint_dir}")

    # Build the train DataLoader from the mmap-backed Arrow dataset.
    # Under DDP, DistributedSampler shards examples across ranks and provides
    # i.i.d. epoch-level shuffling via set_epoch(). drop_last=True keeps per-rank
    # step counts identical so the cosine scheduler stays in sync across the world.
    # Single-process: DataLoader(shuffle=True) gives full i.i.d. shuffling — mmap
    # makes the random access cheap (no shard cache to thrash).
    train_ds = load_arrow_split(args.dataset_name, "train")
    train_sampler = (
        DistributedSampler(train_ds, shuffle=True, drop_last=True) if is_distributed else None
    )
    train_loader = make_arrow_loader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        sampler=train_sampler,
        normalizer=normalizer if normalize_beliefs else None,
        num_workers=args.num_workers,
    )

    # Validation runs on rank 0 only (see epoch loop), so no distributed sampler.
    val_ds = load_arrow_split(args.dataset_name, "val")
    val_loader = make_arrow_loader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=None,
        normalizer=normalizer if normalize_beliefs else None,
        num_workers=args.num_workers,
    )

    model = build_model(
        args.model_type,
        dict(
            num_arms=num_arms,
            num_agents=num_agents,
            num_bandits=num_bandits,
            num_demonstrators=num_demonstrators,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            env_theta_loss=args.env_theta_loss,
            experience_head_loss_type=args.experience_head_loss_type,
            theta_head_input=args.theta_head_input,
            demo_beliefs_head_input=args.demo_beliefs_head_input,
            experience_head_input=args.experience_head_input,
        ),
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
        'model_type': args.model_type,
        'env_theta_loss': args.env_theta_loss,
        'experience_head_loss_type': args.experience_head_loss_type,
        'theta_head_input': args.theta_head_input,
        'demo_beliefs_head_input': args.demo_beliefs_head_input,
        'experience_head_input': args.experience_head_input,
    }

    # Auto-resume from the latest rolling checkpoint when running under a stable
    # --run_id (SLURM requeue). The run dir is derived deterministically from the
    # id (no wandb dependency), so every rank resolves the same shared-FS path.
    # First launch: no file -> trains fresh. Requeue: finds epoch_*_latest.pt -> resumes.
    if args.run_id and not args.resume_from_checkpoint:
        latest_ckpts = sorted(
            (get_training_runs_dir() / f"wandb_{args.run_id}" / "checkpoints").glob("epoch_*_latest.pt")
        )
        if latest_ckpts:
            args.resume_from_checkpoint = str(latest_ckpts[-1])
            args.checkpoint_source = "local"
            if is_rank_0:
                print(f"Auto-resuming from latest checkpoint: {args.resume_from_checkpoint}")

    # If resuming, replace the fresh model with the one loaded from the
    # checkpoint *before* the optimizer is built, so the optimizer references
    # the right parameters. The checkpoint dict is stashed for the optimizer/
    # scheduler state restore below.
    resume_checkpoint = None
    if args.resume_from_checkpoint:
        if is_rank_0:
            print(f"\n{'='*50}")
            print("Loading checkpoint for resume...")
            print(f"{'='*50}")
        if args.checkpoint_source not in ("local", "wandb"):
            raise ValueError(f"Unknown checkpoint_source: {args.checkpoint_source}")
        model, resume_checkpoint = BaseSocialTransformer.load_checkpoint(
            local_path=args.resume_from_checkpoint if args.checkpoint_source == "local" else None,
            wandb_artifact=args.resume_from_checkpoint if args.checkpoint_source == "wandb" else None,
            cache_dir=str(get_training_runs_dir()),
        )
        model = model.to(device)

        # Trust the checkpoint's stored config for the saved architecture so
        # subsequent saves stay consistent with the loaded weights.
        if 'model_config' in resume_checkpoint:
            model_config = resume_checkpoint['model_config']

        # Early exit if training is already complete.
        checkpoint_epoch = resume_checkpoint['epoch']
        if checkpoint_epoch >= args.epochs:
            if is_rank_0:
                print(f"\n{'='*50}")
                print("Training already complete!")
                print(f"{'='*50}")
                print(f"Checkpoint at epoch {checkpoint_epoch}")
                print(f"Target epochs: {args.epochs}")
                print(f"\nTo continue training, specify more epochs:")
                print(f"  --epochs {checkpoint_epoch + 50}")
                print(f"{'='*50}\n")
            if is_distributed:
                dist.destroy_process_group()
            exit(0)
        if is_rank_0:
            print("✓ Loaded model weights")

    # Wrap in DDP *after* the model is on-device and (if applicable) restored
    # from checkpoint, but *before* the optimizer is built so optimizer
    # param_groups reference the wrapped parameters.
    if is_distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

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
            min_lr_ratio = args.min_lr / args.lr
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                min_lr_ratio=min_lr_ratio
            )
            if is_rank_0:
                # Under DDP, len(train_loader) is the per-rank step count; that
                # matches the optimizer's per-rank step rate so the cosine
                # envelope is correct without further adjustment.
                print(f"Using cosine LR schedule with {args.warmup_epochs} epoch warmup")
                print(f"  Total training steps (per-rank): {num_training_steps}")
                print(f"  Warmup steps (per-rank): {num_warmup_steps}")
                print(f"  Min LR: {args.min_lr:.2e}")
        elif args.scheduler == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=args.factor,
                patience=args.patience,
                min_lr=args.min_lr
            )
            if is_rank_0:
                print(f"Using ReduceLROnPlateau scheduler")
                print(f"  Patience: {args.patience}, Factor: {args.factor}")
                print(f"  Min LR: {args.min_lr:.2e}")

    # Initialize training state (overridden below if resuming).
    start_epoch = 1
    best_val_loss = float('inf')

    # Restore optimizer / scheduler state from the resume checkpoint we
    # already loaded above (the model weights themselves were applied there).
    if resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
        if is_rank_0:
            print("✓ Loaded optimizer state")

        if scheduler and resume_checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(resume_checkpoint['scheduler_state_dict'])
            if is_rank_0:
                print("✓ Loaded scheduler state")

        start_epoch = resume_checkpoint['epoch'] + 1
        best_val_loss = resume_checkpoint['val_stats']['loss']

        if is_rank_0:
            print(f"\nResuming from epoch {resume_checkpoint['epoch']}")
            print(f"Will train from epoch {start_epoch} to {args.epochs}")
            print(f"Best val loss so far: {best_val_loss:.4f}")
            print(f"{'='*50}\n")
    
    # Overfitting detection tracking
    overfitting_counter = 0
    val_loss_history = []  # Track recent validation losses
    
    # Track previous checkpoint paths for deletion
    prev_best_checkpoint = None
    prev_latest_checkpoint = None

    # Get the environment theta loss function
    env_loss_fn = get_env_theta_loss(args.env_theta_loss)
    if is_rank_0:
        print(f"Using {args.env_theta_loss} loss for environment theta payoffs")

    # Get the demo belief loss function
    demo_loss_fn = get_demo_belief_loss(args.demo_belief_loss)
    if is_rank_0:
        print(f"Using {args.demo_belief_loss} loss for demonstrator beliefs")

    # Get the experience head loss function (only for profile methods)
    if args.model_type != "social":
        experience_head_loss_fn = get_experience_head_loss(args.experience_head_loss_type)
        if is_rank_0:
            print(f"Using {args.experience_head_loss_type} loss for experience head")
            print(f"Experience head loss weight: {args.experience_head_loss_weight}")
    else:
        experience_head_loss_fn = None

    if is_rank_0:
        print(f"\nStarting training loop for {args.epochs} epochs...")
        print(f"Training batches per epoch (per-rank): {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        print(f"\nNote: First batch load may take 2-3 seconds (loading shard from disk)...")
    
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            if is_rank_0:
                print(f"\n{'='*50}")
                print(f"Starting Epoch {epoch}/{args.epochs}")
                print(f"{'='*50}")

            # Per-epoch shuffling. Under DDP we call sampler.set_epoch() so the
            # global shuffle permutation differs each epoch and stays consistent
            # across ranks. Single-process gets its shuffling for free from
            # DataLoader(shuffle=True), which reshuffles every iteration.
            if is_distributed and train_sampler is not None:
                train_sampler.set_epoch(epoch)

            stats = train_one_epoch(
                model, train_loader, optimizer, device,
                alpha0=alpha0,
                beta0=beta0,
                env_loss_fn=env_loss_fn,
                env_theta_loss=args.env_theta_loss,
                demo_loss_weight=args.demo_loss_weight,
                demo_loss_fn=demo_loss_fn,
                normalizer=normalizer,
                scheduler=scheduler,
                scheduler_type=args.scheduler if args.use_scheduler else "cosine",
                experience_loss_weight=args.experience_loss_weight,
                masking_strategy=args.masking_strategy,
                model_type=args.model_type,
                experience_head_loss_weight=args.experience_head_loss_weight,
                experience_head_loss_fn=experience_head_loss_fn,
                num_arms=num_arms,
                is_rank_0=is_rank_0,
            )

            # Log current learning rate if scheduler is used
            current_lr = optimizer.param_groups[0]['lr']
            if is_rank_0:
                log_str = f"Epoch {epoch:02d} | loss={stats['loss']:.4f}  theta={stats['loss_theta']:.4f}  demo={stats['loss_demo']:.4f}  exp={stats['loss_experience']:.4f}"
                if 'loss_experience_head' in stats and stats['loss_experience_head'] > 0:
                    log_str += f"  exp_head={stats['loss_experience_head']:.4f}"
                log_str += f"  lr={current_lr:.2e}"
                print(log_str)

            # Validation runs on rank 0 only — see the broadcast below for how
            # the other ranks pick up val_stats so scheduler/early-stop logic
            # stays in sync across the world.
            val_stats = None
            if is_rank_0:
                model.eval()
                with torch.no_grad():
                    val_tot = {"loss":0.0,"theta":0.0,"demo":0.0,"experience":0.0,"experience_head":0.0}
                    for batch in tqdm(val_loader, desc="val", unit="b", leave=False):
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

                        # Extract demo predictions for last bandit
                        pred_alpha_demo_val = out["alpha_demo"][batch_indices, last_bandit_idx]    # [B, D, K]
                        pred_beta_demo_val = out["beta_demo"][batch_indices, last_bandit_idx]      # [B, D, K]

                        # Environment theta loss: branch on the configured loss function
                        if args.env_theta_loss == "bce":
                            pred_theta_logits_val = out["theta_logits"][batch_indices, last_bandit_idx]  # [B, K]
                            lt = env_loss_fn(pred_theta_logits_val, y_theta_last)
                        else:
                            pred_alpha_theta_val = out["alpha_theta"][batch_indices, last_bandit_idx]  # [B, K]
                            pred_beta_theta_val = out["beta_theta"][batch_indices, last_bandit_idx]    # [B, K]
                            lt = env_loss_fn(pred_alpha_theta_val, pred_beta_theta_val, y_theta_last)

                        # Denormalize predictions if needed
                        if normalizer is not None:
                            pred_alpha_val = normalizer.denormalize_alpha(pred_alpha_demo_val)
                            pred_beta_val = normalizer.denormalize_beta(pred_beta_demo_val)
                            target_alpha_val = normalizer.denormalize_alpha(y_alpha_demo_last)
                            target_beta_val = normalizer.denormalize_beta(y_beta_demo_last)
                        else:
                            pred_alpha_val = pred_alpha_demo_val
                            pred_beta_val = pred_beta_demo_val
                            target_alpha_val = y_alpha_demo_last
                            target_beta_val = y_beta_demo_last

                        # Per-demonstrator belief loss
                        num_demonstrators = y_alpha_demo_last.shape[1]

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

                    # Log both train and val metrics together (rank 0 only since run is None elsewhere)
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

            # Broadcast val_stats so every rank can drive its scheduler and
            # early-stop logic from the same numbers. Without this, ranks 1+
            # would have val_stats=None and the plateau scheduler would desync.
            if is_distributed:
                payload = [val_stats]
                dist.broadcast_object_list(payload, src=0)
                val_stats = payload[0]

            # Step plateau scheduler on every rank so per-rank optimizer LRs stay in sync.
            if scheduler is not None and args.scheduler == "plateau":
                scheduler.step(val_stats['loss'])

            # Check if this is the best model so far. Computed on every rank so
            # overfitting bookkeeping (val_loss_history, overfitting_counter)
            # stays in lockstep — checkpoint I/O itself happens only on rank 0.
            is_best = val_stats['loss'] < best_val_loss
            if is_best:
                best_val_loss = val_stats['loss']
                early_stopping_counter = 0
                if is_rank_0:
                    print(f"     ✓ New best model! Val loss: {best_val_loss:.4f}")

                    # Delete previous best checkpoint if it exists
                    if prev_best_checkpoint is not None and os.path.exists(prev_best_checkpoint):
                        os.remove(prev_best_checkpoint)
                        print(f"     Deleted previous best checkpoint: {prev_best_checkpoint}")

                    # Delete previous latest checkpoint if it exists
                    if prev_latest_checkpoint is not None and os.path.exists(prev_latest_checkpoint):
                        os.remove(prev_latest_checkpoint)
                        print(f"     Deleted previous latest checkpoint: {prev_latest_checkpoint}")

                    # Save best model checkpoint (logs to W&B with all aliases).
                    # unwrap(model) strips the DDP wrapper so state_dict keys
                    # stay free of the 'module.' prefix that would break
                    # strict loading downstream.
                    checkpoint_name = f"epoch_{epoch:03d}_best.pt"
                    checkpoint_path = save_checkpoint(
                        model=unwrap(model),
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
                        max_experience=max_experience,
                        alpha0=alpha0,
                        beta0=beta0,
                        env_theta_loss=args.env_theta_loss,
                        demo_belief_loss=args.demo_belief_loss,
                        scheduler=scheduler,
                        model_config=model_config,
                        experience_loss_weight=args.experience_loss_weight,
                        log_to_wandb=not offline,
                    )
                    prev_best_checkpoint = checkpoint_path

                    # Also save as latest checkpoint (local only, no W&B logging)
                    checkpoint_name_latest = f"epoch_{epoch:03d}_latest.pt"
                    checkpoint_path_latest = save_checkpoint(
                        model=unwrap(model),
                        optimizer=optimizer,
                        epoch=epoch,
                        train_stats=stats,
                        val_stats=val_stats,
                        checkpoint_dir=args.checkpoint_dir,
                        checkpoint_name=checkpoint_name_latest,
                        wandb_run=run,
                        artifact_name=args.artifact_name,
                        is_best=False,
                        normalizer=normalizer,
                        max_experience=max_experience,
                        alpha0=alpha0,
                        beta0=beta0,
                        env_theta_loss=args.env_theta_loss,
                        demo_belief_loss=args.demo_belief_loss,
                        scheduler=scheduler,
                        model_config=model_config,
                        experience_loss_weight=args.experience_loss_weight,
                        log_to_wandb=False,
                    )
                    prev_latest_checkpoint = checkpoint_path_latest
            else:
                # No improvement - check for overfitting (runs on every rank
                # so the `break` decision is unanimous across the world).
                val_loss_history.append(val_stats['loss'])

                # Only check if overfitting detection is enabled (threshold > 0)
                if args.overfitting_threshold > 0:
                    # Condition A: Current loss exceeds threshold
                    threshold_exceeded = val_stats['loss'] > best_val_loss * (1 + args.overfitting_threshold / 100.0)

                    # Condition B: Moving average shows upward trend
                    upward_trend = False
                    if len(val_loss_history) >= args.overfitting_window:
                        recent_losses = val_loss_history[-args.overfitting_window:]
                        # Simple trend: compare first half vs second half of window
                        first_half_avg = sum(recent_losses[:len(recent_losses)//2]) / (len(recent_losses)//2)
                        second_half_avg = sum(recent_losses[len(recent_losses)//2:]) / (len(recent_losses) - len(recent_losses)//2)
                        upward_trend = second_half_avg > first_half_avg

                    # Both conditions must be true
                    if threshold_exceeded and upward_trend:
                        overfitting_counter += 1
                        if is_rank_0:
                            print(f"     ⚠ Overfitting signal {overfitting_counter}/{args.overfitting_patience}: "
                                  f"val_loss={val_stats['loss']:.4f} vs best={best_val_loss:.4f} "
                                  f"(+{((val_stats['loss']/best_val_loss - 1)*100):.2f}%), trend=↑")

                        if overfitting_counter >= args.overfitting_patience:
                            if is_rank_0:
                                print(f"\n{'='*50}")
                                print(f"Early stopping: Overfitting detected after {epoch} epochs")
                                print(f"Val loss consistently above best by >{args.overfitting_threshold}% with upward trend")
                                print(f"Best val loss: {best_val_loss:.4f}")
                                print(f"Current val loss: {val_stats['loss']:.4f}")
                                print(f"{'='*50}")
                            break
                    else:
                        # Reset counter if conditions not met
                        overfitting_counter = 0

            # Save periodic checkpoints if requested (but skip if already saved as best)
            if args.save_every > 0 and epoch % args.save_every == 0 and not is_best and is_rank_0:
                # Delete previous latest checkpoint if it exists
                if prev_latest_checkpoint is not None and os.path.exists(prev_latest_checkpoint):
                    os.remove(prev_latest_checkpoint)
                    print(f"     Deleted previous latest checkpoint: {prev_latest_checkpoint}")

                # Save as latest checkpoint (logs to W&B since not already logged as best)
                checkpoint_name = f"epoch_{epoch:03d}_latest.pt"
                checkpoint_path = save_checkpoint(
                    model=unwrap(model),
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
                    max_experience=max_experience,
                    alpha0=alpha0,
                    beta0=beta0,
                    env_theta_loss=args.env_theta_loss,
                    demo_belief_loss=args.demo_belief_loss,
                    scheduler=scheduler,
                    model_config=model_config,
                    experience_loss_weight=args.experience_loss_weight,
                    log_to_wandb=not offline,
                )
                prev_latest_checkpoint = checkpoint_path

            # Per-epoch barrier so ranks 1+ don't race ahead while rank 0 is
            # still writing checkpoints / uploading artifacts.
            if is_distributed:
                dist.barrier()

        # Save final model as latest checkpoint (only if not already saved).
        # Rank-0-only; other ranks just skip past this block to the cleanup.
        if is_rank_0:
            final_checkpoint_name = f"epoch_{args.epochs:03d}_latest.pt"
            final_checkpoint_exists = prev_latest_checkpoint is not None and \
                                      final_checkpoint_name in prev_latest_checkpoint

            if not final_checkpoint_exists:
                # Delete previous latest checkpoint if it exists and is for a different epoch
                if prev_latest_checkpoint is not None and os.path.exists(prev_latest_checkpoint):
                    os.remove(prev_latest_checkpoint)
                    print(f"Deleted previous latest checkpoint: {prev_latest_checkpoint}")

                final_checkpoint_path = save_checkpoint(
                    model=unwrap(model),
                    optimizer=optimizer,
                    epoch=args.epochs,
                    train_stats=stats,
                    val_stats=val_stats,
                    checkpoint_dir=args.checkpoint_dir,
                    checkpoint_name=final_checkpoint_name,
                    wandb_run=run,
                    artifact_name=args.artifact_name,
                    is_best=False,
                    normalizer=normalizer,
                    max_experience=max_experience,
                    alpha0=alpha0,
                    beta0=beta0,
                    env_theta_loss=args.env_theta_loss,
                    demo_belief_loss=args.demo_belief_loss,
                    scheduler=scheduler,
                    model_config=model_config,
                    experience_loss_weight=args.experience_loss_weight,
                    log_to_wandb=not offline,
                )
            else:
                print(f"Final epoch checkpoint already saved as latest: {prev_latest_checkpoint}")

        # Training completed successfully
        if is_rank_0:
            if not args.use_wandb:
                finalize_run_yaml(run_dir, status='completed')
            print(f"\n{'='*50}")
            print("Training completed successfully!")
            print(f"{'='*50}")

    except KeyboardInterrupt:
        if is_rank_0:
            print(f"\n{'='*50}")
            print("Training interrupted by user")
            print(f"{'='*50}")
            if not args.use_wandb:
                finalize_run_yaml(run_dir, status='interrupted')
        raise
    except Exception as e:
        if is_rank_0:
            print(f"\n{'='*50}")
            print(f"Training failed with error: {e}")
            print(f"{'='*50}")
            if not args.use_wandb:
                finalize_run_yaml(run_dir, status='failed', error=str(e))
        raise
    finally:
        # Drain pending collectives, then tear down the process group cleanly.
        if is_distributed:
            if dist.is_initialized():
                dist.barrier()
                dist.destroy_process_group()


if __name__ == "__main__":
    main()
