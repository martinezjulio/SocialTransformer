
import argparse
import glob
import os
from typing import Dict, List, Tuple

import torch

from models.BetaModel import SBIPosteriorTransformer, thompson_sample_action
from dataloader import SPECIAL


def find_checkpoint(checkpoint_dir: str, checkpoint_type: str) -> str:
    """
    Find the checkpoint file in the given directory based on type.
    
    Args:
        checkpoint_dir: Directory to search for checkpoints
        checkpoint_type: Either 'best' or 'latest'
    
    Returns:
        Path to the checkpoint file
    
    Raises:
        FileNotFoundError: If no matching checkpoint is found
    """
    pattern = f"epoch_*_{checkpoint_type}.pt"
    search_path = os.path.join(checkpoint_dir, pattern)
    matches = glob.glob(search_path)
    
    if not matches:
        raise FileNotFoundError(
            f"No {checkpoint_type} checkpoint found in {checkpoint_dir}. "
            f"Looking for pattern: {pattern}"
        )
    
    # If multiple matches, take the one with the highest epoch number
    if len(matches) > 1:
        # Extract epoch numbers and sort
        def get_epoch(path):
            filename = os.path.basename(path)
            # epoch_XXX_best.pt or epoch_XXX_latest.pt
            epoch_str = filename.split('_')[1]
            return int(epoch_str)
        
        matches.sort(key=get_epoch, reverse=True)
    
    return matches[0]


def build_tokens_from_events(
    events_context: List[Tuple[int,int,int,int]],
    events_current: List[Tuple[int,int,int,int]],
    K: int,
) -> Dict[str, torch.Tensor]:
    """
    Convert raw events into model token tensors and append a QUERY token.
    events_* are lists of (bandit_id, agent_id, action, reward).
    """
    bandit_ids = []
    agent_ids  = []
    action_ids = []
    reward_ids = []
    special_ids= []

    def add_event(e):
        b, ag, a, r = e
        if b == -1 and ag == -1:
            # NOCTX
            bandit_ids.append(0); agent_ids.append(0); action_ids.append(K); reward_ids.append(2); special_ids.append(SPECIAL["NOCTX"]); return
        if ag == -1:
            # BANDIT_SEP
            bandit_ids.append(max(0,b)); agent_ids.append(0); action_ids.append(K); reward_ids.append(2); special_ids.append(SPECIAL["BANDIT_SEP"]); return
        bandit_ids.append(max(0,b))
        agent_ids.append(2 if ag == 99 else min(ag, 1))
        action_ids.append(a if a >= 0 else K)
        reward_ids.append(r if r in (0,1) else 2)
        special_ids.append(SPECIAL["NONE"])

    if len(events_context) == 0:
        add_event((-1,-1,-1,-1))  # NOCTX
    else:
        for e in events_context: add_event(e)
    for e in events_current: add_event(e)

    # Append QUERY
    bandit_ids.append(0); agent_ids.append(0); action_ids.append(K); reward_ids.append(2); special_ids.append(SPECIAL["QUERY"])

    batch = {
        "bandit_ids": torch.tensor(bandit_ids, dtype=torch.long)[None, :],
        "agent_ids":  torch.tensor(agent_ids,  dtype=torch.long)[None, :],
        "action_ids": torch.tensor(action_ids, dtype=torch.long)[None, :],
        "reward_ids": torch.tensor(reward_ids, dtype=torch.long)[None, :],
        "special_ids":torch.tensor(special_ids,dtype=torch.long)[None, :],
        "pad_mask":   torch.zeros(1, len(bandit_ids), dtype=torch.bool),
    }
    return batch


@torch.no_grad()
def posterior_sample_theta_and_counts(
    model: SBIPosteriorTransformer,
    batch_tokens: Dict[str, torch.Tensor],
    alpha0: float = 1.0,
    beta0: float = 1.0,
    observed_successes: torch.Tensor = None,  # [K] successes accumulated so far (both agents)
    observed_failures: torch.Tensor = None,   # [K]
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Given tokens, sample start-of-tail pseudo-counts (continuous) and then sample theta ~ Beta(alpha*, beta*).
    If observed_successes/failures are provided, they will be added to the shapes before sampling theta.
    """
    for k in batch_tokens:
        batch_tokens[k] = batch_tokens[k].to(device)
    model = model.to(device).eval()
    out = model(batch_tokens)
    alpha_demo0 = out["alpha_demo0"][0]  # [K]
    beta_demo0  = out["beta_demo0"][0]   # [K]
    alpha_theta = out["alpha_theta"][0]
    beta_theta  = out["beta_theta"][0]

    # Combine with observed counts to form posterior over theta
    if observed_successes is None:
        observed_successes = torch.zeros_like(alpha_demo0)
    if observed_failures is None:
        observed_failures = torch.zeros_like(beta_demo0)

    alpha_star = alpha_demo0 + observed_successes
    beta_star  = beta_demo0  + observed_failures

    theta_sample = torch.distributions.Beta(alpha_star, beta_star).sample()

    return {
        "alpha_demo0": alpha_demo0.cpu(),
        "beta_demo0": beta_demo0.cpu(),
        "alpha_star": alpha_star.cpu(),
        "beta_star": beta_star.cpu(),
        "theta_sample": theta_sample.cpu(),
        "theta_beta_head": torch.distributions.Beta(alpha_theta, beta_theta).sample().cpu(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory containing checkpoints")
    parser.add_argument("--checkpoint_type", type=str, default="best", choices=["best", "latest"],
                        help="Type of checkpoint to load: 'best' or 'latest'")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Load checkpoint from wandb artifact instead of local file")
    parser.add_argument("--wandb_entity", type=str,
                        help="wandb entity (required if --use_wandb)")
    parser.add_argument("--wandb_project", type=str,
                        help="wandb project (required if --use_wandb)")
    parser.add_argument("--artifact_name", type=str, default="sbi-transformer-checkpoint",
                        help="wandb artifact name")
    parser.add_argument("--artifact_alias", type=str, default="best",
                        help="wandb artifact alias (e.g., 'best', 'latest', 'epoch_5')")
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    model = SBIPosteriorTransformer(num_arms=args.K, num_agents=3, max_bandits=16)
    
    if args.use_wandb:
        # Load from wandb artifact
        if not args.wandb_entity or not args.wandb_project:
            parser.error("--wandb_entity and --wandb_project are required when --use_wandb is set")
        
        import wandb
        
        api = wandb.Api()
        artifact_path = f"{args.wandb_entity}/{args.wandb_project}/{args.artifact_name}:{args.artifact_alias}"
        print(f"Downloading artifact: {artifact_path}")
        
        artifact = api.artifact(artifact_path, type='model')
        artifact_dir = artifact.download()
        
        # Find checkpoint file based on type
        checkpoint_path = find_checkpoint(artifact_dir, args.checkpoint_type)
        print(f"Found checkpoint: {os.path.basename(checkpoint_path)}")
        
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        
        # Load model state from checkpoint dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from epoch {checkpoint['epoch']}, val loss: {checkpoint['val_stats']['loss']:.4f}")
        else:
            model.load_state_dict(checkpoint)
    else:
        # Load from local file
        checkpoint_path = find_checkpoint(args.checkpoint_dir, args.checkpoint_type)
        print(f"Found checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        
        # Handle both full checkpoint dicts and plain state dicts
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, val loss: {checkpoint['val_stats']['loss']:.4f}")
        else:
            model.load_state_dict(checkpoint)
            print(f"Loaded model from {checkpoint_path}")

    # Example: empty context, a few current events (bandit separators + some demo actions)
    events_context = []  # use NOCTX
    events_current = [
        (0, -1, -1, -1),  # BANDIT_SEP
        (0, 0, 1, 1),     # demo action=1, reward=1
        (0, 0, 1, 0),
        (0, 0, 2, 1),
    ]
    batch_tok = build_tokens_from_events(events_context, events_current, K=args.K)
    post = posterior_sample_theta_and_counts(model, batch_tok, device=args.device)
    print("Sampled theta:", post["theta_sample"].tolist())
    print("Alpha*:", post["alpha_star"].tolist(), " Beta*:", post["beta_star"].tolist())


if __name__ == "__main__":
    main()
