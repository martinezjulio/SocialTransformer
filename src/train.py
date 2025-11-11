
import argparse
from typing import Dict
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataloader import ShardedExamplesDataset, BanditSequenceDataset, collate_batch
from models.BetaModel import SBIPosteriorTransformer


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
    parser.add_argument("--train_path", type=str, default="data/train/manifest.json")
    parser.add_argument("--val_path", type=str, default="data/val/manifest.json")
    parser.add_argument("--num_arms", type=int, default=12)
    parser.add_argument("--max_bandits", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

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

    for epoch in range(1, args.epochs + 1):
        stats = train_one_epoch(model, train_loader, optimizer, device)
        print(f"Epoch {epoch:02d} | loss={stats['loss']:.4f}  theta={stats['loss_theta']:.4f}  demo={stats['loss_demo']:.4f}")

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
            print(f"           val | loss={val_tot['loss']/n:.4f}  theta={val_tot['theta']/n:.4f}  demo={val_tot['demo']/n:.4f}")

    torch.save(model.state_dict(), "sbi_transformer.pt")
    print("Saved model to sbi_transformer.pt")


if __name__ == "__main__":
    main()
