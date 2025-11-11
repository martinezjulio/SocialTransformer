
import argparse
from typing import Dict, List, Tuple

import torch

from models.BetaModel import SBIPosteriorTransformer, thompson_sample_action
from dataloader import SPECIAL

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
    parser.add_argument("--checkpoint", type=str, default="sbi_transformer.pt")
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    model = SBIPosteriorTransformer(num_arms=args.K, num_agents=3, max_bandits=16)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))

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
