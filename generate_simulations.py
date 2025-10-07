
import math
import random
from typing import Dict, List, Tuple

import torch

# ----------- Utilities -----------

def beta_sample(a: float, b: float) -> float:
    return torch.distributions.Beta(torch.tensor(a), torch.tensor(b)).sample().item()

def bernoulli(p: float) -> int:
    return 1 if random.random() < p else 0

def ts_argmax(alpha: torch.Tensor, beta: torch.Tensor) -> int:
    """Standard Thompson Sampling argmax with Beta draws per arm."""
    draws = torch.distributions.Beta(alpha, beta).sample()
    return int(torch.argmax(draws).item())

# ----------- Simulator -----------

def simulate_bandit(
    K: int,
    T_demo: int,
    T_learner: int,
    alpha0: float,
    beta0: float,
    kappa_tot: float,
    dirichlet_eta: float,
    tauD_unused: float,
    demonstrator_id: int = 0,
) -> Dict:
    """
    Simulate a single bandit episode with:
    - true thetas ~ Beta(2,2), with enforced gap so there's a clear best
    - demonstrator with pre-episode counts S0,F0 derived from kappa_tot and a Dirichlet allocation
    - demonstrator acts for T_demo steps via TS; learner also acts for T_learner steps via TS (optional)
    Returns a dict with events and ground truth labels for this bandit.
    """
    # Sample thetas until we get a clear best
    while True:
        thetas = torch.distributions.Beta(torch.tensor([2.0]*K), torch.tensor([2.0]*K)).sample().tolist()
        best = max(thetas)
        second = sorted(thetas, reverse=True)[1]
        if best - second >= 0.08 and best >= 0.6:
            break

    # Allocation of experience across arms (Dirichlet -> proportions -> pseudo-counts -> integer N per arm)
    w = torch.distributions.Dirichlet(torch.tensor([dirichlet_eta]*K)).sample().tolist()
    Nk_real = [max(kappa_tot * wi, 0.0) for wi in w]
    # Round to integers for simulation of pre-episode counts
    Nk = [int(round(n)) for n in Nk_real]

    # Pre-episode successes S0 ~ Binomial(Nk, theta), failures F0 = Nk - S0
    S0 = [torch.distributions.Binomial(total_count=n, probs=t).sample().int().item() if n > 0 else 0
          for n, t in zip(Nk, thetas)]
    F0 = [n - s for n, s in zip(Nk, S0)]

    # Start-of-tail Betas for demonstrator
    alphaD = torch.tensor([alpha0 + s for s in S0], dtype=torch.float32)
    betaD  = torch.tensor([beta0 + f for f in F0], dtype=torch.float32)

    # Learner internal Beta (asocial baseline TS for the synthetic learner; we do not train on learner policy)
    alphaL = torch.tensor([alpha0]*K, dtype=torch.float32)
    betaL  = torch.tensor([beta0]*K, dtype=torch.float32)

    # Event stream: (bandit_id, agent_id, action, reward)
    # agent_id: 0.. is demonstrators, 99 -> learner
    events: List[Tuple[int,int,int,int]] = []

    bandit_id = 0  # local within this function; outer driver will offset

    # Demonstrator block
    for _ in range(T_demo):
        a = ts_argmax(alphaD, betaD)
        r = bernoulli(thetas[a])
        events.append((bandit_id, demonstrator_id, a, r))
        # update demo
        alphaD[a] += r
        betaD[a]  += 1 - r

    # Learner block (optional; include to diversify tokens)
    for _ in range(T_learner):
        a = ts_argmax(alphaL, betaL)
        r = bernoulli(thetas[a])
        events.append((bandit_id, 99, a, r))
        # update learner (this is only for synthetic token variety)
        alphaL[a] += r
        betaL[a]  += 1 - r

    return {
        "theta": thetas,
        "S0": S0,
        "F0": F0,
        "events": events,
        "demonstrator_id": demonstrator_id,
    }


def simulate_multi_bandit_episode(
    K: int = 4,
    num_bandits: int = 3,
    T_demo: int = 12,
    T_learner: int = 12,
    alpha0: float = 1.0,
    beta0: float = 1.0,
    kappa_high: float = 30.0,
    kappa_low: float = 4.0,
    dirichlet_eta: float = 1.0,
    two_demos: bool = False,
) -> Dict:
    """
    Create an episode containing sequences from multiple bandits with (optionally) the same demonstrator.
    If two_demos=True, alternate demonstrator_id between 0 and 1 across bandits.
    """
    epi = {"bandits": []}
    for b in range(num_bandits):
        demonstrator_id = (b % 2) if two_demos else 0
        # Keep same experience "trait" across bandits for same demonstrator
        kappa_tot = kappa_high if (demonstrator_id == 0) else kappa_low
        bandit = simulate_bandit(
            K=K,
            T_demo=T_demo,
            T_learner=T_learner,
            alpha0=alpha0,
            beta0=beta0,
            kappa_tot=kappa_tot,
            dirichlet_eta=dirichlet_eta,
            tauD_unused=4.0,
            demonstrator_id=demonstrator_id,
        )
        # Offset bandit_id inside events
        bandit["events"] = [(b, ag, a, r) for (_, ag, a, r) in bandit["events"]]
        epi["bandits"].append(bandit)
    return epi


def build_training_examples_from_episode(epi: Dict, max_context_bandits: int = 2) -> List[Dict]:
    """
    Turn a multi-bandit episode into training examples.
    For each bandit c >= 0, build:
      - context: up to 'max_context_bandits' previous bandits (their events), regardless of demonstrator.
        (You can filter same-demonstrator-only if you prefer.)
      - current: all events from current bandit (we'll later take a random prefix in the dataloader)
      - labels: theta, alpha_demo0, beta_demo0 for the demonstrator of current bandit
    """
    examples = []
    for c, bandit in enumerate(epi["bandits"]):
        ctx_bandits = epi["bandits"][max(0, c - max_context_bandits):c]
        ctx_events = []
        for cb, bdict in enumerate(ctx_bandits):
            # Insert a bandit separator as (bandit_id, agent_id=-1, action=-1, reward=-1)
            ctx_events.append((bdict["events"][0][0], -1, -1, -1))
            ctx_events.extend(bdict["events"])
        if len(ctx_events) == 0:
            # Put a NOCTX special marker token
            ctx_events = [(-1, -1, -1, -1)]
        current_events = []
        # Insert a bandit separator before current
        current_events.append((bandid := bandit["events"][0][0], -1, -1, -1))
        current_events.extend(bandit["events"])
        labels = {
            "theta": bandit["theta"],
            "alpha_demo0": [1.0 + s for s in bandit["S0"]],
            "beta_demo0":  [1.0 + f for f in bandit["F0"]],
            "demonstrator_id": bandit["demonstrator_id"],
            "K": len(bandit["theta"]),
        }
        examples.append({
            "context_events": ctx_events,
            "current_events": current_events,
            "labels": labels,
        })
    return examples


def generate_dataset(
    out_path: str,
    num_episodes: int = 2000,
    K: int = 4,
    num_bandits: int = 3,
    T_demo: int = 12,
    T_learner: int = 12,
    two_demos: bool = False,
) -> None:
    """
    Generate a synthetic dataset and save as a torch file with a list of examples.
    """
    all_examples = []
    for _ in range(num_episodes):
        epi = simulate_multi_bandit_episode(
            K=K,
            num_bandits=num_bandits,
            T_demo=T_demo,
            T_learner=T_learner,
            two_demos=two_demos,
        )
        exs = build_training_examples_from_episode(epi)
        all_examples.extend(exs)
    torch.save(all_examples, out_path)
    print(f"Saved {len(all_examples)} examples to {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data/train.pt")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--num_bandits", type=int, default=3)
    parser.add_argument("--T_demo", type=int, default=12)
    parser.add_argument("--T_learner", type=int, default=12)
    parser.add_argument("--two_demos", action="store_true")
    args = parser.parse_args()

    generate_dataset(
        out_path=args.out,
        num_episodes=args.episodes,
        K=args.K,
        num_bandits=args.num_bandits,
        T_demo=args.T_demo,
        T_learner=args.T_learner,
        two_demos=args.two_demos,
    )
