#%%
import json
import math
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from datasets import Dataset

from data.encoding import SEGMENT_TYPE_TO_ID, build_arrow_row, build_features

# ----------- Utilities -----------

def beta_sample(alpha: float, beta: float) -> float:
    """Sample one value from Beta(alpha, beta)."""
    return torch.distributions.Beta(torch.tensor(alpha), torch.tensor(beta)).sample().item()

def bernoulli(p: float) -> int:
    return 1 if random.random() < p else 0

def ts_argmax(alphas: torch.Tensor, betas: torch.Tensor) -> int:
    """Thompson Sampling argmax with Beta draws per arm.
    Args:
        alphas: tensor [K]
        betas:  tensor [K]
    """
    draws = torch.distributions.Beta(alphas, betas).sample()  # [K]
    return int(torch.argmax(draws).item())

def random_arm_with_exclusion(K: int, thetas: List[float], num_exclude: int = 6) -> int:
    """
    Select a random arm excluding the top num_exclude arms by payoff.
    
    Args:
        K: number of arms
        thetas: list of arm payoffs
        num_exclude: number of top arms to exclude (default: 6)
    
    Returns:
        Random arm index from the bottom K-num_exclude arms
    """
    # Create list of (arm_idx, theta) pairs and sort by theta descending
    arm_payoffs = [(i, thetas[i]) for i in range(K)]
    arm_payoffs.sort(key=lambda x: x[1], reverse=True)
    
    # Get the bottom K-num_exclude arms
    available_arms = [arm_idx for arm_idx, _ in arm_payoffs[num_exclude:]]
    
    # Return random arm from available set
    return random.choice(available_arms)

def sample_bandit(K: int, min_gap: float = 0.05, min_best: float = 0.70) -> List[float]:
    """Sample K payoffs from Beta(2,2) until there's a clear best arm."""
    while True:
        thetas = torch.distributions.Beta(torch.full((K,), 2.0), torch.full((K,), 2.0)).sample().tolist()
        best, second = sorted(thetas, reverse=True)[:2]
        if best - second >= min_gap and best >= min_best:
            return thetas

def permute_bandit(payoff_values: List[float]) -> List[float]:
    """
    Create a bandit by permuting a fixed set of payoff values.
    
    Args:
        payoff_values: Fixed list of payoff values to permute
    
    Returns:
        Random permutation of the payoff values
    """
    thetas = payoff_values.copy()
    random.shuffle(thetas)
    return thetas

def allocate_prehistory_ts(
    K: int,
    N_tot: int,
    thetas: List[float],
    alpha0: float = 1.0,
    beta0: float = 1.0,
) -> Tuple[torch.Tensor, List[int], List[int]]:
    """
    Thompson Sampling-based pre-episode: simulate N_tot pulls before the observed tail.
    Returns:
      N_by_arm: LongTensor [K] counts of pre-episode pulls per arm
      S0: list[int] successes per arm in the pre-episode
      F0: list[int] failures per arm in the pre-episode
    """
    a = torch.full((K,), alpha0, dtype=torch.float32)
    b = torch.full((K,), beta0,  dtype=torch.float32)
    N_by_arm = torch.zeros(K, dtype=torch.long)
    S0 = [0] * K

    for _ in range(int(N_tot)):
        arm = ts_argmax(a, b)
        r = bernoulli(thetas[arm])
        N_by_arm[arm] += 1
        if r == 1:
            S0[arm] += 1
        # update the demonstrator's internal belief used during pre-history
        a[arm] += r
        b[arm] += 1 - r

    F0 = [int(N_by_arm[k].item()) - S0[k] for k in range(K)]
    return N_by_arm, S0, F0


# ----------- Round-Based Simulators -----------

def simulate_segment(
    *,
    mode: str,  # "demo" or "learner"
    steps: int,
    K: int,
    thetas: List[float],
    alpha: torch.Tensor,  # [K] current belief shapes for this agent
    beta: torch.Tensor,   # [K]
    bandit_id: int,
    agent_id: int,  # demonstrator_id (1, 2, ...) or 0 for learner
    strategy: str = "thompson",  # "thompson" or "random_exclusion"
) -> Tuple[List[Tuple[int,int,int,int]], torch.Tensor, torch.Tensor]:
    """
    Simulate ONE segment (demo or learner acting for 'steps' timesteps).
    Args:
        strategy: "thompson" for Thompson Sampling, "random_exclusion" for random arm selection with exclusion
    Returns:
        events: list of (bandit_id, agent_id, action, reward)
        alpha_after: updated alpha [K]
        beta_after: updated beta [K]
    """
    events = []
    a = alpha.clone()
    b = beta.clone()
    
    for _ in range(steps):
        # Select arm based on strategy
        if strategy == "random_exclusion":
            arm = random_arm_with_exclusion(K, thetas, num_exclude=6)
        else:  # default to thompson
            arm = ts_argmax(a, b)
        
        r = bernoulli(thetas[arm])
        events.append((bandit_id, agent_id, arm, r))
        # Update beliefs regardless of strategy
        a[arm] += r
        b[arm] += 1 - r
    
    return events, a, b


def simulate_rounds_run(
    *,
    K: int = 12,
    num_demonstrators: int = 2,
    num_rounds: int = 3,
    change_bandits_with_rounds: bool = True,
    order: int = 0,  # 0: demo-first, 1: learner-first
    T_demo: int = 6,
    T_learner: int = 6,
    alpha0: float = 1.0,
    beta0: float = 1.0,
    demo_strategy: str = "uniform_thompson",
    expert_N_min: int = 60,
    expert_N_max: int = 72,
    bandit_strategy: str = "sample",
    bandit_payoffs: List[float] = None,
) -> Dict:
    """
    Simulate a run with multiple rounds of alternating demonstrators and learner.

    Args:
        K: number of arms
        num_demonstrators: number of local demonstrators (1, 2, or more)
        num_rounds: number of rounds
        change_bandits_with_rounds: if True, new bandit each round; else same bandit
        order: 0 = demo-first (Demo1, Learner, Demo2, Learner, ...)
               1 = learner-first (Learner, Demo1, Learner, Demo2, ...)
        T_demo: timesteps per demonstrator segment
        T_learner: timesteps per learner segment
        alpha0, beta0: prior parameters
        demo_strategy: "uniform_thompson", "expert_novice_thompson", or "expert_thompson_novice_random"
        expert_N_min: minimum N_tot for experts (used in expert_novice strategies)
        expert_N_max: maximum N_tot for experts / max experience for uniform_thompson
        bandit_strategy: "sample" for random Beta sampling, "permute" for fixed payoffs permuted
        bandit_payoffs: Fixed payoff values to permute (required if bandit_strategy="permute", must have length K)
    
    Returns:
        Dictionary with events, round_boundaries, demonstrators, bandits, meta
    """
    LEARNER_ID = 0
    
    # Sample bandits based on strategy
    num_bandits = num_rounds if change_bandits_with_rounds else 1
    if bandit_strategy == "sample":
        bandit_thetas = [sample_bandit(K) for _ in range(num_bandits)]
    elif bandit_strategy == "permute":
        if bandit_payoffs is None or len(bandit_payoffs) != K:
            raise ValueError(f"bandit_payoffs must have exactly K={K} values for permute strategy, got {len(bandit_payoffs) if bandit_payoffs else 0}")
        bandit_thetas = [permute_bandit(bandit_payoffs) for _ in range(num_bandits)]
    else:
        raise ValueError(f"Unknown bandit_strategy: {bandit_strategy}")
    
    # Sample N_tot for each demonstrator based on strategy
    demo_N_tots = {}
    demo_strategies = {}  # Track which strategy each demonstrator uses
    
    if demo_strategy == "uniform_thompson":
        # Random N_tot for all demonstrators, sampled from [0, expert_N_max]
        for demo_id in range(num_demonstrators):
            demo_N_tots[demo_id] = int(torch.randint(low=0, high=expert_N_max + 1, size=(1,)).item())
            demo_strategies[demo_id] = "thompson"
    
    elif demo_strategy in ["expert_novice_thompson", "expert_thompson_novice_random"]:
        # Expert-novice assignment
        # Randomly assign which demo is expert (50/50 for demo_id 0 or 1)
        expert_id = random.choice([0, 1]) if num_demonstrators == 2 else 0
        novice_id = 1 - expert_id if num_demonstrators == 2 else 1
        
        # Sample expert N_tot from [expert_N_min, expert_N_max]
        expert_N_tot = int(torch.randint(low=expert_N_min, high=expert_N_max + 1, size=(1,)).item())
        
        demo_N_tots[expert_id] = expert_N_tot
        demo_N_tots[novice_id] = 0
        demo_strategies[expert_id] = "thompson"
        
        # Set novice strategy based on demo_strategy
        if demo_strategy == "expert_novice_thompson":
            demo_strategies[novice_id] = "thompson"
        else:  # expert_thompson_novice_random
            demo_strategies[novice_id] = "random_exclusion"
    
    else:
        raise ValueError(f"Unknown demo_strategy: {demo_strategy}")
    
    # Pre-compute pre-episode beliefs for each demonstrator on each bandit
    demo_pre_beliefs = {}  # [demo_id][bandit_idx] -> {N_tot, N_by_arm, S0, F0, alpha0, beta0}
    for demo_id in range(num_demonstrators):
        demo_pre_beliefs[demo_id] = {}
        N_tot = demo_N_tots[demo_id]
        for b_idx, thetas in enumerate(bandit_thetas):
            N_by_arm, S0, F0 = allocate_prehistory_ts(K, N_tot, thetas, alpha0=alpha0, beta0=beta0)
            demo_pre_beliefs[demo_id][b_idx] = {
                "N_tot": int(N_tot),
                "N_by_arm": N_by_arm.tolist(),
                "S0": S0,
                "F0": F0,
                "alpha0": [alpha0 + s for s in S0],
                "beta0": [beta0 + f for f in F0],
            }
    
    # Initialize learner belief (shared across all rounds)
    learner_alpha = torch.full((K,), alpha0, dtype=torch.float32)
    learner_beta = torch.full((K,), beta0, dtype=torch.float32)
    
    # Track demonstrator beliefs (they persist across rounds on same bandit, reset on new bandit)
    demo_alphas = {}  # [demo_id] -> current alpha [K]
    demo_betas = {}   # [demo_id] -> current beta [K]
    
    all_events = []
    round_boundaries = [0]  # Position where each round starts
    round_bandit_map = []  # Which bandit for each round
    segment_boundaries = [0]  # Position where each segment starts
    segment_info = []  # Info about each segment
    
    # Simulate each round
    for round_idx in range(num_rounds):
        bandit_idx = round_idx if change_bandits_with_rounds else 0
        thetas = bandit_thetas[bandit_idx]
        round_bandit_map.append(bandit_idx)
        
        # Reset demonstrator beliefs if new bandit
        if round_idx == 0 or change_bandits_with_rounds:
            for demo_id in range(num_demonstrators):
                beliefs = demo_pre_beliefs[demo_id][bandit_idx]
                demo_alphas[demo_id] = torch.tensor(beliefs["alpha0"], dtype=torch.float32)
                demo_betas[demo_id] = torch.tensor(beliefs["beta0"], dtype=torch.float32)
        
        # Add BANDIT_SEP token at start of round
        all_events.append((bandit_idx, -1, -1, -1))
        # Track BANDIT_SEP as a segment
        segment_boundaries.append(len(all_events))
        segment_info.append({"round": round_idx, "type": "bandit_sep", "agent": -1})
        
        # Define segment order
        if order == 0:
            # Demo-first: Demo1, Learner, Demo2, Learner, Demo3, Learner, ...
            segments = []
            for demo_id in range(num_demonstrators):
                segments.append(("demo", demo_id))
                segments.append(("learner", LEARNER_ID))
        else:  # order == 1
            # Learner-first: Learner, Demo1, Learner, Demo2, Learner, Demo3, ...
            segments = []
            for demo_id in range(num_demonstrators):
                segments.append(("learner", LEARNER_ID))
                segments.append(("demo", demo_id))
        
        # Simulate each segment in the round
        for seg_type, demo_id in segments:
            if seg_type == "demo":
                # Convert demo_id (0, 1, ...) to agent_id (1, 2, ...) for events
                # Learner uses agent_id 0, so demonstrators are 1, 2, ...
                agent_id = demo_id + 1
                events, new_alpha, new_beta = simulate_segment(
                    mode="demo",
                    steps=T_demo,
                    K=K,
                    thetas=thetas,
                    alpha=demo_alphas[demo_id],
                    beta=demo_betas[demo_id],
                    bandit_id=bandit_idx,
                    agent_id=agent_id,
                    strategy=demo_strategies[demo_id],
                )
                demo_alphas[demo_id] = new_alpha
                demo_betas[demo_id] = new_beta
                
                # Learner also observes demo actions
                for (_, _, arm, r) in events:
                    learner_alpha[arm] += r
                    learner_beta[arm] += 1 - r
                    
            else:  # learner
                agent_id = LEARNER_ID
                events, learner_alpha, learner_beta = simulate_segment(
                    mode="learner",
                    steps=T_learner,
                    K=K,
                    thetas=thetas,
                    alpha=learner_alpha,
                    beta=learner_beta,
                    bandit_id=bandit_idx,
                    agent_id=agent_id,
                    strategy="thompson",
                )
            
            all_events.extend(events)
            # Track segment boundary after adding events
            segment_boundaries.append(len(all_events))
            segment_info.append({"round": round_idx, "type": seg_type, "agent": agent_id})
        
        # Add QUERY token at end of round (after all segments)
        # Use -2 in action field to distinguish from BANDIT_SEP (-1)
        all_events.append((bandit_idx, -1, -2, -1))
        segment_boundaries.append(len(all_events))
        segment_info.append({"round": round_idx, "type": "query", "agent": -1, "bandit": bandit_idx})
        
        # Record boundary for next round
        round_boundaries.append(len(all_events))
    
    # Build output structure
    demonstrators_out = []
    for demo_id in range(num_demonstrators):
        demonstrators_out.append({
            "demonstrator_id": demo_id,
            "N_tot": demo_N_tots[demo_id],
            "strategy": demo_strategies[demo_id],
        })
    
    bandits_out = []
    for b_idx, thetas in enumerate(bandit_thetas):
        bandit_info = {
            "bandit_index": b_idx,
            "theta": thetas,
            "demonstrators": {}
        }
        for demo_id in range(num_demonstrators):
            beliefs = demo_pre_beliefs[demo_id][b_idx]
            bandit_info["demonstrators"][demo_id] = {
                "N_tot": beliefs["N_tot"],
                "N_by_arm": beliefs["N_by_arm"],
                "S0": beliefs["S0"],
                "F0": beliefs["F0"],
                "alpha0": beliefs["alpha0"],
                "beta0": beliefs["beta0"],
            }
        bandits_out.append(bandit_info)
    
    return {
        "meta": {
            "K": K,
            "num_demonstrators": num_demonstrators,
            "num_rounds": num_rounds,
            "change_bandits_with_rounds": change_bandits_with_rounds,
            "order": order,
            "T_demo": T_demo,
            "T_learner": T_learner,
            "alpha0": alpha0,
            "beta0": beta0,
            "demo_strategy": demo_strategy,
            "expert_N_min": expert_N_min,
            "expert_N_max": expert_N_max,
            "bandit_strategy": bandit_strategy,
            "bandit_payoffs": bandit_payoffs if bandit_strategy == "permute" else None,
            "prehistory": "TS",
        },
        "events": all_events,
        "round_boundaries": round_boundaries,
        "round_bandit_map": round_bandit_map,
        "segment_boundaries": segment_boundaries,
        "segment_info": segment_info,
        "demonstrators": demonstrators_out,
        "bandits": bandits_out,
    }


def build_training_example(run: Dict) -> Dict:
    """
    Convert a run into a training example.
    
    The example contains all events and targets for predicting demonstrator
    pre-episode beliefs at each round boundary (where bandits appear).
    
    Returns:
        {
            "events": [...],
            "round_boundaries": [...],
            "segment_boundaries": [...],
            "segment_info": [...],
            "labels": {
                "bandits": [bandit_info, ...],  # One per round
                "round_bandit_map": [...],
            }
        }
    """
    # The per-row `meta` field (full config snapshot) is no longer carried —
    # it's identical across all rows in a generation run and now lives in the
    # single root-level metadata.json that `generate_dataset` writes.
    return {
        "events": run["events"],
        "round_boundaries": run["round_boundaries"],
        "round_bandit_map": run["round_bandit_map"],
        "segment_boundaries": run["segment_boundaries"],
        "segment_info": run["segment_info"],
        "labels": {
            "bandits": run["bandits"],
            "K": run["meta"]["K"],
            "num_demonstrators": run["meta"]["num_demonstrators"],
        },
    }


# ------------ Dataset Generation ------------

def generate_split(
    out_root: Path,
    split: str,
    num_runs: int,
    *,
    K: int,
    num_demonstrators: int,
    num_rounds: int,
    change_bandits_with_rounds: bool,
    order: int,
    T_demo: int,
    T_learner: int,
    alpha0: float,
    beta0: float,
    demo_strategy: str,
    expert_N_min: int,
    expert_N_max: int,
    bandit_strategy: str,
    bandit_payoffs: Optional[List[float]],
) -> int:
    """Generate one split and save it as a single mmap-backed Arrow dataset at
    ``out_root/split/``. Returns the number of examples written.

    The per-example encoding (5 token channels, label tensor construction,
    segment_info flattening) happens here via ``build_arrow_row`` so rows hit
    disk pre-encoded and the training inner loop does almost no Python work.
    """
    num_bandits = num_rounds if change_bandits_with_rounds else 1
    num_agents = num_demonstrators + 1
    features = build_features(K, num_demonstrators, num_bandits)

    # Same order pool as the pre-Arrow generator: shuffled mix of order=0/1
    # rows when --order=2, otherwise all rows use the single specified order.
    orders = [0, 1] if order == 2 else [order]
    pool: List[int] = []
    runs_per_order = math.ceil(num_runs / len(orders))
    for ord_val in orders:
        pool.extend([ord_val] * runs_per_order)
    pool = pool[:num_runs]
    random.shuffle(pool)

    def gen():
        for ord_val in pool:
            run = simulate_rounds_run(
                K=K,
                num_demonstrators=num_demonstrators,
                num_rounds=num_rounds,
                change_bandits_with_rounds=change_bandits_with_rounds,
                order=ord_val,
                T_demo=T_demo,
                T_learner=T_learner,
                alpha0=alpha0,
                beta0=beta0,
                demo_strategy=demo_strategy,
                expert_N_min=expert_N_min,
                expert_N_max=expert_N_max,
                bandit_strategy=bandit_strategy,
                bandit_payoffs=bandit_payoffs,
            )
            example = build_training_example(run)
            yield build_arrow_row(
                example,
                K=K,
                num_demonstrators=num_demonstrators,
                num_bandits=num_bandits,
                num_agents=num_agents,
            )

    split_dir = out_root / split
    if split_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing dataset at {split_dir}. "
            "Pick a fresh output directory or delete the old one explicitly."
        )

    ds = Dataset.from_generator(gen, features=features)
    ds.save_to_disk(str(split_dir))
    print(f"Wrote {len(ds)} examples to {split_dir}")
    return len(ds)


def write_root_metadata(
    out_root: Path,
    *,
    K: int,
    num_demonstrators: int,
    num_rounds: int,
    change_bandits_with_rounds: bool,
    order: int,
    T_demo: int,
    T_learner: int,
    alpha0: float,
    beta0: float,
    demo_strategy: str,
    expert_N_min: int,
    expert_N_max: int,
    bandit_strategy: str,
    bandit_payoffs: Optional[List[float]],
    num_train_runs: int,
    num_valid_runs: int,
    num_test_runs: int,
    seed: int,
) -> Path:
    """Write the single root metadata.json (replaces dataset_metadata.json +
    per-split manifest.json from the .pt-shard era). Train.py / eval.py read
    constants like K, num_demonstrators, expert_N_max, alpha0, beta0 from here.
    """
    metadata = {
        "dataset_name": out_root.name,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "format": "huggingface_arrow",
        "seed": seed,
        "parameters": {
            "K": K,
            "num_demonstrators": num_demonstrators,
            "num_rounds": num_rounds,
            "change_bandits_with_rounds": change_bandits_with_rounds,
            "order": order,
            "T_demo": T_demo,
            "T_learner": T_learner,
            "alpha0": alpha0,
            "beta0": beta0,
            "demo_strategy": demo_strategy,
            "expert_N_min": expert_N_min,
            "expert_N_max": expert_N_max,
            "bandit_strategy": bandit_strategy,
            "bandit_payoffs": bandit_payoffs if bandit_strategy == "permute" else None,
            "num_train_runs": num_train_runs,
            "num_valid_runs": num_valid_runs,
            "num_test_runs": num_test_runs,
        },
        # Encoding tables — useful for reproducibility and for any external
        # consumer that wants to inspect rows without importing the codebase.
        "segment_type_to_id": SEGMENT_TYPE_TO_ID,
    }
    metadata_path = out_root / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote dataset metadata to {metadata_path}")
    return metadata_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data")
    parser.add_argument("--K", type=int, default=12)
    parser.add_argument("--num_demonstrators", type=int, default=2)
    parser.add_argument("--num_rounds", type=int, default=3)
    parser.add_argument("--change_bandits_with_rounds", action="store_true", default=True)
    parser.add_argument("--order", type=int, default=2, choices=[0, 1, 2])
    parser.add_argument("--num_train_runs", type=int, default=6000)
    parser.add_argument("--num_valid_runs", type=int, default=400)
    parser.add_argument("--num_test_runs", type=int, default=400)
    parser.add_argument("--T_demo", type=int, default=6)
    parser.add_argument("--T_learner", type=int, default=6)
    parser.add_argument("--alpha0", type=float, default=1.0,
                        help="Beta prior alpha (also stored in metadata.json for the loader)")
    parser.add_argument("--beta0", type=float, default=1.0,
                        help="Beta prior beta (also stored in metadata.json for the loader)")
    parser.add_argument("--demo_strategy", type=str, default="uniform_thompson",
                        choices=["uniform_thompson", "expert_novice_thompson", "expert_thompson_novice_random"],
                        help="Demonstrator experience strategy")
    parser.add_argument("--expert_N_min", type=int, default=60,
                        help="Minimum N_tot for experts")
    parser.add_argument("--expert_N_max", type=int, default=72,
                        help="Maximum N_tot for experts")
    parser.add_argument("--bandit_strategy", type=str, default="sample",
                        choices=["sample", "permute"],
                        help="Bandit generation strategy: 'sample' (random Beta) or 'permute' (fixed payoffs permuted)")
    parser.add_argument("--bandit_payoffs", type=str, default="0.0,0.1,0.2,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.75,0.8",
                        help="Comma-separated payoff values for permute strategy (must have exactly K values)")
    parser.add_argument("--seed", type=int, default=42, help="Global RNG seed")

    args = parser.parse_args()

    # Parse bandit payoffs if needed
    bandit_payoffs = None
    if args.bandit_strategy == "permute":
        bandit_payoffs = [float(x.strip()) for x in args.bandit_payoffs.split(",")]
        if len(bandit_payoffs) != args.K:
            raise ValueError(
                f"bandit_payoffs must have exactly K={args.K} values, got {len(bandit_payoffs)}"
            )

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    common_kwargs = dict(
        K=args.K,
        num_demonstrators=args.num_demonstrators,
        num_rounds=args.num_rounds,
        change_bandits_with_rounds=args.change_bandits_with_rounds,
        order=args.order,
        T_demo=args.T_demo,
        T_learner=args.T_learner,
        alpha0=args.alpha0,
        beta0=args.beta0,
        demo_strategy=args.demo_strategy,
        expert_N_min=args.expert_N_min,
        expert_N_max=args.expert_N_max,
        bandit_strategy=args.bandit_strategy,
        bandit_payoffs=bandit_payoffs,
    )

    for split, n in [("train", args.num_train_runs),
                     ("val",   args.num_valid_runs),
                     ("test",  args.num_test_runs)]:
        generate_split(out_root, split, n, **common_kwargs)

    write_root_metadata(
        out_root,
        num_train_runs=args.num_train_runs,
        num_valid_runs=args.num_valid_runs,
        num_test_runs=args.num_test_runs,
        seed=args.seed,
        **common_kwargs,
    )
    print(f"\nDone. Dataset at {out_root}")
