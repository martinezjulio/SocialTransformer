#%% 
import math, random, os, json
from typing import Dict, List, Tuple
import torch

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

def sample_bandit(K: int, min_gap: float = 0.05, min_best: float = 0.70) -> List[float]:
    """Sample K payoffs from Beta(2,2) until there's a clear best arm."""
    while True:
        thetas = torch.distributions.Beta(torch.full((K,), 2.0), torch.full((K,), 2.0)).sample().tolist()
        best, second = sorted(thetas, reverse=True)[:2]
        if best - second >= min_gap and best >= min_best:
            return thetas

def sample_total_experience(N_max: int) -> int:
    """Sample N_tot ~ DiscreteUniform{0..N_max}"""
    N_tot = int(torch.randint(low=0, high=N_max + 1, size=(1,)).item())
    return N_tot

def sample_dirichlet_allocation(K: int, dirichlet_eta: float, N_tot: int) -> Tuple[int, torch.Tensor]:
    """Draw N_by_arm ~ Multinomial(N_tot, w), w ~ Dir(eta).
    Returns:
        N_by_arm (LongTensor [K] summing to N_tot)
    """
    w = torch.distributions.Dirichlet(torch.full((K,), float(dirichlet_eta))).sample()
    N_by_arm = torch.distributions.Multinomial(total_count=N_tot, probs=w).sample().to(torch.int64)
    return N_tot, N_by_arm

def sample_pre_episode_counts(N_by_arm: torch.Tensor, thetas: List[float]) -> Tuple[List[int], List[int]]:
    """Sample S0,F0 with S0_k ~ Binomial(N_k, theta_k)."""
    S0, F0 = [], []
    for n_k, theta_k in zip(N_by_arm.tolist(), thetas):
        s = int(torch.distributions.Binomial(total_count=int(n_k), probs=theta_k).sample().item()) if n_k > 0 else 0
        S0.append(s)
        F0.append(int(n_k) - s)
    return S0, F0

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


def prehistory_allocate_softmax(K, N_tot, thetas, N_max, tau_min=0.0, tau_max=6.0, p=1.0):
    """Alternative to allocate_prehistory_ts but faster"""
    t = float(N_tot) / max(1, N_max)
    tau = tau_min + (tau_max - tau_min) * (t ** p)
    logits = torch.tensor([math.log(t/(1-t)) for t in thetas], dtype=torch.float32)  # logit(theta)
    probs = torch.softmax(tau * logits, dim=0)
    N_by_arm = torch.distributions.Multinomial(total_count=int(N_tot), probs=probs).sample().to(torch.int64)
    S0 = [int(torch.distributions.Binomial(total_count=int(n), probs=float(thetas[k])).sample().item())
          if N_by_arm[k] > 0 else 0
          for k, n in enumerate(N_by_arm.tolist())]
    F0 = [int(n) - s for n, s in zip(N_by_arm.tolist(), S0)]
    return N_by_arm, S0, F0

# ----------- Simulators -----------

def simulate_bandit_episode_block(
    *,
    mode: str,                      # "demo" or "learner"
    steps: int,                     # T_demo if demo, T_learner if learner
    K: int,
    thetas: List[float],
    alphaD: torch.Tensor,           # [K] current demo internal shapes
    betaD:  torch.Tensor,           # [K]
    alphaL: torch.Tensor,           # [K] current learner evidence shapes
    betaL:  torch.Tensor,           # [K]
    bandit_id: int,
    demonstrator_id: int,
) -> Dict:
    """
    Run ONE episode block and return a dictionary with events and states.

    - If mode=="demo": demo acts 'steps' times with TS on (alphaD,betaD).
      Demo internal updates; learner ALSO updates on demo rewards.
    - If mode=="learner": learner acts 'steps' times with TS on (alphaL,betaL).
      Only learner updates.
    """
    events: List[Tuple[int,int,int,int]] = []

    alphaD_before = alphaD.tolist()
    betaD_before  = betaD.tolist()
    alphaL_before = alphaL.tolist()
    betaL_before  = betaL.tolist()

    aD = alphaD.clone()
    bD = betaD.clone()
    aL = alphaL.clone()
    bL = betaL.clone()

    if steps > 0:
        if mode == "demo":
            for _ in range(steps):
                a = ts_argmax(aD, bD)
                r = bernoulli(thetas[a])
                events.append((bandit_id, demonstrator_id, a, r))
                aD[a] += r
                bD[a] += 1 - r
                aL[a] += r     # learner sees demo reward
                bL[a] += 1 - r
        elif mode == "learner":
            for _ in range(steps):
                a = ts_argmax(aL, bL)
                r = bernoulli(thetas[a])
                events.append((bandit_id, 99, a, r))
                aL[a] += r
                bL[a] += 1 - r
        else:
            raise ValueError(f"mode must be 'demo' or 'learner', got {mode}")

    return {
        "mode": mode,
        "steps": steps,
        "bandit_id": bandit_id,
        "demonstrator_id": demonstrator_id,
        "events": events,
        "alphaD_before": alphaD_before,
        "betaD_before":  betaD_before,
        "alphaL_before": alphaL_before,
        "betaL_before":  betaL_before,
        "alphaD": aD.tolist(),
        "betaD":  bD.tolist(),
        "alphaL": aL.tolist(),
        "betaL":  bL.tolist(),
    }

#%%
def simulate_multi_episode_run(
    K: int = 12,
    num_bandits: int = 3,
    episodes_per_bandit: int = 1,
    order: int = 0,                  # 0: demo -> learner; 1: learner -> demo
    num_demonstrators: int = 1,
    demos_per_bandit: int = 1,
    T_demo: int = 6,
    T_learner: int = 6,
    alpha0: float = 1.0,
    beta0: float  = 1.0,
    N_max: int = 72,
) -> Dict:
    """
    Build a multi-bandit run as a sequence of EPISODES (each is a demo- or learner-block).

    Returns a structured dict with meta, demonstrators (their bandit blocks & N_tot),
    per-bandit info (theta, S0/F0, N_by_arm), and per-episode records
    (events, states_before/after).
    """
    # 1) Sample bandits
    bandit_thetas: List[List[float]] = [sample_bandit(K) for _ in range(num_bandits)]

    # 2) Contiguous assignment of bandits to demonstrators
    if not (1 <= num_demonstrators <= num_bandits):
        raise ValueError("num_demonstrators must be in [1, num_bandits].")
    if num_demonstrators == 1:
        cuts = []
    else:
        cuts = sorted(random.sample(range(1, num_bandits), k=num_demonstrators - 1))
    bounds = [0] + cuts + [num_bandits]

    bandit_to_demo: Dict[int, int] = {}
    demo_blocks: Dict[int, List[int]] = {}
    for demo_id, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
        demo_blocks[demo_id] = list(range(lo, hi))
        for b in range(lo, hi):
            bandit_to_demo[b] = demo_id

    # 3) N_tot per demonstrator (reused)
    demo_N_tot: Dict[int, int] = {
        d: int(torch.randint(low=0, high=N_max + 1, size=(1,)).item())
        for d in range(num_demonstrators)
    }

    bandits_out: List[Dict] = []
    episodes_out: List[Dict] = []
    epi_idx = 0

    # 4) Iterate bandits
    for b in range(num_bandits):
        thetas = bandit_thetas[b]
        
        # Determine which demonstrators act on this bandit
        if demos_per_bandit == 1:
            # Original behavior: one demo handles all episodes
            demonstrator_id = bandit_to_demo[b]
            acting_demos = [demonstrator_id]
        else:
            # Multiple demos per bandit: use first demos_per_bandit demonstrators
            acting_demos = list(range(min(demos_per_bandit, num_demonstrators)))
        
        # Pre-compute pre-histories for all acting demonstrators
        demo_states = {}
        for demo_id in acting_demos:
            N_tot = demo_N_tot[demo_id]
            N_by_arm, S0, F0 = allocate_prehistory_ts(K, N_tot, thetas, alpha0=alpha0, beta0=beta0)
            demo_states[demo_id] = {
                "N_tot": int(N_tot),
                "N_by_arm": N_by_arm.tolist(),
                "S0": S0,
                "F0": F0,
                "alphaD": torch.tensor([alpha0 + s for s in S0], dtype=torch.float32),
                "betaD": torch.tensor([beta0 + f for f in F0], dtype=torch.float32),
            }
        
        # Use first demonstrator's info for bandit metadata (for backwards compatibility)
        primary_demo = acting_demos[0]
        bandits_out.append({
            "bandit_index": b,
            "demonstrator_id": primary_demo,
            "theta": thetas,
            "N_tot": demo_states[primary_demo]["N_tot"],
            "N_by_arm": demo_states[primary_demo]["N_by_arm"],
            "S0": demo_states[primary_demo]["S0"],
            "F0": demo_states[primary_demo]["F0"],
            "acting_demonstrators": acting_demos,  # NEW: list of all demos acting on this bandit
        })
        
        # Initialize learner state (shared across all demos)
        alphaL = torch.full((K,), alpha0, dtype=torch.float32)
        betaL = torch.full((K,), beta0,  dtype=torch.float32)
        
        # Repetitions for this bandit, cycling through demonstrators
        demo_cycle_idx = 0
        for rep in range(episodes_per_bandit):
            # Select which demonstrator acts in this episode
            current_demo_id = acting_demos[demo_cycle_idx % len(acting_demos)]
            alphaD = demo_states[current_demo_id]["alphaD"].clone()
            betaD = demo_states[current_demo_id]["betaD"].clone()
            
            pair = [("demo", T_demo), ("learner", T_learner)] if order == 0 else [("learner", T_learner), ("demo", T_demo)]
            for ep_mode, ep_steps in pair:
                block = simulate_bandit_episode_block(
                    mode=ep_mode, steps=ep_steps, K=K, thetas=thetas,
                    alphaD=alphaD, betaD=betaD, alphaL=alphaL, betaL=betaL,
                    bandit_id=b, demonstrator_id=current_demo_id,
                )
                
                # Update states
                alphaD = torch.tensor(block["alphaD"], dtype=torch.float32)
                betaD  = torch.tensor(block["betaD"],  dtype=torch.float32)
                alphaL = torch.tensor(block["alphaL"], dtype=torch.float32)
                betaL  = torch.tensor(block["betaL"],  dtype=torch.float32)
                
                # Store updated demo state back
                demo_states[current_demo_id]["alphaD"] = alphaD
                demo_states[current_demo_id]["betaD"] = betaD
                
                episodes_out.append({
                    "episode_index": epi_idx,
                    "bandit_index": b,
                    "episode_type": ep_mode,
                    "demonstrator_id": current_demo_id,
                    "steps": ep_steps,
                    "events": block["events"],
                    "states_before": {
                        "alphaD": block["alphaD_before"],
                        "betaD":  block["betaD_before"],
                        "alphaL": block["alphaL_before"],
                        "betaL":  block["betaL_before"],
                    },
                    "states_after": {
                        "alphaD": block["alphaD"],
                        "betaD":  block["betaD"],
                        "alphaL": block["alphaL"],
                        "betaL":  block["betaL"],
                    },
                })
                epi_idx += 1
            
            # Move to next demonstrator for next episode
            demo_cycle_idx += 1

    return {
        "meta": {
            "K": K,
            "num_bandits": num_bandits,
            "episodes_per_bandit": episodes_per_bandit,
            "order": order,  # 0 demo->learner, 1 learner->demo
            "num_demonstrators": num_demonstrators,
            "demos_per_bandit": demos_per_bandit,
            "T_demo": T_demo,
            "T_learner": T_learner,
            "alpha0": alpha0,
            "beta0": beta0,
            "N_max": N_max,
            "prehistory": "TS",   # document allocation mode
        },
        "demonstrators": [
            {
                "demonstrator_id": d,
                "bandit_indices": demo_blocks[d],
                "N_tot": int(demo_N_tot[d]),
            }
            for d in range(num_demonstrators)
        ],
        "bandits": bandits_out,    # per-bandit facts (theta, S0/F0, N_by_arm)
        "episodes": episodes_out,  # chronological episode list with events and states
    }

#%%
def build_training_examples_from_run(
    run: Dict,
    alpha0: float = 1.0,
    beta0: float = 1.0,
    max_context_bandits: int = 2,
) -> List[Dict]:
    examples: List[Dict] = []

    bandit_infos = {b["bandit_index"]: b for b in run["bandits"]}
    episodes = run["episodes"]

    # reconstruct per-bandit streams in chrono order
    per_bandit_events = {idx: [] for idx in bandit_infos.keys()}
    for E in episodes:
        per_bandit_events[E["bandit_index"]].extend(E["events"])

    bandit_indices = sorted(bandit_infos.keys())

    for i, b_idx in enumerate(bandit_indices):
        # context: up to last N prior bandits
        ctx_events = []
        prior_idxs = bandit_indices[max(0, i - max_context_bandits):i]
        if not prior_idxs:
            ctx_events.append((-1, -1, -1, -1))  # NOCTX
        else:
            for pb in prior_idxs:
                ctx_events.append((pb, -1, -1, -1))       # BANDIT_SEP
                ctx_events.extend(per_bandit_events[pb])

        # current bandit
        cur_events = [(b_idx, -1, -1, -1)]                # BANDIT_SEP
        cur_events.extend(per_bandit_events[b_idx])

        binfo = bandit_infos[b_idx]
        thetas = binfo["theta"]
        S0, F0 = binfo["S0"], binfo["F0"]

        labels = {
            "theta": thetas,
            "alpha_demo0": [alpha0 + s for s in S0],
            "beta_demo0":  [beta0  + f for f in F0],
            "demonstrator_id": binfo["demonstrator_id"],
            "K": len(thetas),
            "N_tot": binfo.get("N_tot", None),
            "N_by_arm": binfo.get("N_by_arm", None),
        }

        examples.append({
            "context_events": ctx_events,
            "current_events": cur_events,
            "labels": labels,
        })

    return examples

# ------------ Dataset Generation ------------

#%%
def generate_dataset_streaming(
    out_dir: str,
    num_runs: int,
    *,
    # fixed knobs
    K: int = 12,
    num_bandits: int = 3,
    T_demo: int = 6,
    T_learner: int = 6,
    alpha0: float = 1.0,
    beta0: float = 1.0,
    N_max: int = 72,
    max_context_bandits: int = 2,
    max_num_demos: int = 2,
    max_demos_per_bandit: int = 2,
    max_episodes_per_bandit: int = 2,
    # sharding
    split: str = "train",
    shard_size: int = 10000,       # ~1e4–2e4 is a good start
    manifest_name: str = "manifest.json",
) -> str:
    """
    Stream-generate examples and save shards:
      out_dir/{split}-{00001..}.pt
    Also writes a manifest JSON listing shard paths and sizes.
    Returns manifest path.
    """
    os.makedirs(out_dir, exist_ok=True)

    def _all_valid_combos(num_bandits=3, max_num_demos=2, max_demos_per_bandit=2, max_episodes_per_bandit=2):
        combos = []
        nb = num_bandits  # Use as fixed value, don't iterate
        for epb in range(1, max_episodes_per_bandit + 1):
            for nd in range(1, min(nb, max_num_demos) + 1):
                for dpb in range(1, min(nd, epb, max_demos_per_bandit) + 1):
                    for ord_ in (0, 1):
                        combos.append(dict(
                            num_bandits=nb,
                            episodes_per_bandit=epb,
                            order=ord_,
                            num_demonstrators=nd,
                            demos_per_bandit=dpb,
                        ))
        return combos

    grid = _all_valid_combos(
        num_bandits=num_bandits,
        max_num_demos=max_num_demos,
        max_demos_per_bandit=max_demos_per_bandit,
        max_episodes_per_bandit=max_episodes_per_bandit
    )
    reps = math.ceil(num_runs / len(grid))
    pool = (grid * reps)[:num_runs]
    random.shuffle(pool)
    combo_iter = pool


    shard_idx = 0
    buf: List[Dict] = []
    manifest = {
        "split": split,
        "meta": {
            "K": K, "num_bandits": num_bandits, "T_demo": T_demo, "T_learner": T_learner,
            "alpha0": alpha0, "beta0": beta0, "N_max": N_max,
            "max_context_bandits": max_context_bandits,
            "max_num_demos": max_num_demos,
            "max_demos_per_bandit": max_demos_per_bandit,
            "max_episodes_per_bandit": max_episodes_per_bandit,
            "coverage": "exhaustive_grid",
            "shard_size": shard_size,
        },
        "shards": []  # [{"path": "...", "num_examples": int}]
    }

    def _flush():
        nonlocal shard_idx, buf
        if not buf: return
        shard_idx += 1
        fname = f"{split}-{shard_idx:05d}.pt"
        fpath = os.path.join(out_dir, fname)
        torch.save(buf, fpath)
        manifest["shards"].append({"path": fname, "num_examples": len(buf)})
        buf = []

    for cfg in combo_iter:
        run = simulate_multi_episode_run(
            K=K,
            num_bandits=cfg["num_bandits"],
            episodes_per_bandit=cfg["episodes_per_bandit"],
            order=cfg["order"],
            num_demonstrators=cfg["num_demonstrators"],
            demos_per_bandit=cfg["demos_per_bandit"],
            T_demo=T_demo,
            T_learner=T_learner,
            alpha0=alpha0,
            beta0=beta0,
            N_max=N_max,
        )
        exs = build_training_examples_from_run(
            run, alpha0=alpha0, beta0=beta0, max_context_bandits=max_context_bandits
        )
        for ex in exs:
            buf.append(ex)
            if len(buf) >= shard_size:
                _flush()

    _flush()  # write last partial shard

    manifest_path = os.path.join(out_dir, manifest_name)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {len(manifest['shards'])} shards to {out_dir} (manifest: {manifest_path})")
    return manifest_path



#%%
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data")
    parser.add_argument("--K", type=int, default=12)
    parser.add_argument("--N_max", type=int, default=72)
    parser.add_argument("--num_bandits", type=int, default=3)
    parser.add_argument("--max_num_demos", type=int, default=2)
    parser.add_argument("--max_context_bandits", type=int, default=2)
    parser.add_argument("--num_train_runs", type=int, default=6000)
    parser.add_argument("--num_valid_runs", type=int, default=400)
    parser.add_argument("--num_test_runs", type=int, default=400)
    parser.add_argument("--T_demo", type=int, default=6)
    parser.add_argument("--T_learner", type=int, default=6)
    parser.add_argument("--shard_size", type=int, default=12000)
    parser.add_argument("--max_demos_per_bandit", type=int, default=2)
    parser.add_argument("--max_episodes_per_bandit", type=int, default=2)
    
    args = parser.parse_args()

    random.seed(42); torch.manual_seed(42)
    train_manifest = generate_dataset_streaming(os.path.join(args.out, "train"), num_runs=args.num_train_runs, split="train",
                                                K=args.K, num_bandits=args.num_bandits, T_demo=args.T_demo, T_learner=args.T_learner, N_max=args.N_max,
                                                max_context_bandits=args.max_context_bandits, max_num_demos=args.max_num_demos,
                                                max_demos_per_bandit=args.max_demos_per_bandit,
                                                max_episodes_per_bandit=args.max_episodes_per_bandit,
                                                shard_size=args.shard_size)

    val_manifest   = generate_dataset_streaming(os.path.join(args.out, "val"),   num_runs=args.num_valid_runs,  split="val",
                                                K=args.K, num_bandits=args.num_bandits, T_demo=args.T_demo, T_learner=args.T_learner, N_max=args.N_max,
                                                max_context_bandits=args.max_context_bandits, max_num_demos=args.max_num_demos,
                                                max_demos_per_bandit=args.max_demos_per_bandit,
                                                max_episodes_per_bandit=args.max_episodes_per_bandit,
                                                shard_size=args.shard_size)

    test_manifest  = generate_dataset_streaming(os.path.join(args.out, "test"),  num_runs=args.num_test_runs,  split="test",
                                                K=args.K, num_bandits=args.num_bandits, T_demo=args.T_demo, T_learner=args.T_learner, N_max=args.N_max,
                                                max_context_bandits=args.max_context_bandits, max_num_demos=args.max_num_demos,
                                                max_demos_per_bandit=args.max_demos_per_bandit,
                                                max_episodes_per_bandit=args.max_episodes_per_bandit,
                                                shard_size=args.shard_size)