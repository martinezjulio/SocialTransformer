from typing import Dict, List, Tuple, Sequence, Optional
import torch
from torch.utils.data import Dataset, DataLoader

# Special token channel for structure
SPECIAL = {
    "NONE": 0,        # real events
    "CTX_SEP": 1,     # (unused here—kept for compatibility)
    "NOCTX": 2,       # (-1, -1, -1, -1)
    "QUERY": 3,       # appended at the end for readout
    "BANDIT_SEP": 4,  # (b, -1, -1, -1)
}

LEARNER_RAW_ID = 99  # learner's agent_id in the simulator


# ---------- helpers ----------

def _collect_local_demo_ids(events: List[Tuple[int,int,int,int]]) -> List[int]:
    """Return sorted unique demonstrator raw ids present in a stream."""
    demos = sorted({ag for (_, ag, _, _) in events if ag not in (LEARNER_RAW_ID, -1)})
    return demos


def _remap_agent_ids_local(
    events: List[Tuple[int,int,int,int]],
    num_agents_max: int = 4
) -> Tuple[List[Tuple[int,int,int,int]], int]:
    """
    Remap raw agent ids to a compact local space for this example:
      - Demonstrators present in the example map to 0..D-1 (D <= num_agents_max-1)
      - Learner (raw=99) maps to D (last index used)
      - BANDIT_SEP/NOCTX rows get agent_id = 0 (they're structural)
    Returns:
      remapped_events, num_agents_used (<= num_agents_max)
    """
    demos = _collect_local_demo_ids(events)
    D = len(demos)

    if D + 1 > num_agents_max:
        raise ValueError(
            f"Example has {D} demonstrators + learner = {D+1} agents, "
            f"but num_agents_max={num_agents_max}. Increase num_agents_max and the model's num_agents."
        )

    demo_map = {raw: i for i, raw in enumerate(demos)}
    learner_index = D  # learner goes after demos

    out: List[Tuple[int,int,int,int]] = []
    for (b, ag, a, r) in events:
        if ag == -1:
            # BANDIT_SEP row—use agent_id=0 (arbitrary; it's a structural token)
            out.append((b, 0, a, r))
        elif ag == LEARNER_RAW_ID:
            out.append((b, learner_index, a, r))
        else:
            out.append((b, demo_map[ag], a, r))
    return out, (D + 1)  # demos + learner


# ---------- Dataset ----------

class BanditSequenceDataset(Dataset):
    """
    Consumes a list-like of training examples produced by build_training_examples_from_run(...):
      {
        "context_events": [ (b, -1,-1,-1) | (b, agent, action, reward) ... ] or [(-1,-1,-1,-1)],
        "current_events": [ (b, -1,-1,-1), (b, agent, action, reward) ... ],
        "labels": {
           "theta": [...], "alpha_demo0": [...], "beta_demo0": [...],
           "demonstrator_id": int, "K": int, "N_tot": int, "N_by_arm": [...]
        }
      }

    It builds one token sequence per example:
      [context blocks] + [current block] + [QUERY]
    with five channels: bandit_id, agent_id, action_id, reward_id, special_id.

    Notes:
      - agent_id is remapped locally per example to a compact [0..D] space, with learner at index D.
      - set num_agents_max to cover your maximum demonstrators per run (+1 for learner); default 4.
    """
    def __init__(
        self,
        examples: Sequence[Dict],
        num_arms: int,
        max_bandits: int = 64,
        num_agents_max: int = 4,
        random_prefix: bool = False,  # keep False to use full streams by default
    ):
        self.examples = examples
        self.num_arms = num_arms
        self.max_bandits = max_bandits
        self.num_agents_max = num_agents_max
        self.random_prefix = random_prefix  # if True, will clip current_events to a random prefix (>=1 BANDIT_SEP + >=1 event)

    def __len__(self):
        return len(self.examples)

    def _encode_stream(
        self,
        stream: List[Tuple[int,int,int,int]],
        remap: bool = True,
        local_map: Optional[Dict[int,int]] = None,
    ) -> Dict[str, List[int]]:
        """
        Encode a stream of events into token channels.
        Assumes agent ids already remapped if remap=False.
        """
        bandit_ids, agent_ids, action_ids, reward_ids, special_ids = [], [], [], [], []
        for (b, ag, a, r) in stream:
            if (b, ag, a, r) == (-1, -1, -1, -1):
                # NOCTX
                bandit_ids.append(0)
                agent_ids.append(0)
                action_ids.append(self.num_arms)  # PAD
                reward_ids.append(2)              # PAD reward
                special_ids.append(SPECIAL["NOCTX"])
                continue

            if ag == -1:
                # BANDIT_SEP
                bandit_ids.append(min(max(b, 0), self.max_bandits - 1))
                agent_ids.append(0)               # structural row; any index ok
                action_ids.append(self.num_arms)  # PAD
                reward_ids.append(2)              # PAD
                special_ids.append(SPECIAL["BANDIT_SEP"])
                continue

            # Real event
            bandit_ids.append(min(max(b, 0), self.max_bandits - 1))
            agent_ids.append(ag)
            action_ids.append(a if (0 <= a < self.num_arms) else self.num_arms)
            reward_ids.append(r if r in (0, 1) else 2)
            special_ids.append(SPECIAL["NONE"])

        return {
            "bandit_ids": bandit_ids,
            "agent_ids": agent_ids,
            "action_ids": action_ids,
            "reward_ids": reward_ids,
            "special_ids": special_ids,
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]

        # Build the full list for mapping purposes (structural tokens can be present)
        # We'll concatenate context and current, then remap agent ids once per example.
        combined = list(ex["context_events"]) + list(ex["current_events"])

        # Construct a "real event only" view to collect unique demo ids (skip NOCTX & BANDIT_SEP)
        real_events = [(b, ag, a, r) for (b, ag, a, r) in combined if not (ag == -1 or (b, ag, a, r) == (-1, -1, -1, -1))]

        # Remap agents locally (demos -> 0..D-1, learner -> D)
        remapped_combined, num_agents_used = _remap_agent_ids_local(combined, num_agents_max=self.num_agents_max)

        # Optionally shorten current_events to a random prefix (preserving the first BANDIT_SEP and at least one real event)
        if self.random_prefix:
            cur = [*ex["current_events"]]
            if len(cur) > 2:  # at least [BANDIT_SEP, one real event]
                L = torch.randint(low=2, high=len(cur)+1, size=(1,)).item()
                ex_cur = cur[:L]
            else:
                ex_cur = cur
            remapped_combined = list(ex["context_events"]) + ex_cur
            remapped_combined, _ = _remap_agent_ids_local(remapped_combined, num_agents_max=self.num_agents_max)

        # Encode streams separately (preserve structure: context then current)
        toks_ctx  = self._encode_stream(ex["context_events"])
        toks_curr = self._encode_stream(ex["current_events"])

        # But we need the remapped agent ids; rebuild agent_ids with the remapped sequence
        # Re-encode with remapped combined then split lengths back into ctx/curr
        enc = self._encode_stream(remapped_combined)
        len_ctx = len(ex["context_events"])
        # splice back in remapped agent_ids only (other channels are already correct from toks_ctx/toks_curr)
        agent_ids_ctx  = enc["agent_ids"][:len_ctx]
        agent_ids_curr = enc["agent_ids"][len_ctx:]

        # Append QUERY token at the very end for readout
        bandit_ids = toks_ctx["bandit_ids"] + toks_curr["bandit_ids"] + [0]
        agent_ids  = agent_ids_ctx         + agent_ids_curr         + [0]
        action_ids = toks_ctx["action_ids"]+ toks_curr["action_ids"]+ [self.num_arms]
        reward_ids = toks_ctx["reward_ids"]+ toks_curr["reward_ids"]+ [2]
        special_ids= toks_ctx["special_ids"]+toks_curr["special_ids"]+[SPECIAL["QUERY"]]

        # Labels
        y_theta        = torch.tensor(ex["labels"]["theta"],        dtype=torch.float32)
        y_alpha_demo0  = torch.tensor(ex["labels"]["alpha_demo0"],  dtype=torch.float32)
        y_beta_demo0   = torch.tensor(ex["labels"]["beta_demo0"],   dtype=torch.float32)

        return {
            "bandit_ids": torch.tensor(bandit_ids, dtype=torch.long),
            "agent_ids":  torch.tensor(agent_ids,  dtype=torch.long),
            "action_ids": torch.tensor(action_ids, dtype=torch.long),
            "reward_ids": torch.tensor(reward_ids, dtype=torch.long),
            "special_ids":torch.tensor(special_ids,dtype=torch.long),
            "y_theta":     y_theta,
            "y_alpha_demo0": y_alpha_demo0,
            "y_beta_demo0":  y_beta_demo0,
        }


# ---------- collate ----------

def collate_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    maxT = max(x["bandit_ids"].shape[0] for x in batch)

    def pad1(x: torch.Tensor, pad_val: int) -> torch.Tensor:
        T = x.shape[0]
        if T == maxT:
            return x
        out = torch.full((maxT,), pad_val, dtype=x.dtype)
        out[:T] = x
        return out

    bandit_ids = torch.stack([pad1(x["bandit_ids"], 0) for x in batch], dim=0)
    agent_ids  = torch.stack([pad1(x["agent_ids"],  0) for x in batch], dim=0)
    # PAD action = num_arms (we infer from the first item)
    num_arms = batch[0]["y_theta"].shape[0]
    action_ids = torch.stack([pad1(x["action_ids"], num_arms) for x in batch], dim=0)
    reward_ids = torch.stack([pad1(x["reward_ids"], 2) for x in batch], dim=0)
    special_ids= torch.stack([pad1(x["special_ids"], 0) for x in batch], dim=0)

    lengths = torch.tensor([x["bandit_ids"].shape[0] for x in batch])
    pad_mask = (torch.arange(maxT)[None, :].expand(len(batch), maxT) >= lengths[:, None])

    y_theta        = torch.stack([x["y_theta"]       for x in batch], dim=0)
    y_alpha_demo0  = torch.stack([x["y_alpha_demo0"] for x in batch], dim=0)
    y_beta_demo0   = torch.stack([x["y_beta_demo0"]  for x in batch], dim=0)

    return {
        "bandit_ids": bandit_ids,
        "agent_ids": agent_ids,
        "action_ids": action_ids,
        "reward_ids": reward_ids,
        "special_ids": special_ids,
        "pad_mask": pad_mask,
        "y_theta": y_theta,
        "y_alpha_demo0": y_alpha_demo0,
        "y_beta_demo0": y_beta_demo0,
    }


# ---------- convenience loader ----------

def make_dataloader(
    examples_or_path,
    batch_size: int,
    num_arms: int,
    *,
    num_agents_max: int = 4,
    max_bandits: int = 64,
    shuffle: bool = True,
) -> DataLoader:
    """
    If 'examples_or_path' is a path to a .pt list, loads it; otherwise assumes it is already a list-like of examples.
    """
    if isinstance(examples_or_path, str):
        examples = torch.load(examples_or_path)
    else:
        examples = examples_or_path

    ds = BanditSequenceDataset(
        examples, num_arms=num_arms, max_bandits=max_bandits, num_agents_max=num_agents_max, random_prefix=False
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch)


if __name__ == "__main__":
    import argparse, os, json, torch
    from torch.utils.data import DataLoader
    from collections import OrderedDict
    import bisect

    parser = argparse.ArgumentParser(description="Quick demo for bandit dataloader.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--pt", type=str, help="Path to a .pt file containing a list of examples")
    src.add_argument("--manifest", type=str, help="Path to a streaming manifest.json")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-arms", type=int, default=12, help="K used for PAD index (should match dataset)")
    parser.add_argument("--num-agents-max", type=int, default=4, help="max demonstrators per run + learner")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle batches")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--preview-batches", type=int, default=1)
    parser.add_argument("--alpha0", type=float, default=1.0)  # for label sanity check
    parser.add_argument("--beta0", type=float, default=1.0)
    args = parser.parse_args()

    # --- optional: a tiny loader for streaming shards (manifest) ---
    class ShardedExamplesDataset(torch.utils.data.Dataset):
        """
        Lazily reads shards listed in a manifest.json produced by generate_dataset_streaming.
        Keeps a small LRU cache of loaded shards.
        """
        def __init__(self, manifest_path: str, cache_shards: int = 2):
            with open(manifest_path, "r") as f:
                man = json.load(f)
            self.root = os.path.dirname(manifest_path)
            self.shards = man["shards"]
            self.total = sum(s["num_examples"] for s in self.shards)
            # prefix sums for fast index -> (shard, local)
            self.prefix = []
            c = 0
            for s in self.shards:
                self.prefix.append(c)
                c += s["num_examples"]
            self._cache = OrderedDict()
            self._cap = max(1, cache_shards)

        def __len__(self):
            return self.total

        def _load_shard(self, i: int):
            if i in self._cache:
                self._cache.move_to_end(i)
                return self._cache[i]
            path = os.path.join(self.root, self.shards[i]["path"])
            data = torch.load(path, map_location="cpu")
            self._cache[i] = data
            self._cache.move_to_end(i)
            if len(self._cache) > self._cap:
                self._cache.popitem(last=False)
            return data

        def __getitem__(self, idx: int):
            # find shard via prefix sums
            i = bisect.bisect_right(self.prefix, idx) - 1
            start = self.prefix[i]
            local_idx = idx - start
            shard = self._load_shard(i)
            return shard[local_idx]

    # --- build DataLoader ---
    if args.pt:
        # .pt path: load all examples into RAM (ok for small/medium)
        print(f"Loading examples from {args.pt} ...")
        loader = make_dataloader(
            args.pt,
            batch_size=args.batch_size,
            num_arms=args.num_arms,
            num_agents_max=args.num_agents_max,
            shuffle=args.shuffle,
        )
    else:
        # manifest: lazy shard loading
        print(f"Loading shard manifest from {args.manifest} ...")
        ds = ShardedExamplesDataset(args.manifest, cache_shards=2)
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            num_workers=args.num_workers,
            collate_fn=collate_batch,
            pin_memory=True,
        )

    # --- iterate a few batches and pretty print ---
    def _summarize_batch(b):
        B, T = b["bandit_ids"].shape
        print(f"\nBatch: B={B}, T={T}")
        for k in ("bandit_ids","agent_ids","action_ids","reward_ids","special_ids","pad_mask"):
            print(f"  {k:>12}: {tuple(b[k].shape)}  dtype={b[k].dtype}")

        # preview the first sequence (unpadded length)
        lengths = (~b["pad_mask"]).sum(dim=1).tolist()
        i0 = 0
        L0 = lengths[i0]
        print(f"\nFirst sequence length (no PAD): {L0}")
        preview = min(12, L0)
        cols = ["bandit_ids","agent_ids","action_ids","reward_ids","special_ids"]
        rows = []
        for t in range(preview):
            rows.append([int(b[c][i0, t].item()) for c in cols])
        print("  t  bandit  agent  action  reward  special")
        for t, row in enumerate(rows):
            print(f"{t:>3}  " + "   ".join(f"{v:>5}" for v in row))

        # label sanity for first item
        K = b["y_theta"].shape[1]
        alpha0, beta0 = args.alpha0, args.beta0
        a0 = b["y_alpha_demo0"][i0].tolist()
        b0 = b["y_beta_demo0"][i0].tolist()
        # Check: (alpha_demo0 - alpha0) + (beta_demo0 - beta0) should be N_by_arm if present.
        # We can't access N_by_arm from the collated batch; just print sums as a quick check.
        total_pseudo = sum((a - alpha0) + (bb - beta0) for a, bb in zip(a0, b0))
        print(f"\nLabel check (item 0): sum((alpha_demo0-α0)+(beta_demo0-β0)) = {total_pseudo:.1f}")

        # show theta stats
        theta0 = b["y_theta"][i0].tolist()
        best = max(theta0); sec = sorted(theta0, reverse=True)[1]
        print(f"  theta[0] best={best:.3f}  gap={best-sec:.3f}")

    n = 0
    for batch in loader:
        _summarize_batch(batch)
        n += 1
        if n >= args.preview_batches:
            break

    print("\nDone. Try adjusting --batch-size, --shuffle, or point to --manifest to stream shards.")
