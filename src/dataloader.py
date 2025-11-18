from typing import Any, Dict, List, Tuple, Sequence, Optional
import os
import json
import bisect
from collections import OrderedDict
import torch
from torch.utils.data import Dataset, DataLoader

# Special token channel for structure
SPECIAL = {
    "NONE": 0,        # real events
    "QUERY": 1,       # appended at the end for readout
    "BANDIT_SEP": 2,  # (b, -1, -1, -1)
}

LEARNER_RAW_ID = 99  # learner's agent_id in the simulator


# ---------- helpers ----------

def _collect_local_demo_ids(events: List[Tuple[int,int,int,int]]) -> List[int]:
    """Return sorted unique demonstrator raw ids present in a stream."""
    demos = sorted({ag for (_, ag, _, _) in events if ag not in (LEARNER_RAW_ID, -1)})
    return demos


def _remap_agent_ids_local(
    events: List[Tuple[int,int,int,int]],
    num_agents: int = 4
) -> Tuple[List[Tuple[int,int,int,int]], int]:
    """
    Remap raw agent ids to a compact local space for this example:
      - Learner (raw=99) maps to 0 (primary position)
      - Demonstrators present in the example map to 1..D (D <= num_agents-1)
      - BANDIT_SEP rows get agent_id = 0 (share with learner; distinguished by special_ids)
    Returns:
      remapped_events, num_agents_used (<= num_agents)
    """
    demos = _collect_local_demo_ids(events)
    D = len(demos)

    if D + 1 > num_agents:
        raise ValueError(
            f"Example has {D} demonstrators + learner = {D+1} agents, "
            f"but num_agents={num_agents}. Increase num_agents in the model."
        )

    demo_map = {raw: i + 1 for i, raw in enumerate(demos)}  # demos start at 1
    learner_index = 0  # learner is always at 0

    out: List[Tuple[int,int,int,int]] = []
    for (b, ag, a, r) in events:
        if ag == -1:
            # BANDIT_SEP row—use agent_id=0 (shares with learner; structural token)
            out.append((b, 0, a, r))
        elif ag == LEARNER_RAW_ID:
            out.append((b, learner_index, a, r))
        else:
            out.append((b, demo_map[ag], a, r))
    return out, (D + 1)  # demos + learner


# ---------- Dataset ----------

class BanditSequenceDataset(Dataset):
    """
    Consumes training examples from the round-based generation:
      {
        "events": [(b, agent, action, reward), ...],
        "round_boundaries": [0, 25, 50, 75],
        "round_bandit_map": [0, 1, 2],
        "labels": {
           "bandits": [bandit_info, ...],
           "K": int,
           "num_demonstrators": int,
        },
        "meta": {...}
      }

    Builds one token sequence per example with five channels:
      bandit_id, agent_id, action_id, reward_id, special_id
    Plus a QUERY token at the end.

    Notes:
      - agent_id is remapped locally per example to a compact [0..D] space, with learner at index 0.
      - num_agents includes the learner + all demonstrators (e.g., 3 = 1 learner + 2 demos)
    """
    def __init__(
        self,
        examples: Sequence[Dict],
        num_arms: int,
        num_bandits: int = 64,
        num_agents: int = 4,
    ):
        self.examples = examples
        self.num_arms = num_arms
        self.num_bandits = num_bandits
        self.num_agents = num_agents

    def __len__(self):
        return len(self.examples)

    def _encode_stream(
        self,
        stream: List[Tuple[int,int,int,int]],
    ) -> Dict[str, List[int]]:
        """
        Encode a stream of events into token channels.
        """
        bandit_ids, agent_ids, action_ids, reward_ids, special_ids = [], [], [], [], []
        for (b, ag, a, r) in stream:
            if ag == -1:
                # BANDIT_SEP
                bandit_ids.append(min(max(b, 0), self.num_bandits - 1))
                agent_ids.append(0)               # structural row; any index ok
                action_ids.append(self.num_arms)  # PAD
                reward_ids.append(2)              # PAD
                special_ids.append(SPECIAL["BANDIT_SEP"])
                continue

            # Real event
            bandit_ids.append(min(max(b, 0), self.num_bandits - 1))
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

        # Remap agent ids locally
        events = ex["events"]
        remapped_events, num_agents_used = _remap_agent_ids_local(events, num_agents=self.num_agents)

        # Encode the remapped events
        enc = self._encode_stream(remapped_events)

        # Append QUERY token at the very end for readout
        bandit_ids = enc["bandit_ids"] + [0]
        agent_ids = enc["agent_ids"] + [0]
        action_ids = enc["action_ids"] + [self.num_arms]
        reward_ids = enc["reward_ids"] + [2]
        special_ids = enc["special_ids"] + [SPECIAL["QUERY"]]

        # Extract labels for all demonstrators and all bandits
        labels = ex["labels"]
        K = labels["K"]
        num_demonstrators = labels["num_demonstrators"]
        bandits = labels["bandits"]
        
        # Build tensors for pre-episode beliefs for each demonstrator and each bandit
        # Shape: [num_bandits, num_demonstrators, K]
        num_bandits = len(bandits)
        y_alpha_demo = torch.zeros((num_bandits, num_demonstrators, K), dtype=torch.float32)
        y_beta_demo = torch.zeros((num_bandits, num_demonstrators, K), dtype=torch.float32)
        y_thetas = torch.zeros((num_bandits, K), dtype=torch.float32)
        
        for b_idx, bandit_info in enumerate(bandits):
            y_thetas[b_idx] = torch.tensor(bandit_info["theta"], dtype=torch.float32)
            for demo_id in range(num_demonstrators):
                demo_beliefs = bandit_info["demonstrators"][demo_id]
                y_alpha_demo[b_idx, demo_id] = torch.tensor(demo_beliefs["alpha0"], dtype=torch.float32)
                y_beta_demo[b_idx, demo_id] = torch.tensor(demo_beliefs["beta0"], dtype=torch.float32)

        return {
            "bandit_ids": torch.tensor(bandit_ids, dtype=torch.long),
            "agent_ids": torch.tensor(agent_ids, dtype=torch.long),
            "action_ids": torch.tensor(action_ids, dtype=torch.long),
            "reward_ids": torch.tensor(reward_ids, dtype=torch.long),
            "special_ids": torch.tensor(special_ids, dtype=torch.long),
            "round_boundaries": torch.tensor(ex["round_boundaries"], dtype=torch.long),
            "round_bandit_map": torch.tensor(ex["round_bandit_map"], dtype=torch.long),
            "segment_boundaries": torch.tensor(ex["segment_boundaries"], dtype=torch.long),
            "segment_info": ex["segment_info"],  # Keep as list of dicts
            "y_thetas": y_thetas,
            "y_alpha_demo": y_alpha_demo,
            "y_beta_demo": y_beta_demo,
            "num_demonstrators": num_demonstrators,
            "num_bandits": num_bandits,
        }


# ---------- collate ----------

def collate_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    maxT = max(x["bandit_ids"].shape[0] for x in batch)
    num_arms = batch[0]["y_thetas"].shape[1]  # K from first example
    max_num_bandits = max(x["num_bandits"] for x in batch)
    max_num_demonstrators = max(x["num_demonstrators"] for x in batch)
    max_rounds = max(len(x["round_boundaries"]) for x in batch)

    def pad1(x: torch.Tensor, pad_val: int) -> torch.Tensor:
        T = x.shape[0]
        if T == maxT:
            return x
        out = torch.full((maxT,), pad_val, dtype=x.dtype)
        out[:T] = x
        return out

    bandit_ids = torch.stack([pad1(x["bandit_ids"], 0) for x in batch], dim=0)
    agent_ids = torch.stack([pad1(x["agent_ids"], 0) for x in batch], dim=0)
    action_ids = torch.stack([pad1(x["action_ids"], num_arms) for x in batch], dim=0)
    reward_ids = torch.stack([pad1(x["reward_ids"], 2) for x in batch], dim=0)
    special_ids = torch.stack([pad1(x["special_ids"], 0) for x in batch], dim=0)

    lengths = torch.tensor([x["bandit_ids"].shape[0] for x in batch])
    pad_mask = (torch.arange(maxT)[None, :].expand(len(batch), maxT) >= lengths[:, None])

    # Pad round_boundaries and round_bandit_map
    round_boundaries = torch.zeros((len(batch), max_rounds), dtype=torch.long)
    round_bandit_map = torch.zeros((len(batch), max_rounds - 1), dtype=torch.long)  # -1 because boundaries has one extra
    
    for i, x in enumerate(batch):
        rb_len = len(x["round_boundaries"])
        round_boundaries[i, :rb_len] = x["round_boundaries"]
        rbm_len = len(x["round_bandit_map"])
        round_bandit_map[i, :rbm_len] = x["round_bandit_map"]
    
    # Pad segment_boundaries and collect segment_info
    max_segments = max(len(x["segment_boundaries"]) for x in batch)
    segment_boundaries = torch.zeros((len(batch), max_segments), dtype=torch.long)
    segment_info = []  # List of lists, one per batch item
    
    for i, x in enumerate(batch):
        sb_len = len(x["segment_boundaries"])
        segment_boundaries[i, :sb_len] = x["segment_boundaries"]
        segment_info.append(x["segment_info"])

    # Pad y_thetas, y_alpha_demo, y_beta_demo
    y_thetas = torch.zeros((len(batch), max_num_bandits, num_arms), dtype=torch.float32)
    y_alpha_demo = torch.zeros((len(batch), max_num_bandits, max_num_demonstrators, num_arms), dtype=torch.float32)
    y_beta_demo = torch.zeros((len(batch), max_num_bandits, max_num_demonstrators, num_arms), dtype=torch.float32)
    
    for i, x in enumerate(batch):
        nb = x["num_bandits"]
        nd = x["num_demonstrators"]
        y_thetas[i, :nb] = x["y_thetas"]
        y_alpha_demo[i, :nb, :nd] = x["y_alpha_demo"]
        y_beta_demo[i, :nb, :nd] = x["y_beta_demo"]

    return {
        "bandit_ids": bandit_ids,
        "agent_ids": agent_ids,
        "action_ids": action_ids,
        "reward_ids": reward_ids,
        "special_ids": special_ids,
        "pad_mask": pad_mask,
        "round_boundaries": round_boundaries,
        "round_bandit_map": round_bandit_map,
        "segment_boundaries": segment_boundaries,
        "segment_info": segment_info,
        "y_thetas": y_thetas,
        "y_alpha_demo": y_alpha_demo,
        "y_beta_demo": y_beta_demo,
    }


# ---------- convenience loader ----------

def make_dataloader(
    examples_or_path,
    batch_size: int,
    num_arms: int,
    *,
    num_agents: int = 4,
    num_bandits: int = 64,
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
        examples, num_arms=num_arms, num_bandits=num_bandits, num_agents=num_agents
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch)


# ---------- sharded dataset ----------

class ShardedExamplesDataset(Dataset):
    """
    Lazily reads shards listed in a manifest.json produced by generate_dataset_streaming.
    Keeps a small LRU cache of loaded shards.
    Exposes dataset metadata for automatic configuration.
    """
    def __init__(self, manifest_path: str, cache_shards: int = 2):
        with open(manifest_path, "r") as f:
            man = json.load(f)
        self.root = os.path.dirname(manifest_path)
        self.shards = man["shards"]
        self.meta = man.get("meta", {})  # Expose metadata
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quick demo for bandit dataloader.")
    parser.add_argument("--manifest", type=str, default="data/demonstrations_data_01/train/manifest.json", help="Path to a streaming manifest.json")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--shuffle", action="store_true", help="Shuffle batches")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--preview-batches", type=int, default=5)
    parser.add_argument("--alpha0", type=float, default=1.0)  # for label sanity check
    parser.add_argument("--beta0", type=float, default=1.0)
    args = parser.parse_args()

    # --- build DataLoader ---
    print(f"Loading shard manifest from {args.manifest} ...")
    raw_ds = ShardedExamplesDataset(args.manifest, cache_shards=2)
    
    # Automatically infer parameters from manifest
    num_arms = raw_ds.meta.get("K", 12)
    num_demonstrators = raw_ds.meta.get("num_demonstrators", 2)
    num_agents = num_demonstrators + 1  # +1 for learner
    num_bandits = raw_ds.meta.get("num_rounds", 3) if raw_ds.meta.get("change_bandits_with_rounds", True) else 1
    
    print(f"Auto-detected from manifest: num_arms={num_arms}, num_agents={num_agents}, num_bandits={num_bandits}")
    
    ds = BanditSequenceDataset(
        raw_ds,
        num_arms=num_arms,
        num_agents=num_agents,
        num_bandits=num_bandits,
    )
    print(f"Dataset size: {len(ds)} examples")
    
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=True,
    )

    n = 0
    for batch in loader:

        # print the batch shape and keys of the batch
        B, T = batch["bandit_ids"].shape
        print(f"\nBatch: batch_size={B}, sequence_length={T}")
        print("----------------------------------------------------")
        print("----------------------------------------------------")
        for k in batch.keys():
            if k == "segment_info":
                print(f"  {k:>18}: list of {len(batch[k])} items")
            else:
                print(f"  {k:>18}: {tuple(batch[k].shape)}  dtype={batch[k].dtype}")

        lengths = (~batch["pad_mask"]).sum(dim=1).tolist()
        i0 = 0
        L0 = lengths[i0]
        print(f"\nFirst example of batch sequence length (no PAD): {L0}")
        
        preview = min(12, L0)
        cols = ["bandit_ids","agent_ids","action_ids","reward_ids","special_ids"]
        rows = []
        for t in range(preview):
            rows.append([int(batch[c][i0, t].item()) for c in cols])
        print("  t  bandit  agent  action  reward  special")
        for t, row in enumerate(rows):
            print(f"{t:>3}  " + "   ".join(f"{v:>5}" for v in row))

        # Show round boundaries
        print(f"\nRound boundaries for first example: {batch['round_boundaries'][i0].tolist()}")
        print(f"Round bandit map: {batch['round_bandit_map'][i0].tolist()}")

        # Show demonstrator beliefs for first bandit
        K = num_arms
        alpha0, beta0 = args.alpha0, args.beta0
        print(f"\nDemonstrator Pre-episode beliefs (first bandit):")
        num_demos = (batch["y_alpha_demo"][i0, 0].sum(dim=1) > 0).sum().item()
        for demo_id in range(num_demos):
            a0 = batch["y_alpha_demo"][i0, 0, demo_id].tolist()
            b0 = batch["y_beta_demo"][i0, 0, demo_id].tolist()
            S0 = [a - alpha0 for a in a0]
            F0 = [b - beta0 for b in b0]
            N_tot = sum(S0) + sum(F0)
            print(f"  Demo {demo_id}: N_tot={N_tot:.0f}, S0={S0[:3]}..., F0={F0[:3]}...")

        # Payoffs Ground Truth
        theta = batch["y_thetas"][i0, 0].tolist()
        theta_rounded = [round(t, 2) for t in theta]
        print(f"\nFirst Bandit Thetas: {theta_rounded}\n")

        n += 1
        if n >= args.preview_batches:
            break

    print("\nDone. Try adjusting --batch-size or --shuffle.")
