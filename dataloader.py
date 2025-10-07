
from typing import Dict, List, Tuple
import random

import torch
from torch.utils.data import Dataset, DataLoader

SPECIAL = {
    "NONE": 0,
    "CTX_SEP": 1,
    "NOCTX": 2,
    "QUERY": 3,
    "BANDIT_SEP": 4,
}

AGENT_LEARNER_ID = 99  # in simulation

class BanditSequenceDataset(Dataset):
    """
    Takes a list of examples (from generate_simulations.py) and produces token tensors for the transformer.
    We concatenate context events + current events + [QUERY] token, then build a causal attention mask.
    """
    def __init__(self, examples: List[Dict], num_arms: int, max_bandits: int = 16, random_prefix: bool = True):
        self.examples = examples
        self.num_arms = num_arms
        self.max_bandits = max_bandits
        self.random_prefix = random_prefix

    def __len__(self):
        return len(self.examples)

    def _encode_events(self, events: List[Tuple[int,int,int,int]]) -> Dict[str, List[int]]:
        bandit_ids = []
        agent_ids = []
        action_ids = []
        reward_ids = []
        special_ids = []
        for (bandit_id, agent_id, action, reward) in events:
            if bandit_id == -1 and agent_id == -1:
                # NOCTX token
                bandit_ids.append(0)
                agent_ids.append(0)
                action_ids.append(self.num_arms)  # PAD action slot
                reward_ids.append(2)
                special_ids.append(SPECIAL["NOCTX"])
                continue
            if agent_id == -1:
                # BANDIT_SEP
                bandit_ids.append(min(max(bandit_id, 0), self.max_bandits-1))
                agent_ids.append(0)
                action_ids.append(self.num_arms)  # PAD
                reward_ids.append(2)
                special_ids.append(SPECIAL["BANDIT_SEP"])
                continue
            bandit_ids.append(min(max(bandit_id, 0), self.max_bandits-1))
            # Map agent ids: demonstrators start from 0.., learner maps to last index (2)
            agent_mapped = 2 if agent_id == AGENT_LEARNER_ID else min(agent_id, 1)  # support two demos + 1 learner
            agent_ids.append(agent_mapped)
            action_ids.append(action if action >= 0 else self.num_arms)
            reward_ids.append(reward if reward in (0,1) else 2)
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
        ctx = ex["context_events"]
        cur = ex["current_events"]

        # Optionally take only a random prefix of current events to mimic online filtering
        if self.random_prefix:
            L = random.randint(4, max(4, len(cur)))  # at least 4 tokens (including BANDIT_SEP)
            cur = cur[:L]

        toks_ctx = self._encode_events(ctx)
        toks_cur = self._encode_events(cur)

        # Append QUERY token at the end for readout
        for key in toks_ctx:
            toks_ctx[key] = toks_ctx[key]
        for key in toks_cur:
            toks_cur[key] = toks_cur[key]
        # Build combined
        bandit_ids = toks_ctx["bandit_ids"] + toks_cur["bandit_ids"] + [0]
        agent_ids  = toks_ctx["agent_ids"]  + toks_cur["agent_ids"]  + [0]
        action_ids = toks_ctx["action_ids"] + toks_cur["action_ids"] + [self.num_arms]
        reward_ids = toks_ctx["reward_ids"] + toks_cur["reward_ids"] + [2]
        special_ids= toks_ctx["special_ids"]+ toks_cur["special_ids"]+ [SPECIAL["QUERY"]]

        # Labels
        y_theta = torch.tensor(ex["labels"]["theta"], dtype=torch.float32)
        y_alpha_demo0 = torch.tensor(ex["labels"]["alpha_demo0"], dtype=torch.float32)
        y_beta_demo0  = torch.tensor(ex["labels"]["beta_demo0"], dtype=torch.float32)

        return {
            "bandit_ids": torch.tensor(bandit_ids, dtype=torch.long),
            "agent_ids":  torch.tensor(agent_ids,  dtype=torch.long),
            "action_ids": torch.tensor(action_ids, dtype=torch.long),
            "reward_ids": torch.tensor(reward_ids, dtype=torch.long),
            "special_ids":torch.tensor(special_ids,dtype=torch.long),
            "y_theta": y_theta,
            "y_alpha_demo0": y_alpha_demo0,
            "y_beta_demo0": y_beta_demo0,
        }


def collate_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    # Pad to max length
    maxT = max(item["bandit_ids"].shape[0] for item in batch)
    def pad1d(x, pad_value):
        T = x.shape[0]
        if T == maxT:
            return x
        out = torch.full((maxT,), pad_value, dtype=x.dtype)
        out[:T] = x
        return out

    bandit_ids = torch.stack([pad1d(x["bandit_ids"], 0) for x in batch], dim=0)
    agent_ids  = torch.stack([pad1d(x["agent_ids"], 0) for x in batch], dim=0)
    action_ids = torch.stack([pad1d(x["action_ids"], batch[0]["y_theta"].shape[0]) for x in batch], dim=0)
    reward_ids = torch.stack([pad1d(x["reward_ids"], 2) for x in batch], dim=0)
    special_ids= torch.stack([pad1d(x["special_ids"], 0) for x in batch], dim=0)

    # pad mask: True where PAD
    lengths = torch.tensor([x["bandit_ids"].shape[0] for x in batch])
    pad_mask = (torch.arange(maxT)[None, :].expand(len(batch), maxT) >= lengths[:, None])

    y_theta = torch.stack([x["y_theta"] for x in batch], dim=0)
    y_alpha_demo0 = torch.stack([x["y_alpha_demo0"] for x in batch], dim=0)
    y_beta_demo0  = torch.stack([x["y_beta_demo0"]  for x in batch], dim=0)

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


def make_dataloader(path: str, batch_size: int, num_arms: int, shuffle: bool = True) -> DataLoader:
    examples = torch.load(path)
    ds = BanditSequenceDataset(examples, num_arms=num_arms, random_prefix=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch)
