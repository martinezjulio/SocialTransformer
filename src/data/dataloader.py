"""
Runtime data loading for SocialTransformer.

The on-disk format is a HuggingFace Arrow dataset (one per split) produced by
`generate_simulations.py` via `Dataset.from_generator(...).save_to_disk(...)`.
`load_from_disk` returns an mmap-backed view, so random index access is paid
by the OS page cache — sample-level shuffling under DDP is cheap.

This module is runtime-only. Build-time encoding (5 token channels, label
tensor construction, segment_info flattening) lives in `src/data/encoding.py`
and runs once when the dataset is generated.
"""

import os
from typing import Dict, List, Optional, Tuple

import torch
from datasets import Dataset, load_from_disk
from torch.utils.data import DataLoader

from data.encoding import (  # re-exports
    ID_TO_SEGMENT_TYPE,
    LEARNER_RAW_ID,
    SEGMENT_TYPE_TO_ID,
    SPECIAL,
)

__all__ = [
    "SPECIAL",
    "LEARNER_RAW_ID",
    "SEGMENT_TYPE_TO_ID",
    "ID_TO_SEGMENT_TYPE",
    "collate_batch",
    "load_arrow_split",
    "make_arrow_loader",
    "make_arrow_dataloader",
]


# ---------- collate ----------

# Variable-length integer columns that need padding to per-batch max length.
_VARLEN_TOKEN_COLS = ("bandit_ids", "agent_ids", "action_ids", "reward_ids", "special_ids")
_VARLEN_INDEX_COLS = ("round_boundaries", "round_bandit_map", "segment_boundaries")


def _as_long_tensor(x) -> torch.Tensor:
    """Convert a list / tuple / numpy array / tensor to a 1-D int64 tensor."""
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.long)


def collate_batch(batch: List[Dict], normalizer=None) -> Dict[str, torch.Tensor]:
    """Pad a list of Arrow rows into a batch dict.

    Conventions inherited from the pre-Arrow loader:
    - PAD_VAL for `action_ids` is `num_arms` (out-of-range action id).
    - PAD_VAL for `reward_ids` is 2 (out-of-range reward id).
    - PAD_VAL for the rest is 0.
    - `pad_mask[b, t] == True` for padded positions.
    - `round_bandit_map` is shaped `(B, max_rounds - 1)` because
      `round_boundaries` carries one extra entry (positions = rounds + 1).
    - `segment_round` / `segment_type_id` are kept as Python lists per example
      (not padded) — consumers iterate them by index per example and never feed
      them to the model as a batched tensor.
    """
    B = len(batch)

    # 1) Tensorize variable-length integer columns (Arrow returns them as
    #    Python lists when not covered by set_format("torch")).
    for item in batch:
        for k in _VARLEN_TOKEN_COLS + _VARLEN_INDEX_COLS:
            item[k] = _as_long_tensor(item[k])

    # 2) Token sequence padding
    maxT = max(x["bandit_ids"].shape[0] for x in batch)
    num_arms = batch[0]["y_thetas"].shape[1]  # K — constant per dataset

    def _pad1(x: torch.Tensor, pad_val: int) -> torch.Tensor:
        T = x.shape[0]
        if T == maxT:
            return x
        out = torch.full((maxT,), pad_val, dtype=x.dtype)
        out[:T] = x
        return out

    pad_vals = {
        "bandit_ids":  0,
        "agent_ids":   0,
        "action_ids":  num_arms,
        "reward_ids":  2,
        "special_ids": 0,
    }
    tokens = {
        k: torch.stack([_pad1(x[k], pad_vals[k]) for x in batch], dim=0)
        for k in _VARLEN_TOKEN_COLS
    }

    lengths = torch.tensor([x["bandit_ids"].shape[0] for x in batch])
    pad_mask = (torch.arange(maxT)[None, :].expand(B, maxT) >= lengths[:, None])

    # 3) Round + segment index padding
    max_rounds = max(x["round_boundaries"].shape[0] for x in batch)
    round_boundaries = torch.zeros((B, max_rounds), dtype=torch.long)
    round_bandit_map = torch.zeros((B, max_rounds - 1), dtype=torch.long)
    for i, x in enumerate(batch):
        rb_len  = x["round_boundaries"].shape[0]
        rbm_len = x["round_bandit_map"].shape[0]
        round_boundaries[i, :rb_len] = x["round_boundaries"]
        round_bandit_map[i, :rbm_len] = x["round_bandit_map"]

    max_segments = max(x["segment_boundaries"].shape[0] for x in batch)
    segment_boundaries = torch.zeros((B, max_segments), dtype=torch.long)
    segment_round: List[List[int]] = []
    segment_type_id: List[List[int]] = []
    segment_agent: List[List[int]] = []
    for i, x in enumerate(batch):
        sb_len = x["segment_boundaries"].shape[0]
        segment_boundaries[i, :sb_len] = x["segment_boundaries"]
        for src_key, dst in (
            ("segment_round",   segment_round),
            ("segment_type_id", segment_type_id),
            ("segment_agent",   segment_agent),
        ):
            v = x[src_key]
            dst.append(v.tolist() if isinstance(v, torch.Tensor) else list(v))

    # 4) Fixed-shape labels — already (num_bandits, ...) and consistent across the dataset
    y_thetas     = torch.stack([torch.as_tensor(x["y_thetas"],     dtype=torch.float32) for x in batch], dim=0)
    y_alpha_demo = torch.stack([torch.as_tensor(x["y_alpha_demo"], dtype=torch.float32) for x in batch], dim=0)
    y_beta_demo  = torch.stack([torch.as_tensor(x["y_beta_demo"],  dtype=torch.float32) for x in batch], dim=0)

    if normalizer is not None:
        y_alpha_demo = normalizer.normalize_alpha(y_alpha_demo)
        y_beta_demo  = normalizer.normalize_beta(y_beta_demo)

    return {
        **tokens,
        "pad_mask":           pad_mask,
        "round_boundaries":   round_boundaries,
        "round_bandit_map":   round_bandit_map,
        "segment_boundaries": segment_boundaries,
        "segment_round":      segment_round,
        "segment_type_id":    segment_type_id,
        "segment_agent":      segment_agent,
        "y_thetas":           y_thetas,
        "y_alpha_demo":       y_alpha_demo,
        "y_beta_demo":        y_beta_demo,
    }


# ---------- dataset / loader ----------

def load_arrow_split(data_dir: str, split: str) -> Dataset:
    """Load one split as an mmap-backed HF Dataset with the standard torch format
    applied to the fixed-shape label columns. Variable-length sequence columns
    stay as Python lists (cheaper than per-row nested-tensor materialization)
    and get tensorized inside ``collate_batch``.
    """
    ds = load_from_disk(os.path.join(data_dir, split))
    ds.set_format(
        type="torch",
        columns=["y_thetas", "y_alpha_demo", "y_beta_demo"],
        output_all_columns=True,
    )
    return ds


def make_arrow_loader(
    ds: Dataset,
    batch_size: int,
    *,
    shuffle: bool = True,
    sampler=None,
    normalizer=None,
    num_workers: int = 0,
) -> DataLoader:
    """Wrap a loaded Arrow Dataset in a DataLoader with our collate + optional
    runtime normalization. Split out from ``load_arrow_split`` so callers that
    need the Dataset *first* (e.g., to build a ``DistributedSampler``) don't
    have to load it twice.
    """
    def _collate(batch):
        return collate_batch(batch, normalizer=normalizer)

    return DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(shuffle and sampler is None),
        collate_fn=_collate,
        num_workers=num_workers,
    )


def make_arrow_dataloader(
    data_dir: str,
    split: str,
    batch_size: int,
    *,
    shuffle: bool = True,
    sampler=None,
    normalizer=None,
    num_workers: int = 0,
) -> Tuple[Dataset, DataLoader]:
    """Convenience: ``load_arrow_split`` + ``make_arrow_loader`` in one call.
    Use when you don't need to construct the sampler yourself.
    """
    ds = load_arrow_split(data_dir, split)
    loader = make_arrow_loader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        normalizer=normalizer,
        num_workers=num_workers,
    )
    return ds, loader
