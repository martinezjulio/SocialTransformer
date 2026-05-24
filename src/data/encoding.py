"""
Build-time encoding of raw simulation examples into the on-disk Arrow row schema.

The runtime dataloader never calls this module. The pipeline is:

    simulate_rounds_run() → build_training_example()   (in generate_simulations.py)
        ↓
    build_arrow_row()                                  (here — encodes once at build time)
        ↓
    Dataset.from_generator(...).save_to_disk(...)      (HuggingFace Arrow + mmap)
        ↓
    load_from_disk(...)                                (in src/data/dataloader.py at training time)

Everything here is a pure function of (raw_example, dataset config). No randomness,
no per-run state.
"""

from typing import Dict, List, Tuple

import numpy as np
from datasets import Array2D, Array3D, Features, Sequence, Value


# ---------- constants (shared with runtime dataloader via re-export) ----------

SPECIAL = {
    "NONE": 0,
    "QUERY": 1,
    "BANDIT_SEP": 2,
}

LEARNER_RAW_ID = 0

# segment_info["type"] is encoded as a small int on disk. Comparisons in
# train.py / eval.py use `== SEGMENT_TYPE_TO_ID["learner"]` etc.
SEGMENT_TYPE_TO_ID = {
    "bandit_sep": 0,
    "demo": 1,
    "learner": 2,
    "query": 3,
}
ID_TO_SEGMENT_TYPE = {v: k for k, v in SEGMENT_TYPE_TO_ID.items()}


# ---------- Arrow schema ----------

def build_features(K: int, num_demonstrators: int, num_bandits: int) -> Features:
    """Schema for one row in the on-disk Arrow dataset.

    Variable-length fields use `Sequence(Value(...))`. Fixed-shape labels use
    `Array2D` / `Array3D` so they materialize as contiguous tensors under
    `set_format("torch")` without per-row nested-list conversion.
    """
    return Features({
        # Variable-length token channels
        "bandit_ids":         Sequence(Value("int64")),
        "agent_ids":          Sequence(Value("int64")),
        "action_ids":         Sequence(Value("int64")),
        "reward_ids":         Sequence(Value("int64")),
        "special_ids":        Sequence(Value("int64")),

        # Variable-length structural index columns
        "round_boundaries":   Sequence(Value("int64")),
        "round_bandit_map":   Sequence(Value("int64")),
        "segment_boundaries": Sequence(Value("int64")),

        # Flattened segment_info — three parallel variable-length list columns.
        # eval.py needs `agent` to identify expert-vs-novice ordering for
        # game_sequence; train.py only reads `round`. `bandit` (only present
        # on QUERY segments today) isn't read anywhere downstream.
        "segment_round":      Sequence(Value("int64")),
        "segment_type_id":    Sequence(Value("int64")),
        "segment_agent":      Sequence(Value("int64")),

        # Fixed-shape labels (num_bandits, num_demonstrators, K constant per dataset)
        "y_thetas":     Array2D(shape=(num_bandits, K), dtype="float32"),
        "y_alpha_demo": Array3D(shape=(num_bandits, num_demonstrators, K), dtype="float32"),
        "y_beta_demo":  Array3D(shape=(num_bandits, num_demonstrators, K), dtype="float32"),
    })


# ---------- per-example encoding ----------

def _remap_agent_ids_local(
    events: List[Tuple[int, int, int, int]],
    num_agents: int,
) -> Tuple[List[Tuple[int, int, int, int]], int]:
    """Replace structural-token sentinels (agent_id == -1) with the padding id
    `num_agents`. Real agent ids (learner=0, demos=1..) are already correct
    from the simulator.

    Returns (processed_events, num_agents_used). Raises if the example carries
    more agents than the configured `num_agents`.
    """
    demos = sorted({ag for (_, ag, _, _) in events if ag not in (LEARNER_RAW_ID, -1)})
    num_agents_used = len(demos) + 1  # +1 for learner

    if num_agents_used > num_agents:
        raise ValueError(
            f"Example has {num_agents_used} agents, but num_agents={num_agents}"
        )

    out: List[Tuple[int, int, int, int]] = []
    for (b, ag, a, r) in events:
        if ag == -1:
            out.append((b, num_agents, a, r))
        else:
            out.append((b, ag, a, r))
    return out, num_agents_used


def _encode_stream(
    stream: List[Tuple[int, int, int, int]],
    *,
    num_agents: int,
    num_arms: int,
    num_bandits: int,
) -> Dict[str, List[int]]:
    """Encode a stream of (bandit_id, agent_id, action, reward) tuples into the
    five token channels (bandit_ids, agent_ids, action_ids, reward_ids,
    special_ids). Structural tokens (agent_id == num_agents) are distinguished
    by the action field: -2 means QUERY, -1 means BANDIT_SEP.
    """
    bandit_ids: List[int] = []
    agent_ids:  List[int] = []
    action_ids: List[int] = []
    reward_ids: List[int] = []
    special_ids: List[int] = []

    for (b, ag, a, r) in stream:
        if ag == num_agents:
            # Structural token. action == -2 → QUERY, action == -1 → BANDIT_SEP.
            bandit_ids.append(min(max(b, 0), num_bandits - 1))
            agent_ids.append(num_agents)
            action_ids.append(num_arms)
            reward_ids.append(2)
            special_ids.append(SPECIAL["QUERY"] if a == -2 else SPECIAL["BANDIT_SEP"])
            continue

        # Real event
        bandit_ids.append(min(max(b, 0), num_bandits - 1))
        agent_ids.append(ag)
        action_ids.append(a if (0 <= a < num_arms) else num_arms)
        reward_ids.append(r if r in (0, 1) else 2)
        special_ids.append(SPECIAL["NONE"])

    return {
        "bandit_ids": bandit_ids,
        "agent_ids": agent_ids,
        "action_ids": action_ids,
        "reward_ids": reward_ids,
        "special_ids": special_ids,
    }


def build_arrow_row(
    raw_example: Dict,
    *,
    K: int,
    num_demonstrators: int,
    num_bandits: int,
    num_agents: int,
) -> Dict:
    """Convert one training example (output of `build_training_example`) into a
    row dict that matches `build_features(...)`. Pure function — no normalization,
    no run-state.
    """
    remapped, _ = _remap_agent_ids_local(raw_example["events"], num_agents=num_agents)
    enc = _encode_stream(
        remapped,
        num_agents=num_agents,
        num_arms=K,
        num_bandits=num_bandits,
    )

    bandits = raw_example["labels"]["bandits"]
    y_thetas = np.zeros((num_bandits, K), dtype=np.float32)
    y_alpha  = np.zeros((num_bandits, num_demonstrators, K), dtype=np.float32)
    y_beta   = np.zeros((num_bandits, num_demonstrators, K), dtype=np.float32)
    for b_idx, b_info in enumerate(bandits):
        y_thetas[b_idx] = b_info["theta"]
        for d in range(num_demonstrators):
            y_alpha[b_idx, d] = b_info["demonstrators"][d]["alpha0"]
            y_beta[b_idx, d]  = b_info["demonstrators"][d]["beta0"]

    seg_round  = [s["round"]                    for s in raw_example["segment_info"]]
    seg_typeid = [SEGMENT_TYPE_TO_ID[s["type"]] for s in raw_example["segment_info"]]
    seg_agent  = [s["agent"]                    for s in raw_example["segment_info"]]

    return {
        "bandit_ids":         enc["bandit_ids"],
        "agent_ids":          enc["agent_ids"],
        "action_ids":         enc["action_ids"],
        "reward_ids":         enc["reward_ids"],
        "special_ids":        enc["special_ids"],
        "round_boundaries":   list(raw_example["round_boundaries"]),
        "round_bandit_map":   list(raw_example["round_bandit_map"]),
        "segment_boundaries": list(raw_example["segment_boundaries"]),
        "segment_round":      seg_round,
        "segment_type_id":    seg_typeid,
        "segment_agent":      seg_agent,
        "y_thetas":           y_thetas,
        "y_alpha_demo":       y_alpha,
        "y_beta_demo":        y_beta,
    }
