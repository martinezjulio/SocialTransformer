"""Tests for segment-cutoff attention masking in train.py.

Three invariants, all of which were violated before:

1. No attention row is ever fully masked. A softmax over an all -inf row is NaN,
   and under "current_bandit_only" every position in an earlier round loses all of
   its keys once the cutoff passes that round.
2. The mask is per-example. Each example draws its own cutoff and its loss target is
   derived from that cutoff, so collapsing to one shared [T, T] mask scores each
   example's target against a different example's context.
3. No NaN reaches the model outputs in eval() mode. The NaN in (1) is invisible in
   train() mode -- PyTorch's fused inference path is what surfaces it -- so this must
   be asserted with the model in eval().

Tests needing real data/weights skip when they are absent, matching conftest.py.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from train import create_segment_mask, sample_segment_cutoffs  # noqa: E402

STRATEGIES = ["all_bandits", "current_bandit_only"]

# Fixed dataset geometry: 3 rounds x [SEP, demo, learner, demo, learner, QUERY],
# T_demo = T_learner = 6 -> 18 segments, 78 tokens. Identical for every example.
SEG_BOUNDARIES = [0, 1, 7, 13, 19, 25, 26, 27, 33, 39, 45, 51, 52, 53, 59, 65, 71, 77, 78]
SEG_ROUND = [0] * 6 + [1] * 6 + [2] * 6
NUM_SEGMENTS = len(SEG_ROUND)
SEQ_LEN = 78


def _geometry(batch_size: int = 1):
    return (
        torch.tensor([SEG_BOUNDARIES] * batch_size),
        [list(SEG_ROUND) for _ in range(batch_size)],
        torch.tensor([[0, 1, 2]] * batch_size),
    )


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("cutoff", range(1, NUM_SEGMENTS + 1))
def test_no_fully_masked_rows(strategy, cutoff):
    """Every position must retain at least one visible key at every cutoff."""
    sb, sr, rbm = _geometry()
    mask = create_segment_mask(
        sb, [cutoff], SEQ_LEN, torch.device("cpu"),
        masking_strategy=strategy, segment_round=sr, round_bandit_map=rbm,
    )
    dead = mask.all(dim=-1).sum().item()
    assert dead == 0, f"{strategy} cutoff={cutoff}: {dead} fully-masked rows"


def test_mask_is_per_example():
    """The mask must carry one [T, T] plane per example, not a single shared plane."""
    B = 4
    sb, sr, rbm = _geometry(B)
    cutoffs = [3, 9, 14, 18]
    mask = create_segment_mask(
        sb, cutoffs, SEQ_LEN, torch.device("cpu"),
        masking_strategy="all_bandits", segment_round=sr, round_bandit_map=rbm,
    )
    assert mask.shape == (B, SEQ_LEN, SEQ_LEN)
    # Distinct cutoffs must produce distinct planes.
    for i in range(1, B):
        assert not torch.equal(mask[0], mask[i]), f"plane {i} matches plane 0"


def test_cutoffs_and_targets_agree():
    """sample_segment_cutoffs must resolve the target bandit from each example's own
    cutoff -- the pairing the loss depends on."""
    import random

    B = 8
    _, sr, rbm = _geometry(B)
    batch = {"segment_round": sr, "round_bandit_map": rbm}
    cutoffs, targets = sample_segment_cutoffs(batch, random.Random(0))

    assert len(cutoffs) == len(targets) == B
    for b, (cutoff, target) in enumerate(zip(cutoffs, targets)):
        assert 1 <= cutoff <= NUM_SEGMENTS
        expected_round = sr[b][cutoff - 1]
        assert target == int(rbm[b, expected_round].item())


def test_seeded_rng_is_reproducible():
    """Validation reseeds its generator each pass; identical seeds must give identical
    cutoffs, or val/loss carries resampling noise into the scheduler."""
    import random

    _, sr, rbm = _geometry(8)
    batch = {"segment_round": sr, "round_bandit_map": rbm}
    a = sample_segment_cutoffs(batch, random.Random(42))
    b = sample_segment_cutoffs(batch, random.Random(42))
    assert a == b


# ---- tests requiring a real checkpoint ----

@pytest.fixture(scope="session")
def trained_model():
    """Newest local checkpoint, or skip. Only the architecture matters here."""
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
    out = os.environ.get("OUTPUT_DATA_DIR")
    if not out:
        pytest.skip("OUTPUT_DATA_DIR not set in .env")
    ckpts = sorted(
        (Path(out) / "training_runs").glob("*/checkpoints/*.pt"),
        key=lambda p: p.stat().st_mtime,
    )
    if not ckpts:
        pytest.skip("no local checkpoint under $OUTPUT_DATA_DIR/training_runs/")

    from models import BaseSocialTransformer
    model, ck = BaseSocialTransformer.load_checkpoint(local_path=str(ckpts[-1]))
    return model.eval(), ck["model_config"]


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_no_nan_in_eval_mode(trained_model, strategy):
    """Model outputs must be NaN-free at every cutoff with the model in eval() mode."""
    model, cfg = trained_model
    B = 2
    sb, sr, rbm = _geometry(B)
    num_arms, num_agents = cfg["num_arms"], cfg["num_agents"]

    # Minimal well-formed batch: structural tokens at the segment boundaries, real
    # events elsewhere. Exact values are irrelevant -- only NaN propagation is tested.
    starts = set(SEG_BOUNDARIES[:-1])
    special = torch.zeros(B, SEQ_LEN, dtype=torch.long)
    agent = torch.zeros(B, SEQ_LEN, dtype=torch.long)
    action = torch.zeros(B, SEQ_LEN, dtype=torch.long)
    reward = torch.zeros(B, SEQ_LEN, dtype=torch.long)
    bandit = torch.zeros(B, SEQ_LEN, dtype=torch.long)
    for seg_idx, pos in enumerate(SEG_BOUNDARIES[:-1]):
        # QUERY closes each round (segments 5, 11, 17); the rest of the boundaries
        # that start a round are BANDIT_SEP.
        special[:, pos] = 1 if seg_idx % 6 == 5 else 2
        agent[:, pos] = num_agents
        action[:, pos] = num_arms
        reward[:, pos] = 2
    for pos in range(SEQ_LEN):
        if pos not in starts:
            agent[:, pos] = 1
        bandit[:, pos] = SEG_ROUND[min(range(NUM_SEGMENTS),
                                       key=lambda s: abs(SEG_BOUNDARIES[s] - pos))]

    batch = {
        "bandit_ids": bandit, "agent_ids": agent, "action_ids": action,
        "reward_ids": reward, "special_ids": special,
        "pad_mask": torch.zeros(B, SEQ_LEN, dtype=torch.bool),
    }

    for cutoff in range(1, NUM_SEGMENTS + 1):
        mask = create_segment_mask(
            sb, [cutoff] * B, SEQ_LEN, torch.device("cpu"),
            masking_strategy=strategy, segment_round=sr, round_bandit_map=rbm,
        ).repeat_interleave(cfg["nhead"], dim=0)
        with torch.no_grad():
            out = model(batch, custom_attn_mask=mask)
        for key in ("alpha_demo", "beta_demo"):
            assert not torch.isnan(out[key]).any(), \
                f"{strategy} cutoff={cutoff}: NaN in {key}"
        theta_key = "theta_logits" if "theta_logits" in out else "alpha_theta"
        assert not torch.isnan(out[theta_key]).any(), \
            f"{strategy} cutoff={cutoff}: NaN in {theta_key}"
