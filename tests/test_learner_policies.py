"""Tests for the learner-segment policy machinery in generate_simulations.py.

Covers the three things the rewrite has to get right:
  - the "similar" imitation distribution (reserve form, Option 2),
  - the stratified (expert_id, learner_policy) pool — balance + expert_id locking,
  - the generated data: per-row provenance columns, demo-first layout, no `order`.

These tests generate their own tiny dataset in a tmp dir, so they don't depend
on $TRAINING_DATA_ROOT_DIR or any prebuilt artifacts.
"""

from __future__ import annotations

import collections
import json
import random
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402
from datasets import load_from_disk  # noqa: E402

import generate_simulations as gs  # noqa: E402

PAYOFFS = [0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.75, 0.8]


# ---------- imitation_probs (Option 2, reserve form) ----------

def test_imitation_probs_reserve_form():
    K, floor = 12, 0.1
    counts = [3, 2, 1] + [0] * 9  # demo pulled 3 distinct arms over 6 steps
    p = gs.imitation_probs(counts, K, floor)
    assert torch.isclose(p.sum(), torch.tensor(1.0), atol=1e-6)
    # unpulled arms share exactly `floor`, uniformly
    unpulled = p[3:]
    assert torch.isclose(unpulled.sum(), torch.tensor(floor), atol=1e-6)
    assert torch.allclose(unpulled, torch.full((9,), floor / 9), atol=1e-6)
    # pulled arms share (1 - floor) in proportion to counts
    assert torch.isclose(p[0], torch.tensor((1 - floor) * 3 / 6), atol=1e-6)
    assert torch.isclose(p[1], torch.tensor((1 - floor) * 2 / 6), atol=1e-6)
    assert torch.isclose(p[2], torch.tensor((1 - floor) * 1 / 6), atol=1e-6)


def test_imitation_probs_all_pulled_degenerate():
    # If every arm was pulled there is no unpulled mass to place -> plain empirical.
    p = gs.imitation_probs([2, 1, 1], 3, 0.1)
    assert torch.isclose(p.sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(p, torch.tensor([0.5, 0.25, 0.25]), atol=1e-6)


# ---------- pool: locking + balance ----------

def test_allowed_expert_ids_locking():
    assert gs.allowed_expert_ids("expert-priors-always") == [0, 1]
    assert gs.allowed_expert_ids("naive-learner") == [0, 1]
    assert gs.allowed_expert_ids("expert-similar-always") == [0]
    assert gs.allowed_expert_ids("expert-novice-similar") == [0]
    assert gs.allowed_expert_ids("novice-similar-always") == [1]
    assert gs.allowed_expert_ids("novice-expert-similar") == [1]


def test_build_policy_pool_balanced_and_locked():
    random.seed(0)
    n = 180
    pool = gs.build_policy_pool(n, "mixed")
    assert len(pool) == n
    pol = collections.Counter(p for _, p in pool)
    assert set(pol) == set(gs.LEARNER_POLICIES)
    assert all(c == n // 9 for c in pol.values())  # 180/9 = 20 exactly
    # every cell respects the locking rule
    for eid, p in pool:
        assert eid in gs.allowed_expert_ids(p)
    # global expert_id balance
    eids = collections.Counter(eid for eid, _ in pool)
    assert eids[0] == eids[1]


# ---------- guards ----------

def _run(**overrides):
    kwargs = dict(
        K=12, num_demonstrators=2, num_rounds=2, change_bandits_with_rounds=True,
        expert_id=0, learner_policy="naive-learner", similar_exploration_floor=0.1,
        demo_strategy="expert_novice_thompson", expert_N_min=128, expert_N_max=128,
        bandit_strategy="permute", bandit_payoffs=PAYOFFS,
    )
    kwargs.update(overrides)
    return gs.simulate_rounds_run(**kwargs)


def test_nonnaive_requires_two_demonstrators():
    with pytest.raises(ValueError):
        _run(num_demonstrators=3, learner_policy="expert-similar-always")


def test_nonnaive_requires_expert_novice_strategy():
    with pytest.raises(ValueError):
        _run(learner_policy="expert-priors-always", demo_strategy="uniform_thompson")


# ---------- behavioral: "similar" imitates the targeted demo ----------

def _segment_events(run, k):
    sb = run["segment_boundaries"]
    return run["events"][sb[k]:sb[k + 1]]


def test_similar_learner_imitates_target_demo():
    """expert-similar-always: learner pulls should land overwhelmingly on the
    arms the expert demo pulled that round (expert at agent_id 1, expert_id=0)."""
    random.seed(3)
    torch.manual_seed(3)
    hits = total = 0
    for _ in range(40):
        run = _run(learner_policy="expert-similar-always", expert_id=0)
        si = run["segment_info"]
        expert_arms = collections.defaultdict(set)  # round -> arms the expert pulled
        for k, info in enumerate(si):
            if info["type"] == "demo" and info["agent"] == 1:
                expert_arms[info["round"]].update(a for (_, _, a, _) in _segment_events(run, k))
        for k, info in enumerate(si):
            if info["type"] == "learner":
                allowed = expert_arms[info["round"]]
                for (_, _, a, _) in _segment_events(run, k):
                    total += 1
                    hits += int(a in allowed)
    frac = hits / total
    assert frac > 0.6, f"learner imitation fraction too low: {frac:.3f}"


# ---------- dataset-level invariants ----------

@pytest.fixture(scope="module")
def small_dataset(tmp_path_factory):
    random.seed(7)
    torch.manual_seed(7)
    out = tmp_path_factory.mktemp("lp_ds")
    common = dict(
        K=12, num_demonstrators=2, num_rounds=2, change_bandits_with_rounds=True,
        learner_policy="mixed", similar_exploration_floor=0.1, T_demo=6, T_learner=6,
        alpha0=1.0, beta0=1.0, demo_strategy="expert_novice_thompson",
        expert_N_min=128, expert_N_max=128, bandit_strategy="permute", bandit_payoffs=PAYOFFS,
    )
    gs.generate_split(out, "train", 90, **common)  # 90/9 = 10 per policy
    gs.write_root_metadata(out, num_train_runs=90, num_valid_runs=0,
                           num_test_runs=0, seed=7, **common)
    return out


def test_metadata_drops_order_adds_policy_keys(small_dataset):
    params = json.load(open(small_dataset / "metadata.json"))["parameters"]
    assert "order" not in params
    assert params["learner_policy"] == "mixed"
    assert len(params["learner_policy_set"]) == 9
    assert params["similar_exploration_floor"] == 0.1


def test_dataset_balance_locking_layout_and_provenance(small_dataset):
    ds = load_from_disk(str(small_dataset / "train"))
    assert "learner_policy" in ds.column_names
    assert "expert_id" in ds.column_names

    pol = collections.Counter(ds["learner_policy"])
    assert set(pol) == set(gs.LEARNER_POLICIES)
    assert all(c == 10 for c in pol.values())  # 90/9

    for lp, eid in zip(ds["learner_policy"], ds["expert_id"]):
        assert eid in gs.allowed_expert_ids(lp)  # locking holds in the data

    seg_name = {0: "sep", 1: "demo", 2: "learner", 3: "query"}
    expected = ["sep", "demo", "learner", "demo", "learner", "query"]
    for i in range(len(ds)):
        row = ds[i]
        rounds = collections.defaultdict(list)
        for r, t in zip(row["segment_round"], row["segment_type_id"]):
            rounds[r].append(seg_name[t])
        assert len(rounds) == 2  # num_rounds
        for seq in rounds.values():
            assert seq == expected  # demo-first only; no learner-first layout
