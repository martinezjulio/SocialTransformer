"""Simulation checks: a lapse-1 policy is uniform, the PPC targets reproduce the pilot's descriptive anchors, and
δ_N recovery runs end to end on one tiny simulated dataset (transformer in the loop, GPU tables on a reduced grid).
The last two skip without the pilot E data, the checkpoint the inference run used, and the cached Switch-by-round fit."""
import glob
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import analysis.policies as P  # noqa: E402
from analysis.checks import recover_delta, recovery_verdict, targets, templates  # noqa: E402
from analysis.data import load_sessions, trial_data  # noqa: E402
from analysis.predictive import FittedPolicy  # noqa: E402
from inference.learners import TrialState  # noqa: E402

K = 12


def test_lapse_one_policy_is_uniform():
    rng = np.random.default_rng(0)
    a, b = (torch.tensor(rng.gamma(2.0, 3.0, size=(3, K))) for _ in range(2))
    s = TrialState(a, b, None, None, 0, 1, 1, torch.tensor(rng.integers(0, 3, (2, K)).astype(float)), torch.zeros(2, K), prev_action=4, prev_reward=1, round_index=1)
    params = dict(log_kappa=0.3, log_lambda_w=0.1, lambda_c=0.2, rho0=1.5, rho_win=2.0, logit_omega_E=3.0, logit_omega_N=1.0, delta_E=0.5, delta_N=2.0)
    for spec in (P.ModelSpec("u1", habit=True, lapse=1.0), P.ModelSpec("u2", habit=True, mixture="by_round", lapse=1.0)):
        assert np.allclose(FittedPolicy(spec, params).probs(s), 1 / K, atol=1e-12)
    with_habit, without = (FittedPolicy(P.ModelSpec("h", habit=h), params).probs(s) for h in (True, False))
    assert abs(with_habit.sum() - 1) < 1e-12 and with_habit[4] > without[4], "the habit tilt raises the previous arm"


@pytest.fixture(scope="module")
def pilot():
    root, exp = os.environ.get("OUTPUT_DATA_DIR"), os.environ.get("EXPERIMENT_DATA_DIR")
    fits = glob.glob(str(Path(root or "") / "analysis/pilot_E/choice/fits/Switch-by-round_*.pkl"))
    if not root or not exp or not (Path(root) / "eval/pilot_E/data/experiment_trial_predictions.csv").exists() or not fits:
        pytest.skip("pilot E inference CSVs, experiment data or the cached Switch-by-round fit not present")
    return Path(root), Path(exp) / "pilot_E" / "preprocessed_participant_data.csv", pickle.load(open(sorted(fits)[-1], "rb")).posterior


def test_targets_reproduce_descriptive_anchors(pilot):
    root, _, _ = pilot
    t = targets(trial_data(load_sessions(root / "eval" / "pilot_E", "pilot_E")))
    assert abs(t["after_novice_rise"] - 0.044) < 0.006 and abs(t["entropy_fall"] - 0.236) < 0.006, t   # test_games' cell means
    assert 0.1 < t["zigzag"] < 0.2 and 0.7 < t["stay_win"] < 0.8 and 0.3 < t["stay_loss"] < 0.4, t


def test_delta_recovery_runs_end_to_end_on_a_tiny_dataset(pilot, tmp_path, monkeypatch):
    root, preprocessed, post = pilot
    from inference.readout import load_model
    ckpt = root / "training_runs/wandb_0i8yhqm7/checkpoints/epoch_119_best.pt"
    if not ckpt.exists():
        pytest.skip(f"{ckpt} not present")
    monkeypatch.setattr(P, "KAPPA", P.KAPPA[::4]); monkeypatch.setattr(P, "LAM_W", P.LAM_W[::3]); monkeypatch.setattr(P, "LAM_C", np.concatenate([P.LAM_C[:1], P.LAM_C[1::2]]))
    model, _, norm = load_model(str(ckpt), None, None, "cuda" if torch.cuda.is_available() else "cpu")
    rows = recover_delta(post, templates(preprocessed), model, norm, tmp_path, n_datasets=3, n_people=2, datasets=[1])
    assert len(rows) == 1 and rows.iloc[0]["dataset"] == 1 and np.isfinite(rows[["truth", "mean", "lo", "hi"]].to_numpy()).all()
    assert rows.iloc[0]["lo"] < rows.iloc[0]["mean"] < rows.iloc[0]["hi"] and set(rows.columns) >= {"covered", "sign", "passed"}
    assert (tmp_path / "dataset_01.pkl").exists() and len(list((tmp_path / "tables").glob("dataset_01_*_gpu_*.npy"))) == 2 and len(list((tmp_path / "fits").glob("*.pkl"))) == 1
    v = recovery_verdict(rows)
    assert v["n"] == 1 and v["covered"] in (0, 1) and v["passed"] is False, "one dataset cannot reach the 22-of-25 rule"


def test_recovery_and_model_recovery_verdicts():
    """The pass rules of prereg 4.7.3 on hand-made frames: coverage and the truth's-side sign in ≥ 22 of 25; per comparison, A wins a
    majority of A-generated datasets and at most a minority of B-generated ones (undecided counts as correct there, the nested S4 pair)."""
    import pandas as pd
    from analysis.checks import COMPARISONS, COVER_AT_LEAST, GATING, model_recovery_verdict
    r = pd.DataFrame(dict(covered=[True] * 23 + [False] * 2, sign=[True] * COVER_AT_LEAST + [False] * 3))
    assert recovery_verdict(r) == dict(covered=23, sign=22, n=25, passed=True)
    assert not recovery_verdict(r.assign(sign=[True] * 21 + [False] * 4))["passed"] and not recovery_verdict(r.assign(covered=[True] * 21 + [False] * 4))["passed"]
    (s1, s1b), (s4a, s4b) = COMPARISONS["S1a"], COMPARISONS["S4"]
    rows = pd.DataFrame([dict(comparison="S1a", generating=s1, winner=s1)] * 3 + [dict(comparison="S1a", generating=s1b, winner=s1b)] * 3
                        + [dict(comparison="S4", generating=s4a, winner=s4a)] * 3 + [dict(comparison="S4", generating=s4b, winner="undecided")] * 3)
    v = model_recovery_verdict(rows)
    assert v["S1a"]["passed"] and v["S4"]["passed"] and not v["S2a"]["passed"], "S4 passes through its undecided Switch side; S2a has no rows"
    assert v["S1a"] == dict(a_wins_on_a=3, n_a=3, a_wins_on_b=0, n_b=3, gating=True, passed=True) and not v["S4"]["gating"] and set(GATING) == {"S1a", "S2a"}
    assert not model_recovery_verdict(rows.assign(winner=rows["winner"].where(rows["generating"] != s4b, s4a)))["S4"]["passed"], "the richer model winning on Switch data fails"
    assert not model_recovery_verdict(rows.assign(winner=rows["winner"].where(rows["generating"] != s1, "undecided")))["S1a"]["passed"], "A must win on its own data"
