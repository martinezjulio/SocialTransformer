"""The new ROS and Switch-by-round fits reproduce the legacy pilot E fits within Monte Carlo error.

Legacy trial frames came from the 19 Oct inference, which scrambled the shot order of four social sessions,
so pointwise log densities are compared only on sessions whose action sequences match; participant ids are
relabelled between the two frames and are aligned through the session id. Skips unless both caches exist.
"""
import glob
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from analysis.data import load_sessions, trial_data  # noqa: E402

LEGACY = "legacy_analysis/cached_data/policy_{}_pilot_E_wandb_0i8yhqm7_epoch_119_best.pkl"
MODELS = {"ROS": ("ROS_*.pkl", "C1:H+Both"), "Switch-by-round": ("Switch-by-round_*.pkl", "C4:switch+round")}
GROUP = {"ROS": {"mu_log_kappa": "mu_log_kappa", "mu_rho0": "mu_rho0", "tau_log_kappa": "tau_log_kappa", "log_lambda_w": "log_lambda_within", "lambda_c": "lambda_carry"},
         "Switch-by-round": {"mu_logit_omega_E": "mu_logit_omega_E", "mu_logit_omega_N": "mu_logit_omega_N", "delta_N": "delta_omega_N_0"}}
PERSON = {"ROS": ["log_kappa"], "Switch-by-round": ["log_kappa", "logit_omega_N"]}


@pytest.fixture(scope="module")
def caches():
    root = Path(os.environ.get("OUTPUT_DATA_DIR", ""))
    fits = {m: glob.glob(str(root / "analysis/pilot_E/choice/fits" / pat)) for m, (pat, _) in MODELS.items()}
    if not root.name or not (root / LEGACY.format("fits_all")).exists() or not all(fits.values()):
        pytest.skip("legacy or new pilot E fit caches not present")
    legacy = pickle.load(open(root / LEGACY.format("fits_all"), "rb"))
    trial = pickle.load(open(root / LEGACY.format("logp"), "rb"))["trial"]
    new = {m: pickle.load(open(sorted(p)[-1], "rb")) for m, p in fits.items()}
    T = trial_data(load_sessions(root / "eval" / "pilot_E", "pilot_E"))
    assert (trial["sid"].to_numpy() == np.asarray(T.session)).all()
    return legacy, trial, new, T


def flat(x):
    x = np.asarray(x)
    return x.reshape(-1, *x.shape[2:])


def waic(fit):
    ll = fit.log_lik.reshape(-1, fit.log_lik.shape[-1]).astype(np.float64)
    return logsumexp(ll, 0) - np.log(ll.shape[0]) - ll.var(0)


def test_group_parameters_within_one_posterior_sd(caches):
    legacy, _, new, _ = caches
    for m, sites in GROUP.items():
        for ours, theirs in sites.items():
            a, b = flat(new[m].posterior[ours]), np.asarray(legacy[MODELS[m][1]][theirs])
            assert abs(a.mean() - b.mean()) < b.std(), (m, ours, a.mean(), b.mean(), b.std())


def test_participant_means_aligned_by_session(caches):
    legacy, trial, new, T = caches
    pairs = pd.DataFrame({"legacy": trial["part"].to_numpy(), "new": T.part}).drop_duplicates().sort_values("legacy")
    assert len(pairs) == len(T.people) and pairs["new"].is_unique
    for m, sites in PERSON.items():
        for s in sites:
            a, b = flat(new[m].posterior[s]).mean(0)[pairs["new"].to_numpy()], np.asarray(legacy[MODELS[m][1]][s]).mean(0)
            assert np.corrcoef(a, b)[0, 1] > 0.9, (m, s, np.corrcoef(a, b)[0, 1])


def test_pointwise_elpd_on_unscrambled_sessions(caches):
    legacy, trial, new, T = caches
    sid, mismatch = trial["sid"].to_numpy(), trial["act"].to_numpy() != T.act
    same = ~np.isin(sid, np.unique(sid[mismatch]))
    assert (~same).sum() == 4 * 24, "four scrambled sessions of 24 trials"
    d = {m: waic(new[m]) - np.asarray(legacy[MODELS[m][1]]["elpd"], float) for m in MODELS}
    for m, dm in d.items():
        assert np.abs(dm[same]).max() < 1.5 and abs(dm[same].sum()) < 20, (m, np.abs(dm[same]).max(), dm[same].sum())
    paired = pd.Series(d["Switch-by-round"][same] - d["ROS"][same]).groupby(T.part[same]).sum()
    assert abs(paired.sum()) < 2 * paired.std() * np.sqrt(len(paired)), "the by-round gain over ROS matches legacy within 2 SE"
