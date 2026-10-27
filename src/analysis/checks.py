# %% checks
"""Simulation-based checks: δ_N parameter recovery, model recovery for S1–S3, free-running PPC, split-half.

A simulated dataset replays human social sessions (templates, one per condition per simulated participant,
drawn with replacement) with a fitted policy playing every play game and the transformer re-read before each
one, so the readout entering a round's second play game sees the simulated first one. The frames become
``TrialData`` through ``simulated_trial_data`` and the fitting code runs unchanged.
"""

import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.compare import loo, paired
from analysis.data import TrialData
from analysis.fitting import Fit, fit
from analysis.games import CONDITION
from analysis.policies import REGISTRY, ModelSpec, belief, grids
from analysis.predictive import FittedPolicy, sample_participants, simulated_trial_data
from analysis.thompson import table
from inference.sessions import load_shots, validate_sessions
from inference.simulate import simulate_session

SOCIAL = {seq: cond for seq, cond in CONDITION.items() if cond != "solo"}   # game sequence -> expert-first / novice-first
N_RECOVERY, N_DATASETS, COVER_AT_LEAST = 100, 25, 22   # open decision 5; the 95%-coverage rule
COMPARISONS = {"S1": ("Switch-by-round", "Switch"), "S2": ("H+Both", "src:recent"), "S3": ("H+Both", "vis:last")}   # (A, B) behind each rule
N_MODEL_RECOVERY = 5   # datasets per generating side


def templates(preprocessed: Path) -> dict:
    """Human social sessions as shot frames, by condition: {'expert-first': [frame, ...], 'novice-first': [...]}."""
    shots = load_shots(str(preprocessed))
    validate_sessions(shots)
    return {cond: [g for _, g in shots[shots["game_sequence"] == seq].groupby("session_id", sort=True)] for seq, cond in SOCIAL.items()}


def simulate_dataset(spec: ModelSpec, people: list, pools: dict, model, normalizer, seed: int) -> TrialData:
    """``people`` (parameter dicts, one per simulated participant) each play one session per condition on a
    template drawn with replacement from ``pools``; fresh session and participant ids. Seeded: rewards from a
    ``random.Random``, actions and template draws from a numpy Generator, as ``run_inference.py`` does."""
    rng, np_rng, frames = random.Random(seed), np.random.default_rng(seed), []
    for i, params in enumerate(people):
        policy = FittedPolicy(spec, params)
        for cond, pool in pools.items():
            t = pool[np_rng.integers(len(pool))].assign(session_id=f"sim{seed}_{i:04d}_{cond}", anonymous_id=f"sim{seed}_{i:04d}")
            frames.append(simulate_session(model, normalizer, t, policy, rng, np_rng))
    return simulated_trial_data(frames, f"sim{seed}")


def fit_simulated(spec: ModelSpec, T: TrialData, cache_dir: Path, key: str) -> Fit:
    """GPU-built tables for a simulated dataset (the spec's source, plus novice for a mixture) and its cached fit."""
    k, w, c = grids(spec)
    sources = (spec.source, "novice") if spec.mixture != "none" else (spec.source,)
    tabs = [table(f"{key}_{src}_{int(spec.kappa)}_{int(spec.own)}", belief(T, src), T.own_seg, T.own_prev, cache_dir / "tables", k, w, c, builder="gpu") for src in sources]
    return fit(spec, T, tabs[0], key, cache_dir / "fits", tab2=tabs[1] if len(tabs) > 1 else None)


def recover_delta(posterior: dict, pools: dict, model, normalizer, cache_dir: Path, n_datasets: int = N_DATASETS,
                  n_people: int = N_RECOVERY, seed: int = 0, datasets=None) -> pd.DataFrame:
    """δ_N recovery. Dataset d takes posterior draw d of a seeded draw list (its group values, δ_N among them, are the
    truth), samples ``n_people`` participants from that draw's hierarchy, simulates, fits Switch-by-round. One row per
    dataset: truth, posterior mean, 95% ETI, covered, sign (ETI above 0), gate passed. Simulation and fit are cached
    under ``cache_dir`` by dataset index, so an array task runs the ``datasets`` it is given and the rest assemble."""
    spec, draws, rows = REGISTRY["Switch-by-round"], np.random.default_rng(seed).choice(posterior["delta_N"].size, n_datasets, replace=False), []
    for d in range(n_datasets) if datasets is None else datasets:
        path = Path(cache_dir) / f"dataset_{d:02d}.pkl"
        if not path.exists():
            people = sample_participants(posterior, int(draws[d]), n_people, np.random.default_rng(seed + d))
            path.parent.mkdir(parents=True, exist_ok=True)
            pickle.dump((simulate_dataset(spec, people, pools, model, normalizer, seed + d), people[0]["delta_N"]), open(path, "wb"))
        T, truth = pickle.load(open(path, "rb"))
        f = fit_simulated(spec, T, Path(cache_dir), f"dataset_{d:02d}")
        dn = np.asarray(f.posterior["delta_N"]).reshape(-1)
        lo, hi = np.percentile(dn, [2.5, 97.5])
        rows.append(dict(dataset=d, truth=truth, mean=dn.mean(), lo=lo, hi=hi, covered=bool(lo <= truth <= hi), sign=bool(lo > 0), passed=f.passed))
    return pd.DataFrame(rows)


def recovery_verdict(rows: pd.DataFrame) -> dict:
    """Coverage count, sign count and the pass flag of the 95%-coverage rule (≥ COVER_AT_LEAST of N_DATASETS)."""
    return dict(covered=int(rows["covered"].sum()), sign=int(rows["sign"].sum()), n=len(rows), passed=bool(rows["covered"].sum() >= COVER_AT_LEAST))


def model_recovery(posteriors: dict, pools: dict, model, normalizer, cache_dir: Path, n_per_side: int = N_MODEL_RECOVERY,
                   n_people: int = N_RECOVERY, seed: int = 0, units=None) -> pd.DataFrame:
    """Model recovery for S1–S3. Dataset (comparison, side, d) is generated from ``side``'s fitted model at a posterior
    draw seeded by its indices, both models of the comparison are fitted on it, and the 2 SE rule names the winner
    (A, B or undecided). One row per dataset; ``units`` = [(comparison, side, d), ...] restricts an array task.
    Simulation, tables and fits are cached under ``cache_dir`` like ``recover_delta``'s."""
    rows = []
    for ci, (c, (a, b)) in enumerate(COMPARISONS.items()):
        for si, side in enumerate((a, b)):
            for d in range(n_per_side):
                if units is not None and (c, side, d) not in units:
                    continue
                key, rng = f"{c}_{side.replace(':', '-')}_{d:02d}", np.random.default_rng([seed, ci, si, d])
                path = Path(cache_dir) / f"{key}.pkl"
                if not path.exists():
                    post = posteriors[side]
                    people = sample_participants(post, int(rng.integers(next(iter(post.values())).shape[0] * next(iter(post.values())).shape[1])), n_people, rng)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    pickle.dump(simulate_dataset(REGISTRY[side], people, pools, model, normalizer, int(rng.integers(2**31))), open(path, "wb"))
                T = pickle.load(open(path, "rb"))
                fa, fb = (fit_simulated(REGISTRY[m], T, Path(cache_dir), key) for m in (a, b))
                q = paired(loo(fa), loo(fb), T.part)
                winner = a if q["delta"] > 2 * q["se"] else b if q["delta"] < -2 * q["se"] else "undecided"
                rows.append(dict(comparison=c, generating=side, dataset=d, delta=q["delta"], se=q["se"], winner=winner, correct=winner == side, passed=fa.passed and fb.passed))
    return pd.DataFrame(rows)


def confusion(rows: pd.DataFrame) -> pd.DataFrame:
    """Generating model × winner counts, one block per comparison."""
    return pd.crosstab([rows["comparison"], rows["generating"]], rows["winner"])
