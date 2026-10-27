# %% checks
"""Simulation-based checks: δ_N parameter recovery, model recovery for S1–S3, free-running PPC, split-half.

A simulated dataset replays human social sessions (templates, one per condition per simulated participant,
drawn with replacement) with a fitted policy playing every play game and the transformer re-read before each
one, so the readout entering a round's second play game sees the simulated first one. The frames become
``TrialData`` through ``simulated_trial_data`` and the fitting code runs unchanged.
"""

import random
from pathlib import Path

import numpy as np

from analysis.data import TrialData
from analysis.games import CONDITION
from analysis.policies import ModelSpec
from analysis.predictive import FittedPolicy, simulated_trial_data
from inference.sessions import load_shots, validate_sessions
from inference.simulate import simulate_session

SOCIAL = {seq: cond for seq, cond in CONDITION.items() if cond != "solo"}   # game sequence -> expert-first / novice-first


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
