"""Inference CSVs of one version → the human trials and readouts the choice models read.

``load_sessions`` keeps the complete social sessions (every game has the modal number of trials;
``run_inference.py`` already skips solo sessions), asserts the two-round structure, and parses the
readout strings once. Rows per session: 48 trials (2 rounds × 4 games × 6 shots) and 12 cutoff
readouts, one per segment (cutoff c = beliefs after the first c segments; the readout entering play
game g of round r is cutoff 6r + g + 1 for g ∈ {1, 3}, i.e. right after the demonstration before it).
"""

import ast
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

N_ROUNDS, GAMES_PER_ROUND, SEGMENTS_PER_ROUND = 2, 4, 6
SOLO = "play-play-play-play"
READOUTS = ("expert", "novice", "recent")   # ``<name>_betas`` columns: per-arm (alpha, beta) pairs


def parse_betas(s: str) -> np.ndarray:
    """'[(a1, b1), (a2, b2), ...]' → (K, 2) float array."""
    return np.array(ast.literal_eval(s), dtype=float)


@dataclass(frozen=True)
class Sessions:
    version: str
    trials: pd.DataFrame        # human shots of complete social sessions: session_id, anonymous_id, game_sequence, round_index, game_index, demonstrator_type, position, reward
    readouts: pd.DataFrame      # one row per (session_id, cutoff): theta (K,), and expert/novice/recent (K, 2) arrays
    trials_per_game: int
    dropped: tuple              # incomplete session ids


def load_sessions(eval_dir: Path, version: str) -> Sessions:
    d = Path(eval_dir) / "data"
    trial = pd.read_csv(d / "experiment_trial_predictions.csv", low_memory=False)
    hum = trial[trial["learner_segment_input"] == "human"]
    n = hum.groupby(["session_id", "game_index"]).size()
    trials_per_game = int(n.mode()[0])
    dropped = tuple(sorted({sid for sid, _ in n[n != trials_per_game].index}))
    hum = hum[~hum["session_id"].isin(dropped)]
    trials = hum[["session_id", "anonymous_id", "game_sequence", "round_index", "game_index", "demonstrator_type", "position"]].copy()
    trials["reward"] = hum["reward_generated"].astype(float)
    trials = trials.sort_values(["session_id", "game_index"], kind="stable").reset_index(drop=True)

    cut = pd.read_csv(d / "experiment_cutoff_human_predictions.csv", low_memory=False)
    cut = cut[cut["session_id"].isin(trials["session_id"].unique())].sort_values(["session_id", "cutoff"]).reset_index(drop=True)
    readouts = cut[["session_id", "cutoff", "segment_type", "round_index"]].copy()
    readouts["theta"] = cut["theta_predicted"].map(lambda s: np.array(ast.literal_eval(s), dtype=float))
    for name in READOUTS:   # recent is absent at cutoff 1 (nothing demonstrated yet): kept as None
        readouts[name] = cut[f"{name}_betas"].map(lambda s: parse_betas(s) if isinstance(s, str) else None)

    per = trials.groupby("session_id")
    assert (trials["game_sequence"] != SOLO).all(), "inference CSVs hold social sessions only"
    assert per["round_index"].nunique().eq(N_ROUNDS).all() and per["round_index"].max().eq(N_ROUNDS - 1).all(), "two rounds per session"
    assert per.size().eq(N_ROUNDS * GAMES_PER_ROUND * trials_per_game).all() and per["anonymous_id"].nunique().eq(1).all()
    assert (trials["demonstrator_type"].isna() == (trials["game_index"] % 4).isin([1, 3])).all(), "demonstrator only on watch games"
    assert readouts.groupby("session_id")["cutoff"].apply(lambda c: sorted(c) == list(range(1, N_ROUNDS * SEGMENTS_PER_ROUND + 1))).all()
    K = len(readouts["theta"].iloc[0])
    assert all(a is None or a.shape == (K, 2) for name in READOUTS for a in readouts[name]) and readouts["expert"].notna().all()
    return Sessions(version, trials, readouts, trials_per_game, dropped)
