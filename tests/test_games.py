"""The per-play-game table reproduces the pilot E cell means quoted in the rebuild plan.

Anchors are per-participant means then the mean over participants, quoted at 2 dp from legacy
output printed at 3 dp (so up to 0.0055 off); the 4-dp lock holds today's values exactly.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from analysis.games import play_games  # noqa: E402

ANCHOR = {("after expert", 1): (0.62, 0.65, 0.99), ("after expert", 2): (0.64, 0.63, 0.88), ("after novice", 1): (0.52, 0.50, 1.50),
          ("after novice", 2): (0.56, 0.56, 1.14), ("solo", 1): (0.56, 0.56, 1.27), ("solo", 2): (0.54, 0.54, 1.06),
          ("solo slot 4", 1): (0.57, 0.56, 1.19), ("solo slot 4", 2): (0.56, 0.57, 0.99)}
LOCK = {("after expert", 1): (0.6221, 0.6469, 0.9903), ("after expert", 2): (0.6418, 0.6338, 0.8833), ("after novice", 1): (0.5174, 0.5044, 1.5019),
        ("after novice", 2): (0.5618, 0.5592, 1.1372), ("solo", 1): (0.5577, 0.5548, 1.2651), ("solo", 2): (0.5380, 0.5439, 1.0598),
        ("solo slot 4", 1): (0.5664, 0.5570, 1.1925), ("solo slot 4", 2): (0.5548, 0.5658, 0.9851)}
OUTCOMES = ["payoff", "reward", "entropy"]


@pytest.fixture(scope="module")
def table():
    root = os.environ.get("EXPERIMENT_DATA_DIR")
    p = Path(root or "") / "transfer_bandits_choice_textbox" / "pilot_E" / "preprocessed_participant_data.csv"
    if not root or not p.exists():
        pytest.skip(f"{p} not present")
    return play_games(pd.read_csv(p, dtype={"timestamp": str}, low_memory=False))


def test_row_counts(table):
    assert len(table) == 608 and (table.condition != "solo").sum() == 304 and (table.condition == "solo").sum() == 304
    assert table.exploratory.sum() == 152 and ((table.condition == "solo") & ~table.exploratory).sum() == 152
    assert table.participant.nunique() == 38 and table.session.nunique() == 114


def cell_means(table):
    sub = table[~table.exploratory].copy()
    sub["cell"] = np.where(sub.condition == "solo", "solo", "after " + sub.demonstrator)
    slot4 = sub[(sub.condition == "solo") & (sub.slot == 4)].assign(cell="solo slot 4")
    both = pd.concat([sub, slot4])
    return both.groupby(["cell", "round", "participant"])[OUTCOMES].mean().groupby(["cell", "round"]).mean()


def test_cell_means_match_plan_anchors(table):
    means = cell_means(table)
    for key, anchor in ANCHOR.items():
        assert np.abs(means.loc[key].values - anchor).max() <= 0.006, (key, means.loc[key].round(4).tolist(), anchor)


def test_cell_means_regression_lock(table):
    means = cell_means(table)
    for key, locked in LOCK.items():
        assert np.allclose(means.loc[key].values, locked, atol=5e-5), (key, means.loc[key].round(4).tolist(), locked)
