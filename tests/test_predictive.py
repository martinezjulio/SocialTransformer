"""The simulator-frame path builds the same TrialData as the inference-CSV loader; the hierarchy sampler is calibrated."""
import os
import pickle
import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from analysis.data import load_sessions, trial_data  # noqa: E402
from analysis.predictive import PER_PARTICIPANT, sample_participants, simulated_trial_data  # noqa: E402


@pytest.fixture(scope="module")
def eval_dir():
    root = os.environ.get("OUTPUT_DATA_DIR")
    d = Path(root or "") / "eval" / "pilot_E"
    if not root or not (d / "data" / "experiment_trial_predictions.csv").exists():
        pytest.skip("pilot E inference CSVs not present")
    return d


def test_human_rows_round_trip_through_the_simulator_path(eval_dir):
    loaded = trial_data(load_sessions(eval_dir, "pilot_E"))
    rows = pd.read_csv(eval_dir / "data" / "experiment_trial_predictions.csv", low_memory=False)
    rows = rows[(rows["learner_segment_input"] == "human") & rows["session_id"].isin(np.unique(loaded.session))].drop(columns=["expert_betas", "novice_betas"])
    cut = pd.read_csv(eval_dir / "data" / "experiment_cutoff_human_predictions.csv", low_memory=False)
    within = (cut["cutoff"] - 1) % 6 + 1                       # the trial CSV's per-row readout is the post-game one; the simulator
    cut = cut[within.isin([2, 4])].assign(game_index=(cut["cutoff"] - 1) // 6 * 4 + within - 1)   # records the one entering the game
    rows = rows.merge(cut[["session_id", "game_index", "expert_betas", "novice_betas"]], how="left")
    sim = simulated_trial_data([g for _, g in rows.groupby("session_id", sort=False)], "pilot_E")
    for f in loaded.__dataclass_fields__:
        a, b = getattr(loaded, f), getattr(sim, f)
        if f == "read":
            for k in a:
                assert np.array_equal(a[k], b[k]), k
        elif f == "people":
            assert a == b
        else:
            assert np.array_equal(a, b), f


def test_sample_participants_matches_the_draws_hierarchy():
    fits = glob(str(Path(os.environ.get("OUTPUT_DATA_DIR", "")) / "analysis/pilot_E/choice/fits/Switch-by-round_*.pkl"))
    if not fits:
        pytest.skip("no cached Switch-by-round fit")
    post = pickle.load(open(sorted(fits)[-1], "rb")).posterior
    people = sample_participants(post, draw=7, n=4000, rng=np.random.default_rng(0))
    flat = {s: v.reshape(-1, *v.shape[2:])[7] for s, v in post.items()}
    for s in PER_PARTICIPANT:
        x = np.array([p[s] for p in people])
        assert abs(x.mean() - flat[f"mu_{s}"]) < 4 * flat[f"tau_{s}"] / np.sqrt(len(x)) and abs(x.std() / flat[f"tau_{s}"] - 1) < 0.05, s
    assert all(p["delta_N"] == float(flat["delta_N"]) and p["lambda_c"] == float(flat["lambda_c"]) for p in people)
    assert not any(k.startswith(("z_", "mu_", "tau_", "omega_")) for k in people[0])
