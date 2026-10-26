"""The inference modules reproduce the 25 Oct reference run of run_inference.py on pilot E.

Reference: $OUTPUT_DATA_DIR/eval/pilot_E_2026_10_25 (cuda, 5 rollouts, seed 42). Exact string
equality holds on that device with one cutoff per forward; the batched readout and other
devices agree only to float precision, so the test runs on the reference device or skips.
"""
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from inference import sessions as se  # noqa: E402
from inference.learners import ThompsonLearner  # noqa: E402
from inference.readout import load_model, readout  # noqa: E402
from inference.simulate import betas_str, simulate_session  # noqa: E402
from run_inference import _rollout_seed  # noqa: E402
from utils import get_special_characters  # noqa: E402

N_SESSIONS = 3


@pytest.fixture(scope="module")
def ref(output_data_dir):
    d = output_data_dir / "eval" / "pilot_E_2026_10_25"
    if not (d / "data").exists():
        pytest.skip(f"{d} not present")
    cfg = yaml.safe_load((d / "metadata" / "inference_config.yaml").read_text())
    if cfg["device"] == "cuda" and not torch.cuda.is_available():
        pytest.skip("reference run used cuda")
    data_csv = Path(os.environ.get("EXPERIMENT_DATA_DIR", "")) / "pilot_E" / "preprocessed_participant_data.csv"   # the reference metadata predates the flat layout
    if not Path(cfg["checkpoint_path"]).exists() or not data_csv.exists():
        pytest.skip("reference checkpoint or preprocessed CSV not present")
    model, _, normalizer = load_model(local_path=cfg["checkpoint_path"], device=cfg["device"])
    shots = se.load_shots(str(data_csv))
    sids = sorted(shots.session_id.unique())[:N_SESSIONS]
    csv = {k: pd.read_csv(d / "data" / f"experiment_{k}_predictions.csv", low_memory=False) for k in ("cutoff_human", "cutoff_model", "trial")}
    return dict(model=model, normalizer=normalizer, shots=shots, sids=sids, seed=cfg["seed"], **csv)


def tokenise(sdf):
    info = se.convert_session_to_model_format(sdf, sorted(sdf.game_index.unique()))
    return info, se.encode_events_to_tensors(info["events"], 12, len(info["bandits_info"]), 3, get_special_characters())


def test_tokens_and_human_readouts_match_reference(ref):
    for sid in ref["sids"]:
        sdf = ref["shots"][ref["shots"].session_id == sid]
        info, t = tokenise(sdf)
        assert [s["type"] for s in info["segment_info"]] == ["bandit_sep", "demo", "learner", "demo", "learner", "query"] * 2
        assert t["special_ids"].shape == (1, len(sdf) + 4)   # one token per shot plus 2 BANDIT_SEP and 2 QUERY
        rows = ref["cutoff_human"][ref["cutoff_human"].session_id == sid].set_index("cutoff")
        ex, nv = info["demo_to_agent"]["expert"] - 1, info["demo_to_agent"]["novice"] - 1
        cutoffs = list(range(1, len(info["segment_info"]) + 1))
        batched = readout(ref["model"], t, info["segment_boundaries"], cutoffs, normalizer=ref["normalizer"])
        for i, c in enumerate(cutoffs):
            q = info["round_to_index"][info["segment_info"][c - 1]["round"]]
            single = readout(ref["model"], t, info["segment_boundaries"], [c], normalizer=ref["normalizer"])
            assert betas_str(single["alpha_demo"][0, q, ex], single["beta_demo"][0, q, ex]) == rows.loc[c, "expert_betas"]
            assert betas_str(single["alpha_demo"][0, q, nv], single["beta_demo"][0, q, nv]) == rows.loc[c, "novice_betas"]
            assert torch.allclose(batched["alpha_demo"][i], single["alpha_demo"][0], rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("beta_method", ["novice_betas", "expert_betas", "recent_betas"])
@pytest.mark.parametrize("own", ["carry", "reset"])
def test_thompson_rollouts_match_reference(ref, beta_method, own):
    trial, cut = ref["trial"], ref["cutoff_model"]
    sel = (trial.learner_segment_input == "model") & (trial.beta_method == beta_method) & (trial.own_experience == own) & (trial.rollout_id == 0)
    selc = (cut.beta_method == beta_method) & (cut.own_experience == own) & (cut.rollout_id == 0)
    for sid in ref["sids"]:
        sdf = ref["shots"][ref["shots"].session_id == sid]
        seed = _rollout_seed(ref["seed"], sid, 0)
        np_rng = np.random.default_rng(seed)
        sim = simulate_session(ref["model"], ref["normalizer"], sdf, ThompsonLearner(beta_method, own, np_rng), random.Random(seed), np_rng)
        play = sim[sim.game == "play"]
        want = trial[sel & (trial.session_id == sid) & (trial.game == "play")]
        assert play[["game_index", "position"]].values.tolist() == want[["game_index", "position"]].values.tolist()
        assert play.reward_generated.tolist() == want.reward_generated.astype(bool).tolist()
        pre_play = cut[selc & (cut.session_id == sid) & (cut.segment_type == "learner")].cutoff.values - 1
        entering = cut[selc & (cut.session_id == sid)].set_index("cutoff").loc[pre_play]
        assert play.groupby("game_index", sort=False).expert_betas.first().tolist() == entering.expert_betas.tolist()
        assert play.groupby("game_index", sort=False).novice_betas.first().tolist() == entering.novice_betas.tolist()
