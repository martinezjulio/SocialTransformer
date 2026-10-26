"""`TrialData` reproduces the legacy trial frame, and the current CSVs give a well-formed table.

Legacy cache: $OUTPUT_DATA_DIR/legacy_analysis/cached_data/policy_logp_pilot_E_*.pkl, built from the
19 Oct inference ($OUTPUT_DATA_DIR/eval/pilot_E_2026_10_19), whose shot order was scrambled in 5
sessions; so equality is checked on a TrialData built from that same directory.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from analysis.data import load_sessions, trial_data  # noqa: E402

LEGACY_CACHE = "legacy_analysis/cached_data/policy_logp_pilot_E_wandb_0i8yhqm7_epoch_119_best.pkl"


def build(output_data_dir, version):
    d = output_data_dir / "eval" / version
    if not (d / "data").exists():
        pytest.skip(f"{d} not present")
    return trial_data(load_sessions(d, version))


def test_equals_legacy_frame(output_data_dir):
    if not (output_data_dir / LEGACY_CACHE).exists():
        pytest.skip("legacy cache not present")
    cache = pd.read_pickle(output_data_dir / LEGACY_CACHE)
    T, tr, A, IX = build(output_data_dir, "pilot_E_2026_10_19"), cache["trial"], cache["A"], cache["IX"]
    for ours, legacy in [(T.session, tr.sid), (T.act, tr.act), (T.prev_arm, tr.prev_arm), (T.prev_rew, tr.prev_rew),
                         (T.round_pm + 1.5, tr.rnd), (T.seg, tr.seg), (T.last_exp, tr.last_exp), (T.own_seg, cache["S_SEG"]), (T.own_prev, cache["S_PREV"])]:
        assert np.array_equal(np.asarray(ours), np.asarray(legacy))
    for p in ("expert", "novice", "recent"):   # readout + own counts = the legacy posterior, both learners
        assert np.array_equal(T.read[p] + T.own_prev + T.own_seg, A[IX[p, "carry"]]) and np.array_equal(T.read[p] + T.own_seg, A[IX[p, "reset"]])
    assert np.array_equal(1.0 + T.vis_round, A[IX["naive", "carry"]])
    m = pd.crosstab(T.part, tr.part.values)   # participant index is a relabelling of the legacy integer ids
    assert (m.gt(0).sum(1) == 1).all() and (m.gt(0).sum(0) == 1).all()


def test_current_structure(output_data_dir):
    T = build(output_data_dir, "pilot_E")
    N, K = T.theta.shape
    assert N == 1824 and K == 12 and len(T.people) == 38 and T.read["expert"].shape == (N, K, 2)
    assert (T.prev_arm == -1).sum() == N / 6 and np.array_equal(T.prev_arm == -1, ~T.own_seg.any((1, 2)))
    assert not T.own_prev[T.seg % 2 == 1].any(), "nothing carries into a round's first play segment"
    assert np.array_equal(~T.expert_seen, ~T.expert_first & (T.seg % 2 == 1)) and (~T.expert_seen).sum() == 456
    assert (T.vis_expert[~T.expert_seen] == 0).all() and np.array_equal(T.vis_last[T.last_exp], T.vis_expert[T.last_exp])
    assert T.vis_last.sum((1, 2)).min() == 6 and set(np.unique(T.round_pm)) == {-0.5, 0.5} and set(np.unique(T.seg)) == {1, 2, 3, 4}
    assert ((T.theta >= 0) & (T.theta <= 1)).all() and 0.5 < T.theta[np.arange(N), T.act].mean() < 0.7
