"""The Thompson quadrature equals the legacy tables, and cubic interpolation on the dense grid is accurate.

Interpolation reference: 4-point Lagrange (Catmull-Rom style) per axis, in log κ, log λ_w and log λ_c
(λ_c = 0 is its own rung, never interpolated), as a tensor product; the choice models reimplement it in JAX.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from analysis.data import load_sessions, trial_data  # noqa: E402
from analysis.thompson import KAPPA, LAM_C, LAM_W, p_win, p_win_all  # noqa: E402

LEGACY_CACHE = "legacy_analysis/cached_data/policy_logp_pilot_E_wandb_0i8yhqm7_epoch_119_best.pkl"
LEGACY_KAPPA = np.geomspace(0.25, 16, 13)
LAPSE, K = 0.02, 12


def mix(p):   # the log-likelihood the choice models use
    return np.log((1 - LAPSE) * p + LAPSE / K)


def lagrange4(x, xq):
    """Weights (4,) of the cubic through the 4 knots x (sorted) at xq; (i, w) with i the first knot index."""
    i = int(np.clip(np.searchsorted(x, xq) - 2, 0, len(x) - 4))
    xs = x[i:i + 4]
    return i, np.array([np.prod([(xq - xs[m]) / (xs[j] - xs[m]) for m in range(4) if m != j]) for j in range(4)])


def test_p_win_equals_legacy_and_sums_to_one(output_data_dir):
    if not (output_data_dir / LEGACY_CACHE).exists():
        pytest.skip("legacy cache not present")
    c = pd.read_pickle(output_data_dir / LEGACY_CACHE)
    rows = np.random.default_rng(0).choice(len(c["A"]), 300, replace=False)
    for j in (0, 6, 12):   # κ = 0.25, 1, 16; compared in probability: the analytic tail term adds ~1e-8 of mass the
        p = p_win(LEGACY_KAPPA[j] * c["A"][rows])   # legacy quadrature lost, which is a large log difference only at the 1e-12 floor
        assert np.abs(p.sum(1) - 1).max() < 1e-6 and np.abs(p - np.exp(c["LOGP_U"][rows, j])).max() < 1e-6


def test_cubic_interpolation_on_dense_grid(output_data_dir):
    files = sorted((output_data_dir / "analysis" / "pilot_E" / "tables").glob("expert_*.npy"))
    if not files or not (output_data_dir / "eval" / "pilot_E" / "data").exists():
        pytest.skip("cached expert table or pilot_E inference not present")
    T = trial_data(load_sessions(output_data_dir / "eval" / "pilot_E", "pilot_E"))
    tab = np.load(files[-1], mmap_mode="r")
    assert tab.shape == (len(T.act), len(KAPPA), len(LAM_W), len(LAM_C), K) and tab.dtype == np.float32
    rng = np.random.default_rng(1)
    idx = rng.choice(len(T.act), 60, replace=False)
    read, own_s, own_p, act = T.read["expert"][idx], T.own_seg[idx], T.own_prev[idx], T.act[idx]
    for i, (kj, wj, cj) in enumerate([(0, 0, 0), (12, 8, 5), (24, 15, 9)]):   # grid cells equal the exact quadrature; registered
        exact = np.log(np.clip(p_win(KAPPA[kj] * read + LAM_W[wj] * own_s + LAM_C[cj] * own_p), 1e-12, None))   # belief step (5.23): κ scales the readout only
        assert np.abs(tab[idx, kj, wj, cj] - exact).max() < 1e-5
    lk, lw, lc = np.log(KAPPA), np.log(LAM_W), np.log(LAM_C[1:])
    q = np.column_stack([rng.uniform(lk[1], lk[-2], 40), rng.uniform(lw[1], lw[-2], 40), rng.uniform(lc[1], lc[-2], 40)])
    post = np.concatenate([np.exp(a) * read + np.exp(b) * own_s + np.exp(c) * own_p for a, b, c in q])
    exact = mix(p_win_all(post)).reshape(40, len(idx), K)
    sub = tab[idx][:, :, :, 1:]   # nonzero λ_c rungs only
    err = []
    for (a, b, c), ex in zip(q, exact):
        (ia, wa), (ib, wb), (ic, wc) = lagrange4(lk, a), lagrange4(lw, b), lagrange4(lc, c)
        block = sub[:, ia:ia + 4, ib:ib + 4, ic:ic + 4]
        logp = np.einsum("i,j,k,nijka->na", wa, wb, wc, block)
        err.append(mix(np.exp(logp)) - ex)
    err = np.abs(np.array(err))                                  # (40, 60, K)
    chosen = err[:, np.arange(len(idx)), act]
    assert chosen.mean() < 1e-4 and err.max() < 0.01, (chosen.mean(), err.max())
    # below the first nonzero λ_c knot the model uses a cubic on the linear axis through the zero knot
    (ia, wa), (ib, wb) = lagrange4(lk, lk[16]), lagrange4(lw, lw[7])   # κ = 1, λ_w on a knot: only the λ_c axis is interpolated
    for c in (0.03, 0.137, 0.2):
        exact = mix(p_win_all(np.exp(lk[16]) * read + np.exp(lw[7]) * own_s + c * own_p))
        ic, wc = lagrange4(LAM_C[:4], c)
        logp = np.einsum("i,j,k,nijka->na", wa, wb, wc, tab[idx][:, ia:ia + 4, ib:ib + 4, ic:ic + 4])
        e = np.abs(mix(np.exp(logp)) - exact)
        assert e[np.arange(len(idx)), act].mean() < 1e-4 and e.max() < 0.01, (c, e.mean(), e.max())


def test_table_assembles_from_cached_blocks(tmp_path):
    from analysis.thompson import block_path, table, table_path
    rng = np.random.default_rng(0)
    read, own_seg, own_prev = (rng.gamma(2.0, 2.0, size=(10, K, 2)) for _ in range(3))
    grid = dict(kappa=np.array([0.5, 2.0]), lam_w=np.array([0.5, 1.0]), lam_c=np.array([0.0, 1.0]))
    args = ("toy", np.zeros_like(read), read, own_seg, own_prev, tmp_path)
    direct = table("toy", np.zeros_like(read), read, own_seg, own_prev, tmp_path / "direct", **grid)
    blocks = [table(*args, **grid, block=b, n_blocks=3) for b in range(3)]
    assert [len(b) for b in blocks] == [4, 3, 3] and all(block_path(table_path(*args, **grid), b, 3).exists() for b in range(3))
    assembled = table(*args, **grid, n_blocks=3)
    assert assembled.shape == (10, 2, 2, 2, K) and np.array_equal(assembled, direct) and np.array_equal(np.concatenate(blocks), direct)
    assert table_path(*args, **grid).exists() and not any(block_path(table_path(*args, **grid), b, 3).exists() for b in range(3))
    assert np.array_equal(table(*args, **grid), direct), "the assembled table is served from the cache"


def test_gpu_builder_matches_scipy(tmp_path):
    from analysis.thompson import p_win_gpu, table, table_path
    rng = np.random.default_rng(1)
    ab = np.concatenate([rng.gamma(1.5, s, size=(40, K, 2)) + 0.1 for s in (0.1, 10, 300)] + [rng.uniform(90, 110, size=(40, K, 2))])
    p, g = p_win(ab), p_win_gpu(ab)   # counts 0.1 .. ~3500, plus the hard case: every arm near 100
    assert np.abs(g - p).max() < 1e-5 and np.abs(g.sum(-1) - 1).max() < 5e-5, (np.abs(g - p).max(), np.abs(g.sum(-1) - 1).max())
    read, own_seg, own_prev = (rng.gamma(2.0, 2.0, size=(6, K, 2)) for _ in range(3))
    grid = dict(kappa=np.array([0.5, 16.0]), lam_w=np.array([1.0]), lam_c=np.array([0.0, 4.0]))
    exact, fast = (table("toy", np.zeros_like(read), read, own_seg, own_prev, tmp_path, **grid, builder=b) for b in ("scipy", "gpu"))
    assert table_path("toy_gpu", np.zeros_like(read), read, own_seg, own_prev, tmp_path, **grid).exists() and fast.shape == exact.shape
    big = exact > np.log(1e-4)
    assert np.abs(fast[big] - exact[big]).max() < 5e-3, "log p within 5e-3 where p > 1e-4 (the lapse floor is 1.7e-3)"


def test_expert_table_on_legacy_grid_equals_legacy_both_table(output_data_dir, tmp_path):
    """The registered belief step: our table at the legacy knots reproduces the legacy BOTH table (κ · readout + λ_w · own_seg
    + λ_c · own_prev) on trials of the sessions whose shot order the 19 Oct inference did not scramble."""
    from analysis.policies import belief
    from analysis.thompson import table
    both = output_data_dir / "legacy_analysis/cached_data/policy_both_logp_pilot_E_wandb_0i8yhqm7_epoch_119_best.pkl"
    if not (output_data_dir / LEGACY_CACHE).exists() or not both.exists() or not (output_data_dir / "eval/pilot_E/data/experiment_trial_predictions.csv").exists():
        pytest.skip("legacy caches or pilot E CSVs not present")
    T = trial_data(load_sessions(output_data_dir / "eval" / "pilot_E", "pilot_E"))
    legacy, trial = pd.read_pickle(both), pd.read_pickle(output_data_dir / LEGACY_CACHE)["trial"]
    sid, theirs_all = trial["sid"].to_numpy(), legacy["BOTH_LOGP_U"][legacy["ix_b"]["expert"]]
    prior, social = belief(T, "expert")
    one = np.exp(np.log(np.clip(p_win_all(prior + 2.0 * social + 0.1 * T.own_seg), 1e-12, None))) - np.exp(theirs_all[:, 6, 0, 0].astype(float))
    scrambled = pd.Series(np.abs(one).max(1)).groupby(sid).max().pipe(lambda d: d.index[d > 1e-6])
    assert len(scrambled) == 5, "the five sessions whose shot order the 19 Oct inference scrambled (four differ in actions, one in its readouts)"
    rows = np.random.default_rng(1).choice(np.flatnonzero(~np.isin(sid, scrambled) & (T.own_prev.sum((1, 2)) > 0)), 40, replace=False)   # second play games: own terms active
    ours = table("legacy_grid", prior[rows], social[rows], T.own_seg[rows], T.own_prev[rows], tmp_path, LEGACY_KAPPA, np.array([0.1, 0.25, 0.5, 1, 2, 4, 8, 16]), np.array([0., 0.25, 0.5, 1, 2, 4]))
    theirs = legacy["BOTH_LOGP_U"][legacy["ix_b"]["expert"][rows]]
    assert ours.shape == theirs.shape == (40, 13, 8, 6, K)
    assert np.abs(np.exp(ours.astype(float)) - np.exp(theirs.astype(float))).max() < 1e-6
