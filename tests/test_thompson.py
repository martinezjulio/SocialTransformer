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
    for j in (0, 6, 12):   # κ = 0.25, 1, 16
        p = p_win(LEGACY_KAPPA[j] * c["A"][rows])
        assert np.abs(p.sum(1) - 1).max() < 1e-6 and np.abs(np.log(np.clip(p, 1e-12, None)) - c["LOGP_U"][rows, j]).max() < 1e-6


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
    for i, (kj, wj, cj) in enumerate([(0, 0, 0), (12, 8, 5), (24, 15, 9)]):   # grid cells equal the exact quadrature
        exact = np.log(np.clip(p_win(KAPPA[kj] * (read + LAM_W[wj] * own_s + LAM_C[cj] * own_p)), 1e-12, None))
        assert np.abs(tab[idx, kj, wj, cj] - exact).max() < 1e-5
    lk, lw, lc = np.log(KAPPA), np.log(LAM_W), np.log(LAM_C[1:])
    q = np.column_stack([rng.uniform(lk[1], lk[-2], 40), rng.uniform(lw[1], lw[-2], 40), rng.uniform(lc[1], lc[-2], 40)])
    post = np.concatenate([np.exp(a) * (read + np.exp(b) * own_s + np.exp(c) * own_p) for a, b, c in q])
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
        exact = mix(p_win_all(np.exp(lk[16]) * (read + np.exp(lw[7]) * own_s + c * own_p)))
        ic, wc = lagrange4(LAM_C[:4], c)
        logp = np.einsum("i,j,k,nijka->na", wa, wb, wc, tab[idx][:, ia:ia + 4, ib:ib + 4, ic:ic + 4])
        e = np.abs(mix(np.exp(logp)) - exact)
        assert e[np.arange(len(idx)), act].mean() < 1e-4 and e.max() < 0.01, (c, e.mean(), e.max())
