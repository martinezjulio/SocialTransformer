"""Thompson win probabilities and the per-trial log-policy tables the choice models interpolate.

``p_win`` is the legacy quadrature (notebooks/legacy_analysis/policy.py) unchanged: P(arm k draws the
largest payoff) = ∫ f_k Π_{j≠k} F_j dx on the logit scale, scipy betainc, 4096 nodes, exact to ~1e-7.
``table`` evaluates it for one belief source on the dense (κ, λ_w, λ_c) grid: the posterior entering a
shot is κ · (read + λ_w · own_seg + λ_c · own_prev), stored as floored log p in float32,
shape (N, |κ|, |λ_w|, |λ_c|, K). Tables are cached by a hash of their inputs; a build is
~2.7k rows/s on 16 threads (pilot: ~45 min per source; full experiment: hours, once).
The 5.6 benchmark chose tables over an in-sampler quadrature (see the plan's Runtime section).
"""

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.special import betainc, betaln, expit, log_expit

KAPPA = np.geomspace(0.125, 16, 29)                       # ratio 2^¼; 1.0 on the grid; floor below the least trusting pilot participants
LAM_W = np.geomspace(0.1, 16, 16)                         # within-segment own weight
LAM_C = np.concatenate([[0.0], np.geomspace(0.25, 4, 9)])  # carried own weight; 0 = reset learner
P_FLOOR = 1e-12                                           # log p stored without the lapse; keeps -inf out


def p_win(ab: np.ndarray, X: int = 4096, L: float = 80.0) -> np.ndarray:
    """ab (M, K, 2) Beta counts → (M, K) win probabilities. x = sigmoid(u) removes the endpoint singularity; F
    near 1 comes from the complement so an arm's own tail is not double counted. Beyond u = L the other arms'
    CDFs are ~1, so an arm's density mass past the window (noticeable only for counts below ~0.2) is added
    analytically as its win probability there."""
    u = np.linspace(-L, L, X)
    a, b = ab[..., 0, None], ab[..., 1, None]
    F = betainc(a, b, expit(u))
    S = np.clip(betainc(b, a, expit(-u)), None, 1 - 1e-16)
    logF = np.where(F < 0.5, np.log(np.clip(F, 1e-300, None)), np.log1p(-S))
    logf = a * log_expit(u) + b * log_expit(-u) - betaln(a, b)
    body = np.trapezoid(np.exp(logf + logF.sum(-2, keepdims=True) - logF), u, axis=-1)
    return body + S[..., -1] * np.exp(logF[..., -1].sum(-1, keepdims=True) - logF[..., -1])


def p_win_all(ab: np.ndarray, chunk: int = 128, workers: int = 16) -> np.ndarray:
    """p_win in chunks on a thread pool (the scipy ufuncs release the GIL); bitwise equal to one call."""
    with ThreadPoolExecutor(workers) as ex:
        return np.concatenate(list(ex.map(p_win, [ab[i:i + chunk] for i in range(0, len(ab), chunk)])))


def table_path(name: str, read, own_seg, own_prev, cache_dir, kappa=KAPPA, lam_w=LAM_W, lam_c=LAM_C) -> Path:
    """Where ``table`` caches this source and grid: ``<cache_dir>/<name>_<hash of the inputs>.npy``."""
    key = hashlib.sha256(b"".join(np.ascontiguousarray(x).tobytes() for x in (read, own_seg, own_prev, kappa, lam_w, lam_c))).hexdigest()[:16]
    return Path(cache_dir) / f"{name}_{key}.npy"


def block_path(path: Path, block: int, n_blocks: int) -> Path:
    return path.with_name(f"{path.stem}.block{block:03d}of{n_blocks:03d}.npy")


def build(read, own_seg, own_prev, kappa, lam_w, lam_c, workers) -> np.ndarray:
    """Log-policy table (N, |κ|, |λ_w|, |λ_c|, K) for these trials, float32, floored at P_FLOOR."""
    N, K, _ = read.shape
    k, w, c = kappa.reshape(-1, 1, 1, 1, 1, 1), lam_w.reshape(1, -1, 1, 1, 1, 1), lam_c.reshape(1, 1, -1, 1, 1, 1)
    post = k * (read + w * own_seg + c * own_prev)                                       # (|κ|, |λ_w|, |λ_c|, N, K, 2)
    p = p_win_all(post.reshape(-1, K, 2), workers=workers).reshape(len(kappa), len(lam_w), len(lam_c), N, K)
    assert np.abs(p.sum(-1) - 1).max() < 1e-6, "quadrature lost mass"
    return np.log(np.clip(p, P_FLOOR, None)).transpose(3, 0, 1, 2, 4).astype(np.float32)


def table(name: str, read: np.ndarray, own_seg: np.ndarray, own_prev: np.ndarray, cache_dir: Path, kappa: np.ndarray = KAPPA,
          lam_w: np.ndarray = LAM_W, lam_c: np.ndarray = LAM_C, workers: int = 16, block: int = None, n_blocks: int = 1) -> np.ndarray:
    """Cached log-policy table for a belief source; built on a cache miss. The trials split into ``n_blocks`` contiguous
    blocks: ``block=b`` builds, caches and returns block b alone (one cluster array task); ``block=None`` returns the
    full table, assembling it from the cached blocks (building any that are missing) and then deleting them."""
    path = table_path(name, read, own_seg, own_prev, cache_dir, kappa, lam_w, lam_c)
    if path.exists():
        return np.load(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if block is not None:
        rows, bp = np.array_split(np.arange(len(read)), n_blocks)[block], block_path(path, block, n_blocks)
        if not bp.exists():
            np.save(bp, build(read[rows], own_seg[rows], own_prev[rows], kappa, lam_w, lam_c, workers))
        return np.load(bp)
    args = (name, read, own_seg, own_prev, cache_dir, kappa, lam_w, lam_c, workers)
    out = build(read, own_seg, own_prev, kappa, lam_w, lam_c, workers) if n_blocks == 1 else np.concatenate(
        [table(*args, block=b, n_blocks=n_blocks) for b in range(n_blocks)])
    np.save(path, out)
    for b in range(n_blocks if n_blocks > 1 else 0):
        block_path(path, b, n_blocks).unlink()
    return out
