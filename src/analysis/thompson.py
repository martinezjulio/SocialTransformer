"""Thompson win probabilities and the per-trial log-policy tables the choice models interpolate.

``p_win`` is the legacy quadrature (notebooks/legacy_analysis/policy.py) unchanged: P(arm k draws the
largest payoff) = ∫ f_k Π_{j≠k} F_j dx on the logit scale, scipy betainc, 4096 nodes, exact to ~1e-7.
``table`` evaluates it for one belief source on the dense (κ, λ_w, λ_c) grid: the posterior entering a
shot is prior + κ · social + λ_w · own_seg + λ_c · own_prev (the registered step 1: κ scales the social
evidence only; prior = 0 for a readout, Beta(1, 1) for a count of pulls), stored as floored log p in float32,
shape (N, |κ|, |λ_w|, |λ_c|, K). Tables are cached by a hash of their inputs; a build is
~2.7k rows/s on 16 threads (pilot: ~45 min per source; full experiment: hours, once).
The 5.6 benchmark chose tables over an in-sampler quadrature (see the plan's Runtime section).
"""

import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.special import betainc, betaln, expit, log_expit

from analysis.data import atomic_write

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


def cpu_workers(limit: int = 16) -> int:
    """Threads for the CPU builder: the cores this process may use (a Slurm task's allocation, not the node), at most ``limit``."""
    return max(1, min(limit, len(os.sched_getaffinity(0))))


def p_win_all(ab: np.ndarray, chunk: int = 128, workers: int = None) -> np.ndarray:
    """p_win in chunks on a thread pool (the scipy ufuncs release the GIL); bitwise equal to one call."""
    with ThreadPoolExecutor(workers or cpu_workers()) as ex:
        return np.concatenate(list(ex.map(p_win, [ab[i:i + chunk] for i in range(0, len(ab), chunk)])))


def p_win_gpu(ab: np.ndarray, X: int = None, L: float = 80.0) -> np.ndarray:
    """``p_win`` with the arms' CDFs from a cumulative Simpson rule on the density (O(du⁴)) instead of scipy's
    betainc, in float64 on the GPU (torch, imported here only); the CDF at the window's edges comes from scipy so the
    tails match. The integrand's width shrinks with the counts of every overlapping arm, and the Simpson error scales
    as (du / width)⁴, so rows are processed in chunks of increasing total count (all arms) and each chunk's node count
    grows with √(total) from 4097 at a total of 1000: |Δp| ≈ 1e-5 and row mass within 2e-5 against ``p_win`` on the
    pilot grid (largest total 4100) and on rows whose twelve arms all sit near 300; calibrated for totals up to 20 000.
    For simulated datasets only; human fits keep ``p_win``."""
    import torch
    total = ab.sum((1, 2))
    assert total.max() <= 20000, "calibrated for row totals up to 20 000"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lo, hi = betainc(ab[..., 0], ab[..., 1], expit(-L)), betainc(ab[..., 1], ab[..., 0], expit(-L))   # exact edge masses (M, K)
    order, out, i = np.argsort(total), np.empty(ab.shape[:2]), 0
    while i < len(ab):
        n = X or 2 * int(2048 * np.sqrt(max(total[order[i:i + 1024]].max(), 1000.0) / 1000.0)) + 1
        rows = order[i:i + max(32, 1024 * 4097 // n)]
        u = torch.linspace(-L, L, n, dtype=torch.float64, device=dev)
        du = float(u[1] - u[0])
        w = torch.full((n,), du, dtype=torch.float64, device=dev)
        w[0] = w[-1] = du / 2
        t = torch.as_tensor(ab[rows], dtype=torch.float64, device=dev)
        a, b = t[..., 0, None], t[..., 1, None]
        logf = a * torch.nn.functional.logsigmoid(u) + b * torch.nn.functional.logsigmoid(-u) - (torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b))
        f = logf.exp()
        fe, fo, fn = f[..., 0:-2:2], f[..., 1:-1:2], f[..., 2::2]                     # pairs of intervals
        Fe = torch.cat([torch.zeros_like(f[..., :1]), torch.cumsum(du / 3 * (fe + 4 * fo + fn), -1)], -1)   # even nodes
        Fo = Fe[..., :-1] + du / 12 * (5 * fe + 8 * fo - fn)                                              # odd nodes
        F = torch.cat([torch.stack([Fe[..., :-1], Fo], -1).flatten(-2), Fe[..., -1:]], -1) + torch.as_tensor(lo[rows], device=dev)[..., None]
        logF = torch.where(F < 0.5, F.clamp(min=1e-300).log(), torch.log1p(-(1 - F).clamp(min=1e-16)))
        body = (torch.exp(logf + logF.sum(-2, keepdim=True) - logF) * w).sum(-1)
        tail = torch.as_tensor(hi[rows], device=dev) * torch.exp(logF[..., -1].sum(-1, keepdim=True) - logF[..., -1])
        out[rows], i = (body + tail).cpu().numpy(), i + len(rows)
    return out


def table_path(name: str, prior, social, own_seg, own_prev, cache_dir, kappa=KAPPA, lam_w=LAM_W, lam_c=LAM_C) -> Path:
    """Where ``table`` caches this source and grid: ``<cache_dir>/<name>_<hash of the inputs>.npy``."""
    key = hashlib.sha256(b"".join(np.ascontiguousarray(x).tobytes() for x in (prior, social, own_seg, own_prev, kappa, lam_w, lam_c))).hexdigest()[:16]
    return Path(cache_dir) / f"{name}_{key}.npy"


def block_path(path: Path, block: int, n_blocks: int) -> Path:
    return path.with_name(f"{path.stem}.block{block:03d}of{n_blocks:03d}.npy")


def build(prior, social, own_seg, own_prev, kappa, lam_w, lam_c, workers, builder="scipy") -> np.ndarray:
    """Log-policy table (N, |κ|, |λ_w|, |λ_c|, K) for these trials, float32, floored at P_FLOOR."""
    N, K, _ = social.shape
    k, w, c = kappa.reshape(-1, 1, 1, 1, 1, 1), lam_w.reshape(1, -1, 1, 1, 1, 1), lam_c.reshape(1, 1, -1, 1, 1, 1)
    post = prior + k * social + w * own_seg + c * own_prev                               # (|κ|, |λ_w|, |λ_c|, N, K, 2)
    p = (p_win_all(post.reshape(-1, K, 2), workers=workers) if builder == "scipy" else p_win_gpu(post.reshape(-1, K, 2)))
    p = p.reshape(len(kappa), len(lam_w), len(lam_c), N, K)
    assert np.abs(p.sum(-1) - 1).max() < (1e-6 if builder == "scipy" else 1e-4), "quadrature lost mass"
    return np.log(np.clip(p, P_FLOOR, None)).transpose(3, 0, 1, 2, 4).astype(np.float32)


def table(name: str, prior: np.ndarray, social: np.ndarray, own_seg: np.ndarray, own_prev: np.ndarray, cache_dir: Path, kappa: np.ndarray = KAPPA,
          lam_w: np.ndarray = LAM_W, lam_c: np.ndarray = LAM_C, workers: int = None, block: int = None, n_blocks: int = 1,
          builder: str = "scipy") -> np.ndarray:
    """Cached log-policy table for a belief source; built on a cache miss. The trials split into ``n_blocks`` contiguous
    blocks: ``block=b`` builds, caches and returns block b alone (one cluster array task); ``block=None`` returns the
    full table, assembling it from the cached blocks (building any that are missing) and then deleting them.
    ``builder='gpu'`` uses ``p_win_gpu`` (1e-4 gate) and caches under ``<name>_gpu``; human fits keep the scipy builder."""
    name = name if builder == "scipy" else f"{name}_gpu"
    path = table_path(name, prior, social, own_seg, own_prev, cache_dir, kappa, lam_w, lam_c)
    if path.exists():
        return np.load(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if block is not None:
        rows, bp = np.array_split(np.arange(len(social)), n_blocks)[block], block_path(path, block, n_blocks)
        if not bp.exists():
            atomic_write(bp, lambda t: np.save(t, build(prior[rows], social[rows], own_seg[rows], own_prev[rows], kappa, lam_w, lam_c, workers, builder)))
        return np.load(bp)
    args = (name, prior, social, own_seg, own_prev, cache_dir, kappa, lam_w, lam_c, workers)
    out = build(prior, social, own_seg, own_prev, kappa, lam_w, lam_c, workers, builder) if n_blocks == 1 else np.concatenate(
        [table(*args, block=b, n_blocks=n_blocks, builder=builder) for b in range(n_blocks)])
    atomic_write(path, lambda t: np.save(t, out))
    for b in range(n_blocks if n_blocks > 1 else 0):
        block_path(path, b, n_blocks).unlink()
    return out
