# %% fitting
"""Fit one choice-model spec with NUTS, gate its diagnostics, climb a refit ladder on failure, cache the result.

Gate (meta-plan 3.4): R̂ < 1.01 on every site, bulk and tail ESS ≥ 400, zero divergences. The ladder starts at
4 chains × (1000 warm-up + 1000 draws) and raises the budget and target acceptance on failure; a fit that fails
the last rung is returned with ``passed=False`` and never reported. Pointwise log-likelihoods are kept for
PSIS-LOO (``compare.py``). Models with nothing to sample (Random) get their log-likelihood directly.
``run_many`` fits several registry models at once, one spawned process each sharing the host's devices.
"""

import multiprocessing as mp
import os
import pickle
import re
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import arviz as az

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")   # before JAX's backend starts: on unified-memory boxes the 75% default takes the host's RAM too
import jax  # noqa: E402
import numpy as np  # noqa: E402
from numpyro import handlers
from numpyro.infer import MCMC, NUTS, log_likelihood

from analysis.data import TrialData, load_sessions, trial_data, atomic_write
from analysis.policies import REGISTRY, ModelSpec, model

GATE = dict(rhat=1.01, ess=400, divergences=0)
LADDER = (dict(num_warmup=1000, num_samples=1000, target_accept_prob=0.9), dict(num_warmup=2000, num_samples=2000, target_accept_prob=0.95),
          dict(num_warmup=4000, num_samples=4000, target_accept_prob=0.99))
CHAINS = 4


@dataclass
class Fit:
    name: str
    posterior: dict        # site -> (chains, draws, ...) arrays, deterministic sites included
    log_lik: np.ndarray    # (chains, draws, N) float32 pointwise log-likelihood of the choices
    diagnostics: dict      # rhat_max, ess_bulk_min, ess_tail_min, divergences, rung, seconds
    passed: bool


def fit_once(spec: ModelSpec, T: TrialData, tab, tab2, rung: int, seed: int, chain_method: str = "vectorized") -> Fit:
    """One NUTS run at ladder rung ``rung``; the diagnostics are computed on every site."""
    t0, args = time.time(), (spec, T, tab, True, tab2)
    sites = [k for k, v in handlers.trace(handlers.seed(model, seed)).get_trace(*args).items() if v["type"] == "sample" and not v["is_observed"]]
    if not sites:   # nothing to fit: the chance baseline
        ll = np.asarray(handlers.trace(handlers.seed(model, seed)).get_trace(*args)["a"]["fn"].log_prob(jax.numpy.asarray(T.act)))
        return Fit(spec.name, {}, ll[None, None].astype(np.float32), dict(rhat_max=1.0, ess_bulk_min=np.inf, ess_tail_min=np.inf, divergences=0, rung=rung, seconds=time.time() - t0), True)
    cfg = dict(LADDER[rung])
    mc = MCMC(NUTS(model, target_accept_prob=cfg.pop("target_accept_prob")), num_chains=CHAINS, chain_method=chain_method, progress_bar=False, **cfg)
    mc.run(jax.random.PRNGKey(seed + 1000 * rung), *args, extra_fields=("diverging",))
    post = {k: np.asarray(v) for k, v in mc.get_samples(group_by_chain=True).items()}
    ll = np.asarray(log_likelihood(model, mc.get_samples(), *args)["a"]).reshape(CHAINS, -1, len(T.act)).astype(np.float32)
    idata = az.from_dict(posterior={k: post[k] for k in sites})
    rhat, bulk, tail = az.rhat(idata), az.ess(idata, method="bulk"), az.ess(idata, method="tail")   # reduced per site: to_array() would
    d = dict(rhat_max=max(float(rhat[v].max()) for v in rhat.data_vars), ess_bulk_min=min(float(bulk[v].min()) for v in bulk.data_vars),   # outer-join
             ess_tail_min=min(float(tail[v].min()) for v in tail.data_vars), divergences=int(mc.get_extra_fields()["diverging"].sum()), rung=rung, seconds=time.time() - t0)   # the participant dims
    passed = d["rhat_max"] < GATE["rhat"] and min(d["ess_bulk_min"], d["ess_tail_min"]) >= GATE["ess"] and d["divergences"] <= GATE["divergences"]
    return Fit(spec.name, post, ll, d, passed)


def cache_path(name: str, key: str, cache_dir) -> Path:
    return Path(cache_dir) / (re.sub(r"[^\w.-]", "_", name) + f"_{key}.pkl")


def fit(spec: ModelSpec, T: TrialData, tab, key: str, cache_dir: Path, tab2=None, seed: int = 0) -> Fit:
    """Cached fit of ``spec``: climbs the ladder until the gate passes; ``key`` names the data/table the fit used. Every
    finished rung is cached too (``<name>_<key>_rung<r>.pkl``), so a task killed mid-ladder resumes at the first unfinished rung."""
    path = cache_path(spec.name, key, cache_dir)
    if path.exists():
        return pickle.load(open(path, "rb"))
    for rung in range(len(LADDER)):
        rung_path = cache_path(spec.name, f"{key}_rung{rung}", cache_dir)
        if rung_path.exists():
            result = pickle.load(open(rung_path, "rb"))
        else:
            result = fit_once(spec, T, tab, tab2, rung, seed)
            rung_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write(rung_path, lambda t: pickle.dump(result, open(t, "wb")))
        if result.passed:
            break
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, lambda t: pickle.dump(result, open(t, "wb")))
    return result


def _worker(name, eval_dir, version, tab_path, tab2_path, key, cache_dir, seed):
    T = trial_data(load_sessions(Path(eval_dir), version))
    tab2 = np.load(tab2_path, mmap_mode="r") if tab2_path else None
    return fit(REGISTRY[name], T, np.load(tab_path, mmap_mode="r"), key, cache_dir, tab2, seed).diagnostics


def run_many(names, eval_dir, version, table_paths: dict, key: str, cache_dir, workers: int = None, seed: int = 0) -> dict:
    """Fit the named registry models in spawned processes, ``workers`` at a time (default: one per JAX device; several
    processes on one GPU only contend, see the plan's Findings for 5.13). Each process loads the trials and memory-maps
    its tables; ``table_paths[name] = (table, second table or None)``. Returns name -> Fit, read back from the cache."""
    workers = workers or len(jax.devices())
    jobs = [(n, str(eval_dir), version, str(table_paths[n][0]), table_paths[n][1] and str(table_paths[n][1]), key, str(cache_dir), seed) for n in names]
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        for n, d in zip(names, ex.map(_worker, *zip(*jobs))):
            print(f"{n}: rung {d['rung']}, {d['seconds']:.0f} s, R-hat {d['rhat_max']:.3f}, ESS {min(d['ess_bulk_min'], d['ess_tail_min']):.0f}, divergences {d['divergences']}")
    return {n: pickle.load(open(cache_path(n, key, cache_dir), "rb")) for n in names}
