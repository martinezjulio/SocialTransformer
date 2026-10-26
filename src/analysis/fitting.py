# %% fitting
"""Fit one choice-model spec with NUTS, gate its diagnostics, climb a refit ladder on failure, cache the result.

Gate (meta-plan 3.4): R̂ < 1.01 on every site, bulk and tail ESS ≥ 400, zero divergences. The ladder starts at
4 chains × (1000 warm-up + 1000 draws) and raises the budget and target acceptance on failure; a fit that fails
the last rung is returned with ``passed=False`` and never reported. Pointwise log-likelihoods are kept for
PSIS-LOO (``compare.py``). Models with nothing to sample (Random) get their log-likelihood directly.
"""

import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path

import arviz as az
import jax
import numpy as np
from numpyro import handlers
from numpyro.infer import MCMC, NUTS, log_likelihood

from analysis.data import TrialData
from analysis.policies import ModelSpec, model

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
    d = dict(rhat_max=float(az.rhat(idata).to_array().max()), ess_bulk_min=float(az.ess(idata, method="bulk").to_array().min()),
             ess_tail_min=float(az.ess(idata, method="tail").to_array().min()), divergences=int(mc.get_extra_fields()["diverging"].sum()), rung=rung, seconds=time.time() - t0)
    passed = d["rhat_max"] < GATE["rhat"] and min(d["ess_bulk_min"], d["ess_tail_min"]) >= GATE["ess"] and d["divergences"] <= GATE["divergences"]
    return Fit(spec.name, post, ll, d, passed)


def fit(spec: ModelSpec, T: TrialData, tab, key: str, cache_dir: Path, tab2=None, seed: int = 0) -> Fit:
    """Cached fit of ``spec``: climbs the ladder until the gate passes; ``key`` names the data/table the fit used."""
    path = Path(cache_dir) / (re.sub(r"[^\w.-]", "_", spec.name) + f"_{key}.pkl")
    if path.exists():
        return pickle.load(open(path, "rb"))
    for rung in range(len(LADDER)):
        result = fit_once(spec, T, tab, tab2, rung, seed)
        if result.passed:
            break
    path.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(result, open(path, "wb"))
    return result
