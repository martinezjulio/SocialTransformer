"""Hierarchical Thompson-sampling policy models for the human bandit choices.

The models are defined in docs/Model_0.md and docs/Model_1.md and fitted from
notebooks/analysis.py section 6. This module holds everything that does not depend on the
notebook's state: the grids' constants, the win-probability quadrature that tabulates each
belief's log-policy on a grid, the numpyro model that interpolates that grid at fitted
parameters, and a runner that fits many grids in parallel processes.

Import this module before jax in any process that uses it: numpyro's host device count must
be set before jax touches a backend, and it is set here at import.
"""
import os
import pickle
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.special import betainc, betaln, expit, log_expit

import numpyro
N_CHAINS = 4
numpyro.set_host_device_count(N_CHAINS)          # before jax initialises: one device per chain
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, log_likelihood, init_to_value
from jax.scipy.special import logsumexp

K_ARMS = 12                          # arms per bandit: the choice set of a single trial
KAPPA = np.geomspace(0.25, 16, 13)   # ratio sqrt(2); 1.0 is on the grid. Linear interpolation
                                     # between knots is off by ~0.1 per trial at ratio 2
LAPSE = 0.02                         # so one surprising choice cannot dominate a participant
P_FLOOR = 1e-12                      # log p is stored without the lapse; this keeps -inf out
LAM_W = np.array([0.1, 0.25, 0.5, 1, 2, 4, 8, 16])   # own-experience weights, Model 1 Own and Both
LAM_C = np.array([0., 0.25, 0.5, 1, 2, 4])           # cells: within-segment (log-spaced), carried (has 0)
LOGK, LOG_LAM_W, LAM_C_J = jnp.log(KAPPA), jnp.log(LAM_W), jnp.asarray(LAM_C)


# ---------------------------------------------------------------- the grids: quadrature

def p_win(ab, X=4096, L=80.0):
    """P(arm a draws the largest theta) = int f_a(x) prod_{j!=a} F_j(x) dx.  ab: (M,K,2) -> (M,K).

    Quadrature rather than Monte Carlo: faster and exact to ~1e-7. Substituting x = sigmoid(u)
    makes f(x)dx = x^a (1-x)^b / B(a,b) du, removing the x^(a-1) endpoint singularity that
    appears once kappa*alpha < 1. F near 1 comes from the complement (betainc(b,a,1-x) = 1-F)
    because 1-F underflows in float64 while the density still has mass out there at small
    kappa; without that every arm double-counts its own tail and the probabilities sum to
    ~1.002. scipy.special rather than scipy.stats: ~90x faster.
    """
    u = np.linspace(-L, L, X)
    a, b = ab[..., 0, None], ab[..., 1, None]            # (M,K,1) each
    F = betainc(a, b, expit(u))
    S = np.clip(betainc(b, a, expit(-u)), None, 1 - 1e-16)
    logF = np.where(F < 0.5, np.log(np.clip(F, 1e-300, None)), np.log1p(-S))
    logf = a * log_expit(u) + b * log_expit(-u) - betaln(a, b)
    return np.trapezoid(np.exp(logf + logF.sum(-2, keepdims=True) - logF), u, axis=-1)


def p_win_all(ab, chunk=128, workers=16):
    """p_win over ab (M,K,2) in chunks on a thread pool. The scipy and numpy ufuncs inside
    p_win release the GIL, so 16 threads run ~11x faster than one; the result is bitwise the
    same. Threads, not processes: jax is already initialised, and forking under it is unsafe."""
    with ThreadPoolExecutor(workers) as ex:
        return np.concatenate(list(ex.map(p_win, [ab[i:i + chunk] for i in range(0, len(ab), chunk)])))


def log_policy(p):
    """The grid entry stored for a win-probability table: log p, floored, without the lapse.
    The lapse is applied inside the model (lapse_mix) so a habit tilt can act on p first."""
    return np.log(np.clip(p, P_FLOOR, None))


# ---------------------------------------------------------------- the model

def lapse_mix(logq, xp=np):
    """log((1 - eps) q + eps / K) for a normalised log-policy logq. xp = np or jnp."""
    return xp.logaddexp(xp.log1p(-LAPSE) + logq, xp.log(LAPSE / K_ARMS))


def interp(x, g, v):
    """Linear interpolation of v (N,G,...) along grid g (G,) at per-row queries x (N,), removing
    axis 1. Applied twice on an (N,G1,G2,K) grid it is bilinear."""
    x = jnp.clip(x, g[0], g[-1])
    i = jnp.clip(jnp.searchsorted(g, x), 1, len(g) - 1)
    w = ((x - g[i - 1]) / (g[i] - g[i - 1])).reshape((-1,) + (1,) * (v.ndim - 2))
    n = jnp.arange(v.shape[0])
    return (1 - w) * v[n, i - 1] + w * v[n, i]


def hierarchical(name, plate, mu_scale, tau_scale):
    """One value per participant, non-centred (avoids the funnel): name_i = mu + tau * z_i."""
    mu = numpyro.sample(f"mu_{name}", dist.Normal(0., mu_scale))
    tau = numpyro.sample(f"tau_{name}", dist.HalfNormal(tau_scale))
    with plate:
        z = numpyro.sample(f"z_{name}", dist.Normal(0., 1.))
    return numpyro.deterministic(name, mu + tau * z)


def policy_model(part, logp, act, n_part, prev_arm, prev_rew, kappa=True, own=False, habit=False):
    """Hierarchical sharpened-Thompson policy, optionally with a habit tilt (docs/Model_1.md).

    `logp` is one belief's log-probability grid whose axes after the trial axis are, in order,
    the ones the flags ask for: kappa=True adds a |KAPPA| axis, interpolated at each
    participant's kappa ~ Beta(kappa*alpha, kappa*beta); own=True adds |LAM_W| and |LAM_C| axes,
    interpolated at the group-level own-experience weights (log-spaced in lambda_within,
    linear in lambda_carry because of its zero knot). Both flags: the trilinear Both cells.
    Neither: the grid's only column, a belief with nothing to fit. habit=True adds
    b_{i,t} = rho0_i + rhow_i * prev_rew to the previous arm's log-probability before
    renormalising; the lapse comes last."""
    plate = numpyro.plate("participant", n_part)
    n, lq = logp.shape[0], logp
    if kappa:
        lk = hierarchical("log_kappa", plate, 0.5, 0.3)
        lq = interp(lk[part], LOGK, lq)                     # log p at this participant's kappa
    if own:
        lw = numpyro.sample("log_lambda_within", dist.Normal(0., 1.5))
        lc = numpyro.sample("lambda_carry", dist.HalfNormal(4.))
        lq = interp(jnp.full(n, lc), LAM_C_J, interp(jnp.full(n, lw), LOG_LAM_W, lq))
    if not (kappa or own):
        lq = logp[:, 0]
    if habit:
        rho0 = hierarchical("rho0", plate, 2.0, 1.0)
        rhow = hierarchical("rhow", plate, 2.0, 1.0)
        b = (rho0[part] + rhow[part] * prev_rew) * (prev_arm >= 0)   # no previous arm: no tilt
        lq = lq + b[:, None] * jax.nn.one_hot(jnp.clip(prev_arm, 0), K_ARMS)
    lq = lq - logsumexp(lq, -1, keepdims=True)              # renormalise after the tilt
    numpyro.sample("a", dist.Categorical(logits=lapse_mix(lq, jnp)), obs=act)


# ---------------------------------------------------------------- fitting

class Trials(NamedTuple):
    """The per-trial data every fit shares. act: arm chosen; part: participant index; prev_arm /
    prev_rew: the previous trial of this segment (-1 / 0 on its first trial)."""
    act: np.ndarray
    part: np.ndarray
    n_part: int
    prev_arm: np.ndarray
    prev_rew: np.ndarray


def fit(logp, data, kappa=True, own=False, habit=False, seed=0, init=None,
        num_warmup=1500, num_samples=2500):
    """Fit policy_model to one grid. Returns a dict: every group-level site's draws (mu_*,
    tau_*, the lambdas), the pointwise WAIC elpd, and two r_hats.

    Two r_hats, because they mean different things. The group one covers the scalar sites,
    what the figures plot and what the WAIC comparison rests on. The worst one additionally
    covers every participant's own values; a participant whose choices are consistent with a
    wide range of kappa leaves that individual barely identified, which shows up here and
    nowhere else. `init`: site -> value to start every chain from, for a multimodal cell."""
    args = (jnp.asarray(data.part), jnp.asarray(logp), jnp.asarray(data.act), data.n_part,
            jnp.asarray(data.prev_arm), jnp.asarray(data.prev_rew))
    kw = dict(kappa=kappa, own=own, habit=habit)
    kernel = NUTS(policy_model, target_accept_prob=0.95,
                  **({"init_strategy": init_to_value(values=init)} if init else {}))
    mc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=N_CHAINS,
              progress_bar=False)
    mc.run(jax.random.PRNGKey(seed), *args, **kw)
    s = mc.get_samples()
    ll = log_likelihood(policy_model, s, *args, **kw)["a"]                    # (draws, trials)
    elpd = np.asarray(logsumexp(ll, 0) - jnp.log(ll.shape[0]) - ll.var(0))
    rh = numpyro.diagnostics.summary(mc.get_samples(group_by_chain=True))
    worst = lambda ks: max(np.atleast_1d(rh[k]["r_hat"]).max() for k in ks)
    group = [k for k in s if s[k].ndim == 1]                  # scalar sites: not per participant
    return dict({k: np.asarray(s[k]) for k in group}, elpd=elpd,
                rhat_grp=worst(group), rhat_worst=worst(rh))


def fit_many(jobs, data, workers=4, **kw):
    """Fit several grids at once, each in its own process (N_CHAINS chains each), `workers` at
    a time. jobs: name -> dict(logp=array, kappa=..., own=..., habit=..., init=...). Returns
    name -> fit result, in job order. Extra keyword arguments go to fit() for every job.

    Separate interpreters rather than a multiprocessing pool: jax's device count is fixed at
    import, and a spawned pool would re-import the calling script. Each job's inputs go through
    a temporary pickle; the child runs `python -m analysis.policy JOB OUT`."""
    src = str(Path(__file__).resolve().parents[1])
    env = dict(os.environ, PYTHONPATH=src + os.pathsep + os.environ.get("PYTHONPATH", ""))
    results, pending, running = {}, list(jobs), {}
    with tempfile.TemporaryDirectory(prefix="policy_fits_") as tmp:
        def launch(name):
            job, out = Path(tmp) / f"{len(running) + len(results)}.job", Path(tmp) / f"{len(running) + len(results)}.out"
            with open(job, "wb") as f:
                pickle.dump(dict(jobs[name], data=data, kw=kw), f, protocol=pickle.HIGHEST_PROTOCOL)
            running[name] = (subprocess.Popen([sys.executable, "-m", "analysis.policy", str(job), str(out)],
                                              env=env), out)
        while pending or running:
            while pending and len(running) < workers:
                launch(pending.pop(0))
            for name, (proc, out) in list(running.items()):
                if proc.poll() is None:
                    continue
                if proc.returncode != 0:
                    raise RuntimeError(f"fit {name!r} failed with exit code {proc.returncode}")
                with open(out, "rb") as f:
                    results[name] = pickle.load(f)
                del running[name]
                print(f"  fitted {name}", flush=True)
            if running:
                time.sleep(2)
    return {name: results[name] for name in jobs}


if __name__ == "__main__":                                   # one fit_many worker
    with open(sys.argv[1], "rb") as f:
        job = pickle.load(f)
    result = fit(job["logp"], job["data"], **{k: job.get(k, d) for k, d in
                 dict(kappa=True, own=False, habit=False, init=None).items()}, **job["kw"])
    with open(sys.argv[2], "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
