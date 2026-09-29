"""Descriptive hierarchical regressions of trial outcomes on the session design.

docs/Model_4.md, 4a: the same linear predictor for every outcome (reward, chosen payoff, stay,
distinct arms), fitted from notebooks/analysis.py section 7. The design columns (the condition
x demonstrator x round factorial) are built there; this module takes named regressors. Nothing here reads the belief
grids; the models in policy.py ask which policy explains a choice, these ask how the outcomes
of the choices move across the session. Shares policy.py's participant hierarchy so the two
report the same way.

Imports analysis.policy first: that import sets numpyro's host device count before jax starts.
"""
import numpy as np

from analysis.policy import N_CHAINS, hierarchical          # noqa: F401  (sets the device count)
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

FAMILIES = ("bernoulli", "normal", "binomial")


def outcome_model(part, X, y, n_part, family, trials=None):
    """eta = a_i + sum_j gamma_j * X_j, with participant intercepts a_i, non-centred:
    mu_a ~ N(0, 1.5), gamma ~ N(0, 1), tau_a ~ HalfNormal(1). The random-effects structure of
    lmer's (1 | participant). X maps a name to a per-row regressor, each giving a site gamma_name.
    family: bernoulli (logit), normal (identity, sigma ~ HalfNormal(1)), binomial (logit, y
    successes out of `trials`)."""
    plate = numpyro.plate("participant", n_part)
    a = hierarchical("a", plate, 1.5, 1.0)
    eta = a[part]
    for name, x in X.items():
        eta = eta + numpyro.sample(f"gamma_{name}", dist.Normal(0., 1.)) * x
    if family == "normal":
        sigma = numpyro.sample("sigma", dist.HalfNormal(1.))
        numpyro.sample("y", dist.Normal(eta, sigma), obs=y)
    elif family == "bernoulli":
        numpyro.sample("y", dist.Bernoulli(logits=eta), obs=y)
    elif family == "binomial":
        numpyro.sample("y", dist.Binomial(trials, logits=eta), obs=y)
    else:
        raise ValueError(f"family must be one of {FAMILIES}, not {family!r}")


def fit_outcome(y, part, n_part, X, family, trials=None, seed=0, num_warmup=1000,
                num_samples=2000):
    """Fit outcome_model in this process (N_CHAINS chains). Returns a dict of every site's draws
    except the raw z (group sites (draws,), a (draws, n_part)) and the worst r_hat."""
    X = {k: jnp.asarray(np.asarray(v, float)) for k, v in dict(X).items()}
    args = (jnp.asarray(part), X, jnp.asarray(np.asarray(y, float)), n_part, family, trials)
    mc = MCMC(NUTS(outcome_model, target_accept_prob=0.9), num_warmup=num_warmup,
              num_samples=num_samples, num_chains=N_CHAINS, progress_bar=False)
    mc.run(jax.random.PRNGKey(seed), *args)
    s = mc.get_samples()
    rh = numpyro.diagnostics.summary(mc.get_samples(group_by_chain=True))
    return dict({k: np.asarray(v) for k, v in s.items() if not k.startswith("z_")},
                rhat=max(np.atleast_1d(rh[k]["r_hat"]).max() for k in rh))
