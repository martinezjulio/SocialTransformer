# %% policies
"""Choice-model specifications and the registry of fitted models (plan 'Models' table; preregistration.md).

Every model is H+Both with terms switched on or off: a belief source scaled by κ_i, the learner's own
counts weighted by λ_w (this play game) and λ_c (the round's earlier play game), Thompson sampling, a
win-stay lose-shift tilt, an optional perseveration bonus, a fixed lapse; the Switch models mix the
expert-readout and novice-readout policies with a weight ω per trial. ``belief`` gives the counts a
source contributes before the learner's own are added; the model body (``policies.py`` below) reads them
through the cached tables of ``thompson.py``.
"""

from dataclasses import dataclass

import numpy as np

from analysis.data import TrialData

SOURCES = ("expert", "novice", "recent", "naive", "vis_last", "vis_expert", "uniform")
MIXTURES = ("none", "two", "by_round", "eight")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    source: str = "expert"      # belief entering the Thompson step (SOURCES); Switch models: expert, mixed with novice
    kappa: bool = True          # κ_i (hierarchical) scales the source; False fixes κ = 1
    own: bool = True            # λ_w, λ_c on the learner's own counts; False leaves them out
    reset: bool = False         # λ_c fixed at 0: the exploratory reset learner (needs own)
    habit: bool = False         # win-stay lose-shift tilt ρ0_i + ρwin_i · previous reward on the previous arm
    persev: bool = False        # one group-level bonus on any arm already chosen in this play game
    mixture: str = "none"       # MIXTURES: two = ω_E, ω_N per participant; by_round = + δ_E, δ_N; eight = group weights
    kappa_trend: bool = False   # δ_κ: round slope on log κ (robustness)
    role: str = "reported"      # full | S2 | S3 | S1 | reported | robustness | exploratory

    def __post_init__(self):
        assert self.source in SOURCES and self.mixture in MIXTURES and (self.own or not self.reset)
        assert self.mixture == "none" or (self.source == "expert" and self.kappa), "Switch models mix the two readouts"


def belief(T: TrialData, source: str) -> np.ndarray:
    """(N, K, 2) counts the source contributes before the learner's own: a readout as is; Beta(1, 1) plus
    the visible pulls for the copying and naive sources (naive already holds the learner's own pulls)."""
    if source in T.read:
        return T.read[source]
    return 1.0 + {"naive": T.vis_round, "vis_last": T.vis_last, "vis_expert": T.vis_expert, "uniform": np.zeros_like(T.vis_round)}[source]


REGISTRY = {m.name: m for m in [
    ModelSpec("H+Both", habit=True, role="full"),
    ModelSpec("src:recent", source="recent", habit=True, role="S2"),
    ModelSpec("src:novice", source="novice", habit=True),
    ModelSpec("src:naive", source="naive", own=False, habit=True),
    ModelSpec("vis:last", source="vis_last", habit=True, role="S3"),
    ModelSpec("vis:expert", source="vis_expert", habit=True),
    ModelSpec("Both"), ModelSpec("H+Social", own=False, habit=True), ModelSpec("H+Own", source="uniform", kappa=False, habit=True),
    ModelSpec("H", source="uniform", kappa=False, own=False, habit=True), ModelSpec("Own", source="uniform", kappa=False),
    ModelSpec("Social", own=False), ModelSpec("Random", source="uniform", kappa=False, own=False),
    ModelSpec("Switch", habit=True, mixture="two"),
    ModelSpec("Switch-by-round", habit=True, mixture="by_round", role="S1"),
    ModelSpec("Switch-by-round+ktrend", habit=True, mixture="by_round", kappa_trend=True, role="robustness"),
    ModelSpec("Switch-8", habit=True, mixture="eight", role="exploratory"),
    ModelSpec("H+Both+persev", habit=True, persev=True, role="exploratory"),
    ModelSpec("H+Both-reset", habit=True, reset=True, role="exploratory"),
]}


# %% model body
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from analysis.thompson import KAPPA, LAM_C, LAM_W

LAPSE = 0.02


def grids(spec: ModelSpec):
    """(κ, λ_w, λ_c) knots of the table a spec reads; a single knot where the term is off."""
    return (KAPPA if spec.kappa else np.ones(1), LAM_W if spec.own else np.zeros(1), LAM_C if spec.own else np.zeros(1))


def cubic(xq, knots, tab):
    """4-point Lagrange cubic along axis 1 of tab (N, G, ...) at per-row xq (N,), clipped to the knots' range."""
    if len(knots) == 1:
        return tab[:, 0]
    xq = jnp.clip(xq, knots[0], knots[-1])
    idx = jnp.clip(jnp.searchsorted(knots, xq) - 2, 0, len(knots) - 4)[:, None] + jnp.arange(4)   # (N, 4)
    xs, eye = knots[idx], jnp.eye(4, dtype=bool)
    num = jnp.prod(jnp.where(eye, 1.0, (xq[:, None] - xs)[:, None, :]), -1)                      # Π_{m≠j} (xq - x_m)
    den = jnp.prod(jnp.where(eye, 1.0, xs[:, :, None] - xs[:, None, :]), -1)                      # Π_{m≠j} (x_j - x_m)
    ys = jnp.take_along_axis(tab, idx.reshape(idx.shape + (1,) * (tab.ndim - 2)), axis=1)
    return jnp.einsum("nj,nj...->n...", num / den, ys)


def hierarchical(name, plate, mu_scale, tau_scale):
    """One value per participant, non-centred: name_i = mu + tau * z_i."""
    mu, tau = numpyro.sample(f"mu_{name}", dist.Normal(0.0, mu_scale)), numpyro.sample(f"tau_{name}", dist.HalfNormal(tau_scale))
    with plate:
        z = numpyro.sample(f"z_{name}", dist.Normal(0.0, 1.0))
    return numpyro.deterministic(name, mu + tau * z)


def log_policy(tab, kappa, lam_w, lam_c, log_k, log_w, c, tilt):
    """Normalised log-policy (N, K) from a table (N, |κ|, |λ_w|, |λ_c|, K) at per-trial log κ, group log λ_w and
    λ_c (log axis above the first nonzero knot, linear blend with the λ_c = 0 rung below it) plus a tilt (N, K)."""
    lq = cubic(jnp.broadcast_to(log_w, log_k.shape), jnp.log(lam_w) if len(lam_w) > 1 else lam_w, cubic(log_k, jnp.log(kappa), tab))
    if len(lam_c) > 1:
        f = jnp.clip(c / lam_c[1], 0.0, 1.0)
        lq = f * cubic(jnp.broadcast_to(jnp.log(jnp.maximum(c, lam_c[1])), log_k.shape), jnp.log(lam_c[1:]), lq[:, 1:]) + (1 - f) * lq[:, 0]
    else:
        lq = lq[:, 0]
    lq = lq + tilt
    return lq - jax.scipy.special.logsumexp(lq, -1, keepdims=True)


def model(spec: ModelSpec, T: TrialData, tab, observed: bool = True):
    """numpyro model of one spec on the trials T through its table; observed=False samples choices (simulation)."""
    part, prev_arm, prev_rew, act = (jnp.asarray(x) for x in (T.part, T.prev_arm, T.prev_rew, T.act))
    N, K = T.act.shape[0], T.theta.shape[1]
    plate = numpyro.plate("participant", len(T.people))
    kappa, lam_w, lam_c = grids(spec)
    log_k = hierarchical("log_kappa", plate, 0.5, 0.3)[part] if spec.kappa else jnp.zeros(N)
    log_w = numpyro.sample("log_lambda_w", dist.Normal(0.0, 1.5)) if spec.own else 0.0
    c = 0.0 if spec.reset or not spec.own else numpyro.sample("lambda_c", dist.HalfNormal(4.0))
    tilt = jnp.zeros((N, K))
    if spec.habit:
        b = (hierarchical("rho0", plate, 2.0, 1.0)[part] + hierarchical("rho_win", plate, 2.0, 1.0)[part] * prev_rew) * (prev_arm >= 0)
        tilt = tilt + b[:, None] * jax.nn.one_hot(jnp.clip(prev_arm, 0), K)
    if spec.persev:
        tilt = tilt + numpyro.sample("persev", dist.Normal(0.0, 2.0)) * jnp.asarray(T.own_seg.sum(-1) > 0)
    lq = log_policy(jnp.asarray(tab), kappa, lam_w, lam_c, log_k, log_w, c, tilt)
    lq = jnp.logaddexp(jnp.log1p(-LAPSE) + lq, jnp.log(LAPSE / K))
    numpyro.sample("a", dist.Categorical(logits=lq), obs=act if observed else None)
