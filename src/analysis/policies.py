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
    lapse: float | None = 0.02  # fixed lapse rate; None fits ε at group level, Beta(1, 9) (lapse robustness)
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
    ModelSpec("H+Both+eps", habit=True, lapse=None, role="robustness"),
    ModelSpec("Switch-by-round+eps", habit=True, mixture="by_round", lapse=None, role="robustness"),
]}


# %% model body
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from analysis.thompson import KAPPA, LAM_C, LAM_W


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
    λ_c (log axis above the first nonzero knot; below it a cubic on the linear axis through the λ_c = 0 knot) plus a tilt (N, K)."""
    lq = cubic(jnp.broadcast_to(log_w, log_k.shape), jnp.log(lam_w) if len(lam_w) > 1 else lam_w, cubic(log_k, jnp.log(kappa), tab))
    if len(lam_c) > 1:
        cq = jnp.broadcast_to(c, log_k.shape)
        hi = cubic(jnp.log(jnp.maximum(cq, lam_c[1])), jnp.log(lam_c[1:]), lq[:, 1:])
        lo = cubic(jnp.minimum(cq, lam_c[1]), jnp.asarray(lam_c[:4]), lq[:, :4])
        lq = jnp.where((cq < lam_c[1])[:, None], lo, hi)
    else:
        lq = lq[:, 0]
    lq = lq + tilt
    return lq - jax.scipy.special.logsumexp(lq, -1, keepdims=True)


def mixture_logit(spec, plate, part, last_exp, round_pm, expert_first):
    """Per-trial logit (N,) of the weight on the expert-readout policy: ω_E,i / ω_N,i by the last demonstrator
    (two), plus δ_E / δ_N times the ±½ round (by_round), or eight group weights by (demonstrator, round, condition)
    around which participants deviate with the same τ z_i (eight)."""
    eight = spec.mixture == "eight"
    lo = {}
    for s in "EN":
        mu = 0.0 if eight else numpyro.sample(f"mu_logit_omega_{s}", dist.Normal(0.0, 1.5))
        tau = numpyro.sample(f"tau_logit_omega_{s}", dist.HalfNormal(1.0))
        with plate:
            z = numpyro.sample(f"z_logit_omega_{s}", dist.Normal(0.0, 1.0))
        lo[s] = numpyro.deterministic(f"logit_omega_{s}", mu + tau * z)
        if not eight:
            numpyro.deterministic(f"omega_{s}", jax.nn.sigmoid(lo[s]))
    out = jnp.where(last_exp, lo["E"][part], lo["N"][part])
    if spec.mixture == "by_round":
        d_e, d_n = numpyro.sample("delta_E", dist.Normal(0.0, 1.0)), numpyro.sample("delta_N", dist.Normal(0.0, 1.0))
        out = out + jnp.where(last_exp, d_e, d_n) * round_pm
    if eight:
        out = out + numpyro.sample("logit_omega_8", dist.Normal(0.0, 1.5).expand([8]))[last_exp * 4 + (round_pm > 0) * 2 + expert_first]
    return out


def model(spec: ModelSpec, T: TrialData, tab, observed: bool = True, tab2=None):
    """numpyro model of one spec on the trials T through its table (tab2: the novice-readout table of a Switch
    model, same grids); observed=False samples choices (simulation)."""
    part, prev_arm, prev_rew, act = (jnp.asarray(x) for x in (T.part, T.prev_arm, T.prev_rew, T.act))
    last_exp, round_pm, expert_first = (jnp.asarray(x) for x in (T.last_exp, T.round_pm, T.expert_first))
    N, K = T.act.shape[0], T.theta.shape[1]
    plate = numpyro.plate("participant", len(T.people))
    kappa, lam_w, lam_c = grids(spec)
    log_k = hierarchical("log_kappa", plate, 0.5, 0.3)[part] if spec.kappa else jnp.zeros(N)
    if spec.kappa_trend:
        log_k = log_k + numpyro.sample("delta_kappa", dist.Normal(0.0, 0.5)) * round_pm
    log_w = numpyro.sample("log_lambda_w", dist.Normal(0.0, 1.5)) if spec.own else 0.0
    c = 0.0 if spec.reset or not spec.own else numpyro.sample("lambda_c", dist.HalfNormal(4.0))
    tilt = jnp.zeros((N, K))
    if spec.habit:
        b = (hierarchical("rho0", plate, 2.0, 1.0)[part] + hierarchical("rho_win", plate, 2.0, 1.0)[part] * prev_rew) * (prev_arm >= 0)
        tilt = tilt + b[:, None] * jax.nn.one_hot(jnp.clip(prev_arm, 0), K)
    if spec.persev:
        tilt = tilt + numpyro.sample("persev", dist.Normal(0.0, 2.0)) * jnp.asarray(T.own_seg.sum(-1) > 0)
    lq = log_policy(jnp.asarray(tab), kappa, lam_w, lam_c, log_k, log_w, c, tilt)
    if spec.mixture != "none":
        lo = mixture_logit(spec, plate, part, last_exp, round_pm, expert_first)[:, None]
        lq = jnp.logaddexp(jax.nn.log_sigmoid(lo) + lq, jax.nn.log_sigmoid(-lo) + log_policy(jnp.asarray(tab2), kappa, lam_w, lam_c, log_k, log_w, c, tilt))
    eps = numpyro.sample("eps", dist.Beta(1.0, 9.0)) if spec.lapse is None else spec.lapse
    lq = jnp.logaddexp(jnp.log1p(-eps) + lq, jnp.log(eps / K))
    numpyro.sample("a", dist.Categorical(logits=lq), obs=act if observed else None)
