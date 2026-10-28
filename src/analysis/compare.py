# %% compare
"""PSIS-LOO comparison of fitted choice models (meta-plan 3.4).

ELPD is summed over trials; a difference between two models gets its SE over participants (per-participant
sums are the paired units), so a participant's 48 trials never count as independent. Pareto k > 0.7 marks
trials whose importance weights are unreliable; a model behind a registered claim with such trials falls
back to exact leave-one-session-out refits (``loso``). ``by_cell`` splits a difference over trial subsets
(round, last demonstrator, expert seen) for S2 by round and the unseen-expert ELPD split.
"""

from dataclasses import dataclass, replace

import arviz as az
import numpy as np
import pandas as pd

from analysis.data import TrialData
from analysis.fitting import Fit

K_BAD = 0.7


@dataclass
class Loo:
    name: str
    elpd: float            # sum over trials
    elpd_i: np.ndarray     # (N,) pointwise
    pareto_k: np.ndarray   # (N,)
    n_bad: int             # trials with k > K_BAD


def loo(fit: Fit) -> Loo:
    """PSIS-LOO from the fit's pointwise log-likelihood; a fit with a single draw (Random) is its log-likelihood."""
    if fit.log_lik.shape[0] * fit.log_lik.shape[1] == 1:
        return Loo(fit.name, float(fit.log_lik.sum()), fit.log_lik.reshape(-1).astype(float), np.zeros(fit.log_lik.shape[-1]), 0)
    ll = fit.log_lik.astype(float)
    var = ll.std((0, 1)) > 0            # trials whose log-likelihood is the same in every draw (e.g. a game's first shot
    elpd_i, k = ll[0, 0].copy(), np.zeros(ll.shape[-1])   # under a uniform belief) are exact: no importance weights to fit
    idata = az.from_dict(posterior={"_": np.zeros(ll.shape[:2])}, log_likelihood={"a": ll[:, :, var]})   # arviz wants a posterior group
    r = az.loo(idata, pointwise=True)
    elpd_i[var], k[var] = r.loo_i.values, r.pareto_k.values
    return Loo(fit.name, float(elpd_i.sum()), elpd_i, k, int((k > K_BAD).sum()))


def paired(a: Loo, b: Loo, part: np.ndarray, mask=None) -> dict:
    """ΔELPD = a − b over the trials in ``mask`` (all by default): sum, SE over participants, share of participants favouring a."""
    d = (a.elpd_i - b.elpd_i)[mask] if mask is not None else a.elpd_i - b.elpd_i
    per = np.bincount(part[mask] if mask is not None else part, d, minlength=part.max() + 1)
    return dict(delta=float(d.sum()), se=float(np.sqrt(len(per)) * per.std(ddof=1)), per_participant=per, share=float((per > 0).mean()))


def ladder(loos: dict, base: str, part: np.ndarray) -> pd.DataFrame:
    """One row per model: ELPD, its difference from ``base`` with the paired SE, the ratio, and the bad-k count."""
    rows = [dict(model=n, elpd=l.elpd, n_bad_k=l.n_bad, **{k: v for k, v in paired(l, loos[base], part).items() if k != "per_participant"}) for n, l in loos.items()]
    return pd.DataFrame(rows).assign(ratio=lambda t: t.delta / t.se.where(t.se > 0)).set_index("model")


def by_cell(a: Loo, b: Loo, T: TrialData, cells: dict) -> pd.DataFrame:
    """ΔELPD a − b per named trial subset (``cells``: name -> boolean mask over trials)."""
    return pd.DataFrame({n: {k: v for k, v in paired(a, b, T.part, m).items() if k != "per_participant"} for n, m in cells.items()}).T


def subset(T: TrialData, mask: np.ndarray) -> TrialData:
    """The trials where ``mask`` is True, participant indexing unchanged."""
    per_trial = {f: v[mask] for f, v in vars(T).items() if isinstance(v, np.ndarray) and v.shape[:1] == mask.shape}
    return replace(T, **per_trial, read={k: v[mask] for k, v in T.read.items()})


def loso(l: Loo, spec, T: TrialData, tab, tab2=None, seed: int = 0) -> Loo:
    """Pareto-k fallback: for every session holding a trial with k > K_BAD, refit without that session and replace
    its trials' pointwise elpd by the exact held-out log predictive density (log mean over draws); k set to 0 there."""
    from numpyro.infer import log_likelihood
    from scipy.special import logsumexp
    from analysis.fitting import LADDER, fit_once
    from analysis.policies import model
    elpd_i, k = l.elpd_i.copy(), l.pareto_k.copy()
    for sid in np.unique(T.session[k > K_BAD]):
        held = T.session == sid
        for rung in range(len(LADDER)):
            f = fit_once(spec, subset(T, ~held), tab[~held], None if tab2 is None else tab2[~held], rung, seed)
            if f.passed:
                break
        draws = {s: v.reshape(-1, *v.shape[2:]) for s, v in f.posterior.items()}
        ll = np.asarray(log_likelihood(model, draws, spec, subset(T, held), tab[held], True, None if tab2 is None else tab2[held])["a"])
        elpd_i[held], k[held] = logsumexp(ll, 0) - np.log(len(ll)), 0.0
    return Loo(l.name, float(elpd_i.sum()), elpd_i, k, int((k > K_BAD).sum()))
