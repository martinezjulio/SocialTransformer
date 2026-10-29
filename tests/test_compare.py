"""PSIS-LOO on synthetic log-likelihoods: the exact-trial and flat-tail paths of ``analysis.compare.loo``."""
import os
import sys

import numpy as np
from scipy.special import logsumexp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from analysis.compare import Loo, loo  # noqa: E402
from analysis.fitting import Fit  # noqa: E402


def test_flat_weight_tail_takes_the_plain_importance_estimate():
    rng = np.random.default_rng(0)
    ll = rng.normal(-1.5, 0.3, size=(4, 1000, 3)).astype(np.float32)
    ll[:, :, 1] = -1.0                                       # identical in every draw: exact, no weights
    ll[:, :, 2] = np.maximum(ll[:, :, 2], np.float32(-1.9))   # the large-weight tail tied (9% of draws): arviz returns k = inf
    l = loo(Fit("synthetic", {}, ll, {}, True))
    assert isinstance(l, Loo) and l.elpd_i.shape == (3,) and np.isfinite(l.elpd_i).all() and np.isfinite(l.pareto_k).all()
    assert l.elpd_i[1] == -1.0 and l.pareto_k[1] == 0
    x = ll[:, :, 2].astype(float).reshape(-1)
    assert np.isclose(l.elpd_i[2], np.log(x.size) - logsumexp(-x)) and l.pareto_k[2] == 0
    assert l.n_bad == 0 and np.isclose(l.elpd, l.elpd_i.sum())
