"""Thompson win probabilities in JAX: P(arm k draws the largest payoff) under independent Beta beliefs.

``log_p_win(ab)`` integrates f_k(x) Π_{j≠k} F_j(x) dx by trapezoid quadrature on the logit scale,
x = sigmoid(u), u ∈ [-L, L], which removes the x^(a-1) endpoint singularity once κ·α < 1 (the legacy
quadrature in notebooks/legacy_analysis/policy.py, ported term by term). Everything is in log space;
F near 1 comes from the complement betainc(b, a, 1-x) so an arm's own tail is not double counted.
Node count is a parameter so the 5.6 benchmark can trade accuracy for speed inside a NUTS likelihood.
Importing this module enables float64 in JAX: the Beta counts span 1..~140 and the tails need it.
"""

import jax
import jax.numpy as jnp
from jax.nn import log_sigmoid, sigmoid
from jax.scipy.special import betainc, betaln, logsumexp

jax.config.update("jax_enable_x64", True)


def log_p_win(ab: jnp.ndarray, nodes: int = 4096, L: float = 80.0) -> jnp.ndarray:
    """ab: (..., K, 2) Beta (alpha, beta) per arm → (..., K) log P(arm is the Thompson draw's argmax)."""
    u = jnp.linspace(-L, L, nodes)
    du = u[1] - u[0]
    log_w = jnp.full(nodes, jnp.log(du)).at[jnp.array([0, nodes - 1])].add(jnp.log(0.5))   # trapezoid weights
    a, b = ab[..., 0, None], ab[..., 1, None]                                             # (..., K, 1)
    F = betainc(a, b, sigmoid(u))
    S = jnp.clip(betainc(b, a, sigmoid(-u)), None, 1 - 1e-16)                             # 1 - F, accurate near 1
    logF = jnp.where(F < 0.5, jnp.log(jnp.clip(F, 1e-300, None)), jnp.log1p(-S))         # (..., K, nodes)
    logf = a * log_sigmoid(u) + b * log_sigmoid(-u) - betaln(a, b)                        # Beta density times dx/du
    return logsumexp(logf + logF.sum(-2, keepdims=True) - logF + log_w, axis=-1)


def p_win(ab: jnp.ndarray, nodes: int = 4096, L: float = 80.0) -> jnp.ndarray:
    return jnp.exp(log_p_win(ab, nodes, L))
