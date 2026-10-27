# %% predictive
"""A fitted choice model as a ``Policy`` the simulator can play (recovery, PPC, split-half checks).

``FittedPolicy`` holds one participant's parameters at one posterior draw and turns a ``TrialState`` into
the model's choice probabilities with the exact Thompson quadrature (no tables: simulated beliefs are new).
It implements the same four steps as ``policies.model`` (belief, Thompson, habit tilt, lapse) and the
Switch mixture. Sources available in the loop: expert, novice, recent (last watched), uniform.
"""

import numpy as np
from scipy.special import expit, logsumexp

from analysis.policies import ModelSpec
from analysis.thompson import P_FLOOR, p_win
from inference.learners import TrialState

PER_PARTICIPANT = ("log_kappa", "rho0", "rho_win", "logit_omega_E", "logit_omega_N")


def draw_params(posterior: dict, draw: int, participant: int) -> dict:
    """Site -> value for one flattened draw, per-participant sites indexed at ``participant``."""
    flat = {s: v.reshape(-1, *v.shape[2:])[draw] for s, v in posterior.items()}
    return {s: (v[participant] if s in PER_PARTICIPANT else v) for s, v in flat.items()}


class FittedPolicy:
    def __init__(self, spec: ModelSpec, params: dict):
        self.spec, self.p = spec, params

    def _source(self, s: TrialState, source: str) -> np.ndarray:   # (K, 2) counts before the learner's own
        if source == "uniform":
            return np.ones((s.own_within.shape[1], 2))
        d = {"expert": s.expert, "novice": s.novice, "recent": s.last_watched}[source]
        return np.stack([s.alpha_demo[d].numpy(), s.beta_demo[d].numpy()], -1)

    def _log_policy(self, s: TrialState, source: str) -> np.ndarray:
        belief = self._source(s, source)
        if self.spec.own:
            lam_c = 0.0 if self.spec.reset else self.p["lambda_c"]
            belief = belief + np.exp(self.p["log_lambda_w"]) * s.own_within.numpy().T + lam_c * s.own_carry.numpy().T
        ab = (np.exp(self.p["log_kappa"]) if self.spec.kappa else 1.0) * belief
        lq = np.log(np.clip(p_win(ab[None])[0], P_FLOOR, None))
        if self.spec.habit and s.prev_action is not None:
            lq[s.prev_action] += self.p["rho0"] + self.p["rho_win"] * s.prev_reward
        if self.spec.persev:
            lq = lq + self.p["persev"] * (s.own_within.numpy().sum(0) > 0)
        return lq - logsumexp(lq)

    def probs(self, s: TrialState) -> np.ndarray:
        lq = self._log_policy(s, self.spec.source)
        if self.spec.mixture != "none":
            after_expert = s.last_watched == s.expert
            lo = self.p["logit_omega_E" if after_expert else "logit_omega_N"]
            if self.spec.mixture == "by_round":
                lo = lo + self.p["delta_E" if after_expert else "delta_N"] * (s.round_index - 0.5)
            lq = np.logaddexp(np.log(expit(lo)) + lq, np.log(expit(-lo)) + self._log_policy(s, "novice"))
        eps = self.p["eps"] if self.spec.lapse is None else self.spec.lapse
        return (1 - eps) * np.exp(lq) + eps / len(lq)
