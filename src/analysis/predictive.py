# %% predictive
"""A fitted choice model as a ``Policy`` the simulator can play (recovery, PPC, split-half checks).

``FittedPolicy`` holds one participant's parameters at one posterior draw and turns a ``TrialState`` into
the model's choice probabilities with the exact Thompson quadrature (no tables: simulated beliefs are new).
It implements the same four steps as ``policies.model`` (belief, Thompson, habit tilt, lapse) and the
Switch mixture. Sources available in the loop: expert, novice, recent (last watched), uniform.
``simulated_trial_data`` turns the simulator's session frames into ``TrialData`` through the human
builder, and ``sample_participants`` draws new participants from the fitted hierarchy.
"""

import numpy as np
import pandas as pd
from scipy.special import expit, logsumexp

from analysis.data import SEGMENTS_PER_ROUND, Sessions, TrialData, parse_betas, trial_data
from analysis.policies import ModelSpec
from analysis.thompson import P_FLOOR, p_win
from inference.learners import TrialState

PER_PARTICIPANT = ("log_kappa", "rho0", "rho_win", "logit_omega_E", "logit_omega_N")
TRIAL_COLUMNS = ["session_id", "anonymous_id", "game_sequence", "round_index", "game_index", "demonstrator_type", "position", "zone_bandit"]


def simulated_trial_data(frames, version: str = "simulated") -> TrialData:
    """``simulate_session`` frames (shot rows in the human schema, ``expert_betas``/``novice_betas`` on play rows) →
    ``TrialData`` through the same builder as the human data. 'recent' is the readout of the demonstrator watched
    just before the play game; the theta readout is unused downstream and left NaN. Rows must be in shot order."""
    shots = pd.concat(frames, ignore_index=True).sort_values(["session_id", "game_index"], kind="stable").reset_index(drop=True)
    trials = shots[TRIAL_COLUMNS].assign(reward=shots["reward_generated"].astype(float))
    rows = []
    for (sid, gi), g in shots.groupby(["session_id", "game_index"], sort=True):
        if gi % 4 in (1, 3):   # play game: cutoff right after the demonstration before it, whose type the sequence gives
            e, n, rnd = parse_betas(g["expert_betas"].iloc[0]), parse_betas(g["novice_betas"].iloc[0]), int(g["round_index"].iloc[0])
            last = g["game_sequence"].iloc[0].split("-")[gi % 4 - 1]
            rows.append(dict(session_id=sid, cutoff=SEGMENTS_PER_ROUND * rnd + gi % 4 + 1, segment_type="learner", round_index=rnd,
                             theta=np.full(len(e), np.nan), expert=e, novice=n, recent=e if last == "expert" else n))
    per_game = int(trials.groupby(["session_id", "game_index"]).size().mode()[0])
    return trial_data(Sessions(version, trials, pd.DataFrame(rows), per_game, ()))


def sample_participants(posterior: dict, draw: int, n: int, rng: np.random.Generator) -> list:
    """``n`` new participants from the hierarchy at one posterior draw: per-participant sites as mu + tau · z with fresh
    z, group sites copied into every dict (their values are the truth a recovery check must find back)."""
    flat = {s: v.reshape(-1, *v.shape[2:])[draw] for s, v in posterior.items()}
    group = {s: v for s, v in flat.items() if v.ndim == 0 and not s.startswith(("mu_", "tau_"))}
    people = {s: flat[f"mu_{s}"] + flat[f"tau_{s}"] * rng.standard_normal(n) for s in PER_PARTICIPANT if s in flat}
    return [group | {s: float(v[i]) for s, v in people.items()} for i in range(n)]


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
