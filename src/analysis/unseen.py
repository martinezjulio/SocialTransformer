# %% unseen
"""Unseen-expert section (exploratory, meta-plan §6): what the readout entering a play game contains, and how
choices made after the novice relate to it.

``readout_table`` is output (a) of the section: for each readout (expert, novice), whether the expert has
already demonstrated on this round's bandit, and round, the readout entering the play game scored four ways:
confidence (total Beta count), Spearman rank correlation of its means with the true payoffs, the true payoff
of its favourite arm, and the entropy (bits) of the Thompson policy at κ = 1 on it. Cells are means over
sessions of per-session means, with the SE over sessions.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis.data import TrialData
from analysis.thompson import P_FLOOR, p_win_all

STATISTICS = ["confidence", "rank_corr", "favourite_payoff", "entropy"]


def entering(T: TrialData) -> np.ndarray:
    """Mask of each play game's first shot: the readout is constant within a game, so one row per game."""
    return T.prev_arm == -1


def readout_stats(T: TrialData, source: str) -> pd.DataFrame:
    """One row per play game: the four statistics of ``source``'s readout entering it, with the cell labels."""
    first = entering(T)
    ab, theta = T.read[source][first], T.theta[first]
    mean, P = ab[..., 0] / ab.sum(-1), p_win_all(ab)
    return pd.DataFrame(dict(session=T.session[first], readout=source, expert_seen=T.expert_seen[first], round=(T.round_pm[first] > 0).astype(int) + 1,
                             confidence=ab.sum((1, 2)), rank_corr=[spearmanr(m, t)[0] for m, t in zip(mean, theta)],
                             favourite_payoff=theta[np.arange(len(theta)), mean.argmax(1)], entropy=-(P * np.log2(np.maximum(P, P_FLOOR))).sum(-1)))


def readout_table(T: TrialData, sources=("expert", "novice")) -> pd.DataFrame:
    """Output (a): mean, SE and n (sessions) of the four statistics by readout × expert_seen × round."""
    per_session = pd.concat([readout_stats(T, s) for s in sources]).groupby(["readout", "expert_seen", "round", "session"])[STATISTICS].mean()
    return per_session.groupby(["readout", "expert_seen", "round"]).agg(["mean", "sem", "count"])
