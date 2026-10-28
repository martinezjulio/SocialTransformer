# %% unseen
"""Unseen-expert section (exploratory, meta-plan §6): what the readout entering a play game contains, and how
choices made after the novice relate to it.

``readout_table`` is output (a) of the section: for each readout (expert, novice), whether the expert has
already demonstrated on this round's bandit, and round, the readout entering the play game scored four ways:
confidence (total Beta count), Spearman rank correlation of its means with the true payoffs, the true payoff
of its favourite arm, and the entropy (bits) of the Thompson policy at κ = 1 on it. Cells are means over
sessions of per-session means, with the SE over sessions.
``copying_table`` compares the unseen-expert readout with the copying belief Beta(1 + wins, 1 + losses) built from
the novice's visible shots (rank correlation across arms of the two beliefs' means, and of each with the truth), and
``elpd_by_cell`` gives the ΔELPD of two fitted models on the after-novice cells (expert seen × round), from the LOO
objects of the choice stage: expert belief vs copying (C5) here, expert vs novice belief in 7.3.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis.compare import Loo, by_cell
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


def copying_table(T: TrialData) -> pd.DataFrame:
    """After-novice play games by expert_seen × round: Spearman rank correlation across arms between the copying
    belief's mean and the expert readout's mean (``copy_vs_expert``), and of each with the true payoffs; per-session
    means, SE and n over sessions."""
    first = entering(T) & ~T.last_exp
    copy, expert, theta = 1.0 + T.vis_last[first], T.read["expert"][first], T.theta[first]
    mc, me = copy[..., 0] / copy.sum(-1), expert[..., 0] / expert.sum(-1)
    rows = pd.DataFrame(dict(session=T.session[first], expert_seen=T.expert_seen[first], round=(T.round_pm[first] > 0).astype(int) + 1,
                             copy_vs_expert=[spearmanr(a, b)[0] for a, b in zip(mc, me)], copy_vs_truth=[spearmanr(a, t)[0] for a, t in zip(mc, theta)],
                             expert_vs_truth=[spearmanr(b, t)[0] for b, t in zip(me, theta)]))
    stats = ["copy_vs_expert", "copy_vs_truth", "expert_vs_truth"]
    return rows.groupby(["expert_seen", "round", "session"])[stats].mean().groupby(["expert_seen", "round"]).agg(["mean", "sem", "count"])


def cells(T: TrialData) -> dict:
    """Trial masks: after-novice trials by expert_seen × round, and each round whole (the note's C5 rows)."""
    r2 = T.round_pm > 0
    out = {f"after novice, expert {'seen' if seen else 'unseen'}, round {r}": ~T.last_exp & (T.expert_seen == seen) & (r2 == (r == 2))
           for seen in (False, True) for r in (1, 2)}
    return out | {f"round {r}": r2 == (r == 2) for r in (1, 2)}


def elpd_by_cell(a: Loo, b: Loo, T: TrialData) -> pd.DataFrame:
    """ΔELPD a − b (sum, SE over participants, share of participants favouring a) on ``cells``; rows with no trials dropped."""
    return by_cell(a, b, T, {n: m for n, m in cells(T).items() if m.any()})
