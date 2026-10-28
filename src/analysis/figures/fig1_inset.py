# %% fig1_inset
"""Figure 1(c) inset: one real readout from a pilot session, rendered by the pipeline. For three arms of one bandit
(payoffs 0.8, 0.55 and 0.0 of the fixed payoff set) the expert's and the novice's Beta beliefs entering a round-1
play game after the expert has demonstrated (the game whose expert readout is closest to the truth on those arms), drawn as
densities over P(success) in three shades of each demonstrator's hue, the arms' true payoffs as ticks in the same
shades; the arms are labelled 1, 2, 3 from the worst payoff to the best. The expert's belief is tight on the best arm;
the novice's is flat on every arm.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

from analysis.data import TrialData, load_sessions, trial_data
from analysis.figures.style import COLOR, WIDTH, apply, save

SHADES = {"expert": ["#0072B2", "#4A9BD0", "#9DC6E2"], "novice": ["#D55E00", "#E8945A", "#F0B48F"]}   # dark → light = best → poor arm
PAYOFF_COLOR = "#555555"   # the arms' true payoffs: ticks, their labels and the arm names (dark grey: ground truth, no demonstrator)


PAYOFFS = (0.8, 0.55, 0.0)   # the arms shown: the best, the fourth-best and the poorest of the fixed payoff set


def three_arms(T: TrialData, i: int) -> list:
    """The arms of trial ``i``'s bandit carrying the payoffs in PAYOFFS (every bandit permutes one payoff set)."""
    return [int(np.argmin(np.abs(T.theta[i] - p))) for p in PAYOFFS]


def pick(T: TrialData) -> int:
    """First shot of the round-1 play game, expert already seen, whose expert readout means on the three arms are
    closest to the truth (sum of absolute errors); ties go to the earliest session in sorted order."""
    first = np.flatnonzero((T.prev_arm == -1) & T.expert_seen & (T.seg <= 2))
    err = [np.abs(T.read["expert"][i, a, 0] / T.read["expert"][i, a].sum(1) - T.theta[i, a]).sum() for i, a in ((i, three_arms(T, i)) for i in first)]
    order = np.lexsort((T.session[first], np.round(err, 6)))
    return int(first[order[0]])


def panel_readout(ax, T: TrialData, i: int) -> None:
    arms = three_arms(T, i)[::-1]                                      # labelled 1, 2, 3 from the worst payoff to the best
    name = {k: str(n + 1) for n, k in enumerate(arms)}
    x = np.linspace(0.001, 0.999, 400)
    for who in ("expert", "novice"):
        for shade, k in zip(SHADES[who][::-1], arms):
            a, b = T.read[who][i, k]
            pdf = beta.pdf(x, a, b)
            ax.fill_between(x, pdf, color=shade, alpha=0.25, linewidth=0)
            ax.plot(x, pdf, color=shade, linewidth=0.7, label=f"{who}, arm {name[k]}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, None)
    ax.spines["left"].set_position(("outward", 6))                   # x = 0 sits inside the frame, clear of the y-axis
    top = ax.get_ylim()[1]
    for k in arms:   # the arms' true payoffs, once, as ticks rising from the baseline with the arm name beside
        ax.plot([T.theta[i, k]] * 2, [0, 0.07 * top], color=PAYOFF_COLOR, linewidth=1.5, zorder=5)
        ax.annotate(f"arm {name[k]}", (T.theta[i, k], 0.0), xytext=(3, 2), textcoords="offset points", ha="left", va="bottom", fontsize=plt.rcParams["xtick.labelsize"] - 1, color=PAYOFF_COLOR)
    ax.set_xticks(sorted(PAYOFFS) + [1.0], [f"{v:g}" if v != 0 else "0.0" for v in sorted(PAYOFFS)] + ["1.0"])
    for label in ax.get_xticklabels()[:len(PAYOFFS)]:
        label.set_color(PAYOFF_COLOR)
    ax.set_xlabel("P(success); coloured ticks: true payoffs")
    ax.set_ylabel("belief density")
    ax.set_yticks([])
    handles, labels = ax.get_legend_handles_labels()
    order = [2, 1, 0, 5, 4, 3]                                         # legend from the best arm down, expert then novice
    ax.legend([handles[j] for j in order], [labels[j] for j in order], loc="upper left", ncol=1, handlelength=1.2, handletextpad=0.4, fontsize=plt.rcParams["legend.fontsize"] - 1)


def fig1_inset(eval_dir: Path, version: str, analysis_dir: Path) -> list:
    apply()
    T = trial_data(load_sessions(eval_dir, version))
    fig, ax = plt.subplots(figsize=(WIDTH["single"], 2.3))
    panel_readout(ax, T, pick(T))
    return save(fig, "fig1c_readout_inset", analysis_dir)


if __name__ == "__main__":
    import os
    root, version = Path(os.environ["OUTPUT_DATA_DIR"]), os.environ["VERSION"]
    print(fig1_inset(root / "eval" / version, version, root / "analysis" / version))
