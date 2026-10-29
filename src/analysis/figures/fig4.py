# %% fig4
"""Figure 4, the unseen expert (exploratory). (a) The weight on the expert's belief in each play game laid out in
session order, from the eight-weight switch model (one group weight per demonstrator just watched × round ×
condition), one grey line per condition (solid expert-first, dashed novice-first), markers in the hue of the demonstrator just watched; the games where the
expert has not yet played the bandit (novice-first, first play game of each round) are ringed. (b) What the readout
contains (8.6.2). (c) The exploit statistic: on after-novice trials with the expert unseen, the share of shots on
arms where the novice succeeded, by round, beside chance (the share of such arms) and the share on arms the novice
tried at all; mean and 95% CI over participants.
Inputs: the cached Switch-8 fit, ``unseen/{readout_table,exploit}.csv`` and the inference CSVs (inset).
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import beta

from analysis.data import TrialData, load_sessions, trial_data
from analysis.figures.fig1_inset import PAYOFFS, SHADES, three_arms
from analysis.figures.style import COLOR, ORDER, WIDTH, apply, save

GAMES = {"expert-first": "ENEN", "novice-first": "NENE"}   # the demonstrator watched before play games 1–4
UNSEEN = {"expert-first": (), "novice-first": (1, 3)}     # games on a bandit the expert has not played yet
WHO = {"E": "expert", "N": "novice"}


def omega8(fit) -> np.ndarray:
    """(draws, 8) group-level weights of Switch-8; index = (after the expert) * 4 + (round 2) * 2 + expert-first."""
    return 1 / (1 + np.exp(-fit.posterior["logit_omega_8"].reshape(-1, 8)))


def panel_weights(ax, analysis_dir: Path) -> None:
    """(a) Median and 95% ETI of each game's weight, in session order, by condition."""
    fit = pickle.load(open(max((Path(analysis_dir) / "choice" / "fits").glob("Switch-8_*.pkl"), key=lambda q: q.stat().st_mtime), "rb"))
    om = omega8(fit)
    for cond, dx, ls in (("expert-first", -0.08, "-"), ("novice-first", 0.08, "--")):
        idx = [(w == "E") * 4 + (g >= 3) * 2 + (cond == "expert-first") for g, w in enumerate(GAMES[cond], 1)]
        q = np.percentile(om[:, idx], [50, 2.5, 97.5], axis=0)
        x = np.arange(1, 5) + dx
        ax.plot(x, q[0], color="#8C8C8C", linestyle=ls, linewidth=1.0, zorder=1)   # the condition is the line style; the hue is who was just watched
        for g, (xi, w) in enumerate(zip(x, GAMES[cond]), 1):
            ax.errorbar(xi, q[0, g - 1], yerr=[[q[0, g - 1] - q[1, g - 1]], [q[2, g - 1] - q[0, g - 1]]], marker="o", color=COLOR[WHO[w]], capsize=2, zorder=3,
                        markeredgecolor=COLOR["human"] if g in UNSEEN[cond] else COLOR[WHO[w]], markeredgewidth=1.4 if g in UNSEEN[cond] else 0.0)
    ax.axhline(0.5, color="#BBBBBB", linewidth=0.8, zorder=0)
    ax.set(xticks=range(1, 5), xticklabels=["P1", "P2"] * 2, xlim=(0.5, 4.5), ylim=(0, 1), ylabel="weight on the expert's belief")
    ax.axvline(2.5, color="#BBBBBB", linewidth=0.8, linestyle=(0, (2, 2)), zorder=0)
    for x, lab in ((1.5, "round 1"), (3.5, "round 2")):   # the round under its two play games
        ax.annotate(lab, (x, 0), xytext=(0, -22), textcoords="offset points", ha="center", va="top", annotation_clip=False)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color="#8C8C8C", linestyle=ls, linewidth=1.0, label=c) for c, ls in zip(ORDER[:2], ("-", "--"))]
    handles += [Line2D([], [], linestyle="none", marker="o", color=COLOR[w], label=f"after the {w}") for w in ("expert", "novice")]
    handles += [Line2D([], [], linestyle="none", marker="o", color="white", markeredgecolor=COLOR["human"], markeredgewidth=1.4, label="expert unseen")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.25, -0.3), ncol=5, handletextpad=0.3, handlelength=1.2, columnspacing=0.8)   # one line below the axis: the data fill the panel


def panel_readout(axes, analysis_dir: Path) -> None:
    """(b) Confidence and favourite-arm payoff of the expert readout by round, expert unseen vs seen; novice readout in grey."""
    t = pd.read_csv(Path(analysis_dir) / "unseen" / "readout_table.csv", header=[0, 1], index_col=[0, 1, 2])
    for ax, stat, lab in zip(axes, ("confidence", "favourite_payoff"), ("readout confidence (α + β)", "favourite arm's true payoff")):
        for who, seen, col, ls, name in (("expert", False, COLOR["expert"], "--", "expert\nunseen"), ("expert", True, COLOR["expert"], "-", "expert\nseen"),
                                         ("novice", True, "#8C8C8C", "-", "novice")):   # the expert readout, with the expert unseen or seen; the novice readout
            r = t.loc[(who, seen)]
            ax.errorbar([1, 2], r[(stat, "mean")], yerr=1.96 * r[(stat, "sem")], marker="o", color=col, linestyle=ls, capsize=2,
                        markerfacecolor="white" if not seen else col, markeredgewidth=1.4)   # hollow = expert unseen, as the rings in (a)
            ax.annotate(name, (2, r[(stat, "mean")].iloc[-1]), xytext=(6, 0), textcoords="offset points", ha="left", va="center", color=col, fontsize=plt.rcParams["legend.fontsize"], annotation_clip=False)
        ax.set(xticks=[1, 2], xticklabels=["round 1", "round 2"], xlim=(0.6, 2.4), ylabel=lab)


def unseen_pair(T: TrialData) -> tuple:
    """First shots of round-1 play games 1 and 2 of the novice-first session whose expert readout after the expert's
    demonstration is closest to the truth on the three shown arms (the same bandit in both games)."""
    after = np.flatnonzero((T.prev_arm == -1) & ~T.expert_first & (T.seg == 2))
    err = [np.abs(T.read["expert"][i, a, 0] / T.read["expert"][i, a].sum(1) - T.theta[i, a]).sum() for i, a in ((i, three_arms(T, i)) for i in after)]
    i2 = int(after[np.lexsort((T.session[after], np.round(err, 6)))[0]])
    i1 = int(np.flatnonzero((T.session == T.session[i2]) & (T.prev_arm == -1) & (T.seg == 1))[0])
    return i1, i2


def inset_readout(ax, T: TrialData, i1: int, i2: int) -> None:
    """The expert readout's Beta densities on three arms before (dashed) and after (solid) the expert's demonstration."""
    arms = three_arms(T, i2)[::-1]
    x = np.linspace(0.001, 0.999, 400)
    for shade, k in zip(SHADES["expert"][::-1], arms):
        for i, ls in ((i1, "--"), (i2, "-")):
            ax.plot(x, beta.pdf(x, *T.read["expert"][i, k]), color=shade, linestyle=ls, linewidth=0.7)
        ax.plot([T.theta[i2, k]] * 2, [0, 0.07 * ax.get_ylim()[1]], color="#555555", linewidth=1.5, zorder=5)
    ax.set(xlim=(0, 1), ylim=(0, None), yticks=[], xlabel="P(success)", ylabel="belief density", xticks=sorted(PAYOFFS) + [1.0])
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=COLOR["expert"], linestyle="--", label="expert unseen"), Line2D([], [], color=COLOR["expert"], label="expert seen")],
              loc="upper left", handlelength=1.6, handletextpad=0.4, fontsize=plt.rcParams["legend.fontsize"] - 1)


def panel_exploit(ax, analysis_dir: Path) -> None:
    """(c) Exploit, chance and tried in the unseen-expert cell by round: mean and 95% CI over participants."""
    t = pd.read_csv(Path(analysis_dir) / "unseen" / "exploit.csv", header=[0, 1], index_col=[0, 1]).loc[False]
    for stat, (col, mk) in {"exploit": (COLOR["human"], "o"), "tried": (COLOR["novice"], "s"), "chance": ("#8C8C8C", "x")}.items():
        m, se = t[(stat, "mean")].to_numpy(), t[(stat, "sem")].to_numpy()
        ax.errorbar([1, 2], m, yerr=1.96 * se, marker=mk, color=col, capsize=2, label={"exploit": "novice's successful arms", "tried": "arms the novice tried", "chance": "chance"}[stat])
    ax.set(xticks=[1, 2], xticklabels=["round 1", "round 2"], xlim=(0.6, 2.4), ylim=(0, 1), ylabel="share of shots")
    ax.legend(loc="upper left", handletextpad=0.3, handlelength=1.2)


def fig4(analysis_dir: Path, eval_dir: Path, version: str) -> list:
    """Assemble Figure 4 (a | b, b; one session | c) and save ``fig4_unseen`` plus the panels on their own."""
    apply()
    T = trial_data(load_sessions(eval_dir, version))
    i1, i2 = unseen_pair(T)
    fig = plt.figure(figsize=(WIDTH["double"], 4.8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.3, 1, 1], height_ratios=[1.15, 1], hspace=0.72, wspace=0.75)
    ax_a, ax_b1, ax_b2, ax_in, ax_c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1:])
    panel_weights(ax_a, analysis_dir)
    panel_readout((ax_b1, ax_b2), analysis_dir)
    inset_readout(ax_in, T, i1, i2)
    panel_exploit(ax_c, analysis_dir)
    for ax, letter in ((ax_a, "a"), (ax_b1, "b"), (ax_in, "b, one session"), (ax_c, "c")):
        ax.set_title(letter, loc="left", fontweight="bold")
    paths = save(fig, "fig4_unseen", analysis_dir)
    for panel, name, h in ((panel_weights, "fig4a_weights", 2.6), (panel_exploit, "fig4c_exploit", 2.4)):
        f, ax = plt.subplots(figsize=(WIDTH["single"], h))
        panel(ax, analysis_dir)
        paths += save(f, name, analysis_dir)
    f, axes = plt.subplots(1, 2, figsize=(WIDTH["single"], 2.2), gridspec_kw=dict(wspace=0.6))
    panel_readout(axes, analysis_dir)
    paths += save(f, "fig4b_readout", analysis_dir)
    f, ax = plt.subplots(figsize=(WIDTH["single"] * 0.6, 1.8))
    inset_readout(ax, T, i1, i2)
    return paths + save(f, "fig4b_inset", analysis_dir)


if __name__ == "__main__":
    import os
    v = os.environ["VERSION"]
    print(fig4(Path(os.environ["OUTPUT_DATA_DIR"]) / "analysis" / v, Path(os.environ["OUTPUT_DATA_DIR"]) / "eval" / v, v))
