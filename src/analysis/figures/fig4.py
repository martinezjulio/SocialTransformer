# %% fig4
"""Figure 4, the unseen expert (exploratory). (a) The weight on the expert's belief in each play game laid out in
session order, from the eight-weight switch model (one group weight per demonstrator just watched × round ×
condition), one grey line per condition (solid expert-first, dashed novice-first), markers in the hue of the demonstrator just watched; the games where the
expert has not yet played the bandit (novice-first, first play game of each round) are ringed. (b) What the readout
contains (8.6.2). (c) The exploit statistic: on after-novice trials with the expert unseen, the share of shots on
arms where the novice succeeded, by round, beside chance (the share of such arms) and the share on arms the novice
tried at all; mean and 95% CI over participants.
Inputs: the cached Switch-8 fit and ``unseen/exploit.csv``.
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    ax.set(xticks=range(1, 5), xticklabels=[f"R{(g + 1) // 2}\nP{2 - g % 2}" for g in range(1, 5)], xlim=(0.5, 4.5), ylim=(0, 1),
           xlabel="play game (round, play)", ylabel="weight on the expert's belief")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color="#8C8C8C", linestyle=ls, linewidth=1.0, label=c) for c, ls in zip(ORDER[:2], ("-", "--"))]
    handles += [Line2D([], [], linestyle="none", marker="o", color=COLOR[w], label=f"after the {w}") for w in ("expert", "novice")]
    handles += [Line2D([], [], linestyle="none", marker="o", color="white", markeredgecolor=COLOR["human"], markeredgewidth=1.4, label="expert unseen")]
    ax.legend(handles=handles, loc="lower right", ncol=1, handletextpad=0.3, handlelength=1.2)


def panel_exploit(ax, analysis_dir: Path) -> None:
    """(c) Exploit, chance and tried in the unseen-expert cell by round: mean and 95% CI over participants."""
    t = pd.read_csv(Path(analysis_dir) / "unseen" / "exploit.csv", header=[0, 1], index_col=[0, 1]).loc[False]
    for stat, (col, mk) in {"exploit": (COLOR["human"], "o"), "tried": (COLOR["novice"], "s"), "chance": ("#8C8C8C", "x")}.items():
        m, se = t[(stat, "mean")].to_numpy(), t[(stat, "sem")].to_numpy()
        ax.errorbar([1, 2], m, yerr=1.96 * se, marker=mk, color=col, capsize=2, label={"exploit": "novice's successful arms", "tried": "arms the novice tried", "chance": "chance"}[stat])
    ax.set(xticks=[1, 2], xticklabels=["round 1", "round 2"], xlim=(0.6, 2.4), ylim=(0, 1), ylabel="share of shots")
    ax.legend(loc="upper left", handletextpad=0.3, handlelength=1.2)


def fig4(analysis_dir: Path) -> list:
    """Panels a and c for now (b and the assembly in 8.6.2); each saved on its own."""
    apply()
    paths = []
    for panel, name, h in ((panel_weights, "fig4a_weights", 2.6), (panel_exploit, "fig4c_exploit", 2.4)):
        f, ax = plt.subplots(figsize=(WIDTH["single"], h))
        panel(ax, analysis_dir)
        paths += save(f, name, analysis_dir)
    return paths


if __name__ == "__main__":
    import os
    print(fig4(Path(os.environ["OUTPUT_DATA_DIR"]) / "analysis" / os.environ["VERSION"]))
