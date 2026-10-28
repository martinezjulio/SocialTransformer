# %% fig3
"""Figure 3, policy models. (a) Whose belief: ΔELPD of H+Both over the recent-readout (S1a) and novice-readout (S1b)
models, whole data. (b) A readout, not counts: ΔELPD of H+Both over the visible-pulls models of the last demonstration
(S2a) and of the expert (S2b) and over the naive pooled count (S2c), by round. (c) Ablations: ΔELPD of H+Both over
each rung, with H+Social minus H+Own (S3a) as its own row. (d) The weight on the expert's belief after the expert (ω_E)
and after the novice (ω_N) in round 1 and 2 under Switch-by-round, group level: median and 95% ETI (S4 is the rise in
ω_N). Marks in (a–c): point at the paired ΔELPD with a 2 SE bar, the registered margin; filled where the margin is cleared.
Inputs: ``choice/{ladder,ladder_by_round,verdicts}.csv`` and the cached Switch-by-round fit.
"""

from pathlib import Path

import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.figures.style import COLOR, WIDTH, apply, save

SOURCES = {"src:recent": "recent readout (S1a)", "src:novice": "novice readout (S1b)"}
COUNTS = {"vis:last": "last demonstration's\npulls (S2a)", "vis:expert": "expert's pulls (S2b)", "src:naive": "every pull seen (S2c)"}
ROUND = {"round 1": dict(marker="o", dy=0.18), "round 2": dict(marker="s", dy=-0.18)}
RUNGS = {"H+Social": "H+Social (no own)", "H+Own": "H+Own (no social, S3b)", "H": "H (habit only)", "Both": "Both (no habit)",
         "Social": "Social", "Own": "Own", "Random": "Random"}
XLABEL = "ΔELPD, H+Both minus the rival"


def dot_whisker(ax, y, delta, se, marker="o", label=None):
    """One mark: the ΔELPD with its 2 SE bar; filled when the margin is cleared."""
    ax.errorbar(delta, y, xerr=2 * se, linestyle="none", color=COLOR["human"], capsize=2, zorder=2)
    ax.scatter(delta, y, marker=marker, s=28, color=COLOR["human"], facecolor=COLOR["human"] if delta > 2 * se else "white", zorder=3, label=label)


def panel_source(ax, analysis_dir: Path) -> None:
    """(a) H+Both minus each other readout source, whole data."""
    lad = pd.read_csv(Path(analysis_dir) / "choice" / "ladder.csv", index_col=0)
    for y, (m, lab) in enumerate(reversed(SOURCES.items())):
        dot_whisker(ax, y, -lad.loc[m, "delta"], lad.loc[m, "se"])
    ax.set_yticks(range(len(SOURCES)), [lab for lab in reversed(SOURCES.values())])
    ax.set_ylim(-0.6, len(SOURCES) - 0.4)
    ax.axvline(0, color="#BBBBBB", linewidth=0.8, zorder=1)
    ax.set_xlabel(XLABEL)


def panel_counts(ax, analysis_dir: Path) -> None:
    """(b) H+Both minus each counting model, round 1 (circle) and round 2 (square)."""
    lad = pd.read_csv(Path(analysis_dir) / "choice" / "ladder_by_round.csv", index_col=[0, 1])
    for y, (m, lab) in enumerate(reversed(COUNTS.items())):
        for r, st in ROUND.items():
            dot_whisker(ax, y + st["dy"], -lad.loc[(m, r), "delta"], lad.loc[(m, r), "se"], st["marker"], r if y == 0 else None)
    ax.set_yticks(range(len(COUNTS)), [lab for lab in reversed(COUNTS.values())])
    ax.axvline(0, color="#BBBBBB", linewidth=0.8, zorder=1)
    ax.set_xlabel(XLABEL)
    ax.legend(loc="lower right", handletextpad=0.3)


def panel_ablation(ax, analysis_dir: Path) -> None:
    """(c) H+Both minus each rung of the ladder, and H+Social minus H+Own (S3a) from the verdicts."""
    lad = pd.read_csv(Path(analysis_dir) / "choice" / "ladder.csv", index_col=0)
    v = pd.read_csv(Path(analysis_dir) / "choice" / "verdicts.csv", index_col=0).loc["S3a"]
    rows = [(lab, -lad.loc[m, "delta"], lad.loc[m, "se"]) for m, lab in RUNGS.items()] + [("H+Social minus H+Own (S3a)", v["estimate"], (v["hi"] - v["lo"]) / 4)]
    for y, (lab, delta, se) in enumerate(reversed(rows)):
        dot_whisker(ax, y, delta, se)
    ax.set_yticks(range(len(rows)), [lab for lab, *_ in reversed(rows)])
    ax.axvline(0, color="#BBBBBB", linewidth=0.8, zorder=1)
    ax.set_xlabel(XLABEL)


def panel_omega(ax, analysis_dir: Path) -> None:
    """(d) Group-level ω after the expert and after the novice by round: logit ω_s(r) = μ_s + δ_s · (±½); median, 95% ETI."""
    fit = pickle.load(open(max((Path(analysis_dir) / "choice" / "fits").glob("Switch-by-round_*.pkl"), key=lambda q: q.stat().st_mtime), "rb"))
    for s, who, dx in (("E", "expert", -0.06), ("N", "novice", 0.06)):
        mu, d = fit.posterior[f"mu_logit_omega_{s}"].reshape(-1), fit.posterior[f"delta_{s}"].reshape(-1)
        om = [1 / (1 + np.exp(-(mu + d * pm))) for pm in (-0.5, 0.5)]
        med, lo, hi = (np.array([np.percentile(o, q) for o in om]) for q in (50, 2.5, 97.5))
        ax.errorbar(np.array([1, 2]) + dx, med, yerr=[med - lo, hi - med], marker="o", color=COLOR[who], capsize=2, label=f"after the {who} (ω_{s})")
    ax.set(xticks=[1, 2], xticklabels=["round 1", "round 2"], xlim=(0.6, 2.4), ylim=(0, 1), ylabel="weight on the expert's belief")
    ax.axhline(0.5, color="#BBBBBB", linewidth=0.8, zorder=0)
    ax.legend(loc="lower right", handletextpad=0.3)


def fig3(analysis_dir: Path) -> list:
    """Assemble Figure 3 (a, b | c, d) and save ``fig3_policy`` plus the panels on their own."""
    apply()
    fig = plt.figure(figsize=(WIDTH["double"], 4.6))
    outer = fig.add_gridspec(1, 2, wspace=0.95)   # each column splits its rows to its panels' needs: a is short, c is tall
    cols = [outer[0].subgridspec(2, 1, height_ratios=[1.3, 3], hspace=0.8), outer[1].subgridspec(2, 1, height_ratios=[2.6, 2.4], hspace=0.6)]
    axes = [fig.add_subplot(g[i]) for g in cols for i in range(2)]
    panels = {"a": panel_source, "b": panel_counts, "c": panel_ablation, "d": panel_omega}
    for ax, (letter, panel) in zip(axes, panels.items()):
        panel(ax, analysis_dir)
        ax.set_title(letter, loc="left", fontweight="bold")
    paths = save(fig, "fig3_policy", analysis_dir)
    for letter, panel, name, h in (("a", panel_source, "fig3a_source", 1.6), ("b", panel_counts, "fig3b_counts", 2.4), ("c", panel_ablation, "fig3c_ablation", 2.6), ("d", panel_omega, "fig3d_omega", 2.4)):
        f, ax = plt.subplots(figsize=(WIDTH["single"], h))
        panel(ax, analysis_dir)
        paths += save(f, name, analysis_dir)
    return paths


if __name__ == "__main__":
    import os
    print(fig3(Path(os.environ["OUTPUT_DATA_DIR"]) / "analysis" / os.environ["VERSION"]))
