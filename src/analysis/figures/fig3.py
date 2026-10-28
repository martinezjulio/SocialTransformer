# %% fig3
"""Figure 3, policy models. (a) Whose belief: ΔELPD of H+Both over the recent-readout (S1a) and novice-readout (S1b)
models, whole data. (b) A readout, not counts: ΔELPD of H+Both over the visible-pulls models of the last demonstration
(S2a) and of the expert (S2b) and over the naive pooled count (S2c), by round. Marks: point at the paired ΔELPD with a
2 SE bar, the registered margin; filled where the rival is beaten by more than 2 SE. Panels (c, d) follow in 8.5.
Inputs: ``choice/ladder.csv`` and ``choice/ladder_by_round.csv``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analysis.figures.style import COLOR, WIDTH, apply, save

SOURCES = {"src:recent": "recent readout (S1a)", "src:novice": "novice readout (S1b)"}
COUNTS = {"vis:last": "last demonstration's\npulls (S2a)", "vis:expert": "expert's pulls (S2b)", "src:naive": "every pull seen (S2c)"}
ROUND = {"round 1": dict(marker="o", dy=0.18), "round 2": dict(marker="s", dy=-0.18)}
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


def fig3(analysis_dir: Path) -> list:
    """Assemble Figure 3 (a, b for now) and save ``fig3_policy`` plus the panels on their own."""
    apply()
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(WIDTH["single"], 4.0), height_ratios=[1.3, 3], gridspec_kw=dict(hspace=0.8))
    panel_source(ax_a, analysis_dir)
    panel_counts(ax_b, analysis_dir)
    for ax, letter in ((ax_a, "a"), (ax_b, "b")):
        ax.set_title(letter, loc="left", fontweight="bold")
    paths = save(fig, "fig3_policy", analysis_dir)
    fa, ax = plt.subplots(figsize=(WIDTH["single"], 1.6))
    panel_source(ax, analysis_dir)
    fb, ax = plt.subplots(figsize=(WIDTH["single"], 2.4))
    panel_counts(ax, analysis_dir)
    return paths + save(fa, "fig3a_source", analysis_dir) + save(fb, "fig3b_counts", analysis_dir)


if __name__ == "__main__":
    import os
    print(fig3(Path(os.environ["OUTPUT_DATA_DIR"]) / "analysis" / os.environ["VERSION"]))
