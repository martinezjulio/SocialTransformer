# %% supp
"""Supplementary figures, one file each (``figS_<name>``). Part 1, from the choice and unseen stages: the full ladder
(every model against HOS, whole data) with the S1/S2 rivals by round; the per-participant ΔELPD behind every paired SE
of the registered comparisons; the ELPD split and the copying comparison on the after-novice cells. Part 2 (8.7.2b):
robustness fits, power curves, exclusion counts, solo slots 1 and 3, recency. Part 3 (8.7.3): posteriors. Part 4
(8.7.4): PPC bands, recovery coverage, split-half. Names: H habit, O own, S social (HOS = H+Both).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.figures.fig3 import COUNTS, SHORT, SOURCES, dot_whisker, headed_rows
from analysis.figures.style import COLOR, WIDTH, apply, save

NAME = {**SOURCES, **COUNTS, **{m: s for m, s in SHORT.items() if m != "H+Both"}, "H+Both+persev": "HOS + perseveration", "H+Both-reset": "HOS, no reset",
        "H+Both+eps": "HOS + ε free", "Switch-by-round+eps": "Switch-by-round + ε free", "Switch-by-round+ktrend": "Switch-by-round + κ trend"}   # registry name -> figure name
REGISTERED = {"recent": ("src:recent",), "novice": ("src:novice",), "naive": ("src:naive",), "naive-recent": ("vis:last",), "naive-expert": ("vis:expert",),
              "HOS − HO": ("H+Own",), "HS − HO": ("H+Social", "H+Own")}   # per-participant ΔELPD: HOS minus the model, or the difference of two


def fig_ladder(analysis_dir: Path) -> list:
    """Every model against HOS on all trials (left), and the S1/S2 rivals by round (right)."""
    lad = pd.read_csv(Path(analysis_dir) / "choice" / "ladder.csv", index_col=0).drop(index="H+Both")
    byr = pd.read_csv(Path(analysis_dir) / "choice" / "ladder_by_round.csv", index_col=[0, 1])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(WIDTH["double"], 4.0), width_ratios=[1.3, 1], gridspec_kw=dict(wspace=0.9))
    ys = headed_rows(ax1, {"HOS minus the model": [NAME.get(m, m) for m in lad.index]})
    for y, m in zip(ys, lad.index):
        dot_whisker(ax1, y, -lad.loc[m, "delta"], lad.loc[m, "se"])
    rivals = list(SOURCES) + list(COUNTS)
    ys = headed_rows(ax2, {"by round": [NAME[m] for m in rivals]})
    for y, m in zip(ys, rivals):
        for r, dy, mk in (("round 1", -0.18, "o"), ("round 2", 0.18, "s")):
            dot_whisker(ax2, y + dy, -byr.loc[(m, r), "delta"], byr.loc[(m, r), "se"], mk, r if m == rivals[0] else None)
    for ax in (ax1, ax2):
        ax.axvline(0, color="#BBBBBB", linewidth=0.8, zorder=1)
        ax.set_xlabel("ΔELPD, HOS minus the model")
    ax2.legend(loc="upper right", handletextpad=0.3)
    return save(fig, "figS_ladder", analysis_dir)


def fig_per_participant(analysis_dir: Path) -> list:
    """The per-participant ΔELPD behind each registered comparison's paired SE: dots per participant, the mean as a bar."""
    e = pd.read_csv(Path(analysis_dir) / "choice" / "elpd_per_participant.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(WIDTH["single"], 2.8))
    rng = np.random.default_rng(0)
    ys = headed_rows(ax, {"belief rivals (HOS −)": list(REGISTERED)[:5], "ablations": list(REGISTERED)[5:]})
    for y, (lab, ms) in zip(ys, REGISTERED.items()):
        d = -e[ms[0]] if len(ms) == 1 else e[ms[0]] - e[ms[1]]   # the file holds model − HOS: HOS − model, or (HS − HOS) − (HO − HOS) = HS − HO
        ax.scatter(d, y + rng.uniform(-0.18, 0.18, len(d)), s=6, color=COLOR["model"], alpha=0.8, linewidths=0, zorder=2)
        ax.plot([d.mean()] * 2, [y - 0.3, y + 0.3], color=COLOR["human"], linewidth=1.4, zorder=3)
    ax.axvline(0, color="#BBBBBB", linewidth=0.8, zorder=1)
    ax.set_xlabel("ΔELPD per participant (bar: mean)")
    return save(fig, "figS_per_participant", analysis_dir)


def fig_split(analysis_dir: Path) -> list:
    """The ELPD split (expert vs recent readout) and the copying comparison (expert readout vs naive-recent) on the
    after-novice cells by expert seen × round, and the rounds whole."""
    fig, ax = plt.subplots(figsize=(WIDTH["single"], 2.8))
    tabs = {n: pd.read_csv(Path(analysis_dir) / "unseen" / f"{f}.csv", index_col=0) for n, f in (("expert readout minus recent (S1a by cell)", "elpd_split"), ("expert readout minus naive-recent (S2a by cell)", "copying_elpd"))}
    ys = headed_rows(ax, {n: [i.replace("after novice, ", "") for i in t.index] for n, t in tabs.items()})
    for y, (n, t) in zip([ys[:6], ys[6:]], tabs.items()):
        for yi, (_, r) in zip(y, t.iterrows()):
            dot_whisker(ax, yi, r["delta"], r["se"])
    ax.axvline(0, color="#BBBBBB", linewidth=0.8, zorder=1)
    ax.set_xlabel("ΔELPD on the cell's trials")
    return save(fig, "figS_unseen_split", analysis_dir)


def supp(analysis_dir: Path) -> list:
    apply()
    return fig_ladder(analysis_dir) + fig_per_participant(analysis_dir) + fig_split(analysis_dir)


if __name__ == "__main__":
    import os
    print(supp(Path(os.environ["OUTPUT_DATA_DIR"]) / "analysis" / os.environ["VERSION"]))
