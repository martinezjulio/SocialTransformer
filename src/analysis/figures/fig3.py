# %% fig3
"""Figure 3, policy models. Names: H habit, O own experience, S social belief; HOS is the full model (registry H+Both).
(a) Whose belief, and a readout, not counts: every model is HOS with a different belief; the
expert readout minus the recent (S1a) and novice (S1b) readouts, and minus the visible-pulls models of the last demonstration
(S2a) and of the expert (S2b) and over the naive pooled count (S2c); one panel, all trials, as the claims are judged. (b) Ablations: ΔELPD of H+Both over
each rung, with H+Social minus H+Own (S3a) as its own row. (c) The weight on the expert's belief after the expert (ω_E)
and after the novice (ω_N) in round 1 and 2 under Switch-by-round, group level: median and 95% ETI (S4 is the rise in
ω_N). Marks in (a, b): a point at the paired ΔELPD with a 2 SE bar, the registered margin; filled where the margin is cleared.
Inputs: ``choice/{ladder,verdicts}.csv`` and the cached Switch-by-round fit.
"""

from pathlib import Path

import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.figures.style import COLOR, WIDTH, apply, save

SOURCES = {"src:recent": "recent", "src:novice": "novice"}   # figure names: the readout the belief comes from
COUNTS = {"src:naive": "naive", "vis:last": "naive-recent", "vis:expert": "naive-expert"}   # Beta counts of the pulls seen
SHORT = {"H+Both": "HOS", "H+Social": "HS", "H+Own": "HO", "H": "H", "Both": "OS", "Social": "S", "Own": "O", "Random": "Random"}   # H habit, O own, S social
RUNGS = {m: f"HOS − {SHORT[m]}" for m in ("H+Social", "H+Own", "H", "Both", "Social", "Own", "Random")}


def dot_whisker(ax, y, delta, se, marker="o", label=None):
    """One mark: the ΔELPD with its 2 SE bar; filled when the margin is cleared."""
    ax.errorbar(delta, y, xerr=2 * se, linestyle="none", color=COLOR["human"], capsize=2, zorder=2)
    ax.scatter(delta, y, marker=marker, s=28, color=COLOR["human"], facecolor=COLOR["human"] if delta > 2 * se else "white", zorder=3, label=label)


GROUPS = {"readout source": SOURCES, "Beta counts": COUNTS}


def panel_belief(ax, analysis_dir: Path) -> None:
    """(a) Every model is H+Both with a different belief: the expert readout minus each rival on all trials, with the
    2 SE whisker, as the S1 and S2 parts are judged (the by-round split is a supplementary figure)."""
    lad = pd.read_csv(Path(analysis_dir) / "choice" / "ladder.csv", index_col=0)
    ys = headed_rows(ax, {g: list(labs.values()) for g, labs in GROUPS.items()})
    for y, m in zip(ys, [m for g in GROUPS.values() for m in g]):
        dot_whisker(ax, y, -lad.loc[m, "delta"], lad.loc[m, "se"])
    ax.axvline(0, color="#BBBBBB", linewidth=0.8, zorder=1)
    ax.set_xlabel("ΔELPD, expert readout minus the rival belief (all HOS)")


def headed_rows(ax, groups: dict) -> list:
    """y positions for rows under italic group headers (a gap holds each later header); returns the rows' y values."""
    ys, heads, y = [], {}, 0.0
    for g, labels in groups.items():
        heads[y - 0.9] = g
        ys += [y + i for i in range(len(labels))]
        y += len(labels) + 1.2
    ticks = sorted(list(heads) + ys)
    labels = [lab for labs in groups.values() for lab in labs]
    ax.set_yticks(ticks, [heads.get(t, "") or labels[ys.index(t)].replace("\n", " ") for t in ticks])
    for t, lab in zip(ax.get_yticklabels(), ticks):
        t.set_ha("left")   # left-aligned column: every name starts at the same x
        if lab in heads:
            t.set_fontstyle("italic"), t.set_fontweight("bold")
    ax.figure.canvas.draw()   # the pad is the widest label, measured
    widest = max(t.get_window_extent().width for t in ax.get_yticklabels()) * 72 / ax.figure.dpi
    ax.tick_params(axis="y", length=0, pad=widest + 4)
    ax.spines["left"].set_visible(False)   # the zero line is the reference, not the axis edge
    ax.set_ylim(max(ys) + 0.6, -1.4)
    return ys


def panel_ablation(ax, analysis_dir: Path) -> None:
    """(b) HOS minus each rung of the ladder (S3b is the HO row), and HS minus HO (S3a) from the verdicts."""
    lad = pd.read_csv(Path(analysis_dir) / "choice" / "ladder.csv", index_col=0)
    v = pd.read_csv(Path(analysis_dir) / "choice" / "verdicts.csv", index_col=0).loc["S3a"]
    rows = [(-lad.loc[m, "delta"], lad.loc[m, "se"]) for m in RUNGS] + [(v["estimate"], (v["hi"] - v["lo"]) / 4)]
    ys = headed_rows(ax, {"ablations": list(RUNGS.values()), "social > own": ["HS − HO"]})
    for y, (delta, se) in zip(ys, rows):
        dot_whisker(ax, y, delta, se)
    ax.axvline(0, color="#BBBBBB", linewidth=0.8, zorder=1)
    ax.set_xlabel("ΔELPD")


def panel_omega(ax, analysis_dir: Path) -> None:
    """(d) Group-level ω after the expert and after the novice by round: logit ω_s(r) = μ_s + δ_s · (±½); median, 95% ETI."""
    fit = pickle.load(open(max((Path(analysis_dir) / "choice" / "fits").glob("Switch-by-round_*.pkl"), key=lambda q: q.stat().st_mtime), "rb"))
    for s, who, dx in (("E", "expert", -0.06), ("N", "novice", 0.06)):
        mu, d = fit.posterior[f"mu_logit_omega_{s}"].reshape(-1), fit.posterior[f"delta_{s}"].reshape(-1)
        om = [1 / (1 + np.exp(-(mu + d * pm))) for pm in (-0.5, 0.5)]
        med, lo, hi = (np.array([np.percentile(o, q) for o in om]) for q in (50, 2.5, 97.5))
        ax.errorbar(np.array([1, 2]) + dx, med, yerr=[med - lo, hi - med], marker="o", color=COLOR[who], capsize=2, label=f"after the {who}")
    ax.set(xticks=[1, 2], xticklabels=["round 1", "round 2"], xlim=(0.6, 2.4), ylim=(0, 1), ylabel="weight on the expert's belief")
    ax.axhline(0.5, color="#BBBBBB", linewidth=0.8, zorder=0)
    ax.legend(loc="lower right", handletextpad=0.3, handlelength=1.2, borderaxespad=0.2)


def fig3(analysis_dir: Path) -> list:
    """Assemble Figure 3 (a | c, d) and save ``fig3_policy`` plus the panels on their own."""
    apply()
    fig = plt.figure(figsize=(WIDTH["double"], 3.8))   # low enough that panel a's rows sit close
    outer = fig.add_gridspec(1, 2, wspace=0.95)
    right = outer[1].subgridspec(2, 1, height_ratios=[2.7, 2.3], hspace=0.7)
    axes = [fig.add_subplot(outer[0]), fig.add_subplot(right[0]), fig.add_subplot(right[1])]
    panels = {"a": panel_belief, "b": panel_ablation, "c": panel_omega}
    for ax, (letter, panel) in zip(axes, panels.items()):
        panel(ax, analysis_dir)
        ax.set_title(letter, loc="left", fontweight="bold")
    paths = save(fig, "fig3_policy", analysis_dir)
    for panel, name, h in ((panel_belief, "fig3a_belief", 3.0), (panel_ablation, "fig3b_ablation", 2.6), (panel_omega, "fig3c_omega", 2.4)):
        f, ax = plt.subplots(figsize=(WIDTH["single"], h))
        panel(ax, analysis_dir)
        paths += save(f, name, analysis_dir)
    return paths


if __name__ == "__main__":
    import os
    print(fig3(Path(os.environ["OUTPUT_DATA_DIR"]) / "analysis" / os.environ["VERSION"]))
