# %% elpd
"""ΔELPD figures. Every bar is a model minus ROS (the registered full model, which reads the expert's belief) with its
± 2 paired SE: ROS is the top row at zero and a model that predicts worse is a negative bar, the ``elpd_diff`` of
``loo_compare``. Rows carry the preregistration's model names, best first. The registered ΔELPD (preregistration
@eq-elpd) is ROS minus the rival, the same number with the sign reversed; the numbers and tables keep that direction.

``elpd_S1`` and ``elpd_S2`` (belief rivals), ``elpd_S3`` (ablations, with S3a = RS − RO as a bracket) and
``elpd_ladder`` (every model; the extensions on their own scale; the belief rivals by round). ``panel_elpd`` draws one
set of rows on any axis, so Figure 3 uses the same marks.
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from analysis.figures.style import COLOR, INK, MUTED, apply, model_color, save

REF = "ROS"
SOURCES = ["recent", "novice"]                        # belief rivals whose belief is a readout
COUNTS = ["naive", "naive-recent", "naive-expert"]    # belief rivals whose belief is a Beta count of the pulls seen
ABLATIONS = ["RS", "RO", "R", "OS", "S", "O", "Random"]
REGISTERED = {"S1a": "recent", "S1b": "novice", "S2a": "naive-recent", "S2b": "naive-expert", "S2c": "naive", "S3b": "RO"}   # part: the rival ROS is compared with
XLABEL = "ΔELPD: model minus ROS, ± 2 paired SE"
WHISKER = "#3F3F4A"


def ladder(analysis_dir: Path) -> pd.DataFrame:
    """choice/ladder.csv: ``delta`` is the model's ELPD minus ROS's, ``se`` its paired SE over participants."""
    lad = pd.read_csv(Path(analysis_dir) / "choice" / "ladder.csv", index_col=0)
    assert lad.loc[REF, "delta"] == 0 and np.allclose(lad["delta"], lad["elpd"] - lad.loc[REF, "elpd"]), "delta is not model minus ROS"
    return lad


def verdicts(analysis_dir: Path, lad: pd.DataFrame) -> pd.DataFrame:
    """choice/verdicts.csv, checked against the ladder: a registered part is the negated delta of its rival; S3a is RS − RO."""
    v = pd.read_csv(Path(analysis_dir) / "choice" / "verdicts.csv", index_col=0)
    for part, m in REGISTERED.items():
        assert np.isclose(v.loc[part, "estimate"], -lad.loc[m, "delta"]), (part, m)
    assert np.isclose(v.loc["S3a", "estimate"], lad.loc["RS", "delta"] - lad.loc["RO", "delta"])
    return v


def best_first(lad: pd.DataFrame, models: list) -> list:
    return sorted(models, key=lambda m: -lad.loc[m, "delta"])


def extensions(lad: pd.DataFrame) -> list:
    return [m for m in lad.index if m not in [REF] + SOURCES + COUNTS + ABLATIONS]


def signed(v: float) -> str:
    return f"{v:+,.0f}".replace("-", "−")


def thousands(ax) -> None:
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}".replace("-", "−")))


def reference_row(ax, left: bool, pad: float) -> None:
    """ROS on row 0: a marker at zero (no bar) with its label on the side the bars are on."""
    small = matplotlib.rcParams["font.size"] - 2
    ax.axvline(0, color=INK, linewidth=matplotlib.rcParams["axes.linewidth"], zorder=4)
    ax.plot([0], [0], marker="D", markersize=matplotlib.rcParams["lines.markersize"] * 0.95, color=COLOR["expert"], linestyle="none", zorder=5, clip_on=False)
    ax.text(-pad if left else pad, 0, "0  (reference)", va="center", ha="right" if left else "left", fontsize=small, color=COLOR["expert"])


def panel_elpd(ax, lad: pd.DataFrame, models: list, xlim: tuple, exts: tuple = ()) -> None:
    """ROS on the top row, then one bar per model (``delta`` ± 2 SE) with its value at the bar's outer end."""
    val, se = lad.loc[models, "delta"].to_numpy(float), lad.loc[models, "se"].to_numpy(float)
    font, k = matplotlib.rcParams["font.size"], matplotlib.rcParams["font.size"] / 12
    y = np.arange(1, len(models) + 1)
    ax.set_xlim(*xlim)
    pad = 0.012 * (xlim[1] - xlim[0])
    ax.barh(y, val, height=0.64, color=[model_color(m, exts) for m in models], zorder=2)
    ax.errorbar(val, y, xerr=2 * se, color=WHISKER, linestyle="none", capsize=3 * k, elinewidth=1.3 * k, capthick=1.3 * k, zorder=3)
    for yi, v, s in zip(y, val, se):
        ax.text(v + (2 * s + pad) * (1 if v >= 0 else -1), yi, signed(v), va="center", ha="left" if v >= 0 else "right", fontsize=font - 2, color=WHISKER)
    reference_row(ax, bool((val < 0).all()), 1.6 * pad)
    ax.set_yticks(np.arange(len(models) + 1), [REF] + list(models))
    ax.get_yticklabels()[0].set_color(COLOR["expert"])
    ax.set_ylim(len(models) + 0.55, -0.55)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    thousands(ax)


def panel_belief(ax, analysis_dir: Path, models: list = None) -> None:
    """The belief rivals against ROS (S1: the readout sources; S2: the counts); the same x range for both."""
    lad = ladder(analysis_dir)
    verdicts(analysis_dir, lad)
    panel_elpd(ax, lad, best_first(lad, models or SOURCES + COUNTS), (-2500, 70))
    ax.set_xticks([-2000, -1500, -1000, -500, 0])
    ax.set_xlabel(XLABEL)


def panel_ablation(ax, analysis_dir: Path) -> None:
    """Each ablation against ROS (S3b is the RO bar); S3a, RS − RO, is the bracket between those two bars."""
    lad = ladder(analysis_dir)
    v = verdicts(analysis_dir, lad).loc["S3a"]
    ms = best_first(lad, ABLATIONS)
    assert ms[:2] == ["RS", "RO"], ms
    panel_elpd(ax, lad, ms, (-18500, 400))
    ax.set_xticks([-15000, -10000, -5000, 0])
    a, b = lad.loc["RO", "delta"], lad.loc["RS", "delta"]
    ax.plot([a, a, b, b], [1.62, 1.5, 1.5, 1.38], color=COLOR["expert"], linewidth=matplotlib.rcParams["lines.linewidth"] * 0.6, zorder=6)
    ax.text(a - 2600, 1.5, f"RS − RO = {signed(v['estimate'])}  (± 2 SE: {v['lo']:,.0f} to {v['hi']:,.0f})", va="center", ha="right",
            fontsize=matplotlib.rcParams["font.size"] - 2, color=COLOR["expert"])
    ax.set_xlabel(XLABEL)


def panel_by_round(ax, analysis_dir: Path) -> None:
    """The belief rivals against ROS in each round (lighter: round 1)."""
    lad = ladder(analysis_dir)
    byr = pd.read_csv(Path(analysis_dir) / "choice" / "ladder_by_round.csv", index_col=[0, 1])
    rivals, font, k = best_first(lad, SOURCES + COUNTS), matplotlib.rcParams["font.size"], matplotlib.rcParams["font.size"] / 12
    y = np.arange(1, len(rivals) + 1)
    lo = min(byr.loc[(m, r), "delta"] - 2 * byr.loc[(m, r), "se"] for m in rivals for r in ("round 1", "round 2"))
    ax.set_xlim(lo * 1.22, -lo * 0.02)
    pad = 0.012 * -lo * 1.24
    for r, dy, alpha in (("round 1", -0.2, 0.55), ("round 2", 0.2, 1.0)):
        val = np.array([byr.loc[(m, r), "delta"] for m in rivals])
        se = np.array([byr.loc[(m, r), "se"] for m in rivals])
        assert byr.loc[(REF, r), "delta"] == 0
        ax.barh(y + dy, val, height=0.36, color=[model_color(m) for m in rivals], alpha=alpha, zorder=2)
        ax.errorbar(val, y + dy, xerr=2 * se, color=WHISKER, linestyle="none", capsize=2.5 * k, elinewidth=1.2 * k, capthick=1.2 * k, zorder=3)
        for yi, v, s in zip(y + dy, val, se):
            ax.text(v - 2 * s - pad, yi, f"{r[0].upper()}{r[-1]}  {signed(v)}", va="center", ha="right", fontsize=font - 2, color=WHISKER)
    reference_row(ax, True, 1.6 * pad)
    ax.set_yticks(np.arange(len(rivals) + 1), [REF] + rivals)
    ax.get_yticklabels()[0].set_color(COLOR["expert"])
    ax.set_ylim(len(rivals) + 0.55, -0.55)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    thousands(ax)
    ax.set_xlabel(XLABEL)


def fig_ladder(analysis_dir: Path) -> list:
    """Every model against ROS (left; the extensions on their own scale below) and the belief rivals by round (right)."""
    apply(10.5)
    lad = ladder(analysis_dir)
    exts = best_first(lad, extensions(lad))
    ms = best_first(lad, SOURCES + COUNTS + ABLATIONS)
    fig = plt.figure(figsize=(11.83, 5.3))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1], height_ratios=[len(ms) + 1, max(len(exts), 1) + 1.6], hspace=0.42, wspace=0.3)
    ax1, ax3, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])
    panel_elpd(ax1, lad, ms, (-18500, 400))
    ax1.set_xticks([-15000, -10000, -5000, 0])
    ax1.set_title("belief rivals and ablations", fontsize=10, loc="left", color=MUTED)
    if exts:
        hi, lo = (lad.loc[exts, "delta"] + 2 * lad.loc[exts, "se"]).max(), (lad.loc[exts, "delta"] - 2 * lad.loc[exts, "se"]).min()
        panel_elpd(ax3, lad, exts, (min(lo, 0) - 0.13 * (hi - min(lo, 0)), hi * 1.13), tuple(exts))
    ax3.set_title("extensions, on their own scale (positive: the extension predicts better than ROS)", fontsize=10, loc="left", color=MUTED)
    ax3.set_xlabel(XLABEL)
    panel_by_round(ax2, analysis_dir)
    ax2.set_title("belief rivals by round (lighter: round 1)", fontsize=10, loc="left", color=MUTED)
    return save(fig, "elpd_ladder", analysis_dir)


def elpd(analysis_dir: Path) -> list:
    """``elpd_S1``, ``elpd_S2``, ``elpd_S3`` and ``elpd_ladder``."""
    paths = []
    for name, models, size, font in (("elpd_S1", SOURCES, (5.1, 3.95), 12), ("elpd_S2", COUNTS, (5.1, 3.95), 12), ("elpd_S3", None, (5.83, 4.2), 11.5)):
        apply(font)
        fig, ax = plt.subplots(figsize=size)
        panel_ablation(ax, analysis_dir) if models is None else panel_belief(ax, analysis_dir, models)
        ax.set_xlabel(XLABEL + "\n(0 = ROS, more negative = worse)")
        paths += save(fig, name, analysis_dir)
    return paths + fig_ladder(analysis_dir)


if __name__ == "__main__":
    import os
    print(elpd(Path(os.environ["OUTPUT_DATA_DIR"]) / "analysis" / os.environ["VERSION"]))
