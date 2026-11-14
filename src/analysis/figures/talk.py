# %% talk
"""Figures first drawn for talks, one message each. ``contrast_H1`` … ``_H5``, and all of them in one figure as
``contrast_levels``: the two levels behind each registered contrast, with the registered difference as the title (needs
``descriptive/contrast_levels.csv`` from ``src/r/contrast_levels.R``). They show what each change looks like; the
differences themselves, point and whisker, are ``contrasts_forest``. ``partner_choice_slide``: H4 as bars. ``omega_by_condition``: the weight on the expert's
belief by round, per session type, from the Switch-8 fit (exploratory; its table goes to
``unseen/omega_by_condition.csv``). ``readout_N_scatter``: the experience N the transformer reads out on simulated test
sessions (needs ``eval/data/test_cutoff_model_predictions.csv``). ``exclusions_slide``: exclusions per rule without the
returned submissions. A figure whose input is absent is skipped with a message.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.transforms import offset_copy

from analysis.figures.fits import load_fit
from analysis.figures.style import COLOR, INK, MUTED, apply, save

EXPERT, NOVICE, SOLO = COLOR["expert"], COLOR["novice"], COLOR["solo"]
WHISKER = "#3F3F4A"
SIDES = {   # hypothesis: the level subtracted, then the other
    "H1": [("after the\nnovice", NOVICE), ("after the\nexpert", EXPERT)],
    "H2": [("round 1", NOVICE), ("round 2", NOVICE)],
    "H3": [("round 1", INK), ("round 2", INK)],
    "H5": [("solo,\nfinal game", SOLO), ("after the\nexpert", EXPERT)],
}
CONTRASTS = [("H1", ["payoff", "entropy"], (5.58, 2.75)), ("H2", ["payoff", "entropy"], (5.83, 3.9)), ("H3", ["entropy"], (5.0, 3.9)), ("H5", ["payoff"], (3.2, 2.75))]
LABEL = {"payoff": "chosen arm's payoff", "entropy": "choice entropy (bits)"}
XLABEL = {"H2": "play right after the novice", "H3": "social play (after either demonstrator)"}
XLABEL_WRAPPED = {"H2": "play right after\nthe novice", "H3": "social play (after\neither demonstrator)"}
EXCLUSION_NAME = {"colour_vision": "color vision"}


def minus(text: str) -> str:
    return text.replace("-", "−")


def contrast_tables(analysis_dir: Path):
    """(levels, registered contrasts), or None when the descriptive stage has not written the levels."""
    f = Path(analysis_dir) / "descriptive" / "contrast_levels.csv"
    if not f.exists():
        return None
    return pd.read_csv(f).set_index(["hypothesis", "outcome", "side"]), pd.read_csv(Path(analysis_dir) / "descriptive" / "contrasts.csv")


def panel_contrast_level(ax, lv: pd.DataFrame, reg_all: pd.DataFrame, hyp: str, outcome: str, name: bool = False, compact: bool = False) -> None:
    """The two levels behind one registered contrast: each side's mean with its 95% CI from the registered mixed model (a
    between-participant interval) joined by a line; the title is the registered within-participant difference and its CI
    (under the hypothesis's name when ``name``). Marks scale with the font size; ``compact`` wraps the title and the axis
    label for a narrow panel inside a composite figure."""
    font = plt.rcParams["font.size"]
    k = font / 11
    rows = [lv.loc[(hyp, outcome, s)] for s in ("left", "right")]
    for i, ((_, col), r) in enumerate(zip(SIDES[hyp], rows)):
        ax.errorbar([i], [r["estimate"]], yerr=[[r["estimate"] - r["lower"]], [r["upper"] - r["estimate"]]], color=col, marker="o", markersize=9 * k, capsize=4 * k,
                    elinewidth=1.8 * k, capthick=1.8 * k, linestyle="none", zorder=3)
    ax.plot([0, 1], [rows[0]["estimate"], rows[1]["estimate"]], color=MUTED, linewidth=1.2 * k, zorder=2)
    ax.set_xticks([0, 1], [s[0] for s in SIDES[hyp]])
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylabel(LABEL[outcome])
    if hyp in XLABEL:
        ax.set_xlabel(XLABEL_WRAPPED[hyp] if compact else XLABEL[hyp])
    lo, hi = min(r["lower"] for r in rows), max(r["upper"] for r in rows)
    pad = max(hi - lo, 0.02 if outcome == "payoff" else 0.08)
    ax.set_ylim(lo - 0.3 * pad, hi + 0.45 * pad)
    diff, reg = lv.loc[(hyp, outcome, "diff")], reg_all.query("hypothesis == @hyp and outcome == @outcome").iloc[0]
    assert abs(diff["estimate"] - reg["estimate"]) < 1e-6 and abs(diff["lower"] - reg["lower.CL"]) < 1e-6, (hyp, outcome)
    unit = "" if outcome == "payoff" else " bits"
    est, interval = minus(f"Δ = {reg['estimate']:+.2f}{unit}"), minus(f"[{reg['lower.CL']:.2f}, {reg['upper.CL']:.2f}]")
    if compact:
        ax.set_title(f"{hyp}:  {est}\n{interval}", fontsize=font, pad=6 * k, linespacing=1.25)
        ax.tick_params(axis="x", labelsize=font - 2)
        ax.set_xlim(-0.7, 1.7)
    else:
        ax.set_title(f"{hyp}\n{est}  {interval}" if name else f"{est}  {interval}", fontsize=font, pad=6 * k)


def fig_contrasts(analysis_dir: Path) -> list:
    """``contrast_H*`` (one hypothesis each) and ``contrast_levels`` (all six panels)."""
    tables = contrast_tables(analysis_dir)
    if tables is None:
        print("contrast plots skipped: descriptive/contrast_levels.csv not present (the descriptive stage writes it)")
        return []
    apply(11)
    paths = []
    panels = [(hyp, outcome) for hyp, outcomes, _ in CONTRASTS for outcome in outcomes]
    whole, grid = plt.subplots(2, 3, figsize=(9.6, 6.6))   # every registered contrast in one figure, beside the single ones
    for hyp, outcomes, size in CONTRASTS:
        fig, axes = plt.subplots(1, len(outcomes), figsize=size, squeeze=False)
        for ax, outcome in zip(axes[0], outcomes):
            panel_contrast_level(ax, *tables, hyp, outcome)
            panel_contrast_level(grid.ravel()[panels.index((hyp, outcome))], *tables, hyp, outcome, name=True)
        fig.tight_layout(pad=0.6, w_pad=1.6)
        paths += save(fig, f"contrast_{hyp}", analysis_dir)
    whole.tight_layout(pad=0.6, w_pad=1.8, h_pad=1.6)
    return paths + save(whole, "contrast_levels", analysis_dir)


def fig_partner(analysis_dir: Path) -> list:
    """H4: the share of choice games in which the expert was picked (all sessions, novice-first, expert-first) with the 95%
    intervals of ``partner_choice/rates.csv``, as bars; the mixed model is in ``partner_choice_paper``."""
    apply(12)
    r = pd.read_csv(Path(analysis_dir) / "partner_choice" / "rates.csv").set_index("group").loc[["all", "novice-first", "expert-first"]]
    fig, ax = plt.subplots(figsize=(4.6, 3.9))
    x = np.arange(3)
    ax.bar(x, r["p"], width=0.62, color=[INK, NOVICE, EXPERT], zorder=2)
    ax.errorbar(x, r["p"], yerr=[r["p"] - r["lo"], r["hi"] - r["p"]], color=WHISKER, linestyle="none", capsize=5, elinewidth=1.6, capthick=1.6, zorder=3)
    ax.axhline(0.5, color=MUTED, linewidth=0.9, linestyle="--", zorder=1)
    ax.set_xlim(-0.5, 2.95)
    ax.text(2.93, 0.51, "chance", fontsize=10, color=MUTED, ha="right", va="bottom")
    for xi, row in zip(x, r.itertuples()):
        ax.text(xi, row.hi + 0.025, f"{100 * row.p:.0f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_xticks(x, [f"{n}\n{int(k)} of {int(m)}" for n, k, m in zip(["all sessions", "novice-first", "expert-first"], r["k"], r["n"])])
    ax.set(ylim=(0, 1), yticks=[0, 0.25, 0.5, 0.75, 1.0], ylabel="share of choice games\nin which the expert was picked")
    fig.tight_layout(pad=0.6)
    return save(fig, "partner_choice_slide", analysis_dir)


def fig_omega_by_condition(analysis_dir: Path) -> list:
    """The by-condition version of ``omega_by_round``. Switch-8 has one group-level weight per (demonstrator just watched,
    round, session type); this draws the two rounds for each and the round change computed draw by draw (median, 95% ETI).
    Exploratory, not a registered quantity. Index into ``logit_omega_8``: (after the expert) * 4 + (round 2) * 2 + expert-first."""
    try:
        lo8 = np.asarray(load_fit(analysis_dir, "Switch-8").posterior["logit_omega_8"]).reshape(-1, 8)
    except FileNotFoundError:
        print("omega_by_condition skipped: no Switch-8 fit")
        return []
    om = 1 / (1 + np.exp(-lo8))
    q = lambda v: np.percentile(v, [50, 2.5, 97.5])
    apply(12)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3), sharey=True)
    rows = []
    for ax, (cond, ef) in zip(axes, (("expert-first", 1), ("novice-first", 0))):
        for who, we, col, dx in (("expert", 1, EXPERT, -0.05), ("novice", 0, NOVICE, 0.05)):
            i1, i2 = we * 4 + 0 + ef, we * 4 + 2 + ef
            r1, r2 = q(om[:, i1]), q(om[:, i2])
            dw, dl = q(om[:, i2] - om[:, i1]), q(lo8[:, i2] - lo8[:, i1])
            unseen = cond == "novice-first" and who == "novice"   # that play game comes before the expert is seen on the bandit
            ax.errorbar(np.array([1, 2]) + dx, [r1[0], r2[0]], yerr=[[r1[0] - r1[1], r2[0] - r2[1]], [r1[2] - r1[0], r2[2] - r2[0]]], marker="o", markersize=8, color=col,
                        capsize=4, linewidth=2, elinewidth=1.8, capthick=1.8, markeredgecolor=INK if unseen else col, markeredgewidth=1.6 if unseen else 0)
            ax.text(2.22, 0.63 if who == "expert" else 0.55, minus(f"after the {who}:  Δ = {dw[0]:+.2f} [{dw[1]:+.2f}, {dw[2]:+.2f}]"), va="center", ha="left", color=col, fontsize=10)
            rows.append(dict(session=cond, after=who, round1=r1[0], round1_lo=r1[1], round1_hi=r1[2], round2=r2[0], round2_lo=r2[1], round2_hi=r2[2], change=dw[0],
                             change_lo=dw[1], change_hi=dw[2], change_logit=dl[0], change_logit_lo=dl[1], change_logit_hi=dl[2], p_change_positive=float((om[:, i2] > om[:, i1]).mean())))
        ax.text(2.22, 0.71, "change, round 2 − round 1", va="center", ha="left", color=MUTED, fontsize=9.5)
        ax.axhline(0.5, color="#BBBBBB", linewidth=0.8, zorder=0, xmax=0.56)
        ax.set(xticks=[1, 2], xticklabels=["round 1", "round 2"], xlim=(0.6, 3.75), ylim=(0, 1.06), yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_title(f"{cond} sessions", fontsize=12, loc="left")
    axes[0].set_ylabel("weight on the expert's belief")
    h = [Line2D([], [], color=EXPERT, marker="o", markersize=8, linewidth=2, label="play after the expert"), Line2D([], [], color=NOVICE, marker="o", markersize=8, linewidth=2, label="play after the novice"),
         Line2D([], [], color=NOVICE, marker="o", markersize=8, markeredgecolor=INK, markeredgewidth=1.6, linestyle="none", label="expert not yet seen on this bandit")]
    fig.legend(handles=h, ncol=3, loc="lower center", fontsize=10, handlelength=2.0, columnspacing=1.6)
    fig.tight_layout(pad=0.6, w_pad=1.5, rect=(0, 0.07, 1, 1))
    pd.DataFrame(rows).to_csv(Path(analysis_dir) / "unseen" / "omega_by_condition.csv", index=False)
    return save(fig, "omega_by_condition", analysis_dir)


def fig_readout_n(analysis_dir: Path, n_points: int = 300, seed: int = 0) -> list:
    """The experience N read out for each demonstrator after each game of rounds 1 and 2, on the simulated test split.
    Dots are a random ``n_points`` sessions per watch order (blue: the expert's N, true value 128; vermilion: the novice's,
    true value 0); lines are the mean over all sessions of that watch order (solid: expert watched first)."""
    f = Path(analysis_dir).parents[1] / "eval" / "data" / "test_cutoff_model_predictions.csv"
    if not f.exists():
        print(f"readout_N_scatter skipped: {f} not present (src/eval/eval.py on the simulated test split writes it)")
        return []
    e = pd.read_csv(f)
    e = e[(e["round_index"] <= 1) & (e["segment_type"] != "query")]
    e = e.assign(x=e["round_index"] * 5 + e.groupby(["example_id", "round_index"]).cumcount())
    orders = {"expert-play-novice-play-choice": ("-", -0.17), "novice-play-expert-play-choice": ("--", 0.17)}
    rng = np.random.default_rng(seed)
    apply(12)
    fig, ax = plt.subplots(figsize=(7.3, 5.15))
    for seq, (ls, dx) in orders.items():
        q = e[e["game_sequence"] == seq]
        s = q[q["example_id"].isin(rng.choice(q["example_id"].unique(), size=n_points, replace=False))]
        jit = rng.uniform(-0.1, 0.1, len(s))
        mean = q.groupby("x")[["expert_N_tot_pred", "novice_N_tot_pred"]].mean()
        for col, who in (("expert_N_tot_pred", EXPERT), ("novice_N_tot_pred", NOVICE)):
            ax.scatter(s["x"] + dx + jit, s[col], s=9, color=who, alpha=0.13, linewidths=0, zorder=2, rasterized=True)
            for r0 in (0, 5):
                seg = mean.loc[r0:r0 + 4]
                ax.plot(seg.index + dx, seg[col], color=who, linestyle=ls, marker="o", markersize=6, markeredgecolor="white", markeredgewidth=0.9, linewidth=2.0, zorder=4)
    ax.axhline(128, color=EXPERT, linewidth=1.0, linestyle=":", zorder=1)
    ax.axhline(0, color=NOVICE, linewidth=1.0, linestyle=":", zorder=1)
    ax.text(9.5, 131, "expert's true N = 128", color=EXPERT, fontsize=10, va="bottom", ha="right")
    ax.text(9.5, -4, "novice's true N = 0", color=NOVICE, fontsize=10, va="top", ha="right")
    ax.axvline(4.5, color="#555555", linewidth=0.8, linestyle="--")
    ax.set(ylabel="experience N read out", ylim=(-14, 146), yticks=[0, 25, 50, 75, 100, 125], xlim=(-0.55, 9.55))
    ax.set_xticks(range(10), ["start", "watch", "play", "watch", "play"] * 2)
    off = -(12 * 2.0 + 6)
    for x, r in ((2, "round 1"), (7, "round 2")):
        ax.annotate(r, (x, 0), xycoords=("data", "axes fraction"), xytext=(0, off), textcoords="offset points", ha="center", va="top", fontweight="bold")
    for x0, x1 in ((-0.35, 4.35), (4.65, 9.35)):
        ax.add_line(Line2D([x0, x1], [0, 0], transform=offset_copy(ax.get_xaxis_transform(), fig=fig, y=off + 3, units="points"), color=MUTED, linewidth=0.9, clip_on=False))
    h = [Line2D([], [], color=INK, linestyle="-", linewidth=2, label="mean, expert watched first"), Line2D([], [], color=INK, linestyle="--", linewidth=2, label="mean, novice watched first"),
         Line2D([], [], color=EXPERT, marker="o", markersize=6, linestyle="none", label="expert's N"), Line2D([], [], color=NOVICE, marker="o", markersize=6, linestyle="none", label="novice's N")]
    ax.legend(handles=h, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=10, handlelength=2.4, columnspacing=1.6)
    fig.subplots_adjust(left=0.11, right=0.985, top=0.9, bottom=0.14)
    return save(fig, "readout_N_scatter", analysis_dir)


def fig_exclusions(experiment_dir: Path, analysis_dir: Path) -> list:
    """Participants and sessions excluded per rule, from the preprocessing ledger, with each count printed. Returned
    submissions are left out (those people never entered the study); ``exclusions_paper`` has every row."""
    e = pd.read_csv(Path(experiment_dir) / "exclusions.csv")
    e = e[~e["rule"].isin(["kept", "returned"])].sort_values("level", kind="stable")   # participants first, then sessions
    apply(11)
    fig, ax = plt.subplots(figsize=(5.83, 3.14))
    ax.barh(range(len(e)), e["n"], color=[INK if lvl == "participant" else COLOR["model"] for lvl in e["level"]], height=0.62)
    for i, v in enumerate(e["n"]):
        ax.text(v + 0.15, i, str(v), va="center", fontsize=10, color=MUTED)
    ax.set_yticks(range(len(e)), [f"{EXCLUSION_NAME.get(r, r.replace('_', ' '))} ({lvl})" for lvl, r in zip(e["level"], e["rule"])])
    ax.invert_yaxis()
    ax.set(xlim=(0, e["n"].max() + 1.5), xlabel="excluded (participants in black, sessions in gray)")
    fig.tight_layout(pad=0.5)
    return save(fig, "exclusions_slide", analysis_dir)


def talk(analysis_dir: Path, experiment_dir: Path) -> list:
    return fig_contrasts(analysis_dir) + fig_partner(analysis_dir) + fig_omega_by_condition(analysis_dir) + fig_readout_n(analysis_dir) + fig_exclusions(experiment_dir, analysis_dir)


if __name__ == "__main__":
    import os
    v = os.environ["VERSION"]
    print(talk(Path(os.environ["OUTPUT_DATA_DIR"]) / "analysis" / v, Path(os.environ["EXPERIMENT_DATA_DIR"]) / v))
