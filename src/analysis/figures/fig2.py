# %% fig2
"""Figure 2, primary results. (a) The session in order: payoff, reward and entropy of each play game as the
descriptive model's cell estimates (95% CI), one line per session type, with the demonstrators' own six shots
scored the same way at the watch slots; solo's exploratory slots 1 and 3 as plain means (both with the 95% interval of the mean); a dashed line at
random choice and, for payoff and reward, a dotted line at the best arm. (b) Forest plot of
the confirmatory contrasts (H1, H2, H3, H5) × outcomes from the descriptive stage, reward beside them as a reported
outcome without a rule. (c) The partner choice (H4): the mixed model's probabilities as bars (``talk.panel_partner``).
"""

from pathlib import Path

from math import factorial

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from matplotlib.lines import Line2D
from matplotlib.transforms import offset_copy

from analysis.figures.style import COLOR, MARK, MUTED, WIDTH, apply, panel_letters, save
from analysis.games import entropy_bits

OUTCOMES = {"payoff": "payoff", "reward": "observed reward", "entropy": "entropy (bits)"}
PLAY_SLOT = {("first", 1): 2, ("second", 1): 4, ("first", 2): 6, ("second", 2): 8}   # play games sit at even slots
REFERENCE = {"random": dict(linestyle=(0, (5, 3)), linewidth=0.9), "best": dict(linestyle=(0, (1, 2)), linewidth=1.1)}   # reference lines: dashed, dotted
DODGE = {"expert-first": -0.16, "solo": 0.0, "novice-first": 0.16}   # play-game points staggered in a fixed order; watch slots unmoved
WATCHED = {"expert-first": {2: "expert", 4: "novice", 6: "expert", 8: "novice"},   # who was watched just before each play game
           "novice-first": {2: "novice", 4: "expert", 6: "novice", 8: "expert"}}


def mean_ci(v) -> pd.Series:
    """Mean with its 95% t interval over independent units (participants or sessions)."""
    v = np.asarray(v, float)
    half = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return pd.Series({"mean": v.mean(), "lo": v.mean() - half, "hi": v.mean() + half})


def watch_games(preprocessed: Path) -> pd.DataFrame:
    """Demonstrators' watch games scored like play games: one row per (condition, slot) with the mean payoff, reward and
    entropy over sessions (one value per session; each participant has one session per condition), the 95% interval of
    each mean as ``<outcome>_lo`` and ``<outcome>_hi``, and who demonstrated."""
    df = pd.read_csv(preprocessed, dtype={"timestamp": str}, low_memory=False)
    w = df[df["shot"].astype(str).str.upper().isin(["TRUE", "T", "1"]) & (df["game"] == "watch")]
    rows = []
    for (sid, seq, gi), g in w.groupby(["session_id", "game_sequence", "game_index"]):
        theta = np.array([float(x) for x in g["zone_bandit"].iloc[0].strip("[]").split(",")])
        rows.append(dict(condition=seq.split("-play")[0] + "-first", slot=int(gi) % 4 + 1 + 4 * int(g["round_index"].iloc[0]), demonstrator=g["demonstrator_type"].iloc[0],
                         payoff=theta[g["position"].to_numpy()].mean(), reward=g["reward_generated"].astype(str).str.upper().isin(["TRUE", "T", "1"]).mean(),
                         entropy=entropy_bits(g["position"].to_numpy())))
    per = pd.DataFrame(rows).groupby(["condition", "slot"])
    out = per.agg(demonstrator=("demonstrator", "first"))
    for o in OUTCOMES:
        ci = per[o].apply(mean_ci).unstack()
        out[o], out[f"{o}_lo"], out[f"{o}_hi"] = ci["mean"], ci["lo"], ci["hi"]
    return out.reset_index()


def expected_entropy(n_arms: int, shots: int) -> float:
    """Expected ``entropy_bits`` of ``shots`` choices made uniformly at random over ``n_arms`` arms (exact: summed over the
    ways the shots can split across arms)."""
    def splits(n, largest):
        if n == 0:
            yield ()
        for first in range(min(n, largest), 0, -1):
            for rest in splits(n - first, first):
                yield (first,) + rest
    total = 0.0
    for c in splits(shots, shots):
        if len(c) > n_arms:
            continue
        ways = factorial(n_arms) / factorial(n_arms - len(c)) / np.prod([factorial(m) for m in np.unique(c, return_counts=True)[1]])   # which arms
        ways *= factorial(shots) / np.prod([factorial(x) for x in c])                                                              # which shots
        q = np.array(c) / shots
        total += ways / n_arms ** shots * -(q * np.log2(q)).sum()
    return float(total)


def references(preprocessed: Path) -> dict:
    """Reference levels per outcome: ``random`` is choosing uniformly at random (the mean arm payoff; the expected entropy
    of a game's shots), ``best`` always choosing the best arm. Every bandit holds the same payoffs in a different order."""
    df = pd.read_csv(preprocessed, usecols=["game", "shot", "session_id", "game_index", "zone_bandit"], low_memory=False)
    theta = np.array([[float(x) for x in t.strip("[]").split(",")] for t in df["zone_bandit"].dropna().unique()])
    assert len({tuple(sorted(t)) for t in theta}) == 1, "bandits differ in their payoffs"
    shots = df[df["shot"].astype(str).str.upper().isin(["TRUE", "T", "1"]) & (df["game"] == "play")].groupby(["session_id", "game_index"]).size().unique()
    assert len(shots) == 1, shots
    level = dict(random=float(theta[0].mean()), best=float(theta[0].max()))
    return dict(payoff=level, reward=level, entropy=dict(random=expected_entropy(theta.shape[1], int(shots[0]))))


def session_data(analysis_dir: Path, preprocessed: Path) -> dict:
    """What the session plot draws: the descriptive model's cell estimates by slot, the play-game table (solo's exploratory
    slots) and the demonstrators' watch games."""
    cells = pd.read_csv(Path(analysis_dir) / "descriptive" / "cell_means.csv")
    cells["slot"] = [PLAY_SLOT[(p, r)] for p, r in zip(cells["play"], cells["round"])]
    return dict(cells=cells, games=pd.read_csv(Path(analysis_dir) / "games" / "play_games.csv"), watch=watch_games(preprocessed), reference=references(preprocessed))


def solo_open(games: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Solo's exploratory play games (slots 1, 3, 5, 7): plain means over participants with their 95% intervals
    (columns mean, lo, hi); these games are not in the descriptive model."""
    s = games.query("exploratory").groupby(["participant", "round", "slot"])[outcome].mean().groupby(["round", "slot"]).apply(mean_ci).unstack()
    s.index = [r * 4 + sl - 4 for r, sl in s.index]
    return s


def draw_session(ax, d: dict, outcome: str, n_novice: int = 4, n_expert: int = 4, show_solo: bool = True, show_watch: bool = True) -> None:
    """Solo first, then the demonstrators' diamonds, then the first ``n_novice`` play games of novice-first and the first
    ``n_expert`` of expert-first (the slides build the plot up in that order). Lines take the session's colour; each
    play-game point takes the colour of the demonstrator watched just before it."""
    ms, lw = matplotlib.rcParams["lines.markersize"] * MARK, matplotlib.rcParams["lines.linewidth"] * 0.75
    k = matplotlib.rcParams["font.size"] / 15
    cells = d["cells"][d["cells"]["outcome"] == outcome]
    if show_solo:
        c = cells[cells["condition"] == "solo"].sort_values("slot")
        op = solo_open(d["games"], outcome)
        y = pd.concat([c.set_index("slot")["emmean"], op["mean"]]).sort_index()
        ax.plot(y.index, y.to_numpy(), color=COLOR["solo"], linewidth=lw, zorder=2)
        ax.errorbar(op.index, op["mean"], yerr=[op["mean"] - op["lo"], op["hi"] - op["mean"]], linestyle="none", marker="o", mfc="white", color=COLOR["solo"], mew=1.4 * k,
                    markersize=ms, capsize=3 * k, elinewidth=1.4 * k, capthick=1.4 * k, zorder=3)
        ax.errorbar(c["slot"], c["emmean"], yerr=[c["emmean"] - c["lower.CL"], c["upper.CL"] - c["emmean"]], color=COLOR["solo"], marker="o", markersize=ms,
                    linestyle="none", capsize=3 * k, elinewidth=1.4 * k, capthick=1.4 * k, zorder=4)
    if show_watch:
        w = d["watch"]
        for r in w.itertuples():
            v, lo, hi = getattr(r, outcome), getattr(r, f"{outcome}_lo"), getattr(r, f"{outcome}_hi")
            ax.errorbar([r.slot], [v], yerr=[[v - lo], [hi - v]], marker="D", markersize=ms, mfc="white", color=COLOR[r.demonstrator], mew=1.4 * k, linestyle="none",
                        capsize=3 * k, elinewidth=1.4 * k, capthick=1.4 * k, zorder=3)
    for cond, n in (("novice-first", n_novice), ("expert-first", n_expert)):
        c = cells[cells["condition"] == cond].sort_values("slot").head(n).rename(columns={"lower.CL": "lo", "upper.CL": "hi"})
        x = c["slot"] + DODGE[cond]
        ax.plot(x, c["emmean"], color=COLOR[cond], linewidth=lw, zorder=2, alpha=0.55)
        for xi, row in zip(x, c.itertuples()):
            ax.errorbar([xi], [row.emmean], yerr=[[row.emmean - row.lo], [row.hi - row.emmean]], color=COLOR[WATCHED[cond][row.slot]], marker="o", markersize=ms,
                        linestyle="none", capsize=3 * k, elinewidth=1.4 * k, capthick=1.4 * k, zorder=5)


def session_ylim(d: dict, outcome: str) -> tuple:
    """The range of everything the complete plot shows, with a margin, so every build-up frame has the same axis."""
    c = d["cells"][d["cells"]["outcome"] == outcome]
    op = solo_open(d["games"], outcome)
    v = np.concatenate([c["lower.CL"], c["upper.CL"], d["watch"][f"{outcome}_lo"], d["watch"][f"{outcome}_hi"], op["lo"], op["hi"], list(d["reference"][outcome].values())])
    return v.min() - 0.06 * np.ptp(v), v.max() + 0.06 * np.ptp(v)


def session_axes(ax, d: dict, outcome: str, top: bool = True, bottom: bool = True) -> None:
    """Axis furniture: the social sessions' watch/play labels and round brackets below (``bottom``), the solo session's
    eight play games above (``top``), each axis named at its left end."""
    font = matplotlib.rcParams["font.size"]
    ax.set_xlim(0.4, 8.6)
    ax.set_ylim(*session_ylim(d, outcome))
    ax.set_ylabel(OUTCOMES[outcome])
    ax.set_xticks(range(1, 9), ["watch", "play"] * 4)
    ax.axvline(4.5, color="#555555", linewidth=0.8, linestyle="--")
    for kind, level in d["reference"][outcome].items():   # random choice and, for payoff and reward, always the best arm
        ax.axhline(level, color=MUTED, zorder=1, **REFERENCE[kind])
    if bottom:
        off = -(font * 2.0 + 6)
        for x, r in ((2.5, "round 1"), (6.5, "round 2")):
            ax.annotate(r, (x, 0), xycoords=("data", "axes fraction"), xytext=(0, off), textcoords="offset points", ha="center", va="top", fontweight="bold")
        for x0, x1 in ((0.7, 4.3), (4.7, 8.3)):   # a bracket under each round's four games
            ax.add_line(Line2D([x0, x1], [0, 0], transform=offset_copy(ax.get_xaxis_transform(), fig=ax.figure, y=off + 3, units="points"), color=MUTED, linewidth=0.9, clip_on=False))
        ax.annotate("social\nsession", (0, 0), xycoords="axes fraction", xytext=(-4, -font * 0.55), textcoords="offset points", ha="right", va="top", fontsize=font - 1, color=MUTED)
    if top:
        sec = ax.secondary_xaxis("top")
        sec.set_xticks(range(1, 9), ["play"] * 8)
        sec.tick_params(colors=COLOR["solo"])
        sec.spines["top"].set_visible(True)
        sec.spines["top"].set_color(COLOR["solo"])
        ax.annotate("solo\nsession", (0, 1), xycoords="axes fraction", xytext=(-4, font * 0.45), textcoords="offset points", ha="right", va="bottom", fontsize=font - 1, color=COLOR["solo"])


def session_legend(ax, **kw) -> None:
    e, n, so, d = COLOR["expert"], COLOR["novice"], COLOR["solo"], matplotlib.rcParams["lines.markersize"] * MARK
    h = [Line2D([], [], color=e, alpha=0.55, label="expert-first session"), Line2D([], [], color=n, alpha=0.55, label="novice-first session"),
         Line2D([], [], color=so, marker="o", markersize=d, label="solo play"), Line2D([], [], color=so, marker="o", markersize=d, mfc="white", label="solo play (not in the model)"),
         Line2D([], [], linestyle="none", marker="o", markersize=d, color=e, label="after expert"), Line2D([], [], linestyle="none", marker="o", markersize=d, color=n, label="after novice"),
         Line2D([], [], linestyle="none", marker="D", markersize=d, mfc="white", mec=e, label="expert"),
         Line2D([], [], linestyle="none", marker="D", markersize=d, mfc="white", mec=n, label="novice"),
         Line2D([], [], color=MUTED, label="chance", **REFERENCE["random"]), Line2D([], [], color=MUTED, label="best arm", **REFERENCE["best"])]
    ax.legend(handles=h, handlelength=1.8, **kw)


def panel_session(axes, analysis_dir: Path, preprocessed: Path, outcomes: tuple = tuple(OUTCOMES), stacked: bool = True) -> None:
    """(a) One axis per outcome. Stacked: the solo labels above the first axis, the social labels and the legend under the
    last. Side by side (``stacked=False``): every axis carries both sets of labels and the caller places the legend."""
    d = session_data(analysis_dir, preprocessed)
    for ax, outcome in zip(axes, outcomes):
        draw_session(ax, d, outcome)
        session_axes(ax, d, outcome, top=not stacked or ax is axes[0], bottom=not stacked or ax is axes[-1])
    if stacked:
        session_legend(axes[-1], ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.5), columnspacing=1.0, labelspacing=0.35)


def session_figure(d: dict, name: str, analysis_dir: Path, outcome: str = "payoff", n_novice: int = 4, n_expert: int = 4, show_watch: bool = True) -> list:
    """One outcome on its own, wide, with the legend at the right; the layout is fixed so the build-up frames register."""
    apply(15)
    fig, ax = plt.subplots(figsize=(11.83, 5.2))
    session_axes(ax, d, outcome)
    draw_session(ax, d, outcome, n_novice, n_expert, show_watch=show_watch)
    fig.subplots_adjust(left=0.085, right=0.745, top=0.86, bottom=0.2)
    session_legend(ax, ncol=1, loc="center left", bbox_to_anchor=(1.02, 0.5), labelspacing=0.9, fontsize=13)
    return save(fig, name, analysis_dir, tight=False)


def session(analysis_dir: Path, preprocessed: Path) -> list:
    """``session_payoff``, ``session_entropy``, ``session_reward``, and the payoff plot built up in ten frames for the
    slides (``session_payoff_slide_01`` … ``_10``): solo, the demonstrators' games, then one play game at a time."""
    d = session_data(analysis_dir, preprocessed)
    paths = [q for outcome in OUTCOMES for q in session_figure(d, f"session_{outcome}", analysis_dir, outcome)]
    stages = [(0, 0, False), (0, 0, True)] + [(k, 0, True) for k in range(1, 5)] + [(4, k, True) for k in range(1, 5)]
    for i, (nn, ne, sw) in enumerate(stages, 1):
        paths += session_figure(d, f"session_payoff_slide_{i:02d}", analysis_dir, "payoff", nn, ne, sw)
    return paths


def panel_forest(ax_pr, ax_ent, analysis_dir: Path) -> None:
    """(b) The confirmatory contrasts × outcomes: estimate and 95% CI, payoff and reward on one axis (probability units),
    entropy on its own; payoff circle, reward square, entropy star; filled where the Holm-corrected verdict is supported,
    hollow where not; reward rows are reported (interval, no rule; advisor, 28 Oct) and stay hollow."""
    c = pd.read_csv(Path(analysis_dir) / "descriptive" / "contrasts.csv").query("role in ('confirmatory', 'reported')")
    hyps = list(dict.fromkeys(c["hypothesis"]))
    y = {h: len(hyps) - i for i, h in enumerate(hyps)}
    for ax, outcomes, off in ((ax_pr, {"payoff": -0.15, "reward": 0.15}, 0), (ax_ent, {"entropy": 0.0}, 0)):
        for outcome, dy in outcomes.items():
            r = c[c["outcome"] == outcome]
            yy = [y[h] + dy for h in r["hypothesis"]]
            ax.errorbar(r["estimate"], yy, xerr=[r["estimate"] - r["lower.CL"], r["upper.CL"] - r["estimate"]], linestyle="none", color=COLOR["human"], capsize=2, zorder=2)
            ax.scatter(r["estimate"], yy, marker={"payoff": "o", "reward": "s", "entropy": "*"}[outcome], s=28 if outcome != "entropy" else 60, zorder=3, color=COLOR["human"],
                       facecolor=[COLOR["human"] if v == "supported" else "white" for v in r["verdict"]], label=outcome if outcome != "reward" else "reward (reported)")
        ax.axvline(0, color="#BBBBBB", linewidth=0.8, zorder=1)
        ax.set_yticks(list(y.values()), list(y))
    ax_pr.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, handletextpad=0.3, columnspacing=0.8)
    ax_ent.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), handletextpad=0.3)   # same placement as the left legend
    ax_pr.set_xlabel("Δ payoff, Δ reward")
    ax_ent.set_xlabel("Δ entropy (bits)")
    ax_ent.set_ylim(ax_pr.get_ylim())


def fig2(analysis_dir: Path, preprocessed: Path) -> list:
    """Assemble Figure 2 and save ``fig2_primary``: (a) payoff and entropy by game, (b) the two levels behind each
    registered contrast (H1, H2, H5 on payoff above; H1, H2, H3 on entropy below; one y axis per row), (c) the partner choice (H4). Also saves the panels on their own (the session plot
    with all three outcomes, the forest plot of the contrasts) and the single-outcome session plots; the partner choice on
    its own is ``talk.fig_partner``."""
    from analysis.figures.talk import CONTRASTS, contrast_tables, panel_contrast_level, panel_partner   # here: talk imports nothing from this module
    apply()
    fig = plt.figure(figsize=(WIDTH["double"], 6.7))
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 1.7], hspace=0.62)
    gs_a = outer[0].subgridspec(1, 2, wspace=0.26)
    axes_a = [fig.add_subplot(gs_a[i]) for i in range(2)]
    panel_session(axes_a, analysis_dir, preprocessed, outcomes=("payoff", "entropy"), stacked=False)
    session_legend(axes_a[0], ncol=5, loc="upper center", bbox_to_anchor=(1.1, -0.34), columnspacing=1.2, labelspacing=0.35)
    lower = outer[1].subgridspec(1, 2, width_ratios=[3.7, 1], wspace=0.24)
    gs_b = lower[0].subgridspec(2, 3, hspace=0.78, wspace=0.1)
    axes_b = [fig.add_subplot(gs_b[i]) for i in range(6)]
    tables = contrast_tables(analysis_dir)
    panels = sorted([(h, o) for h, outcomes, _ in CONTRASTS for o in outcomes], key=lambda p: p[1] != "payoff")   # payoff on the top row, entropy below
    assert [o for _, o in panels] == ["payoff"] * 3 + ["entropy"] * 3, panels
    for ax, (hyp, outcome) in zip(axes_b, panels):
        if tables is None:
            ax.axis("off")
        else:
            panel_contrast_level(ax, *tables, hyp, outcome, compact=True)
    for row in (axes_b[:3], axes_b[3:]):   # one y axis per row: the same limits, named and numbered on the first panel only
        lims = [ax.get_ylim() for ax in row]
        for ax in row:
            ax.set_ylim(min(lo for lo, _ in lims), max(hi for _, hi in lims))
        for ax in row[1:]:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)
    gs_c = lower[1].subgridspec(3, 1, height_ratios=[0.16, 1, 0.16])
    ax_c = fig.add_subplot(gs_c[1])
    panel_partner(ax_c, analysis_dir, compact=True)
    panel_letters(fig, [("a", axes_a[0], 0, 0), ("b", axes_b[0], 1, 0), ("c", ax_c, 1, 1)])
    paths = save(fig, "fig2_primary", analysis_dir)
    fa, axes = plt.subplots(3, 1, figsize=(WIDTH["single"], 6.6), sharex=True)   # the panels on their own as well
    panel_session(axes, analysis_dir, preprocessed)
    fb, (bx1, bx2) = plt.subplots(1, 2, figsize=(WIDTH["single"], 2.8), width_ratios=[1.4, 1], gridspec_kw=dict(wspace=0.12))
    bx2.tick_params(labelleft=False)
    panel_forest(bx1, bx2, analysis_dir)
    paths += save(fa, "session_all", analysis_dir) + save(fb, "contrasts_forest", analysis_dir)
    return paths + session(analysis_dir, preprocessed)


if __name__ == "__main__":
    import os
    apply()
    version = os.environ["VERSION"]
    print(fig2(Path(os.environ["OUTPUT_DATA_DIR"]) / "analysis" / version, Path(os.environ["EXPERIMENT_DATA_DIR"]) / version / "preprocessed_participant_data.csv"))
