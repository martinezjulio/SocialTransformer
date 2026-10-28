# %% fig2
"""Figure 2, primary results. (a) The session in order: payoff, reward and entropy of each play game as the
descriptive model's cell estimates (95% CI), one line per session type, with the demonstrators' own six shots
scored the same way at the watch slots; solo's exploratory slots 1 and 3 as plain means. (b) Forest plot of
the confirmatory contrasts (H1, H2, H3, H5) × outcomes from the descriptive stage, reward beside them as a reported
outcome without a rule. (c) The partner choice (H4).
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.figures.style import COLOR, ORDER, WIDTH, apply, save
from analysis.games import entropy_bits

OUTCOMES = {"payoff": "chosen arm's payoff", "reward": "reward", "entropy": "entropy (bits)"}
PLAY_SLOT = {("first", 1): 2, ("second", 1): 4, ("first", 2): 6, ("second", 2): 8}   # play games sit at even slots
SLOT_LABEL = {1: "W", 2: "P", 3: "W", 4: "P", 5: "W", 6: "P", 7: "W", 8: "P"}
DODGE = {"expert-first": -0.18, "solo": 0.0, "novice-first": 0.18}   # play-game points staggered in a fixed order; watch slots unmoved


def watch_games(preprocessed: Path) -> pd.DataFrame:
    """Demonstrators' watch games scored like play games: one row per (condition, slot) with mean payoff, reward,
    entropy over sessions (per-session means first) and who demonstrated."""
    df = pd.read_csv(preprocessed, dtype={"timestamp": str}, low_memory=False)
    w = df[df["shot"].astype(str).str.upper().isin(["TRUE", "T", "1"]) & (df["game"] == "watch")]
    rows = []
    for (sid, seq, gi), g in w.groupby(["session_id", "game_sequence", "game_index"]):
        theta = np.array([float(x) for x in g["zone_bandit"].iloc[0].strip("[]").split(",")])
        rows.append(dict(condition=seq.split("-play")[0] + "-first", slot=int(gi) % 4 + 1 + 4 * int(g["round_index"].iloc[0]), demonstrator=g["demonstrator_type"].iloc[0],
                         payoff=theta[g["position"].to_numpy()].mean(), reward=g["reward_generated"].astype(str).str.upper().isin(["TRUE", "T", "1"]).mean(),
                         entropy=entropy_bits(g["position"].to_numpy())))
    return pd.DataFrame(rows).groupby(["condition", "slot"]).agg(payoff=("payoff", "mean"), reward=("reward", "mean"), entropy=("entropy", "mean"), demonstrator=("demonstrator", "first")).reset_index()


def panel_session(axes, analysis_dir: Path, preprocessed: Path) -> None:
    """(a) on three axes, one per outcome; the legend goes under the last axis."""
    from matplotlib.lines import Line2D
    cells = pd.read_csv(Path(analysis_dir) / "descriptive" / "cell_means.csv")
    cells["slot"] = [PLAY_SLOT[(p, r)] for p, r in zip(cells["play"], cells["round"])]
    solo = pd.read_csv(Path(analysis_dir) / "games" / "play_games.csv").query("exploratory").groupby(["participant", "round", "slot"])[list(OUTCOMES)].mean().groupby(["round", "slot"]).mean()
    solo.index = [r * 4 + s - 4 for r, s in solo.index]                         # slots 1, 3, 5, 7
    watch = watch_games(preprocessed)
    for ax, (outcome, label) in zip(axes, OUTCOMES.items()):
        for cond in ORDER:
            c = cells[(cells["outcome"] == outcome) & (cells["condition"] == cond)].sort_values("slot")
            x = c["slot"] + DODGE[cond]
            if cond == "solo":   # one line through all eight play games: cell estimates at even slots, plain means (open) at odd
                y = pd.concat([c.assign(x=x).set_index("x")["emmean"], solo[outcome].rename(index=lambda i: i + DODGE[cond])]).sort_index()
                ax.plot(y.index, y.to_numpy(), color=COLOR[cond], zorder=2)
                ax.plot(solo.index + DODGE[cond], solo[outcome], linestyle="none", marker="o", mfc="white", color=COLOR[cond], zorder=3)
            else:
                ax.plot(x, c["emmean"], color=COLOR[cond], zorder=2)
                w = watch[watch["condition"] == cond]
                ax.scatter(w["slot"], w[outcome], marker="D", s=22, facecolor="white", edgecolor=[COLOR[d] for d in w["demonstrator"]], linewidth=1.2, zorder=3)
            ax.errorbar(x, c["emmean"], yerr=[c["emmean"] - c["lower.CL"], c["upper.CL"] - c["emmean"]], color=COLOR[cond], marker="o", linestyle="none", capsize=2, zorder=4)
        ax.set_xticks(range(1, 9), [SLOT_LABEL[s] for s in range(1, 9)])
        ax.set_ylabel(label)
        ax.axvline(4.5, color="#555555", linewidth=0.8, linestyle="--")
    top = axes[0].secondary_xaxis("top")   # solo sessions are eight play games: their own slot labels
    top.set_xticks(range(1, 9), ["P"] * 8)
    top.set_xlabel("solo session", fontsize=matplotlib.rcParams["legend.fontsize"])
    for x, r in ((2.5, "round 1"), (6.5, "round 2")):   # the hierarchical level under the slot labels
        axes[-1].annotate(r, (x, 0), xycoords=("data", "axes fraction"), xytext=(0, -22), textcoords="offset points", ha="center", va="top")
    axes[-1].set_xlabel("W: watch, P: play", labelpad=22)
    handles = [Line2D([], [], color=COLOR[c], marker="o", label=c) for c in ORDER]
    handles += [Line2D([], [], linestyle="none", marker="D", mfc="white", mec=COLOR[d], label=f"{d}'s demonstration") for d in ("expert", "novice")]
    handles += [Line2D([], [], linestyle="none", marker="o", mfc="white", mec=COLOR["solo"], label="solo, exploratory slots")]
    axes[-1].legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.5), ncol=2)


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
    ax_pr.legend(loc="lower center", bbox_to_anchor=(0.62, 1.0), ncol=2, handletextpad=0.3, columnspacing=0.8)
    ax_ent.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), handletextpad=0.3)   # same placement as the left legend
    ax_pr.set_xlabel("Δ payoff, Δ reward")
    ax_ent.set_xlabel("Δ entropy (bits)")
    ax_ent.set_ylim(ax_pr.get_ylim())


def panel_partner(ax, analysis_dir: Path, conditions_in_legend: bool = False) -> None:
    """(c) H4: the expert-choice rate with 95% CI overall (circle) and by condition (the session-type hues), and the mixed
    model's probability (diamond); rows labelled by n only, the identities carried by the legend."""
    from matplotlib.lines import Line2D
    rates = pd.read_csv(Path(analysis_dir) / "partner_choice" / "rates.csv")
    model = pd.read_csv(Path(analysis_dir) / "partner_choice" / "model.csv").query("role == 'confirmatory'").iloc[0]
    rows = list(rates.itertuples())
    labels = [f"n = {r.n}" for r in rows] + [f"n = {rows[0].n}"]
    est = [r.p for r in rows] + [model["prob_estimate"]]
    lo, hi = [r.lo for r in rows] + [model["prob_lower"]], [r.hi for r in rows] + [model["prob_upper"]]
    colors = [COLOR.get(r.group, COLOR["human"]) for r in rows] + [COLOR["human"]]
    markers = ["o"] * len(rows) + ["D"]
    yy = np.arange(len(labels))[::-1]
    for e, l, h, y0, col, m in zip(est, lo, hi, yy, colors, markers):
        ax.errorbar(e, y0, xerr=[[e - l], [h - e]], marker=m, color=col, capsize=2, zorder=2)
    ax.axvline(0.5, color="#BBBBBB", linewidth=0.8, zorder=1)
    ax.set_yticks(yy, labels)
    handles = [Line2D([], [], linestyle="none", marker="o", color=COLOR["human"], label="all sessions"), Line2D([], [], linestyle="none", marker="D", color=COLOR["human"], label="mixed model")]
    if conditions_in_legend:
        handles[1:1] = [Line2D([], [], linestyle="none", marker="o", color=COLOR[c], label=c) for c in ORDER[:2]]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=2, handletextpad=0.3, columnspacing=0.8)
    ax.set_xlim(0.3, 1.0)
    ax.set_xlabel("P(chose the expert)")


def fig2(analysis_dir: Path, preprocessed: Path) -> list:
    """Assemble Figure 2 (a | b, c) and save ``fig2_primary``."""
    apply()
    fig = plt.figure(figsize=(WIDTH["double"], 6.4))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.4)
    gs_a, gs_r = outer[0].subgridspec(3, 1, hspace=0.3), outer[1].subgridspec(2, 1, height_ratios=[2, 1], hspace=0.55)
    axes_a = [fig.add_subplot(gs_a[i]) for i in range(3)]
    for ax in axes_a[:-1]:
        ax.tick_params(labelbottom=False)
    gs_b = gs_r[0].subgridspec(1, 2, width_ratios=[1.4, 1], wspace=0.12)
    ax_pr, ax_ent, ax_c = fig.add_subplot(gs_b[0]), fig.add_subplot(gs_b[1]), fig.add_subplot(gs_r[1])
    ax_ent.tick_params(labelleft=False)
    panel_session(axes_a, analysis_dir, preprocessed)
    panel_forest(ax_pr, ax_ent, analysis_dir)
    panel_partner(ax_c, analysis_dir)
    for ax, letter in ((axes_a[0], "a"), (ax_pr, "b"), (ax_c, "c")):
        ax.set_title(letter, loc="left", fontweight="bold")
    paths = save(fig, "fig2_primary", analysis_dir)
    fa, axes = plt.subplots(3, 1, figsize=(WIDTH["single"], 6.6), sharex=True)   # the panels on their own as well
    panel_session(axes, analysis_dir, preprocessed)
    fb, (bx1, bx2) = plt.subplots(1, 2, figsize=(WIDTH["single"], 2.8), width_ratios=[1.4, 1], gridspec_kw=dict(wspace=0.12))
    bx2.tick_params(labelleft=False)
    panel_forest(bx1, bx2, analysis_dir)
    fc, cx = plt.subplots(figsize=(WIDTH["single"], 1.8))
    panel_partner(cx, analysis_dir, conditions_in_legend=True)
    return paths + save(fa, "fig2a_session", analysis_dir) + save(fb, "fig2b_forest", analysis_dir) + save(fc, "fig2c_partner", analysis_dir)


if __name__ == "__main__":
    import os
    apply()
    version = os.environ["VERSION"]
    print(fig2(Path(os.environ["OUTPUT_DATA_DIR"]) / "analysis" / version, Path(os.environ["EXPERIMENT_DATA_DIR"]) / version / "preprocessed_participant_data.csv"))
