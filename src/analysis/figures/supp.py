# %% supp
"""Supplementary figures, one file each (``figS_<name>``). Part 1, from the choice and unseen stages: the full ladder
(every model against ROS, whole data) with the S1/S2 rivals by round; the per-participant ΔELPD behind every paired SE
of the registered comparisons; the ELPD split and the copying comparison on the after-novice cells. Part 2, from the
descriptive, power, partner-choice, games and preprocessing outputs: the confirmatory contrasts under the robustness
fits; decision rates by N at the pilot and half effect; exclusion counts per rule; the solo session's four slots by
round (slots 1 and 3 are exploratory); the partner choice's intercept and recency term. Part 3, from the cached fits:
the group-level posteriors of ROS and Switch-by-round (κ, λ_w, λ_c, ρ0, ρ_win, ω_E, ω_N, δ_E, δ_N) and the lapse ε of
the two free-lapse fits, as 95% ETI bars with the median.
Part 4 (8.7.4): PPC bands, recovery coverage, split-half. Names: R repeat, O own, S social.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.figures.fig3 import COUNTS, SOURCES, dot_whisker, headed_rows
from analysis.figures.fits import load_fit
from analysis.figures.style import COLOR, FONT, WIDTH, apply, save

NAME = {"ROS+persev": "ROS + perseveration", "ROS-reset": "ROS, no reset", "ROS+eps": "ROS + ε free", "Switch-by-round+eps": "Switch-by-round + ε free",
        "Switch-by-round+ktrend": "Switch-by-round + κ trend"}   # registry name -> figure name where they differ
REGISTERED = {**{m: (m,) for m in SOURCES + COUNTS}, "ROS − RO": ("RO",), "RS − RO": ("RS", "RO")}   # per-participant ΔELPD: ROS minus the model, or the difference of two


def fig_ladder(analysis_dir: Path) -> list:
    """Every model against ROS on all trials (left), and the S1/S2 rivals by round (right)."""
    lad = pd.read_csv(Path(analysis_dir) / "choice" / "ladder.csv", index_col=0).drop(index="ROS")
    byr = pd.read_csv(Path(analysis_dir) / "choice" / "ladder_by_round.csv", index_col=[0, 1])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(WIDTH["double"], 4.0), width_ratios=[1.3, 1], gridspec_kw=dict(wspace=0.9))
    ys = headed_rows(ax1, {"ROS minus the model": [NAME.get(m, m) for m in lad.index]})
    for y, m in zip(ys, lad.index):
        dot_whisker(ax1, y, -lad.loc[m, "delta"], lad.loc[m, "se"])
    rivals = SOURCES + COUNTS
    ys = headed_rows(ax2, {"by round": rivals})
    for y, m in zip(ys, rivals):
        for r, dy, mk in (("round 1", -0.18, "o"), ("round 2", 0.18, "s")):
            dot_whisker(ax2, y + dy, -byr.loc[(m, r), "delta"], byr.loc[(m, r), "se"], mk, r if m == rivals[0] else None)
    for ax in (ax1, ax2):
        ax.axvline(0, color="#BBBBBB", linewidth=0.8, zorder=1)
        ax.set_xlabel("ΔELPD, ROS minus the model")
    ax2.legend(loc="upper right", handletextpad=0.3)
    return save(fig, "figS_ladder", analysis_dir)


def fig_per_participant(analysis_dir: Path) -> list:
    """The per-participant ΔELPD behind each registered comparison's paired SE: dots per participant, the mean as a bar."""
    e = pd.read_csv(Path(analysis_dir) / "choice" / "elpd_per_participant.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(WIDTH["single"], 2.8))
    rng = np.random.default_rng(0)
    ys = headed_rows(ax, {"belief rivals (ROS −)": list(REGISTERED)[:5], "ablations": list(REGISTERED)[5:]})
    for y, (lab, ms) in zip(ys, REGISTERED.items()):
        d = -e[ms[0]] if len(ms) == 1 else e[ms[0]] - e[ms[1]]   # the file holds model − ROS: ROS − model, or (RS − ROS) − (RO − ROS) = RS − RO
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


FITS = {"main": ("main", "o", 0.0), "random_slopes": ("random slopes", "s", -0.22), "session_order": ("session order", "^", 0.22)}


def fig_robustness(analysis_dir: Path) -> list:
    """The confirmatory contrasts (payoff, entropy; reward beside them, reported) under the main fit and the two
    sensitivity fits: estimate and 95% CI. The random-slope fit falls back to the intercept-only model when singular; the
    specification reached is named under each panel."""
    d = Path(analysis_dir) / "descriptive"
    main = pd.read_csv(d / "contrasts.csv").query("role != 'exploratory'").assign(fit="main")
    rob = pd.read_csv(d / "robustness.csv")
    both = pd.concat([main, rob]).query("outcome != 'reward'")
    hyps = list(dict.fromkeys(main["hypothesis"]))
    fig, (ax_p, ax_e) = plt.subplots(1, 2, figsize=(WIDTH["single"], 2.6), width_ratios=[1.2, 1], gridspec_kw=dict(wspace=0.15))
    for ax, outcome in ((ax_p, "payoff"), (ax_e, "entropy")):
        for fit, (lab, mk, dy) in FITS.items():
            r = both.query("outcome == @outcome and fit == @fit")
            yy = [hyps.index(h) + dy for h in r["hypothesis"]]
            ax.errorbar(r["estimate"], yy, xerr=[r["estimate"] - r["lower.CL"], r["upper.CL"] - r["estimate"]], marker=mk, linestyle="none", color=COLOR["human"], capsize=2, markersize=4, label=lab if ax is ax_p else None)
        ax.axvline(0, color="#BBBBBB", linewidth=0.8, zorder=1)
        ax.set(yticks=range(len(hyps)), yticklabels=hyps, xlabel=f"Δ {outcome}" + (" (bits)" if outcome == "entropy" else ""))
        ax.invert_yaxis()
    specs = {o: rob.query("fit == 'random_slopes' and outcome == @o")["spec"].iloc[0].replace("_", " ") for o in ("payoff", "entropy")}
    reached = specs["payoff"] if len(set(specs.values())) == 1 else ", ".join(f"{v} ({o})" for o, v in specs.items())
    fig.text(0.5, -0.04, f"random slopes: {reached} reached", ha="center", va="top", fontsize=6, color="#666666")
    ax_e.tick_params(labelleft=False)
    ax_p.legend(loc="lower center", bbox_to_anchor=(0.95, 1.0), ncol=3, handletextpad=0.3, columnspacing=0.8)
    return save(fig, "figS_robustness", analysis_dir)


def fig_power(analysis_dir: Path) -> list:
    """Decision rate of each confirmatory test by N, at the pilot effect and at half of it; the 0.9 target."""
    r = pd.read_csv(Path(analysis_dir) / "power" / "decision_rates.csv")
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH["double"], 2.6), sharey=True, gridspec_kw=dict(wspace=0.15))
    for ax, effect in zip(axes, ("pilot", "half")):
        for (h, o), g in r[r["effect"] == effect].groupby(["hypothesis", "outcome"]):   # the weakest test in black, the rest grey
            weakest = (h, o) == ("H2", "payoff")
            ax.plot(g["N"], g["rate"], marker="o", markersize=3, linewidth=1.0, color=COLOR["human"] if weakest else COLOR["band"], zorder=3 if weakest else 2,
                    label=f"{h} {o}" if weakest else ("the other tests" if (h, o) == ("H1", "entropy") else None))
        ax.axhline(0.9, color="#BBBBBB", linewidth=0.8, zorder=0)
        ax.set(title=f"{effect} effect", xlabel="N participants", ylim=(0, 1.02))
    axes[0].set_ylabel("rate of 'supported' (500 simulations)")
    axes[1].legend(loc="lower right", ncol=2, handletextpad=0.3, columnspacing=0.8)
    return save(fig, "figS_power", analysis_dir)


def fig_exclusions(experiment_dir: Path, analysis_dir: Path) -> list:
    """Participants and sessions dropped per rule, from the preprocessing ledger."""
    e = pd.read_csv(Path(experiment_dir) / "exclusions.csv")
    e = e[e["rule"] != "kept"]
    fig, ax = plt.subplots(figsize=(WIDTH["single"], 2.4))
    ax.barh(range(len(e)), e["n"], color=[COLOR["human"] if l == "participant" else COLOR["model"] for l in e["level"]], height=0.6)
    ax.set(yticks=range(len(e)), yticklabels=[f"{r.replace('_', ' ')} ({l})" for l, r in zip(e["level"], e["rule"])], xlabel="excluded (participants in black, sessions in grey)")
    ax.invert_yaxis()
    return save(fig, "figS_exclusions", analysis_dir)


def fig_solo(analysis_dir: Path) -> list:
    """The solo session's four play games by round: mean payoff and entropy with 95% CI over participants; slots 2 and 4
    (the ones in the descriptive model) filled, slots 1 and 3 (exploratory) hollow."""
    g = pd.read_csv(Path(analysis_dir) / "games" / "play_games.csv").query("condition == 'solo'")
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH["single"], 2.2), gridspec_kw=dict(wspace=0.5))
    for ax, outcome in zip(axes, ("payoff", "entropy")):
        for rnd, ls in ((1, "-"), (2, "--")):
            m = g[g["round"] == rnd].groupby("slot")[outcome].agg(["mean", "sem"])
            ax.errorbar(m.index, m["mean"], yerr=1.96 * m["sem"], linestyle=ls, color=COLOR["solo"], capsize=2, marker="none", label=f"round {rnd}")
            ax.scatter(m.index, m["mean"], marker="o", s=22, color=COLOR["solo"], facecolor=[COLOR["solo"] if sl in (2, 4) else "white" for sl in m.index], zorder=3)
        ax.set(xticks=[1, 2, 3, 4], xlabel="play game (slot)", ylabel=outcome + (" (bits)" if outcome == "entropy" else ""))
    axes[0].legend(loc="lower right", handletextpad=0.3, handlelength=1.6)
    return save(fig, "figS_solo", analysis_dir)


def fig_recency(analysis_dir: Path) -> list:
    """The partner choice's mixed logistic model: the intercept (H4, log-odds of choosing the expert) and the recency
    term (the condition effect: expert-first minus novice-first), estimate and 95% CI."""
    m = pd.read_csv(Path(analysis_dir) / "partner_choice" / "model.csv")
    fig, ax = plt.subplots(figsize=(WIDTH["single"], 1.4))
    labels = {"(Intercept)": "chose the expert (log-odds)", "condition1": "recency: condition term"}
    for y, (_, r) in enumerate(m.iterrows()):
        ax.errorbar(r["estimate"], y, xerr=[[r["estimate"] - r["lower"]], [r["upper"] - r["estimate"]]], marker="o", color=COLOR["human"], capsize=2, linestyle="none")
    ax.set(yticks=range(len(m)), yticklabels=[labels.get(t, t) for t in m["term"]], xlabel="log-odds, 95% CI", ylim=(len(m) - 0.4, -0.6))
    ax.axvline(0, color="#BBBBBB", linewidth=0.8, zorder=1)
    return save(fig, "figS_recency", analysis_dir)


PARAMS = {"κ (group)": ("mu_log_kappa", np.exp), "λ_w": ("log_lambda_w", np.exp), "λ_c": ("lambda_c", None), "ρ0 (group)": ("mu_rho0", None), "ρ_win (group)": ("mu_rho_win", None),
          "ω_E (group)": ("mu_logit_omega_E", lambda x: 1 / (1 + np.exp(-x))), "ω_N (group)": ("mu_logit_omega_N", lambda x: 1 / (1 + np.exp(-x))), "δ_E": ("delta_E", None), "δ_N": ("delta_N", None),
          "ε (free lapse)": ("eps", None)}
POST_MODELS = ["ROS", "Switch-by-round", "ROS+eps", "Switch-by-round+eps"]


def fig_posteriors(analysis_dir: Path) -> list:
    """Group-level posteriors: median and 95% ETI per parameter (columns) and model (rows, the same in every panel;
    blank where the model has no such parameter)."""
    fits = {m: load_fit(analysis_dir, m) for m in POST_MODELS}
    fig, axes = plt.subplots(2, 5, figsize=(WIDTH["double"], 3.2), gridspec_kw=dict(hspace=0.8, wspace=0.35))
    for ax, (lab, (site, f)) in zip(axes.ravel(), PARAMS.items()):
        for y, (m, fit) in enumerate(fits.items()):
            if site in fit.posterior:
                v = fit.posterior[site].reshape(-1)
                lo, med, hi = np.percentile(f(v) if f else v, [2.5, 50, 97.5])
                ax.errorbar(med, y, xerr=[[med - lo], [hi - med]], marker="o", color=COLOR["human"], capsize=2, linestyle="none")
        ax.set(yticks=range(len(fits)), yticklabels=[NAME.get(m, m) for m in POST_MODELS], ylim=(len(fits) - 0.4, -0.6), title=lab)
        ax.tick_params(labelleft=ax in axes[:, 0], length=0, axis="y")
        if site.startswith("delta"):
            ax.axvline(0, color="#BBBBBB", linewidth=0.8, zorder=0)
    return save(fig, "figS_posteriors", analysis_dir)


TARGETS = {"after_novice_rise": "after-novice\npayoff rise", "entropy_fall": "entropy fall\nround 1 − 2", "zigzag": "novice-first\nzig-zag",
           "stay_win": "stay after\na win", "stay_loss": "stay after\na loss"}
PPC_MODELS = ["ROS", "Switch-by-round"]


def fig_ppc(analysis_dir: Path) -> list:
    """Free-running PPC of the two headline models on the five target statistics: the 95% band over posterior draws
    (whisker), its mean (dot, hollow when the human value falls outside) and the human value (black line). Top row
    on all data; bottom row split-half (fitted on one half of the participants, the band and human value on the other)."""
    r = pd.read_csv(Path(analysis_dir) / "checks" / "ppc.csv").set_index(["model", "target"])
    fig, axes = plt.subplots(2, len(TARGETS), figsize=(WIDTH["double"], 2.4), gridspec_kw=dict(wspace=0.35, hspace=0.55))
    for row, (suffix, label) in enumerate((("", "all data"), (" split_half", "split half"))):
        for ax, (t, name) in zip(axes[row], TARGETS.items()):
            for y, m in enumerate(PPC_MODELS):
                q = r.loc[(m + suffix, t)]
                ax.plot([q["lo"], q["hi"]], [y, y], color=COLOR["model"], linewidth=2.2, solid_capstyle="butt", zorder=2)
                ax.scatter(q["mean"], y, s=22, color=COLOR["human"], facecolor=COLOR["human"] if q["inside"] else "white", zorder=3)
                ax.axvline(q["human"], color=COLOR["human"], linewidth=1.0, zorder=1)
            lo, hi = r.loc[[(m + suffix, t) for m in PPC_MODELS]][["lo", "human"]].min().min(), r.loc[[(m + suffix, t) for m in PPC_MODELS]][["hi", "human"]].max().max()
            ax.set(xlim=(lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo)), ylim=(-0.6, 1.6), yticks=[0, 1], yticklabels=PPC_MODELS if ax is axes[row][0] else [])
            ax.tick_params(axis="x", labelsize=FONT - 2)
            if row == 0:
                ax.set_title(name, fontsize=FONT)
        axes[row][0].set_ylabel(label, fontsize=FONT)
    return save(fig, "figS_ppc", analysis_dir)


def fig_recovery(analysis_dir: Path) -> list:
    """(a) δ_N recovery: posterior mean and 95% ETI against the simulated truth, one dataset per mark (hollow when the
    interval misses the truth). (b) Model recovery: ΔELPD (A − B, 2 SE) of each simulated dataset, rows by comparison and
    generating model; a mark on the generating model's side of zero that clears the margin is a correct recovery."""
    rec, mr = (pd.read_csv(Path(analysis_dir) / "checks" / f"{f}.csv") for f in ("recovery", "model_recovery"))
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH["double"], 2.6), gridspec_kw=dict(width_ratios=[1, 1.4], wspace=0.5))
    for _, q in rec.iterrows():
        a.errorbar(q["truth"], q["mean"], yerr=[[q["mean"] - q["lo"]], [q["hi"] - q["mean"]]], linestyle="none", color=COLOR["human"], capsize=2, zorder=2)
        a.scatter(q["truth"], q["mean"], s=22, color=COLOR["human"], facecolor=COLOR["human"] if q["covered"] else "white", zorder=3)
    lim = (min(rec["truth"].min(), rec["lo"].min()) - 0.3, max(rec["truth"].max(), rec["hi"].max()) + 0.3)
    a.plot(lim, lim, color="#BBBBBB", linewidth=0.8, zorder=1)
    a.set(xlim=lim, ylim=lim, xlabel="δ_N simulated", ylabel="δ_N recovered (95% ETI)", title=f"{int(rec['covered'].sum())} of {len(rec)} covered")
    groups = list(mr.groupby(["comparison", "generating"], sort=False))
    for y, ((c, g), rows) in enumerate(groups):
        for _, q in rows.iterrows():                       # ΔELPD is A − B; flipped for B-generated rows so "right of zero" is always correct
            dot_whisker(b, y, q["delta"] if g == q["a"] else -q["delta"], q["se"])
    b.axvline(0, color="#BBBBBB", linewidth=0.8, zorder=1)
    b.set(yticks=range(len(groups)), yticklabels=[f"{c}: {NAME.get(g, g)}" for (c, g), _ in groups],
          xlabel="ΔELPD towards the generating model (2 SE)", title=f"{int(mr['correct'].sum())} of {len(mr)} recovered")
    b.invert_yaxis()
    return save(fig, "figS_recovery", analysis_dir)


def supp(analysis_dir: Path, experiment_dir: Path) -> list:
    apply()
    return (fig_ladder(analysis_dir) + fig_per_participant(analysis_dir) + fig_split(analysis_dir) + fig_robustness(analysis_dir) + fig_power(analysis_dir)
            + fig_exclusions(experiment_dir, analysis_dir) + fig_solo(analysis_dir) + fig_recency(analysis_dir) + fig_posteriors(analysis_dir)
            + fig_ppc(analysis_dir) + fig_recovery(analysis_dir))


if __name__ == "__main__":
    import os
    v = os.environ["VERSION"]
    print(supp(Path(os.environ["OUTPUT_DATA_DIR"]) / "analysis" / v, Path(os.environ["EXPERIMENT_DATA_DIR"]) / v))
