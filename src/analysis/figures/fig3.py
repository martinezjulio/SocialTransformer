# %% fig3
"""Figure 3, policy models. Names: R repeat, O own experience, S social belief; ROS is the full model.
(a) Whose belief, and a readout, not counts: every model is ROS with a different belief; the recent (S1a) and novice
(S1b) readouts and the visible-pulls models of the last demonstration (S2a), of the expert (S2b) and of every pull seen
(S2c), each minus ROS, all trials. (b) Ablations: each rung minus ROS (S3b is RO), with RS minus RO (S3a) as a bracket.
Marks: the bars of ``analysis.figures.elpd`` (model minus ROS, ± 2 paired SE; the registered ΔELPD is the same number
with the sign reversed). Saves ``fig3_policy``; the panels on their own are ``elpd_S1``, ``elpd_S2``, ``elpd_S3``.
Also here, for Figure 4 and on its own as ``omega_by_round``: the weight on the expert's belief after the expert (ω_E)
and after the novice (ω_N) in round 1 and 2 under Switch-by-round, group level, median and 95% ETI (S4 is the rise in ω_N).
Inputs: ``choice/{ladder,verdicts}.csv`` and the cached Switch-by-round fit.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.figures.elpd import panel_ablation, panel_belief
from analysis.figures.fits import load_fit
from analysis.figures.style import COLOR, WIDTH, apply, save


def panel_omega(ax, analysis_dir: Path) -> None:
    """Group-level ω after the expert and after the novice by round: logit ω_s(r) = μ_s + δ_s · (±½); median, 95% ETI."""
    fit = load_fit(analysis_dir, "Switch-by-round")
    for s, who, dx in (("E", "expert", -0.06), ("N", "novice", 0.06)):
        mu, d = fit.posterior[f"mu_logit_omega_{s}"].reshape(-1), fit.posterior[f"delta_{s}"].reshape(-1)
        om = [1 / (1 + np.exp(-(mu + d * pm))) for pm in (-0.5, 0.5)]
        med, lo, hi = (np.array([np.percentile(o, q) for o in om]) for q in (50, 2.5, 97.5))
        ax.errorbar(np.array([1, 2]) + dx, med, yerr=[med - lo, hi - med], marker="o", color=COLOR[who], capsize=2, label=f"after the {who}")
    ax.set(xticks=[1, 2], xticklabels=["round 1", "round 2"], xlim=(0.6, 2.4), ylim=(0, 1.06), yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], ylabel="weight on the expert's belief")
    ax.axhline(0.5, color="#BBBBBB", linewidth=0.8, zorder=0)
    ax.legend(loc="lower right", handletextpad=0.3, handlelength=1.2, borderaxespad=0.2)


def fig3(analysis_dir: Path) -> list:
    """Assemble Figure 3 (a | b) and save ``fig3_policy``, and the weights on their own as ``omega_by_round``."""
    apply()
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH["double"], 2.9), gridspec_kw=dict(wspace=0.34))
    for ax, (letter, panel) in zip(axes, {"a": panel_belief, "b": panel_ablation}.items()):
        panel(ax, analysis_dir)
        ax.set_title(letter, loc="left", fontweight="bold")
    paths = save(fig, "fig3_policy", analysis_dir)
    f, ax = plt.subplots(figsize=(WIDTH["single"], 2.4))
    panel_omega(ax, analysis_dir)
    return paths + save(f, "omega_by_round", analysis_dir)


if __name__ == "__main__":
    import os
    print(fig3(Path(os.environ["OUTPUT_DATA_DIR"]) / "analysis" / os.environ["VERSION"]))
