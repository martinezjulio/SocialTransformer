# %% fig1
"""Figure 1, the task and the account. (a) A watch game: the top of the game canvas from a screenshot (twelve arms,
the demonstrator's shots marked above the arms it chose, the reward tank). (b) The three conditions as strips of games:
each round is four games on a new bandit; the social conditions end with a choice game. (c) The account, left to
right: the observations (the games so far, in order) enter the transformer, whose latents are each demonstrator's
belief about every arm as a Beta distribution (a real readout is drawn); ROS, drawn as a generative model in plate
notation, turns the expert's latents (S), the participant's own outcomes (O) and the last arm and its reward (R, the
win-stay lose-shift tilt) into the observed choice through a belief per arm, Thompson sampling and a softmax with a
lapse (preregistration, @eq-belief to @eq-choice), scored by leave-one-out ELPD. Each stage is named underneath. The history is written h_{t-1}, everything before
shot t; the readout itself is taken once, before each play game, and the shots within the game enter through O and R.
Symbols are set by Matplotlib's mathtext in Computer Modern (``style.rc``), so no LaTeX install is needed.

The screenshots are not in the repo: ``<analysis_dir>/paradigm_screenshots/paradigm_screenshot_{watch,play}.png``.
Where they are absent ``fig1_paradigm`` and ``paradigm_screens`` are skipped with a message.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

from analysis.data import load_sessions, trial_data
from analysis.figures.fig1_inset import panel_readout, pick
from analysis.figures.style import COLOR, INK, MUTED, WIDTH, apply, panel_letters, save

CROP = (0.029, 0.973, 0.030, 0.705)   # left, right, top, bottom as fractions of the watch screenshot: the whole game canvas inside its border
ROUND = {"expert-first": "EPNP", "novice-first": "NPEP", "solo": "PPPP"}   # the four games of a round (preregistration, Conditions)
CHOICE = {"expert-first": True, "novice-first": True, "solo": False}
FILL = {"E": COLOR["expert"], "N": COLOR["novice"], "P": "white", "C": INK}
TINT_S = "#CFE3F1"   # the social node of the ROS graph: observed, in the expert's hue
PLAY_EDGE = {"expert-first": INK, "novice-first": INK, "solo": COLOR["solo"]}


def screenshot(analysis_dir: Path, game: str) -> Path:
    return Path(analysis_dir) / "paradigm_screenshots" / f"paradigm_screenshot_{game}.png"


def panel_screenshot(ax, path: Path, crop: tuple = CROP) -> None:
    """(a) The screenshot cut to ``crop`` (fractions of its width and height)."""
    im = plt.imread(path)
    h, w = im.shape[:2]
    l, r, t, b = crop
    ax.imshow(im[int(t * h):int(b * h), int(l * w):int(r * w)], interpolation="lanczos")
    ax.set(xticks=[], yticks=[])
    for s in ax.spines.values():
        s.set_visible(False)


def panel_conditions(ax) -> None:
    """(b) One row per condition; a block per game, in session order, the two rounds apart; the choice game last."""
    font = plt.rcParams["font.size"]
    w, gap, round_gap, h = 1.0, 0.14, 0.42, 0.6
    x0 = {r: r * (4 * (w + gap) + round_gap) for r in (0, 1)}
    x_choice = x0[1] + 4 * (w + gap) + round_gap - gap
    for row, (cond, games) in enumerate(ROUND.items()):
        y = -row
        for r in (0, 1):
            for g, kind in enumerate(games):
                x = x0[r] + g * (w + gap)
                play = kind == "P"
                ax.add_patch(Rectangle((x, y - h / 2), w, h, facecolor=FILL[kind], edgecolor=PLAY_EDGE[cond] if play else FILL[kind], linewidth=0.8))
                ax.text(x + w / 2, y, "play" if play else "watch", ha="center", va="center", fontsize=font - 2, color=PLAY_EDGE[cond] if play else "white")
        if CHOICE[cond]:
            ax.add_patch(Rectangle((x_choice, y - h / 2), w * 1.2, h, facecolor=FILL["C"], edgecolor=FILL["C"], linewidth=0.8))
            ax.text(x_choice + w * 0.6, y, "choose", ha="center", va="center", fontsize=font - 2, color="white")
        ax.text(-0.3, y, cond, ha="right", va="center", color=COLOR[cond])
    for r, lab, note in ((0, "round 1", "first bandit"), (1, "round 2", "new bandit, new payoffs")):
        xa, xb = x0[r], x0[r] + 4 * w + 3 * gap
        ax.plot([xa, xb], [0.55, 0.55], color=MUTED, linewidth=0.8)
        ax.text((xa + xb) / 2, 1.0, lab, ha="center", va="bottom", fontweight="bold")
        ax.text((xa + xb) / 2, 0.63, note, ha="center", va="bottom", fontsize=font - 1, color=MUTED)
    ax.text(x_choice + w * 0.6, 0.63, "partner\nchoice", ha="center", va="bottom", fontsize=font - 1, color=MUTED)
    key = [("watch expert", FILL["E"], FILL["E"]), ("watch novice", FILL["N"], FILL["N"]), ("play", "white", INK), ("choose demonstrator", FILL["C"], FILL["C"])]
    for i, (lab, face, edge) in enumerate(key):
        kx, ky = x0[0] + (i % 2) * 4.0, -2.95 - (i // 2) * 0.55
        ax.add_patch(Rectangle((kx, ky - 0.17), 0.42, 0.34, facecolor=face, edgecolor=edge, linewidth=0.8))
        ax.text(kx + 0.58, ky, lab, ha="left", va="center", fontsize=font - 1)
    ax.set(xlim=(-2.55, x_choice + w * 1.2 + 0.05), ylim=(-3.85, 1.55))
    ax.axis("off")


def network(ax, x: float, y: float, w: float, h: float, layers: tuple = (4, 5, 5, 3)) -> None:
    """A small layered network (units as circles, every unit joined to the next layer) in the box (x, y, w, h)."""
    pos = [[(x + w * i / (len(layers) - 1), y + h * (j + 0.5) / n + h * 0 if n > 1 else y + h / 2) for j in range(n)] for i, n in enumerate(layers)]
    pos = [[(px, y + h / 2 + (j - (len(col) - 1) / 2) * h / max(layers)) for j, (px, _) in enumerate(col)] for col in pos]
    for a, b in zip(pos[:-1], pos[1:]):
        for xa, ya in a:
            for xb, yb in b:
                ax.plot([xa, xb], [ya, yb], color=MUTED, linewidth=0.35, alpha=0.7, transform=ax.transAxes, clip_on=False, zorder=1)
    for col in pos:
        ax.plot([q[0] for q in col], [q[1] for q in col], linestyle="none", marker="o", markersize=5.5, markerfacecolor="white", markeredgecolor=INK, markeredgewidth=0.9,
                transform=ax.transAxes, clip_on=False, zorder=2)


def arrow(ax, x0: float, x1: float, y: float, y1: float = None) -> None:
    """An arrow in axes fractions: level at ``y``, or from (x0, y) to (x1, y1)."""
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y if y1 is None else y1), arrowstyle="-|>,head_length=0.5,head_width=0.22", mutation_scale=7, color=INK, linewidth=0.7, shrinkA=0, shrinkB=0, transform=ax.transAxes, clip_on=False))


def panel_model(ax, T) -> None:
    """(c) Observations (the games so far, in order) → the transformer (drawn as a network) → latents (a Beta belief per
    arm, a real readout) → ROS (a generative model of each choice) → leave-one-out ELPD. Each stage is named in a
    caption of at most two lines underneath."""
    font = plt.rcParams["font.size"]
    body, small = font - 1, font - 2
    ax.axis("off")
    y, h = 0.2, 0.78
    mid = y + h / 2

    def caption(x, text, **kw):
        ax.text(x, y - 0.035, text, ha="center", va="top", fontsize=body, linespacing=1.3, transform=ax.transAxes, **kw)

    # the input: the games of the session so far, in the order they happened
    tops = []
    for i, (kind, lab) in enumerate((("E", "watch expert"), ("P", "play"), ("N", "watch novice"))):
        yy = mid + 0.115 - i * 0.2
        tops.append(yy)
        ax.add_patch(Rectangle((0.0, yy), 0.115, 0.15, facecolor=FILL[kind], edgecolor=INK if kind == "P" else FILL[kind], linewidth=0.8, transform=ax.transAxes, clip_on=False))
        ax.text(0.0575, yy + 0.075, lab, ha="center", va="center", fontsize=small, color=INK if kind == "P" else "white", transform=ax.transAxes)
    ax.add_patch(FancyArrowPatch((-0.014, tops[0] + 0.15), (-0.014, tops[-1]), arrowstyle="-|>", mutation_scale=7, color=MUTED, linewidth=0.8, transform=ax.transAxes, clip_on=False))
    ax.text(-0.021, (tops[0] + 0.15 + tops[-1]) / 2, "history of observations", rotation=90, ha="right", va="center", fontsize=small, color=MUTED, transform=ax.transAxes)
    caption(0.04, r"$h_{t-1} = (a_{1:t-1},\, r_{1:t-1})$")   # centered under the blocks and their label
    arrow(ax, 0.121, 0.15, mid)
    network(ax, 0.16, y + 0.12, 0.1, h - 0.24)
    caption(0.229, r"$f_\phi : h_{t-1} \mapsto (\alpha,\, \beta)^{\mathrm{demo}}$")
    arrow(ax, 0.276, 0.303, mid)
    # the latents: a real readout, drawn by the inset's own panel
    ins = ax.inset_axes([0.312, y + 0.2, 0.16, h - 0.32])
    panel_readout(ins, T, pick(T))
    ins.get_legend().remove()
    for t in list(ins.texts):   # the arm names of the standalone inset
        t.remove()
    ins.spines["left"].set_visible(False)
    ins.set_xlabel("payoff", fontsize=small, labelpad=1)
    ins.set_ylabel("")
    ins.tick_params(labelsize=small, pad=1)
    ins.text(0.72, 0.8, "expert's\nbelief", color=COLOR["expert"], fontsize=small, ha="right", va="center", transform=ins.transAxes)
    ins.text(0.1, 0.36, "novice's\nbelief", color=COLOR["novice"], fontsize=small, ha="left", va="center", transform=ins.transAxes)
    caption(0.404, r"Beta$(\alpha_k^{\mathrm{demo}},\, \beta_k^{\mathrm{demo}})$" + "\nper arm $k$")
    box = (0.518, y + 0.01, 0.385, h - 0.02)
    node = panel_ros(ax, *box)
    s_node, a_y = node["S"], node["a"][1]
    ax.plot([0.478, 0.498, 0.498], [mid, mid, s_node[1]], color=INK, linewidth=0.7, solid_joinstyle="miter", solid_capstyle="butt", transform=ax.transAxes, clip_on=False)
    arrow(ax, 0.498, s_node[0] - 0.021, s_node[1])      # the latents leave level with the other arrows, then turn down and enter at S
    caption(box[0] + box[2] / 2 - 0.02, "ROS: repeat (R), own (O), social (S)\n" + r"hierarchical: $\kappa_i$, $\rho_i$; group-level: $\lambda$")
    arrow(ax, 0.908, 0.935, a_y)   # level with the choice node
    ax.text(0.94, a_y, r"$\pi(a_t)$", ha="left", va="center", fontsize=font, transform=ax.transAxes)   # the policy: a distribution over the arms
    ax.plot([0.925, 1.0], [a_y - 0.15, a_y - 0.15], color=MUTED, linewidth=0.7, linestyle=(0, (3, 2)), transform=ax.transAxes, clip_on=False)   # above: what the model produces; below: how it is scored
    ax.text(0.962, a_y - 0.25, r"$\sum_t \log \pi(a_t)$", ha="center", va="top", fontsize=body, transform=ax.transAxes)
    caption(0.962, "ELPD\n(LOO)")


def panel_ros(ax, x: float, y: float, w: float, h: float) -> dict:
    """ROS as a generative model of one shot, in plate notation (preregistration, @eq-belief to @eq-choice). Observed
    (shaded): R, the last arm and its reward; O, the participant's own wins and losses on each arm; S, the expert's
    latents; and the choice a_t. Per arm, the belief θ_k is Beta with S scaled by κ_i and O by λ_w (this game) and λ_c
    (the round's earlier game); Thompson sampling turns the beliefs into p; the win-stay lose-shift tilt
    ρ_0,i + ρ_win,i · r_{t-1} is added on the last arm, and the choice is a softmax with a lapse ε. The inner plate is
    the arm, the outer one every shot of every participant. Returns the nodes' positions."""
    font = plt.rcParams["font.size"]
    body, small = font - 1, font - 2
    T = ax.transAxes
    at = lambda u, v: (x + u * w, y + v * h)   # box coordinates to axes fractions
    ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor=MUTED, linewidth=0.7, transform=T, clip_on=False))
    ax.text(*at(0.985, 0.97), r"shot $t$, participant $i$", ha="right", va="top", fontsize=small, color=MUTED, transform=T)
    ax.add_patch(Rectangle(at(0.012, 0.02), 0.5 * w, 0.685 * h, facecolor="none", edgecolor=MUTED, linewidth=0.7, linestyle=(0, (3, 2)), transform=T, clip_on=False))
    ax.text(*at(0.5, 0.69), r"arm $k$", ha="right", va="top", fontsize=small, color=MUTED, transform=T)
    node = {"R": at(0.095, 0.88), "O": at(0.095, 0.56), "S": at(0.095, 0.23), "b": at(0.39, 0.37), "p": at(0.64, 0.37), "a": at(0.885, 0.6)}
    inside = {"R": "R", "O": "O", "S": "S", "b": r"$\theta_k$", "p": r"$p$", "a": r"$a_t$"}
    face = {"S": TINT_S, "O": "#D9D9DE", "R": "#D9D9DE", "b": "white", "p": "white", "a": "#D9D9DE"}
    for u, v in (("S", "b"), ("O", "b"), ("b", "p"), ("p", "a"), ("R", "a")):
        ax.add_patch(FancyArrowPatch(node[u], node[v], arrowstyle="-|>", mutation_scale=7, color=INK, linewidth=0.8, shrinkA=9, shrinkB=9, transform=T, clip_on=False, zorder=1))
    for k, (nx, ny) in node.items():
        ax.plot([nx], [ny], marker="o", markersize=15, markerfacecolor=face[k], markeredgecolor=COLOR["expert"] if k == "S" else INK, markeredgewidth=0.9, linestyle="none",
                transform=T, clip_on=False, zorder=2)
        ax.text(nx, ny, inside[k], ha="center", va="center", fontsize=body, color=COLOR["expert"] if k == "S" else INK, transform=T, zorder=3)
    under = {"R": r"$a_{t-1},\, r_{t-1}$", "O": r"$s_k,\, f_k$", "S": r"$\alpha_k^{\mathrm{demo}},\, \beta_k^{\mathrm{demo}}$"}   # what each input holds
    for k, text in under.items():
        ax.text(x + 0.03 * w, node[k][1] - 0.085 * h, text, ha="left", va="top", fontsize=small, color=COLOR["expert"] if k == "S" else INK, transform=T)
    ax.text(*at(0.215, 0.355), r"$\kappa_i$", ha="center", va="center", fontsize=body, color=COLOR["expert"], transform=T)
    ax.text(*at(0.27, 0.56), r"$\lambda_{\mathrm{w}},\, \lambda_{\mathrm{c}}$", ha="center", va="center", fontsize=body, transform=T)
    ax.text(*at(0.39, 0.25), "belief\n" + r"Beta$(\alpha_k, \beta_k)$", ha="center", va="top", fontsize=small, color=MUTED, linespacing=1.25, transform=T)
    ax.text(*at(0.64, 0.25), "Thompson\nsampling", ha="center", va="top", fontsize=small, color=MUTED, linespacing=1.25, transform=T)
    ax.text(*at(0.885, 0.72), "choice", ha="center", va="bottom", fontsize=small, color=MUTED, transform=T)
    ax.text(*at(0.9, 0.47), "softmax,\n" + r"lapse $\varepsilon$", ha="center", va="top", fontsize=small, color=MUTED, linespacing=1.25, transform=T)
    ax.text(*at(0.56, 0.795), r"$\rho_{0,i} + \rho_{\mathrm{win},i}\, r_{t-1}$", ha="center", va="bottom", fontsize=body, transform=T)
    ax.text(*at(0.17, 0.97), "win-stay lose-shift", ha="left", va="top", fontsize=small, color=MUTED, transform=T)
    return node


def fig1(analysis_dir: Path, eval_dir: Path, version: str) -> list:
    """Assemble Figure 1 (a | b over c) as ``fig1_paradigm``, and the two full screenshots as ``paradigm_screens``."""
    apply()
    watch, play = screenshot(analysis_dir, "watch"), screenshot(analysis_dir, "play")
    if not watch.exists():
        print(f"fig1_paradigm skipped: {watch} not present")
        return []
    T = trial_data(load_sessions(eval_dir, version))
    fig = plt.figure(figsize=(WIDTH["double"], 3.95))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.3], width_ratios=[1, 1.7], hspace=0.08, wspace=0.05)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]
    panel_screenshot(axes[0], watch)
    axes[0].set_anchor("C")   # centered in its cell, level with the middle of panel b
    panel_conditions(axes[1])
    panel_model(axes[2], T)
    panel_letters(fig, [("a", axes[0], 0, 0), ("b", axes[1], 0, 1), ("c", axes[2], 1, 0)])
    paths = save(fig, "fig1_paradigm", analysis_dir)
    shots = [q for q in (watch, play) if q.exists()]
    fig, axs = plt.subplots(1, len(shots), figsize=(WIDTH["double"], 3.4), squeeze=False)
    for ax, q, name in zip(axs[0], shots, ("watch game", "play game")):
        panel_screenshot(ax, q, (0, 1, 0, 1))
        ax.set_title(name, loc="left", fontweight="bold")
    fig.tight_layout(pad=0.4, w_pad=1.0)
    return paths + save(fig, "paradigm_screens", analysis_dir)


if __name__ == "__main__":
    import os
    root, version = Path(os.environ["OUTPUT_DATA_DIR"]), os.environ["VERSION"]
    print(fig1(root / "analysis" / version, root / "eval" / version, version))
