# %% fig1
"""Figure 1, the task and the account. (a) A watch game: the top of the game canvas from a screenshot (twelve arms,
the demonstrator's shots marked above the arms it chose, the reward tank). (b) The three conditions as strips of games:
each round is four games on a new bandit; the social conditions end with a choice game. (c) The account, left to
right: the observations (the games so far, in order) enter the transformer, whose latents are each demonstrator's
belief about every arm as a Beta distribution (a real readout is drawn); ROS, drawn as a generative model in plate
notation, turns the expert's latents (S), the participant's own outcomes (O) and the last arm (R) into the observed
choice (preregistration, @eq-belief to @eq-choice), scored by leave-one-out ELPD.
Symbols are set by Matplotlib's mathtext in Computer Modern (``style.rc``), so no LaTeX install is needed.

The screenshots are not in the repo: ``<analysis_dir>/paradigm_screenshots/paradigm_screenshot_{watch,play}.png``.
Where they are absent ``fig1_paradigm`` and ``paradigm_screens`` are skipped with a message.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

from analysis.data import load_sessions, trial_data
from analysis.figures.fig1_inset import panel_readout, pick
from analysis.figures.style import COLOR, INK, MUTED, WIDTH, apply, save

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
    w, gap, round_gap, h = 1.0, 0.14, 0.75, 0.6
    x0 = {r: r * (4 * (w + gap) + round_gap) for r in (0, 1)}
    x_choice = x0[1] + 4 * (w + gap) + round_gap - gap
    for row, (cond, games) in enumerate(ROUND.items()):
        y = -row
        for r in (0, 1):
            for g, kind in enumerate(games):
                x = x0[r] + g * (w + gap)
                play = kind == "P"
                ax.add_patch(Rectangle((x, y - h / 2), w, h, facecolor=FILL[kind], edgecolor=PLAY_EDGE[cond] if play else FILL[kind], linewidth=0.8))
                ax.text(x + w / 2, y, "play" if play else "watch", ha="center", va="center", fontsize=font - 2.5, color=PLAY_EDGE[cond] if play else "white")
        if CHOICE[cond]:
            ax.add_patch(Rectangle((x_choice, y - h / 2), w * 1.2, h, facecolor=FILL["C"], edgecolor=FILL["C"], linewidth=0.8))
            ax.text(x_choice + w * 0.6, y, "choose", ha="center", va="center", fontsize=font - 2.5, color="white")
        ax.text(-0.3, y, cond, ha="right", va="center", color=COLOR[cond])
    for r, lab, note in ((0, "round 1", "first bandit"), (1, "round 2", "new bandit, new payoffs")):
        xa, xb = x0[r], x0[r] + 4 * w + 3 * gap
        ax.plot([xa, xb], [0.55, 0.55], color=MUTED, linewidth=0.8)
        ax.text((xa + xb) / 2, 1.0, lab, ha="center", va="bottom", fontweight="bold")
        ax.text((xa + xb) / 2, 0.63, note, ha="center", va="bottom", fontsize=font - 1.5, color=MUTED)
    ax.text(x_choice + w * 0.6, 0.63, "partner\nchoice", ha="center", va="bottom", fontsize=font - 1.5, color=MUTED)
    key = [("watch expert", FILL["E"], FILL["E"]), ("watch novice", FILL["N"], FILL["N"]), ("play", "white", INK), ("choose demonstrator", FILL["C"], FILL["C"])]
    for i, (lab, face, edge) in enumerate(key):
        kx, ky = x0[0] + (i % 2) * 4.0, -2.95 - (i // 2) * 0.55
        ax.add_patch(Rectangle((kx, ky - 0.17), 0.42, 0.34, facecolor=face, edgecolor=edge, linewidth=0.8))
        ax.text(kx + 0.58, ky, lab, ha="left", va="center", fontsize=font - 1.5)
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


def arrow(ax, x0: float, x1: float, y: float) -> None:
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=9, color=INK, linewidth=1.0, transform=ax.transAxes, clip_on=False))


def panel_model(ax, T) -> None:
    """(c) Observations (the games so far, in order) → the transformer (drawn as a network) → latents (a Beta belief per
    arm, a real readout) → ROS (repeat + own + social) → the choice probability."""
    font = plt.rcParams["font.size"]
    ax.axis("off")
    y, h = 0.12, 0.74
    mid, head = y + h / 2, y + h + 0.04
    # the input: the games of the session so far, in the order they happened
    ax.text(0.06, head, "observations", ha="center", va="bottom", fontweight="bold", transform=ax.transAxes)
    tops = []
    for i, (kind, lab) in enumerate((("E", "watch expert"), ("P", "play"), ("N", "watch novice"))):
        yy = y + h - 0.2 - i * 0.2
        tops.append(yy)
        ax.add_patch(Rectangle((0.0, yy), 0.12, 0.15, facecolor=FILL[kind], edgecolor=INK if kind == "P" else FILL[kind], linewidth=0.8, transform=ax.transAxes, clip_on=False))
        ax.text(0.06, yy + 0.075, lab, ha="center", va="center", fontsize=font - 2, color=INK if kind == "P" else "white", transform=ax.transAxes)
    ax.add_patch(FancyArrowPatch((-0.016, tops[0] + 0.15), (-0.016, tops[-1]), arrowstyle="-|>", mutation_scale=7, color=MUTED, linewidth=0.8, transform=ax.transAxes, clip_on=False))
    ax.text(-0.024, (tops[0] + 0.15 + tops[-1]) / 2, "time", rotation=90, ha="right", va="center", fontsize=font - 2, color=MUTED, transform=ax.transAxes)
    ax.text(0.06, y + 0.1, "obs: actions and rewards", ha="center", va="top", fontsize=font - 2, color=MUTED, transform=ax.transAxes)
    arrow(ax, 0.128, 0.158, mid)
    network(ax, 0.175, y + 0.14, 0.135, h - 0.2)
    ax.text(0.2425, head, "transformer", ha="center", va="bottom", fontweight="bold", transform=ax.transAxes)
    arrow(ax, 0.33, 0.36, mid)
    # the latents: a real readout, drawn by the inset's own panel
    ax.text(0.475, head, "latents", ha="center", va="bottom", fontweight="bold", transform=ax.transAxes)
    ins = ax.inset_axes([0.375, y + 0.22, 0.2, h - 0.3])
    panel_readout(ins, T, pick(T))
    ins.get_legend().remove()
    for t in list(ins.texts):   # the arm names of the standalone inset
        t.remove()
    ins.spines["left"].set_visible(False)
    ins.set_xlabel("payoff", fontsize=font - 2, labelpad=1)
    ins.set_ylabel("")
    ins.tick_params(labelsize=font - 2, pad=1)
    ins.text(0.72, 0.8, "expert's\nbelief", color=COLOR["expert"], fontsize=font - 1.5, ha="right", va="center", transform=ins.transAxes)
    ins.text(0.1, 0.36, "novice's\nbelief", color=COLOR["novice"], fontsize=font - 1.5, ha="left", va="center", transform=ins.transAxes)
    ax.text(0.475, y + 0.02, r"$\mathrm{Beta}(\alpha_k^{\mathrm{demo}},\, \beta_k^{\mathrm{demo}})$ per arm $k$", ha="center", va="top", fontsize=font - 0.5, color=MUTED,
            transform=ax.transAxes)
    up = 0.06                                   # the ROS plate sits a little above the row's baseline, its key underneath
    arrow(ax, 0.585, 0.612, y + up + 0.18)      # the latents enter at S
    panel_ros(ax, 0.618, y + up, 0.262, h, head)
    arrow(ax, 0.888, 0.914, mid + up)
    ax.text(0.92, mid + up, r"$\mathrm{ELPD}$", ha="left", va="center", fontsize=font + 1, transform=ax.transAxes)
    ax.text(0.92, mid + up - 0.09, "leave-one-out", ha="left", va="center", fontsize=font - 2, color=MUTED, transform=ax.transAxes)


def panel_ros(ax, x: float, y: float, w: float, h: float, head: float) -> None:
    """ROS as a generative model of one shot, in plate notation: the observed inputs, top to bottom, R (the last arm and its
    reward), O (own wins and losses) and S (the expert's latents); the belief about each arm, Beta with S and O weighted by κ_i and λ;
    the observed choice a_t, tilted toward the last arm by ρ_i. Shaded nodes are observed; the weights on the edges are the
    fitted parameters. The plate is every shot of every participant."""
    font = plt.rcParams["font.size"]
    T = ax.transAxes
    ax.text(x + w / 2, head, "ROS", ha="center", va="bottom", fontweight="bold", transform=T)
    ax.add_patch(Rectangle((x, y + 0.02), w, h - 0.06, facecolor="none", edgecolor=MUTED, linewidth=0.7, transform=T, clip_on=False))
    ax.text(x + w - 0.006, y + h - 0.055, r"shot $t$, participant $i$", ha="right", va="top", fontsize=font - 2.5, color=MUTED, transform=T)
    node = {"R": (x + 0.035, y + 0.6), "O": (x + 0.035, y + 0.39), "S": (x + 0.035, y + 0.18), "b": (x + 0.13, y + 0.285), "a": (x + 0.225, y + 0.47)}
    label = {"S": r"$\alpha^{\mathrm{demo}}\!,\beta^{\mathrm{demo}}$", "O": r"$s, f$", "R": r"$a_{t-1}, r_{t-1}$", "b": r"$\theta_k$", "a": r"$a_t$"}
    face = {"S": TINT_S, "O": "#D9D9DE", "R": "#D9D9DE", "b": "white", "a": "#D9D9DE"}
    for (u, v), weight in {("S", "b"): r"$\kappa_i$", ("O", "b"): r"$\lambda$", ("b", "a"): None, ("R", "a"): r"$\rho_i$"}.items():
        ax.add_patch(FancyArrowPatch(node[u], node[v], arrowstyle="-|>", mutation_scale=7, color=INK, linewidth=0.8, shrinkA=9, shrinkB=9, transform=T, clip_on=False, zorder=1))
        if weight:
            mx, my = (node[u][0] + node[v][0]) / 2, (node[u][1] + node[v][1]) / 2
            ax.text(mx + 0.006, my + (-0.065 if u == "S" else 0.06), weight, ha="center", va="center", fontsize=font, color=COLOR["expert"] if u == "S" else INK, transform=T)
    for k, (nx, ny) in node.items():
        ax.plot([nx], [ny], marker="o", markersize=15, markerfacecolor=face[k], markeredgecolor=COLOR["expert"] if k == "S" else INK, markeredgewidth=0.9, linestyle="none",
                transform=T, clip_on=False, zorder=2)
        ax.text(nx, ny, {"S": "S", "O": "O", "R": "R"}.get(k, label[k]), ha="center", va="center", fontsize=font - 1, color=COLOR["expert"] if k == "S" else INK, transform=T, zorder=3)
    ax.text(node["b"][0] + 0.024, node["b"][1] - 0.075, "belief per arm", ha="left", va="center", fontsize=font - 2.5, color=MUTED, transform=T)
    ax.text(node["a"][0], node["a"][1] + 0.105, "choice", ha="center", va="center", fontsize=font - 2.5, color=MUTED, transform=T)
    for i, (text, col) in enumerate((("R: arm repetition", INK), ("O: own observations", INK), ("S: expert latents (social)", COLOR["expert"]))):
        ax.text(x, y - 0.02 - i * 0.07, text, ha="left", va="top", fontsize=font - 2, color=col, transform=T)


def fig1(analysis_dir: Path, eval_dir: Path, version: str) -> list:
    """Assemble Figure 1 (a | b over c) as ``fig1_paradigm``, and the two full screenshots as ``paradigm_screens``."""
    apply()
    watch, play = screenshot(analysis_dir, "watch"), screenshot(analysis_dir, "play")
    if not watch.exists():
        print(f"fig1_paradigm skipped: {watch} not present")
        return []
    T = trial_data(load_sessions(eval_dir, version))
    fig = plt.figure(figsize=(WIDTH["double"], 4.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1, 1.45], hspace=0.2, wspace=0.07)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]
    panel_screenshot(axes[0], watch)
    axes[0].set_anchor("C")   # centered in its cell, level with the middle of panel b
    panel_conditions(axes[1])
    panel_model(axes[2], T)
    for ax, letter in zip(axes, "abc"):
        ax.set_title(letter, loc="left", fontweight="bold")
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
