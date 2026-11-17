# %% style
"""Figure conventions shared by every figure module: palette, 95% intervals, sizes, and the saver.

Palette (Okabe–Ito hues, validated colour-blind safe for the two coloured series): expert blue, novice vermilion,
solo grey (deliberately neutral: the condition with no social information), humans in near-black ink, models as
lighter tints of the demonstrator hues. A policy model takes the hue of the demonstrations it uses (``model_color``).
Every interval drawn is 95% (ETI for posteriors, CI for frequentist fits) unless the axis says ± 2 SE.

One folder holds every figure, for the paper, the slides and the project webpage: ``<analysis_dir>/figures/<name>``
as ``.png`` (viewing), and as ``.pdf`` (paper, slides) and ``.svg`` (webpage, slides) when asked for: the formats are
the comma-separated ``FIGURE_FORMATS`` (default ``png`` while the figures are being finalised; ``pdf,png,svg`` for all). Names say what the figure shows
(``elpd_S1``, ``session_payoff``); only the paper's composite figures carry a number (``fig2_primary``). Where two
versions of one figure exist, the name ends in ``_paper`` or ``_slide``.

Sizes: a standalone panel is drawn at single-column proportions, 8 pt type on 3.4 in. ``apply(font)`` scales the type
and the marks together, so a figure drawn larger (12 pt on 5.1 in) has the same look once placed.
"""

from pathlib import Path

import logging
import os

import matplotlib
import matplotlib.pyplot as plt

logging.getLogger("fontTools").setLevel(logging.ERROR)   # font-subsetting chatter when a PDF embeds a system font

INK, MUTED, GRID = "#16161A", "#5B5B66", "#E3E3E8"
COLOR = {"expert": "#0072B2", "novice": "#D55E00", "solo": "#7F7F7F", "none": "#7F7F7F", "human": "#1A1A1A",
         "expert-first": "#0072B2", "novice-first": "#D55E00", "model": "#8C8C8C", "band": "#C9C9C9"}
TINT = {"expert": "#9DC6E2", "novice": "#F0B48F"}     # model bands on a demonstrator's hue
# policy models: the hue of whose demonstrations the belief uses; a readout is solid, a count of shots a lighter tint
MODEL_COLOR = {"ROS": COLOR["expert"], "recent": "#E69F00", "novice": COLOR["novice"], "naive": "#B0B0B6", "naive-recent": "#F3CF80", "naive-expert": "#7FB8D8"}
ABLATION_COLOR, EXTENSION_COLOR = "#4A5568", "#009E73"
ORDER = ["expert-first", "novice-first", "solo"]       # session types, fixed order in every legend
INTERVAL = 0.95
LINE, MARKER, FONT = 1.6, 5, 8                         # thin marks, small readable type
MARK = 0.72                                            # markers that carry an interval, relative to the rc size: small enough that a narrow interval shows
WIDTH = {"single": 3.4, "double": 7.0}                 # column widths in inches
FORMATS = tuple(f.strip() for f in os.environ.get("FIGURE_FORMATS", "png").split(","))


def rc(font: float = FONT) -> dict:
    k = font / FONT
    return {"font.size": font, "axes.titlesize": font + 1, "axes.labelsize": font, "legend.fontsize": font - 1, "xtick.labelsize": font - 1,
            "ytick.labelsize": font - 1, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6 * k, "xtick.major.width": 0.6 * k,
            "ytick.major.width": 0.6 * k, "lines.linewidth": LINE * k, "lines.markersize": MARKER * k, "legend.frameon": False, "axes.grid": False,
            "figure.dpi": 110, "savefig.dpi": 300, "pdf.fonttype": 42, "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "cm", "svg.fonttype": "path", "svg.hashsalt": "social-bandits", "axes.edgecolor": INK, "text.color": INK, "axes.labelcolor": INK,
            "xtick.color": INK, "ytick.color": INK}


RC = rc()


def apply(font: float = FONT) -> None:
    """Set the rc conventions for the current process (every figure function calls it first); ``font`` scales type and marks."""
    matplotlib.rcParams.update(rc(font))


def model_color(model: str, extensions: tuple = ()) -> str:
    return MODEL_COLOR.get(model, EXTENSION_COLOR if model in extensions else ABLATION_COLOR)


def panel_letters(fig, letters: list, pad: float = 3) -> None:
    """Panel letters on a grid. ``letters`` is a list of (text, axis, row, column): letters that share a row key sit at one
    height, just above the tallest of that row's panels (axis, title and top labels included); letters that share a
    column key sit at one x, the left edge of the widest of that column's panels (tick and axis labels included).
    Legends are left out of the measure. Call it once the panels are drawn."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()

    def box(ax):
        parts = [ax.get_window_extent(r)] + [a.get_tightbbox(r) for a in (ax.xaxis, ax.yaxis, *ax.child_axes) if ax.axison]
        parts += [t.get_window_extent(r) for t in (ax.title, *ax.texts) if t.get_text() and t.get_visible()]
        return matplotlib.transforms.Bbox.union([b for b in parts if b is not None])

    boxes = [box(ax) for _, ax, _, _ in letters]
    top = {k: max(b.y1 for b, (_, _, row, _) in zip(boxes, letters) if row == k) for k in {row for _, _, row, _ in letters}}
    left = {k: min(b.x0 for b, (_, _, _, col) in zip(boxes, letters) if col == k) for k in {col for _, _, _, col in letters}}
    for text, _, row, col in letters:
        fig.text(*fig.transFigure.inverted().transform((left[col], top[row] + pad * fig.dpi / 72)), text, fontweight="bold",
                 fontsize=matplotlib.rcParams["axes.titlesize"], ha="left", va="bottom")


def ci(lo: float, hi: float) -> str:
    return f"[{lo:.2f}, {hi:.2f}]"


def save(fig, name: str, analysis_dir: Path, tight: bool = True) -> list:
    """Write ``figures/<name>.<format>`` under ``analysis_dir`` for each of ``FORMATS``; returns the paths. ``tight=False``
    keeps the figure's own canvas, for a series whose frames must register."""
    out = Path(analysis_dir) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{name}.{ext}" for ext in FORMATS]
    for p in paths:
        fig.savefig(p, bbox_inches="tight" if tight else None, facecolor="white", metadata={"Date": None} if p.suffix == ".svg" else ({"CreationDate": None} if p.suffix == ".pdf" else None))
    plt.close(fig)
    return paths
