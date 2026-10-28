# %% style
"""Figure conventions shared by every figure module: palette, 95% intervals, sizes, and the PDF + PNG saver.

Palette (Okabe–Ito hues, validated colour-blind safe for the two coloured series): expert blue, novice vermilion,
solo grey (deliberately neutral: the condition with no social information), humans in near-black ink, models as
lighter tints of the demonstrator hues. Every interval drawn is 95% (ETI for posteriors, CI for frequentist fits).
Figures are saved under ``<analysis_dir>/figures/<name>.pdf`` and ``.png`` with the fixed names the paper and the
report include (``fig2_primary``, ``figS3_ppc``, …).
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

COLOR = {"expert": "#0072B2", "novice": "#D55E00", "solo": "#7F7F7F", "human": "#1A1A1A",
         "expert-first": "#0072B2", "novice-first": "#D55E00", "model": "#8C8C8C", "band": "#C9C9C9"}
TINT = {"expert": "#9DC6E2", "novice": "#F0B48F"}     # model bands on a demonstrator's hue
ORDER = ["expert-first", "novice-first", "solo"]       # session types, fixed order in every legend
INTERVAL = 0.95
LINE, MARKER, FONT = 1.6, 5, 8                         # thin marks, small readable type
RC = {"font.size": FONT, "axes.titlesize": FONT + 1, "axes.labelsize": FONT, "legend.fontsize": FONT - 1, "xtick.labelsize": FONT - 1,
      "ytick.labelsize": FONT - 1, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
      "ytick.major.width": 0.6, "lines.linewidth": LINE, "lines.markersize": MARKER, "legend.frameon": False, "axes.grid": False,
      "figure.dpi": 110, "savefig.dpi": 300, "pdf.fonttype": 42, "font.family": "sans-serif"}
WIDTH = {"single": 3.4, "double": 7.0}                 # column widths in inches


def apply() -> None:
    """Set the rc conventions for the current process (every figure module calls it first)."""
    matplotlib.rcParams.update(RC)


def ci(lo: float, hi: float) -> str:
    return f"[{lo:.2f}, {hi:.2f}]"


def save(fig, name: str, analysis_dir: Path) -> list:
    """Write ``figures/<name>.pdf`` and ``.png`` under ``analysis_dir``; returns the two paths."""
    out = Path(analysis_dir) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{name}.pdf", out / f"{name}.png"]
    for p in paths:
        fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return paths
