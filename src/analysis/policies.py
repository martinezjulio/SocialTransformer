# %% policies
"""Choice-model specifications and the registry of fitted models (plan 'Models' table; preregistration.md).

Every model is H+Both with terms switched on or off: a belief source scaled by κ_i, the learner's own
counts weighted by λ_w (this play game) and λ_c (the round's earlier play game), Thompson sampling, a
win-stay lose-shift tilt, an optional perseveration bonus, a fixed lapse; the Switch models mix the
expert-readout and novice-readout policies with a weight ω per trial. ``belief`` gives the counts a
source contributes before the learner's own are added; the model body (``policies.py`` below) reads them
through the cached tables of ``thompson.py``.
"""

from dataclasses import dataclass

import numpy as np

from analysis.data import TrialData

SOURCES = ("expert", "novice", "recent", "naive", "vis_last", "vis_expert", "uniform")
MIXTURES = ("none", "two", "by_round", "eight")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    source: str = "expert"      # belief entering the Thompson step (SOURCES); Switch models: expert, mixed with novice
    kappa: bool = True          # κ_i (hierarchical) scales the source; False fixes κ = 1
    own: bool = True            # λ_w, λ_c on the learner's own counts; False leaves them out
    reset: bool = False         # λ_c fixed at 0: the exploratory reset learner (needs own)
    habit: bool = False         # win-stay lose-shift tilt ρ0_i + ρwin_i · previous reward on the previous arm
    persev: bool = False        # one group-level bonus on any arm already chosen in this play game
    mixture: str = "none"       # MIXTURES: two = ω_E, ω_N per participant; by_round = + δ_E, δ_N; eight = group weights
    kappa_trend: bool = False   # δ_κ: round slope on log κ (robustness)
    role: str = "reported"      # full | S2 | S3 | S1 | reported | robustness | exploratory

    def __post_init__(self):
        assert self.source in SOURCES and self.mixture in MIXTURES and (self.own or not self.reset)
        assert self.mixture == "none" or (self.source == "expert" and self.kappa), "Switch models mix the two readouts"


def belief(T: TrialData, source: str) -> np.ndarray:
    """(N, K, 2) counts the source contributes before the learner's own: a readout as is; Beta(1, 1) plus
    the visible pulls for the copying and naive sources (naive already holds the learner's own pulls)."""
    if source in T.read:
        return T.read[source]
    return 1.0 + {"naive": T.vis_round, "vis_last": T.vis_last, "vis_expert": T.vis_expert, "uniform": np.zeros_like(T.vis_round)}[source]


REGISTRY = {m.name: m for m in [
    ModelSpec("H+Both", habit=True, role="full"),
    ModelSpec("src:recent", source="recent", habit=True, role="S2"),
    ModelSpec("src:novice", source="novice", habit=True),
    ModelSpec("src:naive", source="naive", own=False, habit=True),
    ModelSpec("vis:last", source="vis_last", habit=True, role="S3"),
    ModelSpec("vis:expert", source="vis_expert", habit=True),
    ModelSpec("Both"), ModelSpec("H+Social", own=False, habit=True), ModelSpec("H+Own", source="uniform", kappa=False, habit=True),
    ModelSpec("H", source="uniform", kappa=False, own=False, habit=True), ModelSpec("Own", source="uniform", kappa=False),
    ModelSpec("Social", own=False), ModelSpec("Random", source="uniform", kappa=False, own=False),
    ModelSpec("Switch", habit=True, mixture="two"),
    ModelSpec("Switch-by-round", habit=True, mixture="by_round", role="S1"),
    ModelSpec("Switch-by-round+ktrend", habit=True, mixture="by_round", kappa_trend=True, role="robustness"),
    ModelSpec("Switch-8", habit=True, mixture="eight", role="exploratory"),
    ModelSpec("H+Both+persev", habit=True, persev=True, role="exploratory"),
    ModelSpec("H+Both-reset", habit=True, reset=True, role="exploratory"),
]}
