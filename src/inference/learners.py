"""Policies that pick arms during a simulated play game.

``TrialState`` is what the simulation loop knows before each shot; a ``Policy`` turns it into
a probability vector over arms. ``ThompsonLearner`` reproduces today's ``run_inference.py``
learner: one Beta draw per arm from the chosen readout plus the learner's own counts, so its
"probabilities" are a sampled one-hot rather than the exact win probabilities.
"""

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import numpy as np
import torch

from inference.inference import thompson_sample_action


@dataclass
class TrialState:
    alpha_demo: torch.Tensor            # [D, K] readout entering this play game, denormalised
    beta_demo: torch.Tensor             # [D, K]
    alpha_theta: Optional[torch.Tensor]  # [K], None for BCE-theta checkpoints
    beta_theta: Optional[torch.Tensor]
    expert: int                         # demonstrator index of the expert in the readout axis
    novice: int
    last_watched: Optional[int]         # most recent demonstrator before this play game, None if none
    own_within: torch.Tensor            # [2, K] (wins, losses) so far in this play game
    own_carry: torch.Tensor             # [2, K] from the earlier play game of this round
    prev_action: Optional[int] = None   # None on the first shot of the game
    prev_reward: Optional[int] = None
    round_index: int = 0
    trial: int = 0
    vis_last: Optional[torch.Tensor] = None   # [2, K] (wins, losses) visible in the demonstration watched last
    vis_expert: Optional[torch.Tensor] = None   # [2, K] visible in the expert's demonstrations this round (zeros if not yet seen)
    vis_round: Optional[torch.Tensor] = None    # [2, K] every pull visible this round before this game: the demonstrations and the learner's earlier play game


@runtime_checkable
class Policy(Protocol):
    def probs(self, state: TrialState) -> np.ndarray: ...  # (K,) summing to 1


class ThompsonLearner:
    def __init__(self, beta_method: str, own_experience: str, np_rng: np.random.Generator):
        assert own_experience in ("carry", "reset"), own_experience
        self.beta_method, self.own_experience, self.np_rng = beta_method, own_experience, np_rng

    def belief(self, s: TrialState) -> tuple:
        if self.beta_method == "theta_betas":
            if s.alpha_theta is None:
                raise ValueError("beta_method='theta_betas' needs a Beta theta head; this checkpoint uses BCE")
            alpha, beta = s.alpha_theta, s.beta_theta
        elif self.beta_method in ("novice_betas", "expert_betas"):
            d = s.novice if self.beta_method == "novice_betas" else s.expert
            alpha, beta = s.alpha_demo[d], s.beta_demo[d]
        elif self.beta_method == "recent_betas":
            if s.last_watched is None and s.alpha_theta is None:
                raise ValueError("beta_method='recent_betas' found no earlier demonstrator and no theta posterior to fall back on")
            alpha, beta = (s.alpha_theta, s.beta_theta) if s.last_watched is None else (s.alpha_demo[s.last_watched], s.beta_demo[s.last_watched])
        else:
            raise ValueError(f"Unknown beta_method: {self.beta_method}")
        own = s.own_within + (s.own_carry if self.own_experience == "carry" else 0)
        return alpha + own[0].to(alpha), beta + own[1].to(beta)

    def probs(self, state: TrialState) -> np.ndarray:
        alpha, beta = self.belief(state)
        one_hot = np.zeros(alpha.shape[-1])
        one_hot[thompson_sample_action(alpha, beta, self.np_rng)] = 1.0
        return one_hot
