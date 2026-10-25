"""Simulate play games with a ``Policy`` in the loop.

``play_segment`` fills one play game: the policy picks an arm from the current
``TrialState``, the reward is a Bernoulli draw from the bandit's true payoff, the
(action, reward) pair is spliced into the session's token tensors where the human's
shot sat, and the state's own counts advance. Rewards use a ``random.Random`` and
actions a numpy Generator, as ``run_inference.py`` does, so seeded rollouts reproduce.
"""

import random
from typing import Dict, List, Tuple

import numpy as np
import torch

from inference.learners import Policy, TrialState


def sample_action(probs: np.ndarray, np_rng: np.random.Generator) -> int:
    """Draw an arm from ``probs``; a one-hot policy (Thompson) takes its arm with no draw."""
    if probs.max() >= 1.0:
        return int(probs.argmax())
    return int(np_rng.choice(len(probs), p=probs))


def play_segment(policy: Policy, state: TrialState, tensors: Dict[str, torch.Tensor], start: int, end: int,
                 payoffs: List[float], rng: random.Random, np_rng: np.random.Generator) -> List[Tuple[int, int]]:
    """Play positions ``start:end`` of ``tensors`` in place; returns the (action, reward) trajectory.

    ``state.own_within`` must hold the counts so far in this game (zeros at its start);
    on return it holds the whole game's counts and ``state.prev_action/prev_reward/trial``
    point past the last shot.
    """
    num_arms = state.own_within.shape[1]
    trajectory = []
    for pos in range(start, end):
        action = sample_action(policy.probs(state), np_rng)
        assert 0 <= action < num_arms, action
        reward = 1 if rng.random() < payoffs[action] else 0
        trajectory.append((action, reward))
        tensors["action_ids"][0, pos] = action
        tensors["reward_ids"][0, pos] = reward
        state.own_within[1 - reward, action] += 1.0
        state.prev_action, state.prev_reward, state.trial = action, reward, state.trial + 1
    return trajectory
