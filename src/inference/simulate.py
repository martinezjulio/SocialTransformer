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
import pandas as pd
import torch

from inference.learners import Policy, TrialState
from inference.readout import readout
from inference.sessions import convert_session_to_model_format, encode_events_to_tensors, parse_bandit_payoffs
from utils import get_special_characters


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


def betas_str(alpha: torch.Tensor, beta: torch.Tensor) -> str:
    """Same serialisation as ``run_inference.py``: "[(a1, b1), (a2, b2), ...]"."""
    return str([(float(a), float(b)) for a, b in zip(alpha.cpu(), beta.cpu())])


def simulate_session(model: torch.nn.Module, normalizer, session_df: pd.DataFrame, policy: Policy, rng: random.Random,
                     np_rng: np.random.Generator, masking_strategy: str = "all_bandits", num_arms: int = 12, num_agents: int = 3) -> pd.DataFrame:
    """Replay one social session with ``policy`` playing every play game.

    Watch rows are the human's; play rows hold the simulated shots in the human trial schema
    (``step`` 0.. within the game, no timestamp) plus the readout entering the game as
    ``expert_betas``/``novice_betas``. The readout is recomputed before each play game on the
    tensors spliced so far; own counts carry across the two play games of a round only.
    """
    games = sorted(session_df["game_index"].unique())
    info = convert_session_to_model_format(session_df, games, num_arms)
    tensors = encode_events_to_tensors(info["events"], num_arms, len(info["bandits_info"]), num_agents, get_special_characters())
    segs, bounds = info["segment_info"], info["segment_boundaries"]
    expert, novice = info["demo_to_agent"]["expert"] - 1, info["demo_to_agent"]["novice"] - 1
    next_game, last_watched, own, out = iter(games), None, {}, []
    for seg_idx, seg in enumerate(segs):
        if seg["type"] == "demo":
            last_watched = seg["agent"] - 1
            out.append(session_df[session_df["game_index"] == next(next_game)].assign(expert_betas=np.nan, novice_betas=np.nan))
        elif seg["type"] == "learner":
            game_df, r = session_df[session_df["game_index"] == next(next_game)], seg["round"]
            q = info["round_to_index"][r]
            o = readout(model, tensors, bounds, [seg_idx], masking_strategy, [s["round"] for s in segs], info["round_bandit_map"], normalizer)
            a_demo, b_demo = o["alpha_demo"][0, q].cpu(), o["beta_demo"][0, q].cpu()
            theta = (o["alpha_theta"][0, q].cpu(), o["beta_theta"][0, q].cpu()) if "alpha_theta" in o else (None, None)
            carry = own.get(r, torch.zeros(2, num_arms))
            state = TrialState(a_demo, b_demo, *theta, expert, novice, last_watched, torch.zeros(2, num_arms), carry, round_index=r)
            traj = play_segment(policy, state, tensors, bounds[seg_idx], bounds[seg_idx + 1], parse_bandit_payoffs(game_df.iloc[0]["zone_bandit"]), rng, np_rng)
            own[r] = carry + state.own_within
            rows = pd.DataFrame([{**game_df.iloc[0].to_dict(), "step": i, "position": a, "reward_generated": bool(rw), "timestamp": np.nan} for i, (a, rw) in enumerate(traj)])
            out.append(rows.assign(expert_betas=betas_str(a_demo[expert], b_demo[expert]), novice_betas=betas_str(a_demo[novice], b_demo[novice])))
    return pd.concat(out, ignore_index=True)
