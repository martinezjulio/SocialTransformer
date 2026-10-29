"""Inference CSVs of one version → the human trials and readouts the choice models read.

``load_sessions`` keeps the complete social sessions (every game has the modal number of trials;
``run_inference.py`` already skips solo sessions), asserts the two-round structure, and parses the
readout strings once. Rows per session: 48 trials (2 rounds × 4 games × 6 shots) and 12 cutoff
readouts, one per segment (cutoff c = beliefs after the first c segments; the readout entering play
game g of round r is cutoff 6r + g + 1 for g ∈ {1, 3}, i.e. right after the demonstration before it).
"""

import ast
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

N_ROUNDS, GAMES_PER_ROUND, SEGMENTS_PER_ROUND = 2, 4, 6
SOLO = "play-play-play-play"
READOUTS = ("expert", "novice", "recent")   # ``<name>_betas`` columns: per-arm (alpha, beta) pairs


def parse_betas(s: str) -> np.ndarray:
    """'[(a1, b1), (a2, b2), ...]' → (K, 2) float array."""
    return np.array(ast.literal_eval(s), dtype=float)


def atomic_write(path: Path, write) -> None:
    """``write(tmp)`` to a hidden temporary name beside ``path`` (same suffix, so ``np.save`` keeps it), then rename into
    place: a reader sees the whole cache file or none, so a task killed mid-write (preemption) leaves nothing that looks complete."""
    tmp = Path(path).with_name(f".{Path(path).stem}.tmp{Path(path).suffix}")
    write(tmp)
    os.replace(tmp, path)


@dataclass(frozen=True)
class Sessions:
    version: str
    trials: pd.DataFrame        # human shots of complete social sessions: session_id, anonymous_id, game_sequence, round_index, game_index, demonstrator_type, position, zone_bandit, reward
    readouts: pd.DataFrame      # one row per (session_id, cutoff): theta (K,), and expert/novice/recent (K, 2) arrays
    trials_per_game: int
    dropped: tuple              # incomplete session ids


def checkpoint_path(eval_dir: Path) -> Path:
    """The checkpoint the inference run used, under this machine's output root. The run's metadata records the paths of
    the machine it ran on; the checkpoint's place relative to that machine's output root (the grandparent of the recorded
    eval dir) is taken under the current one (the grandparent of ``eval_dir``), so the metadata stays a record."""
    cfg = yaml.safe_load((Path(eval_dir) / "metadata" / "inference_config.yaml").read_text())
    recorded, old_root = Path(cfg["checkpoint_path"]), Path(cfg["eval_dir"]).parents[1]
    return Path(eval_dir).parents[1] / recorded.relative_to(old_root) if recorded.is_relative_to(old_root) else recorded


def load_sessions(eval_dir: Path, version: str) -> Sessions:
    d = Path(eval_dir) / "data"
    trial = pd.read_csv(d / "experiment_trial_predictions.csv", low_memory=False)
    hum = trial[trial["learner_segment_input"] == "human"]
    n = hum.groupby(["session_id", "game_index"]).size()
    trials_per_game = int(n.mode()[0])
    dropped = tuple(sorted({sid for sid, _ in n[n != trials_per_game].index}))
    hum = hum[~hum["session_id"].isin(dropped)]
    trials = hum[["session_id", "anonymous_id", "game_sequence", "round_index", "game_index", "demonstrator_type", "position", "zone_bandit"]].copy()
    trials["reward"] = hum["reward_generated"].astype(float)
    trials = trials.sort_values(["session_id", "game_index"], kind="stable").reset_index(drop=True)

    cut = pd.read_csv(d / "experiment_cutoff_human_predictions.csv", low_memory=False)
    cut = cut[cut["session_id"].isin(trials["session_id"].unique())].sort_values(["session_id", "cutoff"]).reset_index(drop=True)
    readouts = cut[["session_id", "cutoff", "segment_type", "round_index"]].copy()
    readouts["theta"] = cut["theta_predicted"].map(lambda s: np.array(ast.literal_eval(s), dtype=float))
    for name in READOUTS:   # recent is absent at cutoff 1 (nothing demonstrated yet): kept as None
        readouts[name] = cut[f"{name}_betas"].map(lambda s: parse_betas(s) if isinstance(s, str) else None)

    per = trials.groupby("session_id")
    assert (trials["game_sequence"] != SOLO).all(), "inference CSVs hold social sessions only"
    assert per["round_index"].nunique().eq(N_ROUNDS).all() and per["round_index"].max().eq(N_ROUNDS - 1).all(), "two rounds per session"
    assert per.size().eq(N_ROUNDS * GAMES_PER_ROUND * trials_per_game).all() and per["anonymous_id"].nunique().eq(1).all()
    assert (trials["demonstrator_type"].isna() == (trials["game_index"] % 4).isin([1, 3])).all(), "demonstrator only on watch games"
    assert readouts.groupby("session_id")["cutoff"].apply(lambda c: sorted(c) == list(range(1, N_ROUNDS * SEGMENTS_PER_ROUND + 1))).all()
    K = len(readouts["theta"].iloc[0])
    assert all(a is None or a.shape == (K, 2) for name in READOUTS for a in readouts[name]) and readouts["expert"].notna().all()
    return Sessions(version, trials, readouts, trials_per_game, dropped)


@dataclass(frozen=True)
class TrialData:
    """One entry per human play shot (N), in (session, game, shot) order. Count arrays are (N, K, 2) of
    (successes, failures) on the round's bandit, as the learner sees them before the shot."""
    session: np.ndarray      # session id per shot
    part: np.ndarray         # participant index 0..P-1 (sorted anonymous ids)
    act: np.ndarray          # arm chosen
    prev_arm: np.ndarray     # previous arm in this play segment, -1 on its first shot
    prev_rew: np.ndarray     # its reward, 0.0 on the first shot
    round_pm: np.ndarray     # -0.5 round 1, +0.5 round 2
    seg: np.ndarray          # play segment 1..4 (2 per round)
    expert_first: np.ndarray  # condition: the expert demonstrated first in this session
    last_exp: np.ndarray     # the demonstrator watched just before this segment was the expert
    expert_seen: np.ndarray  # the expert has demonstrated on this round's bandit before this shot
    read: dict               # 'expert', 'novice', 'recent' -> (N, K, 2) readout entering the segment
    own_seg: np.ndarray      # learner's own counts earlier in this segment (S_SEG)
    own_prev: np.ndarray     # learner's own counts from the round's earlier play segment (S_PREV)
    vis_last: np.ndarray     # visible pulls of the demonstration watched last
    vis_expert: np.ndarray   # visible pulls of the expert's demonstration this round (zeros if unseen)
    vis_round: np.ndarray    # every visible pull this round so far, demonstrations and own (legacy naive)
    theta: np.ndarray        # (N, K) true payoffs of the round's bandit
    people: tuple            # anonymous ids in participant-index order


def trial_data(s: Sessions) -> TrialData:
    K = len(s.readouts["theta"].iloc[0])
    people = tuple(sorted(s.trials["anonymous_id"].unique())); pidx = {p: i for i, p in enumerate(people)}
    read = {(r.session_id, r.cutoff): r for r in s.readouts.itertuples()}
    cols = {k: [] for k in ("session", "part", "act", "prev_arm", "prev_rew", "round_pm", "seg", "expert_first", "last_exp", "expert_seen",
                            "own_seg", "own_prev", "vis_last", "vis_expert", "vis_round", "theta")}
    reads = {name: [] for name in READOUTS}
    for sid, sess in s.trials.groupby("session_id", sort=True):
        expert_first = sess["game_sequence"].iloc[0] == "expert-play-novice-play-choice"
        for rnd, games in sess.groupby("round_index", sort=True):
            theta = np.array(ast.literal_eval(games["zone_bandit"].iloc[0]), dtype=float)   # one bandit per round
            carried, vis_round, vis_expert, vis_last, last_exp = np.zeros((K, 2)), np.zeros((K, 2)), np.zeros((K, 2)), None, None
            for gi, g in games.groupby("game_index", sort=True):
                pos, rew = g["position"].to_numpy(), g["reward"].to_numpy()
                if gi % 4 in (0, 2):                                 # a demonstration: tally its visible pulls
                    vis_last = np.zeros((K, 2)); np.add.at(vis_last, (pos, 0), rew); np.add.at(vis_last, (pos, 1), 1 - rew)
                    vis_round += vis_last; last_exp = g["demonstrator_type"].iloc[0] == "expert"
                    if last_exp: vis_expert = vis_expert + vis_last
                    continue
                pre = read[(sid, SEGMENTS_PER_ROUND * rnd + gi % 4 + 1)]   # readout right after that demonstration
                within = np.zeros((K, 2))
                for t, (a, r) in enumerate(zip(pos, rew)):
                    for k, v in dict(session=sid, part=pidx[sess["anonymous_id"].iloc[0]], act=a, prev_arm=pos[t - 1] if t else -1,
                                     prev_rew=rew[t - 1] if t else 0.0, round_pm=rnd - 0.5, seg=2 * rnd + (gi % 4 == 3) + 1, expert_first=expert_first,
                                     last_exp=last_exp, expert_seen=bool(vis_expert.any()), own_seg=within.copy(), own_prev=carried.copy(),
                                     vis_last=vis_last.copy(), vis_expert=vis_expert.copy(), vis_round=vis_round.copy(), theta=theta).items():
                        cols[k].append(v)
                    for name in READOUTS: reads[name].append(getattr(pre, name))
                    within[a, 0] += r; within[a, 1] += 1 - r; vis_round[a, 0] += r; vis_round[a, 1] += 1 - r
                carried = carried + within
    return TrialData(**{k: np.array(v) for k, v in cols.items()}, read={n: np.stack(v) for n, v in reads.items()}, people=people)
