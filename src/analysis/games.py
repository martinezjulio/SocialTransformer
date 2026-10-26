"""Per-play-game table for the descriptive stage (one row per play game, all sessions).

Columns: participant, session, session_order (1..3, randomised per participant), condition
(expert-first, novice-first, solo), round (1, 2), slot (position of the game in the round, 1..4),
play (first, second: the social play games; solo slots 2 and 4 by information state), demonstrator
(expert, novice, none: who was watched just before), payoff (mean true payoff of the chosen arms),
reward (mean observed reward), entropy (Shannon bits over the six choices), n_shots, exploratory
(True for solo slots 1 and 3, which have no social counterpart and stay out of the primary fits).

Run directly with VERSION set to write $OUTPUT_DATA_DIR/analysis/<version>/games/play_games.csv.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

CONDITION = {"expert-play-novice-play-choice": "expert-first", "novice-play-expert-play-choice": "novice-first",
             "play-play-play-play": "solo"}
PLAY_OF_SLOT = {2: "first", 4: "second"}   # social play games sit at slots 2 and 4; solo's 2 and 4 match them
SHOTS_PER_GAME, N_ROUNDS = 6, 2


def entropy_bits(arms: np.ndarray) -> float:
    p = np.bincount(arms)[np.bincount(arms) > 0] / len(arms)
    return float(-(p * np.log2(p)).sum()) + 0.0   # + 0.0 turns the -0.0 of a constant game into 0.0


def play_games(preprocessed: pd.DataFrame) -> pd.DataFrame:
    """One row per play game of every session, in (participant, session_order, round, slot) order."""
    shots = preprocessed[preprocessed["shot"].astype(bool) & (preprocessed["game"] == "play")]
    rows = []
    for (pid, sid, order, seq, game), g in shots.groupby(["anonymous_id", "session_id", "session_index", "game_sequence", "game_index"]):
        payoffs = np.array([float(x) for x in g["zone_bandit"].iloc[0].strip("[]").split(",")])
        slot, cond = int(game) % 4 + 1, CONDITION[seq]
        demonstrator = "none" if cond == "solo" else seq.split("-")[slot - 2]
        rows.append(dict(participant=pid, session=sid, session_order=int(order), condition=cond,
                         round=int(g["round_index"].iloc[0]) + 1, slot=slot, play=PLAY_OF_SLOT.get(slot),
                         demonstrator=demonstrator, payoff=payoffs[g["position"].to_numpy()].mean(),
                         reward=g["reward_generated"].astype(float).mean(), entropy=entropy_bits(g["position"].to_numpy()),
                         n_shots=len(g)))
    table = pd.DataFrame(rows).sort_values(["participant", "session_order", "round", "slot"]).reset_index(drop=True)
    per_session = table.groupby("session")
    assert (table["n_shots"] == SHOTS_PER_GAME).all(), "every play game has six shots"
    assert per_session["round"].nunique().eq(N_ROUNDS).all(), "two rounds per session"
    assert per_session["condition"].nunique().eq(1).all()
    social, solo = table[table["condition"] != "solo"], table[table["condition"] == "solo"]
    assert social["slot"].isin([2, 4]).all() and social.groupby("session").size().eq(2 * N_ROUNDS).all()
    assert solo.groupby("session").size().eq(4 * N_ROUNDS).all() and solo["demonstrator"].eq("none").all()
    assert table.groupby("participant")["condition"].nunique().eq(3).all(), "each participant has one session per condition"
    assert table.groupby("participant")["session_order"].agg(lambda s: sorted(set(s)) == [1, 2, 3]).all()
    assert (social["demonstrator"] == np.where((social["condition"] == "expert-first") == (social["slot"] == 2), "expert", "novice")).all()
    table["exploratory"] = table["play"].isna()
    assert table["exploratory"].sum() == len(solo) // 2 and (table.loc[table["exploratory"], "condition"] == "solo").all()
    return table


def write_play_games(preprocessed_path: Path, analysis_dir: Path) -> Path:
    """Build the table from ``preprocessed_path`` and write ``<analysis_dir>/games/play_games.csv``."""
    out = Path(analysis_dir) / "games" / "play_games.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    play_games(pd.read_csv(preprocessed_path, dtype={"timestamp": str}, low_memory=False)).to_csv(out, index=False)
    return out


if __name__ == "__main__":
    version = os.environ["VERSION"]
    print(write_play_games(Path(os.environ["EXPERIMENT_DATA_DIR"]) / version / "preprocessed_participant_data.csv",
                           Path(os.environ["OUTPUT_DATA_DIR"]) / "analysis" / version))
