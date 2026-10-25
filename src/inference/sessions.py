"""Load the preprocessed human sessions and check their structure.

Lifted from ``run_inference.py``. Shots are ordered by ``step`` within each game and
that order is checked against the client timestamps, so the file's row order never
matters (the 19 Oct inference read shots in file order; see the plan's Findings).
"""

import pandas as pd

SHOT_COLUMNS = [
    "session_id", "session_index", "anonymous_id", "position", "action", "game",
    "game_sequence", "game_index", "round_index", "demonstrator_type", "zone_bandit",
    "zone_global_index", "reward_generated", "step", "timestamp",
]
TRUE_STRINGS = ["TRUE", "T", "1", "YES"]


def load_shots(csv_path: str) -> pd.DataFrame:
    """Shot rows of watch/play games in social sessions, in (session, round, game, step) order."""
    df = pd.read_csv(csv_path, dtype={"timestamp": str})
    for col in ("shot", "reward_generated"):
        df[col] = df[col].astype(str).str.upper().isin(TRUE_STRINGS)
    shots = df[df["shot"] & df["game"].isin(["watch", "play"])][SHOT_COLUMNS]
    shots = shots.sort_values(["session_id", "round_index", "game_index", "step"]).reset_index(drop=True)
    game_steps = shots.groupby(["session_id", "game_index"])["step"]
    assert not game_steps.apply(lambda s: s.duplicated().any()).any(), "duplicate step within a game"
    in_time_order = shots.groupby(["session_id", "game_index"])["timestamp"].apply(lambda t: t.is_monotonic_increasing)
    assert in_time_order.all(), f"step order disagrees with timestamps in games: {in_time_order.index[~in_time_order].tolist()}"
    # Solo sessions have no demonstrators, so there is nothing for the network to read out.
    has_watch = shots.groupby("session_id")["game"].apply(lambda g: (g == "watch").any())
    solo = sorted(has_watch.index[~has_watch])
    if solo:
        print(f"Skipping {len(solo)} solo session(s) with no watch games")
    return shots[~shots["session_id"].isin(solo)].reset_index(drop=True)


def validate_sessions(shots: pd.DataFrame, n_rounds: int = 2, games_per_round: int = 4, shots_per_play: int = 6) -> None:
    """Raise ValueError unless every session has all rounds, all games, and full play games."""
    expected_rounds = set(range(n_rounds))
    bad = {}
    for sid, sdf in shots.groupby("session_id"):
        reasons = []
        rounds = set(int(r) for r in sdf["round_index"].unique())
        if rounds != expected_rounds or sdf["game_index"].nunique() != games_per_round * n_rounds:
            reasons.append(f"incomplete-rounds (rounds={sorted(rounds)}, n_games={sdf['game_index'].nunique()})")
        play_counts = sdf[sdf["game"] == "play"].groupby("game_index").size()
        if (play_counts != shots_per_play).any():
            reasons.append(f"short-play {play_counts[play_counts != shots_per_play].to_dict()}")
        if reasons:
            bad[sid] = reasons
    if bad:
        lines = [f"  {sid}: {'; '.join(rs)}" for sid, rs in bad.items()]
        raise ValueError(
            f"{len(bad)} session(s) with invalid structure in preprocessed_participant_data.csv. "
            "Re-run src/r/preprocess_participant_data.R to regenerate.\n" + "\n".join(lines)
        )
