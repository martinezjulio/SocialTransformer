"""Load the preprocessed human sessions and check their structure.

Lifted from ``run_inference.py``. Shots are ordered by ``step`` within each game and
that order is checked against the client timestamps, so the file's row order never
matters (the 19 Oct inference read shots in file order; see the plan's Findings).
"""

from typing import Dict, List, Tuple

import pandas as pd
import torch

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
    cols = SHOT_COLUMNS if "timestamp" in df.columns else [c for c in SHOT_COLUMNS if c != "timestamp"]
    shots = df[df["shot"] & df["game"].isin(["watch", "play"])][cols]
    shots = shots.sort_values(["session_id", "round_index", "game_index", "step"]).reset_index(drop=True)
    game_steps = shots.groupby(["session_id", "game_index"])["step"]
    assert not game_steps.apply(lambda s: s.duplicated().any()).any(), "duplicate step within a game"
    if "timestamp" in shots.columns:   # absent from the public release, which keeps only the step order
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


def parse_bandit_payoffs(zone_bandit_str: str) -> List[float]:
    """
    Parse zone_bandit string to list of float payoffs.
    
    Args:
        zone_bandit_str: e.g., "0.55,0,0.8,0.5,0.4,0.3,0.6,0.75,0.1,0.35,0.45,0.2"
        
    Returns:
        List of payoff probabilities
    """
    cleaned = zone_bandit_str.strip().strip('[]')
    return [float(x) for x in cleaned.split(',')]


def convert_session_to_model_format(
    session_df: pd.DataFrame,
    visible_game_indices: List[int],
    num_arms: int = 12
) -> Dict:
    """
    Convert session data to model input format.
    
    Args:
        session_df: DataFrame for a single session
        visible_game_indices: Which game_index values to include
        num_arms: Number of bandit arms
        
    Returns:
        Dictionary with model input tensors and metadata
    """
    # Filter to visible games
    visible_df = session_df[session_df['game_index'].isin(visible_game_indices)].copy()
    
    # Map zone_global_index to local bandit IDs (0-indexed per session)
    zone_to_bandit = {zone: idx for idx, zone in enumerate(sorted(visible_df['zone_global_index'].unique()))}
    visible_df['bandit_id'] = visible_df['zone_global_index'].map(zone_to_bandit)
    
    # Map demonstrator types to agent IDs
    # Learner = 0, then assign IDs to demonstrators in order they appear
    demo_types = []
    for _, row in visible_df.iterrows():
        if row['game'] == 'watch' and pd.notna(row['demonstrator_type']):
            if row['demonstrator_type'] not in demo_types:
                demo_types.append(row['demonstrator_type'])
    
    demo_to_agent = {demo_type: idx + 1 for idx, demo_type in enumerate(demo_types)}
    
    # Build events list with segment boundaries matching training data format:
    # Each segment (watch/play game) gets a boundary, plus QUERY tokens at end of each round
    events = []
    segment_boundaries = [0]
    segment_info = []
    round_boundaries = [0]
    current_round = None
    current_game_idx = None
    rounds_seen = []
    
    for idx, (_, row) in enumerate(visible_df.iterrows()):
        bandit_id = int(row['bandit_id'])
        action = int(row['position'])
        reward = 1 if row['reward_generated'] else 0

        if row['game'] == 'watch':
            agent_id = demo_to_agent[row['demonstrator_type']]
            seg_type = "demo"
        else:
            agent_id = 0
            seg_type = "learner"

        if current_round is None:
            current_round = row['round_index']
            current_game_idx = row['game_index']
            rounds_seen.append(current_round)

            # BANDIT_SEP token at start of first round
            events.append((bandit_id, -1, -1, -1))
            segment_boundaries.append(len(events))
            segment_info.append({"round": int(current_round), "type": "bandit_sep", "agent": -1})
        elif row['round_index'] != current_round:
            # End of previous segment
            segment_boundaries.append(len(events))
            prev_row = visible_df.iloc[idx - 1]
            prev_agent = demo_to_agent[prev_row['demonstrator_type']] if prev_row['game'] == 'watch' else 0
            segment_info.append({
                "round": int(current_round),
                "type": "demo" if prev_row['game'] == 'watch' else "learner",
                "agent": prev_agent
            })

            # QUERY token at end of previous round
            prev_bandit = int(visible_df.iloc[idx - 1]['bandit_id'])
            events.append((prev_bandit, -1, -2, -1))
            segment_boundaries.append(len(events))
            segment_info.append({
                "round": int(current_round),
                "type": "query",
                "agent": -1,
                "bandit": prev_bandit
            })

            round_boundaries.append(len(events))
            current_round = row['round_index']
            current_game_idx = row['game_index']
            rounds_seen.append(current_round)

            # BANDIT_SEP token at start of new round
            events.append((bandit_id, -1, -1, -1))
            segment_boundaries.append(len(events))
            segment_info.append({"round": int(current_round), "type": "bandit_sep", "agent": -1})
        elif row['game_index'] != current_game_idx:
            # New segment within same round
            segment_boundaries.append(len(events))
            prev_row = visible_df.iloc[idx - 1]
            prev_agent = demo_to_agent[prev_row['demonstrator_type']] if prev_row['game'] == 'watch' else 0
            segment_info.append({
                "round": int(current_round),
                "type": "demo" if prev_row['game'] == 'watch' else "learner",
                "agent": prev_agent
            })
            current_game_idx = row['game_index']

        events.append((bandit_id, agent_id, action, reward))
    
    # Final segment info for last segment
    last_row = visible_df.iloc[-1]
    last_agent = demo_to_agent[last_row['demonstrator_type']] if last_row['game'] == 'watch' else 0
    segment_boundaries.append(len(events))
    segment_info.append({
        "round": int(last_row['round_index']),
        "type": "demo" if last_row['game'] == 'watch' else "learner",
        "agent": last_agent
    })
    
    # QUERY token at end of last round
    last_bandit = int(last_row['bandit_id'])
    events.append((last_bandit, -1, -2, -1))
    segment_boundaries.append(len(events))
    segment_info.append({
        "round": int(last_row['round_index']),
        "type": "query",
        "agent": -1,
        "bandit": last_bandit
    })
    
    round_boundaries.append(len(events))
    
    # Round-to-query-index mapping (each round produces one QUERY, indexed 0, 1, 2...)
    sorted_rounds = sorted(visible_df['round_index'].unique())
    round_to_index = {r: i for i, r in enumerate(sorted_rounds)}
    
    round_bandit_map = []
    for round_idx in sorted_rounds:
        round_df = visible_df[visible_df['round_index'] == round_idx]
        round_bandit_map.append(int(round_df.iloc[0]['bandit_id']))
    
    bandits_info = []
    for zone_idx in sorted(zone_to_bandit.keys()):
        zone_df = visible_df[visible_df['zone_global_index'] == zone_idx]
        payoffs = parse_bandit_payoffs(zone_df.iloc[0]['zone_bandit'])
        bandits_info.append({
            'zone_global_index': int(zone_idx),
            'payoffs': payoffs
        })
    
    return {
        'events': events,
        'segment_boundaries': segment_boundaries,
        'segment_info': segment_info,
        'round_boundaries': round_boundaries,
        'round_bandit_map': round_bandit_map,
        'round_to_index': round_to_index,
        'bandits_info': bandits_info,
        'zone_to_bandit': zone_to_bandit,
        'demo_to_agent': demo_to_agent,
        'num_demonstrators': len(demo_types)
    }


def encode_events_to_tensors(
    events: List[Tuple[int, int, int, int]],
    num_arms: int,
    num_bandits: int,
    num_agents: int,
    SPECIAL: Dict[str, int]
) -> Dict[str, torch.Tensor]:
    """
    Encode events to tensor format for model input.
    Handles structural tokens (BANDIT_SEP and QUERY) which use ag=-1 sentinel.
    
    Args:
        events: List of (bandit_id, agent_id, action, reward) tuples
                QUERY tokens: (bandit_id, -1, -2, -1)
                BANDIT_SEP tokens: (bandit_id, -1, -1, -1) 
        num_arms: Number of bandit arms
        num_bandits: Number of unique bandits
        num_agents: Number of agents (for padding)
        SPECIAL: Dictionary of special token IDs
        
    Returns:
        Dictionary of tensors
    """
    bandit_ids = []
    agent_ids = []
    action_ids = []
    reward_ids = []
    special_ids = []
    
    for (b, ag, a, r) in events:
        # Check if structural token (uses -1 sentinel)
        if ag == -1:
            # Distinguish BANDIT_SEP (a=-1) from QUERY (a=-2)
            if a == -2:
                # QUERY token
                bandit_ids.append(min(max(b, 0), num_bandits - 1))
                agent_ids.append(num_agents)  # padding
                action_ids.append(num_arms)   # padding
                reward_ids.append(2)          # padding
                special_ids.append(SPECIAL["QUERY"])
            else:
                # BANDIT_SEP token
                bandit_ids.append(min(max(b, 0), num_bandits - 1))
                agent_ids.append(num_agents)  # padding
                action_ids.append(num_arms)   # padding
                reward_ids.append(2)          # padding
                special_ids.append(SPECIAL["BANDIT_SEP"])
        else:
            # Regular event
            bandit_ids.append(min(max(b, 0), num_bandits - 1))
            agent_ids.append(ag)
            action_ids.append(a if (0 <= a < num_arms) else num_arms)
            reward_ids.append(r if r in (0, 1) else 2)
            special_ids.append(SPECIAL["NONE"])
    
    return {
        'bandit_ids': torch.tensor(bandit_ids, dtype=torch.long).unsqueeze(0),  # [1, T]
        'agent_ids': torch.tensor(agent_ids, dtype=torch.long).unsqueeze(0),
        'action_ids': torch.tensor(action_ids, dtype=torch.long).unsqueeze(0),
        'reward_ids': torch.tensor(reward_ids, dtype=torch.long).unsqueeze(0),
        'special_ids': torch.tensor(special_ids, dtype=torch.long).unsqueeze(0),
    }
