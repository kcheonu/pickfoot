from copy import deepcopy

import pandas as pd
import pytest

from pickfoot.data import normalize_matches
from pickfoot.features import (
    HOME_ADVANTAGE_ELO,
    INITIAL_ELO,
    FeatureGenerationError,
    build_features,
)


def matches(rows):
    defaults = {
        "league": "Premier League",
        "season": "2023-24",
        "home_team": "Alpha",
        "away_team": "Beta",
    }
    return normalize_matches(pd.DataFrame([{**defaults, **row} for row in rows]))


def test_first_match_gets_initial_elo_and_empty_form():
    build = build_features(matches([{
        "match_date": "2023-08-01", "home_goals": 2, "away_goals": 0, "result": "H",
    }]))
    first = build.features.iloc[0]
    assert first["home_elo"] == INITIAL_ELO
    assert first["away_elo"] == INITIAL_ELO
    assert first["elo_difference"] == 0
    assert first["home_advantage"] == HOME_ADVANTAGE_ELO
    assert first["home_recent_matches"] == 0
    assert first["home_recent_points"] == 0


def test_previous_result_updates_future_elo_and_recent_form():
    build = build_features(matches([
        {"match_date": "2023-08-01", "home_goals": 2, "away_goals": 0, "result": "H"},
        {"match_date": "2023-08-08", "home_goals": 1, "away_goals": 1, "result": "D"},
    ]))
    next_match = build.features.iloc[1]
    assert next_match["home_elo"] > next_match["away_elo"]
    assert next_match["home_recent_points"] == 3
    assert next_match["home_recent_win_rate"] == 1
    assert next_match["home_recent_goals_for_avg"] == 2
    assert next_match["home_recent_goals_against_avg"] == 0
    assert next_match["home_recent_goal_diff_avg"] == 2
    assert next_match["home_elo"] == pytest.approx(
        INITIAL_ELO + 20 * (1 - 1 / (1 + 10 ** (-HOME_ADVANTAGE_ELO / 400)))
    )


def test_same_date_results_never_change_same_date_features():
    first = matches([
        {"match_date": "2023-08-01", "home_team": "Alpha", "away_team": "Beta",
         "home_goals": 2, "away_goals": 0, "result": "H"},
        {"match_date": "2023-08-01", "home_team": "Gamma", "away_team": "Delta",
         "home_goals": 1, "away_goals": 0, "result": "H"},
        {"match_date": "2023-08-02", "home_team": "Beta", "away_team": "Gamma",
         "home_goals": 0, "away_goals": 0, "result": "D"},
    ])
    changed = first.copy()
    changed.loc[changed["home_team"] == "Alpha", ["home_goals", "away_goals", "result"]] = [
        0, 3, "A"
    ]
    changed_build = build_features(changed)
    first_build = build_features(first)
    same_day = first["match_date"] == pd.Timestamp("2023-08-01").date()
    pd.testing.assert_frame_equal(
        first_build.features.loc[same_day].reset_index(drop=True),
        changed_build.features.loc[same_day].reset_index(drop=True),
    )
    assert first_build.features.loc[~same_day, "home_elo"].iloc[0] != (
        changed_build.features.loc[~same_day, "home_elo"].iloc[0]
    )


def test_recent_form_uses_only_five_prior_matches_and_caps_history():
    rows = []
    for day in range(1, 8):
        rows.append({
            "match_date": f"2023-08-{day:02d}", "home_team": "Alpha", "away_team": f"Team {day}",
            "home_goals": 1, "away_goals": 0, "result": "H",
        })
    rows.append({
        "match_date": "2023-08-10", "home_team": "Alpha", "away_team": "New Opponent",
        "home_goals": 0, "away_goals": 0, "result": "D",
    })
    final = build_features(matches(rows)).features.iloc[-1]
    assert final["home_recent_matches"] == 5
    assert final["home_recent_points"] == 15
    assert final["home_recent_win_rate"] == 1


def test_new_team_starts_at_initial_rating_and_leagues_cannot_mix():
    build = build_features(matches([
        {"match_date": "2023-08-01", "home_goals": 2, "away_goals": 1, "result": "H"},
    ]))
    assert build.state.elo["alpha"] != INITIAL_ELO
    assert build.state.features_for("Alpha", "Beta")["away_elo"] != INITIAL_ELO
    mixed = pd.concat(
        [matches([{"match_date": "2023-08-01", "home_goals": 2, "away_goals": 1,
                   "result": "H"}]),
         normalize_matches(pd.DataFrame([{
             "league": "La Liga", "season": "2023-24", "match_date": "2023-08-01",
             "home_team": "Celta", "away_team": "Osasuna", "home_goals": 1,
             "away_goals": 0, "result": "H",
         }]))],
        ignore_index=True,
    )
    with pytest.raises(FeatureGenerationError, match="one league"):
        build_features(mixed)


def test_elo_and_recent_form_carry_across_season_boundary():
    build = build_features(matches([
        {"match_date": "2023-05-20", "season": "2022/23", "home_team": "Alpha",
         "away_team": "Beta", "home_goals": 2, "away_goals": 0, "result": "H"},
        {"match_date": "2023-08-12", "season": "2023/24", "home_team": "Alpha",
         "away_team": "Gamma", "home_goals": 1, "away_goals": 0, "result": "H"},
    ]))
    second = build.features.iloc[1]
    assert second["home_elo"] > INITIAL_ELO
    assert second["home_recent_matches"] == 1
    assert build.state.elo["alpha"] > second["home_elo"]


def test_unseen_team_has_actionable_error():
    build = build_features(matches([
        {"match_date": "2023-08-01", "home_goals": 2, "away_goals": 1, "result": "H"},
    ]))
    with pytest.raises(FeatureGenerationError, match="Unknown team"):
        build.state.features_for("Alpha", "Unknown")
