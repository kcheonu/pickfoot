"""League and team views built from the shared chronological Elo engine."""

from __future__ import annotations

import pandas as pd

from pickfoot.features import FeatureBuild, build_features


def build_league_analytics(matches: pd.DataFrame) -> FeatureBuild:
    """Replay one league and return model features plus match-level Elo history."""
    return build_features(matches)


def team_match_history(elo_history: pd.DataFrame, team: str) -> pd.DataFrame:
    """Return a team's matches in date order, with a consistent team-side view."""
    if elo_history.empty:
        return elo_history.copy()
    key = team.casefold()
    home = elo_history["home_team"].str.casefold() == key
    away = elo_history["away_team"].str.casefold() == key
    records: list[dict[str, object]] = []
    for _, match in elo_history.loc[home | away].iterrows():
        is_home = str(match["home_team"]).casefold() == key
        result = str(match["result"])
        team_result = result if is_home else {"H": "A", "A": "H", "D": "D"}[result]
        records.append({
            "match_date": match["match_date"],
            "season": match["season"],
            "opponent": match["away_team"] if is_home else match["home_team"],
            "venue": "Home" if is_home else "Away",
            "team_goals": match["home_goals"] if is_home else match["away_goals"],
            "opponent_goals": match["away_goals"] if is_home else match["home_goals"],
            "result": team_result,
            "elo_before": match["home_elo_before"] if is_home else match["away_elo_before"],
            "elo_after": match["home_elo_after"] if is_home else match["away_elo_after"],
            "elo_change": match["home_elo_change"] if is_home else match["away_elo_change"],
            "expected_score": match["expected_home_score"] if is_home
            else match["expected_away_score"],
            "home_team": match["home_team"],
            "away_team": match["away_team"],
        })
    return pd.DataFrame(records).sort_values(
        ["match_date", "opponent"], kind="stable"
    ).reset_index(drop=True)


def league_standings(matches: pd.DataFrame, feature_build: FeatureBuild) -> pd.DataFrame:
    """Summarize current Elo, last match, newest-season games, and recent points."""
    state = feature_build.state
    if matches.empty:
        return pd.DataFrame()
    latest_season = str(matches["season"].max())
    season_counts: dict[str, int] = {}
    last_dates: dict[str, object] = {}
    for _, match in matches.iterrows():
        for team_column in ("home_team", "away_team"):
            key = str(match[team_column]).casefold()
            if str(match["season"]) == latest_season:
                season_counts[key] = season_counts.get(key, 0) + 1
            last_dates[key] = max(last_dates.get(key, match["match_date"]), match["match_date"])
    rows = []
    for key, team in state.team_names.items():
        form = state._form_features(key, "team")
        rows.append({
            "team": team,
            "elo": state.elo.get(key, 1500.0),
            "last_match_date": last_dates.get(key),
            "season": latest_season,
            "season_matches": season_counts.get(key, 0),
            "recent_points": form["team_recent_points"],
            "recent_matches": int(form["team_recent_matches"]),
        })
    standings = pd.DataFrame(rows).sort_values(
        ["elo", "team"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
    standings.insert(0, "rank", standings["elo"].rank(method="min", ascending=False).astype(int))
    return standings


def active_season_roster(
    matches: pd.DataFrame,
    *,
    registered_teams: list[str] | None = None,
) -> set[str]:
    """Use observed latest-season teams plus manually confirmed teams as the active roster."""
    if matches.empty:
        return {team.casefold() for team in (registered_teams or [])}
    latest_season = str(matches["season"].max())
    latest = matches.loc[matches["season"] == latest_season]
    teams = set(latest["home_team"].str.casefold()) | set(latest["away_team"].str.casefold())
    teams.update(team.casefold() for team in (registered_teams or []))
    return teams
