"""Leakage-safe chronological Elo and recent-form features."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from math import pow

import pandas as pd

INITIAL_ELO = 1500.0
HOME_ADVANTAGE_ELO = 60.0
ELO_K_FACTOR = 20.0
ELO_DIVISOR = 400.0
RECENT_MATCH_LIMIT = 5

FEATURE_COLUMNS = (
    "home_elo",
    "away_elo",
    "elo_difference",
    "home_advantage",
    "home_recent_points",
    "home_recent_win_rate",
    "home_recent_goals_for_avg",
    "home_recent_goals_against_avg",
    "home_recent_goal_diff_avg",
    "home_recent_matches",
    "away_recent_points",
    "away_recent_win_rate",
    "away_recent_goals_for_avg",
    "away_recent_goals_against_avg",
    "away_recent_goal_diff_avg",
    "away_recent_matches",
)


class FeatureGenerationError(ValueError):
    """Feature input is not a single normalized competition."""


@dataclass
class LeagueState:
    """Latest known team strength and form for one league."""

    league: str
    elo: dict[str, float] = field(default_factory=dict)
    recent: dict[str, list[dict[str, float]]] = field(default_factory=dict)
    team_names: dict[str, str] = field(default_factory=dict)
    latest_date: date | None = None

    @property
    def teams(self) -> list[str]:
        return sorted(self.team_names.values(), key=str.casefold)

    def register_team(self, team_name: str) -> None:
        """Add an explicitly registered team with the documented new-team prior."""
        display_name = " ".join(team_name.strip().split())
        if not display_name:
            raise FeatureGenerationError("A registered team name cannot be empty.")
        key = display_name.casefold()
        self.team_names.setdefault(key, display_name)
        self.elo.setdefault(key, INITIAL_ELO)
        self.recent.setdefault(key, [])

    def features_for(self, home_team: str, away_team: str) -> dict[str, float]:
        home_key, away_key = home_team.casefold(), away_team.casefold()
        unknown = [team for team, key in ((home_team, home_key), (away_team, away_key))
                   if key not in self.team_names]
        if unknown:
            raise FeatureGenerationError(
                f"Unknown team(s) for {self.league}: {', '.join(unknown)}. "
                "Train this league with matches for both teams first."
            )
        home_elo = self.elo.get(home_key, INITIAL_ELO)
        away_elo = self.elo.get(away_key, INITIAL_ELO)
        return {
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_difference": home_elo - away_elo,
            "home_advantage": HOME_ADVANTAGE_ELO,
            **self._form_features(home_key, "home"),
            **self._form_features(away_key, "away"),
        }

    def _form_features(self, team_key: str, prefix: str) -> dict[str, float]:
        matches = self.recent.get(team_key, [])[-RECENT_MATCH_LIMIT:]
        count = len(matches)
        return {
            f"{prefix}_recent_points": sum(match["points"] for match in matches),
            f"{prefix}_recent_win_rate": (
                sum(match["won"] for match in matches) / count if count else 0.0
            ),
            f"{prefix}_recent_goals_for_avg": (
                sum(match["goals_for"] for match in matches) / count if count else 0.0
            ),
            f"{prefix}_recent_goals_against_avg": (
                sum(match["goals_against"] for match in matches) / count if count else 0.0
            ),
            f"{prefix}_recent_goal_diff_avg": (
                sum(match["goals_for"] - match["goals_against"] for match in matches) / count
                if count else 0.0
            ),
            f"{prefix}_recent_matches": float(count),
        }


@dataclass
class FeatureBuild:
    """Pre-match feature rows, labels, and the resulting latest league state."""

    features: pd.DataFrame
    labels: pd.Series
    state: LeagueState
    elo_history: pd.DataFrame


def _expected_home(elo_home: float, elo_away: float) -> float:
    return 1.0 / (1.0 + pow(10.0, (elo_away - (elo_home + HOME_ADVANTAGE_ELO)) / ELO_DIVISOR))


def build_features(matches: pd.DataFrame) -> FeatureBuild:
    """Create pre-match features; apply each date's results only after its features."""
    if matches.empty:
        raise FeatureGenerationError("Cannot build features from an empty match set.")
    leagues = matches["league"].unique()
    if len(leagues) != 1:
        raise FeatureGenerationError(
            "Feature state is isolated per league; filter the CSV to one league first."
        )

    league = str(leagues[0])
    state = LeagueState(league=league)
    feature_rows: list[dict[str, float]] = []
    labels: list[str] = []
    history_rows: list[dict[str, object]] = []
    ordered = matches.sort_values(["match_date", "home_team", "away_team"], kind="stable")

    for match_date, day_matches in ordered.groupby("match_date", sort=True):
        daily = day_matches.to_dict("records")
        daily_updates: dict[str, float] = {}
        daily_history: list[tuple[str, str, dict[str, float], dict[str, float]]] = []
        daily_elo_records: list[tuple[int, str, str, float, float]] = []

        for match in daily:
            home_team, away_team = str(match["home_team"]), str(match["away_team"])
            home_key, away_key = home_team.casefold(), away_team.casefold()
            state.team_names.setdefault(home_key, home_team)
            state.team_names.setdefault(away_key, away_team)

            # Every fixture on the date sees the unchanged start-of-day state.
            feature_rows.append(state.features_for(home_team, away_team))
            labels.append(str(match["result"]))

            home_goals, away_goals = int(match["home_goals"]), int(match["away_goals"])
            actual = 1.0 if home_goals > away_goals else 0.5 if home_goals == away_goals else 0.0
            rating_home = state.elo.get(home_key, INITIAL_ELO)
            rating_away = state.elo.get(away_key, INITIAL_ELO)
            expected_home = _expected_home(rating_home, rating_away)
            change = ELO_K_FACTOR * (actual - expected_home)
            daily_updates[home_key] = daily_updates.get(home_key, 0.0) + change
            daily_updates[away_key] = daily_updates.get(away_key, 0.0) - change
            history_rows.append({
                "match_date": pd.Timestamp(match["match_date"]).date(),
                "league": league,
                "season": str(match["season"]),
                "home_team": home_team,
                "away_team": away_team,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "result": str(match["result"]),
                "home_elo_before": rating_home,
                "away_elo_before": rating_away,
                "expected_home_score": expected_home,
                "expected_away_score": 1.0 - expected_home,
                "actual_home_score": actual,
                "actual_away_score": 1.0 - actual,
                "home_elo_change": change,
                "away_elo_change": -change,
            })
            daily_elo_records.append(
                (len(history_rows) - 1, home_key, away_key, rating_home, rating_away)
            )

            home_points = 3.0 if actual == 1.0 else 1.0 if actual == 0.5 else 0.0
            away_points = 3.0 if actual == 0.0 else 1.0 if actual == 0.5 else 0.0
            daily_history.append(
                (
                    home_key,
                    away_key,
                    {
                        "points": home_points,
                        "won": float(actual == 1.0),
                        "goals_for": float(home_goals),
                        "goals_against": float(away_goals),
                    },
                    {
                        "points": away_points,
                        "won": float(actual == 0.0),
                        "goals_for": float(away_goals),
                        "goals_against": float(home_goals),
                    },
                )
            )

        for index, home_key, away_key, rating_home, rating_away in daily_elo_records:
            history_rows[index]["home_elo_after"] = rating_home + daily_updates[home_key]
            history_rows[index]["away_elo_after"] = rating_away + daily_updates[away_key]
        for team_key, delta in daily_updates.items():
            state.elo[team_key] = state.elo.get(team_key, INITIAL_ELO) + delta
        for home_key, away_key, home_history, away_history in daily_history:
            state.recent.setdefault(home_key, []).append(home_history)
            state.recent.setdefault(away_key, []).append(away_history)
        state.latest_date = pd.Timestamp(match_date).date()

    feature_frame = pd.DataFrame(feature_rows, columns=FEATURE_COLUMNS)
    feature_frame.index = ordered.index
    feature_frame = feature_frame.sort_index()
    label_series = pd.Series(labels, index=ordered.index, name="result").sort_index()
    return FeatureBuild(feature_frame, label_series, state, pd.DataFrame(history_rows))
