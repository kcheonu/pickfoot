"""Bridges persistent deployment records to PickFoot's existing analysis engine."""

from __future__ import annotations

from datetime import date
import re
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from pickfoot.analytics import league_standings, team_match_history
from pickfoot.api.config import get_settings
from pickfoot.api.models import ActiveModel, DataSource, Match, ModelVersion, Team
from pickfoot.data import CSVValidationError, normalize_matches
from pickfoot.features import FeatureBuild, build_features
from pickfoot.model import ModelArtifact, ModelError, load_artifact

SUPPORTED_LEAGUES = (
    "Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"
)


def canonical_league(league: str) -> str:
    if league not in SUPPORTED_LEAGUES:
        raise CSVValidationError(f"Unsupported league: {league}.")
    return league


def completed_database_matches(session: Session, league: str | None = None) -> pd.DataFrame:
    query = select(Match).where(Match.status == "completed")
    if league:
        query = query.where(Match.league == league)
    rows = session.scalars(query.order_by(Match.match_date, Match.home_team)).all()
    if not rows:
        return pd.DataFrame(columns=[
            "league", "season", "match_date", "home_team", "away_team",
            "home_goals", "away_goals", "result",
        ])
    return pd.DataFrame([{
        "league": row.league,
        "season": row.season,
        "match_date": row.match_date,
        "home_team": row.home_team,
        "away_team": row.away_team,
        "home_goals": row.home_goals,
        "away_goals": row.away_goals,
        "result": row.result,
    } for row in rows])


def available_matches(session: Session, league: str | None = None) -> pd.DataFrame:
    settings = get_settings()
    pieces: list[pd.DataFrame] = []
    active_source = session.scalar(
        select(DataSource).where(DataSource.active.is_(True)).order_by(DataSource.uploaded_at.desc())
    )
    if active_source:
        root = settings.data_storage_dir.resolve()
        base_csv = Path(active_source.file_path).resolve()
        if root not in base_csv.parents:
            raise ModelError("Active dataset path is outside configured persistent data storage.")
    else:
        base_csv = settings.base_matches_csv
    if base_csv.is_file():
        try:
            from pickfoot.data import load_matches

            base = load_matches(base_csv)
            if league:
                base = base.loc[base["league"] == league]
            if not base.empty:
                pieces.append(base)
        except CSVValidationError as exc:
            raise ModelError(f"Base match dataset is invalid: {exc}") from exc
    stored = completed_database_matches(session, league)
    if not stored.empty:
        pieces.append(stored)
    if not pieces:
        raise ModelError(f"No completed match data is available for {league or 'the leagues'}.")
    try:
        return normalize_matches(pd.concat(pieces, ignore_index=True))
    except CSVValidationError as exc:
        raise ModelError(
            "The base dataset and saved results conflict or duplicate a fixture: "
            f"{exc}"
        ) from exc


def current_state(session: Session, league: str) -> tuple[pd.DataFrame, FeatureBuild]:
    matches = available_matches(session, league)
    build = build_features(matches)
    registered = session.scalars(
        select(Team.name).where(Team.league == league)
    ).all()
    for name in registered:
        build.state.register_team(name)
    return matches, build


def model_slug(league: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", league.casefold()).strip("-")


def model_for_league(session: Session, league: str) -> tuple[ModelArtifact, str | None]:
    settings = get_settings()
    active = session.get(ActiveModel, league)
    if active:
        version = session.get(ModelVersion, active.model_version_id)
        if version is None:
            raise ModelError(f"Active model metadata is missing for {league}.")
        root = settings.model_storage_dir.resolve()
        artifact_path = Path(version.artifact_path).resolve()
        if root not in artifact_path.parents:
            raise ModelError("Active artifact path is outside the configured model storage.")
        return load_artifact(artifact_path), version.id
    original = settings.model_storage_dir / f"{model_slug(league)}.joblib"
    return load_artifact(original), None


def match_payload(match: Match) -> dict[str, Any]:
    return {
        "id": match.id,
        "league": match.league,
        "season": match.season,
        "round": match.round_name,
        "match_date": match.match_date.isoformat(),
        "home_team": match.home_team,
        "away_team": match.away_team,
        "home_goals": match.home_goals,
        "away_goals": match.away_goals,
        "result": match.result,
        "status": match.status,
        "version": match.version,
        "updated_at": match.updated_at.isoformat() if match.updated_at else None,
    }


def standings_payload(
    session: Session,
    league: str,
    *,
    team: str | None = None,
) -> dict[str, Any]:
    matches, built = current_state(session, league)
    standings = league_standings(matches, built)
    registered = session.scalars(
        select(Team).where(Team.league == league)
    ).all()
    updates = [entry.created_at for entry in registered]
    saved_match = session.scalar(
        select(Match.updated_at).where(Match.league == league)
        .order_by(Match.updated_at.desc()).limit(1)
    )
    if saved_match is not None:
        updates.append(saved_match)
    last_update = max(updates, default=None)
    records = standings.to_dict("records")
    if team:
        history = team_match_history(built.elo_history, team)
        matching = [row for row in records if str(row["team"]).casefold() == team.casefold()]
        if not matching:
            raise ModelError(f"Team {team!r} is not present in {league} data.")
        return {
            "league": league,
            "as_of": built.state.latest_date,
            "last_updated_at": last_update,
            "team": matching[0],
            "history": history.to_dict("records"),
        }
    return {
        "league": league,
        "as_of": built.state.latest_date,
        "last_updated_at": last_update,
        "rows": records,
        "elo_history": built.elo_history.to_dict("records"),
    }


def outcome(home_goals: int, away_goals: int) -> str:
    return "H" if home_goals > away_goals else "A" if home_goals < away_goals else "D"


def date_in_2026_27(match_date: date) -> bool:
    return date(2026, 7, 1) <= match_date <= date(2027, 6, 30)
