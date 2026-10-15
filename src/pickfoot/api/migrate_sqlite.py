"""Idempotently import local Streamlit records into the PostgreSQL schema."""

from __future__ import annotations

import argparse
from contextlib import closing
from datetime import date, datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from pickfoot.api.db import session_factory
from pickfoot.api.models import AdminAudit, Match, MatchAudit, Prediction, Team, TeamAlias


class SQLiteImportError(ValueError):
    """Actionable error raised when a legacy database cannot be imported safely."""


def _timestamp(value: str | None) -> datetime | None:
    if value is None or not value.strip():
        return None
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def _required_timestamp(value: str | None, field: str) -> datetime:
    parsed = _timestamp(value)
    if parsed is None:
        raise SQLiteImportError(f"Legacy SQLite field {field} is empty.")
    return parsed


def _day(value: str) -> date:
    return date.fromisoformat(value[:10])


def _fixture_key(home: str, away: str) -> str:
    return "\x1f".join(sorted((home.casefold(), away.casefold())))


def _stable_id(kind: str, identity: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"pickfoot-legacy-sqlite:{kind}:{identity}"))


def _read_legacy_database(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.is_file():
        raise SQLiteImportError(f"Legacy SQLite database does not exist: {path}")
    uri = path.resolve().as_uri() + "?mode=ro"
    try:
        with closing(sqlite3.connect(uri, uri=True)) as connection:
            connection.row_factory = sqlite3.Row
            tables = {
                str(row["name"])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            expected = {
                "uploads",
                "uploaded_matches",
                "team_aliases",
                "registered_teams",
                "match_corrections",
                "predictions",
            }
            missing = expected - tables
            if missing:
                raise SQLiteImportError(
                    "Legacy SQLite database is missing required tables: "
                    + ", ".join(sorted(missing))
                )
            return {
                table: [
                    dict(row)
                    for row in connection.execute(f'SELECT * FROM "{table}"')
                ]
                for table in sorted(expected)
            }
    except sqlite3.Error as exc:
        raise SQLiteImportError(f"Could not read legacy SQLite database: {exc}") from exc


def _audit(
    session: Session,
    *,
    identity: str,
    action: str,
    entity_type: str,
    entity_id: str,
    details: dict[str, Any],
    created_at: datetime | None,
) -> bool:
    audit_id = _stable_id("admin-audit", identity)
    if session.get(AdminAudit, audit_id) is not None:
        return False
    session.add(AdminAudit(
        id=audit_id,
        actor="legacy-sqlite-import",
        action=action,
        entity_type=entity_type,
        entity_id=entity_id,
        details=details,
        created_at=created_at or datetime.now(timezone.utc),
    ))
    return True


def _team(session: Session, league: str, name: str, *, registered: bool = False) -> Team:
    key = " ".join(name.split()).casefold()
    team = session.scalar(
        select(Team).where(Team.league == league, Team.team_key == key)
    )
    if team is None:
        team = Team(
            league=league,
            team_key=key,
            name=" ".join(name.split()),
            manually_registered=registered,
        )
        session.add(team)
        session.flush()
    elif registered:
        team.manually_registered = True
    return team


def _find_match(
    session: Session,
    *,
    league: str,
    season: str,
    match_date: date,
    home_key: str,
    away_key: str,
) -> Match | None:
    return session.scalar(
        select(Match).where(
            Match.league == league,
            Match.season == season,
            Match.match_date == match_date,
            or_(
                (Match.home_team_key == home_key) & (Match.away_team_key == away_key),
                (Match.home_team_key == away_key) & (Match.away_team_key == home_key),
            ),
        )
    )


def _match_is_same_result(match: Match, row: dict[str, Any]) -> bool:
    return (
        match.status == "completed"
        and match.home_team.casefold() == str(row["home_team"]).casefold()
        and match.away_team.casefold() == str(row["away_team"]).casefold()
        and match.home_goals == int(row["home_goals"])
        and match.away_goals == int(row["away_goals"])
        and match.result == str(row["result"])
    )


def import_legacy_sqlite(
    source: Path,
    *,
    factory=None,
) -> dict[str, int]:
    """Import all SQLite operational records atomically and safely on reruns."""
    records = _read_legacy_database(source)
    factory = factory or session_factory()
    counts = {
        "matches_inserted": 0,
        "matches_existing": 0,
        "teams_registered": 0,
        "aliases_inserted": 0,
        "predictions_inserted": 0,
        "predictions_existing": 0,
        "uploads_audited": 0,
        "corrections_audited": 0,
    }
    upload_by_hash = {row["file_sha256"]: row for row in records["uploads"]}
    legacy_match_ids: dict[tuple[str, str, str, str], str] = {}

    try:
        with factory.begin() as session:
            for row in records["uploads"]:
                imported = _audit(
                    session,
                    identity=f"upload:{row['file_sha256']}",
                    action="legacy_upload_imported",
                    entity_type="legacy_upload",
                    entity_id=str(row["upload_id"]),
                    details={
                        "source_database": source.name,
                        "upload_id": row["upload_id"],
                        "file_sha256": row["file_sha256"],
                        "file_name": row["file_name"],
                        "source_name": row["source_name"],
                        "license_reference": row["license_reference"],
                        "uploaded_at_utc": row["uploaded_at_utc"],
                        "inserted_matches": int(row["inserted_matches"]),
                        "corrected_matches": int(row.get("corrected_matches", 0)),
                        "note": "Legacy SQLite stored provenance and hash, not uploaded file bytes.",
                    },
                    created_at=_required_timestamp(
                        row["uploaded_at_utc"], "uploads.uploaded_at_utc"
                    ),
                )
                counts["uploads_audited"] += int(imported)

            for row in records["uploaded_matches"]:
                league = str(row["league"])
                season = str(row["season"])
                match_date = _day(str(row["match_date"]))
                home = str(row["home_team"])
                away = str(row["away_team"])
                home_key, away_key = home.casefold(), away.casefold()
                fixture = _fixture_key(home, away)
                identity = f"{league}:{season}:{match_date}:{home_key}:{away_key}"
                match_id = _stable_id("match", identity)
                existing = session.get(Match, match_id) or _find_match(
                    session,
                    league=league,
                    season=season,
                    match_date=match_date,
                    home_key=home_key,
                    away_key=away_key,
                )
                if existing is not None:
                    if not _match_is_same_result(existing, row):
                        raise SQLiteImportError(
                            f"Legacy result conflicts with PostgreSQL match "
                            f"{league} {match_date} {home} – {away}; no records were imported."
                        )
                    match = existing
                    counts["matches_existing"] += 1
                else:
                    upload = upload_by_hash.get(row["file_sha256"])
                    if upload is None:
                        raise SQLiteImportError(
                            f"Legacy result references missing upload hash {row['file_sha256']}."
                        )
                    _team(session, league, home)
                    _team(session, league, away)
                    match = Match(
                        id=match_id,
                        league=league,
                        season=season,
                        round_name="Unspecified (legacy SQLite)",
                        match_date=match_date,
                        home_team=home,
                        away_team=away,
                        home_team_key=home_key,
                        away_team_key=away_key,
                        status="completed",
                        home_goals=int(row["home_goals"]),
                        away_goals=int(row["away_goals"]),
                        result=str(row["result"]),
                        source_name=str(upload["source_name"]),
                        license_reference=str(upload["license_reference"]),
                        version=1,
                    )
                    session.add(match)
                    session.flush()
                    _audit(
                        session,
                        identity=f"match:{identity}",
                        action="legacy_match_imported",
                        entity_type="match",
                        entity_id=match.id,
                        details={
                            "file_sha256": row["file_sha256"],
                            "round_name": None,
                            "source_database": source.name,
                        },
                        created_at=_required_timestamp(
                            upload["uploaded_at_utc"], "uploads.uploaded_at_utc"
                        ),
                    )
                    counts["matches_inserted"] += 1
                legacy_match_ids[(league, season, str(match_date), fixture)] = match.id

            for row in records["registered_teams"]:
                team = _team(session, str(row["league"]), str(row["team_name"]), registered=True)
                imported = _audit(
                    session,
                    identity=f"registered-team:{row['league']}:{row['team_key']}",
                    action="legacy_team_registration_imported",
                    entity_type="team",
                    entity_id=team.id,
                    details={
                        "team_name": row["team_name"],
                        "registration_note": row["registration_note"],
                    },
                    created_at=_required_timestamp(
                        row["registered_at_utc"], "registered_teams.registered_at_utc"
                    ),
                )
                counts["teams_registered"] += int(imported)

            for row in records["team_aliases"]:
                league = str(row["league"])
                alias_key = str(row["alias_key"])
                identity = (league, alias_key)
                existing = session.get(TeamAlias, identity)
                canonical_team = str(row["canonical_team"])
                if existing is not None:
                    if existing.canonical_team.casefold() != canonical_team.casefold():
                        raise SQLiteImportError(
                            f"Legacy alias {league}:{alias_key} conflicts with an existing PostgreSQL mapping."
                        )
                    continue
                session.add(TeamAlias(
                    league=league,
                    alias_key=alias_key,
                    alias=str(row["alias"]),
                    canonical_team=canonical_team,
                    created_at=_required_timestamp(
                        row["created_at_utc"], "team_aliases.created_at_utc"
                    ),
                ))
                counts["aliases_inserted"] += 1

            for row in records["predictions"]:
                prediction_id = str(row["prediction_id"])
                existing = session.get(Prediction, prediction_id)
                if existing is not None:
                    same = (
                        existing.league == str(row["league"])
                        and existing.match_date == _day(str(row["match_date"]))
                        and existing.model_version == str(row["model_version"])
                        and existing.home_probability == float(row["home_win_probability"])
                        and existing.draw_probability == float(row["draw_probability"])
                        and existing.away_probability == float(row["away_win_probability"])
                    )
                    if not same:
                        raise SQLiteImportError(
                            f"Legacy prediction {prediction_id} differs from its PostgreSQL copy."
                        )
                    counts["predictions_existing"] += 1
                    continue
                data_cutoff = _timestamp(row["data_cutoff_at_utc"])
                session.add(Prediction(
                    id=prediction_id,
                    league=str(row["league"]),
                    season=str(row["season"]),
                    match_date=_day(str(row["match_date"])),
                    home_team=str(row["home_team"]),
                    away_team=str(row["away_team"]),
                    prediction_mode=str(row["prediction_mode"]),
                    predicted_at=_required_timestamp(
                        row["predicted_at_utc"], "predictions.predicted_at_utc"
                    ),
                    data_cutoff_at=data_cutoff,
                    model_version=str(row["model_version"]),
                    model_training_cutoff=_day(str(row["model_training_cutoff"])),
                    team_state_cutoff=_day(str(row["team_state_cutoff"])),
                    state_fingerprint=str(row["team_state_hash"]),
                    home_probability=float(row["home_win_probability"]),
                    draw_probability=float(row["draw_probability"]),
                    away_probability=float(row["away_win_probability"]),
                    actual_result=row["actual_result"],
                    evaluation_status=str(row["evaluation_status"]),
                    feature_snapshot={
                        "legacy_sqlite_import": True,
                        "legacy_data_cutoff_status": row["data_cutoff_status"],
                        "feature_values_available": False,
                    },
                ))
                counts["predictions_inserted"] += 1

            for row in records["match_corrections"]:
                league = str(row["league"])
                season = str(row["season"])
                match_date = str(row["match_date"])
                fixture = str(row["fixture_key"])
                match_id = legacy_match_ids.get((league, season, match_date, fixture))
                if match_id is None:
                    candidates = records["uploaded_matches"]
                    for match_row in candidates:
                        if (
                            str(match_row["league"]) == league
                            and str(match_row["season"]) == season
                            and str(match_row["match_date"]) == match_date
                            and _fixture_key(
                                str(match_row["home_team"]), str(match_row["away_team"])
                            ) == fixture
                        ):
                            match_id = legacy_match_ids.get(
                                (league, season, match_date, fixture)
                            )
                            break
                if match_id is None:
                    raise SQLiteImportError(
                        f"Legacy correction {row['correction_id']} has no imported match."
                    )
                correction_id = str(row["correction_id"])
                audit_id = _stable_id("match-correction", correction_id)
                if session.get(MatchAudit, audit_id) is not None:
                    continue
                session.add(MatchAudit(
                    id=audit_id,
                    match_id=match_id,
                    action="legacy_result_corrected",
                    previous_payload=json.loads(row["previous_result_json"]),
                    new_payload=json.loads(row["corrected_result_json"]),
                    actor="legacy-sqlite-import",
                    reason=(
                        f"Imported correction {correction_id}; source "
                        f"{row['source_name']}; permission {row['license_reference']}."
                    ),
                    created_at=_required_timestamp(
                        row["corrected_at_utc"], "match_corrections.corrected_at_utc"
                    ),
                ))
                counts["corrections_audited"] += 1

            for row in records["team_aliases"]:
                _audit(
                    session,
                    identity=f"team-alias:{row['league']}:{row['alias_key']}",
                    action="legacy_team_alias_imported",
                    entity_type="team_alias",
                    entity_id=f"{row['league']}:{row['alias_key']}",
                    details={
                        "alias": row["alias"],
                        "canonical_team": row["canonical_team"],
                    },
                    created_at=_required_timestamp(
                        row["created_at_utc"], "team_aliases.created_at_utc"
                    ),
                )
    except SQLiteImportError:
        raise
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SQLiteImportError(f"Legacy SQLite data is malformed: {exc}") from exc
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Idempotently import local SQLite uploads and predictions into PostgreSQL."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--confirm-import",
        action="store_true",
        help="confirm that the selected PostgreSQL database is the intended target",
    )
    args = parser.parse_args()
    if not args.confirm_import:
        parser.error("pass --confirm-import after backing up SQLite and verifying the PostgreSQL target")
    try:
        counts = import_legacy_sqlite(args.source)
    except SQLiteImportError as exc:
        parser.exit(2, f"SQLite import stopped without committing: {exc}\n")
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
