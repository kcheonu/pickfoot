"""Transactional local persistence for result uploads and prediction records."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from hashlib import sha256
from io import BytesIO
import json
import sqlite3
from typing import Any
from uuid import uuid4

import pandas as pd

from pickfoot.data import CSVValidationError, LEAGUES, normalize_matches


class StorageError(ValueError):
    """Actionable local-storage error."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _connect(database: str | Path) -> sqlite3.Connection:
    path = Path(database)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS uploads (
            upload_id TEXT PRIMARY KEY,
            file_sha256 TEXT NOT NULL UNIQUE,
            file_name TEXT NOT NULL,
            source_name TEXT NOT NULL,
            license_reference TEXT NOT NULL,
            uploaded_at_utc TEXT NOT NULL,
            inserted_matches INTEGER NOT NULL,
            corrected_matches INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS uploaded_matches (
            league TEXT NOT NULL,
            season TEXT NOT NULL,
            match_date TEXT NOT NULL,
            fixture_key TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            home_goals INTEGER NOT NULL,
            away_goals INTEGER NOT NULL,
            result TEXT NOT NULL,
            file_sha256 TEXT NOT NULL,
            PRIMARY KEY (league, season, match_date, fixture_key),
            FOREIGN KEY (file_sha256) REFERENCES uploads(file_sha256)
        );
        CREATE TABLE IF NOT EXISTS team_aliases (
            league TEXT NOT NULL,
            alias_key TEXT NOT NULL,
            alias TEXT NOT NULL,
            canonical_team TEXT NOT NULL,
            created_at_utc TEXT NOT NULL,
            PRIMARY KEY (league, alias_key)
        );
        CREATE TABLE IF NOT EXISTS registered_teams (
            league TEXT NOT NULL,
            team_key TEXT NOT NULL,
            team_name TEXT NOT NULL,
            registration_note TEXT NOT NULL,
            registered_at_utc TEXT NOT NULL,
            PRIMARY KEY (league, team_key)
        );
        CREATE TABLE IF NOT EXISTS match_corrections (
            correction_id TEXT PRIMARY KEY,
            league TEXT NOT NULL,
            season TEXT NOT NULL,
            match_date TEXT NOT NULL,
            fixture_key TEXT NOT NULL,
            previous_result_json TEXT NOT NULL,
            corrected_result_json TEXT NOT NULL,
            corrected_at_utc TEXT NOT NULL,
            source_name TEXT NOT NULL,
            license_reference TEXT NOT NULL,
            file_sha256 TEXT NOT NULL,
            FOREIGN KEY (file_sha256) REFERENCES uploads(file_sha256)
        );
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id TEXT PRIMARY KEY,
            league TEXT NOT NULL,
            season TEXT NOT NULL,
            match_date TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            prediction_mode TEXT NOT NULL,
            predicted_at_utc TEXT NOT NULL,
            data_cutoff_at_utc TEXT,
            data_cutoff_status TEXT NOT NULL,
            model_version TEXT NOT NULL,
            model_training_cutoff TEXT NOT NULL,
            team_state_cutoff TEXT NOT NULL,
            team_state_hash TEXT NOT NULL,
            home_win_probability REAL NOT NULL,
            draw_probability REAL NOT NULL,
            away_win_probability REAL NOT NULL,
            actual_result TEXT,
            evaluation_status TEXT NOT NULL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS uq_prediction_state
        ON predictions (
            league, season, match_date, lower(home_team), lower(away_team),
            prediction_mode, model_version, team_state_hash
        );
        """
        )
        upload_columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(uploads)").fetchall()
        }
        if "corrected_matches" not in upload_columns:
            connection.execute(
                "ALTER TABLE uploads ADD COLUMN corrected_matches INTEGER NOT NULL DEFAULT 0"
            )
        return connection
    except OSError as exc:
        raise StorageError(f"Could not create local storage at {path}: {exc}") from exc
    except sqlite3.Error as exc:
        if "connection" in locals():
            connection.close()
        raise StorageError(f"Could not initialize local database {path}: {exc}") from exc


def fixture_key(home_team: str, away_team: str) -> str:
    """Create an orientation-independent key for duplicate/conflict checks."""
    return "\x1f".join(sorted((home_team.casefold(), away_team.casefold())))


def read_uploaded_matches(database: str | Path) -> pd.DataFrame:
    path = Path(database)
    if not path.is_file():
        return pd.DataFrame(columns=[
            "league", "season", "match_date", "home_team", "away_team",
            "home_goals", "away_goals", "result",
        ])
    connection = _connect(path)
    try:
        rows = connection.execute(
            """
            SELECT league, season, match_date, home_team, away_team,
                   home_goals, away_goals, result
            FROM uploaded_matches
            ORDER BY match_date, league, home_team, away_team
            """
        ).fetchall()
    except sqlite3.Error as exc:
        raise StorageError(f"Could not read uploaded match results: {exc}") from exc
    finally:
        connection.close()
    if not rows:
        return pd.DataFrame(columns=[
            "league", "season", "match_date", "home_team", "away_team",
            "home_goals", "away_goals", "result",
        ])
    return normalize_matches(pd.DataFrame([dict(row) for row in rows]))


def read_team_aliases(database: str | Path, league: str | None = None) -> dict[str, str]:
    path = Path(database)
    if not path.is_file():
        return {}
    connection = _connect(path)
    try:
        if league:
            rows = connection.execute(
                "SELECT alias_key, canonical_team FROM team_aliases WHERE league = ?",
                (league,),
            ).fetchall()
        else:
            rows = connection.execute(
                "SELECT league || ':' || alias_key AS alias_key, canonical_team "
                "FROM team_aliases"
            ).fetchall()
    except sqlite3.Error as exc:
        raise StorageError(f"Could not read team aliases: {exc}") from exc
    finally:
        connection.close()
    return {str(row["alias_key"]): str(row["canonical_team"]) for row in rows}


def read_registered_teams(database: str | Path, league: str | None = None) -> dict[str, list[str]]:
    path = Path(database)
    if not path.is_file():
        return {}
    connection = _connect(path)
    try:
        rows = connection.execute(
            """
            SELECT league, team_name FROM registered_teams
            WHERE (? IS NULL OR league = ?)
            ORDER BY team_name COLLATE NOCASE
            """,
            (league, league),
        ).fetchall()
    except sqlite3.Error as exc:
        raise StorageError(f"Could not read registered teams: {exc}") from exc
    finally:
        connection.close()
    output: dict[str, list[str]] = {}
    for row in rows:
        output.setdefault(str(row["league"]), []).append(str(row["team_name"]))
    return output


def register_team(
    database: str | Path,
    *,
    league: str,
    team_name: str,
    registration_note: str,
) -> bool:
    """Register an explicitly user-confirmed team; return whether it was new."""
    normalized_name = " ".join(team_name.strip().split())
    if not normalized_name or not registration_note.strip():
        raise StorageError("A team name and the basis for registering it are required.")
    connection = _connect(database)
    team_key = normalized_name.casefold()
    try:
        connection.execute("BEGIN IMMEDIATE")
        exists = connection.execute(
            "SELECT 1 FROM registered_teams WHERE league = ? AND team_key = ?",
            (league, team_key),
        ).fetchone()
        if exists:
            connection.rollback()
            return False
        connection.execute(
            """
            INSERT INTO registered_teams
            (league, team_key, team_name, registration_note, registered_at_utc)
            VALUES (?, ?, ?, ?, ?)
            """,
            (league, team_key, normalized_name, registration_note.strip(), utc_now()),
        )
        connection.commit()
        return True
    except sqlite3.Error as exc:
        connection.rollback()
        raise StorageError(f"Could not register team: {exc}") from exc
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def normalize_uploaded_bytes(
    content: bytes,
    *,
    database: str | Path,
) -> tuple[pd.DataFrame, str]:
    """Decode UTF-8/UTF-8-BOM CSV and normalize registered team aliases."""
    try:
        frame = pd.read_csv(
            BytesIO(content),
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
        )
    except (UnicodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise CSVValidationError(f"Could not read uploaded CSV as UTF-8: {exc}") from exc
    aliases = read_team_aliases(database)
    columns = {str(column).strip().casefold(): column for column in frame.columns}
    for team_column in ("home_team", "away_team"):
        original_column = columns.get(team_column)
        if original_column is None:
            continue
        league_column = columns.get("league")
        if league_column is None:
            continue
        normalized_names = []
        for league_value, team_value in zip(
            frame[league_column].astype(str),
            frame[original_column].astype(str),
            strict=True,
        ):
            normalized_league = league_value.strip().casefold()
            normalized_names.append(
                aliases.get(
                    f"{LEAGUES.get(normalized_league, league_value.strip())}:"
                    f"{team_value.strip().casefold()}",
                    team_value,
                )
            )
        frame[original_column] = normalized_names
    try:
        matches = normalize_matches(frame)
    except CSVValidationError:
        raise
    if set(matches["season"]) != {"2026-27"}:
        raise CSVValidationError("Result uploads must contain only the 2026-27 season.")
    return matches, sha256(content).hexdigest()


def preview_upload(
    incoming: pd.DataFrame,
    existing: pd.DataFrame,
) -> tuple[pd.DataFrame, int, list[dict[str, Any]]]:
    """Classify new, exact duplicate, and conflicting fixture rows without writes."""
    known: dict[tuple[str, str, str], dict[str, Any]] = {}
    for source in (existing,):
        for row in source.to_dict("records"):
            identity = (
                str(row["league"]),
                str(row["season"]),
                str(row["match_date"]),
            )
            known[(*identity, fixture_key(str(row["home_team"]), str(row["away_team"])))] = row
    new_rows: list[dict[str, Any]] = []
    duplicate_count = 0
    conflicts: list[dict[str, Any]] = []
    for row in incoming.to_dict("records"):
        identity = (
            str(row["league"]),
            str(row["season"]),
            str(row["match_date"]),
            fixture_key(str(row["home_team"]), str(row["away_team"])),
        )
        previous = known.get(identity)
        if previous is None:
            known[identity] = row
            new_rows.append(row)
            continue
        exact = (
            str(previous["home_team"]).casefold() == str(row["home_team"]).casefold()
            and str(previous["away_team"]).casefold() == str(row["away_team"]).casefold()
            and int(previous["home_goals"]) == int(row["home_goals"])
            and int(previous["away_goals"]) == int(row["away_goals"])
        ) or (
            str(previous["home_team"]).casefold() == str(row["away_team"]).casefold()
            and str(previous["away_team"]).casefold() == str(row["home_team"]).casefold()
            and int(previous["home_goals"]) == int(row["away_goals"])
            and int(previous["away_goals"]) == int(row["home_goals"])
        )
        if exact:
            duplicate_count += 1
        else:
            conflicts.append({"incoming": row, "existing": previous})
    return normalize_matches(pd.DataFrame(new_rows)) if new_rows else incoming.iloc[0:0].copy(), duplicate_count, conflicts


def commit_upload(
    database: str | Path,
    *,
    matches: pd.DataFrame,
    content_sha256: str,
    file_name: str,
    source_name: str,
    license_reference: str,
    rights_confirmed: bool,
    base_matches: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Atomically save a validated result batch, provenance, and prediction links."""
    if not rights_confirmed:
        raise StorageError("Confirm that the uploaded data permits the intended local use.")
    if not source_name.strip() or not license_reference.strip():
        raise StorageError("Data source and license/permission reference are required.")
    connection = _connect(database)
    inserted = duplicates = linked = 0
    upload_id = str(uuid4())
    try:
        connection.execute("BEGIN IMMEDIATE")
        existing_hash = connection.execute(
            "SELECT upload_id, inserted_matches FROM uploads WHERE file_sha256 = ?",
            (content_sha256,),
        ).fetchone()
        if existing_hash:
            connection.rollback()
            return {
                "upload_id": str(existing_hash["upload_id"]),
                "inserted_matches": 0,
                "duplicate_matches": len(matches),
                "linked_predictions": 0,
                "already_uploaded": True,
                "uploaded_at_utc": None,
            }

        existing_rows = connection.execute(
            """
            SELECT league, season, match_date, home_team, away_team,
                   home_goals, away_goals, result
            FROM uploaded_matches
            """
        ).fetchall()
        existing_frames = []
        if base_matches is not None and not base_matches.empty:
            existing_frames.append(base_matches)
        if existing_rows:
            existing_frames.append(pd.DataFrame([dict(row) for row in existing_rows]))
        existing_frame = (
            normalize_matches(pd.concat(existing_frames, ignore_index=True))
            if existing_frames
            else matches.iloc[0:0].copy()
        )
        new_matches, duplicates, conflicts = preview_upload(matches, existing_frame)
        if conflicts:
            raise StorageError(
                f"{len(conflicts)} uploaded fixture(s) conflict with already stored results; "
                "no rows were saved."
            )

        uploaded_at = utc_now()
        connection.execute(
            """
            INSERT INTO uploads
            (upload_id, file_sha256, file_name, source_name, license_reference,
             uploaded_at_utc, inserted_matches)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                upload_id,
                content_sha256,
                Path(file_name).name,
                source_name.strip(),
                license_reference.strip(),
                uploaded_at,
                len(new_matches),
            ),
        )
        for row in new_matches.to_dict("records"):
            identity = (
                str(row["league"]),
                str(row["season"]),
                str(row["match_date"]),
                fixture_key(str(row["home_team"]), str(row["away_team"])),
            )
            found = connection.execute(
                """
                SELECT home_team, away_team, home_goals, away_goals
                FROM uploaded_matches
                WHERE league = ? AND season = ? AND match_date = ? AND fixture_key = ?
                """,
                identity,
            ).fetchone()
            if found:
                duplicates += 1
            else:
                connection.execute(
                    """
                    INSERT INTO uploaded_matches
                    (league, season, match_date, fixture_key, home_team, away_team,
                     home_goals, away_goals, result, file_sha256)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        *identity,
                        str(row["home_team"]),
                        str(row["away_team"]),
                        int(row["home_goals"]),
                        int(row["away_goals"]),
                        str(row["result"]),
                        content_sha256,
                    ),
                )
                inserted += 1
        # Link every uploaded result, including exact duplicates, to a saved prediction.
        for row in matches.to_dict("records"):
            cursor = connection.execute(
                """
                UPDATE predictions
                SET actual_result = ?, evaluation_status = 'unverified_kickoff_time'
                WHERE league = ? AND season = ? AND match_date = ?
                  AND lower(home_team) = lower(?) AND lower(away_team) = lower(?)
                  AND actual_result IS NULL
                """,
                (
                    str(row["result"]),
                    str(row["league"]),
                    str(row["season"]),
                    str(row["match_date"]),
                    str(row["home_team"]),
                    str(row["away_team"]),
                ),
            )
            linked += cursor.rowcount
        connection.commit()
        return {
            "upload_id": upload_id,
            "inserted_matches": inserted,
            "duplicate_matches": duplicates,
            "linked_predictions": linked,
            "already_uploaded": False,
            "uploaded_at_utc": uploaded_at,
        }
    except sqlite3.Error as exc:
        connection.rollback()
        raise StorageError(f"Could not save uploaded results: {exc}") from exc
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def correct_uploaded_conflicts(
    database: str | Path,
    *,
    conflicts: list[dict[str, Any]],
    content_sha256: str,
    file_name: str,
    source_name: str,
    license_reference: str,
    correction_confirmed: bool,
    base_matches: pd.DataFrame,
) -> dict[str, Any]:
    """Apply reviewed corrections only to previously uploaded 2026/27 rows."""
    if not correction_confirmed:
        raise StorageError("Explicitly confirm this separate correction action.")
    if not conflicts:
        raise StorageError("There are no reviewed result conflicts to correct.")
    if not source_name.strip() or not license_reference.strip():
        raise StorageError("Correction source and license/permission reference are required.")
    connection = _connect(database)
    correction_id = str(uuid4())
    uploaded_at = utc_now()
    corrected = 0
    try:
        connection.execute("BEGIN IMMEDIATE")
        if connection.execute(
            "SELECT 1 FROM uploads WHERE file_sha256 = ?", (content_sha256,)
        ).fetchone():
            raise StorageError("This correction file was already applied.")
        correction_rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for conflict in conflicts:
            incoming = conflict["incoming"]
            existing = conflict["existing"]
            key = fixture_key(str(incoming["home_team"]), str(incoming["away_team"]))
            date_key = str(incoming["match_date"])
            if not base_matches.empty and any(
                row["league"] == incoming["league"]
                and str(row["match_date"]) == date_key
                and fixture_key(str(row["home_team"]), str(row["away_team"])) == key
                for row in base_matches.to_dict("records")
            ):
                raise StorageError(
                    "Conflicts against immutable base CSV rows cannot be corrected from the upload page."
                )
            stored = connection.execute(
                """
                SELECT * FROM uploaded_matches
                WHERE league = ? AND season = ? AND match_date = ? AND fixture_key = ?
                """,
                (
                    str(incoming["league"]),
                    str(incoming["season"]),
                    date_key,
                    key,
                ),
            ).fetchone()
            if stored is None:
                raise StorageError(
                    "A conflicting result is not a stored 2026/27 upload; nothing was changed."
                )
            current = {
                "home_team": str(stored["home_team"]),
                "away_team": str(stored["away_team"]),
                "home_goals": int(stored["home_goals"]),
                "away_goals": int(stored["away_goals"]),
                "result": str(stored["result"]),
            }
            if current != {
                "home_team": str(existing["home_team"]),
                "away_team": str(existing["away_team"]),
                "home_goals": int(existing["home_goals"]),
                "away_goals": int(existing["away_goals"]),
                "result": str(existing["result"]),
            }:
                raise StorageError("Stored result changed after preview; reload and review again.")
            correction_rows.append((incoming, current))

        upload_id = str(uuid4())
        connection.execute(
            """
            INSERT INTO uploads
            (upload_id, file_sha256, file_name, source_name, license_reference,
             uploaded_at_utc, inserted_matches, corrected_matches)
            VALUES (?, ?, ?, ?, ?, ?, 0, ?)
            """,
            (
                upload_id,
                content_sha256,
                Path(file_name).name,
                source_name.strip(),
                license_reference.strip(),
                uploaded_at,
                len(correction_rows),
            ),
        )
        for incoming, current in correction_rows:
            key = fixture_key(str(incoming["home_team"]), str(incoming["away_team"]))
            date_key = str(incoming["match_date"])
            new_result = {
                "home_team": str(incoming["home_team"]),
                "away_team": str(incoming["away_team"]),
                "home_goals": int(incoming["home_goals"]),
                "away_goals": int(incoming["away_goals"]),
                "result": str(incoming["result"]),
            }
            connection.execute(
                """
                INSERT INTO match_corrections
                (correction_id, league, season, match_date, fixture_key,
                 previous_result_json, corrected_result_json, corrected_at_utc,
                 source_name, license_reference, file_sha256)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid4()),
                    str(incoming["league"]),
                    str(incoming["season"]),
                    date_key,
                    key,
                    json.dumps(current, ensure_ascii=False, sort_keys=True),
                    json.dumps(new_result, ensure_ascii=False, sort_keys=True),
                    uploaded_at,
                    source_name.strip(),
                    license_reference.strip(),
                    content_sha256,
                ),
            )
            connection.execute(
                """
                UPDATE uploaded_matches
                SET home_team = ?, away_team = ?, home_goals = ?, away_goals = ?,
                    result = ?, file_sha256 = ?
                WHERE league = ? AND season = ? AND match_date = ? AND fixture_key = ?
                """,
                (
                    new_result["home_team"],
                    new_result["away_team"],
                    new_result["home_goals"],
                    new_result["away_goals"],
                    new_result["result"],
                    content_sha256,
                    str(incoming["league"]),
                    str(incoming["season"]),
                    date_key,
                    key,
                ),
            )
            connection.execute(
                """
                UPDATE predictions
                SET actual_result = ?, evaluation_status = 'unverified_kickoff_time'
                WHERE league = ? AND season = ? AND match_date = ?
                  AND lower(home_team) = lower(?) AND lower(away_team) = lower(?)
                """,
                (
                    new_result["result"],
                    str(incoming["league"]),
                    str(incoming["season"]),
                    date_key,
                    new_result["home_team"],
                    new_result["away_team"],
                ),
            )
            corrected += 1
        connection.commit()
        return {
            "upload_id": upload_id,
            "inserted_matches": 0,
            "duplicate_matches": 0,
            "corrected_matches": corrected,
            "linked_predictions": 0,
            "already_uploaded": False,
            "uploaded_at_utc": uploaded_at,
        }
    except sqlite3.Error as exc:
        connection.rollback()
        raise StorageError(f"Could not save result correction: {exc}") from exc
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def save_team_alias(
    database: str | Path,
    *,
    league: str,
    alias: str,
    canonical_team: str,
) -> None:
    alias_key = alias.strip().casefold()
    if not alias_key or not canonical_team.strip():
        raise StorageError("Both the team alias and canonical team name are required.")
    if alias_key == canonical_team.strip().casefold():
        raise StorageError("Alias and canonical team name are already identical.")
    path = Path(database)
    connection = _connect(path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        current = connection.execute(
            "SELECT canonical_team FROM team_aliases WHERE league = ? AND alias_key = ?",
            (league, alias_key),
        ).fetchone()
        if current and str(current["canonical_team"]).casefold() != canonical_team.strip().casefold():
            raise StorageError("This alias already maps to a different canonical team.")
        connection.execute(
            """
            INSERT OR IGNORE INTO team_aliases
            (league, alias_key, alias, canonical_team, created_at_utc)
            VALUES (?, ?, ?, ?, ?)
            """,
            (league, alias_key, alias.strip(), canonical_team.strip(), utc_now()),
        )
        connection.commit()
    except sqlite3.Error as exc:
        connection.rollback()
        raise StorageError(f"Could not save team alias: {exc}") from exc
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def record_prediction(
    database: str | Path,
    *,
    league: str,
    season: str,
    match_date: str,
    home_team: str,
    away_team: str,
    model_version: str,
    model_training_cutoff: str,
    team_state_cutoff: str,
    team_state_hash: str,
    probabilities: dict[str, float],
) -> tuple[str, bool]:
    """Insert a prediction once per fixture/model/team-state snapshot."""
    if abs(sum(probabilities.values()) - 1.0) > 1e-8:
        raise StorageError("Prediction probabilities must sum to 1.")
    prediction_id = str(uuid4())
    connection = _connect(database)
    try:
        connection.execute(
            """
            INSERT OR IGNORE INTO predictions
            (prediction_id, league, season, match_date, home_team, away_team,
             prediction_mode, predicted_at_utc, data_cutoff_at_utc,
             data_cutoff_status, model_version, model_training_cutoff,
             team_state_cutoff, team_state_hash, home_win_probability,
             draw_probability, away_win_probability, actual_result, evaluation_status)
            VALUES (?, ?, ?, ?, ?, ?, 'PRE_MATCH', ?, NULL, 'unknown',
                    ?, ?, ?, ?, ?, ?, ?, NULL, 'awaiting_result')
            """,
            (
                prediction_id,
                league,
                season,
                match_date,
                home_team,
                away_team,
                utc_now(),
                model_version,
                model_training_cutoff,
                team_state_cutoff,
                team_state_hash,
                probabilities["H"],
                probabilities["D"],
                probabilities["A"],
            ),
        )
        inserted = connection.execute("SELECT changes()").fetchone()[0] == 1
        if not inserted:
            row = connection.execute(
                """
                SELECT prediction_id FROM predictions
                WHERE league = ? AND season = ? AND match_date = ?
                  AND lower(home_team) = lower(?) AND lower(away_team) = lower(?)
                  AND prediction_mode = 'PRE_MATCH' AND model_version = ?
                  AND team_state_hash = ?
                """,
                (
                    league,
                    season,
                    match_date,
                    home_team,
                    away_team,
                    model_version,
                    team_state_hash,
                ),
            ).fetchone()
            if row is None:
                raise StorageError("Could not find an existing duplicate prediction.")
            prediction_id = str(row["prediction_id"])
        connection.commit()
    except sqlite3.Error as exc:
        if "connection" in locals():
            connection.rollback()
        raise StorageError(f"Could not save prediction record: {exc}") from exc
    finally:
        if "connection" in locals():
            connection.close()
    return prediction_id, inserted


def read_predictions(database: str | Path) -> pd.DataFrame:
    path = Path(database)
    if not path.is_file():
        return pd.DataFrame()
    connection = _connect(path)
    try:
        rows = connection.execute(
            "SELECT * FROM predictions ORDER BY predicted_at_utc DESC"
        ).fetchall()
    except sqlite3.Error as exc:
        raise StorageError(f"Could not read prediction records: {exc}") from exc
    finally:
        connection.close()
    return pd.DataFrame([dict(row) for row in rows])


def read_uploads(database: str | Path) -> pd.DataFrame:
    path = Path(database)
    if not path.is_file():
        return pd.DataFrame()
    connection = _connect(path)
    try:
        rows = connection.execute(
            "SELECT * FROM uploads ORDER BY uploaded_at_utc DESC"
        ).fetchall()
    except sqlite3.Error as exc:
        raise StorageError(f"Could not read upload records: {exc}") from exc
    finally:
        connection.close()
    return pd.DataFrame([dict(row) for row in rows])


def save_backtest_results(path: str | Path, results: dict[str, Any]) -> None:
    """Atomically replace the generated walk-forward report, never source data."""
    target = Path(path)
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(target)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise StorageError(f"Could not save walk-forward results to {target}: {exc}") from exc


def load_backtest_results(path: str | Path) -> dict[str, Any] | None:
    target = Path(path)
    if not target.is_file():
        return None
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StorageError(f"Could not read walk-forward results at {target}: {exc}") from exc
