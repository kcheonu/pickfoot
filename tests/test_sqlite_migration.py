from datetime import date, datetime, timezone
import json

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

from pickfoot.api.migrate_sqlite import SQLiteImportError, import_legacy_sqlite
from pickfoot.api.models import (
    AdminAudit,
    Base,
    Match,
    MatchAudit,
    Prediction,
    Team,
    TeamAlias,
)
from pickfoot.storage import _connect, fixture_key


def _legacy_database(path):
    connection = _connect(path)
    uploaded_at = "2026-08-20T12:30:00Z"
    file_hash = "a" * 64
    connection.execute(
        """
        INSERT INTO uploads
        (upload_id, file_sha256, file_name, source_name, license_reference,
         uploaded_at_utc, inserted_matches, corrected_matches)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "legacy-upload-1", file_hash, "results.csv", "OpenFootball",
            "Public domain statement", uploaded_at, 1, 1,
        ),
    )
    connection.execute(
        """
        INSERT INTO uploaded_matches
        (league, season, match_date, fixture_key, home_team, away_team,
         home_goals, away_goals, result, file_sha256)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "Premier League", "2026-27", "2026-08-15",
            fixture_key("Arsenal FC", "Liverpool FC"),
            "Arsenal FC", "Liverpool FC", 2, 1, "H", file_hash,
        ),
    )
    connection.execute(
        """
        INSERT INTO team_aliases
        (league, alias_key, alias, canonical_team, created_at_utc)
        VALUES (?, ?, ?, ?, ?)
        """,
        ("Premier League", "arsenal", "Arsenal", "Arsenal FC", uploaded_at),
    )
    connection.execute(
        """
        INSERT INTO registered_teams
        (league, team_key, team_name, registration_note, registered_at_utc)
        VALUES (?, ?, ?, ?, ?)
        """,
        ("Premier League", "new fc", "New FC", "Confirmed 2026/27 team", uploaded_at),
    )
    connection.execute(
        """
        INSERT INTO predictions
        (prediction_id, league, season, match_date, home_team, away_team,
         prediction_mode, predicted_at_utc, data_cutoff_at_utc, data_cutoff_status,
         model_version, model_training_cutoff, team_state_cutoff, team_state_hash,
         home_win_probability, draw_probability, away_win_probability, actual_result,
         evaluation_status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "legacy-prediction-1", "Premier League", "2026-27", "2026-08-15",
            "Arsenal FC", "Liverpool FC", "PRE_MATCH", uploaded_at, None, "unknown",
            "legacy-model-v1", "2026-05-24", "2026-05-24", "b" * 64,
            0.55, 0.25, 0.20, "H", "unverified_kickoff_time",
        ),
    )
    connection.execute(
        """
        INSERT INTO match_corrections
        (correction_id, league, season, match_date, fixture_key,
         previous_result_json, corrected_result_json, corrected_at_utc,
         source_name, license_reference, file_sha256)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "legacy-correction-1", "Premier League", "2026-27", "2026-08-15",
            fixture_key("Arsenal FC", "Liverpool FC"),
            json.dumps({"home_goals": 1, "away_goals": 1, "result": "D"}),
            json.dumps({"home_goals": 2, "away_goals": 1, "result": "H"}),
            uploaded_at, "OpenFootball", "Public domain statement", file_hash,
        ),
    )
    connection.commit()
    connection.close()


@pytest.fixture
def postgres_factory(tmp_path):
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'target.sqlite3'}")
    Base.metadata.create_all(engine)
    yield sessionmaker(bind=engine, expire_on_commit=False)
    engine.dispose()


def test_sqlite_import_preserves_records_and_is_idempotent(tmp_path, postgres_factory):
    source = tmp_path / "legacy.sqlite3"
    _legacy_database(source)

    first = import_legacy_sqlite(source, factory=postgres_factory)
    second = import_legacy_sqlite(source, factory=postgres_factory)

    assert first["matches_inserted"] == 1
    assert first["aliases_inserted"] == 1
    assert first["predictions_inserted"] == 1
    assert first["corrections_audited"] == 1
    assert second["matches_inserted"] == 0
    assert second["matches_existing"] == 1
    assert second["aliases_inserted"] == 0
    assert second["predictions_inserted"] == 0
    assert second["predictions_existing"] == 1
    with postgres_factory() as session:
        assert session.scalar(select(func.count()).select_from(Match)) == 1
        assert session.scalar(select(func.count()).select_from(Team)) == 3
        assert session.scalar(select(func.count()).select_from(TeamAlias)) == 1
        assert session.scalar(select(func.count()).select_from(Prediction)) == 1
        assert session.scalar(select(func.count()).select_from(MatchAudit)) == 1
        assert session.scalar(select(func.count()).select_from(AdminAudit)) == 4
        prediction = session.get(Prediction, "legacy-prediction-1")
        assert prediction.actual_result == "H"
        assert prediction.evaluation_status == "unverified_kickoff_time"
        assert prediction.data_cutoff_at is None
        assert session.scalar(
            select(Team).where(Team.team_key == "new fc")
        ).manually_registered is True


def test_sqlite_import_conflict_rolls_back_everything(tmp_path, postgres_factory):
    source = tmp_path / "legacy.sqlite3"
    _legacy_database(source)
    with postgres_factory.begin() as session:
        session.add(Match(
            id="existing-conflicting-match",
            league="Premier League",
            season="2026-27",
            round_name="Round 1",
            match_date=date(2026, 8, 15),
            home_team="Arsenal FC",
            away_team="Liverpool FC",
            home_team_key="arsenal fc",
            away_team_key="liverpool fc",
            status="completed",
            home_goals=0,
            away_goals=0,
            result="D",
            version=1,
            updated_at=datetime.now(timezone.utc),
        ))

    with pytest.raises(SQLiteImportError, match="conflicts"):
        import_legacy_sqlite(source, factory=postgres_factory)

    with postgres_factory() as session:
        assert session.scalar(select(func.count()).select_from(Match)) == 1
        assert session.scalar(select(func.count()).select_from(AdminAudit)) == 0
        assert session.scalar(select(func.count()).select_from(Prediction)) == 0
