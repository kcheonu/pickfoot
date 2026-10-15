import asyncio
import sys
from datetime import date

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from pickfoot.api import app as app_module
from pickfoot.api import auth as auth_module
from pickfoot.api.auth import hash_password
from pickfoot.api.config import Settings, get_settings
from pickfoot.api.db import get_db
from pickfoot.api.models import ActiveModel, Base, Match, MatchAudit, ModelVersion
from tests.test_model import synthetic_matches


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    get_settings.cache_clear()
    auth_module._request_times.clear()
    original_policy = asyncio.get_event_loop_policy()
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    base_csv = tmp_path / "base.csv"
    synthetic_matches().to_csv(base_csv, index=False)
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    config = Settings(
        database_url="sqlite+pysqlite://",
        web_origin="http://localhost:3000",
        admin_username="operator",
        admin_password_hash=hash_password("test-only-password-2026"),
        session_secret="test-only-session-secret-with-more-than-32-bytes",
        cookie_secure=False,
        base_matches_csv=base_csv,
        model_storage_dir=model_dir,
        data_storage_dir=tmp_path / "persistent",
    )
    monkeypatch.setattr(app_module, "settings", config)
    monkeypatch.setattr("pickfoot.api.services.get_settings", lambda: config)
    monkeypatch.setattr("pickfoot.api.app.get_settings", lambda: config)
    monkeypatch.setattr(auth_module, "get_settings", lambda: config)
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = lambda: Session(engine, expire_on_commit=False)

    def override_get_db():
        session = factory()
        try:
            yield session
        finally:
            session.close()

    app_module.app.dependency_overrides[get_db] = override_get_db
    try:
        with TestClient(app_module.app) as client:
            yield client
    finally:
        app_module.app.dependency_overrides.clear()
        engine.dispose()
        asyncio.set_event_loop_policy(original_policy)
        get_settings.cache_clear()


def _admin_login(client: TestClient) -> None:
    client.headers.update({"Origin": "http://localhost:3000"})
    response = client.post(
        "/api/v1/auth/login",
        headers={"Origin": "http://localhost:3000"},
        json={"username": "operator", "password": "test-only-password-2026"},
    )
    assert response.status_code == 200
    assert response.json()["authenticated"] == "true"


def test_public_health_and_league_routes(api_client):
    assert api_client.get("/api/v1/health/live").json() == {"status": "ok"}
    assert api_client.get("/api/v1/leagues").json()["leagues"] == [
        "Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"
    ]


def test_login_requires_valid_origin_and_credentials(api_client):
    wrong_origin = api_client.post(
        "/api/v1/auth/login",
        headers={"Origin": "https://untrusted.example"},
        json={"username": "operator", "password": "test-only-password-2026"},
    )
    assert wrong_origin.status_code == 403
    wrong_password = api_client.post(
        "/api/v1/auth/login",
        headers={"Origin": "http://localhost:3000"},
        json={"username": "operator", "password": "wrong-password"},
    )
    assert wrong_password.status_code == 401
    _admin_login(api_client)
    assert api_client.get("/api/v1/auth/me").json()["authenticated"] is True
    logout = api_client.post(
        "/api/v1/auth/logout", headers={"Origin": "http://localhost:3000"}
    )
    assert logout.status_code == 200
    assert api_client.get("/api/v1/auth/me").json()["authenticated"] is False


def test_result_workflow_is_protected_versioned_and_audited(api_client):
    unauthenticated = api_client.post(
        "/api/v1/admin/matches",
        headers={"Origin": "http://localhost:3000"},
        json={
            "league": "Premier League",
            "season": "2026-27",
            "round_name": "Matchday 1",
            "match_date": "2026-08-15",
            "home_team": "Alpha",
            "away_team": "Beta",
        },
    )
    assert unauthenticated.status_code == 401
    _admin_login(api_client)

    fixture = {
        "league": "Premier League",
        "season": "2026-27",
        "round_name": "Matchday 1",
        "match_date": "2026-08-15",
        "home_team": "Alpha",
        "away_team": "Beta",
    }
    created = api_client.post("/api/v1/admin/matches", json=fixture)
    assert created.status_code == 201
    match = created.json()
    duplicate = api_client.post("/api/v1/admin/matches", json=fixture)
    assert duplicate.status_code == 409
    assert api_client.get(
        "/api/v1/rounds",
        params={"league": "Premier League", "season": "2026-27"},
    ).json()["rounds"] == ["Matchday 1"]

    result = {
        "home_goals": 2,
        "away_goals": 0,
        "expected_version": match["version"],
        "source_name": "administrator supplied result",
        "license_reference": "local-use permission confirmed by administrator",
        "rights_confirmed": True,
        "reason": "Final score checked against the permitted source.",
    }
    api_client.cookies.clear()
    missing_auth = api_client.put(
        f"/api/v1/admin/matches/{match['id']}/result", json=result
    )
    assert missing_auth.status_code == 401
    _admin_login(api_client)
    confirmed = api_client.put(
        f"/api/v1/admin/matches/{match['id']}/result", json=result
    )
    assert confirmed.status_code == 200
    assert confirmed.json()["match"]["result"] == "H"
    assert confirmed.json()["elo"]["home_change"] > 0

    repeated = api_client.put(
        f"/api/v1/admin/matches/{match['id']}/result",
        json={**result, "expected_version": 2},
    )
    assert repeated.json()["idempotent"] is True
    correction = {
        **result,
        "home_goals": 0,
        "away_goals": 1,
        "expected_version": 2,
    }
    conflict = api_client.put(
        f"/api/v1/admin/matches/{match['id']}/result", json=correction
    )
    assert conflict.status_code == 409
    corrected = api_client.put(
        f"/api/v1/admin/matches/{match['id']}/result",
        json={**correction, "correction_confirmed": True, "reason": "Official correction verified."},
    )
    assert corrected.status_code == 200
    assert corrected.json()["match"]["result"] == "A"
    audit = api_client.get(f"/api/v1/admin/matches/{match['id']}/audit")
    assert audit.status_code == 200
    assert [row["action"] for row in audit.json()["history"]] == [
        "fixture_created", "result_confirmed", "result_corrected"
    ]


def test_training_status_and_model_listing_are_safe_to_read(api_client):
    _admin_login(api_client)
    assert api_client.get("/api/v1/models").json() == {"models": []}
    assert api_client.get("/api/v1/training-jobs").status_code == 200


def test_model_listing_serializes_registered_versions(api_client):
    db_generator = app_module.app.dependency_overrides[get_db]()
    session = next(db_generator)
    version = ModelVersion(
        league="Premier League",
        version="test-candidate-v1",
        artifact_path="unused-test-artifact.joblib",
        source_data_hash="0" * 64,
        evaluation={"accuracy": 0.5},
        comparison=None,
        training_cutoff=date(2026, 5, 24),
        candidate_status="candidate",
    )
    session.add(version)
    session.flush()
    session.add(ActiveModel(
        league="Premier League",
        model_version_id=version.id,
    ))
    session.commit()
    try:
        next(db_generator)
    except StopIteration:
        pass

    response = api_client.get("/api/v1/models")

    assert response.status_code == 200
    models = response.json()["models"]
    assert len(models) == 1
    assert models[0]["version"] == "test-candidate-v1"
    assert models[0]["candidate_status"] == "candidate"
    assert models[0]["active"] is True


def test_training_and_backtest_queue_audits_persist_job_ids(api_client):
    _admin_login(api_client)
    content = synthetic_matches().to_csv(index=False).encode("utf-8")
    dataset = api_client.post(
        "/api/v1/admin/datasets",
        params={
            "source_name": "Synthetic test fixture",
            "license_reference": "Test-only synthetic data",
            "rights_confirmed": "true",
        },
        files={"file": ("matches.csv", content, "text/csv")},
    )
    assert dataset.status_code == 201

    training = api_client.post(
        "/api/v1/admin/training-jobs",
        json={"league": "Premier League"},
    )
    assert training.status_code == 202
    training_job = api_client.get(
        f"/api/v1/admin/training-jobs/{training.json()['job_id']}"
    )
    assert training_job.status_code == 200

    backtest = api_client.post(
        "/api/v1/admin/backtests",
        params={"league": "Premier League"},
    )
    assert backtest.status_code == 202
    backtest_job = api_client.get(
        f"/api/v1/admin/training-jobs/{backtest.json()['job_id']}"
    )
    assert backtest_job.status_code == 200
    assert backtest_job.json()["job_type"] == "backtest"


def test_admin_team_registration_is_real_idempotence_safe_and_visible(api_client):
    _admin_login(api_client)
    payload = {
        "league": "Premier League",
        "team_name": "  New City  ",
        "registration_note": "Confirmed as a 2026-27 league participant.",
    }
    registered = api_client.post("/api/v1/admin/teams", json=payload)
    assert registered.status_code == 201
    assert registered.json()["team_name"] == "New City"
    assert registered.json()["elo_policy"].startswith("1500 initial rating")
    duplicate = api_client.post(
        "/api/v1/admin/teams",
        json={**payload, "team_name": "New City"},
    )
    assert duplicate.status_code == 409
    standings = api_client.get("/api/v1/elo/Premier%20League")
    assert standings.status_code == 200
    new_team = next(row for row in standings.json()["rows"] if row["team"] == "New City")
    assert new_team["elo"] == 1500
    assert new_team["season_matches"] == 0
