from datetime import timedelta
from hashlib import sha256

import pandas as pd
import pytest

from pickfoot.analytics import league_standings, team_match_history
from pickfoot.data import CSVValidationError, normalize_matches
from pickfoot.features import ELO_K_FACTOR, HOME_ADVANTAGE_ELO, INITIAL_ELO, build_features
from pickfoot.model import ModelError, predict_match, predict_match_with_state
from pickfoot.service import current_league_state, prepare_upload, state_hash
from pickfoot.storage import (
    StorageError,
    commit_upload,
    correct_uploaded_conflicts,
    normalize_uploaded_bytes,
    read_predictions,
    read_registered_teams,
    read_team_aliases,
    read_uploaded_matches,
    preview_upload,
    register_team,
    record_prediction,
    save_team_alias,
)
from tests.test_model import synthetic_matches, train


def _upload_bytes(*, home="Alpha", away="Beta", home_goals=2, away_goals=0):
    return pd.DataFrame([{
        "league": "PL",
        "season": "2026/27",
        "match_date": "2026-08-15",
        "home_team": home,
        "away_team": away,
        "home_goals": str(home_goals),
        "away_goals": str(away_goals),
        "result": "H" if home_goals > away_goals else "D" if home_goals == away_goals else "A",
    }]).to_csv(index=False).encode("utf-8-sig")


def test_elo_match_ledger_contains_expected_and_real_rating_changes():
    frame = normalize_matches(pd.DataFrame([{
        "league": "Premier League",
        "season": "2025/26",
        "match_date": "2025-08-01",
        "home_team": "Alpha",
        "away_team": "Beta",
        "home_goals": 0,
        "away_goals": 2,
        "result": "A",
    }]))
    event = build_features(frame).elo_history.iloc[0]
    expected_home = 1 / (1 + 10 ** (-HOME_ADVANTAGE_ELO / 400))
    assert event["home_elo_before"] == INITIAL_ELO
    assert event["away_elo_before"] == INITIAL_ELO
    assert event["expected_home_score"] == pytest.approx(expected_home)
    assert event["actual_home_score"] == 0
    assert event["home_elo_change"] == pytest.approx(ELO_K_FACTOR * (0 - expected_home))
    assert event["away_elo_change"] == pytest.approx(-event["home_elo_change"])
    assert event["home_elo_after"] == pytest.approx(INITIAL_ELO + event["home_elo_change"])
    assert event["result"] == "A"


def test_standings_sort_and_share_ties_and_team_history_is_chronological():
    matches = synthetic_matches()
    built = build_features(matches)
    for key in built.state.elo:
        built.state.elo[key] = 1500.0
    table = league_standings(matches, built)
    assert table["team"].tolist() == sorted(table["team"].tolist())
    assert set(table["rank"]) == {1}
    assert table["last_match_date"].notna().all()
    history = team_match_history(built.elo_history, "Alpha")
    assert len(history) > 5
    assert history["match_date"].is_monotonic_increasing
    assert set(history["result"]) <= {"H", "D", "A"}


def test_upload_preview_does_not_write_and_utf8_bom_hash_is_source_hash(tmp_path):
    database = tmp_path / "runtime" / "pickfoot.sqlite3"
    raw = _upload_bytes()
    base = synthetic_matches()
    uploaded = read_uploaded_matches(database)
    incoming, digest, new, duplicates, conflicts = prepare_upload(
        raw,
        database=database,
        base_matches=base,
        uploaded_matches=uploaded,
    )
    assert digest == sha256(raw).hexdigest()
    assert len(incoming) == len(new) == 1
    assert duplicates == 0
    assert conflicts == []
    assert not database.exists()


def test_upload_rejects_wrong_season_and_malformed_utf8(tmp_path):
    database = tmp_path / "pickfoot.sqlite3"
    malformed = b"\xff\xfe"
    with pytest.raises(CSVValidationError, match="UTF-8"):
        normalize_uploaded_bytes(malformed, database=database)
    wrong_season = pd.DataFrame([{
        "league": "PL", "season": "2025/26", "match_date": "2026-08-15",
        "home_team": "Alpha", "away_team": "Beta", "home_goals": 1,
        "away_goals": 0, "result": "H",
    }]).to_csv(index=False).encode()
    with pytest.raises(CSVValidationError, match="only the 2026-27"):
        normalize_uploaded_bytes(wrong_season, database=database)


def test_alias_maps_league_code_and_team_name_explicitly(tmp_path):
    database = tmp_path / "pickfoot.sqlite3"
    save_team_alias(
        database,
        league="Premier League",
        alias="Arsenal",
        canonical_team="Arsenal FC",
    )
    raw = _upload_bytes(home="Arsenal", away="Liverpool FC")
    normalized, _ = normalize_uploaded_bytes(raw, database=database)
    assert normalized.loc[0, "league"] == "Premier League"
    assert normalized.loc[0, "home_team"] == "Arsenal FC"
    assert read_team_aliases(database)["Premier League:arsenal"] == "Arsenal FC"
    with pytest.raises(StorageError, match="different canonical"):
        save_team_alias(
            database,
            league="Premier League",
            alias="Arsenal",
            canonical_team="Different Team",
        )


def test_registered_new_team_gets_initial_elo_without_fabricated_history(tmp_path):
    database = tmp_path / "runtime.sqlite3"
    assert register_team(
        database,
        league="Premier League",
        team_name="Promoted FC",
        registration_note="User confirmed 2026/27 promotion",
    )
    assert not register_team(
        database,
        league="Premier League",
        team_name="Promoted FC",
        registration_note="repeat",
    )
    registered = read_registered_teams(database)
    assert registered["Premier League"] == ["Promoted FC"]
    base = synthetic_matches()
    artifact = train(base)
    built = build_features(base)
    built.state.register_team(registered["Premier League"][0])
    standings = league_standings(base, built)
    team_row = standings.loc[standings["team"] == "Promoted FC"].iloc[0]
    assert team_row["elo"] == INITIAL_ELO
    assert team_row["season_matches"] == 0
    assert team_row["recent_matches"] == 0
    assert artifact.state.team_names.get("promoted fc") is None


def test_upload_commits_atomically_idempotently_and_rejects_conflicts(tmp_path):
    database = tmp_path / "runtime" / "pickfoot.sqlite3"
    raw = _upload_bytes()
    base = synthetic_matches()
    incoming, digest, _, _, _ = prepare_upload(
        raw,
        database=database,
        base_matches=base,
        uploaded_matches=read_uploaded_matches(database),
    )
    with pytest.raises(StorageError, match="Confirm"):
        commit_upload(
            database,
            matches=incoming,
            content_sha256=digest,
            file_name="round.csv",
            source_name="Local source",
            license_reference="confirmed terms",
            rights_confirmed=False,
        )
    assert not database.exists()
    first = commit_upload(
        database,
        matches=incoming,
        content_sha256=digest,
        file_name="round.csv",
        source_name="Local source",
        license_reference="confirmed terms",
        rights_confirmed=True,
        base_matches=base,
    )
    assert first["inserted_matches"] == 1
    second = commit_upload(
        database,
        matches=incoming,
        content_sha256=digest,
        file_name="round.csv",
        source_name="Local source",
        license_reference="confirmed terms",
        rights_confirmed=True,
        base_matches=base,
    )
    assert second["already_uploaded"]
    assert len(read_uploaded_matches(database)) == 1


def test_stored_2026_27_conflict_requires_separate_review_and_audits_correction(tmp_path):
    database = tmp_path / "runtime.sqlite3"
    base = synthetic_matches()
    original_bytes = _upload_bytes(home_goals=2, away_goals=0)
    original, original_hash = normalize_uploaded_bytes(original_bytes, database=database)
    commit_upload(
        database,
        matches=original,
        content_sha256=original_hash,
        file_name="first.csv",
        source_name="first source",
        license_reference="first permission",
        rights_confirmed=True,
        base_matches=base,
    )
    corrected_bytes = _upload_bytes(home_goals=1, away_goals=0)
    incoming, corrected_hash = normalize_uploaded_bytes(corrected_bytes, database=database)
    _, _, conflicts = preview_upload(incoming, read_uploaded_matches(database))
    assert len(conflicts) == 1
    arguments = {
        "database": database,
        "conflicts": conflicts,
        "content_sha256": corrected_hash,
        "file_name": "corrected.csv",
        "source_name": "correction source",
        "license_reference": "correction permission",
        "base_matches": base,
    }
    with pytest.raises(StorageError, match="separate correction"):
        correct_uploaded_conflicts(**arguments, correction_confirmed=False)
    assert int(read_uploaded_matches(database).iloc[0]["home_goals"]) == 2
    result = correct_uploaded_conflicts(**arguments, correction_confirmed=True)
    corrected = read_uploaded_matches(database)
    assert result["corrected_matches"] == 1
    assert int(corrected.iloc[0]["home_goals"]) == 1
    import sqlite3

    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM match_corrections").fetchone()[0] == 1
        assert connection.execute(
            "SELECT corrected_matches FROM uploads WHERE file_sha256 = ?",
            (corrected_hash,),
        ).fetchone()[0] == 1

    conflict_bytes = _upload_bytes(home_goals=0, away_goals=2)
    conflicting, conflict_hash = normalize_uploaded_bytes(conflict_bytes, database=database)
    with pytest.raises(StorageError, match="conflict"):
        commit_upload(
            database,
            matches=conflicting,
            content_sha256=conflict_hash,
            file_name="changed.csv",
            source_name="Local source",
            license_reference="confirmed terms",
            rights_confirmed=True,
            base_matches=base,
        )
    assert len(read_uploaded_matches(database)) == 1


def test_saved_prediction_is_idempotent_and_result_link_is_unverified(tmp_path):
    database = tmp_path / "runtime.sqlite3"
    probabilities = {"H": 0.5, "D": 0.3, "A": 0.2}
    arguments = {
        "league": "Premier League",
        "season": "2026-27",
        "match_date": "2026-08-15",
        "home_team": "Alpha",
        "away_team": "Beta",
        "model_version": "test-model",
        "model_training_cutoff": "2026-05-24",
        "team_state_cutoff": "2026-05-24",
        "team_state_hash": "same-state",
        "probabilities": probabilities,
    }
    prediction_id, inserted = record_prediction(database, **arguments)
    repeated_id, repeated_inserted = record_prediction(database, **arguments)
    assert inserted and not repeated_inserted and repeated_id == prediction_id
    upload = normalize_matches(pd.DataFrame([{
        "league": "Premier League", "season": "2026-27", "match_date": "2026-08-15",
        "home_team": "Alpha", "away_team": "Beta", "home_goals": 2,
        "away_goals": 0, "result": "H",
    }]))
    commit_upload(
        database,
        matches=upload,
        content_sha256="a" * 64,
        file_name="results.csv",
        source_name="test data",
        license_reference="unit test",
        rights_confirmed=True,
    )
    record = read_predictions(database).iloc[0]
    assert record["actual_result"] == "H"
    assert record["evaluation_status"] == "unverified_kickoff_time"
    assert not record["data_cutoff_at_utc"]


def test_updated_state_changes_features_without_mutating_model_and_new_team_is_rejected():
    artifact = train()
    matches = synthetic_matches()
    original_prediction = predict_match(
        artifact,
        home_team="Alpha",
        away_team="Beta",
        match_date=artifact.state.latest_date + timedelta(days=1),
    )
    extra = normalize_matches(pd.DataFrame([{
        "league": "Premier League", "season": "2026/27",
        "match_date": (artifact.state.latest_date + timedelta(days=1)).isoformat(),
        "home_team": "Alpha", "away_team": "Beta",
        "home_goals": 6, "away_goals": 0, "result": "H",
    }]))
    combined = normalize_matches(pd.concat([matches, extra], ignore_index=True))
    _, updated = current_league_state(
        artifact, base_matches=matches, uploaded_matches=extra
    )
    assert artifact.pipeline is not None
    updated_prediction = predict_match_with_state(
        artifact,
        state=updated.state,
        home_team="Alpha",
        away_team="Beta",
        match_date=updated.state.latest_date + timedelta(days=1),
    )
    assert updated.state.latest_date > artifact.state.latest_date
    assert updated_prediction["probabilities"] != original_prediction["probabilities"]
    assert updated_prediction["model_training_cutoff"] == artifact.state.latest_date.isoformat()
    assert updated_prediction["team_state_updated"]
    assert state_hash(updated_prediction["features"])
    assert combined["season"].str.startswith("2020").any()

    new_team_row = normalize_matches(pd.DataFrame([{
        "league": "Premier League", "season": "2026/27",
        "match_date": (artifact.state.latest_date + timedelta(days=1)).isoformat(),
        "home_team": "Promoted FC", "away_team": "Alpha",
        "home_goals": 0, "away_goals": 1, "result": "A",
    }]))
    _, new_team_state = current_league_state(
        artifact, base_matches=matches, uploaded_matches=new_team_row
    )
    assert new_team_state.state.elo["promoted fc"] != INITIAL_ELO
    with pytest.raises(ModelError, match="Unknown team"):
        predict_match_with_state(
            artifact,
            state=new_team_state.state,
            home_team="Promoted FC",
            away_team="Alpha",
            match_date=new_team_state.state.latest_date + timedelta(days=1),
        )
