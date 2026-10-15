from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from pickfoot.data import normalize_matches
from pickfoot.features import build_features
from pickfoot.model import (
    MIN_TRAIN_MATCHES,
    ModelError,
    load_artifact,
    ordered_probabilities,
    predict_match,
    save_artifact,
    train_model,
)


def synthetic_matches(count=60):
    teams = ["Alpha", "Beta", "Gamma", "Delta"]
    rows = []
    for index in range(count):
        outcome = ("H", "D", "A")[index % 3]
        home_goals, away_goals = {"H": (2, 0), "D": (1, 1), "A": (0, 2)}[outcome]
        rows.append({
            "league": "Premier League",
            "season": "2020/21",
            "match_date": (date(2020, 8, 1) + timedelta(days=index)).isoformat(),
            "home_team": teams[index % len(teams)],
            "away_team": teams[(index + 1) % len(teams)],
            "home_goals": home_goals,
            "away_goals": away_goals,
            "result": outcome,
        })
    return normalize_matches(pd.DataFrame(rows))


def train(matches=None):
    return train_model(
        matches if matches is not None else synthetic_matches(),
        source_name="Synthetic test fixture only",
        license_reference="Unit test; not a real dataset",
    )


def test_chronological_holdout_metrics_and_date_counts():
    artifact = train()
    report = artifact.evaluation
    assert report.train_end < report.validation_start
    assert report.train_matches >= MIN_TRAIN_MATCHES
    assert report.validation_matches > 0
    assert report.train_matches + report.validation_matches == 60
    for metric in (
        report.accuracy, report.log_loss, report.brier_score,
        report.baseline_accuracy, report.baseline_log_loss, report.baseline_brier_score,
    ):
        assert np.isfinite(metric)
    assert report.log_loss >= 0
    assert report.brier_score >= 0


def test_holdout_scaler_is_train_only_and_final_pipeline_refits_all_rows():
    matches = synthetic_matches()
    artifact = train(matches)
    built = build_features(matches)
    train_end = date.fromisoformat(artifact.evaluation.train_end)
    training_rows = matches["match_date"] <= train_end
    expected_mean = built.features.loc[training_rows.to_numpy()].mean(axis=0).to_numpy()
    validation_scaler_mean = artifact.evaluation_pipeline.named_steps["scaler"].mean_
    np.testing.assert_allclose(validation_scaler_mean, expected_mean)
    final_scaler_mean = artifact.pipeline.named_steps["scaler"].mean_
    np.testing.assert_allclose(final_scaler_mean, built.features.mean(axis=0).to_numpy())


def test_validation_outcomes_do_not_change_their_own_features():
    matches = synthetic_matches()
    artifact = train(matches)
    validation_date = date.fromisoformat(artifact.evaluation.validation_start)
    altered = matches.copy()
    row_index = altered.index[altered["match_date"] == validation_date][0]
    original_outcome = altered.loc[row_index, "result"]
    altered.loc[row_index, "home_goals"] = 0 if original_outcome != "A" else 3
    altered.loc[row_index, "away_goals"] = 3 if original_outcome != "A" else 0
    altered.loc[row_index, "result"] = "A" if original_outcome != "A" else "H"
    changed_build = build_features(altered)
    original_build = build_features(matches)
    before_or_same_day = matches["match_date"] <= validation_date
    pd.testing.assert_frame_equal(
        original_build.features.loc[before_or_same_day.to_numpy()].reset_index(drop=True),
        changed_build.features.loc[before_or_same_day.to_numpy()].reset_index(drop=True),
    )


def test_probability_columns_are_mapped_by_class_not_estimator_order():
    class ReorderedEstimator:
        classes_ = np.array(["A", "H", "D"])

        @staticmethod
        def predict_proba(features):
            return np.tile([0.2, 0.5, 0.3], (len(features), 1))

    probabilities = ordered_probabilities(ReorderedEstimator(), pd.DataFrame([{}]))
    np.testing.assert_allclose(probabilities[0], [0.5, 0.3, 0.2])
    assert probabilities.sum() == pytest.approx(1.0)


def test_saved_artifact_loads_and_returns_normalized_future_prediction(tmp_path):
    artifact = train()
    saved_path = save_artifact(artifact, tmp_path)
    loaded = load_artifact(saved_path)
    prediction = predict_match(
        loaded,
        home_team="Alpha",
        away_team="Beta",
        match_date=loaded.state.latest_date + timedelta(days=1),
    )
    assert prediction["mode"] == "PRE_MATCH"
    assert list(prediction["probabilities"]) == ["H", "D", "A"]
    assert sum(prediction["probabilities"].values()) == pytest.approx(1.0)
    assert prediction["home_elo"] == loaded.state.elo["alpha"]


def test_unseen_team_and_same_team_are_rejected():
    artifact = train()
    future = artifact.state.latest_date + timedelta(days=1)
    with pytest.raises(ModelError, match="Unknown team"):
        predict_match(artifact, home_team="Alpha", away_team="Unknown", match_date=future)
    with pytest.raises(ModelError, match="must be different"):
        predict_match(artifact, home_team="Alpha", away_team="alpha", match_date=future)


def test_historical_or_cutoff_date_prediction_is_rejected():
    artifact = train()
    with pytest.raises(ModelError, match="must be after the model data cutoff"):
        predict_match(
            artifact,
            home_team="Alpha",
            away_team="Beta",
            match_date=artifact.state.latest_date,
        )


def test_missing_training_class_and_insufficient_data_are_actionable():
    all_home = synthetic_matches()
    all_home["result"] = "H"
    with pytest.raises(ModelError, match="missing outcome class"):
        train(all_home)
    with pytest.raises(ModelError, match="at least"):
        train(synthetic_matches(20))


def test_unknown_source_collection_time_is_recorded_as_unknown():
    matches = synthetic_matches()
    with pytest.raises(ModelError, match="must include a UTC timezone"):
        train_model(
            matches,
            source_name="reviewed dataset",
            license_reference="private permission record",
            source_collected_at="2026-10-14T06:00:00",
        )
    artifact = train_model(
        matches,
        source_name="synthetic test fixture",
        license_reference="unit-test synthetic fixture",
        source_collected_at="2026-10-14T06:00:00+00:00",
    )
    assert artifact.source_metadata["source_collected_at_utc"] == "2026-10-14T06:00:00Z"
    assert artifact.source_metadata["data_kind"] == "synthetic_test"
    without_timestamp = train(matches)
    assert without_timestamp.source_metadata["source_collected_at_utc"] is None
    assert without_timestamp.source_metadata["collection_time_status"].startswith("not supplied")
