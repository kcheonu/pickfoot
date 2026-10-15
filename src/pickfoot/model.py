"""Chronological model training, evaluation, storage, and match prediction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
import re
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pickfoot.data import OUTCOMES
from pickfoot.features import (
    FEATURE_COLUMNS,
    FeatureBuild,
    FeatureGenerationError,
    LeagueState,
    build_features,
)

MODEL_VERSION = "pickfoot-pre-match-logreg-1"
VALIDATION_FRACTION = 0.2
MIN_TRAIN_MATCHES = 30


class ModelError(ValueError):
    """Actionable model-training or prediction error."""


@dataclass
class Evaluation:
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    train_matches: int
    validation_matches: int
    accuracy: float
    log_loss: float
    brier_score: float
    baseline_accuracy: float
    baseline_log_loss: float
    baseline_brier_score: float


@dataclass
class ModelArtifact:
    league: str
    model_version: str
    pipeline: Pipeline
    evaluation_pipeline: Pipeline
    state: LeagueState
    evaluation: Evaluation
    source_metadata: dict[str, str | None]
    feature_columns: tuple[str, ...]

    @property
    def teams(self) -> list[str]:
        return self.state.teams


def _brier_score(y_true: pd.Series, probabilities: np.ndarray) -> float:
    truth = np.zeros((len(y_true), len(OUTCOMES)), dtype=float)
    class_to_index = {label: index for index, label in enumerate(OUTCOMES)}
    for row, label in enumerate(y_true):
        truth[row, class_to_index[str(label)]] = 1.0
    return float(np.mean(np.sum((truth - probabilities) ** 2, axis=1)))


def _multiclass_log_loss(y_true: pd.Series, probabilities: np.ndarray) -> float:
    class_to_index = {label: index for index, label in enumerate(OUTCOMES)}
    rows = np.arange(len(y_true))
    columns = np.asarray([class_to_index[str(label)] for label in y_true])
    selected = np.clip(probabilities[rows, columns], 1e-15, 1.0)
    return float(-np.mean(np.log(selected)))


def ordered_probabilities(estimator: Any, features: pd.DataFrame) -> np.ndarray:
    """Reindex estimator probabilities to the stable H, D, A order."""
    raw = estimator.predict_proba(features)
    classes = list(estimator.classes_)
    missing = set(OUTCOMES) - set(classes)
    if missing:
        raise ModelError(f"Trained estimator is missing outcome class(es): {', '.join(sorted(missing))}.")
    ordered = np.zeros((len(features), len(OUTCOMES)), dtype=float)
    for source_index, label in enumerate(classes):
        ordered[:, OUTCOMES.index(label)] = raw[:, source_index]
    totals = ordered.sum(axis=1)
    if not np.all(np.isfinite(ordered)) or np.any(totals <= 0):
        raise ModelError("Model returned invalid probability values.")
    return ordered / totals[:, None]


def _chronological_split(matches: pd.DataFrame, fraction: float) -> tuple[set[date], set[date]]:
    dates = sorted(set(matches["match_date"]))
    if len(dates) < 2:
        raise ModelError("At least two distinct match dates are required for chronological validation.")
    split_at = int(len(dates) * (1.0 - fraction))
    split_at = min(max(split_at, 1), len(dates) - 1)
    return set(dates[:split_at]), set(dates[split_at:])


def train_model(
    matches: pd.DataFrame,
    *,
    source_name: str,
    license_reference: str,
    data_kind: Literal["authorized_real", "synthetic_test"] = "synthetic_test",
    source_collected_at: str | None = None,
    source_url: str | None = None,
) -> ModelArtifact:
    """Train one league model using date-separated validation and pre-match features."""
    leagues = matches["league"].unique()
    if len(leagues) != 1:
        raise ModelError("Train one independent model per league; filter matches to a single league.")
    league = str(leagues[0])
    if len(matches) < MIN_TRAIN_MATCHES + 2:
        raise ModelError(
            f"{league} has {len(matches)} matches; at least {MIN_TRAIN_MATCHES + 2} are required "
            f"to leave {MIN_TRAIN_MATCHES} training matches and a validation sample."
        )
    if not source_name.strip() or not license_reference.strip():
        raise ModelError("Source name and license/terms reference are required.")
    if data_kind not in ("authorized_real", "synthetic_test"):
        raise ModelError("data_kind must be 'authorized_real' or 'synthetic_test'.")
    collection_timestamp: str | None = None
    if source_collected_at:
        try:
            parsed_timestamp = datetime.fromisoformat(source_collected_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ModelError("Source collection time must be an ISO-8601 UTC timestamp.") from exc
        if parsed_timestamp.tzinfo is None or parsed_timestamp.utcoffset() != timezone.utc.utcoffset(None):
            raise ModelError("Source collection time must include a UTC timezone, for example 2026-10-14T06:00:00Z.")
        collection_timestamp = parsed_timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    try:
        built: FeatureBuild = build_features(matches)
    except FeatureGenerationError as exc:
        raise ModelError(str(exc)) from exc

    train_dates, validation_dates = _chronological_split(matches, VALIDATION_FRACTION)
    train_mask = matches["match_date"].isin(train_dates)
    validation_mask = matches["match_date"].isin(validation_dates)
    train_x = built.features.loc[train_mask.to_numpy()].reset_index(drop=True)
    validation_x = built.features.loc[validation_mask.to_numpy()].reset_index(drop=True)
    train_y = built.labels.loc[train_mask.to_numpy()].reset_index(drop=True)
    validation_y = built.labels.loc[validation_mask.to_numpy()].reset_index(drop=True)
    if len(train_x) < MIN_TRAIN_MATCHES:
        raise ModelError(
            f"Chronological split leaves only {len(train_x)} training matches; "
            f"at least {MIN_TRAIN_MATCHES} are required."
        )
    train_classes = set(train_y)
    missing_classes = set(OUTCOMES) - train_classes
    if missing_classes:
        raise ModelError(
            f"Training period is missing outcome class(es): {', '.join(sorted(missing_classes))}. "
            "Add authorized historical matches; no class will be fabricated."
        )
    if validation_x.empty:
        raise ModelError("Chronological split produced no validation matches.")

    evaluation_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, random_state=0)),
        ]
    )
    evaluation_pipeline.fit(train_x.loc[:, FEATURE_COLUMNS], train_y)
    probabilities = ordered_probabilities(
        evaluation_pipeline, validation_x.loc[:, FEATURE_COLUMNS]
    )

    train_counts = train_y.value_counts().reindex(OUTCOMES, fill_value=0).to_numpy(dtype=float)
    baseline_probabilities = np.tile(train_counts / train_counts.sum(), (len(validation_y), 1))
    evaluation = Evaluation(
        train_start=min(train_dates).isoformat(),
        train_end=max(train_dates).isoformat(),
        validation_start=min(validation_dates).isoformat(),
        validation_end=max(validation_dates).isoformat(),
        train_matches=len(train_y),
        validation_matches=len(validation_y),
        accuracy=float(accuracy_score(
            validation_y, [OUTCOMES[index] for index in probabilities.argmax(axis=1)]
        )),
        log_loss=_multiclass_log_loss(validation_y, probabilities),
        brier_score=_brier_score(validation_y, probabilities),
        baseline_accuracy=float(accuracy_score(
            validation_y, [OUTCOMES[int(train_counts.argmax())]] * len(validation_y)
        )),
        baseline_log_loss=_multiclass_log_loss(validation_y, baseline_probabilities),
        baseline_brier_score=_brier_score(validation_y, baseline_probabilities),
    )
    final_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, random_state=0)),
        ]
    )
    # Freeze holdout metrics first, then refit the deployable estimator on
    # the complete authorized history for the next-match prediction state.
    final_pipeline.fit(built.features.loc[:, FEATURE_COLUMNS], built.labels)

    return ModelArtifact(
        league=league,
        model_version=MODEL_VERSION,
        pipeline=final_pipeline,
        evaluation_pipeline=evaluation_pipeline,
        state=built.state,
        evaluation=evaluation,
        source_metadata={
            "source_name": source_name.strip(),
            "license_reference": license_reference.strip(),
            "source_url": source_url,
            "data_kind": data_kind,
            "source_collected_at_utc": collection_timestamp,
            "collection_time_status": (
                "recorded"
                if collection_timestamp
                else "not supplied; original acquisition time is unknown"
            ),
            "seasons": ", ".join(sorted(matches["season"].astype(str).unique())),
            "matches_in_source": str(len(matches)),
            "source_coverage_start": min(matches["match_date"]).isoformat(),
            "source_coverage_end": max(matches["match_date"]).isoformat(),
        },
        feature_columns=tuple(FEATURE_COLUMNS),
    )


def save_artifact(artifact: ModelArtifact, directory: str | Path) -> Path:
    """Persist an artifact to a deterministic, per-league joblib path."""
    model_dir = Path(directory)
    model_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", artifact.league.casefold()).strip("-")
    target = model_dir / f"{slug}.joblib"
    joblib.dump(artifact, target)
    return target


def load_artifact(path: str | Path) -> ModelArtifact:
    """Load a trusted local PickFoot joblib artifact; never load untrusted files."""
    target = Path(path)
    if not target.is_file():
        raise ModelError(f"Model artifact not found: {target}. Train a model for this league first.")
    try:
        loaded = joblib.load(target)
    except (OSError, EOFError, ValueError, TypeError) as exc:
        raise ModelError(f"Could not load model artifact {target}: {exc}") from exc
    if not isinstance(loaded, ModelArtifact):
        raise ModelError(f"File is not a PickFoot model artifact: {target}.")
    return loaded


def predict_match(
    artifact: ModelArtifact,
    *,
    home_team: str,
    away_team: str,
    match_date: date,
) -> dict[str, Any]:
    """Predict a future fixture from the model's saved latest team state."""
    return predict_match_with_state(
        artifact,
        state=artifact.state,
        home_team=home_team,
        away_team=away_team,
        match_date=match_date,
    )


def predict_match_with_state(
    artifact: ModelArtifact,
    *,
    state: LeagueState,
    home_team: str,
    away_team: str,
    match_date: date,
) -> dict[str, Any]:
    """Use a refreshed results state without changing the fitted model pipeline."""
    if home_team.strip().casefold() == away_team.strip().casefold():
        raise ModelError("Home and away teams must be different.")
    if state.league != artifact.league:
        raise ModelError(f"Team state for {state.league} cannot be used with {artifact.league}.")
    if state.latest_date is None:
        raise ModelError("Model has no training data cutoff.")
    if match_date <= state.latest_date:
        raise ModelError(
            f"Prediction date {match_date.isoformat()} must be after the model data cutoff "
            f"and current team-state cutoff {state.latest_date.isoformat()}."
        )
    trained_keys = set(artifact.state.team_names)
    unsupported = [
        team for team in (home_team, away_team)
        if team.strip().casefold() not in trained_keys
    ]
    if unsupported:
        raise ModelError(
            "Unknown team(s) for PRE_MATCH model training roster: "
            f"{', '.join(unsupported)}. Their Elo can be displayed, but no probability "
            "is produced until a model supports those teams."
        )
    try:
        features = state.features_for(home_team, away_team)
    except FeatureGenerationError as exc:
        raise ModelError(str(exc)) from exc
    row = pd.DataFrame([features], columns=artifact.feature_columns)
    probabilities = ordered_probabilities(artifact.pipeline, row)[0]
    return {
        "league": artifact.league,
        "match_date": match_date.isoformat(),
        "mode": "PRE_MATCH",
        "model_version": artifact.model_version,
        "model_training_cutoff": artifact.state.latest_date.isoformat(),
        "team_state_cutoff": state.latest_date.isoformat(),
        "team_state_updated": state.latest_date > artifact.state.latest_date,
        "probabilities": dict(zip(OUTCOMES, map(float, probabilities), strict=True)),
        "features": features,
        "home_elo": features["home_elo"],
        "away_elo": features["away_elo"],
        "home_form": {key.removeprefix("home_"): value for key, value in features.items()
                      if key.startswith("home_recent_")},
        "away_form": {key.removeprefix("away_"): value for key, value in features.items()
                      if key.startswith("away_recent_")},
    }


def evaluation_dict(artifact: ModelArtifact) -> dict[str, Any]:
    """Return serializable evaluation metadata for CLI and UI display."""
    return asdict(artifact.evaluation)


def compare_models_on_candidate_window(
    incumbent: ModelArtifact,
    candidate: ModelArtifact,
    matches: pd.DataFrame,
) -> dict[str, Any]:
    """Score both estimators on the candidate's exact later-date holdout."""
    if incumbent.league != candidate.league:
        raise ModelError("Incumbent and candidate models must target the same league.")
    validation_start = date.fromisoformat(candidate.evaluation.validation_start)
    if incumbent.state.latest_date is None or validation_start <= incumbent.state.latest_date:
        raise ModelError(
            "A same-window comparison requires a candidate validation period strictly after "
            "the incumbent model's training cutoff."
        )
    try:
        built = build_features(matches)
    except FeatureGenerationError as exc:
        raise ModelError(str(exc)) from exc
    mask = matches["match_date"].map(
        lambda item: item >= validation_start
        and item <= date.fromisoformat(candidate.evaluation.validation_end)
    )
    if not mask.any():
        raise ModelError("The candidate validation window has no matching fixtures.")
    x = built.features.loc[mask.to_numpy(), list(candidate.feature_columns)].reset_index(drop=True)
    y = built.labels.loc[mask.to_numpy()].reset_index(drop=True)
    old_probabilities = ordered_probabilities(incumbent.pipeline, x)
    new_probabilities = ordered_probabilities(candidate.evaluation_pipeline, x)
    return {
        "validation_start": candidate.evaluation.validation_start,
        "validation_end": candidate.evaluation.validation_end,
        "matches": len(y),
        "incumbent": _evaluation_metrics(y, old_probabilities),
        "candidate": _evaluation_metrics(y, new_probabilities),
        "training_cutoffs": {
            "incumbent": incumbent.state.latest_date.isoformat(),
            "candidate": candidate.evaluation.train_end,
        },
        "note": "Both estimators were scored on the same candidate holdout; "
        "the holdout starts after the incumbent's complete training cutoff.",
    }


def _evaluation_metrics(y: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(
            y, [OUTCOMES[index] for index in probabilities.argmax(axis=1)]
        )),
        "log_loss": _multiclass_log_loss(y, probabilities),
        "brier_score": _brier_score(y, probabilities),
    }
