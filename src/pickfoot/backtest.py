"""Expanding-window, match-date-safe historical PRE_MATCH evaluation."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pickfoot.data import OUTCOMES
from pickfoot.features import FEATURE_COLUMNS, build_features
from pickfoot.model import (
    MIN_TRAIN_MATCHES,
    _brier_score,
    _multiclass_log_loss,
    ordered_probabilities,
)
from pickfoot.data import REQUIRED_COLUMNS


def matches_fingerprint(matches: pd.DataFrame) -> str:
    """Stable digest used to prevent showing a report for a different dataset."""
    frame = matches.loc[:, REQUIRED_COLUMNS].copy()
    frame["match_date"] = frame["match_date"].map(lambda value: value.isoformat())
    payload = json.dumps(frame.to_dict("records"), ensure_ascii=False, sort_keys=True)
    return sha256(payload.encode("utf-8")).hexdigest()


def expanding_window_backtest(matches: pd.DataFrame) -> dict[str, Any]:
    """Refit only on strictly earlier dates; score all fixtures on each date together."""
    reports: dict[str, Any] = {}
    for league, league_matches in matches.groupby("league", sort=True):
        ordered = league_matches.sort_values(
            ["match_date", "home_team", "away_team"], kind="stable"
        ).reset_index(drop=True)
        built = build_features(ordered)
        dates = pd.Series(ordered["match_date"].tolist())
        predictions: list[dict[str, Any]] = []
        for match_date in sorted(ordered["match_date"].unique()):
            train_mask = dates < match_date
            test_mask = dates == match_date
            train_y = built.labels.loc[train_mask.to_numpy()].reset_index(drop=True)
            if len(train_y) < MIN_TRAIN_MATCHES or set(train_y) != set(OUTCOMES):
                continue
            train_x = built.features.loc[train_mask.to_numpy(), FEATURE_COLUMNS].reset_index(drop=True)
            test_x = built.features.loc[test_mask.to_numpy(), FEATURE_COLUMNS].reset_index(drop=True)
            test_y = built.labels.loc[test_mask.to_numpy()].reset_index(drop=True)

            estimator = Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=2000, random_state=0)),
            ])
            estimator.fit(train_x, train_y)
            probabilities = ordered_probabilities(estimator, test_x)
            counts = train_y.value_counts().reindex(OUTCOMES, fill_value=0).to_numpy(dtype=float)
            baseline = counts / counts.sum()
            daily = ordered.loc[test_mask.to_numpy()]
            for row_index, (_, match) in enumerate(daily.iterrows()):
                predictions.append({
                    "league": str(league),
                    "season": str(match["season"]),
                    "match_date": match_date.isoformat(),
                    "home_team": str(match["home_team"]),
                    "away_team": str(match["away_team"]),
                    "home_goals": int(match["home_goals"]),
                    "away_goals": int(match["away_goals"]),
                    "actual_result": str(test_y.iloc[row_index]),
                    "probabilities": {
                        outcome: float(probabilities[row_index, index])
                        for index, outcome in enumerate(OUTCOMES)
                    },
                    "baseline_probabilities": {
                        outcome: float(baseline[index])
                        for index, outcome in enumerate(OUTCOMES)
                    },
                    "training_cutoff": max(dates.loc[train_mask]).isoformat(),
                })

        if not predictions:
            reports[str(league)] = {
                "matches": 0,
                "message": (
                    f"Need at least {MIN_TRAIN_MATCHES} earlier matches and all H/D/A "
                    "classes before producing a walk-forward prediction."
                ),
                "predictions": [],
            }
            continue

        truth = pd.Series([item["actual_result"] for item in predictions])
        probability_array = pd.DataFrame(
            [item["probabilities"] for item in predictions], columns=OUTCOMES
        ).to_numpy(dtype=float)
        baseline_array = pd.DataFrame(
            [item["baseline_probabilities"] for item in predictions], columns=OUTCOMES
        ).to_numpy(dtype=float)
        predicted_labels = [
            OUTCOMES[int(index)] for index in probability_array.argmax(axis=1)
        ]
        baseline_labels = [
            OUTCOMES[int(index)] for index in baseline_array.argmax(axis=1)
        ]
        reports[str(league)] = {
            "matches": len(predictions),
            "evaluation_start": min(item["match_date"] for item in predictions),
            "evaluation_end": max(item["match_date"] for item in predictions),
            "accuracy": float((truth.to_numpy() == predicted_labels).mean()),
            "log_loss": _multiclass_log_loss(truth, probability_array),
            "brier_score": _brier_score(truth, probability_array),
            "baseline_accuracy": float((truth.to_numpy() == baseline_labels).mean()),
            "baseline_log_loss": _multiclass_log_loss(truth, baseline_array),
            "baseline_brier_score": _brier_score(truth, baseline_array),
            "predictions": predictions,
        }
    return {
        "method": "expanding-window by match date; fit on strictly earlier fixtures only",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_data_hash": matches_fingerprint(matches),
        "source_match_count": int(len(matches)),
        "same_date_policy": "all fixtures use start-of-day state and earlier dates for model fitting",
        "leakage_note": "not an actual pre-match prediction record; historical time-safe reconstruction",
        "leagues": reports,
    }
