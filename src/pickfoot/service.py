"""Application services for combining immutable training data and result updates."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from pickfoot.data import CSVValidationError, load_matches, normalize_matches
from pickfoot.features import FeatureBuild, build_features
from pickfoot.model import ModelArtifact, ModelError, predict_match_with_state
from pickfoot.storage import normalize_uploaded_bytes, preview_upload, read_uploaded_matches


def load_base_matches(path: str | Path = "data/raw/matches.csv") -> pd.DataFrame:
    return load_matches(path)


def merged_matches(
    base_matches: pd.DataFrame,
    uploaded_matches: pd.DataFrame,
    league: str | None = None,
) -> pd.DataFrame:
    frames = [frame for frame in (base_matches, uploaded_matches) if not frame.empty]
    if not frames:
        raise CSVValidationError("No completed match results are available.")
    combined = normalize_matches(pd.concat(frames, ignore_index=True))
    if league:
        combined = combined.loc[combined["league"] == league].copy()
        if combined.empty:
            raise CSVValidationError(f"No match results are available for {league}.")
    return combined.reset_index(drop=True)


def current_league_state(
    artifact: ModelArtifact,
    *,
    base_matches: pd.DataFrame,
    uploaded_matches: pd.DataFrame,
    registered_teams: list[str] | None = None,
) -> tuple[pd.DataFrame, FeatureBuild]:
    matches = merged_matches(base_matches, uploaded_matches, artifact.league)
    if matches.empty:
        raise ModelError(f"No match results are available for {artifact.league}.")
    built = build_features(matches)
    for team in registered_teams or []:
        built.state.register_team(team)
    return matches, built


def prepare_upload(
    content: bytes,
    *,
    database: str | Path,
    base_matches: pd.DataFrame,
    uploaded_matches: pd.DataFrame,
) -> tuple[pd.DataFrame, str, pd.DataFrame, int, list[dict[str, Any]]]:
    incoming, content_sha256 = normalize_uploaded_bytes(content, database=database)
    invalid_dates = incoming["match_date"].map(
        lambda value: value < date(2026, 7, 1) or value > date(2027, 6, 30)
    )
    if invalid_dates.any():
        raise CSVValidationError("2026-27 match dates must fall between 2026-07-01 and 2027-06-30.")
    future_matches = incoming["match_date"].map(lambda value: value > date.today())
    if future_matches.any():
        raise CSVValidationError("Only completed results can be uploaded; future match dates are not accepted.")

    league_cutoffs = base_matches.groupby("league")["match_date"].max().to_dict()
    for _, row in incoming.iterrows():
        cutoff = league_cutoffs.get(row["league"])
        if cutoff and row["match_date"] <= cutoff:
            raise CSVValidationError(
                f"{row['league']} upload date {row['match_date']} must be later than "
                f"the base-data cutoff {cutoff}."
            )
    existing_frames = [
        frame for frame in (base_matches, uploaded_matches) if not frame.empty
    ]
    existing = normalize_matches(pd.concat(existing_frames, ignore_index=True)) \
        if existing_frames else incoming.iloc[0:0].copy()
    new_matches, duplicate_count, conflicts = preview_upload(incoming, existing)
    return incoming, content_sha256, new_matches, duplicate_count, conflicts


def predict_with_latest_state(
    artifact: ModelArtifact,
    *,
    matches: pd.DataFrame,
    feature_build: FeatureBuild,
    home_team: str,
    away_team: str,
    match_date: date,
) -> dict[str, Any]:
    return predict_match_with_state(
        artifact,
        state=feature_build.state,
        home_team=home_team,
        away_team=away_team,
        match_date=match_date,
    )


def state_hash(features: dict[str, float]) -> str:
    """Stable fingerprint of the actual model input snapshot."""
    import hashlib
    import json

    payload = json.dumps(features, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
