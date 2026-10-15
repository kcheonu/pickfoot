"""CSV loading, normalization, and validation for completed matches."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

LEAGUES = {
    "pl": "Premier League",
    "premier league": "Premier League",
    "epl": "Premier League",
    "england premier league": "Premier League",
    "pd": "La Liga",
    "la liga": "La Liga",
    "laliga": "La Liga",
    "spanish la liga": "La Liga",
    "bl1": "Bundesliga",
    "bundesliga": "Bundesliga",
    "german bundesliga": "Bundesliga",
    "sa": "Serie A",
    "serie a": "Serie A",
    "italian serie a": "Serie A",
    "fl1": "Ligue 1",
    "ligue 1": "Ligue 1",
    "ligue1": "Ligue 1",
    "french ligue 1": "Ligue 1",
}
REQUIRED_COLUMNS = (
    "league",
    "season",
    "match_date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "result",
)
OUTCOMES = ("H", "D", "A")


class CSVValidationError(ValueError):
    """User-readable validation failure for a match-results CSV."""


def _clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().split())


def _normalize_season(value: object, row_number: int) -> str:
    season = _clean_text(value)
    if not season:
        raise CSVValidationError(f"Row {row_number}: season is required.")
    year_only = re.fullmatch(r"(?:19|20)\d{2}", season)
    if year_only:
        return season
    match = re.fullmatch(r"((?:19|20)\d{2})\s*[/_-]\s*(\d{2}|(?:19|20)\d{2})", season)
    if not match:
        raise CSVValidationError(
            f"Row {row_number}: invalid season {season!r}; use YYYY, YYYY-YY, or YYYY-YYYY."
        )
    start = int(match.group(1))
    end_text = match.group(2)
    end = int(end_text) if len(end_text) == 4 else (start // 100) * 100 + int(end_text)
    if end not in (start, start + 1):
        raise CSVValidationError(f"Row {row_number}: season must describe one year or adjacent years.")
    return f"{start}-{end % 100:02d}"


def load_matches(path: str | Path) -> pd.DataFrame:
    """Read and validate a CSV; return normalized rows in stable time order."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise CSVValidationError(
            f"Match CSV not found: {csv_path}. Place an authorized CSV in data/raw/matches.csv."
        )
    try:
        frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except (OSError, UnicodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise CSVValidationError(f"Could not read CSV: {exc}") from exc
    return normalize_matches(frame)


def normalize_matches(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate required match fields and canonicalize teams, leagues, seasons and dates."""
    if frame.empty:
        raise CSVValidationError("The CSV contains no match rows.")

    normalized_columns = {str(column).strip().lower(): column for column in frame.columns}
    missing = [column for column in REQUIRED_COLUMNS if column not in normalized_columns]
    if missing:
        raise CSVValidationError(f"Missing required CSV columns: {', '.join(missing)}.")

    result = frame.rename(
        columns={source: target for target, source in normalized_columns.items()}
    ).copy()
    result = result.loc[:, list(REQUIRED_COLUMNS)].copy()
    result["_row_number"] = range(2, len(result) + 2)

    for column in ("league", "home_team", "away_team", "result"):
        result[column] = result[column].map(_clean_text)
    for index, row in result.iterrows():
        row_number = int(row["_row_number"])
        league_key = row["league"].casefold()
        if league_key not in LEAGUES:
            raise CSVValidationError(
                f"Row {row_number}: unsupported league {row['league']!r}; use one of "
                f"{', '.join(dict.fromkeys(LEAGUES.values()))}."
            )
        result.at[index, "league"] = LEAGUES[league_key]
        result.at[index, "season"] = _normalize_season(row["season"], row_number)
        for team_column in ("home_team", "away_team"):
            if not row[team_column]:
                raise CSVValidationError(f"Row {row_number}: {team_column} is required.")
        if row["home_team"].casefold() == row["away_team"].casefold():
            raise CSVValidationError(f"Row {row_number}: home_team and away_team must differ.")

    dates = pd.to_datetime(result["match_date"].map(_clean_text), errors="coerce", format="mixed")
    invalid_dates = dates.isna()
    if invalid_dates.any():
        row_number = int(result.loc[invalid_dates, "_row_number"].iloc[0])
        raise CSVValidationError(f"Row {row_number}: match_date must be a valid date (YYYY-MM-DD).")
    result["match_date"] = dates.dt.date

    for column in ("home_goals", "away_goals"):
        text = result[column].map(_clean_text)
        values = pd.to_numeric(text, errors="coerce")
        invalid = values.isna() | (values < 0) | (values % 1 != 0)
        if invalid.any():
            row_number = int(result.loc[invalid, "_row_number"].iloc[0])
            raise CSVValidationError(
                f"Row {row_number}: {column} must be a non-negative whole number."
            )
        result[column] = values.astype(int)

    invalid_result = ~result["result"].str.upper().isin(OUTCOMES)
    if invalid_result.any():
        row_number = int(result.loc[invalid_result, "_row_number"].iloc[0])
        raise CSVValidationError(f"Row {row_number}: result must be H, D, or A.")
    result["result"] = result["result"].str.upper()
    derived = result.apply(
        lambda row: "H" if row.home_goals > row.away_goals
        else "A" if row.home_goals < row.away_goals
        else "D",
        axis=1,
    )
    mismatch = result["result"] != derived
    if mismatch.any():
        row_number = int(result.loc[mismatch, "_row_number"].iloc[0])
        raise CSVValidationError(f"Row {row_number}: result does not match the supplied goals.")

    result["_home_key"] = result["home_team"].str.casefold()
    result["_away_key"] = result["away_team"].str.casefold()
    result["_pair_key"] = result.apply(
        lambda row: tuple(sorted((row["_home_key"], row["_away_key"]))), axis=1
    )
    duplicate_key = result.duplicated(["league", "match_date", "_pair_key"], keep=False)
    if duplicate_key.any():
        row_number = int(result.loc[duplicate_key, "_row_number"].iloc[0])
        raise CSVValidationError(
            f"Row {row_number}: duplicate match for this league, date, and team pair."
        )

    result = result.sort_values(
        ["league", "match_date", "_home_key", "_away_key"], kind="stable"
    ).reset_index(drop=True)
    return result.drop(columns=["_row_number", "_home_key", "_away_key", "_pair_key"])


def filter_league(matches: pd.DataFrame, league: str) -> pd.DataFrame:
    """Return a copy of one canonical league's matches."""
    canonical = LEAGUES.get(league.strip().casefold())
    if canonical is None:
        raise CSVValidationError(f"Unsupported league: {league}.")
    subset = matches.loc[matches["league"] == canonical].copy()
    if subset.empty:
        raise CSVValidationError(f"No matches found for {canonical}.")
    return subset.reset_index(drop=True)
