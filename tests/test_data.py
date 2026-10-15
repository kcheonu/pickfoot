from datetime import date

import pandas as pd
import pytest

from pickfoot.data import CSVValidationError, filter_league, normalize_matches


def rows():
    return pd.DataFrame(
        [
            {
                "league": "epl",
                "season": "2023/24",
                "match_date": "2023-08-12",
                "home_team": "  Alpha   FC ",
                "away_team": "Beta FC",
                "home_goals": "2",
                "away_goals": "0",
                "result": "h",
            },
            {
                "league": "Premier League",
                "season": "2023-24",
                "match_date": "2023-08-19",
                "home_team": "Beta FC",
                "away_team": "Alpha FC",
                "home_goals": "1",
                "away_goals": "1",
                "result": "D",
            },
        ]
    )


def test_normalizes_and_sorts_csv_rows():
    normalized = normalize_matches(rows().iloc[::-1])
    assert normalized["league"].tolist() == ["Premier League", "Premier League"]
    assert normalized["season"].tolist() == ["2023-24", "2023-24"]
    assert normalized["match_date"].tolist() == [date(2023, 8, 12), date(2023, 8, 19)]
    assert normalized.loc[0, "home_team"] == "Alpha FC"
    assert normalized["result"].tolist() == ["H", "D"]


def test_requires_schema_columns():
    with pytest.raises(CSVValidationError, match="Missing required CSV columns"):
        normalize_matches(pd.DataFrame({"league": ["epl"]}))


def test_empty_csv_has_actionable_error(tmp_path):
    from pickfoot.data import load_matches

    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("", encoding="utf-8")
    with pytest.raises(CSVValidationError, match="Could not read CSV"):
        load_matches(empty_csv)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("home_goals", "-1", "non-negative whole number"),
        ("away_goals", "1.5", "non-negative whole number"),
        ("home_team", "  ", "home_team is required"),
        ("match_date", "not-a-date", "match_date must be a valid date"),
        ("result", "X", "result must be H, D, or A"),
    ],
)
def test_rejects_invalid_values(column, value, message):
    data = rows()
    data.loc[0, column] = value
    with pytest.raises(CSVValidationError, match=message):
        normalize_matches(data)


def test_rejects_same_team_and_result_score_mismatch():
    data = rows()
    data.loc[0, "away_team"] = "alpha fc"
    with pytest.raises(CSVValidationError, match="must differ"):
        normalize_matches(data)
    data = rows()
    data.loc[0, "result"] = "D"
    with pytest.raises(CSVValidationError, match="does not match"):
        normalize_matches(data)


def test_rejects_duplicate_fixture_even_if_teams_reversed():
    data = rows().iloc[[0]].copy()
    reversed_match = data.copy()
    reversed_match.loc[:, "home_team"] = data.iloc[0]["away_team"]
    reversed_match.loc[:, "away_team"] = data.iloc[0]["home_team"]
    reversed_match.loc[:, "home_goals"] = "0"
    reversed_match.loc[:, "away_goals"] = "2"
    reversed_match.loc[:, "result"] = "A"
    with pytest.raises(CSVValidationError, match="duplicate match"):
        normalize_matches(pd.concat([data, reversed_match], ignore_index=True))


def test_filter_league_keeps_competitions_separate():
    matches = normalize_matches(rows())
    assert len(filter_league(matches, "EPL")) == 2


@pytest.mark.parametrize(
    ("code", "canonical"),
    [
        ("PL", "Premier League"),
        ("PD", "La Liga"),
        ("BL1", "Bundesliga"),
        ("SA", "Serie A"),
        ("FL1", "Ligue 1"),
    ],
)
def test_openfootball_league_codes_normalize(code, canonical):
    data = rows().iloc[[0]].copy()
    data.loc[:, "league"] = code
    normalized = normalize_matches(data)
    assert normalized.loc[0, "league"] == canonical
