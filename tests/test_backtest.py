from datetime import date, timedelta

import pandas as pd

from pickfoot.backtest import expanding_window_backtest
from tests.test_model import synthetic_matches


def test_expanding_backtest_trains_only_before_target_date_and_scores_all_classes():
    matches = synthetic_matches()
    report = expanding_window_backtest(matches)
    league = report["leagues"]["Premier League"]
    assert league["matches"] > 0
    assert league["evaluation_start"] > "2020-08-01"
    assert league["accuracy"] >= 0
    assert league["log_loss"] >= 0
    assert league["brier_score"] >= 0
    assert len(league["predictions"][0]["probabilities"]) == 3
    for row in league["predictions"]:
        assert row["training_cutoff"] < row["match_date"]
        assert abs(sum(row["probabilities"].values()) - 1.0) < 1e-8


def test_target_and_future_results_cannot_change_walk_forward_prediction():
    matches = synthetic_matches()
    original = expanding_window_backtest(matches)["leagues"]["Premier League"]["predictions"]
    target_date = original[8]["match_date"]
    altered = matches.copy()
    target_mask = altered["match_date"] == date.fromisoformat(target_date)
    target_index = altered.index[target_mask][0]
    outcome = altered.loc[target_index, "result"]
    altered.loc[target_index, ["home_goals", "away_goals", "result"]] = (
        (0, 4, "A") if outcome != "A" else (4, 0, "H")
    )
    future_mask = altered["match_date"] > date.fromisoformat(target_date)
    altered.loc[future_mask, ["home_goals", "away_goals", "result"]] = [3, 0, "H"]
    rerun = expanding_window_backtest(altered)["leagues"]["Premier League"]["predictions"]
    original_row = next(row for row in original if row["match_date"] == target_date)
    rerun_row = next(row for row in rerun if row["match_date"] == target_date)
    assert rerun_row["probabilities"] == original_row["probabilities"]
