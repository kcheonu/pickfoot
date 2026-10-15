import asyncio
from datetime import timedelta
import json
from pathlib import Path
import sys

import pytest
from streamlit.testing.v1 import AppTest

from pickfoot.cli import main
from pickfoot.model import predict_match, save_artifact, train_model
from pickfoot.storage import read_predictions
from tests.test_model import synthetic_matches


@pytest.fixture(scope="module", autouse=True)
def windows_selector_loop_for_streamlit():
    if sys.platform != "win32":
        yield
        return
    original_policy = asyncio.get_event_loop_policy()
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    yield
    asyncio.set_event_loop_policy(original_policy)


def test_cli_refuses_to_train_without_rights_confirmation(capsys):
    code = main([
        "train",
        "--csv", "missing.csv",
        "--source-name", "unknown",
        "--license-reference", "unknown",
    ])
    captured = capsys.readouterr()
    assert code == 2
    assert "verify" in captured.err


def test_cli_reports_missing_local_csv_after_confirmation(capsys):
    code = main([
        "train",
        "--csv", "data\\raw\\not-present.csv",
        "--source-name", "reviewed source",
        "--license-reference", "permission record",
        "--confirm-usage-rights",
    ])
    captured = capsys.readouterr()
    assert code == 2
    assert "not found" in captured.err


def test_cli_trains_and_saves_explicit_synthetic_test_csv(tmp_path, capsys):
    csv_path = tmp_path / "authorized-synthetic-test-only.csv"
    model_directory = tmp_path / "saved-models"
    synthetic_matches().to_csv(csv_path, index=False)
    code = main([
        "train",
        "--csv", str(csv_path),
        "--artifact-dir", str(model_directory),
        "--source-name", "Synthetic test only",
        "--license-reference", "Unit test fixture; not a real data license",
        "--source-url", "https://example.invalid/test",
        "--synthetic-test-data",
    ])
    assert code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["models"][0]["league"] == "Premier League"
    assert Path(output["models"][0]["artifact"]).is_file()
    assert output["models"][0]["evaluation"]["validation_matches"] > 0
    assert output["models"][0]["model_version"] == "pickfoot-pre-match-logreg-1"
    assert output["models"][0]["data_kind"] == "synthetic_test"


def test_cli_generates_time_safe_backtest_report(tmp_path, capsys):
    csv_path = tmp_path / "fixtures.csv"
    report_path = tmp_path / "walk-forward.json"
    synthetic_matches().to_csv(csv_path, index=False)
    code = main([
        "backtest",
        "--csv", str(csv_path),
        "--database", str(tmp_path / "no-existing-db.sqlite3"),
        "--output", str(report_path),
        "--confirm-usage-rights",
    ])
    assert code == 0
    output = json.loads(capsys.readouterr().out)
    assert report_path.is_file()
    assert output["leagues"]["Premier League"]["matches"] > 0
    assert "strictly earlier" in output["method"]


def test_prediction_service_basic_and_team_form_payload():
    artifact = train_model(
        synthetic_matches(),
        source_name="synthetic test only",
        license_reference="unit-test fixture",
    )
    prediction = predict_match(
        artifact,
        home_team="Alpha",
        away_team="Beta",
        match_date=artifact.state.latest_date + timedelta(days=1),
    )
    assert prediction["mode"] == "PRE_MATCH"
    assert set(prediction["probabilities"]) == {"H", "D", "A"}
    assert prediction["home_form"]["recent_matches"] <= 5


def test_streamlit_app_shows_clear_no_model_state(monkeypatch, tmp_path):
    monkeypatch.setenv("PICKFOOT_MODEL_DIR", str(tmp_path / "empty-model-dir"))
    app_path = Path(__file__).parents[1] / "src" / "pickfoot" / "app.py"
    app = AppTest.from_file(app_path).run()
    assert not app.exception
    assert any("PickFoot" in element.value for element in app.title)
    assert any("모델이 없습니다" in element.value for element in app.info)
    assert any(element.disabled for element in app.button if "POST_LINEUP" in element.label)


def test_streamlit_app_predicts_with_trained_synthetic_test_artifact(monkeypatch, tmp_path):
    model_dir = tmp_path / "models"
    csv_path = tmp_path / "synthetic-matches.csv"
    synthetic_matches().to_csv(csv_path, index=False)
    artifact = train_model(
        synthetic_matches(),
        source_name="synthetic test only",
        license_reference="unit-test fixture",
    )
    save_artifact(artifact, model_dir)
    monkeypatch.setenv("PICKFOOT_MODEL_DIR", str(model_dir))
    monkeypatch.setenv("PICKFOOT_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("PICKFOOT_MATCHES_CSV", str(csv_path))
    app_path = Path(__file__).parents[1] / "src" / "pickfoot" / "app.py"
    app = AppTest.from_file(app_path).run()
    assert not app.exception
    assert app.selectbox[0].options == [
        "Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"
    ]
    assert app.selectbox[1].options == artifact.teams
    assert app.selectbox[2].options == [team for team in artifact.teams if team != "Alpha"]
    next(button for button in app.button if button.label == "PRE_MATCH 예측 실행").click()
    app.run()
    assert not app.exception
    assert any("합성 테스트 데모" in element.value for element in app.subheader)
    assert any("SYNTHETIC TEST MODEL" in element.value for element in app.error)
    assert len(app.metric) >= 7
    assert any(
        "synthetic test only" in element.value
        and "원본에 기록되지 않음" in element.value
        for element in app.caption
    )
    assert len(read_predictions(tmp_path / "runtime" / "pickfoot.sqlite3")) == 1


def test_streamlit_elo_rankings_page_renders(monkeypatch, tmp_path):
    model_dir = tmp_path / "models"
    csv_path = tmp_path / "synthetic-matches.csv"
    synthetic_matches().to_csv(csv_path, index=False)
    artifact = train_model(
        synthetic_matches(),
        source_name="synthetic test only",
        license_reference="unit-test fixture",
    )
    save_artifact(artifact, model_dir)
    monkeypatch.setenv("PICKFOOT_MODEL_DIR", str(model_dir))
    monkeypatch.setenv("PICKFOOT_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("PICKFOOT_MATCHES_CSV", str(csv_path))
    app_path = Path(__file__).parents[1] / "src" / "pickfoot" / "app.py"
    app = AppTest.from_file(app_path)
    app.session_state["pickfoot_navigation"] = "Elo 순위"
    app.run()
    assert not app.exception
    assert any("리그별 Elo 순위" in element.value for element in app.header)


def test_streamlit_backtest_page_renders(monkeypatch, tmp_path):
    csv_path = tmp_path / "synthetic-matches.csv"
    synthetic_matches().to_csv(csv_path, index=False)
    monkeypatch.setenv("PICKFOOT_MODEL_DIR", str(tmp_path / "empty-models"))
    monkeypatch.setenv("PICKFOOT_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("PICKFOOT_MATCHES_CSV", str(csv_path))
    monkeypatch.setenv(
        "PICKFOOT_BACKTEST_PATH", str(tmp_path / "runtime" / "missing-backtest.json")
    )
    app_path = Path(__file__).parents[1] / "src" / "pickfoot" / "app.py"
    app = AppTest.from_file(app_path)
    app.session_state["pickfoot_navigation"] = "예측 백테스트"
    app.run()
    assert not app.exception
    assert any("예측 백테스트" in element.value for element in app.header)
    assert any("저장된 과거 시점 재현 결과가 없습니다" in element.value for element in app.info)
