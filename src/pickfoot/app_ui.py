"""Streamlit dashboard rendering for local PickFoot artifacts."""

from __future__ import annotations

from datetime import date, timedelta
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from pickfoot.analytics import active_season_roster
from pickfoot.data import CSVValidationError
from pickfoot.dashboard_pages import (
    LEAGUE_NAMES,
    render_backtest_page,
    render_elo_rankings,
    render_match_elo_analysis,
    render_team_analysis,
    render_upload_page,
)
from pickfoot.model import (
    ModelArtifact,
    ModelError,
    load_artifact,
    predict_match_with_state,
)
from pickfoot.service import current_league_state, load_base_matches, state_hash
from pickfoot.storage import (
    StorageError,
    read_registered_teams,
    read_uploaded_matches,
    record_prediction,
)

SUPPORTED_LEAGUES = LEAGUE_NAMES


@st.cache_resource
def load_models(model_directory: str) -> tuple[dict[str, ModelArtifact], dict[str, str]]:
    models: dict[str, ModelArtifact] = {}
    errors: dict[str, str] = {}
    directory = Path(model_directory)
    if not directory.is_dir():
        return models, errors
    for path in sorted(directory.glob("*.joblib")):
        try:
            artifact = load_artifact(path)
        except ModelError as exc:
            errors[path.name] = str(exc)
            continue
        models[artifact.league] = artifact
    return models, errors


def _render_form(form: dict[str, float]) -> str:
    return (
        f"경기 {int(form['recent_matches'])}회 · 승점 {form['recent_points']:.0f} · "
        f"승률 {form['recent_win_rate']:.0%} · 득점 {form['recent_goals_for_avg']:.2f} · "
        f"실점 {form['recent_goals_against_avg']:.2f} · "
        f"득실차 {form['recent_goal_diff_avg']:+.2f}"
    )


def run(model_directory: str | Path | None = None) -> None:
    st.set_page_config(page_title="PickFoot", page_icon="⚽", layout="wide")
    model_directory = model_directory or os.environ.get("PICKFOOT_MODEL_DIR", "models")
    st.title("⚽ PickFoot")
    st.caption("유럽 5대 리그 경기의 H/D/A 사전 추정 대시보드")
    st.warning(
        "확률은 검증이 필요한 모델 추정치이며 실제 결과나 보장된 승률이 아닙니다. "
        "실제 경기 데이터 권한이 확인되지 않으면 실제 예측 모델을 표시하지 않습니다."
    )

    model_view = st.radio("예측 모드", ["PRE_MATCH"], horizontal=True)
    st.button("POST_LINEUP — 추후 지원", disabled=True, help="공식 선발 명단 기반 모델은 아직 지원하지 않습니다.")
    st.caption(f"현재 모드: **{model_view}**")

    models, load_errors = load_models(str(Path(model_directory).resolve()))
    if load_errors:
        for filename, message in load_errors.items():
            st.error(f"모델 파일 {filename}을 읽지 못했습니다: {message}")

    data_directory = Path(os.environ.get("PICKFOOT_DATA_DIR", "data/runtime"))
    database = data_directory / "pickfoot.sqlite3"
    backtest_path = Path(
        os.environ.get("PICKFOOT_BACKTEST_PATH", str(data_directory / "walk_forward_backtest.json"))
    )
    try:
        base_matches = load_base_matches(
            os.environ.get("PICKFOOT_MATCHES_CSV", "data/raw/matches.csv")
        )
    except CSVValidationError as exc:
        base_matches = pd.DataFrame()
        st.warning(f"기준 경기 데이터를 읽지 못했습니다: {exc}")
    try:
        uploaded_matches = read_uploaded_matches(database)
    except (CSVValidationError, StorageError) as exc:
        uploaded_matches = pd.DataFrame()
        st.error(f"저장된 2026/27 경기 결과를 읽지 못했습니다: {exc}")
    registered_teams = read_registered_teams(database)

    pages = [
        "PRE_MATCH 예측",
        "Elo 순위",
        "팀 상세 분석",
        "경기별 Elo 분석",
        "경기 결과 업로드",
        "예측 백테스트",
    ]
    next_page = st.session_state.pop("pickfoot_next_page", None)
    if next_page in pages:
        st.session_state["pickfoot_navigation"] = next_page
    page = st.radio("대시보드 페이지", pages, key="pickfoot_navigation")
    if page == "Elo 순위":
        render_elo_rankings(
            models, base_matches=base_matches, uploaded_matches=uploaded_matches,
            registered_teams=registered_teams,
        )
        return
    if page == "팀 상세 분석":
        render_team_analysis(
            models, base_matches=base_matches, uploaded_matches=uploaded_matches,
            registered_teams=registered_teams,
        )
        return
    if page == "경기별 Elo 분석":
        render_match_elo_analysis(
            base_matches=base_matches, uploaded_matches=uploaded_matches
        )
        return
    if page == "경기 결과 업로드":
        render_upload_page(models, database=database, base_matches=base_matches)
        return
    if page == "예측 백테스트":
        render_backtest_page(
            database=database,
            backtest_path=backtest_path,
            base_matches=base_matches,
            uploaded_matches=uploaded_matches,
        )
        return

    league = st.selectbox("리그", SUPPORTED_LEAGUES)
    artifact = models.get(league)
    if artifact is None:
        st.info(
            f"{league} 모델이 없습니다. 사용 권한을 확인한 경기 CSV를 "
            "`data/raw/matches.csv`에 준비한 뒤 README의 학습 명령을 실행하세요. "
            "테스트용 합성 데이터는 실제 예측 모델로 표시하지 않습니다."
        )
        return

    synthetic_only = artifact.source_metadata.get("data_kind") == "synthetic_test"
    if artifact.source_metadata.get("data_kind") not in ("authorized_real", "synthetic_test"):
        st.error("모델 데이터 출처/사용 유형이 확인되지 않아 결과를 표시하지 않습니다.")
        return
    if synthetic_only:
        st.error(
            "SYNTHETIC TEST MODEL — 아래 화면은 파이프라인/UI 확인 전용이며 "
            "실제 경기 예측이나 성능 결과가 아닙니다."
        )
    report = artifact.evaluation
    state = artifact.state
    current_matches = pd.DataFrame()
    if artifact.source_metadata.get("data_kind") == "authorized_real" and not base_matches.empty:
        try:
            current_matches, current_build = current_league_state(
                artifact,
                base_matches=base_matches,
                uploaded_matches=uploaded_matches,
            )
            state = current_build.state
        except CSVValidationError as exc:
            st.error(f"최신 팀 상태를 계산하지 못했습니다: {exc}")
            return
    st.markdown(f"**모델 버전:** `{artifact.model_version}` · **데이터 기준일:** "
                f"`{artifact.state.latest_date.isoformat()}` · "
                f"**팀 상태 기준일:** `{state.latest_date.isoformat()}`")
    uploaded_for_league = (
        uploaded_matches.loc[uploaded_matches["league"] == league]
        if not uploaded_matches.empty else uploaded_matches
    )
    updated_2026_27 = (
        not uploaded_for_league.empty
        and bool((uploaded_for_league["season"] == "2026-27").any())
    )
    st.info(
        f"학습 시즌: {artifact.source_metadata.get('seasons', '미기록')} · "
        f"원천 경기 {artifact.source_metadata.get('matches_in_source', '미기록')}경기 · "
        f"수집 시각(UTC): "
        f"{artifact.source_metadata.get('source_collected_at_utc') or '원본에 기록되지 않음'} · "
        f"2026/27 업로드 반영: {'예' if updated_2026_27 else '아니오'} · "
        f"학습 cutoff는 {artifact.state.latest_date.isoformat()}, "
        f"현재 Elo/최근 폼 cutoff는 {state.latest_date.isoformat()}입니다. "
        "기존 모델 계수·scaler는 재학습하지 않았습니다. "
        "신규/승격팀은 Elo 분석만 가능하며 기존 모델의 예측은 지원하지 않습니다."
    )
    metric_columns = st.columns(4)
    metric_columns[0].metric("검증 경기 수", report.validation_matches)
    metric_columns[1].metric("Accuracy", f"{report.accuracy:.3f}")
    metric_columns[2].metric("Log Loss", f"{report.log_loss:.3f}")
    metric_columns[3].metric("Multiclass Brier", f"{report.brier_score:.3f}")
    st.caption(
        f"검증 기간: {report.validation_start} – {report.validation_end} · "
        f"학습 경기: {report.train_matches} · "
        f"빈도 baseline (Acc/Log Loss/Brier): "
        f"{report.baseline_accuracy:.3f} / {report.baseline_log_loss:.3f} / "
        f"{report.baseline_brier_score:.3f}"
    )

    if updated_2026_27 and not current_matches.empty:
        active_keys = active_season_roster(
            current_matches,
            registered_teams=registered_teams.get(league, []),
        )
        teams = sorted(
            (name for key, name in state.team_names.items() if key in active_keys),
            key=str.casefold,
        )
    else:
        teams = state.teams
    if len(teams) < 2:
        st.error("예측하려면 학습된 팀이 두 팀 이상 필요합니다.")
        return
    left, right = st.columns(2)
    home_team = left.selectbox("홈팀", teams, key="home_team")
    away_options = [team for team in teams if team.casefold() != home_team.casefold()]
    away_team = right.selectbox("원정팀", away_options, key="away_team")
    min_date = state.latest_date + timedelta(days=1)
    suggested_date = max(date.today(), min_date)
    match_date = st.date_input(
        "경기 날짜",
        value=suggested_date,
        min_value=min_date,
        help="저장된 최신 상태의 과거/동일 날짜 예측은 지원하지 않습니다.",
    )

    if st.button("PRE_MATCH 예측 실행", type="primary"):
        try:
            prediction = predict_match_with_state(
                artifact,
                state=state,
                home_team=home_team,
                away_team=away_team,
                match_date=match_date,
            )
        except ModelError as exc:
            st.error(str(exc))
            return
        year = match_date.year if match_date.month >= 7 else match_date.year - 1
        season = f"{year}-{(year + 1) % 100:02d}"
        try:
            prediction_id, inserted = record_prediction(
                database,
                league=league,
                season=season,
                match_date=match_date.isoformat(),
                home_team=home_team,
                away_team=away_team,
                model_version=artifact.model_version,
                model_training_cutoff=artifact.state.latest_date.isoformat(),
                team_state_cutoff=state.latest_date.isoformat(),
                team_state_hash=state_hash(prediction["features"]),
                probabilities=prediction["probabilities"],
            )
        except (StorageError, OSError) as exc:
            st.error(f"예측은 계산했지만 기록을 저장하지 못했습니다: {exc}")
            return
        st.caption(
            f"예측 기록 ID: `{prediction_id}` · "
            f"{'새 기록 저장' if inserted else '동일 모델/팀 상태의 기존 기록 재사용'} · "
            "정확한 데이터 취득 시각은 미확인"
        )
        st.caption(
            f"모델 학습 기준일: {artifact.state.latest_date} · "
            f"팀 상태 업데이트 기준일: {state.latest_date} · "
            f"2026/27 데이터 반영: {'예' if updated_2026_27 else '아니오'}"
        )
        st.subheader(
            "합성 테스트 데모 — 실제 예측 아님"
            if synthetic_only
            else "검증이 필요한 추정 확률"
        )
        probabilities = prediction["probabilities"]
        probability_columns = st.columns(3)
        for column, label, title in zip(
            probability_columns, ("H", "D", "A"), ("홈 승", "무승부", "원정 승"), strict=True
        ):
            column.metric(title, f"{probabilities[label]:.1%}")
        chart = pd.DataFrame(
            {"결과": ["홈 승 (H)", "무승부 (D)", "원정 승 (A)"],
             "추정 확률": [probabilities["H"], probabilities["D"], probabilities["A"]]}
        ).set_index("결과")
        st.bar_chart(chart, y="추정 확률")

        st.subheader("팀 상태 비교")
        team_columns = st.columns(2)
        team_columns[0].metric(f"{home_team} Elo", f"{prediction['home_elo']:.1f}")
        team_columns[1].metric(f"{away_team} Elo", f"{prediction['away_elo']:.1f}")
        team_columns[0].write("최근 5경기: " + _render_form(prediction["home_form"]))
        team_columns[1].write("최근 5경기: " + _render_form(prediction["away_form"]))
        st.expander("PRE_MATCH 입력 피처").json(prediction["features"])
        st.caption(
            "평가 지표는 저장된 시간순 검증 구간 결과입니다. 데이터 출처: "
            f"{artifact.source_metadata['source_name']} · "
            f"원본: {artifact.source_metadata.get('source_url') or 'URL 미기록'} · "
            f"이용 조건 참조: {artifact.source_metadata['license_reference']} · "
            f"로컬 수집 시각(UTC): "
            f"{artifact.source_metadata.get('source_collected_at_utc') or '원본에 기록되지 않음'}"
        )
