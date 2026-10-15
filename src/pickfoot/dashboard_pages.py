"""Streamlit pages for Elo, result management, and backtest analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from pickfoot.analytics import active_season_roster, league_standings, team_match_history
from pickfoot.backtest import matches_fingerprint
from pickfoot.data import CSVValidationError
from pickfoot.features import build_features
from pickfoot.model import ModelArtifact
from pickfoot.service import merged_matches, prepare_upload
from pickfoot.storage import (
    StorageError,
    commit_upload,
    correct_uploaded_conflicts,
    fixture_key,
    load_backtest_results,
    read_predictions,
    read_registered_teams,
    read_team_aliases,
    read_uploaded_matches,
    read_uploads,
    save_team_alias,
    register_team,
)

LEAGUE_NAMES = [
    "Premier League",
    "La Liga",
    "Bundesliga",
    "Serie A",
    "Ligue 1",
]


def _league_matches(
    league: str,
    *,
    base_matches: pd.DataFrame,
    uploaded_matches: pd.DataFrame,
) -> pd.DataFrame:
    return merged_matches(base_matches, uploaded_matches, league)


def _ranked_league(
    league: str,
    *,
    base_matches: pd.DataFrame,
    uploaded_matches: pd.DataFrame,
    registered_teams: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, Any]:
    matches = _league_matches(
        league, base_matches=base_matches, uploaded_matches=uploaded_matches
    )
    features = build_features(matches)
    manually_registered = registered_teams.get(league, [])
    for team in manually_registered:
        features.state.register_team(team)
    standings = league_standings(matches, features)
    roster = active_season_roster(matches, registered_teams=manually_registered)
    standings = standings.loc[
        standings["team"].str.casefold().isin(roster)
    ].reset_index(drop=True)
    standings["rank"] = standings["elo"].rank(method="min", ascending=False).astype(int)
    return matches, standings, features


def render_elo_rankings(
    models: dict[str, ModelArtifact],
    *,
    base_matches: pd.DataFrame,
    uploaded_matches: pd.DataFrame,
    registered_teams: dict[str, list[str]],
) -> None:
    st.header("리그별 Elo 순위")
    league = st.selectbox("Elo 순위 리그", LEAGUE_NAMES, key="elo_league")
    try:
        matches, table, feature_build = _ranked_league(
            league, base_matches=base_matches, uploaded_matches=uploaded_matches,
            registered_teams=registered_teams,
        )
    except CSVValidationError as exc:
        st.info(str(exc))
        return

    artifact = models.get(league)
    trained_teams = set(artifact.teams) if artifact else set()
    table["모델 지원"] = table["team"].map(
        lambda team: "지원" if team in trained_teams else "신규/미학습 — 예측 불가"
    )
    as_of = feature_build.state.latest_date.isoformat()
    has_2026_27 = bool((matches["season"] == "2026-27").any())
    latest_season = str(matches["season"].max())
    st.caption(
        f"데이터 기준일: {as_of} · 표시 시즌: {latest_season} · "
        f"2026/27 업로드 반영: {'예' if has_2026_27 else '아니오'} · "
        f"Elo는 이 리그 결과만으로 계산한 상대적 전력 지표입니다."
    )
    if has_2026_27:
        st.caption(
            "현재 팀 목록은 2026/27 업로드에 등장한 팀과 사용자가 수동 확인해 등록한 팀입니다. "
            "아직 업로드 데이터에 나오지 않은 참가팀은 ‘신규 팀 등록’에서 등록해야 표시됩니다."
        )
    st.info(
        "동일 Elo는 같은 순위를 부여하고 팀 이름 오름차순으로 표시합니다. "
        "Elo 초기값은 1500이며 시즌 경계에서 평균 회귀를 적용하지 않습니다. "
        f"마지막 경기까지 총 {len(matches)}경기 상태를 다시 계산했습니다."
    )
    st.dataframe(
        table.rename(columns={
            "rank": "순위",
            "team": "팀",
            "elo": "Elo",
            "last_match_date": "마지막 경기",
            "season": "시즌",
            "season_matches": "시즌 경기 수",
            "recent_points": "최근 5경기 승점",
            "recent_matches": "최근 표본 경기 수",
        }),
        hide_index=True,
        use_container_width=True,
    )
    teams = table["team"].tolist()
    selected = st.selectbox("상세 분석할 팀", teams, key="ranking_detail_team")
    if st.button("팀 상세 분석 열기", key="open_team_analysis"):
        st.session_state["pickfoot_detail_team"] = selected
        st.session_state["pickfoot_next_page"] = "팀 상세 분석"
        st.rerun()


def render_team_analysis(
    models: dict[str, ModelArtifact],
    *,
    base_matches: pd.DataFrame,
    uploaded_matches: pd.DataFrame,
    registered_teams: dict[str, list[str]],
) -> None:
    st.header("팀 상세 분석")
    league = st.selectbox("상세 분석 리그", LEAGUE_NAMES, key="detail_league")
    try:
        matches, table, feature_build = _ranked_league(
            league, base_matches=base_matches, uploaded_matches=uploaded_matches,
            registered_teams=registered_teams,
        )
    except CSVValidationError as exc:
        st.info(str(exc))
        return
    teams = table["team"].tolist()
    requested_team = st.session_state.get("pickfoot_detail_team")
    default_index = teams.index(requested_team) if requested_team in teams else 0
    team = st.selectbox("팀", teams, index=default_index, key="detail_team")
    key = team.casefold()
    history = team_match_history(feature_build.elo_history, team)
    standing = table.loc[table["team"] == team].iloc[0]
    model_teams = set(models[league].teams) if league in models else set()
    if history.empty:
        st.warning("이 팀의 경기 기록이 없습니다. 데이터가 부족해 Elo/최근 폼을 계산할 수 없습니다.")
        st.metric("초기 Elo", f"{feature_build.state.elo.get(key, 1500.0):.1f}")
        st.caption(
            "이 팀은 명시적으로 등록됐지만 경기 기록은 없습니다. "
            "신규 팀 Elo는 1500으로 시작하며 기존 Logistic Regression 예측은 지원하지 않습니다."
        )
        return
    recent = history.tail(5)
    points_by_result = {"H": 3, "D": 1, "A": 0}
    recent_points = int(recent["result"].map(points_by_result).sum())
    recent_count = len(recent)
    rank = int(standing["rank"])
    st.caption(
        f"{league} Elo 순위 {rank}위 · 데이터 기준일 {feature_build.state.latest_date} · "
        f"{standing['season']}시즌 경기 {int(standing['season_matches'])}회"
    )
    columns = st.columns(4)
    columns[0].metric("현재 Elo", f"{feature_build.state.elo[key]:.1f}")
    columns[1].metric("초기 Elo 대비", f"{feature_build.state.elo[key] - 1500:+.1f}")
    columns[2].metric("최근 5경기 승점", f"{recent_points}/{recent_count * 3}")
    win_rate = float((recent["result"] == "H").sum() / recent_count)
    columns[3].metric("최근 승률", f"{win_rate:.0%}")
    columns = st.columns(3)
    columns[0].metric("최근 평균 득점", f"{recent['team_goals'].mean():.2f}")
    columns[1].metric("최근 평균 실점", f"{recent['opponent_goals'].mean():.2f}")
    columns[2].metric(
        "모델 지원",
        "지원" if team in model_teams else "미지원",
        help="새 팀의 Elo 계산과 기존 Logistic Regression의 학습 팀 범위는 별개입니다.",
    )
    if team not in model_teams:
        st.warning("이 팀은 기존 모델 학습 roster에 없습니다. Elo 분석은 가능하지만 예측 확률은 제공하지 않습니다.")

    history["그래프 표기"] = history.apply(
        lambda row: f"{row['match_date']} vs {row['opponent']}", axis=1
    )
    chart = history.set_index("그래프 표기")[["elo_after"]].rename(
        columns={"elo_after": "경기일 종료 Elo"}
    )
    st.subheader("경기별 Elo 이력")
    st.caption("같은 날짜 경기는 누수를 막기 위해 당일 종료 Elo를 공유합니다. 날짜만 있고 킥오프 시각은 없습니다.")
    st.line_chart(chart)
    match_options = list(range(len(history)))
    selected_index = st.selectbox(
        "경기 지점 선택",
        match_options,
        index=len(match_options) - 1,
        format_func=lambda index: (
            f"{history.iloc[index]['match_date']} · vs {history.iloc[index]['opponent']} · "
            f"{history.iloc[index]['result']} · Elo {history.iloc[index]['elo_change']:+.1f}"
        ),
    )
    selected_match = history.iloc[selected_index]
    st.info(
        f"상대: {selected_match['opponent']} · 결과: {selected_match['result']} · "
        f"경기 전 Elo: {selected_match['elo_before']:.1f} · "
        f"변화: {selected_match['elo_change']:+.1f} · "
        f"경기일 종료 Elo: {selected_match['elo_after']:.1f}"
    )
    st.subheader("최근 경기")
    st.dataframe(
        history.tail(10).sort_values("match_date", ascending=False).rename(columns={
            "match_date": "날짜", "season": "시즌", "opponent": "상대",
            "venue": "장소", "team_goals": "득점", "opponent_goals": "실점",
            "result": "결과", "elo_before": "경기 전 Elo",
            "elo_after": "경기일 종료 Elo", "elo_change": "Elo 변화",
        }),
        hide_index=True,
        use_container_width=True,
    )


def render_match_elo_analysis(
    *,
    base_matches: pd.DataFrame,
    uploaded_matches: pd.DataFrame,
) -> None:
    st.header("경기별 Elo 분석")
    league = st.selectbox("경기 분석 리그", LEAGUE_NAMES, key="match_elo_league")
    try:
        matches, feature_build = _league_matches(
            league, base_matches=base_matches, uploaded_matches=uploaded_matches
        ), None
        feature_build = build_features(matches)
    except CSVValidationError as exc:
        st.info(str(exc))
        return
    history = feature_build.elo_history.sort_values("match_date", ascending=False)
    st.caption(
        f"데이터 기준일: {feature_build.state.latest_date} · Elo 초기값 1500 · "
        "K=20 · 홈 보정 +60 Elo · 시즌 평균 회귀 없음"
    )
    st.dataframe(
        history.rename(columns={
            "match_date": "날짜", "league": "리그", "season": "시즌",
            "home_team": "홈팀", "away_team": "원정팀",
            "home_goals": "홈 점수", "away_goals": "원정 점수",
            "home_elo_before": "홈 경기 전 Elo", "away_elo_before": "원정 경기 전 Elo",
            "home_elo_after": "홈 경기일 종료 Elo", "away_elo_after": "원정 경기일 종료 Elo",
            "home_elo_change": "홈 Elo 변화", "away_elo_change": "원정 Elo 변화",
            "expected_home_score": "홈 기대 점수", "expected_away_score": "원정 기대 점수",
            "result": "실제 결과",
        }),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "기대 점수는 Elo의 기대 score(승=1, 무=0.5, 패=0)이지 H/D/A 확률이 아닙니다. "
        "변화식: Δ홈 = 20 × (실제 홈 score − 기대 홈 score), Δ원정 = −Δ홈; "
        "기대 홈 score = 1 / (1 + 10^((Elo원정 − (Elo홈 + 60))/400))."
    )
    chart_data = history.set_index("match_date")[["home_elo_change", "away_elo_change"]]
    st.subheader("최근 경기별 Elo 변화량")
    st.bar_chart(chart_data.head(30).sort_index().rename(columns={
        "home_elo_change": "홈팀 변화",
        "away_elo_change": "원정팀 변화",
    }))
    teams = feature_build.state.teams
    if teams:
        team = st.selectbox("추세 팀", teams, key="elo_trend_team")
        team_history = team_match_history(feature_build.elo_history, team)
        st.line_chart(
            team_history.set_index("match_date")[["elo_after"]].rename(
                columns={"elo_after": team}
            )
        )
    swing_rows = []
    for team in feature_build.state.teams:
        team_history = team_match_history(feature_build.elo_history, team)
        change = float(team_history.tail(5)["elo_change"].sum())
        swing_rows.append({"팀": team, "최근 5경기 누적 Elo 변화": change})
    swings = pd.DataFrame(swing_rows).sort_values(
        "최근 5경기 누적 Elo 변화", ascending=False
    )
    left, right = st.columns(2)
    left.subheader("최근 Elo 상승")
    left.dataframe(swings.head(5), hide_index=True)
    right.subheader("최근 Elo 하락")
    right.dataframe(swings.tail(5).sort_values(
        "최근 5경기 누적 Elo 변화"
    ), hide_index=True)
    selected = history.iloc[0]
    elo_difference = float(selected["home_elo_before"] - selected["away_elo_before"])
    st.info(
        f"상승/하락 원인은 경기 전 Elo 차이({elo_difference:+.1f}), 홈 보정 "
        f"{60:.0f}, 실제 결과 {selected['result']} 및 기대 홈 score "
        f"{selected['expected_home_score']:.3f}에 따른 K=20 업데이트입니다. "
        "강팀을 이기는 경우 실제 score가 기대 score보다 크게 높아져 Elo 변동 폭이 커집니다."
    )


def render_upload_page(
    models: dict[str, ModelArtifact],
    *,
    database: Path,
    base_matches: pd.DataFrame,
) -> None:
    st.header("2026/27 경기 결과 업로드")
    st.caption("UTF-8/UTF-8 BOM CSV · 선택은 미리보기일 뿐이며, 확인 버튼을 눌러야 저장됩니다.")
    previous_upload = st.session_state.pop("pickfoot_last_upload", None)
    if previous_upload:
        st.success(
            f"저장 결과: 신규 {previous_upload['inserted_matches']}경기, "
            f"중복 {previous_upload['duplicate_matches']}경기, "
            f"정정 {previous_upload.get('corrected_matches', 0)}경기, "
            f"연결 예측 {previous_upload['linked_predictions']}건."
        )
    uploads = read_uploads(database)
    if not uploads.empty:
        st.caption("최근 업로드 이력")
        st.dataframe(uploads.head(10), hide_index=True, use_container_width=True)

    known_registered_teams = read_registered_teams(database)
    alias_candidates = {}
    for league in LEAGUE_NAMES:
        known = set(models[league].teams if league in models else [])
        for source in (base_matches, read_uploaded_matches(database)):
            if not source.empty:
                league_rows = source.loc[source["league"] == league]
                known.update(league_rows["home_team"].tolist())
                known.update(league_rows["away_team"].tolist())
        known.update(known_registered_teams.get(league, []))
        alias_candidates[league] = sorted(known, key=str.casefold)
    with st.expander("신규 팀 등록"):
        new_team_league = st.selectbox("신규 팀 리그", LEAGUE_NAMES, key="new_team_league")
        new_team_name = st.text_input("신규 팀 정식 표기", key="new_team_name")
        registration_note = st.text_input(
            "등록 근거", key="new_team_note",
            help="예: 사용자가 확인한 2026/27 승격팀 명단. 자동으로 사실을 추론하지 않습니다.",
        )
        registration_confirmed = st.checkbox(
            "이 팀이 해당 리그의 팀임을 확인했습니다.",
            key="new_team_confirmed",
        )
        if st.button(
            "새 팀 등록 및 Elo 1500 초기화",
            key="register_new_team",
            disabled=not registration_confirmed,
        ):
            try:
                created = register_team(
                    database,
                    league=new_team_league,
                    team_name=new_team_name,
                    registration_note=registration_note,
                )
                if created:
                    st.success(
                        f"{new_team_name.strip()}을 {new_team_league}의 신규 팀으로 등록했습니다. "
                        "Elo는 1500, 최근 폼은 표본 없음으로 시작합니다."
                    )
                else:
                    st.info("해당 팀은 이미 등록되어 있습니다.")
                st.rerun()
            except StorageError as exc:
                st.error(str(exc))
    with st.expander("팀 이름 별칭 관리"):
        alias_league = st.selectbox("별칭 리그", LEAGUE_NAMES, key="alias_league")
        canonical_options = alias_candidates[alias_league]
        if canonical_options:
            canonical = st.selectbox(
                "기준 팀명", canonical_options, key="alias_canonical_team"
            )
            alias = st.text_input("CSV에서 사용할 별칭", key="alias_value")
            if st.button("별칭 저장", key="save_alias"):
                try:
                    save_team_alias(
                        database,
                        league=alias_league,
                        alias=alias,
                        canonical_team=canonical,
                    )
                    st.success(f"{alias} → {canonical} 별칭을 저장했습니다.")
                    st.rerun()
                except StorageError as exc:
                    st.error(str(exc))
        else:
            st.info("선택한 리그의 학습 팀이 없어 별칭 기준 팀을 선택할 수 없습니다.")
        aliases = read_team_aliases(database, alias_league)
        st.write(aliases or "등록된 별칭 없음")

    uploaded_file = st.file_uploader(
        "경기 결과 CSV",
        type=["csv"],
        key="match_results_upload",
        help="필수 컬럼: league, season, match_date, home_team, away_team, home_goals, away_goals, result",
    )
    if uploaded_file is None:
        st.info("2026-27 시즌 완료 경기 결과 CSV를 선택하면 저장 전 검증·충돌 확인·Elo 미리보기가 표시됩니다.")
        return
    try:
        stored = read_uploaded_matches(database)
        incoming, digest, new_matches, duplicate_count, conflicts = prepare_upload(
            uploaded_file.getvalue(),
            database=database,
            base_matches=base_matches,
            uploaded_matches=stored,
        )
    except (CSVValidationError, StorageError) as exc:
        st.error(str(exc))
        return
    st.write(
        f"검증된 행: {len(incoming)} · 신규 행: {len(new_matches)} · "
        f"기존과 동일한 중복: {duplicate_count} · SHA-256: `{digest}`"
    )
    if conflicts:
        st.error(f"기존 점수와 다른 충돌 {len(conflicts)}건입니다. 자동 덮어쓰기는 하지 않습니다.")
        st.dataframe(pd.DataFrame([
            {
                "incoming": str(item["incoming"]),
                "existing": str(item["existing"]),
            }
            for item in conflicts
        ]), hide_index=True, use_container_width=True)
        stored_keys = {
            (
                str(row["league"]),
                str(row["season"]),
                str(row["match_date"]),
                fixture_key(str(row["home_team"]), str(row["away_team"])),
            )
            for row in stored.to_dict("records")
        }
        conflicts_are_uploads = all(
            (
                str(item["incoming"]["league"]),
                str(item["incoming"]["season"]),
                str(item["incoming"]["match_date"]),
                fixture_key(
                    str(item["incoming"]["home_team"]),
                    str(item["incoming"]["away_team"]),
                ),
            ) in stored_keys
            for item in conflicts
        )
        if conflicts_are_uploads:
            st.warning(
                "기존 2026/27 업로드 행만 별도 정정할 수 있습니다. 이전 값과 새 값, 출처 및 권한을 "
                "확인한 뒤 정정 확인을 해야 저장됩니다."
            )
            correction_source = st.text_input(
                "정정 데이터 출처", value="사용자 제공 정정 CSV", key="correction_source"
            )
            correction_license = st.text_input(
                "정정 데이터 이용 권한", key="correction_license"
            )
            correction_confirmed = st.checkbox(
                "기존 2026/27 결과를 새 점수로 정정하는 작업을 확인했습니다.",
                key="correction_confirmed",
            )
            if st.button(
                "검토한 경기 결과 정정",
                type="primary",
                disabled=not correction_confirmed,
                key="commit_result_correction",
            ):
                try:
                    result = correct_uploaded_conflicts(
                        database,
                        conflicts=conflicts,
                        content_sha256=digest,
                        file_name=uploaded_file.name,
                        source_name=correction_source,
                        license_reference=correction_license,
                        correction_confirmed=correction_confirmed,
                        base_matches=base_matches,
                    )
                    st.session_state["pickfoot_last_upload"] = result
                    st.rerun()
                except (StorageError, OSError) as exc:
                    st.error(f"정정 저장 실패: 기존 결과는 유지되었습니다. {exc}")
        else:
            st.warning(
                "기준 2025/26 CSV 또는 아직 저장되지 않은 fixture 충돌은 이 화면에서 수정할 수 없습니다. "
                "원본 기준 데이터는 불변으로 유지됩니다."
            )
        return
    st.subheader("저장 예정 경기")
    st.dataframe(new_matches, hide_index=True, use_container_width=True)
    existing_parts = [part for part in (base_matches, stored) if not part.empty]
    current = merged_matches(base_matches, stored) if existing_parts else incoming.iloc[0:0]
    projected_uploads = pd.concat([stored, new_matches], ignore_index=True)
    projected_parts = [part for part in (base_matches, projected_uploads) if not part.empty]
    projected = merged_matches(base_matches, projected_uploads) if projected_parts else incoming.iloc[0:0]
    for league in sorted(new_matches["league"].unique()):
        if league not in models or models[league].source_metadata.get("data_kind") != "authorized_real":
            continue
        before = build_features(current[current["league"] == league])
        after = build_features(projected[projected["league"] == league])
        before_ratings, after_ratings = before.state.elo, after.state.elo
        delta_rows = []
        for key, name in after.state.team_names.items():
            old = before_ratings.get(key, 1500.0)
            new = after_ratings.get(key, 1500.0)
            if new != old:
                delta_rows.append({"팀": name, "현재 Elo": old, "적용 후 Elo": new, "변화": new - old})
        st.subheader(f"{league} 예상 Elo 변경")
        if delta_rows:
            st.dataframe(pd.DataFrame(delta_rows).sort_values("변화", ascending=False),
                         hide_index=True, use_container_width=True)
        else:
            st.caption("예상 Elo 변동 없음")

    source_name = st.text_input("데이터 제공자/출처", value="사용자 제공 CSV", key="upload_source")
    license_reference = st.text_input(
        "이용 권한 또는 라이선스 기록",
        value="",
        key="upload_license",
        help="학습/로컬 분석 권한을 확인한 기록을 입력하세요. 공개 열람만으로 권한을 추정하지 않습니다.",
    )
    rights_confirmed = st.checkbox(
        "이 데이터의 출처와 로컬 저장·분석 이용 권한을 확인했습니다.",
        key="upload_rights_confirmed",
    )
    disabled = bool(conflicts) or new_matches.empty or not rights_confirmed
    if st.button(
        "검증된 2026/27 결과 반영",
        type="primary",
        disabled=disabled,
        key="commit_match_upload",
    ):
        try:
            result = commit_upload(
                database,
                matches=incoming,
                content_sha256=digest,
                file_name=uploaded_file.name,
                source_name=source_name,
                license_reference=license_reference,
                rights_confirmed=rights_confirmed,
                base_matches=base_matches,
            )
            st.session_state["pickfoot_last_upload"] = result
            st.rerun()
        except (StorageError, OSError) as exc:
            st.error(f"저장 실패: 기존 데이터는 유지되었습니다. {exc}")


def render_backtest_page(
    *,
    database: Path,
    backtest_path: Path,
    base_matches: pd.DataFrame,
    uploaded_matches: pd.DataFrame,
) -> None:
    st.header("예측 백테스트")
    stored_predictions = read_predictions(database)
    confirmed = stored_predictions.loc[
        (stored_predictions.get("evaluation_status", pd.Series(dtype=str)) == "verified_pre_match")
    ] if not stored_predictions.empty else pd.DataFrame()
    linked = stored_predictions.loc[
        stored_predictions.get("actual_result", pd.Series(dtype=object)).notna()
    ] if not stored_predictions.empty else pd.DataFrame()
    real_tab, replay_tab = st.tabs(["저장된 실제 사전 예측", "과거 시점 재현"])
    with real_tab:
        st.caption(
            "실제 사전 예측 평가는 경기 시작 전에 생성됐음을 확인할 수 있는 기록만 포함합니다. "
            "현재 결과 CSV에는 날짜만 있고 킥오프 시각이 없어 예측 시각만으로 선후관계를 증명할 수 없습니다."
        )
        st.metric("검증 가능한 사전 예측 수", len(confirmed))
        st.metric("결과가 연결됐으나 킥오프 미확인인 기록", len(linked.loc[
            linked["evaluation_status"] == "unverified_kickoff_time"
        ]) if not linked.empty else 0)
        if confirmed.empty:
            st.info("현재는 검증된 사전 예측 기록이 없습니다. 날짜만 있는 경기 결과는 실제 사전 예측 지표에서 제외합니다.")
        else:
            st.dataframe(confirmed, hide_index=True, use_container_width=True)

    with replay_tab:
        try:
            report = load_backtest_results(backtest_path)
        except StorageError as exc:
            st.error(str(exc))
            return
        if report is None:
            st.info(
                "저장된 과거 시점 재현 결과가 없습니다. 권한 확인 후 다음 명령으로 생성하세요: "
                "`python -m pickfoot.cli backtest --confirm-usage-rights`"
            )
            return
        available = [frame for frame in (base_matches, uploaded_matches) if not frame.empty]
        current = merged_matches(base_matches, uploaded_matches) if available else pd.DataFrame()
        if not current.empty and report.get("source_data_hash") != matches_fingerprint(current):
            st.warning("저장된 백테스트의 입력 데이터가 현재 경기 결과와 다릅니다. CLI로 다시 생성하세요.")
            return
        st.caption(
            f"방법: {report.get('method')} · 생성 시각 UTC: {report.get('generated_at_utc')} · "
            "이는 실제 예측 기록이 아니라 과거 시점의 expanding-window 재현입니다."
        )
        league_rows = []
        all_predictions = []
        for league, details in report.get("leagues", {}).items():
            if not details.get("matches"):
                continue
            league_rows.append({
                "리그": league,
                "경기 수": details["matches"],
                "평가 기간": f"{details['evaluation_start']} – {details['evaluation_end']}",
                "Accuracy": details["accuracy"],
                "Baseline Accuracy": details["baseline_accuracy"],
                "Log Loss": details["log_loss"],
                "Baseline Log Loss": details["baseline_log_loss"],
                "Brier": details["brier_score"],
                "Baseline Brier": details["baseline_brier_score"],
            })
            all_predictions.extend(details["predictions"])
        if not league_rows:
            st.warning("데이터가 부족해 재현 백테스트 예측을 만들지 못했습니다.")
            return
        summary = pd.DataFrame(league_rows)
        st.dataframe(summary, hide_index=True, use_container_width=True)
        st.caption(
            "평가 표본이 작거나 단일 시즌일 수 있어 지표는 불확실합니다. "
            "모든 모델은 해당 경기 날짜보다 앞선 경기로만 학습했고, 같은 날짜 결과는 학습·피처에 반영하지 않았습니다."
        )
        metric_column = st.selectbox(
            "리그별 비교 지표",
            ["Accuracy", "Log Loss", "Brier"],
            key="backtest_metric",
        )
        chart_column = metric_column
        baseline_column = {
            "Accuracy": "Baseline Accuracy",
            "Log Loss": "Baseline Log Loss",
            "Brier": "Baseline Brier",
        }[chart_column]
        st.bar_chart(summary.set_index("리그")[[chart_column, baseline_column]])
        predictions = pd.DataFrame(all_predictions)
        if not predictions.empty:
            predictions["date"] = pd.to_datetime(predictions["match_date"])
            predictions["month"] = predictions["date"].dt.to_period("M").astype(str)
            probabilities = pd.DataFrame(predictions["probabilities"].tolist())
            predictions = pd.concat([predictions, probabilities.add_prefix("P_")], axis=1)
            predictions["hit"] = (
                predictions[["P_H", "P_D", "P_A"]].to_numpy().argmax(axis=1)
                == predictions["actual_result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
            )
            st.subheader("예측 확률과 실제 결과")
            st.dataframe(
                predictions[[
                    "league", "match_date", "home_team", "away_team",
                    "actual_result", "P_H", "P_D", "P_A", "hit", "training_cutoff",
                ]].rename(columns={
                    "league": "리그", "match_date": "날짜", "home_team": "홈팀",
                    "away_team": "원정팀", "actual_result": "실제 결과",
                    "P_H": "P(H)", "P_D": "P(D)", "P_A": "P(A)",
                    "hit": "최고 확률 적중", "training_cutoff": "학습 기준일",
                }),
                hide_index=True,
                use_container_width=True,
            )
            st.subheader("기간별 적중률")
            period = predictions.groupby("month", as_index=True)["hit"].mean().to_frame("Accuracy")
            st.line_chart(period)
            st.subheader("예측 확률 구간별 실제 적중률")
            calibration_rows = []
            for outcome in ("H", "D", "A"):
                probability_column = f"P_{outcome}"
                for bucket in range(0, 10):
                    low, high = bucket / 10, (bucket + 1) / 10
                    mask = (
                        (predictions[probability_column] >= low)
                        & (predictions[probability_column] < high if bucket < 9
                           else predictions[probability_column] <= high)
                    )
                    sample = predictions.loc[mask]
                    if not sample.empty:
                        calibration_rows.append({
                            "결과 클래스": outcome,
                            "예측 구간": f"{low:.1f}–{high:.1f}",
                            "표본 수": len(sample),
                            "평균 예측 확률": float(sample[probability_column].mean()),
                            "실제 적중률": float((sample["actual_result"] == outcome).mean()),
                        })
            st.dataframe(pd.DataFrame(calibration_rows), hide_index=True, use_container_width=True)
