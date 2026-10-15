"use client";

import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";
import { api, jsonBody } from "@/lib/api";

const LEAGUES = ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"];
const PAGES = [
  "대시보드", "승무패 예측", "Elo 순위", "팀 분석", "경기별 Elo 분석",
  "2026/27 경기 관리", "모델 학습 관리", "예측 백테스트",
] as const;
type PageName = (typeof PAGES)[number];
type Auth = { authenticated: boolean; username: string };
type Rank = {
  rank: number; team: string; elo: number; last_match_date: string | null;
  season: string; season_matches: number; recent_points: number; recent_matches: number;
};
type EloEvent = {
  match_date: string; league: string; season: string; home_team: string; away_team: string;
  home_goals: number; away_goals: number; result: string; home_elo_before: number;
  away_elo_before: number; home_elo_after: number; away_elo_after: number;
  home_elo_change: number; away_elo_change: number; expected_home_score: number;
};
type Match = {
  id: string; league: string; season: string; round: string; match_date: string;
  home_team: string; away_team: string; home_goals: number | null; away_goals: number | null;
  result: string | null; status: string; version: number;
};
type Model = {
  id: string | null; league: string; version: string; training_cutoff: string | null;
  evaluation: Record<string, number | string>; comparison: Record<string, unknown> | null;
  candidate_status: string; active: boolean; created_at: string | null;
};
type Job = {
  id: string; job_type: string; league: string; status: string; progress_message: string;
  created_at: string; result: Record<string, unknown> | null; error_message: string | null;
  candidate_version_id: string | null;
};
type Prediction = {
  id: string; league: string; season: string; match_date: string; home_team: string;
  away_team: string; probabilities: Record<"H" | "D" | "A", number>; model_version: string;
  model_training_cutoff: string; team_state_cutoff: string; predicted_at: string;
  actual_result: string | null; evaluation_status: string;
};
type FormSummary = {
  recent_points: number; recent_win_rate: number;
  recent_goals_for_avg: number; recent_goals_against_avg: number;
};
type PredictionOutput = {
  prediction_id: string; league: string; season: string; match_date: string;
  home_team: string; away_team: string; probabilities: Prediction["probabilities"];
  home_elo: number; away_elo: number; home_form: FormSummary; away_form: FormSummary;
  model_version: string; model_training_cutoff: string; team_state_cutoff: string;
  evaluation: Record<string, number | string>; note: string;
};
type PredictionComparison = {
  operating: { version: string; probabilities: Prediction["probabilities"] };
  candidate: { version: string; probabilities: Prediction["probabilities"] };
  team_state_cutoff: string; warning: string;
};
type Dataset = {
  id: string; file_name: string; file_sha256: string; source_name: string;
  license_reference: string; league_coverage: Record<string, number>; row_count: number;
  coverage_start: string; coverage_end: string; uploaded_at: string; active: boolean;
};
type JobSubmission = { job_id: string; status: string; deduplicated?: boolean };
type ResultConfirmation = {
  elo?: { home_after?: number; away_after?: number; home_change?: number; away_change?: number } | null;
};
type BacktestMetric = {
  evaluation_start: string; evaluation_end: string; matches: number;
  accuracy: number; baseline_accuracy: number; log_loss: number; baseline_log_loss: number;
  brier_score: number; baseline_brier_score: number;
};
type BacktestReport = { method: string; leagues: Record<string, BacktestMetric> };

const pct = (value: number) => `${(value * 100).toFixed(1)}%`;
const day = (value?: string | null) => value ? value.slice(0, 10) : "—";
const fmt = (value: unknown, digits = 3) =>
  typeof value === "number" ? value.toFixed(digits) : "—";
const statusLabel: Record<string, string> = {
  scheduled: "예정", completed: "완료", postponed: "연기", cancelled: "취소",
  queued: "대기", running: "진행 중", completed_job: "완료", failed: "실패",
  candidate: "후보", active: "운영 중", superseded: "이전 운영",
};

function ErrorBox({ message }: { message: string }) {
  return <div className="alert" role="alert">{message}</div>;
}

function Loading({ text = "데이터를 불러오는 중입니다." }: { text?: string }) {
  return <div className="empty"><strong>{text}</strong><span className="text-muted">잠시만 기다려 주세요.</span></div>;
}

function Empty({ title, text }: { title: string; text: string }) {
  return <div className="empty"><strong>{title}</strong><span>{text}</span></div>;
}

function SelectField({
  label, value, onChange, options,
}: { label: string; value: string; onChange: (value: string) => void; options: string[] }) {
  return (
    <div className="field">
      <label>{label}</label>
      <select className="select" value={value} onChange={(event) => onChange(event.target.value)}>
        {options.map((option) => <option key={option} value={option}>{option}</option>)}
      </select>
    </div>
  );
}

function Card({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return <section className={`card ${className}`}>{children}</section>;
}

function PageTitle({ eyebrow, title, description, aside }: {
  eyebrow: string; title: string; description: string; aside?: React.ReactNode;
}) {
  return (
    <div className="page-title">
      <div><div className="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{description}</p></div>
      {aside}
    </div>
  );
}

function LoginPanel({ onLogin, error, busy }: {
  onLogin: (username: string, password: string) => Promise<void>;
  error: string; busy: boolean;
}) {
  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("");
  return (
    <div className="login-panel">
      <Card>
        <div className="eyebrow">관리자 인증</div>
        <h2>운영 콘솔 로그인</h2>
        <p className="text-muted">경기 결과 변경, 데이터 업로드, 모델 학습은 관리자만 사용할 수 있습니다.</p>
        {error && <ErrorBox message={error} />}
        <form className="stack" onSubmit={(event) => {
          event.preventDefault();
          void onLogin(username, password);
        }}>
          <div className="field"><label htmlFor="username">사용자 이름</label>
            <input id="username" className="input" autoComplete="username" value={username}
              onChange={(event) => setUsername(event.target.value)} /></div>
          <div className="field"><label htmlFor="password">비밀번호</label>
            <input id="password" className="input" type="password" autoComplete="current-password"
              value={password} onChange={(event) => setPassword(event.target.value)} /></div>
          <button className="button button-primary" disabled={busy || !password}>
            {busy ? "확인 중..." : "관리자 로그인"}
          </button>
        </form>
        <div className="footer-note">로그인 세션은 서버 서명 HttpOnly 쿠키로 유지됩니다. HTTPS 배포에서는 Secure 쿠키가 필수입니다.</div>
      </Card>
    </div>
  );
}

function DashboardPage({ league, onLeague }: { league: string; onLeague: (value: string) => void }) {
  const [ranking, setRanking] = useState<Rank[]>([]);
  const [models, setModels] = useState<Model[]>([]);
  const [error, setError] = useState("");
  const [cutoff, setCutoff] = useState("");
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    setLoading(true);
    Promise.all([
      api<{ rows: Rank[]; as_of: string }>(`/elo/${encodeURIComponent(league)}`),
      api<{ models: Model[] }>("/models"),
    ]).then(([elo, modelData]) => {
      setRanking(elo.rows ?? []);
      setCutoff(elo.as_of);
      setModels(modelData.models ?? []);
      setError("");
    }).catch((reason: Error) => setError(reason.message)).finally(() => setLoading(false));
  }, [league]);
  const model = models.find((row) => row.league === league && row.active);
  return (
    <div className="stack">
      <PageTitle eyebrow="PickFoot / overview" title="경기 흐름을 한눈에"
        description="운영 모델, 리그 전력, 데이터 기준일을 확인하는 분석 홈"
        aside={<div style={{ width: 210 }}><SelectField label="리그" value={league} onChange={onLeague} options={LEAGUES} /></div>} />
      {error && <ErrorBox message={error} />}
      {loading ? <Loading /> : (
        <>
          <div className="grid grid-4">
            <Card><span className="metric-label">선택 리그</span><strong className="metric-value" style={{ fontSize: 20 }}>{league}</strong><div className="metric-note">유럽 5대 리그</div></Card>
            <Card><span className="metric-label">모델 학습 기준일</span><strong className="metric-value" style={{ fontSize: 22 }}>{day(model?.training_cutoff)}</strong><div className="metric-note">{model?.version ?? "운영 모델 없음"}</div></Card>
            <Card><span className="metric-label">Elo 팀 수</span><strong className="metric-value">{ranking.length}</strong><div className="metric-note">현재 상대 전력 목록</div></Card>
            <Card><span className="metric-label">전력 데이터 기준일</span><strong className="metric-value" style={{ fontSize: 22 }}>{day(cutoff)}</strong><div className="metric-note">모델 기준일과 다를 수 있습니다</div></Card>
          </div>
          <div className="grid grid-2">
            <Card>
              <h2>리그 Elo 상위 팀</h2>
              {ranking.length ? <div className="table-wrap"><table><thead><tr><th>순위</th><th>팀</th><th className="text-right">Elo</th><th className="text-right">최근 5경기</th></tr></thead>
                <tbody>{ranking.slice(0, 6).map((row) => <tr key={row.team}><td>{row.rank}</td><td><strong>{row.team}</strong></td><td className="text-right">{row.elo.toFixed(1)}</td><td className="text-right">{row.recent_points}/{row.recent_matches * 3}</td></tr>)}</tbody></table></div>
                : <Empty title="Elo 데이터 없음" text="허가된 경기 데이터 또는 팀 결과가 저장되면 순위가 표시됩니다." />}
            </Card>
            <Card>
              <h2>운영 모델 평가</h2>
              {model ? <div className="grid grid-3">
                <div><span className="metric-label">Accuracy</span><strong className="metric-value">{fmt(model.evaluation.accuracy)}</strong></div>
                <div><span className="metric-label">Log Loss</span><strong className="metric-value">{fmt(model.evaluation.log_loss)}</strong></div>
                <div><span className="metric-label">Brier</span><strong className="metric-value">{fmt(model.evaluation.brier_score)}</strong></div>
                <div className="notice" style={{ gridColumn: "1 / -1" }}>검증 지표는 과거 시간순 평가의 추정치이며 향후 결과를 보장하지 않습니다. H/D/A 성능은 빈도 baseline과 함께 해석하세요.</div>
              </div> : <Empty title="운영 모델이 아직 없습니다" text="관리자가 승인된 데이터셋으로 후보 모델을 학습하고 적용해야 합니다." />}
            </Card>
          </div>
          <Card><h2>PickFoot 운영 상태</h2><div className="notice">경기와 결과는 PostgreSQL에 저장되고, Elo는 확정 결과에서 다시 계산됩니다. 후보 모델 학습은 별도 worker에서 실행되므로 웹 요청이 장시간 대기하지 않습니다. 학습 데이터 기준일과 팀 상태 기준일은 각각 표시됩니다.</div></Card>
        </>
      )}
    </div>
  );
}

function Probabilities({ values }: { values: Record<"H" | "D" | "A", number> }) {
  const labels = [["H", "홈 승"], ["D", "무승부"], ["A", "원정 승"]] as const;
  const total = Object.values(values).reduce((sum, value) => sum + value, 0);
  return (
    <div className="prob-grid">
      {labels.map(([key, label]) => (
        <div className="prob" key={key}>
          <small>{label} · {key}</small><strong>{pct(values[key])}</strong>
          <div className="bar-track"><div className="bar-fill" style={{ width: `${Math.min(100, Math.max(0, values[key] * 100))}%` }} /></div>
        </div>
      ))}
      <div className="footer-note" style={{ gridColumn: "1 / -1" }}>합계 {pct(total)} · 검증이 필요한 모델 추정 확률이며 확정 승률이 아닙니다.</div>
    </div>
  );
}

function PredictionPage({ league, onLeague, auth }: {
  league: string; onLeague: (value: string) => void; auth: Auth;
}) {
  const [teams, setTeams] = useState<string[]>([]);
  const [models, setModels] = useState<Model[]>([]);
  const [home, setHome] = useState("");
  const [away, setAway] = useState("");
  const [matchDate, setMatchDate] = useState(new Date().toISOString().slice(0, 10));
  const [candidateId, setCandidateId] = useState("");
  const [result, setResult] = useState<PredictionOutput | null>(null);
  const [comparison, setComparison] = useState<PredictionComparison | null>(null);
  const [history, setHistory] = useState<Prediction[]>([]);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const candidates = models.filter((row) => row.league === league && row.id && row.candidate_status === "candidate");
  useEffect(() => {
    setResult(null); setComparison(null); setError("");
    Promise.all([
      api<{ rows: Rank[] }>(`/elo/${encodeURIComponent(league)}`),
      api<{ models: Model[] }>("/models"),
    ]).then(([elo, modelData]) => {
      const names = (elo.rows ?? []).map((item) => item.team).sort((a, b) => a.localeCompare(b));
      setTeams(names);
      setHome(names[0] ?? "");
      setAway(names.find((name) => name !== names[0]) ?? "");
      setModels(modelData.models ?? []);
    }).catch((reason: Error) => setError(reason.message));
  }, [league]);
  useEffect(() => {
    if (!auth.authenticated) return;
    api<{ predictions: Prediction[] }>(`/predictions?league=${encodeURIComponent(league)}`)
      .then((data) => setHistory(data.predictions ?? []))
      .catch(() => setHistory([]));
  }, [auth.authenticated, league, result]);
  const evaluate = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true); setError(""); setComparison(null);
    try {
      const prediction = await api<PredictionOutput>("/predictions", {
        method: "POST",
        body: jsonBody({ league, home_team: home, away_team: away, match_date: matchDate }),
      });
      setResult(prediction);
    } catch (reason) { setError((reason as Error).message); }
    finally { setBusy(false); }
  };
  const compare = async () => {
    setBusy(true); setError("");
    try {
      const output = await api<PredictionComparison>(
        `/predictions/compare?candidate_version_id=${encodeURIComponent(candidateId)}`,
        { method: "POST", body: jsonBody({ league, home_team: home, away_team: away, match_date: matchDate }) },
      );
      setComparison(output);
    } catch (reason) { setError((reason as Error).message); }
    finally { setBusy(false); }
  };
  return (
    <div className="stack">
      <PageTitle eyebrow="PRE_MATCH" title="승무패 예측" description="경기 결과를 보장하지 않는, 검증이 필요한 사전 확률 추정치" />
      <Card>
        <form className="form-grid" onSubmit={(event) => void evaluate(event)}>
          <SelectField label="리그" value={league} onChange={onLeague} options={LEAGUES} />
          <SelectField label="홈팀" value={home} onChange={setHome} options={teams} />
          <SelectField label="원정팀" value={away} onChange={setAway} options={teams.filter((team) => team !== home)} />
          <div className="field"><label htmlFor="predict-date">경기 날짜</label>
            <input id="predict-date" className="input" type="date" value={matchDate} onChange={(event) => setMatchDate(event.target.value)} /></div>
          <div className="inline" style={{ gridColumn: "span 2" }}>
            <button className="button button-primary" disabled={busy || teams.length < 2 || !home || !away || home === away}>
              {busy ? "모델 계산 중..." : "운영 모델로 예측"}
            </button>
            <span className="badge">PRE_MATCH · 선발 라인업 미사용</span>
          </div>
        </form>
      </Card>
      {error && <ErrorBox message={error} />}
      {result && <>
        <div className="notice notice-warn">데이터 기준일 이후 경기를 선택해야 합니다. 역사적 경기의 과거 시점 점수는 expanding-window 백테스트에서만 산출합니다. 현재 상태를 과거 예측에 소급 적용하지 않습니다.</div>
        <Card>
          <div className="inline" style={{ justifyContent: "space-between", marginBottom: 18 }}>
            <div><h2 style={{ marginBottom: 6 }}>{result.home_team} <span className="text-muted">vs</span> {result.away_team}</h2>
              <span className="text-muted">{day(result.match_date)} · 모델 {result.model_version}</span></div>
            <span className="badge badge-green">운영 모델</span>
          </div>
          <Probabilities values={result.probabilities} />
          <div className="grid grid-4" style={{ marginTop: 20 }}>
            <div><span className="metric-label">{result.home_team} Elo</span><strong className="metric-value">{fmt(result.home_elo, 1)}</strong></div>
            <div><span className="metric-label">{result.away_team} Elo</span><strong className="metric-value">{fmt(result.away_elo, 1)}</strong></div>
            <div><span className="metric-label">모델 학습 기준일</span><strong className="metric-value" style={{ fontSize: 19 }}>{day(result.model_training_cutoff)}</strong></div>
            <div><span className="metric-label">팀 상태 기준일</span><strong className="metric-value" style={{ fontSize: 19 }}>{day(result.team_state_cutoff)}</strong></div>
          </div>
          <div className="grid grid-2" style={{ marginTop: 20 }}>
            <div className="notice"><strong>{result.home_team} 최근 5경기</strong><br />승점 {result.home_form.recent_points} · 승률 {pct(result.home_form.recent_win_rate)} · 득점 {fmt(result.home_form.recent_goals_for_avg, 2)} · 실점 {fmt(result.home_form.recent_goals_against_avg, 2)}</div>
            <div className="notice"><strong>{result.away_team} 최근 5경기</strong><br />승점 {result.away_form.recent_points} · 승률 {pct(result.away_form.recent_win_rate)} · 득점 {fmt(result.away_form.recent_goals_for_avg, 2)} · 실점 {fmt(result.away_form.recent_goals_against_avg, 2)}</div>
          </div>
          <div className="footer-note">Accuracy {fmt(result.evaluation.accuracy)} · Log Loss {fmt(result.evaluation.log_loss)} · Brier {fmt(result.evaluation.brier_score)} · 검증 {result.evaluation.validation_matches}경기. 예측 ID {result.prediction_id} · {result.note}</div>
        </Card>
        {candidates.length > 0 && <Card>
          <h2>운영 모델과 후보 모델 입력 비교</h2>
          <div className="inline">
            <select className="select" style={{ maxWidth: 430 }} value={candidateId} onChange={(event) => setCandidateId(event.target.value)}>
              <option value="">후보 모델 선택</option>
              {candidates.map((model) => <option key={model.id} value={model.id!}>{model.version} · cutoff {day(model.training_cutoff)}</option>)}
            </select>
            <button className="button" disabled={!candidateId || busy} onClick={() => void compare()}>동일 입력으로 비교</button>
          </div>
          {comparison && <div className="grid grid-2" style={{ marginTop: 14 }}>
            <div className="notice">운영 · {comparison.operating.version}<br />H {pct(comparison.operating.probabilities.H)} / D {pct(comparison.operating.probabilities.D)} / A {pct(comparison.operating.probabilities.A)}</div>
            <div className="notice">후보 · {comparison.candidate.version}<br />H {pct(comparison.candidate.probabilities.H)} / D {pct(comparison.candidate.probabilities.D)} / A {pct(comparison.candidate.probabilities.A)}</div>
            <div className="footer-note" style={{ gridColumn: "1 / -1" }}>{comparison.warning}</div>
          </div>}
        </Card>}
      </>}
      {auth.authenticated && <Card><h2>저장된 예측 기록</h2>
        {history.length ? <div className="table-wrap"><table><thead><tr><th>경기일</th><th>매치업</th><th>H/D/A</th><th>실제</th><th>검증 상태</th></tr></thead><tbody>
          {history.slice(0, 8).map((row) => <tr key={row.id}><td>{day(row.match_date)}</td><td>{row.home_team} – {row.away_team}</td><td>{pct(row.probabilities.H)} / {pct(row.probabilities.D)} / {pct(row.probabilities.A)}</td><td>{row.actual_result ?? "—"}</td><td>{row.evaluation_status}</td></tr>)}
        </tbody></table></div> : <Empty title="저장된 예측 없음" text="예측을 실행하면 관리 기록에 남습니다." />}
      </Card>}
    </div>
  );
}

function EloPage({ league, onLeague, teamMode = false }: {
  league: string; onLeague: (value: string) => void; teamMode?: boolean;
}) {
  const [rows, setRows] = useState<Rank[]>([]);
  const [history, setHistory] = useState<EloEvent[]>([]);
  const [selected, setSelected] = useState("");
  const [cutoff, setCutoff] = useState("");
  const [lastUpdated, setLastUpdated] = useState("");
  const [search, setSearch] = useState("");
  const [sortBy, setSortBy] = useState<"rank" | "team" | "elo" | "points">("rank");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const load = useCallback(async () => {
    setLoading(true);
    try {
      const data = await api<{ rows: Rank[]; elo_history: EloEvent[]; as_of: string; last_updated_at: string }>(`/elo/${encodeURIComponent(league)}`);
      setRows(data.rows ?? []); setHistory(data.elo_history ?? []); setCutoff(data.as_of ?? "");
      setLastUpdated(data.last_updated_at ?? ""); setError("");
      if (!selected || !(data.rows ?? []).some((row) => row.team === selected)) setSelected(data.rows?.[0]?.team ?? "");
    } catch (reason) { setError((reason as Error).message); }
    finally { setLoading(false); }
  }, [league, selected]);
  useEffect(() => { void load(); }, [load]);
  const filtered = useMemo(() => {
    const result = rows.filter((row) => row.team.toLocaleLowerCase().includes(search.toLocaleLowerCase()));
    return result.sort((a, b) => sortBy === "team" ? a.team.localeCompare(b.team)
      : sortBy === "points" ? b.recent_points - a.recent_points || a.team.localeCompare(b.team)
      : sortBy === "elo" ? b.elo - a.elo || a.team.localeCompare(b.team)
      : a.rank - b.rank || a.team.localeCompare(b.team));
  }, [rows, search, sortBy]);
  const selectedHistory = useMemo(() => history.filter((row) =>
    row.home_team === selected || row.away_team === selected), [history, selected]);
  const historyForChart = selectedHistory.slice(-24);
  const low = historyForChart.length ? Math.min(...historyForChart.map((row) => row.home_team === selected ? row.home_elo_after : row.away_elo_after)) - 20 : 1400;
  const high = historyForChart.length ? Math.max(...historyForChart.map((row) => row.home_team === selected ? row.home_elo_after : row.away_elo_after)) + 20 : 1600;
  const points = historyForChart.map((row, index) => {
    const rating = row.home_team === selected ? row.home_elo_after : row.away_elo_after;
    return `${historyForChart.length < 2 ? 8 : 8 + index * 84 / (historyForChart.length - 1)},${190 - (rating - low) / Math.max(high - low, 1) * 160}`;
  }).join(" ");
  const selectedTeam = rows.find((row) => row.team === selected);
  return (
    <div className="stack">
      <PageTitle eyebrow={teamMode ? "Team profile" : "League ratings"} title={teamMode ? "팀 분석" : "Elo 순위"}
        description="리그별 경기 결과를 재생한 상대 전력 지표 · 초기 Elo 1500 · K=20 · 홈 보정 +60"
        aside={<div style={{ width: 210 }}><SelectField label="리그" value={league} onChange={onLeague} options={LEAGUES} /></div>} />
      <div className="notice">Elo는 승무패 확률이 아니라 리그 결과를 기반으로 한 상대적 전력 지표입니다. 기준일은 {day(cutoff)}이며, 마지막 데이터 갱신 정보는 {lastUpdated ? new Date(lastUpdated).toLocaleString("ko-KR") : "기록 없음"}입니다. 동일 Elo는 동순위이며 이름순으로 정렬합니다.</div>
      {error && <ErrorBox message={error} />}
      {loading ? <Loading /> : rows.length === 0 ? <Card><Empty title="표시할 경기 데이터가 없습니다" text="관리자가 허가된 CSV 데이터 또는 경기 결과를 저장하면 Elo가 계산됩니다." /></Card> : (
        <>
          <div className="toolbar">
            <input className="input" style={{ maxWidth: 280 }} placeholder="팀 검색" value={search} onChange={(event) => setSearch(event.target.value)} />
            <select className="select" style={{ width: 180 }} value={sortBy} onChange={(event) => setSortBy(event.target.value as typeof sortBy)}>
              <option value="rank">Elo 순위</option><option value="elo">Elo 높은 순</option>
              <option value="points">최근 5경기 승점</option><option value="team">팀 이름</option>
            </select>
            <span className="badge">{rows.length} teams · {day(cutoff)}</span>
          </div>
          <div className="grid grid-2">
            <Card>
              <h2>{league} Elo 순위</h2>
              <div className="table-wrap"><table><thead><tr><th>#</th><th>팀</th><th className="text-right">Elo</th><th className="text-right">시즌 경기</th><th className="text-right">최근 승점</th><th>마지막 경기</th></tr></thead><tbody>
                {filtered.map((row) => <tr key={row.team} onClick={() => setSelected(row.team)} style={{ cursor: "pointer", background: selected === row.team ? "#f1f8f4" : undefined }}>
                  <td>{row.rank}</td><td><strong>{row.team}</strong></td><td className="text-right">{row.elo.toFixed(1)}</td><td className="text-right">{row.season_matches}</td><td className="text-right">{row.recent_points}/{row.recent_matches * 3}</td><td>{day(row.last_match_date)}</td>
                </tr>)}
              </tbody></table></div>
            </Card>
            <Card>
              <h2>{selected || "팀"} Elo 흐름</h2>
              {selectedTeam ? <>
                <div className="grid grid-3">
                  <div><span className="metric-label">현재 Elo</span><strong className="metric-value">{selectedTeam.elo.toFixed(1)}</strong></div>
                  <div><span className="metric-label">최근 승점</span><strong className="metric-value">{selectedTeam.recent_points}/{selectedTeam.recent_matches * 3}</strong></div>
                  <div><span className="metric-label">시즌 경기</span><strong className="metric-value">{selectedTeam.season_matches}</strong></div>
                </div>
                {historyForChart.length > 1 ? <svg className="team-history" viewBox="0 0 100 205" preserveAspectRatio="none" role="img" aria-label={`${selected} Elo 추세`}>
                  <polyline points={points} fill="none" stroke="#087f5b" strokeWidth="2.6" vectorEffect="non-scaling-stroke" />
                  {historyForChart.map((row, index) => {
                    const rating = row.home_team === selected ? row.home_elo_after : row.away_elo_after;
                    const x = historyForChart.length < 2 ? 8 : 8 + index * 84 / (historyForChart.length - 1);
                    const y = 190 - (rating - low) / Math.max(high - low, 1) * 160;
                    return <circle key={`${row.match_date}-${row.home_team}`} cx={x} cy={y} r="1.8" fill="#087f5b"><title>{day(row.match_date)} · {rating.toFixed(1)}</title></circle>;
                  })}
                </svg> : <Empty title="이력 부족" text="팀의 완료된 경기 결과가 반영되면 그래프를 그립니다." />}
                <div className="table-wrap"><table><thead><tr><th>날짜</th><th>상대</th><th>결과</th><th className="text-right">Elo 변화</th><th className="text-right">경기 후 Elo</th></tr></thead><tbody>
                  {selectedHistory.slice(-8).reverse().map((row) => {
                    const home = row.home_team === selected;
                    const delta = home ? row.home_elo_change : row.away_elo_change;
                    const rating = home ? row.home_elo_after : row.away_elo_after;
                    return <tr key={`${row.match_date}-${row.home_team}`}><td>{day(row.match_date)}</td><td>{home ? row.away_team : row.home_team}</td><td>{home ? row.result : ({ H: "A", A: "H", D: "D" } as Record<string, string>)[row.result]}</td><td className={`text-right ${delta >= 0 ? "positive" : "negative"}`}>{delta >= 0 ? "+" : ""}{delta.toFixed(1)}</td><td className="text-right">{rating.toFixed(1)}</td></tr>;
                  })}
                </tbody></table></div>
              </> : <Empty title="신규 팀 또는 기록 없음" text="초기 Elo로 관리되며 실제 결과가 저장되기 전까지 경기 이력을 만들지 않습니다." />}
            </Card>
          </div>
        </>
      )}
    </div>
  );
}

function MatchAnalysisPage({ league, onLeague }: { league: string; onLeague: (value: string) => void }) {
  const [events, setEvents] = useState<EloEvent[]>([]);
  const [error, setError] = useState("");
  const [team, setTeam] = useState("");
  useEffect(() => {
    api<{ elo_history: EloEvent[] }>(`/elo/${encodeURIComponent(league)}`)
      .then((data) => { setEvents(data.elo_history ?? []); setError(""); })
      .catch((reason: Error) => setError(reason.message));
  }, [league]);
  const teams = useMemo(() => [...new Set(events.flatMap((row) => [row.home_team, row.away_team]))].sort(), [events]);
  const recent = events.slice(-100).reverse();
  const movement = useMemo(() => {
    const totals = new Map<string, number>();
    for (const row of events.slice(-5)) {
      totals.set(row.home_team, (totals.get(row.home_team) ?? 0) + row.home_elo_change);
      totals.set(row.away_team, (totals.get(row.away_team) ?? 0) + row.away_elo_change);
    }
    return [...totals.entries()].sort((a, b) => b[1] - a[1]);
  }, [events]);
  return (
    <div className="stack">
      <PageTitle eyebrow="Match impact" title="경기별 Elo 분석" description="경기 전 기대 score와 실제 결과에서 계산한 양 팀 Elo 변화"
        aside={<div style={{ width: 210 }}><SelectField label="리그" value={league} onChange={onLeague} options={LEAGUES} /></div>} />
      <div className="notice">Elo 기대 score는 H/D/A 확률이 아닙니다. 경기 전 Elo, 홈 보정 +60, 실제 결과로 Δ홈 = 20 × (실제 score − 기대 score)를 계산합니다. 같은 날짜 경기는 같은 경기일 시작 rating을 사용합니다.</div>
      {error && <ErrorBox message={error} />}
      {!events.length ? <Card><Empty title="완료된 경기 결과가 없습니다" text="결과를 저장한 뒤 경기별 변화를 분석할 수 있습니다." /></Card> : <>
        <div className="grid grid-2">
          <Card><h2>최근 경기별 Elo 변화</h2>
            <div className="table-wrap"><table><thead><tr><th>날짜</th><th>매치업 / 결과</th><th className="text-right">홈 Δ</th><th className="text-right">원정 Δ</th><th className="text-right">기대 score</th></tr></thead><tbody>
              {recent.slice(0, 20).map((row) => <tr key={`${row.match_date}-${row.home_team}`}><td>{day(row.match_date)}</td><td>{row.home_team} {row.home_goals}–{row.away_goals} {row.away_team}<br /><span className="text-muted">{row.result}</span></td><td className={`text-right ${row.home_elo_change >= 0 ? "positive" : "negative"}`}>{row.home_elo_change >= 0 ? "+" : ""}{row.home_elo_change.toFixed(1)}</td><td className={`text-right ${row.away_elo_change >= 0 ? "positive" : "negative"}`}>{row.away_elo_change >= 0 ? "+" : ""}{row.away_elo_change.toFixed(1)}</td><td className="text-right">{row.expected_home_score.toFixed(3)}</td></tr>)}
            </tbody></table></div>
          </Card>
          <Card>
            <h2>최근 5경기 Elo 상승 / 하락</h2>
            {movement.length ? movement.slice(0, 14).map(([name, delta]) => <div key={name} style={{ marginBottom: 13 }}>
              <div className="inline" style={{ justifyContent: "space-between", marginBottom: 5 }}><span>{name}</span><strong className={delta >= 0 ? "positive" : "negative"}>{delta >= 0 ? "+" : ""}{delta.toFixed(1)}</strong></div>
              <div className="bar-track"><div className={`bar-fill ${delta < 0 ? "down" : ""}`} style={{ width: `${Math.max(2, Math.min(100, Math.abs(delta) * 3))}%` }} /></div>
            </div>) : <Empty title="데이터 없음" text="경기 결과를 추가하면 집계됩니다." />}
            <div className="field" style={{ marginTop: 22 }}><label>팀 경기 전후 상세</label>
              <select className="select" value={team} onChange={(event) => setTeam(event.target.value)}><option value="">팀 선택</option>{teams.map((name) => <option key={name}>{name}</option>)}</select></div>
            {team && <div className="table-wrap" style={{ marginTop: 12 }}><table><thead><tr><th>경기일</th><th>상대팀</th><th>결과</th><th>경기 전→후</th></tr></thead><tbody>
              {recent.filter((row) => row.home_team === team || row.away_team === team).slice(0, 8).map((row) => {
                const isHome = row.home_team === team;
                return <tr key={`${row.match_date}-${row.home_team}`}><td>{day(row.match_date)}</td><td>{isHome ? row.away_team : row.home_team}</td><td>{isHome ? row.result : ({ H: "A", A: "H", D: "D" } as Record<string, string>)[row.result]}</td><td>{(isHome ? row.home_elo_before : row.away_elo_before).toFixed(0)} → {(isHome ? row.home_elo_after : row.away_elo_after).toFixed(0)}</td></tr>;
              })}
            </tbody></table></div>}
          </Card>
        </div>
      </>}
    </div>
  );
}

function MatchManagementPage({ league, onLeague, auth }: {
  league: string; onLeague: (value: string) => void; auth: Auth;
}) {
  const [rounds, setRounds] = useState<string[]>([]);
  const [round, setRound] = useState("");
  const [matches, setMatches] = useState<Match[]>([]);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);
  const [preview, setPreview] = useState<Record<string, { home: string; away: string }>>({});
  const [homeGoals, setHomeGoals] = useState<Record<string, string>>({});
  const [awayGoals, setAwayGoals] = useState<Record<string, string>>({});
  const [source, setSource] = useState("");
  const [license, setLicense] = useState("");
  const [rights, setRights] = useState(false);
  const [newFixture, setNewFixture] = useState({ round: "1라운드", date: "2026-08-15", home: "", away: "" });
  const [teamName, setTeamName] = useState("");
  const [teamReason, setTeamReason] = useState("");
  const [statusReason, setStatusReason] = useState("일정 상태를 관리자가 확인했습니다.");
  const [correctionId, setCorrectionId] = useState("");

  const refresh = useCallback(async (selectedRound = round) => {
    setBusy(true);
    try {
      const [roundData, matchData] = await Promise.all([
        api<{ rounds: string[] }>(`/rounds?league=${encodeURIComponent(league)}&season=2026-27`),
        api<{ matches: Match[] }>(`/matches?league=${encodeURIComponent(league)}&season=2026-27${selectedRound ? `&round_name=${encodeURIComponent(selectedRound)}` : ""}`),
      ]);
      setRounds(roundData.rounds ?? []);
      if (!selectedRound && roundData.rounds?.length) setRound(roundData.rounds[0]);
      setMatches(matchData.matches ?? []);
      setError("");
    } catch (reason) { setError((reason as Error).message); }
    finally { setBusy(false); }
  }, [league, round]);
  useEffect(() => { void refresh(round); }, [refresh, round]);
  const createFixture = async (event: FormEvent) => {
    event.preventDefault(); setBusy(true); setError(""); setNotice("");
    try {
      await api("/admin/matches", { method: "POST", body: jsonBody({
        league, season: "2026-27", round_name: newFixture.round,
        match_date: newFixture.date, home_team: newFixture.home, away_team: newFixture.away,
      }) });
      setNotice("일정이 서버에 등록되었습니다. 점수 확정 전에는 Elo에 반영되지 않습니다.");
      setRound(newFixture.round); await refresh(newFixture.round);
    } catch (reason) { setError((reason as Error).message); }
    finally { setBusy(false); }
  };
  const registerTeam = async (event: FormEvent) => {
    event.preventDefault(); setBusy(true); setError(""); setNotice("");
    try {
      const response = await api<Record<string, string>>("/admin/teams", { method: "POST", body: jsonBody({
        league, team_name: teamName, registration_note: teamReason,
      }) });
      setNotice(`${response.team_name} 등록 완료 · ${response.elo_policy}`);
      setTeamName(""); setTeamReason("");
    } catch (reason) { setError((reason as Error).message); }
    finally { setBusy(false); }
  };
  const confirmResult = async (match: Match) => {
    const home = homeGoals[match.id] ?? (match.home_goals === null ? "" : String(match.home_goals));
    const away = awayGoals[match.id] ?? (match.away_goals === null ? "" : String(match.away_goals));
    const homeNumber = Number(home); const awayNumber = Number(away);
    if (!Number.isInteger(homeNumber) || !Number.isInteger(awayNumber) || homeNumber < 0 || awayNumber < 0) {
      setError("양 팀 점수는 0 이상의 정수로 입력하세요."); return;
    }
    const current = preview[match.id];
    if (!current || current.home !== home || current.away !== away) {
      setPreview({ ...preview, [match.id]: { home, away } }); setError("");
      setNotice(`${match.home_team} ${home}–${away} ${match.away_team} 결과를 미리 확인한 뒤 다시 확정 버튼을 누르세요.`);
      return;
    }
    if (!rights || !source.trim() || !license.trim()) { setError("출처, 이용 권한 참조를 입력하고 확인 체크를 해야 합니다."); return; }
    if (match.status === "completed" && correctionId !== match.id) {
      setCorrectionId(match.id);
      setError("이미 확정된 결과입니다. 정정 사유를 확인한 뒤 다시 눌러 명시적으로 승인하세요.");
      return;
    }
    setBusy(true); setError(""); setNotice("");
    try {
      const response = await api<ResultConfirmation>(`/admin/matches/${match.id}/result`, {
        method: "PUT",
        body: jsonBody({
          home_goals: homeNumber, away_goals: awayNumber, expected_version: match.version,
          source_name: source, license_reference: license, rights_confirmed: rights,
          correction_confirmed: correctionId === match.id,
          reason: correctionId === match.id ? "관리자 화면에서 점수 정정 확인" : "관리자 화면에서 최종 점수 확인",
        }),
      });
      setNotice(response.elo?.home_after !== undefined
        ? `저장 완료 · 홈 Elo ${Number(response.elo.home_after).toFixed(1)} (${Number(response.elo.home_change) >= 0 ? "+" : ""}${Number(response.elo.home_change).toFixed(1)}) · 원정 Elo ${Number(response.elo.away_after).toFixed(1)}`
        : "경기 결과가 영구 저장되었습니다.");
      setPreview((current) => {
        const updated = { ...current };
        delete updated[match.id];
        return updated;
      });
      setCorrectionId("");
      await refresh(round);
    } catch (reason) { setError((reason as Error).message); }
    finally { setBusy(false); }
  };
  const setFixtureStatus = async (match: Match, status: "postponed" | "cancelled") => {
    if (!window.confirm(`${match.home_team} – ${match.away_team} 경기를 ${status === "postponed" ? "연기" : "취소"} 처리할까요?`)) return;
    setBusy(true);
    try {
      await api(`/admin/matches/${match.id}/status`, {
        method: "PATCH",
        body: jsonBody({ status, expected_version: match.version, reason: statusReason }),
      });
      setNotice("경기 상태가 저장되었습니다."); await refresh(round);
    } catch (reason) { setError((reason as Error).message); }
    finally { setBusy(false); }
  };
  if (!auth.authenticated) return <Empty title="관리자 로그인이 필요합니다" text="경기 일정과 점수 변경은 인증된 관리자만 처리합니다." />;
  return (
    <div className="stack">
      <PageTitle eyebrow="2026 / 27 season" title="경기 관리" description="라운드 일정을 등록하고, 출처 확인 후 결과를 영구 저장합니다." aside={<div style={{ width: 210 }}><SelectField label="리그" value={league} onChange={onLeague} options={LEAGUES} /></div>} />
      {notice && <div className="success">{notice}</div>}{error && <ErrorBox message={error} />}
      <Card><h2>일정 등록</h2><form className="form-grid" onSubmit={(event) => void createFixture(event)}>
        <div className="field"><label>라운드</label><input className="input" value={newFixture.round} onChange={(event) => setNewFixture({ ...newFixture, round: event.target.value })} required /></div>
        <div className="field"><label>경기 날짜</label><input className="input" type="date" value={newFixture.date} onChange={(event) => setNewFixture({ ...newFixture, date: event.target.value })} required /></div>
        <div className="field"><label>홈팀</label><input className="input" value={newFixture.home} onChange={(event) => setNewFixture({ ...newFixture, home: event.target.value })} required /></div>
        <div className="field"><label>원정팀</label><input className="input" value={newFixture.away} onChange={(event) => setNewFixture({ ...newFixture, away: event.target.value })} required /></div>
        <button className="button button-primary" disabled={busy}>일정 등록</button>
      </form></Card>
      <Card><div className="toolbar" style={{ justifyContent: "space-between" }}>
        <h2 style={{ margin: 0 }}>라운드 결과</h2>
        <div className="inline">
          <div style={{ minWidth: 210 }}><SelectField label="라운드 선택" value={round} onChange={setRound} options={rounds.length ? rounds : [""]} /></div>
          <button className="button" onClick={() => void refresh(round)} disabled={busy}>새로고침</button>
        </div>
      </div>
      <div className="field" style={{ margin: "12px 0" }}><label>결과 데이터 출처 및 권한 기록</label>
        <div className="grid grid-2"><input className="input" placeholder="출처명" value={source} onChange={(event) => setSource(event.target.value)} />
          <input className="input" placeholder="라이선스 또는 이용 권한 확인 참조" value={license} onChange={(event) => setLicense(event.target.value)} /></div>
        <label className="checkbox"><input type="checkbox" checked={rights} onChange={(event) => setRights(event.target.checked)} />경기 데이터의 출처와 로컬 저장·모델 학습 이용 권한을 확인했습니다.</label>
        <input className="input" aria-label="경기 상태 변경 사유" value={statusReason} onChange={(event) => setStatusReason(event.target.value)} placeholder="연기·취소 사유" />
      </div>
      {matches.length ? <div className="table-wrap"><table><thead><tr><th>날짜</th><th>매치업</th><th>상태</th><th>점수</th><th>결과</th><th>관리</th></tr></thead><tbody>
        {matches.map((match) => <tr key={match.id}>
          <td>{day(match.match_date)}</td><td><strong>{match.home_team}</strong> – {match.away_team}</td>
          <td><span className={`badge ${match.status === "completed" ? "badge-green" : match.status === "postponed" ? "badge-amber" : ""}`}>{statusLabel[match.status] ?? match.status}</span></td>
          <td><div className="inline"><input aria-label={`${match.home_team} 점수`} className="input" style={{ width: 64 }} inputMode="numeric" value={homeGoals[match.id] ?? (match.home_goals === null ? "" : String(match.home_goals))} onChange={(event) => setHomeGoals({ ...homeGoals, [match.id]: event.target.value })} />
            <span>:</span><input aria-label={`${match.away_team} 점수`} className="input" style={{ width: 64 }} inputMode="numeric" value={awayGoals[match.id] ?? (match.away_goals === null ? "" : String(match.away_goals))} onChange={(event) => setAwayGoals({ ...awayGoals, [match.id]: event.target.value })} /></div></td>
          <td>{match.result ?? "—"}</td><td><div className="inline">
            <button className="button button-primary" disabled={busy || match.status === "cancelled"} onClick={() => void confirmResult(match)}>{match.status === "completed" ? "결과 정정" : "미리보기 / 확정"}</button>
            {match.status === "scheduled" && <><button className="button" disabled={busy} onClick={() => void setFixtureStatus(match, "postponed")}>연기</button><button className="button button-danger" disabled={busy} onClick={() => void setFixtureStatus(match, "cancelled")}>취소</button></>}
          </div></td>
        </tr>)}
      </tbody></table></div> : <Empty title="선택 라운드에 일정이 없습니다" text="상단 양식으로 경기를 등록하거나 다른 라운드를 선택하세요." />}
      <div className="footer-note">확정 경기는 PostgreSQL 트랜잭션과 변경 이력에 저장됩니다. 일정/경기 관리는 로그인 사용자만 볼 수 있으며, 결과 저장 뒤 Elo는 기존 엔진으로 다시 계산됩니다.</div>
      </Card>
      <Card><h2>팀 등록 상태</h2>
        <form className="inline" onSubmit={(event) => void registerTeam(event)}>
          <input className="input" style={{ maxWidth: 280 }} placeholder="새 팀 공식 이름" value={teamName} onChange={(event) => setTeamName(event.target.value)} />
          <input className="input" style={{ maxWidth: 360 }} placeholder="승격/참가 확인 근거" value={teamReason} onChange={(event) => setTeamReason(event.target.value)} />
          <button className="button" disabled={!teamName || !teamReason}>팀 추가</button>
        </form>
        <div className="footer-note">등록 팀은 Elo 1500에서 시작하며 경기 이력과 폼은 만들지 않습니다. Logistic Regression 예측 가능 여부와 팀 등록은 별도로 검증됩니다.</div>
      </Card>
    </div>
  );
}

function ModelManagementPage({ league, onLeague, auth }: {
  league: string; onLeague: (value: string) => void; auth: Auth;
}) {
  const [models, setModels] = useState<Model[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [file, setFile] = useState<File | null>(null);
  const [source, setSource] = useState("");
  const [license, setLicense] = useState("");
  const [rights, setRights] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);
  const refresh = useCallback(async () => {
    if (!auth.authenticated) return;
    try {
      const [modelData, jobData, datasetData] = await Promise.all([
        api<{ models: Model[] }>("/models"),
        api<{ jobs: Job[] }>("/training-jobs"),
        api<{ datasets: Dataset[] }>("/datasets"),
      ]);
      setModels(modelData.models); setJobs(jobData.jobs); setDatasets(datasetData.datasets); setError("");
    } catch (reason) { setError((reason as Error).message); }
  }, [auth.authenticated]);
  useEffect(() => { void refresh(); }, [refresh]);
  useEffect(() => {
    if (!jobs.some((job) => job.status === "queued" || job.status === "running")) return;
    const timer = window.setInterval(() => void refresh(), 4000);
    return () => window.clearInterval(timer);
  }, [jobs, refresh]);
  const submitDataset = async (event: FormEvent) => {
    event.preventDefault();
    if (!file || !source.trim() || !license.trim() || !rights) return;
    setBusy(true); setError(""); setNotice("");
    const form = new FormData(); form.set("file", file);
    try {
      const query = new URLSearchParams({ source_name: source, license_reference: license, rights_confirmed: "true" });
      const response = await api<{ row_count: number; file_sha256: string }>(`/admin/datasets?${query.toString()}`, { method: "POST", body: form });
      setNotice(`데이터셋 검증·영구 저장 완료 · ${response.row_count}경기 · SHA-256 ${response.file_sha256}`);
      await refresh();
    } catch (reason) { setError((reason as Error).message); }
    finally { setBusy(false); }
  };
  const startJob = async (kind: "retrain" | "backtest") => {
    setBusy(true); setError(""); setNotice("");
    try {
      const response = await api<JobSubmission>(
        kind === "retrain" ? "/admin/training-jobs" : `/admin/backtests?league=${encodeURIComponent(league)}`,
        { method: "POST", body: kind === "retrain" ? jsonBody({ league }) : undefined },
      );
      setNotice(`${kind === "retrain" ? "후보 재학습" : "재현 백테스트"} 작업 ${response.status}: ${response.job_id}`);
      await refresh();
    } catch (reason) { setError((reason as Error).message); }
    finally { setBusy(false); }
  };
  const promote = async (model: Model) => {
    if (!model.id || !window.confirm(`${model.version} 모델을 운영 모델로 적용할까요?`)) return;
    setBusy(true);
    try {
      await api(`/admin/models/${encodeURIComponent(league)}/promote?version_id=${encodeURIComponent(model.id)}`, { method: "POST" });
      setNotice(`운영 모델을 ${model.version}으로 변경했습니다.`); await refresh();
    } catch (reason) { setError((reason as Error).message); }
    finally { setBusy(false); }
  };
  const rollback = async () => {
    if (!window.confirm(`${league}의 직전 운영 모델로 롤백할까요?`)) return;
    setBusy(true);
    try {
      const response = await api<{ active_version: string }>(`/admin/models/${encodeURIComponent(league)}/rollback`, { method: "POST" });
      setNotice(`이전 모델로 롤백했습니다: ${response.active_version}`); await refresh();
    } catch (reason) { setError((reason as Error).message); }
    finally { setBusy(false); }
  };
  if (!auth.authenticated) return <Empty title="관리자 로그인이 필요합니다" text="데이터 업로드, 학습, 모델 운영 변경은 관리자 전용입니다." />;
  const leagueModels = models.filter((model) => model.league === league);
  return (
    <div className="stack">
      <PageTitle eyebrow="Model operations" title="모델 학습 관리" description="데이터 권한 확인 → 비동기 후보 학습 → 동일 검증창 비교 → 운영 적용 또는 롤백" aside={<div style={{ width: 210 }}><SelectField label="대상 리그" value={league} onChange={onLeague} options={LEAGUES} /></div>} />
      {notice && <div className="success">{notice}</div>}{error && <ErrorBox message={error} />}
      <div className="grid grid-2">
        <Card><h2>승인된 결과 데이터 등록</h2>
          <form className="stack" onSubmit={(event) => void submitDataset(event)}>
            <div className="field"><label>기준 경기 이력 CSV (최대 25 MB, UTF-8/BOM)</label><input className="input" type="file" accept=".csv,text/csv" onChange={(event) => setFile(event.target.files?.[0] ?? null)} /></div>
            <div className="field"><label>데이터 제공자</label><input className="input" value={source} onChange={(event) => setSource(event.target.value)} placeholder="예: OpenFootball" /></div>
            <div className="field"><label>라이선스 / 이용 권한 기록</label><input className="input" value={license} onChange={(event) => setLicense(event.target.value)} placeholder="출처 문서 및 확인 날짜" /></div>
            <label className="checkbox"><input type="checkbox" checked={rights} onChange={(event) => setRights(event.target.checked)} />이 데이터의 서버 저장, 분석 및 모델 학습 권한을 확인했습니다.</label>
            <div className="footer-note">기준 CSV에는 이 화면에서 별도로 확정할 2026/27 결과를 중복 포함하지 마세요. 중복 경기 기록은 분석·학습을 차단합니다.</div>
            <button className="button button-primary" disabled={busy || !file || !source || !license || !rights}>검증하고 데이터셋 활성화</button>
          </form>
        </Card>
        <Card><h2>활성 데이터와 작업자 상태</h2>
          {datasets.length ? datasets.slice(0, 4).map((dataset) => <div key={dataset.id} className="notice" style={{ marginBottom: 10 }}>
            <div className="inline" style={{ justifyContent: "space-between" }}><strong>{dataset.source_name} · {dataset.file_name}</strong><span className={`badge ${dataset.active ? "badge-green" : ""}`}>{dataset.active ? "활성" : "보관"}</span></div>
            <div>{dataset.row_count}경기 · {day(dataset.coverage_start)} – {day(dataset.coverage_end)} · 업로드 {new Date(dataset.uploaded_at).toLocaleString("ko-KR")}</div>
            <div className="footer-note">{Object.entries(dataset.league_coverage ?? {}).map(([name, count]) => `${name} ${count}`).join(" · ")}</div>
          </div>) : <Empty title="등록된 데이터셋이 없습니다" text="학습 전에 CSV와 출처·권한 기록을 등록해야 합니다." />}
          <div className="notice notice-warn">PostgreSQL 메타데이터와 파일은 별도 영속 볼륨에 저장됩니다. 배포 플랫폼에서 두 경로 모두 영속 디스크로 연결해야 재시작 후 보존됩니다.</div>
        </Card>
      </div>
      <Card>
        <div className="toolbar" style={{ justifyContent: "space-between" }}><h2 style={{ margin: 0 }}>리그 후보 모델</h2>
          <div className="inline"><button className="button button-primary" disabled={busy} onClick={() => void startJob("retrain")}>후보 모델 재학습</button><button className="button" disabled={busy} onClick={rollback}>이전 모델로 롤백</button></div>
        </div>
        {leagueModels.length ? <div className="grid grid-2">
          {leagueModels.map((model) => <div className="card" key={model.id ?? model.version} style={{ boxShadow: "none" }}>
            <div className="inline" style={{ justifyContent: "space-between" }}><strong>{model.version}</strong><span className={`badge ${model.active ? "badge-green" : model.candidate_status === "candidate" ? "badge-amber" : ""}`}>{model.active ? "운영 모델" : statusLabel[model.candidate_status] ?? model.candidate_status}</span></div>
            <div className="metric-note">학습 기준일 {day(model.training_cutoff)} · {model.evaluation.validation_matches ?? "—"} 검증 경기</div>
            <div className="grid grid-3" style={{ marginTop: 16 }}>
              <div><span className="metric-label">Accuracy</span><strong className="metric-value" style={{ fontSize: 22 }}>{fmt(model.evaluation.accuracy)}</strong></div>
              <div><span className="metric-label">Log Loss</span><strong className="metric-value" style={{ fontSize: 22 }}>{fmt(model.evaluation.log_loss)}</strong></div>
              <div><span className="metric-label">Brier</span><strong className="metric-value" style={{ fontSize: 22 }}>{fmt(model.evaluation.brier_score)}</strong></div>
            </div>
            {model.comparison && <div className="notice" style={{ marginTop: 13 }}>동일 검증창 {String(model.comparison.matches ?? "—")}경기 · 모델/운영 비교</div>}
            {!model.active && model.id && model.candidate_status === "candidate" && <button className="button button-primary" style={{ marginTop: 14 }} disabled={busy} onClick={() => void promote(model)}>후보를 운영 적용</button>}
          </div>)}
        </div> : <Empty title="등록된 모델 버전이 없습니다" text="활성 데이터셋을 올리고 초기 모델 학습을 시작하세요." />}
        <div className="footer-note">후보 평가는 동일한 날짜 구간으로 운영 모델과 비교합니다. 검증 기간이 운영 모델 학습 기준일보다 뒤에 있지 않으면 비교를 중단합니다. 성능이 나빠져도 자동 승격하지 않습니다.</div>
      </Card>
      <Card><h2>비동기 작업 상태</h2>
        {jobs.length ? <div className="table-wrap"><table><thead><tr><th>작업</th><th>리그</th><th>종류</th><th>상태</th><th>진행 정보</th><th>오류</th></tr></thead><tbody>
          {jobs.slice(0, 12).map((job) => <tr key={job.id}><td><code>{job.id.slice(0, 8)}</code></td><td>{job.league}</td><td>{job.job_type === "backtest" ? "백테스트" : "재학습"}</td><td><span className={`badge ${job.status === "failed" ? "badge-red" : job.status === "completed" ? "badge-green" : "badge-amber"}`}>{statusLabel[job.status] ?? job.status}</span></td><td>{job.progress_message}</td><td>{job.error_message ?? "—"}</td></tr>)}
        </tbody></table></div> : <Empty title="작업 이력 없음" text="재학습 또는 백테스트 요청이 생성되면 여기서 상태를 확인합니다." />}
      </Card>
    </div>
  );
}

function BacktestPage({ league, onLeague, auth }: {
  league: string; onLeague: (value: string) => void; auth: Auth;
}) {
  const [report, setReport] = useState<BacktestReport | null>(null);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);
  const refresh = useCallback(async () => {
    if (!auth.authenticated) return;
    try {
      const [backtestData, predictionData] = await Promise.all([
        api<{ available: boolean; report: BacktestReport | null }>("/backtests"),
        api<{ predictions: Prediction[] }>(`/predictions?league=${encodeURIComponent(league)}`),
      ]);
      setReport(backtestData.available ? backtestData.report : null);
      setPredictions(predictionData.predictions);
      setError("");
    } catch (reason) { setError((reason as Error).message); }
  }, [auth.authenticated, league]);
  useEffect(() => { void refresh(); }, [refresh]);
  const runBacktest = async () => {
    setBusy(true); setError(""); setNotice("");
    try {
      const response = await api<JobSubmission>(`/admin/backtests?league=${encodeURIComponent(league)}`, { method: "POST" });
      setNotice(`시간순 재현 백테스트 작업 ${response.status}: ${response.job_id}`);
      await refresh();
    } catch (reason) { setError((reason as Error).message); }
    finally { setBusy(false); }
  };
  if (!auth.authenticated) return <Empty title="관리자 로그인이 필요합니다" text="예측 이력과 재현 백테스트는 운영자 전용 화면입니다." />;
  const row = report?.leagues[league];
  return (
    <div className="stack">
      <PageTitle eyebrow="Evaluation" title="예측 백테스트" description="실제 저장 예측과 과거 시점의 재현 평가를 섞지 않고 보여줍니다." aside={<div style={{ width: 210 }}><SelectField label="리그" value={league} onChange={onLeague} options={LEAGUES} /></div>} />
      {notice && <div className="success">{notice}</div>}{error && <ErrorBox message={error} />}
      <div className="notice notice-warn">경기 날짜만 저장되어 정확한 킥오프 시각이 확인되지 않으면, 예측 생성 시각이 경기보다 앞섰다고 보장할 수 없습니다. 해당 기록은 검증된 사전 예측 점수에서 제외합니다.</div>
      <div className="grid grid-2">
        <Card><h2>A · 실제 사전 예측 기록</h2>
          {predictions.length ? <div className="table-wrap"><table><thead><tr><th>예측 시각</th><th>경기</th><th>H/D/A</th><th>실제</th><th>상태</th></tr></thead><tbody>
            {predictions.map((item) => <tr key={item.id}><td>{new Date(item.predicted_at).toLocaleString("ko-KR")}</td><td>{item.home_team} – {item.away_team}<br /><span className="text-muted">{day(item.match_date)}</span></td><td>{pct(item.probabilities.H)} / {pct(item.probabilities.D)} / {pct(item.probabilities.A)}</td><td>{item.actual_result ?? "—"}</td><td><span className="badge badge-amber">{item.evaluation_status}</span></td></tr>)}
          </tbody></table></div> : <Empty title="기록된 예측이 없습니다" text="예측 화면에서 예측을 실행하면 이력에 저장됩니다." />}
          <div className="footer-note">현재 데이터에는 kickoff 시각이 없어 verified pre-match Accuracy/Log Loss/Brier 표본은 0일 수 있습니다.</div>
        </Card>
        <Card><div className="inline" style={{ justifyContent: "space-between" }}><h2> B · 과거 시점 재현</h2>
          <button className="button button-primary" disabled={busy} onClick={() => void runBacktest()}>{busy ? "요청 중..." : "시간순 재현 실행"}</button></div>
          {row ? <>
            <div className="notice">방법: {report?.method ?? "expanding-window"} · 평가 기간 {day(row.evaluation_start)} – {day(row.evaluation_end)} · {row.matches} 경기</div>
            <div className="grid grid-3" style={{ marginTop: 14 }}>
              <div><span className="metric-label">Accuracy / baseline</span><strong className="metric-value" style={{ fontSize: 21 }}>{fmt(row.accuracy)} / {fmt(row.baseline_accuracy)}</strong></div>
              <div><span className="metric-label">Log Loss / baseline</span><strong className="metric-value" style={{ fontSize: 21 }}>{fmt(row.log_loss)} / {fmt(row.baseline_log_loss)}</strong></div>
              <div><span className="metric-label">Brier / baseline</span><strong className="metric-value" style={{ fontSize: 21 }}>{fmt(row.brier_score)} / {fmt(row.baseline_brier_score)}</strong></div>
            </div>
            <div className="table-wrap" style={{ marginTop: 17 }}><table><thead><tr><th>지표</th><th>모델</th><th>빈도 baseline</th></tr></thead><tbody>
              {(["accuracy", "log_loss", "brier_score"] as const).map((key) => <tr key={key}><td>{key}</td><td>{fmt(row[key])}</td><td>{fmt(row[`baseline_${key}`])}</td></tr>)}
            </tbody></table></div>
          </> : <Empty title="재현 평가 결과가 없습니다" text="백테스트 작업을 시작하면 worker가 과거 날짜보다 이른 결과만으로 평가합니다." />}
          <div className="footer-note">확장형 시간순 재현은 실제 사전 예측 기록이 아닙니다. 각 예측 시점 이후 결과를 해당 시점의 학습에 사용하지 않습니다. 평가 표본이 적으면 수치 해석에 주의하세요.</div>
        </Card>
      </div>
    </div>
  );
}

export default function Home() {
  const [page, setPage] = useState<PageName>("대시보드");
  const [league, setLeague] = useState(LEAGUES[0]);
  const [auth, setAuth] = useState<Auth>({ authenticated: false, username: "" });
  const [apiReady, setApiReady] = useState<boolean | null>(null);
  const [loginError, setLoginError] = useState("");
  const [loginBusy, setLoginBusy] = useState(false);
  useEffect(() => {
    api<{ status: string }>("/health/live").then(() => setApiReady(true)).catch(() => setApiReady(false));
    api<Auth>("/auth/me").then(setAuth).catch(() => setAuth({ authenticated: false, username: "" }));
  }, []);
  const adminPage = ["2026/27 경기 관리", "모델 학습 관리", "예측 백테스트"].includes(page);
  const onLogin = async (username: string, password: string) => {
    setLoginBusy(true); setLoginError("");
    try {
      const loggedIn = await api<{ username: string }>("/auth/login", { method: "POST", body: jsonBody({ username, password }) });
      setAuth({ authenticated: true, username: loggedIn.username });
      setLoginError("");
    } catch (reason) { setLoginError((reason as Error).message); }
    finally { setLoginBusy(false); }
  };
  const logout = async () => {
    try { await api("/auth/logout", { method: "POST" }); }
    finally { setAuth({ authenticated: false, username: "" }); }
  };
  const goTo = (next: PageName) => {
    setPage(next);
    if (["대시보드", "승무패 예측", "Elo 순위", "팀 분석", "경기별 Elo 분석"].includes(next)) return;
    if (!auth.authenticated) setLoginError("");
  };
  return (
    <main className="shell">
      <header className="topbar">
        <div className="brand"><span className="brand-mark">⚽</span><span>PickFoot<span style={{ color: "#8c9b94", fontWeight: 500 }}> / intelligence</span></span></div>
        <nav className="nav" aria-label="주요 메뉴">
          {PAGES.map((name, index) => <button key={name} className={page === name ? "active" : ""}
            onClick={() => goTo(name)}>{String(index + 1).padStart(2, "0")} · {name}</button>)}
        </nav>
        <div className="account">
          <span className="badge">{apiReady === null ? "연결 확인" : apiReady ? <><i className="status-dot" /> API 연결</> : "API 오프라인"}</span>
          {auth.authenticated
            ? <><span className="text-muted">{auth.username}</span><button className="button" onClick={() => void logout()}>로그아웃</button></>
            : <button className="button" onClick={() => { setPage("모델 학습 관리"); setLoginError(""); }}>관리자 로그인</button>}
        </div>
      </header>
      <div className="page">
        {apiReady === false && <div className="alert" style={{ marginBottom: 18 }}>백엔드 API에 연결할 수 없습니다. 로컬 FastAPI와 Docker 상태 또는 관리자에게 Tunnel 연결 상태를 확인하세요.</div>}
        {adminPage && !auth.authenticated ? <LoginPanel onLogin={onLogin} error={loginError} busy={loginBusy} /> : <>
          {page === "대시보드" && <DashboardPage league={league} onLeague={setLeague} />}
          {page === "승무패 예측" && <PredictionPage league={league} onLeague={setLeague} auth={auth} />}
          {page === "Elo 순위" && <EloPage league={league} onLeague={setLeague} />}
          {page === "팀 분석" && <EloPage league={league} onLeague={setLeague} teamMode />}
          {page === "경기별 Elo 분석" && <MatchAnalysisPage league={league} onLeague={setLeague} />}
          {page === "2026/27 경기 관리" && <MatchManagementPage league={league} onLeague={setLeague} auth={auth} />}
          {page === "모델 학습 관리" && <ModelManagementPage league={league} onLeague={setLeague} auth={auth} />}
          {page === "예측 백테스트" && <BacktestPage league={league} onLeague={setLeague} auth={auth} />}
        </>}
        <footer className="footer-note" style={{ marginTop: 36, textAlign: "center" }}>
          PickFoot PRE_MATCH · 추정치는 검증이 필요하며 실제 경기 결과를 보장하지 않습니다. · POST_LINEUP은 추후 지원
        </footer>
      </div>
    </main>
  );
}
