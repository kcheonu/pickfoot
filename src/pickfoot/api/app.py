"""FastAPI application for the deployable PickFoot service."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import date
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import re

from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import and_, or_, select, text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from pickfoot.api.auth import (
    SESSION_COOKIE,
    admin_user,
    create_session,
    enforce_rate_limit,
    require_same_origin,
    set_session_cookie,
    verify_password,
)
from pickfoot.api.config import get_settings
from pickfoot.api.db import get_db, session_factory
from pickfoot.api.models import (
    ActiveModel,
    AdminAudit,
    Base,
    DataSource,
    Match,
    MatchAudit,
    ModelVersion,
    Prediction,
    Team,
    TrainingJob,
    utc_now,
)
from pickfoot.api.schemas import (
    LoginRequest,
    MatchCreate,
    MatchResultUpdate,
    MatchStatusUpdate,
    PredictionRequest,
    TeamRegistration,
    TrainingRequest,
)
from pickfoot.api.services import (
    SUPPORTED_LEAGUES,
    available_matches,
    canonical_league,
    current_state,
    date_in_2026_27,
    match_payload,
    model_for_league,
    model_slug,
    outcome,
    standings_payload,
)
from pickfoot.data import CSVValidationError
from pickfoot.model import (
    ModelError,
    compare_models_on_candidate_window,
    evaluation_dict,
    load_artifact,
    predict_match_with_state,
)
from pickfoot.service import state_hash

settings = get_settings()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if settings.environment == "production":
        missing = [
            name for name, value in (
                ("ADMIN_PASSWORD_HASH", settings.admin_password_hash),
                ("SESSION_SECRET", settings.session_secret),
            ) if not value
        ]
        if settings.session_secret and len(settings.session_secret) < 32:
            missing.append("SESSION_SECRET (minimum 32 characters)")
        if missing or not settings.cookie_secure:
            raise RuntimeError(
                "Production requires " + ", ".join(
                    missing + ([] if settings.cookie_secure else ["COOKIE_SECURE=true"])
                )
            )
    yield


app = FastAPI(
    title="PickFoot API",
    version="1.0.0",
    description="Persisted match operations and PRE_MATCH H/D/A estimates.",
    lifespan=lifespan,
)
origins = list(dict.fromkeys([settings.web_origin, *settings.cors_origins]))
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "OPTIONS"],
    allow_headers=["Content-Type"],
)


def _raise_domain_error(exc: Exception) -> None:
    if isinstance(exc, (CSVValidationError, ModelError)):
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    raise exc


def _write_admin(request: Request) -> str:
    username = admin_user(request)
    require_same_origin(request)
    enforce_rate_limit(request, scope="admin-write", maximum=60, window_seconds=60)
    return username


def _audit(
    session: Session,
    *,
    actor: str,
    action: str,
    entity_type: str,
    entity_id: str,
    details: dict,
) -> None:
    session.add(AdminAudit(
        actor=actor,
        action=action,
        entity_type=entity_type,
        entity_id=entity_id,
        details=details,
    ))


@app.get("/api/v1/health/live")
def live_health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v1/health/ready")
def ready_health(session: Session = Depends(get_db)) -> dict[str, str]:
    try:
        session.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=503, detail="Database is not ready.") from exc
    root = settings.model_storage_dir
    if not root.exists() or not root.is_dir():
        raise HTTPException(status_code=503, detail="Persistent model storage is unavailable.")
    return {"status": "ready", "database": "connected", "model_storage": "available"}


@app.post("/api/v1/auth/login")
def login(
    body: LoginRequest,
    request: Request,
    response: Response,
) -> dict[str, str]:
    require_same_origin(request)
    enforce_rate_limit(request, scope="login", maximum=5, window_seconds=300)
    if not settings.admin_password_hash or not settings.session_secret:
        raise HTTPException(
            status_code=503,
            detail="Admin authentication is not configured on the server.",
        )
    valid_user = body.username == settings.admin_username
    valid_password = verify_password(body.password, settings.admin_password_hash)
    if not (valid_user and valid_password):
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    token = create_session(
        settings.admin_username, settings.session_secret, settings.session_ttl_seconds
    )
    set_session_cookie(response, token)
    return {"username": settings.admin_username, "authenticated": "true"}


@app.post("/api/v1/auth/logout")
def logout(request: Request, response: Response) -> dict[str, str]:
    require_same_origin(request)
    response.delete_cookie(
        SESSION_COOKIE,
        path="/",
        secure=settings.cookie_secure,
        httponly=True,
        samesite="none" if settings.cookie_secure else "lax",
    )
    return {"authenticated": "false"}


@app.get("/api/v1/auth/me")
def who_am_i(request: Request) -> dict[str, bool | str]:
    try:
        return {"authenticated": True, "username": admin_user(request)}
    except HTTPException as exc:
        if exc.status_code == status.HTTP_401_UNAUTHORIZED:
            return {"authenticated": False, "username": ""}
        raise


@app.get("/api/v1/leagues")
def leagues() -> dict[str, list[str]]:
    return {"leagues": list(SUPPORTED_LEAGUES)}


@app.get("/api/v1/datasets")
def datasets(
    request: Request,
    session: Session = Depends(get_db),
    actor: str = Depends(admin_user),
) -> dict:
    rows = session.scalars(
        select(DataSource).order_by(DataSource.uploaded_at.desc()).limit(50)
    ).all()
    return {"datasets": [{
        "id": row.id,
        "file_name": row.file_name,
        "file_sha256": row.file_sha256,
        "source_name": row.source_name,
        "license_reference": row.license_reference,
        "league_coverage": row.league_coverage,
        "row_count": row.row_count,
        "coverage_start": row.coverage_start,
        "coverage_end": row.coverage_end,
        "uploaded_at": row.uploaded_at,
        "active": row.active,
    } for row in rows]}


@app.post("/api/v1/admin/datasets", status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    request: Request,
    file: UploadFile = File(...),
    source_name: str = "",
    license_reference: str = "",
    rights_confirmed: bool = False,
    session: Session = Depends(get_db),
    actor: str = Depends(_write_admin),
) -> dict:
    if not rights_confirmed or not source_name.strip() or not license_reference.strip():
        raise HTTPException(
            status_code=422,
            detail="Provide the source, license/permission reference, and explicit rights confirmation.",
        )
    if Path(file.filename or "").suffix.casefold() != ".csv":
        raise HTTPException(status_code=415, detail="Only CSV datasets are accepted.")
    content = await file.read(25 * 1024 * 1024 + 1)
    if len(content) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="CSV upload exceeds the 25 MB limit.")
    digest = sha256(content).hexdigest()
    existing = session.scalar(select(DataSource).where(DataSource.file_sha256 == digest))
    if existing:
        return {"dataset_id": existing.id, "active": existing.active, "idempotent": True}
    try:
        from pickfoot.data import normalize_matches
        import pandas as pd

        frame = pd.read_csv(
            BytesIO(content), dtype=str, keep_default_na=False, encoding="utf-8-sig"
        )
        normalized = normalize_matches(frame)
    except (UnicodeError, pd.errors.EmptyDataError, pd.errors.ParserError, CSVValidationError) as exc:
        raise HTTPException(status_code=422, detail=f"CSV validation failed: {exc}") from exc
    except OSError as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse uploaded CSV: {exc}") from exc
    destination_dir = settings.data_storage_dir.resolve() / "base"
    destination = destination_dir / f"{digest}.csv"
    temporary = destination.with_suffix(".csv.tmp")
    try:
        destination_dir.mkdir(parents=True, exist_ok=True)
        temporary.write_bytes(content)
        temporary.replace(destination)
        counts = normalized.groupby("league").size().to_dict()
        dataset = DataSource(
            file_path=str(destination),
            file_name=Path(file.filename or "matches.csv").name[:200],
            file_sha256=digest,
            source_name=source_name.strip(),
            license_reference=license_reference.strip(),
            league_coverage={str(league): int(count) for league, count in counts.items()},
            row_count=len(normalized),
            coverage_start=min(normalized["match_date"]),
            coverage_end=max(normalized["match_date"]),
            active=True,
        )
        session.query(DataSource).filter(DataSource.active.is_(True)).update(
            {DataSource.active: False}, synchronize_session=False
        )
        session.add(dataset)
        session.flush()
        _audit(
            session,
            actor=actor,
            action="base_dataset_uploaded",
            entity_type="data_source",
            entity_id=dataset.id,
            details={
                "sha256": digest,
                "row_count": len(normalized),
                "source_name": source_name.strip(),
                "rights_confirmed": True,
            },
        )
        session.commit()
    except (OSError, SQLAlchemyError) as exc:
        session.rollback()
        temporary.unlink(missing_ok=True)
        if destination.exists():
            destination.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Could not activate the uploaded dataset.") from exc
    return {
        "dataset_id": dataset.id,
        "file_sha256": digest,
        "row_count": len(normalized),
        "league_coverage": dataset.league_coverage,
        "coverage_start": dataset.coverage_start,
        "coverage_end": dataset.coverage_end,
        "active": True,
        "idempotent": False,
    }


@app.get("/api/v1/rounds")
def rounds(
    request: Request,
    league: str,
    season: str = "2026-27",
    session: Session = Depends(get_db),
    actor: str = Depends(admin_user),
) -> dict[str, list[str]]:
    canonical_league(league)
    rows = session.scalars(
        select(Match.round_name)
        .where(Match.league == league, Match.season == season)
        .distinct()
        .order_by(Match.round_name)
    ).all()
    return {"rounds": list(rows)}


@app.get("/api/v1/matches")
def matches(
    request: Request,
    league: str,
    season: str = "2026-27",
    round_name: str | None = None,
    session: Session = Depends(get_db),
    actor: str = Depends(admin_user),
) -> dict[str, list[dict]]:
    canonical_league(league)
    query = select(Match).where(Match.league == league, Match.season == season)
    if round_name:
        query = query.where(Match.round_name == round_name)
    rows = session.scalars(
        query.order_by(Match.match_date, Match.home_team, Match.away_team)
    ).all()
    return {"matches": [match_payload(row) for row in rows]}


@app.post("/api/v1/admin/matches", status_code=status.HTTP_201_CREATED)
def create_match(
    body: MatchCreate,
    request: Request,
    session: Session = Depends(get_db),
    actor: str = Depends(_write_admin),
) -> dict:
    canonical_league(body.league)
    if not date_in_2026_27(body.match_date):
        raise HTTPException(status_code=422, detail="Match date must be in the 2026-27 season.")
    home_team = " ".join(body.home_team.split())
    away_team = " ".join(body.away_team.split())
    home_key, away_key = home_team.casefold(), away_team.casefold()
    duplicate = session.scalar(
        select(Match.id).where(
            Match.league == body.league,
            Match.season == body.season,
            Match.match_date == body.match_date,
            or_(
                and_(Match.home_team_key == home_key, Match.away_team_key == away_key),
                and_(Match.home_team_key == away_key, Match.away_team_key == home_key),
            ),
        )
    )
    if duplicate:
        raise HTTPException(status_code=409, detail="This fixture is already registered.")
    match = Match(
        league=body.league,
        season=body.season,
        round_name=body.round_name,
        match_date=body.match_date,
        home_team=home_team,
        away_team=away_team,
        home_team_key=home_key,
        away_team_key=away_key,
        status="scheduled",
    )
    try:
        session.add(match)
        for team_name, team_key in ((home_team, home_key), (away_team, away_key)):
            exists = session.scalar(
                select(Team.id).where(Team.league == body.league, Team.team_key == team_key)
            )
            if not exists:
                session.add(Team(
                    league=body.league,
                    team_key=team_key,
                    name=team_name,
                    manually_registered=True,
                ))
        session.flush()
        session.add(MatchAudit(
            match_id=match.id,
            action="fixture_created",
            previous_payload=None,
            new_payload=match_payload(match),
            actor=actor,
            reason="Fixture registered by an authenticated administrator.",
        ))
        _audit(
            session, actor=actor, action="fixture_created",
            entity_type="match", entity_id=match.id,
            details={"league": body.league, "season": body.season, "round": body.round_name},
        )
        session.commit()
    except IntegrityError as exc:
        session.rollback()
        raise HTTPException(status_code=409, detail="A duplicate fixture already exists.") from exc
    return match_payload(match)


@app.post("/api/v1/admin/teams", status_code=status.HTTP_201_CREATED)
def register_team(
    body: TeamRegistration,
    request: Request,
    session: Session = Depends(get_db),
    actor: str = Depends(_write_admin),
) -> dict:
    canonical_league(body.league)
    key = " ".join(body.team_name.split()).casefold()
    existing = session.scalar(
        select(Team).where(Team.league == body.league, Team.team_key == key)
    )
    if existing:
        raise HTTPException(status_code=409, detail="This team is already registered in the league.")
    team = Team(
        league=body.league,
        team_key=key,
        name=" ".join(body.team_name.split()),
        manually_registered=True,
    )
    try:
        session.add(team)
        session.flush()
        _audit(
            session, actor=actor, action="team_registered",
            entity_type="team", entity_id=team.id,
            details={"league": body.league, "team": team.name, "basis": body.registration_note},
        )
        session.commit()
    except IntegrityError as exc:
        session.rollback()
        raise HTTPException(status_code=409, detail="This team is already registered in the league.") from exc
    return {
        "id": team.id,
        "league": team.league,
        "team_name": team.name,
        "elo_policy": "1500 initial rating; no match history is fabricated",
    }


@app.put("/api/v1/admin/matches/{match_id}/result")
def confirm_result(
    match_id: str,
    body: MatchResultUpdate,
    request: Request,
    session: Session = Depends(get_db),
    actor: str = Depends(_write_admin),
) -> dict:
    match = session.get(Match, match_id)
    if match is None:
        raise HTTPException(status_code=404, detail="Fixture not found.")
    if match.version != body.expected_version:
        raise HTTPException(status_code=409, detail="Fixture changed; reload before confirming.")
    if match.match_date > date.today():
        raise HTTPException(status_code=422, detail="A future fixture cannot be confirmed as completed.")
    if match.status == "cancelled":
        raise HTTPException(status_code=409, detail="A cancelled fixture cannot receive a score.")
    calculated = outcome(body.home_goals, body.away_goals)
    old_payload = match_payload(match)
    same_score = (
        match.status == "completed"
        and match.home_goals == body.home_goals
        and match.away_goals == body.away_goals
    )
    if same_score:
        return {"match": old_payload, "idempotent": True, "elo": None}
    if match.status == "completed" and not body.correction_confirmed:
        raise HTTPException(
            status_code=409,
            detail="A different score is already confirmed. Review and explicitly confirm a correction.",
        )

    action = "result_corrected" if match.status == "completed" else "result_confirmed"
    match.home_goals = body.home_goals
    match.away_goals = body.away_goals
    match.result = calculated
    match.status = "completed"
    match.source_name = body.source_name.strip()
    match.license_reference = body.license_reference.strip()
    match.version += 1
    match.updated_at = utc_now()
    try:
        session.add(MatchAudit(
            match_id=match.id,
            action=action,
            previous_payload=old_payload,
            new_payload=match_payload(match),
            actor=actor,
            reason=body.reason.strip(),
        ))
        _audit(
            session, actor=actor, action=action, entity_type="match",
            entity_id=match.id,
            details={
                "source_name": body.source_name.strip(),
                "license_reference": body.license_reference.strip(),
                "rights_confirmed": body.rights_confirmed,
                "previous_result": old_payload.get("result"),
                "new_result": calculated,
            },
        )
        linked = session.scalars(select(Prediction).where(
            Prediction.league == match.league,
            Prediction.season == match.season,
            Prediction.match_date == match.match_date,
            Prediction.home_team == match.home_team,
            Prediction.away_team == match.away_team,
        )).all()
        for prediction in linked:
            prediction.actual_result = calculated
            prediction.evaluation_status = "unverified_kickoff_time"
        session.commit()
    except SQLAlchemyError as exc:
        session.rollback()
        raise HTTPException(status_code=500, detail="Could not persist the result transaction.") from exc

    elo_update = None
    try:
        _, built = current_state(session, match.league)
        history = built.elo_history
        selected = history.loc[history["home_team"].str.casefold().eq(match.home_team_key)
                               & history["away_team"].str.casefold().eq(match.away_team_key)
                               & history["match_date"].eq(match.match_date)]
        if not selected.empty:
            row = selected.iloc[-1]
            elo_update = {
                "as_of": built.state.latest_date,
                "home_before": float(row["home_elo_before"]),
                "away_before": float(row["away_elo_before"]),
                "home_after": float(row["home_elo_after"]),
                "away_after": float(row["away_elo_after"]),
                "home_change": float(row["home_elo_change"]),
                "away_change": float(row["away_elo_change"]),
                "expected_home_score": float(row["expected_home_score"]),
            }
    except (CSVValidationError, ModelError):
        elo_update = {"status": "saved", "detail": "Elo state will be available when match history exists."}
    return {
        "match": match_payload(match),
        "idempotent": False,
        "linked_predictions": len(linked),
        "elo": elo_update,
    }


@app.patch("/api/v1/admin/matches/{match_id}/status")
def update_match_status(
    match_id: str,
    body: MatchStatusUpdate,
    request: Request,
    session: Session = Depends(get_db),
    actor: str = Depends(_write_admin),
) -> dict:
    match = session.get(Match, match_id)
    if match is None:
        raise HTTPException(status_code=404, detail="Fixture not found.")
    if match.version != body.expected_version:
        raise HTTPException(status_code=409, detail="Fixture changed; reload before updating.")
    if match.status == "completed":
        raise HTTPException(status_code=409, detail="A completed result cannot be postponed or cancelled.")
    previous = match_payload(match)
    match.status = body.status
    match.version += 1
    match.updated_at = utc_now()
    session.add(MatchAudit(
        match_id=match.id,
        action=f"status_{body.status}",
        previous_payload=previous,
        new_payload=match_payload(match),
        actor=actor,
        reason=body.reason.strip(),
    ))
    _audit(
        session, actor=actor, action=f"status_{body.status}",
        entity_type="match", entity_id=match.id, details={"reason": body.reason.strip()},
    )
    session.commit()
    return match_payload(match)


@app.get("/api/v1/elo/{league}")
def elo_data(
    league: str,
    team: str | None = None,
    session: Session = Depends(get_db),
) -> dict:
    canonical_league(league)
    try:
        return standings_payload(session, league, team=team)
    except (CSVValidationError, ModelError) as exc:
        _raise_domain_error(exc)
    raise AssertionError("unreachable")


@app.get("/api/v1/models")
def list_models(session: Session = Depends(get_db)) -> dict:
    versions = session.scalars(
        select(ModelVersion).order_by(ModelVersion.league, ModelVersion.created_at.desc())
    ).all()
    active_by_league = {
        row.league: row.model_version_id
        for row in session.scalars(select(ActiveModel)).all()
    }
    output = [{
        "id": row.id,
        "league": row.league,
        "version": row.version,
        "training_cutoff": row.training_cutoff,
        "evaluation": row.evaluation,
        "comparison": row.comparison,
        "candidate_status": row.candidate_status,
        "active": active_by_league.get(row.league) == row.id,
        "created_at": row.created_at,
    } for row in versions]
    registered = {item["league"] for item in output}
    for league in SUPPORTED_LEAGUES:
        if league in registered:
            continue
        path = settings.model_storage_dir / f"{model_slug(league)}.joblib"
        if path.is_file():
            try:
                artifact = load_artifact(path)
            except ModelError:
                continue
            output.append({
                "id": None,
                "league": league,
                "version": artifact.model_version,
                "training_cutoff": artifact.state.latest_date,
                "evaluation": evaluation_dict(artifact),
                "comparison": None,
                "candidate_status": "legacy",
                "active": True,
                "created_at": None,
            })
    return {"models": output}


@app.get("/api/v1/training-jobs")
def training_jobs(
    request: Request,
    session: Session = Depends(get_db),
    actor: str = Depends(admin_user),
) -> dict:
    rows = session.scalars(
        select(TrainingJob).order_by(TrainingJob.created_at.desc()).limit(100)
    ).all()
    return {"jobs": [{
        "id": row.id,
        "job_type": row.job_type,
        "league": row.league,
        "status": row.status,
        "created_at": row.created_at,
        "started_at": row.started_at,
        "completed_at": row.completed_at,
        "progress_message": row.progress_message,
        "result": (
            {key: value for key, value in (row.result or {}).items() if key != "report"}
            if row.job_type == "backtest" else row.result
        ),
        "error_message": row.error_message,
        "candidate_version_id": row.candidate_version_id,
    } for row in rows]}


@app.post("/api/v1/admin/training-jobs", status_code=status.HTTP_202_ACCEPTED)
def enqueue_training(
    body: TrainingRequest,
    request: Request,
    session: Session = Depends(get_db),
    actor: str = Depends(_write_admin),
) -> dict:
    canonical_league(body.league)
    existing = session.scalar(
        select(TrainingJob).where(
            TrainingJob.job_type == "retrain",
            TrainingJob.league == body.league,
            TrainingJob.status.in_(("queued", "running")),
        ).order_by(TrainingJob.created_at.desc()).limit(1)
    )
    if existing:
        return {"job_id": existing.id, "status": existing.status, "deduplicated": True}
    try:
        data = available_matches(session, body.league)
    except ModelError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    base_source = session.scalar(
        select(DataSource.id).where(DataSource.active.is_(True)).limit(1)
    )
    if base_source is None:
        raise HTTPException(
            status_code=409,
            detail="Activate an authorized base dataset in Model Management before training.",
        )
    try:
        model, _ = model_for_league(session, body.league)
    except ModelError:
        model = None
    if model is not None:
        new_completed = session.scalar(
            select(Match.id).where(
                Match.league == body.league,
                Match.status == "completed",
                Match.match_date > model.state.latest_date,
            ).limit(1)
        )
        if new_completed is None:
            raise HTTPException(
                status_code=422,
                detail="No confirmed match results newer than the operating model cutoff are available.",
            )
    job = TrainingJob(
        job_type="retrain",
        league=body.league,
        status="queued",
        requested_by=actor,
        progress_message=f"Queued with {len(data)} completed matches",
    )
    session.add(job)
    session.flush()
    session.add(AdminAudit(
        actor=actor,
        action="training_queued",
        entity_type="training_job",
        entity_id=job.id,
        details={"league": body.league, "match_count": len(data)},
    ))
    session.commit()
    return {"job_id": job.id, "status": job.status, "deduplicated": False}


@app.post("/api/v1/admin/backtests", status_code=status.HTTP_202_ACCEPTED)
def enqueue_backtest(
    request: Request,
    league: str,
    session: Session = Depends(get_db),
    actor: str = Depends(_write_admin),
) -> dict:
    canonical_league(league)
    source = session.scalar(select(DataSource.id).where(DataSource.active.is_(True)).limit(1))
    if source is None:
        raise HTTPException(
            status_code=409,
            detail="Upload an authorized base dataset in Model Management before running a backtest.",
        )
    existing = session.scalar(
        select(TrainingJob).where(
            TrainingJob.job_type == "backtest",
            TrainingJob.league == league,
            TrainingJob.status.in_(("queued", "running")),
        ).order_by(TrainingJob.created_at.desc()).limit(1)
    )
    if existing:
        return {"job_id": existing.id, "status": existing.status, "deduplicated": True}
    try:
        matches = available_matches(session, league)
    except ModelError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    job = TrainingJob(
        job_type="backtest",
        league=league,
        status="queued",
        requested_by=actor,
        progress_message=f"Queued with {len(matches)} completed matches",
    )
    session.add(job)
    session.flush()
    _audit(
        session, actor=actor, action="backtest_queued",
        entity_type="training_job", entity_id=job.id,
        details={"league": league, "match_count": len(matches)},
    )
    session.commit()
    return {"job_id": job.id, "status": job.status, "deduplicated": False}


@app.post("/api/v1/admin/models/{league}/promote")
def promote_model(
    league: str,
    version_id: str,
    request: Request,
    session: Session = Depends(get_db),
    actor: str = Depends(_write_admin),
) -> dict:
    canonical_league(league)
    candidate = session.get(ModelVersion, version_id)
    if candidate is None or candidate.league != league:
        raise HTTPException(status_code=404, detail="Candidate model not found.")
    path = Path(candidate.artifact_path).resolve()
    if settings.model_storage_dir.resolve() not in path.parents or not path.is_file():
        raise HTTPException(status_code=409, detail="Candidate artifact is not available in persistent storage.")
    previous = session.get(ActiveModel, league)
    if previous and previous.model_version_id == candidate.id:
        return {"active_version": candidate.version, "idempotent": True}
    previous_id = previous.model_version_id if previous else None
    if previous:
        previous_version = session.get(ModelVersion, previous.model_version_id)
        if previous_version:
            previous_version.candidate_status = "superseded"
    candidate.candidate_status = "active"
    if previous:
        previous.model_version_id = candidate.id
        previous.updated_at = utc_now()
    else:
        session.add(ActiveModel(league=league, model_version_id=candidate.id))
    _audit(
        session, actor=actor, action="model_promoted",
        entity_type="model_version", entity_id=candidate.id,
        details={"league": league, "previous_id": previous_id},
    )
    session.commit()
    return {"active_version": candidate.version, "idempotent": False}


@app.post("/api/v1/admin/models/{league}/rollback")
def rollback_model(
    league: str,
    request: Request,
    session: Session = Depends(get_db),
    actor: str = Depends(_write_admin),
) -> dict:
    canonical_league(league)
    current = session.get(ActiveModel, league)
    if current is None:
        raise HTTPException(status_code=409, detail="No versioned active model is available to roll back.")
    candidate = session.get(ModelVersion, current.model_version_id)
    previous = session.scalar(
        select(ModelVersion).where(
            ModelVersion.league == league,
            ModelVersion.candidate_status == "superseded",
        ).order_by(ModelVersion.created_at.desc()).limit(1)
    )
    if previous is None:
        raise HTTPException(status_code=409, detail="No previous promoted model is available.")
    path = Path(previous.artifact_path).resolve()
    if settings.model_storage_dir.resolve() not in path.parents or not path.is_file():
        raise HTTPException(status_code=409, detail="Previous model artifact is unavailable.")
    if candidate:
        candidate.candidate_status = "candidate"
    previous.candidate_status = "active"
    current.model_version_id = previous.id
    current.updated_at = utc_now()
    _audit(
        session, actor=actor, action="model_rollback",
        entity_type="model_version", entity_id=previous.id,
        details={"league": league, "from_version": candidate.version if candidate else None},
    )
    session.commit()
    return {"active_version": previous.version, "rolled_back": True}


@app.post("/api/v1/predictions")
def predict(
    body: PredictionRequest,
    request: Request,
    session: Session = Depends(get_db),
) -> dict:
    enforce_rate_limit(request, scope="public-prediction", maximum=120, window_seconds=60)
    canonical_league(body.league)
    try:
        artifact, version_id = model_for_league(session, body.league)
        if artifact.source_metadata.get("data_kind") != "authorized_real":
            raise HTTPException(
                status_code=409, detail="Synthetic or unverified model artifacts cannot serve predictions."
            )
        _, built = current_state(session, body.league)
        prediction = predict_match_with_state(
            artifact,
            state=built.state,
            home_team=body.home_team,
            away_team=body.away_team,
            match_date=body.match_date,
        )
    except (CSVValidationError, ModelError) as exc:
        _raise_domain_error(exc)
        raise AssertionError("unreachable")
    fingerprint = state_hash(prediction["features"])
    row = session.scalar(
        select(Prediction).where(
            Prediction.league == body.league,
            Prediction.season == body.season,
            Prediction.match_date == body.match_date,
            Prediction.home_team == body.home_team,
            Prediction.away_team == body.away_team,
            Prediction.model_version == artifact.model_version,
            Prediction.state_fingerprint == fingerprint,
        )
    )
    inserted = row is None
    if row is None:
        probabilities = prediction["probabilities"]
        row = Prediction(
            league=body.league,
            season=body.season,
            match_date=body.match_date,
            home_team=body.home_team,
            away_team=body.away_team,
            model_version=artifact.model_version,
            model_training_cutoff=artifact.state.latest_date,
            team_state_cutoff=built.state.latest_date,
            state_fingerprint=fingerprint,
            home_probability=probabilities["H"],
            draw_probability=probabilities["D"],
            away_probability=probabilities["A"],
            evaluation_status="awaiting_result",
            feature_snapshot=prediction["features"],
        )
        session.add(row)
        session.commit()
    return {
        "prediction_id": row.id,
        "inserted": inserted,
        "league": body.league,
        "season": body.season,
        "match_date": body.match_date,
        "home_team": body.home_team,
        "away_team": body.away_team,
        "prediction_mode": "PRE_MATCH",
        "probabilities": {
            "H": row.home_probability,
            "D": row.draw_probability,
            "A": row.away_probability,
        },
        "home_elo": prediction["home_elo"],
        "away_elo": prediction["away_elo"],
        "home_form": prediction["home_form"],
        "away_form": prediction["away_form"],
        "model_version": artifact.model_version,
        "active_model_version_id": version_id,
        "model_training_cutoff": artifact.state.latest_date,
        "team_state_cutoff": built.state.latest_date,
        "evaluation": evaluation_dict(artifact),
        "evaluation_status": row.evaluation_status,
        "note": "Date-only records do not prove the prediction preceded kickoff.",
    }


@app.post("/api/v1/predictions/compare")
def compare_prediction(
    body: PredictionRequest,
    request: Request,
    candidate_version_id: str,
    session: Session = Depends(get_db),
) -> dict:
    enforce_rate_limit(request, scope="public-model-comparison", maximum=60, window_seconds=60)
    canonical_league(body.league)
    incumbent, _ = model_for_league(session, body.league)
    candidate_record = session.get(ModelVersion, candidate_version_id)
    if candidate_record is None or candidate_record.league != body.league:
        raise HTTPException(status_code=404, detail="Candidate model not found for this league.")
    candidate_path = Path(candidate_record.artifact_path).resolve()
    if settings.model_storage_dir.resolve() not in candidate_path.parents:
        raise HTTPException(status_code=409, detail="Candidate artifact path is invalid.")
    try:
        candidate = load_artifact(candidate_path)
        _, built = current_state(session, body.league)
        old_result = predict_match_with_state(
            incumbent, state=built.state, home_team=body.home_team,
            away_team=body.away_team, match_date=body.match_date,
        )
        new_result = predict_match_with_state(
            candidate, state=built.state, home_team=body.home_team,
            away_team=body.away_team, match_date=body.match_date,
        )
    except (CSVValidationError, ModelError) as exc:
        _raise_domain_error(exc)
        raise AssertionError("unreachable")
    return {
        "operating": {
            "version": incumbent.model_version,
            "probabilities": old_result["probabilities"],
        },
        "candidate": {
            "version": candidate.model_version,
            "probabilities": new_result["probabilities"],
        },
        "team_state_cutoff": built.state.latest_date,
        "warning": "This is a same-input model comparison, not evidence of a future performance gain.",
    }


@app.get("/api/v1/predictions")
def prediction_history(
    request: Request,
    league: str | None = None,
    session: Session = Depends(get_db),
    actor: str = Depends(admin_user),
) -> dict:
    query = select(Prediction).order_by(Prediction.predicted_at.desc()).limit(500)
    if league:
        canonical_league(league)
        query = query.where(Prediction.league == league)
    rows = session.scalars(query).all()
    return {"predictions": [{
        "id": row.id,
        "league": row.league,
        "season": row.season,
        "match_date": row.match_date,
        "home_team": row.home_team,
        "away_team": row.away_team,
        "probabilities": {
            "H": row.home_probability,
            "D": row.draw_probability,
            "A": row.away_probability,
        },
        "model_version": row.model_version,
        "model_training_cutoff": row.model_training_cutoff,
        "team_state_cutoff": row.team_state_cutoff,
        "predicted_at": row.predicted_at,
        "actual_result": row.actual_result,
        "evaluation_status": row.evaluation_status,
    } for row in rows]}


@app.get("/api/v1/backtests")
def backtests(request: Request, actor: str = Depends(admin_user), session: Session = Depends(get_db)) -> dict:
    latest_job = session.scalar(
        select(TrainingJob).where(
            TrainingJob.job_type == "backtest",
            TrainingJob.status == "completed",
            TrainingJob.result.is_not(None),
        ).order_by(TrainingJob.completed_at.desc()).limit(1)
    )
    if latest_job and latest_job.result and "report" in latest_job.result:
        return {
            "available": True,
            "report": latest_job.result["report"],
            "generated_at": latest_job.completed_at,
            "job_id": latest_job.id,
        }
    path = settings.base_matches_csv.parent.parent / "runtime" / "walk_forward_backtest.json"
    if not path.is_file():
        return {"available": False, "report": None}
    import json

    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail="Stored walk-forward report cannot be read.") from exc
    return {"available": True, "report": report}


@app.get("/api/v1/admin/training-jobs/{job_id}")
def training_job_detail(
    job_id: str,
    request: Request,
    session: Session = Depends(get_db),
    actor: str = Depends(admin_user),
) -> dict:
    job = session.get(TrainingJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Training job not found.")
    return {
        "id": job.id,
        "league": job.league,
        "status": job.status,
        "progress_message": job.progress_message,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
        "result": job.result,
        "error_message": job.error_message,
        "candidate_version_id": job.candidate_version_id,
        "job_type": job.job_type,
    }


@app.get("/api/v1/admin/matches/{match_id}/audit")
def match_audit(
    match_id: str,
    request: Request,
    session: Session = Depends(get_db),
    actor: str = Depends(admin_user),
) -> dict:
    rows = session.scalars(
        select(MatchAudit).where(MatchAudit.match_id == match_id)
        .order_by(MatchAudit.created_at)
    ).all()
    return {"history": [{
        "action": row.action,
        "previous": row.previous_payload,
        "new": row.new_payload,
        "actor": row.actor,
        "reason": row.reason,
        "created_at": row.created_at,
    } for row in rows]}
