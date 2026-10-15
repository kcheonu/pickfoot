"""Persistent operational entities for matches, predictions, and model lifecycle."""

from datetime import date, datetime, timezone
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Team(Base):
    __tablename__ = "teams"
    __table_args__ = (UniqueConstraint("league", "team_key", name="uq_team_league_key"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    league: Mapped[str] = mapped_column(String(40), index=True)
    team_key: Mapped[str] = mapped_column(String(160))
    name: Mapped[str] = mapped_column(String(160))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    manually_registered: Mapped[bool] = mapped_column(Boolean, default=False)


class TeamAlias(Base):
    __tablename__ = "team_aliases"

    league: Mapped[str] = mapped_column(String(40), primary_key=True)
    alias_key: Mapped[str] = mapped_column(String(160), primary_key=True)
    alias: Mapped[str] = mapped_column(String(160))
    canonical_team: Mapped[str] = mapped_column(String(160))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class DataSource(Base):
    __tablename__ = "data_sources"
    __table_args__ = (
        UniqueConstraint("file_sha256", name="uq_dataset_sha256"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    file_path: Mapped[str] = mapped_column(String(500), unique=True)
    file_name: Mapped[str] = mapped_column(String(200))
    file_sha256: Mapped[str] = mapped_column(String(64), index=True)
    source_name: Mapped[str] = mapped_column(String(200))
    license_reference: Mapped[str] = mapped_column(String(500))
    league_coverage: Mapped[dict] = mapped_column(JSON)
    row_count: Mapped[int] = mapped_column(Integer)
    coverage_start: Mapped[date] = mapped_column(Date)
    coverage_end: Mapped[date] = mapped_column(Date)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    active: Mapped[bool] = mapped_column(Boolean, default=False, index=True)


class Match(Base):
    __tablename__ = "matches"
    __table_args__ = (
        UniqueConstraint(
            "league", "season", "match_date", "home_team_key", "away_team_key",
            name="uq_match_fixture",
        ),
        CheckConstraint(
            "status IN ('scheduled', 'completed', 'postponed', 'cancelled')",
            name="ck_match_status",
        ),
        CheckConstraint(
            "home_goals IS NULL OR home_goals >= 0", name="ck_match_home_goals"
        ),
        CheckConstraint(
            "away_goals IS NULL OR away_goals >= 0", name="ck_match_away_goals"
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    league: Mapped[str] = mapped_column(String(40), index=True)
    season: Mapped[str] = mapped_column(String(16), index=True)
    round_name: Mapped[str] = mapped_column(String(80), index=True)
    match_date: Mapped[date] = mapped_column(Date, index=True)
    home_team: Mapped[str] = mapped_column(String(160))
    away_team: Mapped[str] = mapped_column(String(160))
    home_team_key: Mapped[str] = mapped_column(String(160))
    away_team_key: Mapped[str] = mapped_column(String(160))
    status: Mapped[str] = mapped_column(String(16), default="scheduled", index=True)
    home_goals: Mapped[int | None] = mapped_column(Integer)
    away_goals: Mapped[int | None] = mapped_column(Integer)
    result: Mapped[str | None] = mapped_column(String(1))
    source_name: Mapped[str | None] = mapped_column(String(200))
    license_reference: Mapped[str | None] = mapped_column(String(500))
    version: Mapped[int] = mapped_column(Integer, default=1)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)


class MatchAudit(Base):
    __tablename__ = "match_audit"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    match_id: Mapped[str] = mapped_column(ForeignKey("matches.id"), index=True)
    action: Mapped[str] = mapped_column(String(24))
    previous_payload: Mapped[dict | None] = mapped_column(JSON)
    new_payload: Mapped[dict] = mapped_column(JSON)
    actor: Mapped[str] = mapped_column(String(120))
    reason: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    league: Mapped[str] = mapped_column(String(40), index=True)
    season: Mapped[str] = mapped_column(String(16), index=True)
    match_date: Mapped[date] = mapped_column(Date, index=True)
    home_team: Mapped[str] = mapped_column(String(160))
    away_team: Mapped[str] = mapped_column(String(160))
    prediction_mode: Mapped[str] = mapped_column(String(24), default="PRE_MATCH")
    predicted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    data_cutoff_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    model_version: Mapped[str] = mapped_column(String(120))
    model_training_cutoff: Mapped[date] = mapped_column(Date)
    team_state_cutoff: Mapped[date] = mapped_column(Date)
    state_fingerprint: Mapped[str] = mapped_column(String(64))
    home_probability: Mapped[float] = mapped_column(Float)
    draw_probability: Mapped[float] = mapped_column(Float)
    away_probability: Mapped[float] = mapped_column(Float)
    actual_result: Mapped[str | None] = mapped_column(String(1))
    evaluation_status: Mapped[str] = mapped_column(String(32), default="awaiting_result")
    feature_snapshot: Mapped[dict] = mapped_column(JSON)


class ModelVersion(Base):
    __tablename__ = "model_versions"
    __table_args__ = (
        UniqueConstraint("league", "version", name="uq_model_league_version"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    league: Mapped[str] = mapped_column(String(40), index=True)
    version: Mapped[str] = mapped_column(String(120))
    artifact_path: Mapped[str] = mapped_column(String(500))
    source_data_hash: Mapped[str] = mapped_column(String(64))
    evaluation: Mapped[dict] = mapped_column(JSON)
    comparison: Mapped[dict | None] = mapped_column(JSON)
    training_cutoff: Mapped[date] = mapped_column(Date)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    candidate_status: Mapped[str] = mapped_column(String(24), default="candidate")


class ActiveModel(Base):
    __tablename__ = "active_models"

    league: Mapped[str] = mapped_column(String(40), primary_key=True)
    model_version_id: Mapped[str] = mapped_column(ForeignKey("model_versions.id"))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    job_type: Mapped[str] = mapped_column(String(16), default="retrain")
    league: Mapped[str] = mapped_column(String(40), index=True)
    status: Mapped[str] = mapped_column(String(20), default="queued", index=True)
    requested_by: Mapped[str] = mapped_column(String(120))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    candidate_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("model_versions.id")
    )
    progress_message: Mapped[str] = mapped_column(String(300), default="Waiting for worker")
    result: Mapped[dict | None] = mapped_column(JSON)
    error_message: Mapped[str | None] = mapped_column(Text)


class AdminAudit(Base):
    __tablename__ = "admin_audit"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    actor: Mapped[str] = mapped_column(String(120))
    action: Mapped[str] = mapped_column(String(80))
    entity_type: Mapped[str] = mapped_column(String(80))
    entity_id: Mapped[str] = mapped_column(String(120))
    details: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
