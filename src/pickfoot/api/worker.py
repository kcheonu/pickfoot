"""Database-queued model trainer; run as a separate persistent worker process."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import time
from uuid import uuid4

import joblib
from sqlalchemy import select
from sqlalchemy.orm import Session

from pickfoot.api.config import get_settings
from pickfoot.api.db import session_factory
from pickfoot.api.models import (
    ActiveModel,
    AdminAudit,
    DataSource,
    ModelVersion,
    TrainingJob,
    utc_now,
)
from pickfoot.api.services import available_matches, model_for_league, model_slug
from pickfoot.backtest import expanding_window_backtest, matches_fingerprint
from pickfoot.model import (
    ModelArtifact,
    ModelError,
    compare_models_on_candidate_window,
    evaluation_dict,
    train_model,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("pickfoot.worker")
DATABASE_SECRETS = re.compile(r"(?i)(?:postgres(?:ql)?(?:\+\w+)?://)[^\s'\"<>]+")


def _safe_error(exc: Exception) -> str:
    message = DATABASE_SECRETS.sub("postgresql://[redacted]", str(exc))
    return f"{type(exc).__name__}: {message}"[:2000]


def _claim_job(factory) -> str | None:
    with factory.begin() as session:
        job = session.scalar(
            select(TrainingJob)
            .where(TrainingJob.status == "queued")
            .order_by(TrainingJob.created_at)
            .with_for_update(skip_locked=True)
            .limit(1)
        )
        if job is None:
            return None
        job.status = "running"
        job.started_at = utc_now()
        job.progress_message = "Loading authorized match history"
        return job.id


def _training_sources(session: Session) -> tuple[str, str]:
    source = session.scalar(
        select(DataSource).where(DataSource.active.is_(True))
        .order_by(DataSource.uploaded_at.desc())
    )
    if source is None:
        raise ModelError(
            "Upload the authorized base CSV and confirm its source and permitted use "
            "before starting model training."
        )
    return source.source_name, source.license_reference


def _save_versioned_artifact(artifact: ModelArtifact) -> Path:
    settings = get_settings()
    root = settings.model_storage_dir.resolve()
    directory = root / model_slug(artifact.league)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{artifact.model_version}.joblib"
    temporary = target.with_suffix(".joblib.tmp")
    try:
        joblib.dump(artifact, temporary)
        temporary.replace(target)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise
    return target


def process_job(job_id: str, factory=None) -> None:
    factory = factory or session_factory()
    with factory() as session:
        job = session.get(TrainingJob, job_id)
        if job is None or job.status != "running":
            raise RuntimeError("Training job is no longer in the running state.")
        league = job.league
        job.progress_message = "Building chronological features and fitting candidate"
        session.commit()

    artifact_path: Path | None = None
    try:
        with factory() as session:
            source_name, license_reference = _training_sources(session)
            matches = available_matches(session, league)
            job = session.get(TrainingJob, job_id)
            if job and job.job_type == "backtest":
                report = expanding_window_backtest(matches)
                job.status = "completed"
                job.completed_at = utc_now()
                job.progress_message = "Time-safe backtest completed"
                job.result = {
                    "match_count": len(matches),
                    "report": report,
                }
                session.add(AdminAudit(
                    actor="pickfoot-worker",
                    action="backtest_completed",
                    entity_type="training_job",
                    entity_id=job_id,
                    details={
                        "league": league,
                        "source_data_hash": report["source_data_hash"],
                    },
                ))
                session.commit()
                logger.info("Backtest job %s completed for %s", job_id, league)
                return
            incumbent = None
            active_pointer = session.get(ActiveModel, league)
            default_artifact = (
                get_settings().model_storage_dir / f"{model_slug(league)}.joblib"
            )
            if active_pointer is not None or default_artifact.is_file():
                incumbent, _ = model_for_league(session, league)
            job = session.get(TrainingJob, job_id)
            if job:
                job.progress_message = f"Training on {len(matches)} completed matches"
                session.commit()

            candidate = train_model(
                matches,
                source_name=source_name,
                license_reference=license_reference,
                data_kind="authorized_real",
                source_url=None,
            )
            candidate.model_version = (
                "pickfoot-pre-match-logreg-1+"
                + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                + "-" + uuid4().hex[:8]
            )
            comparison = (
                compare_models_on_candidate_window(incumbent, candidate, matches)
                if incumbent is not None
                else None
            )
            artifact_path = _save_versioned_artifact(candidate)
            version = ModelVersion(
                league=league,
                version=candidate.model_version,
                artifact_path=str(artifact_path),
                source_data_hash=matches_fingerprint(matches),
                evaluation=evaluation_dict(candidate),
                comparison=comparison,
                training_cutoff=candidate.state.latest_date,
                candidate_status="candidate",
            )
            session.add(version)
            session.flush()
            job = session.get(TrainingJob, job_id)
            if job is None:
                raise RuntimeError("Training job disappeared before completion.")
            job.status = "completed"
            job.completed_at = utc_now()
            job.progress_message = "Candidate ready for review"
            job.candidate_version_id = version.id
            job.result = {
                "match_count": len(matches),
                "training_cutoff": candidate.state.latest_date.isoformat(),
                "evaluation": evaluation_dict(candidate),
                "comparison": comparison,
                "artifact_path": str(artifact_path),
            }
            session.add(AdminAudit(
                actor="pickfoot-worker",
                action="candidate_trained",
                entity_type="model_version",
                entity_id=version.id,
                details={
                    "job_id": job_id,
                    "league": league,
                    "source_data_hash": version.source_data_hash,
                },
            ))
            session.commit()
            logger.info("Training job %s completed for %s; candidate %s", job_id, league, version.version)
    except Exception as exc:
        message = _safe_error(exc)
        with factory() as session:
            job = session.get(TrainingJob, job_id)
            if job:
                job.status = "failed"
                job.completed_at = utc_now()
                job.progress_message = "Candidate training failed"
                job.error_message = message
                session.commit()
        if artifact_path and artifact_path.exists():
            artifact_path.unlink(missing_ok=True)
        logger.exception("Training job %s failed: %s", job_id, message)


def _mark_interrupted_jobs(factory) -> None:
    with factory.begin() as session:
        interrupted = session.scalars(
            select(TrainingJob).where(TrainingJob.status == "running")
        ).all()
        for job in interrupted:
            job.status = "failed"
            job.completed_at = utc_now()
            job.progress_message = "Worker restarted before job completion"
            job.error_message = "The worker stopped while this job was running. Queue a new candidate run."


def run_worker(*, once: bool = False) -> None:
    factory = session_factory()
    _mark_interrupted_jobs(factory)
    while True:
        try:
            job_id = _claim_job(factory)
            if job_id:
                process_job(job_id, factory=factory)
            elif once:
                return
            else:
                time.sleep(get_settings().worker_poll_seconds)
        except Exception:
            logger.exception("Worker loop failed")
            if once:
                raise
            time.sleep(get_settings().worker_poll_seconds)


def main() -> None:
    run_worker()


if __name__ == "__main__":
    main()
