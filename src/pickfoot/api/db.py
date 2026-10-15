"""SQLAlchemy engine and request-scoped sessions."""

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from pickfoot.api.config import get_settings

_engine = None
_session_factory = None


def session_factory() -> sessionmaker[Session]:
    global _engine, _session_factory
    if _session_factory is None:
        settings = get_settings()
        connect_args = (
            {"check_same_thread": False}
            if settings.database_url.startswith("sqlite")
            else {}
        )
        _engine = create_engine(
            settings.database_url,
            pool_pre_ping=True,
            connect_args=connect_args,
        )
        _session_factory = sessionmaker(
            bind=_engine, autoflush=False, expire_on_commit=False
        )
    return _session_factory


def get_db() -> Generator[Session, None, None]:
    session = session_factory()()
    try:
        yield session
    finally:
        session.close()
