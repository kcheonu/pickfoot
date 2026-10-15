"""Environment-backed configuration for the deployable services."""

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://pickfoot:pickfoot@localhost:5432/pickfoot"
    web_origin: str = "http://localhost:3000"
    admin_username: str = "admin"
    admin_password_hash: str = ""
    session_secret: str = ""
    cookie_secure: bool = False
    environment: str = "development"
    base_matches_csv: Path = Path("data/raw/matches.csv")
    model_storage_dir: Path = Path("models")
    data_storage_dir: Path = Path("data/persistent")
    cors_origins: list[str] = []
    session_ttl_seconds: int = 8 * 60 * 60
    worker_poll_seconds: float = 2.0

    @field_validator("database_url")
    @classmethod
    def normalize_database_url(cls, value: str) -> str:
        if value.startswith("postgres://"):
            return "postgresql+psycopg://" + value.removeprefix("postgres://")
        if value.startswith("postgresql://"):
            return "postgresql+psycopg://" + value.removeprefix("postgresql://")
        return value

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
