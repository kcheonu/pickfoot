"""Validated request and response shapes for the HTTP API."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=120)
    password: str = Field(min_length=1, max_length=1024)


class MatchCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    league: str
    season: str = "2026-27"
    round_name: str = Field(min_length=1, max_length=80)
    match_date: date
    home_team: str = Field(min_length=1, max_length=160)
    away_team: str = Field(min_length=1, max_length=160)

    @model_validator(mode="after")
    def valid_fixture(self) -> "MatchCreate":
        if self.season != "2026-27":
            raise ValueError("Only the 2026-27 season can be managed in this MVP.")
        if self.home_team.casefold() == self.away_team.casefold():
            raise ValueError("Home and away teams must differ.")
        if self.league not in {
            "Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"
        }:
            raise ValueError("Unsupported league.")
        if not (date(2026, 7, 1) <= self.match_date <= date(2027, 6, 30)):
            raise ValueError("Match date must fall within the 2026-27 season.")
        return self


class TeamRegistration(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    league: str
    team_name: str = Field(min_length=1, max_length=160)
    registration_note: str = Field(min_length=5, max_length=500)

    @model_validator(mode="after")
    def valid_team(self) -> "TeamRegistration":
        if self.league not in {
            "Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"
        }:
            raise ValueError("Unsupported league.")
        return self


class MatchResultUpdate(BaseModel):
    home_goals: int = Field(ge=0, le=30)
    away_goals: int = Field(ge=0, le=30)
    expected_version: int = Field(ge=1)
    source_name: str = Field(min_length=1, max_length=200)
    license_reference: str = Field(min_length=1, max_length=500)
    rights_confirmed: bool
    correction_confirmed: bool = False
    reason: str = Field(min_length=4, max_length=1000)

    @model_validator(mode="after")
    def require_rights(self) -> "MatchResultUpdate":
        if not self.rights_confirmed:
            raise ValueError("Confirm the source and permitted use before saving a result.")
        return self


class MatchStatusUpdate(BaseModel):
    status: Literal["postponed", "cancelled", "scheduled"]
    expected_version: int = Field(ge=1)
    reason: str = Field(min_length=4, max_length=1000)


class PredictionRequest(BaseModel):
    league: str
    home_team: str
    away_team: str
    match_date: date
    season: str = "2026-27"


class TrainingRequest(BaseModel):
    league: str
