"""Runtime settings (environment / .env) and model defaults (config/defaults.yaml)."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from ib_agent.errors import ConfigError

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    sec_user_agent: str | None = None
    llm_provider: Literal["fake", "openai", "anthropic", "ollama"] = "fake"
    llm_model: str | None = None
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    ollama_base_url: str = "http://localhost:11434"
    fred_api_key: SecretStr | None = None

    config_dir: Path = Field(
        default=_PROJECT_ROOT / "config",
        validation_alias=AliasChoices("IB_AGENT_CONFIG_DIR", "config_dir"),
    )
    cache_dir: Path = Field(
        default=_PROJECT_ROOT / ".cache",
        validation_alias=AliasChoices("IB_AGENT_CACHE_DIR", "cache_dir"),
    )
    output_dir: Path = Field(
        default=_PROJECT_ROOT / "outputs",
        validation_alias=AliasChoices("IB_AGENT_OUTPUT_DIR", "output_dir"),
    )

    def require_sec_user_agent(self) -> str:
        agent = (self.sec_user_agent or "").strip()
        if not _EMAIL_RE.search(agent):
            raise ConfigError(
                'SEC_USER_AGENT must be set to "Your Name your@email.com" '
                "(required by SEC fair-access policy). See .env.example."
            )
        return agent


class EdgarDefaults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requests_per_second: float = Field(gt=0, le=10)
    timeout_seconds: float = Field(gt=0)
    max_attempts: int = Field(ge=1, le=10)
    metadata_ttl_hours: float = Field(ge=0)


class ValuationDefaults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    forecast_years: int = Field(ge=3, le=10)
    history_years: int = Field(ge=3, le=10)
    equity_risk_premium: float = Field(gt=0, lt=0.2)
    terminal_growth: float = Field(ge=-0.02, lt=0.1)
    max_terminal_growth: float = Field(ge=0, lt=0.1)
    statutory_tax_rate: float = Field(ge=0, lt=1)
    max_tax_rate: float = Field(gt=0, lt=1)
    mid_year_convention: bool


class Defaults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    edgar: EdgarDefaults
    valuation: ValuationDefaults


def load_defaults(config_dir: Path) -> Defaults:
    path = config_dir / "defaults.yaml"
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Missing defaults file: {path}") from exc
    try:
        return Defaults.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"Invalid {path}:\n{exc}") from exc


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
