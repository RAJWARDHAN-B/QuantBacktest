from __future__ import annotations

from pathlib import Path

import pytest

from ib_agent.config import Settings, load_defaults
from ib_agent.errors import ConfigError

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def test_repo_defaults_load() -> None:
    defaults = load_defaults(CONFIG_DIR)
    assert defaults.edgar.requests_per_second <= 10
    assert defaults.valuation.terminal_growth <= defaults.valuation.max_terminal_growth


def test_missing_defaults_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Missing defaults"):
        load_defaults(tmp_path)


def test_unknown_key_rejected(tmp_path: Path) -> None:
    text = (CONFIG_DIR / "defaults.yaml").read_text() + "\ntypo_section: 1\n"
    (tmp_path / "defaults.yaml").write_text(text)
    with pytest.raises(ConfigError, match="typo_section"):
        load_defaults(tmp_path)


def test_settings_read_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SEC_USER_AGENT", "Jane Doe jane@example.com")
    monkeypatch.setenv("IB_AGENT_CACHE_DIR", str(tmp_path))
    settings = Settings(_env_file=None)
    assert settings.require_sec_user_agent() == "Jane Doe jane@example.com"
    assert settings.cache_dir == tmp_path


@pytest.mark.parametrize("agent", [None, "", "Jane Doe", "jane at example"])
def test_user_agent_requires_email(agent: str | None) -> None:
    settings = Settings(_env_file=None, sec_user_agent=agent)
    with pytest.raises(ConfigError, match="SEC_USER_AGENT"):
        settings.require_sec_user_agent()
