from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from ib_agent import __version__, cli
from ib_agent.config import Settings

runner = CliRunner()


def test_help_lists_commands() -> None:
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    assert "fetch" in result.output and "version" in result.output


def test_version() -> None:
    result = runner.invoke(cli.app, ["version"])
    assert result.exit_code == 0
    assert result.output.strip() == __version__


def _use_settings(monkeypatch: pytest.MonkeyPatch, **overrides: object) -> None:
    settings = Settings(_env_file=None, **overrides)  # type: ignore[arg-type]
    monkeypatch.setattr(cli, "get_settings", lambda: settings)


def test_fetch_rejects_invalid_ticker(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_settings(monkeypatch, sec_user_agent="T t@example.com")
    result = runner.invoke(cli.app, ["fetch", "../etc"])
    assert result.exit_code == 1
    assert "Invalid ticker" in result.output


def test_fetch_requires_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_settings(monkeypatch, sec_user_agent=None)
    result = runner.invoke(cli.app, ["fetch", "ACME"])
    assert result.exit_code == 1
    assert "SEC_USER_AGENT" in result.output


def test_fetch_end_to_end(monkeypatch: pytest.MonkeyPatch, make_client, tmp_path: Path) -> None:
    _use_settings(monkeypatch, sec_user_agent="T t@example.com", cache_dir=tmp_path)
    client, _ = make_client()
    monkeypatch.setattr(cli, "build_edgar_client", lambda *a, **k: client)

    result = runner.invoke(cli.app, ["fetch", "ACME"])

    assert result.exit_code == 0, result.output
    assert "Acme Widgets Inc. (ACME)  CIK 0000000001" in result.output
    assert "Latest 10-K: 0000000001-25-000005" in result.output
    assert "HTTP requests: 5  cache hits: 0" in result.output
