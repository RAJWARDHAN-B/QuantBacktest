from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from ib_agent.data.cache import FileCache
from ib_agent.data.edgar import TICKERS_URL, EdgarClient
from ib_agent.errors import ConfigError, EdgarError

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK0000000001.json"


def test_requires_contact_email(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        EdgarClient("No Email", FileCache(tmp_path))


def test_sends_user_agent_and_parses_json(make_client) -> None:
    client, fake = make_client()
    payload = client.submissions(1)
    assert payload["name"] == "Acme Widgets Inc."
    assert fake.requests[0].headers["User-Agent"] == "IB Agent Tests tests@example.com"


def test_second_call_served_from_cache(make_client) -> None:
    client, fake = make_client()
    client.company_tickers()
    client.company_tickers()
    assert len(fake.requests) == 1
    assert client.network_requests == 1
    assert [s.from_cache for s in client.sources] == [False, True]


def test_zero_ttl_refetches_metadata(make_client) -> None:
    client, fake = make_client(metadata_ttl_seconds=0.0)
    client.company_tickers()
    client.company_tickers()
    assert len(fake.requests) == 2


@pytest.mark.parametrize(
    "url",
    [
        "http://www.sec.gov/files/company_tickers.json",
        "https://evil.example.com/x.json",
        "https://www.sec.gov.evil.example.com/x.json",
    ],
)
def test_rejects_non_sec_urls(make_client, url: str) -> None:
    client, fake = make_client()
    with pytest.raises(EdgarError, match="non-SEC"):
        client.get_bytes(url, ttl_seconds=None)
    assert fake.requests == []


def test_retries_transient_errors_then_succeeds(make_client) -> None:
    body = b'{"ok": true}'
    client, fake = make_client(
        {
            SUBMISSIONS_URL: [
                httpx.Response(503),
                httpx.Response(429),
                httpx.Response(200, content=body),
            ]
        }
    )
    assert client.submissions(1) == {"ok": True}
    assert len(fake.requests) == 3


def test_gives_up_after_max_attempts(make_client) -> None:
    client, fake = make_client({SUBMISSIONS_URL: httpx.Response(503)}, max_attempts=3)
    with pytest.raises(EdgarError, match="after 3 attempts"):
        client.submissions(1)
    assert len(fake.requests) == 3


def test_client_errors_are_not_retried(make_client) -> None:
    client, fake = make_client({})
    with pytest.raises(EdgarError, match="HTTP 404"):
        client.get_bytes(TICKERS_URL, ttl_seconds=None)
    assert len(fake.requests) == 1


def test_transport_errors_are_retried(make_client) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("boom", request=request)
        return httpx.Response(200, content=b"{}")

    client, _ = make_client(http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert client.submissions(1) == {}
    assert calls["n"] == 2


def test_invalid_json_raises(make_client) -> None:
    client, _ = make_client({SUBMISSIONS_URL: httpx.Response(200, content=b"<html>")})
    with pytest.raises(EdgarError, match="Invalid JSON"):
        client.submissions(1)
