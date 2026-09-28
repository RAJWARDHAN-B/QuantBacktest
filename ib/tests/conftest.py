from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from tenacity import wait_none

from ib_agent.data.cache import FileCache
from ib_agent.data.edgar import TICKERS_URL, EdgarClient

FIXTURES = Path(__file__).parent / "fixtures"
EDGAR_FIXTURES = FIXTURES / "edgar"
TEST_USER_AGENT = "IB Agent Tests tests@example.com"

ACME_10K_URL = "https://www.sec.gov/Archives/edgar/data/1/000000000125000005/acme-20241231.htm"
ACME_10Q_URL = "https://www.sec.gov/Archives/edgar/data/1/000000000125000012/acme-20250331.htm"


def default_routes() -> dict[str, httpx.Response]:
    def ok(name: str) -> httpx.Response:
        return httpx.Response(200, content=(EDGAR_FIXTURES / name).read_bytes())

    return {
        TICKERS_URL: ok("company_tickers.json"),
        "https://data.sec.gov/submissions/CIK0000000001.json": ok("submissions_CIK0000000001.json"),
        "https://data.sec.gov/submissions/CIK0000000002.json": ok("submissions_CIK0000000002.json"),
        "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json": ok(
            "companyfacts_CIK0000000001.json"
        ),
        ACME_10K_URL: ok("acme-20241231.htm"),
        ACME_10Q_URL: ok("acme-20250331.htm"),
    }


class FakeSEC:
    """In-memory SEC. A route may be a response, or a list of responses served in order."""

    def __init__(self, routes: dict[str, httpx.Response | list[httpx.Response]]) -> None:
        self.routes = routes
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        route = self.routes.get(str(request.url))
        if route is None:
            return httpx.Response(404)
        if isinstance(route, list):
            return route.pop(0) if len(route) > 1 else route[0]
        return route

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler))


class NoopLimiter:
    def acquire(self) -> None:
        return None


MakeClient = Callable[..., tuple[EdgarClient, FakeSEC]]


@pytest.fixture
def make_client(tmp_path: Path) -> MakeClient:
    def factory(
        routes: dict[str, httpx.Response | list[httpx.Response]] | None = None,
        **kwargs: object,
    ) -> tuple[EdgarClient, FakeSEC]:
        fake = FakeSEC(default_routes() if routes is None else routes)
        options: dict[str, object] = {
            "http_client": fake.client(),
            "rate_limiter": NoopLimiter(),
            "retry_wait": wait_none(),
        }
        options.update(kwargs)
        client = EdgarClient(TEST_USER_AGENT, FileCache(tmp_path / "http"), **options)  # type: ignore[arg-type]
        return client, fake

    return factory
