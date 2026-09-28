"""SEC EDGAR client: mandatory User-Agent, host allow-list, rate limit, retries, disk cache."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from types import TracebackType
from typing import Any
from urllib.parse import urlsplit

import httpx
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)
from tenacity.wait import wait_base

from ib_agent.config import Defaults, Settings
from ib_agent.data.cache import FileCache
from ib_agent.data.rate_limit import RateLimiter
from ib_agent.errors import ConfigError, EdgarError
from ib_agent.models.company import FilingRef
from ib_agent.models.sources import SourceRecord

SEC_HOSTS = frozenset({"www.sec.gov", "data.sec.gov"})
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


class _RetryableStatus(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class EdgarClient:
    def __init__(
        self,
        user_agent: str,
        cache: FileCache,
        *,
        requests_per_second: float = 5.0,
        timeout_seconds: float = 30.0,
        max_attempts: int = 5,
        metadata_ttl_seconds: float = 86_400.0,
        http_client: httpx.Client | None = None,
        rate_limiter: RateLimiter | None = None,
        retry_wait: wait_base | None = None,
    ) -> None:
        if "@" not in user_agent:
            raise ConfigError("SEC User-Agent must include a contact email address.")
        self._headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
        self._cache = cache
        self._http = http_client or httpx.Client(timeout=timeout_seconds)
        self._limiter = rate_limiter or RateLimiter(requests_per_second)
        self._max_attempts = max_attempts
        self._retry_wait = retry_wait or wait_exponential_jitter(initial=1, max=30)
        self.metadata_ttl_seconds = metadata_ttl_seconds
        self.sources: list[SourceRecord] = []
        self.network_requests = 0

    @classmethod
    def from_settings(
        cls, settings: Settings, defaults: Defaults, *, refresh: bool = False
    ) -> EdgarClient:
        edgar = defaults.edgar
        return cls(
            settings.require_sec_user_agent(),
            FileCache(settings.cache_dir / "http"),
            requests_per_second=edgar.requests_per_second,
            timeout_seconds=edgar.timeout_seconds,
            max_attempts=edgar.max_attempts,
            metadata_ttl_seconds=0.0 if refresh else edgar.metadata_ttl_hours * 3600,
        )

    def __enter__(self) -> EdgarClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    # ---- endpoints ----

    def company_tickers(self) -> dict[str, Any]:
        return self.get_json(TICKERS_URL, ttl_seconds=self.metadata_ttl_seconds)

    def submissions(self, cik: int) -> dict[str, Any]:
        url = f"https://data.sec.gov/submissions/CIK{cik:010d}.json"
        return self.get_json(url, ttl_seconds=self.metadata_ttl_seconds)

    def company_facts(self, cik: int) -> dict[str, Any]:
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
        return self.get_json(url, ttl_seconds=self.metadata_ttl_seconds)

    def filing_document(self, filing: FilingRef, cik: int) -> bytes:
        # Archived filings are immutable, so they never expire from the cache.
        return self.get_bytes(filing.document_url(cik), ttl_seconds=None)

    # ---- transport ----

    def get_json(self, url: str, *, ttl_seconds: float | None) -> Any:
        body = self.get_bytes(url, ttl_seconds=ttl_seconds)
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise EdgarError(f"Invalid JSON from {url}") from exc

    def get_bytes(self, url: str, *, ttl_seconds: float | None) -> bytes:
        _check_sec_url(url)
        entry = self._cache.get(url, ttl_seconds)
        if entry is not None:
            self._record(url, entry.fetched_at, from_cache=True)
            return entry.body
        entry = self._cache.put(url, self._fetch(url))
        self._record(url, entry.fetched_at, from_cache=False)
        return entry.body

    def _fetch(self, url: str) -> bytes:
        retrying = Retrying(
            stop=stop_after_attempt(self._max_attempts),
            wait=self._retry_wait,
            retry=retry_if_exception_type((httpx.TransportError, _RetryableStatus)),
            reraise=True,
        )
        try:
            for attempt in retrying:
                with attempt:
                    self._limiter.acquire()
                    self.network_requests += 1
                    response = self._http.get(url, headers=self._headers)
                    if response.status_code == 429 or response.status_code >= 500:
                        raise _RetryableStatus(response.status_code)
                    if response.status_code != 200:
                        raise EdgarError(f"SEC returned HTTP {response.status_code} for {url}")
                    return response.content
        except (httpx.TransportError, _RetryableStatus) as exc:
            raise EdgarError(
                f"SEC request failed after {self._max_attempts} attempts ({exc}): {url}"
            ) from exc
        raise AssertionError("unreachable")

    def _record(self, url: str, fetched_at: float, *, from_cache: bool) -> None:
        self.sources.append(
            SourceRecord(
                url=url,
                retrieved_at=datetime.fromtimestamp(fetched_at, tz=UTC),
                from_cache=from_cache,
            )
        )


def _check_sec_url(url: str) -> None:
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.hostname not in SEC_HOSTS:
        raise EdgarError(f"Refusing to fetch non-SEC URL: {url}")
