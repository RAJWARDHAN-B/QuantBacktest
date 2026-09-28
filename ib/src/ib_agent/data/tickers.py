"""Ticker validation and ticker -> CIK resolution."""

from __future__ import annotations

import re

from ib_agent.data.edgar import EdgarClient
from ib_agent.errors import InvalidTickerError, TickerNotFoundError
from ib_agent.models.company import CompanyRef

_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


def normalize_ticker(raw: str) -> str:
    """Upper-case and validate. The result is safe to use in URLs and file paths."""
    ticker = raw.strip().upper()
    if not _TICKER_RE.fullmatch(ticker):
        raise InvalidTickerError(f"Invalid ticker: {raw!r}")
    return ticker


def resolve_ticker(client: EdgarClient, ticker: str) -> CompanyRef:
    ticker = normalize_ticker(ticker)
    # SEC lists share classes with a dash (BRK-B); users often type a dot (BRK.B).
    candidates = {ticker, ticker.replace(".", "-")}
    for row in client.company_tickers().values():
        if str(row.get("ticker", "")).upper() in candidates:
            return CompanyRef(
                ticker=row["ticker"].upper(), cik=int(row["cik_str"]), name=row["title"]
            )
    raise TickerNotFoundError(f"Ticker {ticker} not found in SEC company list.")
