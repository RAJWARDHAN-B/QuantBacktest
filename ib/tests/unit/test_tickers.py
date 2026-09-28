from __future__ import annotations

import pytest

from ib_agent.data.tickers import normalize_ticker, resolve_ticker
from ib_agent.errors import InvalidTickerError, TickerNotFoundError


@pytest.mark.parametrize(
    ("raw", "expected"), [("msft", "MSFT"), (" brk.b ", "BRK.B"), ("BF-B", "BF-B")]
)
def test_normalize_valid(raw: str, expected: str) -> None:
    assert normalize_ticker(raw) == expected


@pytest.mark.parametrize("raw", ["", "1ABC", "../etc", "A/B", "TOOLONGTICKER", "MS FT"])
def test_normalize_invalid(raw: str) -> None:
    with pytest.raises(InvalidTickerError):
        normalize_ticker(raw)


def test_resolve(make_client) -> None:
    client, _ = make_client()
    ref = resolve_ticker(client, "acme")
    assert (ref.ticker, ref.cik, ref.name) == ("ACME", 1, "Acme Widgets Inc.")


def test_resolve_dot_share_class(make_client) -> None:
    client, _ = make_client()
    assert resolve_ticker(client, "brk.b").ticker == "BRK-B"


def test_resolve_unknown(make_client) -> None:
    client, _ = make_client()
    with pytest.raises(TickerNotFoundError):
        resolve_ticker(client, "ZZZZ")
