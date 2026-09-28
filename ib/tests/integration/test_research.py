from __future__ import annotations

import json

import httpx
import pytest

from ib_agent.agents.research import run_research
from ib_agent.errors import FilingNotFoundError, UnsupportedCompanyError

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK0000000001.json"


def test_research_happy_path(make_client) -> None:
    client, fake = make_client()
    result = run_research(client, "acme")

    assert result.company.ticker == "ACME"
    assert result.latest_10k.accession_number == "0000000001-25-000005"
    assert result.latest_10q is not None
    assert result.latest_10q.accession_number == "0000000001-25-000012"
    assert "Revenues" in result.company_facts["facts"]["us-gaap"]

    fetched = [str(r.url) for r in fake.requests]
    assert any(u.endswith("acme-20241231.htm") for u in fetched)
    assert any(u.endswith("acme-20250331.htm") for u in fetched)
    assert len(result.sources) == len(fetched) == 5


def test_second_run_hits_cache_only(make_client) -> None:
    client, fake = make_client()
    run_research(client, "ACME")
    before = len(fake.requests)
    result = run_research(client, "ACME")
    assert len(fake.requests) == before
    assert all(s.from_cache for s in result.sources[before:])


def test_financials_are_refused(make_client) -> None:
    client, fake = make_client()
    with pytest.raises(UnsupportedCompanyError, match="SIC 6022"):
        run_research(client, "BNK")
    assert not any("companyfacts" in str(r.url) for r in fake.requests)


def test_missing_10k(make_client) -> None:
    client, fake = make_client()
    payload = json.loads(fake.routes[SUBMISSIONS_URL].content)
    recent = payload["filings"]["recent"]
    recent["form"] = ["10-Q" if f == "10-Q" else "8-K" for f in recent["form"]]
    fake.routes[SUBMISSIONS_URL] = httpx.Response(200, content=json.dumps(payload).encode())
    with pytest.raises(FilingNotFoundError):
        run_research(client, "ACME")


def test_10q_older_than_10k_is_dropped(make_client) -> None:
    client, fake = make_client()
    payload = json.loads(fake.routes[SUBMISSIONS_URL].content)
    payload["filings"]["recent"]["filingDate"][0] = "2025-01-15"
    fake.routes[SUBMISSIONS_URL] = httpx.Response(200, content=json.dumps(payload).encode())
    result = run_research(client, "ACME")
    assert result.latest_10q is None
