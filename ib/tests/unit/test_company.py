from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from ib_agent.models.company import FilingRef, parse_submissions

FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "edgar" / "submissions_CIK0000000001.json"
)


def test_parse_submissions() -> None:
    profile = parse_submissions("ACME", json.loads(FIXTURE.read_text()))
    assert profile.cik == 1
    assert profile.sic == 3570
    assert profile.fiscal_year_end == "1231"
    assert not profile.is_financial_sector
    # The Form 4 with an XSL path is skipped.
    assert [f.form for f in profile.filings] == ["10-Q", "8-K", "10-K/A", "10-K", "10-K"]


def test_latest_filing_by_form() -> None:
    profile = parse_submissions("ACME", json.loads(FIXTURE.read_text()))
    ten_k = profile.latest_filing(["10-K"])
    assert ten_k is not None
    assert ten_k.accession_number == "0000000001-25-000005"
    assert ten_k.report_date == date(2024, 12, 31)
    assert profile.latest_filing(["S-1"]) is None


def test_financial_sector_detection() -> None:
    payload = json.loads(FIXTURE.read_text())
    payload["sic"] = "6022"
    assert parse_submissions("ACME", payload).is_financial_sector


def test_document_url() -> None:
    filing = FilingRef(
        accession_number="0000320193-24-000123",
        form="10-K",
        filing_date=date(2024, 11, 1),
        primary_document="aapl-20240928.htm",
    )
    assert filing.document_url(320193) == (
        "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm"
    )
