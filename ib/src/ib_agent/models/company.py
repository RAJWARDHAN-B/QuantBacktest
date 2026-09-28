from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from typing import Any

from pydantic import BaseModel, Field

# SIC 6000-6799: banks, insurers, REITs, and other financials; UFCF DCF is not appropriate.
FINANCIAL_SIC_RANGE = range(6000, 6800)


class CompanyRef(BaseModel):
    ticker: str
    cik: int = Field(gt=0)
    name: str


class FilingRef(BaseModel):
    accession_number: str = Field(pattern=r"^\d{10}-\d{2}-\d{6}$")
    form: str
    filing_date: date
    report_date: date | None = None
    primary_document: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

    def document_url(self, cik: int) -> str:
        folder = self.accession_number.replace("-", "")
        return f"https://www.sec.gov/Archives/edgar/data/{cik}/{folder}/{self.primary_document}"


class CompanyProfile(BaseModel):
    cik: int
    ticker: str
    name: str
    sic: int | None = None
    sic_description: str | None = None
    fiscal_year_end: str | None = None
    exchanges: list[str] = Field(default_factory=list)
    filings: list[FilingRef] = Field(default_factory=list)

    @property
    def is_financial_sector(self) -> bool:
        return self.sic is not None and self.sic in FINANCIAL_SIC_RANGE

    def latest_filing(self, forms: Iterable[str]) -> FilingRef | None:
        wanted = set(forms)
        matches = [f for f in self.filings if f.form in wanted]
        return max(matches, key=lambda f: f.filing_date, default=None)


def parse_submissions(ticker: str, payload: dict[str, Any]) -> CompanyProfile:
    """Build a profile from the EDGAR submissions JSON.

    Filings whose primary document is not a plain file name (e.g. XSL-rendered
    ownership forms) are skipped; they are never needed for valuation.
    """
    recent = payload.get("filings", {}).get("recent", {})
    columns = zip(
        recent.get("accessionNumber", []),
        recent.get("form", []),
        recent.get("filingDate", []),
        recent.get("reportDate", []),
        recent.get("primaryDocument", []),
        strict=True,
    )
    filings: list[FilingRef] = []
    for accession, form, filed, reported, document in columns:
        try:
            filings.append(
                FilingRef(
                    accession_number=accession,
                    form=form,
                    filing_date=filed,
                    report_date=reported or None,
                    primary_document=document,
                )
            )
        except ValueError:
            continue

    sic_raw = payload.get("sic")
    return CompanyProfile(
        cik=int(payload["cik"]),
        ticker=ticker,
        name=payload.get("name", ""),
        sic=int(sic_raw) if sic_raw else None,
        sic_description=payload.get("sicDescription") or None,
        fiscal_year_end=payload.get("fiscalYearEnd") or None,
        exchanges=[e for e in payload.get("exchanges", []) if e],
        filings=filings,
    )
