"""Research agent: gather everything the pipeline needs from SEC EDGAR for one ticker."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from ib_agent.data.edgar import EdgarClient
from ib_agent.data.tickers import normalize_ticker, resolve_ticker
from ib_agent.errors import FilingNotFoundError, UnsupportedCompanyError
from ib_agent.models.company import CompanyProfile, FilingRef, parse_submissions
from ib_agent.models.sources import SourceRecord

# Amendments (10-K/A) are often partial (e.g. Part III only), so only originals are used.
ANNUAL_FORMS = ("10-K",)
QUARTERLY_FORMS = ("10-Q",)


class ResearchResult(BaseModel):
    company: CompanyProfile
    latest_10k: FilingRef
    latest_10q: FilingRef | None
    company_facts: dict[str, Any]
    sources: list[SourceRecord]


def run_research(client: EdgarClient, ticker: str) -> ResearchResult:
    ref = resolve_ticker(client, normalize_ticker(ticker))
    profile = parse_submissions(ref.ticker, client.submissions(ref.cik))

    if profile.is_financial_sector:
        raise UnsupportedCompanyError(
            f"{profile.name} ({profile.ticker}) has SIC {profile.sic} "
            f"({profile.sic_description}). Banks, insurers, and REITs are not supported: "
            "an unlevered free cash flow DCF is not an appropriate valuation method for them."
        )

    latest_10k = profile.latest_filing(ANNUAL_FORMS)
    if latest_10k is None:
        raise FilingNotFoundError(f"No 10-K found in recent filings for {profile.ticker}.")

    latest_10q = profile.latest_filing(QUARTERLY_FORMS)
    if latest_10q is not None and latest_10q.filing_date <= latest_10k.filing_date:
        # A 10-Q older than the latest 10-K adds nothing for LTM.
        latest_10q = None

    facts = client.company_facts(ref.cik)
    client.filing_document(latest_10k, ref.cik)
    if latest_10q is not None:
        client.filing_document(latest_10q, ref.cik)

    return ResearchResult(
        company=profile,
        latest_10k=latest_10k,
        latest_10q=latest_10q,
        company_facts=facts,
        sources=list(client.sources),
    )
