"""Command-line interface: `ib-agent`."""

from __future__ import annotations

from typing import Annotated

import typer

from ib_agent import __version__
from ib_agent.agents.research import ResearchResult, run_research
from ib_agent.config import Defaults, Settings, get_settings, load_defaults
from ib_agent.data.edgar import EdgarClient
from ib_agent.errors import IBAgentError

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Autonomous AI investment banking analyst: SEC filings to DCF model and pitch deck.",
)


def build_edgar_client(settings: Settings, defaults: Defaults, *, refresh: bool) -> EdgarClient:
    return EdgarClient.from_settings(settings, defaults, refresh=refresh)


@app.callback()
def main() -> None:
    """Autonomous AI investment banking analyst."""


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command()
def fetch(
    ticker: Annotated[str, typer.Argument(help="US-listed ticker, e.g. MSFT")],
    refresh: Annotated[
        bool, typer.Option("--refresh", help="Ignore cached SEC metadata (filings stay cached).")
    ] = False,
) -> None:
    """Download and cache SEC data for TICKER: profile, XBRL facts, latest 10-K / 10-Q."""
    try:
        settings = get_settings()
        defaults = load_defaults(settings.config_dir)
        with build_edgar_client(settings, defaults, refresh=refresh) as client:
            result = run_research(client, ticker)
            _print_research(result, client.network_requests)
    except IBAgentError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def _print_research(result: ResearchResult, network_requests: int) -> None:
    company = result.company
    ten_k, ten_q = result.latest_10k, result.latest_10q
    concepts = len(result.company_facts.get("facts", {}).get("us-gaap", {}))
    cache_hits = sum(s.from_cache for s in result.sources)

    typer.echo(f"{company.name} ({company.ticker})  CIK {company.cik:010d}")
    typer.echo(f"SIC {company.sic} {company.sic_description or ''}  FYE {company.fiscal_year_end}")
    typer.echo(
        f"Latest 10-K: {ten_k.accession_number} filed {ten_k.filing_date} "
        f"(period {ten_k.report_date})"
    )
    if ten_q is not None:
        typer.echo(
            f"Latest 10-Q: {ten_q.accession_number} filed {ten_q.filing_date} "
            f"(period {ten_q.report_date})"
        )
    else:
        typer.echo("Latest 10-Q: none since the latest 10-K")
    typer.echo(f"XBRL us-gaap concepts: {concepts}")
    typer.echo(f"HTTP requests: {network_requests}  cache hits: {cache_hits}")
