# IB Analyst Agent

> **Status: planning.** No code has been written yet. The design and milestones are in [implementation.md](implementation.md). Commands below show the planned interface.

Give the agent a ticker. It returns what a junior investment banking analyst would spend a weekend building:

- **`model.xlsx`**: a live-formula DCF model (Inputs → Historicals → Forecast → WACC → DCF → Sensitivity → Comps → Checks).
- **`deck.pptx` / `deck.pdf`**: a preliminary valuation pitch deck with a football field, sensitivity tables, trading comps, and cited commentary.
- **`run_manifest.json`**: a full audit trail showing where every number came from (SEC accession number, XBRL tag, period), the assumptions used, and every LLM call.

## How It Works

```mermaid
flowchart LR
    T[Ticker] --> R[Research<br/>SEC EDGAR · market data]
    R --> E[Extraction<br/>XBRL → IS/BS/CF + QA]
    E --> A[Assumptions<br/>baseline + cited LLM tweaks]
    A --> V[Valuation<br/>WACC · DCF · sensitivities]
    R --> C[Comps<br/>peers · multiples]
    V --> N[Narrative<br/>cited commentary]
    C --> N
    N --> B[Excel · PPTX · PDF]
    B --> RV[Review<br/>consistency + sanity checks]
```

**Design rule: numbers come from code, words come from the LLM.** Financials come from SEC XBRL data and are valued with deterministic Python. The LLM summarizes 10-K sections, suggests assumption adjustments (bounded, with citations), proposes peers (then validated), and writes slide text. Every number in generated text is checked against the model. Every citation is checked against the filing.

## Features (planned)

- **SEC EDGAR ingestion:** ticker → CIK, company facts (XBRL), and 10-K / 10-Q documents. Rate-limited and cached, following SEC fair-access rules.
- **Three-statement extraction:** tag-fallback mapping, restatement handling, LTM roll-forward, and QA checks (balance sheet balances, cash-flow tie-out, required items).
- **DCF:** UFCF forecast, CAPM WACC with a regression beta (Blume-adjusted), Gordon growth and exit-multiple terminal values, mid-year convention, EV → equity bridge, and cross-check implied metrics.
- **Sensitivities:** implied share price across WACC × terminal growth and WACC × exit multiple.
- **Trading comps:** EV/Revenue, EV/EBITDA, and P/E with percentile statistics and implied valuation ranges.
- **Banker-style Excel:** blue inputs, black formulas, green links, named ranges, and a Checks sheet. The formulas are recalculated in LibreOffice and verified against the Python engine.
- **Pitch deck:** native, editable PowerPoint charts, a football field, and sourced footnotes. Converted to PDF.
- **Human in the loop:** `--review` pauses so you can edit `assumptions.yaml` before valuation.
- **LLM-optional:** OpenAI, Anthropic, or a local Ollama model. `--no-llm` still produces the full model and deck.
- **Fully containerized:** Python, LibreOffice, and fonts are all inside Docker.

## Quickstart (planned)

Prerequisites: Docker Desktop (or Docker Engine + Compose v2).

```bash
cd ib
cp .env.example .env          # set SEC_USER_AGENT and, optionally, an LLM provider
make build                    # or: docker compose build

# Full run
docker compose run --rm analyst analyze MSFT

# Deterministic run, no LLM
docker compose run --rm analyst analyze MSFT --no-llm

# Specify peers and pause to review assumptions
docker compose run --rm analyst analyze MSFT --peers AAPL,GOOGL,ORCL,CRM --review

# Local LLM via Ollama
docker compose --profile local-llm up -d ollama
LLM_PROVIDER=ollama docker compose run --rm analyst analyze MSFT

# Tests (network disabled inside the container)
make test
```

Artifacts are written to `ib/outputs/{TICKER}/{timestamp}/` on the host.

## Configuration

| Variable | Required | Description |
|---|---|---|
| `SEC_USER_AGENT` | yes | `"Your Name your@email.com"`. The SEC requires this on every request |
| `LLM_PROVIDER` | no | `openai`, `anthropic`, `ollama`, or `fake` (default `fake`) |
| `LLM_MODEL` | no | Model name for the chosen provider |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | if using that provider | API key |
| `OLLAMA_BASE_URL` | no | Defaults to `http://ollama:11434` inside Compose |
| `FRED_API_KEY` | no | Optional risk-free source. Default is the US Treasury yield curve |

Valuation defaults (ERP, terminal growth bounds, forecast horizon, tax fallback) live in `config/defaults.yaml`. XBRL tag mappings live in `config/xbrl_tag_map.yaml`.

## CLI (planned)

```
ib-agent analyze TICKER [--peers A,B,C] [--review] [--no-llm] [--valuation-date YYYY-MM-DD]
ib-agent fetch   TICKER                     # download + cache data
ib-agent extract TICKER                     # statements + QA report
ib-agent value   TICKER --assumptions FILE  # valuation from edited assumptions
ib-agent build   RUN_DIR                    # rebuild Excel / PPTX / PDF
ib-agent verify  RUN_DIR                    # re-run parity and review checks
```

## Project Layout (planned)

```
ib/
├── Dockerfile · docker-compose.yml · Makefile · pyproject.toml
├── config/            # defaults, XBRL tag map, curated peers
├── src/ib_agent/
│   ├── data/          # EDGAR, filings, market data, rates
│   ├── extraction/    # XBRL → statements, QA checks
│   ├── valuation/     # beta, WACC, forecast, DCF, sensitivities, comps
│   ├── llm/           # providers, prompts, guardrails
│   ├── agents/        # orchestrator + agent steps
│   ├── outputs/       # Excel, deck, PDF, recalc/parity
│   └── api/           # FastAPI (later milestone)
└── tests/             # unit, integration, golden; offline fixtures
```

## Scope and Limitations

- US filers with US-GAAP XBRL (10-K / 10-Q) only.
- Banks, insurers, and REITs (SIC 6000–6799) are refused, because an unlevered-FCF DCF isn't the right method for them.
- No consensus estimates or paid data. Multiples are LTM-based.
- Market data uses `yfinance` (unofficial) behind a swappable interface. You can override values manually.

## Roadmap

M0 scaffolding + Docker → M1 EDGAR → M2 extraction + QA → M3 WACC → M4 DCF → M5 Excel + parity → M6 comps → M7 LLM layer → M8 orchestrator → M9 deck + PDF → M10 review agent → M11 API + polish. Details and completion criteria are in [implementation.md](implementation.md#15-milestones).

## Disclaimer

For educational and portfolio purposes only. The outputs are preliminary and automatically generated, and they are **not investment advice**. Always check figures against the original filings.
