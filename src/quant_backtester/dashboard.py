"""Local interactive dashboard for historical strategy research."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from quant_backtester.backtest import run_backtest
from quant_backtester.data import download_history, load_csv, validate_ohlcv
from quant_backtester.multi_asset import (
    asset_return_correlation,
    exposure_report,
    run_multi_asset_backtest,
)
from quant_backtester.performance import drawdown_series, performance_report
from quant_backtester.returns import buy_and_hold
from quant_backtester.strategy import MeanReversionStrategy


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Quant Terminal", page_icon="Q", layout="wide")
    st.markdown(
        """
        <style>
        :root { color-scheme: dark; --terminal-bg: #101416; --terminal-panel: #171d1f; --terminal-line: #354044; --terminal-text: #e4e9e6; --terminal-lime: #c8f169; --terminal-cyan: #5ec8d4; }
        .stApp { background: var(--terminal-bg); color: var(--terminal-text); }
        [data-testid="stSidebar"] { background: var(--terminal-panel); border-right: 1px solid var(--terminal-line); }
        h1, h2, h3 { color: var(--terminal-text); letter-spacing: 0; }
        .terminal-kicker { color: #91a39d; font: 12px monospace; text-transform: uppercase; }
        .terminal-title { color: var(--terminal-lime); font: 700 30px monospace; margin: 0 0 6px 0; }
        .terminal-rule { border-bottom: 1px solid var(--terminal-line); margin: 10px 0 20px 0; }
        [data-testid="stMetric"] { background: var(--terminal-panel); border: 1px solid var(--terminal-line); padding: 10px 12px; }
        [data-testid="stMetricValue"] { color: var(--terminal-lime); font-family: monospace; font-size: 20px; white-space: nowrap; }
        [data-testid="stMetricLabel"] { font-size: 13px; }
        [data-testid="stAlert"] { background: var(--terminal-panel); border: 1px solid var(--terminal-line); color: var(--terminal-text); }
        .stButton > button { background: var(--terminal-lime); border: 1px solid var(--terminal-lime); color: var(--terminal-bg); font-weight: 700; }
        .stButton > button:hover { background: #d9ff8a; border-color: #d9ff8a; color: var(--terminal-bg); }
        button[role="tab"][aria-selected="true"] { color: var(--terminal-lime); border-bottom-color: var(--terminal-lime); }
        .stTabs [data-baseweb="tab-highlight"] { background-color: var(--terminal-lime); }
        div[data-testid="stDataFrame"] { border: 1px solid var(--terminal-line); }
        </style>
        <div class="terminal-kicker">Quant Research / Historical Simulation</div>
        <div class="terminal-title">MARKET LAB</div>
        <div class="terminal-rule"></div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.subheader("Run configuration")
        universe = st.radio("Universe", ["Single asset", "Multi-asset"], horizontal=True)
        source = st.radio("Market data", ["Local CSV", "Upload CSV", "Yahoo Finance"])
        symbol_input = st.text_input("Symbol(s)", value="SPY")
        symbols = [value.strip().upper() for value in symbol_input.split(",") if value.strip()]
        uploaded = None
        local_path = None
        local_paths = []
        start = None
        end = None
        if source == "Local CSV":
            available_files = sorted(Path("data").glob("*.csv"))
            if available_files:
                if universe == "Multi-asset":
                    local_paths = st.multiselect(
                        "Datasets",
                        available_files,
                        format_func=lambda path: path.name,
                    )
                else:
                    local_path = st.selectbox("Dataset", available_files, format_func=lambda path: path.name)
            else:
                st.caption("No local CSV files found in data/.")
        elif source == "Upload CSV":
            uploaded = st.file_uploader(
                "OHLCV CSV",
                type=["csv"],
                accept_multiple_files=universe == "Multi-asset",
            )
        else:
            start = st.date_input("Start date", value=date.today() - timedelta(days=365 * 5))
            end = st.date_input("End date", value=date.today())

        lookback = st.number_input("Lookback bars", min_value=2, max_value=500, value=20)
        threshold = st.number_input("Entry z-score", min_value=0.1, max_value=5.0, value=1.5, step=0.1)
        initial_cash = st.number_input("Initial cash", min_value=1.0, value=10000.0, step=1000.0)
        commission = st.number_input("Commission per fill", min_value=0.0, value=0.0, step=0.5)
        slippage_bps = st.number_input("Slippage (bps)", min_value=0.0, value=5.0, step=1.0)
        market_impact_bps = st.number_input("Impact at 100% volume (bps)", min_value=0.0, value=0.0, step=1.0)
        max_position_size = st.number_input("Max shares (0 = uncapped)", min_value=0, value=0)
        max_portfolio_exposure = st.slider("Max gross exposure", min_value=0.1, max_value=1.0, value=1.0, step=0.05)
        run_clicked = st.button("Run backtest", type="primary", use_container_width=True)

    if run_clicked:
        try:
            if universe == "Multi-asset":
                if source == "Local CSV":
                    if not local_paths:
                        raise ValueError("Choose at least two local CSV datasets")
                    prices_by_symbol = {path.stem.upper(): load_csv(path) for path in local_paths}
                elif source == "Upload CSV":
                    uploads = uploaded if isinstance(uploaded, list) else ([uploaded] if uploaded else [])
                    if len(uploads) < 2:
                        raise ValueError("Upload at least two OHLCV CSV files")
                    prices_by_symbol = {
                        Path(file.name).stem.upper(): validate_ohlcv(
                            pd.read_csv(file, index_col=0, parse_dates=True)
                        )
                        for file in uploads
                    }
                else:
                    if len(set(symbols)) < 2:
                        raise ValueError("Enter at least two comma-separated symbols")
                    if start is None or end is None or start >= end:
                        raise ValueError("The end date must be later than the start date")
                    prices_by_symbol = {
                        symbol: download_history(symbol, start.isoformat(), end.isoformat())
                        for symbol in symbols
                    }
                if len(prices_by_symbol) < 2:
                    raise ValueError("Choose at least two uniquely named assets")
                strategies = {
                    symbol: MeanReversionStrategy(
                        lookback=int(lookback), z_threshold=float(threshold)
                    )
                    for symbol in prices_by_symbol
                }
                result = run_multi_asset_backtest(
                    prices_by_symbol,
                    strategies,
                    initial_cash=float(initial_cash),
                    commission=float(commission),
                    slippage_bps=float(slippage_bps),
                    max_position_size=int(max_position_size) or None,
                    max_portfolio_exposure=float(max_portfolio_exposure),
                    market_impact_bps=float(market_impact_bps),
                )
                metrics = performance_report(result.equity_curve, result.trades)
                st.session_state["backtest_correlation"] = asset_return_correlation(prices_by_symbol)
                st.session_state["backtest_exposure"] = exposure_report(result.equity_curve)
                st.session_state["backtest_prices_by_symbol"] = prices_by_symbol
                st.session_state["backtest_universe"] = "Multi-asset"
                st.session_state["backtest_symbol"] = ", ".join(prices_by_symbol)
            else:
                symbol = symbols[0] if symbols else ""
                if not symbol:
                    raise ValueError("Enter a symbol")
                if source == "Local CSV":
                    if local_path is None:
                        raise ValueError("Add a CSV under data/ or choose another data source")
                    prices = load_csv(local_path)
                elif source == "Upload CSV":
                    if uploaded is None:
                        raise ValueError("Choose a CSV file to upload")
                    prices = validate_ohlcv(pd.read_csv(uploaded, index_col=0, parse_dates=True))
                else:
                    if start is None or end is None or start >= end:
                        raise ValueError("The end date must be later than the start date")
                    prices = download_history(symbol, start.isoformat(), end.isoformat())

                result = run_backtest(
                    prices,
                    symbol=symbol,
                    strategy=MeanReversionStrategy(lookback=int(lookback), z_threshold=float(threshold)),
                    initial_cash=float(initial_cash),
                    commission=float(commission),
                    slippage_bps=float(slippage_bps),
                    max_position_size=int(max_position_size) or None,
                    max_portfolio_exposure=float(max_portfolio_exposure),
                    market_impact_bps=float(market_impact_bps),
                )
                st.session_state["backtest_prices"] = prices
                st.session_state["backtest_universe"] = "Single asset"
                st.session_state["backtest_symbol"] = symbol
                st.session_state.pop("backtest_correlation", None)
                st.session_state.pop("backtest_exposure", None)

            st.session_state["backtest_result"] = result
            st.session_state["backtest_metrics"] = metrics
        except (ValueError, OSError) as error:
            st.error(str(error))

    result = st.session_state.get("backtest_result")
    if result is None:
        st.info("Load historical OHLCV data, set execution assumptions, and run a backtest.")
        return

    metrics = st.session_state["backtest_metrics"]
    symbol = st.session_state["backtest_symbol"]
    multi_asset = st.session_state.get("backtest_universe") == "Multi-asset"
    if multi_asset:
        datasets = st.session_state["backtest_prices_by_symbol"]
        first_date = min(frame.index[0] for frame in datasets.values()).date()
        last_date = max(frame.index[-1] for frame in datasets.values()).date()
        bar_count = sum(len(frame) for frame in datasets.values())
    else:
        prices = st.session_state["backtest_prices"]
        first_date, last_date, bar_count = prices.index[0].date(), prices.index[-1].date(), len(prices)
    st.caption(f"{symbol}  |  {first_date} to {last_date}  |  {bar_count:,} daily bars  |  next-open fills")

    metric_columns = st.columns(5)
    metric_columns[0].metric("Final equity", f"${metrics['final_equity']:,.2f}")
    metric_columns[1].metric("Total return", f"{metrics['total_return']:.2%}")
    metric_columns[2].metric("Sharpe", "N/A" if metrics["sharpe_ratio"] is None else f"{metrics['sharpe_ratio']:.2f}")
    metric_columns[3].metric("Max DD", f"{metrics['max_drawdown']:.2%}")
    metric_columns[4].metric("Closed trades", str(metrics["trade_count"]))

    if multi_asset:
        chart_data = result.equity_curve[["equity"]].rename(columns={"equity": "Portfolio equity"})
        tab_names = ["Portfolio", "Risk", "Trades", "Metrics", "Allocation"]
    else:
        baseline = buy_and_hold(prices, initial_capital=float(metrics["start_equity"]))
        chart_data = pd.DataFrame(
            {
                "Strategy": result.equity_curve["equity"],
                "Buy and hold": baseline["equity"].reindex(result.equity_curve.index),
            }
        )
        tab_names = ["Equity", "Risk", "Trades", "Metrics"]
    tabs = st.tabs(tab_names)
    chart_tab, risk_tab, trades_tab, metrics_tab = tabs[:4]
    with chart_tab:
        colors = ["#c8f169"] if multi_asset else ["#c8f169", "#5ec8d4"]
        st.line_chart(chart_data, color=colors, height=360)
    with risk_tab:
        st.area_chart(drawdown_series(result.equity_curve), color="#e7a84b", height=300)
        st.caption(f"Maximum underwater duration: {metrics['max_drawdown_duration']} bars")
    with trades_tab:
        st.subheader("Filled trades")
        st.dataframe(result.trades, use_container_width=True, hide_index=True)
        st.subheader("Unfilled quantities")
        st.dataframe(result.rejections, use_container_width=True, hide_index=True)
        st.download_button(
            "Download trade ledger",
            result.trades.to_csv(index=False),
            file_name=f"{symbol.lower()}_trades.csv",
            mime="text/csv",
        )
    with metrics_tab:
        st.dataframe(
            pd.DataFrame([metrics]).T.rename(columns={0: "Value"}),
            use_container_width=True,
        )
    if multi_asset:
        with tabs[4]:
            correlation = st.session_state["backtest_correlation"]
            st.subheader("Return correlation")
            st.dataframe(
                correlation.style.format("{:.2f}").background_gradient(
                    cmap="RdYlGn", vmin=-1.0, vmax=1.0
                ),
                use_container_width=True,
            )
            exposure = st.session_state["backtest_exposure"]
            st.subheader("Gross exposure and concentration")
            st.area_chart(exposure[["gross_exposure_fraction"]], color="#e7a84b", height=220)
            st.dataframe(exposure, use_container_width=True)


if __name__ == "__main__":
    main()