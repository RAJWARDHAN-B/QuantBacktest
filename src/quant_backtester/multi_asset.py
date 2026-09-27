"""Shared-cash event-driven backtests across multiple symbols."""

from __future__ import annotations

from heapq import merge
from itertools import groupby
import pandas as pd

from quant_backtester.backtest import (
    BacktestResult,
    ExecutionConfig,
    _order_from_signal,
    _size_buy_order,
)
from quant_backtester.data import validate_ohlcv
from quant_backtester.events import FillEvent, OrderEvent, stream_market_events
from quant_backtester.portfolio import Portfolio
from quant_backtester.strategy import MeanReversionStrategy


def run_multi_asset_backtest(
    prices: dict[str, pd.DataFrame],
    strategies: dict[str, MeanReversionStrategy],
    initial_cash: float = 10000.0,
    commission: float = 0.0,
    slippage_bps: float = 0.0,
    max_position_size: int | None = None,
    max_portfolio_exposure: float | None = None,
    market_impact_bps: float = 0.0,
) -> BacktestResult:
    """Replay per-symbol bars in timestamp order against one shared portfolio."""
    if not prices:
        raise ValueError("At least one symbol is required")
    if set(prices) != set(strategies):
        raise ValueError("Price and strategy symbol sets must match")

    execution = ExecutionConfig(
        commission=commission,
        slippage_bps=slippage_bps,
        max_position_size=max_position_size,
        market_impact_bps=market_impact_bps,
        max_portfolio_exposure=max_portfolio_exposure,
    )
    portfolio = Portfolio(initial_cash=initial_cash)
    cleaned = {symbol: validate_ohlcv(frame) for symbol, frame in prices.items()}
    market_events = merge(
        *(stream_market_events(frame, symbol) for symbol, frame in cleaned.items()),
        key=lambda event: (event.timestamp, event.symbol),
    )
    close_history = {symbol: pd.Series(dtype=float) for symbol in cleaned}
    pending_orders: dict[str, OrderEvent] = {}
    latest_prices: dict[str, float] = {}
    equity_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
    rejection_rows: list[dict[str, object]] = []

    for timestamp, timestamp_events in groupby(market_events, key=lambda event: event.timestamp):
        for market_event in timestamp_events:
            symbol = market_event.symbol
            order = pending_orders.pop(symbol, None)
            if order is not None:
                rejection_reason = ""
                if order.side == "BUY":
                    marks = {**latest_prices, symbol: market_event.open}
                    marked_equity = portfolio.total_equity(marks)
                    current_exposure = sum(
                        portfolio.position_value(held_symbol, price)
                        for held_symbol, price in marks.items()
                    )
                    quantity, rejection_reason = _size_buy_order(
                        order.quantity,
                        market_event.open,
                        market_event.volume,
                        portfolio.cash,
                        execution.commission,
                        execution.slippage_bps,
                        execution.market_impact_bps,
                        portfolio.positions.get(symbol, 0),
                        execution.max_position_size,
                        execution.max_portfolio_exposure,
                        marked_equity,
                        current_exposure,
                    )
                else:
                    quantity = min(order.quantity, portfolio.positions.get(symbol, 0))
                    if quantity < order.quantity:
                        rejection_reason = "insufficient_position"

                if quantity < order.quantity:
                    rejection_rows.append(
                        {
                            "timestamp": market_event.timestamp,
                            "symbol": symbol,
                            "side": order.side,
                            "requested_quantity": order.quantity,
                            "filled_quantity": quantity,
                            "rejected_quantity": order.quantity - quantity,
                            "reason": rejection_reason,
                        }
                    )

                if quantity > 0:
                    fill = FillEvent(
                        timestamp=market_event.timestamp,
                        symbol=symbol,
                        side=order.side,
                        quantity=quantity,
                        price=market_event.open,
                        commission=execution.commission,
                        slippage_bps=execution.slippage_bps,
                        market_impact_bps=execution.market_impact_bps,
                        volume=market_event.volume,
                    )
                    portfolio.update_from_fill(fill)
                    trade_rows.append(
                        {
                            "timestamp": fill.timestamp,
                            "symbol": fill.symbol,
                            "side": fill.side,
                            "quantity": fill.quantity,
                            "price": fill.execution_price,
                            "commission": fill.commission,
                            "execution_price": fill.execution_price,
                            "effective_cost": fill.effective_cost,
                            "impact_bps": fill.impact_bps,
                        }
                    )

            latest_prices[symbol] = market_event.close
            close_history[symbol].loc[market_event.timestamp] = market_event.close
            signal = strategies[symbol].generate_signal(
                market_event,
                close_history[symbol],
                position=portfolio.positions.get(symbol, 0),
            )
            if signal is not None:
                next_order = _order_from_signal(signal, portfolio, market_event.close)
                if next_order is not None:
                    pending_orders[symbol] = next_order

        equity_row: dict[str, object] = {
            "timestamp": timestamp,
            "cash": portfolio.cash,
            "equity": portfolio.total_equity(latest_prices),
        }
        for symbol in cleaned:
            equity_row[f"position_{symbol}"] = portfolio.positions.get(symbol, 0)
            equity_row[f"close_{symbol}"] = latest_prices.get(symbol)
        equity_rows.append(equity_row)

    equity_curve = pd.DataFrame(equity_rows).set_index("timestamp")
    trades = pd.DataFrame(
        trade_rows,
        columns=[
            "timestamp",
            "symbol",
            "side",
            "quantity",
            "price",
            "commission",
            "execution_price",
            "effective_cost",
            "impact_bps",
        ],
    )
    rejections = pd.DataFrame(
        rejection_rows,
        columns=[
            "timestamp",
            "symbol",
            "side",
            "requested_quantity",
            "filled_quantity",
            "rejected_quantity",
            "reason",
        ],
    )
    return BacktestResult(
        equity_curve=equity_curve,
        trades=trades,
        portfolio=portfolio,
        rejections=rejections,
    )


def asset_return_correlation(prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return pairwise correlations of aligned close-to-close asset returns."""
    if len(prices) < 2:
        raise ValueError("At least two assets are required for correlation analysis")

    returns = {
        symbol: validate_ohlcv(frame)["Close"].pct_change(fill_method=None)
        for symbol, frame in prices.items()
    }
    aligned_returns = pd.concat(returns, axis=1)
    if aligned_returns.dropna(how="all").empty:
        raise ValueError("At least one return observation is required")
    return aligned_returns.corr()


def exposure_report(equity_curve: pd.DataFrame) -> pd.DataFrame:
    """Report marked gross exposure, portfolio weights, and HHI by timestamp."""
    if "equity" not in equity_curve:
        raise ValueError("Equity curve must contain an 'equity' column")
    position_columns = [column for column in equity_curve if column.startswith("position_")]
    symbols = [column.removeprefix("position_") for column in position_columns]
    if not symbols:
        raise ValueError("Equity curve must contain multi-asset position columns")

    exposures = pd.DataFrame(index=equity_curve.index)
    for symbol in symbols:
        price_column = f"close_{symbol}"
        if price_column not in equity_curve:
            raise ValueError(f"Equity curve is missing '{price_column}'")
        positions = equity_curve[f"position_{symbol}"].astype(float)
        prices = equity_curve[price_column].astype(float).fillna(0.0)
        exposures[symbol] = (positions * prices).abs()

    gross_exposure = exposures.sum(axis=1)
    equity = equity_curve["equity"].astype(float)
    if equity.empty or not equity.map(lambda value: pd.notna(value) and value > 0).all():
        raise ValueError("Equity values must be positive and non-empty")

    weights = exposures.div(gross_exposure.where(gross_exposure > 0), axis=0).fillna(0.0)
    report = pd.DataFrame(index=equity_curve.index)
    report["gross_exposure"] = gross_exposure
    report["gross_exposure_fraction"] = gross_exposure / equity
    report["concentration_hhi"] = weights.pow(2).sum(axis=1)
    for symbol in symbols:
        report[f"exposure_{symbol}"] = exposures[symbol]
        report[f"weight_{symbol}"] = weights[symbol]
    return report