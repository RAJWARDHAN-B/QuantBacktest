"""Event-driven backtest orchestration for daily OHLCV data."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor

import pandas as pd

from quant_backtester.data import validate_ohlcv
from quant_backtester.events import FillEvent, OrderEvent, SignalEvent, stream_market_events
from quant_backtester.portfolio import Portfolio
from quant_backtester.strategy import Strategy


@dataclass(frozen=True)
class ExecutionConfig:
    """Execution assumptions applied while converting signals into fills."""

    commission: float = 0.0
    slippage_bps: float = 0.0
    max_position_size: int | None = None
    market_impact_bps: float = 0.0
    max_portfolio_exposure: float | None = None

    def __post_init__(self) -> None:
        if self.commission < 0:
            raise ValueError("Commission cannot be negative")
        if self.slippage_bps < 0:
            raise ValueError("Slippage basis points cannot be negative")
        if self.max_position_size is not None and self.max_position_size <= 0:
            raise ValueError("Max position size must be positive when provided")
        if self.market_impact_bps < 0:
            raise ValueError("Market impact basis points cannot be negative")
        if self.max_portfolio_exposure is not None and not 0.0 <= self.max_portfolio_exposure <= 1.0:
            raise ValueError("Max portfolio exposure must be between 0 and 1")


@dataclass
class BacktestResult:
    """Outputs from one reproducible backtest run."""

    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    portfolio: Portfolio
    rejections: pd.DataFrame

    @property
    def final_equity(self) -> float:
        """Return the final marked-to-market portfolio value."""
        return float(self.equity_curve["equity"].iloc[-1])


def _order_from_signal(signal: SignalEvent, portfolio: Portfolio, price: float) -> OrderEvent | None:
    if signal.direction == "LONG" and portfolio.positions.get(signal.symbol, 0) == 0:
        quantity = floor(portfolio.cash / price)
        if quantity > 0:
            return OrderEvent(signal.timestamp, signal.symbol, "BUY", quantity)
    if signal.direction == "EXIT":
        quantity = portfolio.positions.get(signal.symbol, 0)
        if quantity > 0:
            return OrderEvent(signal.timestamp, signal.symbol, "SELL", quantity)
    return None


def _size_buy_order(
    requested_quantity: int,
    market_price: float,
    volume: float,
    cash: float,
    commission: float,
    slippage_bps: float,
    market_impact_bps: float,
    current_position: int,
    max_position_size: int | None,
    max_portfolio_exposure: float | None,
    current_equity: float,
    current_exposure: float,
) -> tuple[int, str]:
    """Apply cash, share, and projected gross-exposure limits to a buy."""
    def affordable(quantity: int) -> bool:
        impact = market_impact_bps * quantity / volume if volume > 0 else 0.0
        execution_price = market_price * (1.0 + (slippage_bps + impact) / 10000.0)
        return quantity * execution_price + commission <= cash

    low, high = 0, requested_quantity
    while low < high:
        midpoint = (low + high + 1) // 2
        if affordable(midpoint):
            low = midpoint
        else:
            high = midpoint - 1
    quantity = low
    reason = "insufficient_cash_after_execution_costs" if quantity < requested_quantity else ""

    if max_position_size is not None:
        capped_quantity = min(quantity, max(0, max_position_size - current_position))
        if capped_quantity < quantity:
            quantity = capped_quantity
            reason = "max_position_size"

    if max_portfolio_exposure is not None:
        exposure_budget = max(0.0, current_equity * max_portfolio_exposure - current_exposure)
        exposure_quantity = floor(exposure_budget / market_price)
        if exposure_quantity < quantity:
            quantity = exposure_quantity
            reason = "max_portfolio_exposure"

    return quantity, reason


def prime_strategy(strategy: Strategy, history: pd.DataFrame) -> None:
    """Call a strategy's optional ``prime`` hook with validated history."""
    prime = getattr(strategy, "prime", None)
    if callable(prime):
        prime(history)


def run_backtest(
    prices: pd.DataFrame,
    symbol: str,
    strategy: Strategy,
    initial_cash: float = 10000.0,
    commission: float = 0.0,
    slippage_bps: float = 0.0,
    max_position_size: int | None = None,
    max_portfolio_exposure: float | None = None,
    market_impact_bps: float = 0.0,
    warmup_prices: pd.DataFrame | None = None,
) -> BacktestResult:
    """Replay bars, execute signals on the following bar's open, and track equity."""
    execution = ExecutionConfig(
        commission=commission,
        slippage_bps=slippage_bps,
        max_position_size=max_position_size,
        market_impact_bps=market_impact_bps,
        max_portfolio_exposure=max_portfolio_exposure,
    )
    cleaned = validate_ohlcv(prices)
    warmup = validate_ohlcv(warmup_prices) if warmup_prices is not None else None
    if warmup is not None and len(warmup) and warmup.index[-1] >= cleaned.index[0]:
        raise ValueError("Warm-up data must end before test data begins")
    prime_strategy(strategy, warmup if warmup is not None else cleaned.iloc[:0])
    portfolio = Portfolio(initial_cash=initial_cash)
    close_history = warmup["Close"].copy() if warmup is not None else pd.Series(dtype=float)
    pending_order: OrderEvent | None = None
    equity_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
    rejection_rows: list[dict[str, object]] = []

    for market_event in stream_market_events(cleaned, symbol):
        if pending_order is not None:
            rejection_reason = ""
            if pending_order.side == "BUY":
                marked_equity = portfolio.total_equity({symbol: market_event.open})
                current_exposure = portfolio.position_value(symbol, market_event.open)
                quantity, rejection_reason = _size_buy_order(
                    pending_order.quantity,
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
                quantity = min(
                    pending_order.quantity,
                    portfolio.positions.get(pending_order.symbol, 0),
                )
                if quantity < pending_order.quantity:
                    rejection_reason = "insufficient_position"

            if quantity < pending_order.quantity:
                rejection_rows.append(
                    {
                        "timestamp": market_event.timestamp,
                        "symbol": pending_order.symbol,
                        "side": pending_order.side,
                        "requested_quantity": pending_order.quantity,
                        "filled_quantity": quantity,
                        "rejected_quantity": pending_order.quantity - quantity,
                        "reason": rejection_reason,
                    }
                )

            if quantity > 0:
                fill = FillEvent(
                    timestamp=market_event.timestamp,
                    symbol=pending_order.symbol,
                    side=pending_order.side,
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
            pending_order = None

        close_history.loc[market_event.timestamp] = market_event.close
        signal = strategy.generate_signal(
            market_event,
            close_history,
            position=portfolio.positions.get(symbol, 0),
        )
        if signal is not None:
            pending_order = _order_from_signal(signal, portfolio, market_event.close)

        equity_rows.append(
            {
                "timestamp": market_event.timestamp,
                "cash": portfolio.cash,
                "position": portfolio.positions.get(symbol, 0),
                "close": market_event.close,
                "equity": portfolio.total_equity({symbol: market_event.close}),
            }
        )

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