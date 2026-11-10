import pandas as pd
import pytest

from quant_backtester.backtest import run_backtest
from quant_backtester.strategy import MeanReversionStrategy


def test_backtest_executes_signal_on_next_bar_open() -> None:
    prices = pd.DataFrame(
        {
            "Open": [100.0, 100.0, 100.0, 90.0],
            "High": [101.0, 101.0, 101.0, 121.0],
            "Low": [99.0, 99.0, 79.0, 89.0],
            "Close": [100.0, 100.0, 80.0, 120.0],
            "Volume": [1000.0] * 4,
        },
        index=pd.date_range("2024-01-01", periods=4),
    )

    result = run_backtest(
        prices,
        symbol="TEST",
        strategy=MeanReversionStrategy(lookback=3, z_threshold=1.0),
        initial_cash=1000.0,
    )

    assert list(result.trades["side"]) == ["BUY"]
    assert result.trades.iloc[0]["timestamp"] == pd.Timestamp("2024-01-04")
    assert result.trades.iloc[0]["price"] == pytest.approx(90.0)
    assert result.equity_curve.index.is_monotonic_increasing
    assert result.final_equity == pytest.approx(1330.0)


def test_backtest_applies_commission_and_slippage_to_trade_costs() -> None:
    prices = pd.DataFrame(
        {
            "Open": [100.0, 100.0, 80.0, 100.0, 100.0, 100.0],
            "High": [101.0, 101.0, 100.0, 101.0, 101.0, 101.0],
            "Low": [99.0, 99.0, 79.0, 99.0, 99.0, 99.0],
            "Close": [100.0, 100.0, 80.0, 100.0, 100.0, 100.0],
            "Volume": [1000.0] * 6,
        },
        index=pd.date_range("2024-01-01", periods=6),
    )

    strategy = MeanReversionStrategy(lookback=3, z_threshold=1.0)
    baseline = run_backtest(
        prices,
        symbol="TEST",
        strategy=strategy,
        initial_cash=1000.0,
    )
    result = run_backtest(
        prices,
        symbol="TEST",
        strategy=strategy,
        initial_cash=1000.0,
        commission=2.5,
        slippage_bps=50.0,
    )

    assert list(result.trades["side"]) == ["BUY", "SELL"]
    assert result.trades.iloc[0]["execution_price"] == pytest.approx(100.5)
    assert result.trades.iloc[1]["execution_price"] == pytest.approx(99.5)
    assert result.trades["commission"].tolist() == [2.5, 2.5]
    assert result.final_equity < baseline.final_equity


def test_backtest_records_order_rejected_after_next_open_price_gap() -> None:
    prices = pd.DataFrame(
        {
            "Open": [100.0, 100.0, 80.0, 1000.0],
            "High": [101.0, 101.0, 100.0, 1001.0],
            "Low": [99.0, 99.0, 79.0, 999.0],
            "Close": [100.0, 100.0, 80.0, 1000.0],
            "Volume": [1000.0] * 4,
        },
        index=pd.date_range("2024-01-01", periods=4),
    )

    result = run_backtest(
        prices,
        symbol="TEST",
        strategy=MeanReversionStrategy(lookback=3, z_threshold=1.0),
        initial_cash=1000.0,
    )

    assert result.trades.iloc[0]["quantity"] == 1
    assert result.rejections.iloc[0]["reason"] == "insufficient_cash_after_execution_costs"
    assert result.rejections.iloc[0]["rejected_quantity"] == 11


def test_backtest_caps_buy_at_maximum_portfolio_exposure() -> None:
    prices = pd.DataFrame(
        {
            "Open": [100.0, 100.0, 80.0, 100.0],
            "High": [101.0, 101.0, 100.0, 101.0],
            "Low": [99.0, 99.0, 79.0, 99.0],
            "Close": [100.0, 100.0, 80.0, 100.0],
            "Volume": [1000.0] * 4,
        },
        index=pd.date_range("2024-01-01", periods=4),
    )

    result = run_backtest(
        prices,
        symbol="TEST",
        strategy=MeanReversionStrategy(lookback=3, z_threshold=1.0),
        initial_cash=1000.0,
        max_portfolio_exposure=0.5,
    )

    assert result.trades.iloc[0]["quantity"] == 5
    assert result.rejections.iloc[0]["reason"] == "max_portfolio_exposure"
    assert result.final_equity == pytest.approx(1000.0)


def test_backtest_volume_impact_increases_fill_cost_at_lower_volume() -> None:
    prices = pd.DataFrame(
        {
            "Open": [100.0, 100.0, 80.0, 100.0],
            "High": [101.0, 101.0, 100.0, 101.0],
            "Low": [99.0, 99.0, 79.0, 99.0],
            "Close": [100.0, 100.0, 80.0, 100.0],
            "Volume": [1000.0, 1000.0, 1000.0, 10.0],
        },
        index=pd.date_range("2024-01-01", periods=4),
    )

    result = run_backtest(
        prices,
        symbol="TEST",
        strategy=MeanReversionStrategy(lookback=3, z_threshold=1.0),
        initial_cash=1000.0,
        market_impact_bps=100.0,
    )

    assert result.trades.iloc[0]["impact_bps"] > 0.0
    assert result.trades.iloc[0]["execution_price"] > 100.0


def test_backtest_primes_strategy_with_warmup_or_empty_history() -> None:
    class RecordingStrategy:
        def __init__(self) -> None:
            self.primed: list[list[pd.Timestamp]] = []

        def prime(self, history: pd.DataFrame) -> None:
            self.primed.append(list(history.index))

        def generate_signal(self, market_event, close_prices, position=0):
            return None

    prices = pd.DataFrame(
        {
            "Open": [100.0] * 4,
            "High": [101.0] * 4,
            "Low": [99.0] * 4,
            "Close": [100.0] * 4,
            "Volume": [1000.0] * 4,
        },
        index=pd.date_range("2024-01-01", periods=4),
    )
    strategy = RecordingStrategy()

    run_backtest(prices.iloc[2:], symbol="TEST", strategy=strategy, warmup_prices=prices.iloc[:2])
    run_backtest(prices, symbol="TEST", strategy=strategy)

    assert strategy.primed == [list(prices.index[:2]), []]