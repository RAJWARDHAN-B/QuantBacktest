import pandas as pd
import pytest

from quant_backtester.multi_asset import (
    asset_return_correlation,
    exposure_report,
    run_multi_asset_backtest,
)
from quant_backtester.strategy import MeanReversionStrategy


def build_prices(scale: float) -> pd.DataFrame:
    close = [100.0, 100.0, 80.0, 100.0, 100.0, 100.0]
    open_prices = [100.0, 100.0, 80.0, 100.0, 100.0, 100.0]
    frame = pd.DataFrame(
        {
            "Open": [value * scale for value in open_prices],
            "High": [value * scale + scale for value in [100.0, 100.0, 100.0, 100.0, 100.0, 100.0]],
            "Low": [value * scale - scale for value in [100.0, 100.0, 80.0, 100.0, 100.0, 100.0]],
            "Close": [value * scale for value in close],
            "Volume": [1000.0] * 6,
        },
        index=pd.date_range("2024-01-01", periods=6),
    )
    return frame


def test_multi_asset_backtest_shares_cash_and_marks_all_positions() -> None:
    result = run_multi_asset_backtest(
        {"AAA": build_prices(1.0), "BBB": build_prices(0.5)},
        strategies={
            "AAA": MeanReversionStrategy(lookback=3, z_threshold=1.0),
            "BBB": MeanReversionStrategy(lookback=3, z_threshold=1.0),
        },
        initial_cash=1000.0,
        max_position_size=5,
    )

    assert list(result.trades["symbol"].drop_duplicates()) == ["AAA", "BBB"]
    assert result.portfolio.cash == pytest.approx(1000.0)
    assert set(result.rejections["reason"]) == {"max_position_size"}
    assert result.equity_curve.index.is_monotonic_increasing
    assert result.final_equity == pytest.approx(1000.0)


def test_multi_asset_backtest_requires_matching_strategy_symbols() -> None:
    with pytest.raises(ValueError, match="symbol sets"):
        run_multi_asset_backtest(
            {"AAA": build_prices(1.0)},
            strategies={"BBB": MeanReversionStrategy()},
        )


def test_multi_asset_backtest_enforces_shared_portfolio_exposure_limit() -> None:
    result = run_multi_asset_backtest(
        {"AAA": build_prices(1.0), "BBB": build_prices(1.0)},
        strategies={
            "AAA": MeanReversionStrategy(lookback=3, z_threshold=1.0),
            "BBB": MeanReversionStrategy(lookback=3, z_threshold=1.0),
        },
        initial_cash=1000.0,
        max_portfolio_exposure=0.5,
    )

    buy_trades = result.trades.loc[result.trades["side"] == "BUY"]
    assert len(buy_trades) == 1
    assert buy_trades.iloc[0]["quantity"] == 5
    assert set(result.rejections["reason"]) == {"max_portfolio_exposure"}


def test_asset_return_correlation_uses_aligned_close_returns() -> None:
    correlation = asset_return_correlation(
        {"AAA": build_prices(1.0), "BBB": build_prices(2.0)}
    )

    assert correlation.loc["AAA", "BBB"] == pytest.approx(1.0)


def test_exposure_report_calculates_gross_exposure_and_concentration() -> None:
    curve = pd.DataFrame(
        {
            "equity": [100.0, 100.0],
            "position_AAA": [0, 2],
            "close_AAA": [10.0, 10.0],
            "position_BBB": [0, 1],
            "close_BBB": [None, 20.0],
        },
        index=pd.date_range("2024-01-01", periods=2),
    )

    report = exposure_report(curve)

    assert report.iloc[1]["gross_exposure"] == pytest.approx(40.0)
    assert report.iloc[1]["gross_exposure_fraction"] == pytest.approx(0.4)
    assert report.iloc[1]["concentration_hhi"] == pytest.approx(0.5)
    assert report.iloc[0]["exposure_BBB"] == pytest.approx(0.0)