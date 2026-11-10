"""Strategy interface and a simple mean-reversion strategy built on close prices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from quant_backtester.events import MarketEvent, SignalEvent


class Strategy(Protocol):
    """Turns the latest bar into an optional signal without touching the portfolio.

    Engines also call an optional ``prime(history: pd.DataFrame) -> None`` method
    with validated warm-up bars (possibly empty) before replaying a run.
    """

    def generate_signal(
        self,
        market_event: MarketEvent,
        close_prices: pd.Series,
        position: int = 0,
    ) -> SignalEvent | None: ...


@dataclass
class MeanReversionStrategy:
    """Generate LONG or EXIT signals based on a rolling z-score."""

    lookback: int = 20
    z_threshold: float = 1.5
    exit_threshold: float = 0.25
    long_only: bool = True

    def __post_init__(self) -> None:
        if self.lookback <= 1:
            raise ValueError("lookback must be greater than 1")
        if self.z_threshold <= 0:
            raise ValueError("z_threshold must be positive")

    def _rolling_stats(self, close_prices: pd.Series) -> tuple[float, float]:
        window = close_prices.iloc[-self.lookback :]
        mean = float(window.mean())
        std = float(window.std(ddof=0))
        if std == 0:
            return mean, 0.0
        return mean, std

    def generate_signal(
        self,
        market_event: MarketEvent,
        close_prices: pd.Series,
        position: int = 0,
    ) -> SignalEvent | None:
        """Return a signal when the close departs sufficiently from its rolling mean."""
        price_history = close_prices.copy()
        current_close = float(market_event.close)

        if price_history.empty:
            price_history = pd.Series([current_close], index=[pd.Timestamp(market_event.timestamp)])
        elif price_history.index[-1] != pd.Timestamp(market_event.timestamp):
            price_history = pd.concat(
                [
                    price_history,
                    pd.Series([current_close], index=[pd.Timestamp(market_event.timestamp)]),
                ]
            )
        elif abs(float(price_history.iloc[-1]) - current_close) > 1e-12:
            price_history.iloc[-1] = current_close

        if len(price_history) < self.lookback:
            return None

        mean, std = self._rolling_stats(price_history)
        if std == 0:
            return None

        latest_close = float(price_history.iloc[-1])
        z_score = (latest_close - mean) / std

        if position == 0 and z_score <= -self.z_threshold:
            return SignalEvent(
                timestamp=market_event.timestamp,
                symbol=market_event.symbol,
                direction="LONG",
                strength=abs(z_score),
            )

        if self.long_only and position > 0 and z_score >= -self.exit_threshold:
            return SignalEvent(
                timestamp=market_event.timestamp,
                symbol=market_event.symbol,
                direction="EXIT",
                strength=abs(z_score),
            )

        return None
