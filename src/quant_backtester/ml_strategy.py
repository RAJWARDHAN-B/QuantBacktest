"""Event-driven strategy that trades on a fitted classifier's probabilities."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quant_backtester.data import validate_ohlcv
from quant_backtester.events import MarketEvent, SignalEvent
from quant_backtester.features import FEATURE_COLUMNS, FEATURE_WARMUP, feature_matrix
from quant_backtester.models import FittedModel


@dataclass
class MLSignalStrategy:
    """Go long when P(up) reaches ``entry_probability``; exit when it drops below exit.

    Features need full OHLCV bars, so the strategy keeps its own bounded buffer of
    market events and records every probability it computes in ``predictions``.
    """

    model: FittedModel
    entry_probability: float = 0.55
    exit_probability: float = 0.50
    min_history: int = FEATURE_WARMUP
    predictions: list[tuple[pd.Timestamp, float]] = field(
        default_factory=list, init=False, repr=False
    )
    _bars: deque = field(init=False, repr=False)
    _symbol: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.exit_probability <= self.entry_probability < 1.0:
            raise ValueError(
                "Probabilities must satisfy 0 < exit_probability <= entry_probability < 1"
            )
        if self.min_history < FEATURE_WARMUP:
            raise ValueError(f"min_history must be at least {FEATURE_WARMUP} bars")
        self._bars = deque(maxlen=self.min_history)

    def prime(self, history: pd.DataFrame) -> None:
        """Reset state and seed the bar buffer with warm-up bars."""
        self._bars.clear()
        self.predictions.clear()
        self._symbol = None
        for row in validate_ohlcv(history).tail(self.min_history).itertuples(name=None):
            self._bars.append((pd.Timestamp(row[0]), *map(float, row[1:])))

    def generate_signal(
        self,
        market_event: MarketEvent,
        close_prices: pd.Series,
        position: int = 0,
    ) -> SignalEvent | None:
        """Score the latest bar and emit LONG/EXIT signals around the thresholds."""
        if self._symbol is None:
            self._symbol = market_event.symbol
        elif market_event.symbol != self._symbol:
            raise ValueError("MLSignalStrategy tracks one symbol; create one instance per symbol")

        timestamp = pd.Timestamp(market_event.timestamp)
        bar = (
            timestamp,
            market_event.open,
            market_event.high,
            market_event.low,
            market_event.close,
            market_event.volume,
        )
        if self._bars and self._bars[-1][0] == timestamp:
            self._bars[-1] = bar
        else:
            self._bars.append(bar)
        if len(self._bars) < self.min_history:
            return None

        history = np.array([bar[1:] for bar in self._bars])
        latest = feature_matrix(*history.T)[-1]
        if np.isnan(latest).any():
            return None
        row = pd.DataFrame([latest], index=[timestamp], columns=list(FEATURE_COLUMNS))
        probability = float(self.model.predict_proba(row)[0])
        self.predictions.append((timestamp, probability))

        if position == 0 and probability >= self.entry_probability:
            return SignalEvent(timestamp, market_event.symbol, "LONG", strength=probability)
        if position > 0 and probability < self.exit_probability:
            return SignalEvent(timestamp, market_event.symbol, "EXIT", strength=probability)
        return None
