"""Leakage-safe OHLCV features and forward-return labels for ML signals.

Every feature at bar t uses only bars at or before t. Labels look forward by
definition, so they are only used inside training windows, where the final
``horizon`` rows have no label and are therefore purged automatically.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd

from quant_backtester.data import validate_ohlcv


FEATURE_COLUMNS = (
    "ret_1",
    "ret_5",
    "ret_20",
    "zscore_20",
    "vol_20",
    "vol_ratio",
    "range_hl",
    "gap",
    "volume_z_20",
    "rsi_14",
    "dist_sma_50",
    "dist_sma_200",
)
# Bars needed before every feature is defined (the 200-bar moving average).
FEATURE_WARMUP = 200
LABEL_PRICES = ("close", "open")


def _lagged_difference(values: np.ndarray, lag: int) -> np.ndarray:
    result = np.full(len(values), np.nan)
    result[lag:] = values[lag:] - values[:-lag]
    return result


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    result = np.full(len(values), np.nan)
    if len(values) >= window:
        result[window - 1 :] = sliding_window_view(values, window).mean(axis=1)
    return result


def _rolling_std(values: np.ndarray, window: int, ddof: int) -> np.ndarray:
    result = np.full(len(values), np.nan)
    if len(values) >= window:
        windows = sliding_window_view(values, window)
        spread = windows.std(axis=1, ddof=ddof)
        # Flat windows are exactly zero rather than rounding noise.
        spread[np.ptp(windows, axis=1) == 0] = 0.0
        result[window - 1 :] = spread
    return result


def _rolling_zscore(values: np.ndarray, window: int, ddof: int) -> np.ndarray:
    spread = _rolling_std(values, window, ddof)
    deviation = values - _rolling_mean(values, window)
    # A flat window has no deviation from its mean; score it as zero rather than undefined.
    return np.where(spread == 0, 0.0, deviation / spread)


def feature_matrix(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
) -> np.ndarray:
    """Rows of ``FEATURE_COLUMNS`` for aligned OHLCV arrays; undefined values are NaN.

    Each value is computed from its own trailing window, so a buffer of the last
    ``FEATURE_WARMUP`` bars reproduces full-history features exactly.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        log_close = np.log(close)
        log_return = _lagged_difference(log_close, 1)
        change = _lagged_difference(close, 1)
        average_gain = _rolling_mean(np.clip(change, 0.0, None), 14)
        average_loss = _rolling_mean(np.clip(-change, 0.0, None), 14)
        movement = average_gain + average_loss
        previous_close = np.concatenate(([np.nan], close[:-1]))
        columns = (
            log_return,
            _lagged_difference(log_close, 5),
            _lagged_difference(log_close, 20),
            _rolling_zscore(close, 20, ddof=0),
            _rolling_std(log_return, 20, ddof=1),
            _rolling_std(log_return, 5, ddof=1) / _rolling_std(log_return, 60, ddof=1),
            (high - low) / close,
            open_ / previous_close - 1.0,
            _rolling_zscore(np.log1p(volume), 20, ddof=1),
            # Simple-average RSI: finite memory keeps buffered and full-history values equal.
            np.where(movement == 0, 50.0, 100.0 * average_gain / movement),
            close / _rolling_mean(close, 50) - 1.0,
            close / _rolling_mean(close, 200) - 1.0,
        )
    matrix = np.column_stack(columns)
    matrix[~np.isfinite(matrix)] = np.nan
    return matrix


def compute_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute ``FEATURE_COLUMNS`` for validated OHLCV data; warm-up rows are NaN."""
    values = prices.loc[:, ["Open", "High", "Low", "Close", "Volume"]].to_numpy(dtype=float)
    return pd.DataFrame(
        feature_matrix(*values.T), index=prices.index, columns=list(FEATURE_COLUMNS)
    )


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Validate OHLCV data and compute features aligned to its index."""
    return compute_features(validate_ohlcv(prices))


def forward_returns(
    prices: pd.DataFrame,
    horizon: int = 5,
    label_price: str = "close",
) -> pd.Series:
    """Return realized after bar t: ``C[t+h]/C[t] - 1`` or ``O[t+1+h]/O[t+1] - 1``.

    The open-to-open version matches a signal at the close of t that fills at the
    next open. Rows without a complete forward window are NaN.
    """
    if horizon < 1:
        raise ValueError("Label horizon must be at least one bar")
    if label_price not in LABEL_PRICES:
        raise ValueError(f"label_price must be one of {LABEL_PRICES}")
    cleaned = validate_ohlcv(prices)
    if label_price == "close":
        return (cleaned["Close"].shift(-horizon) / cleaned["Close"] - 1.0).rename("forward_return")
    entry = cleaned["Open"].shift(-1)
    return (cleaned["Open"].shift(-(horizon + 1)) / entry - 1.0).rename("forward_return")


def build_labels(
    prices: pd.DataFrame,
    horizon: int = 5,
    threshold: float = 0.0,
    label_price: str = "close",
) -> pd.Series:
    """1.0 when the forward return exceeds ``threshold``, 0.0 otherwise, NaN when unknown."""
    forward = forward_returns(prices, horizon, label_price)
    return (forward > threshold).astype(float).where(forward.notna()).rename("label")


def build_training_set(
    prices: pd.DataFrame,
    horizon: int = 5,
    threshold: float = 0.0,
    label_price: str = "close",
) -> tuple[pd.DataFrame, pd.Series]:
    """Feature rows paired with labels, without warm-up rows or the unlabeled tail.

    Labels come only from ``prices``, so the last ``horizon`` rows (``horizon + 1``
    for open labels) are purged and no training row depends on later prices.
    """
    cleaned = validate_ohlcv(prices)
    features = compute_features(cleaned)
    labels = build_labels(cleaned, horizon, threshold, label_price)
    valid = features.notna().all(axis=1) & labels.notna()
    return features.loc[valid], labels.loc[valid]
