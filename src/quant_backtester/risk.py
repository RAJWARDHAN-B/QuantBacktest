"""Value-at-Risk, Expected Shortfall, VaR backtests, and stress tests.

Losses are negated simple returns, so VaR and ES are reported as fractions of
equity where positive values are losses.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from math import ceil, erfc, isfinite, log, nan, sqrt
from statistics import NormalDist

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd

from quant_backtester.data import validate_ohlcv
from quant_backtester.multi_asset import exposure_report


VAR_METHODS = ("historical", "parametric", "cornish_fisher")
SIMULATION_METHODS = ("monte_carlo", "bootstrap")
MIN_PARAMETRIC_OBSERVATIONS = 30
HORIZON_SCALING = {
    "historical": "overlapping compounded returns",
    "parametric": "square-root-of-time",
    "cornish_fisher": "square-root-of-time with i.i.d. skewness and kurtosis aggregation",
    "monte_carlo": "simulated compounded multi-bar paths",
    "bootstrap": "resampled blocks of consecutive bars",
}
DEFAULT_SCENARIOS: dict[str, tuple[str, str]] = {
    "Global Financial Crisis": ("2008-09-15", "2009-03-09"),
    "COVID crash": ("2020-02-19", "2020-03-23"),
    "2022 rate shock": ("2022-01-03", "2022-10-12"),
}
STRESS_COLUMNS = (
    "scenario",
    "kind",
    "start",
    "end",
    "symbol",
    "exposure",
    "scenario_return",
    "pnl",
    "worst_drawdown",
    "status",
)

_STANDARD_NORMAL = NormalDist()


@dataclass(frozen=True)
class RiskEstimate:
    """One VaR/ES estimate as fractions of equity; positive values are losses."""

    method: str
    confidence: float
    horizon: int
    var: float
    expected_shortfall: float
    observations: int


def _check_confidence(confidence: float) -> float:
    if not 0.0 < confidence < 1.0:
        raise ValueError("Confidence must be strictly between 0 and 1")
    return float(confidence)


def _check_levels(confidence_levels: Sequence[float]) -> list[float]:
    levels = [_check_confidence(confidence) for confidence in confidence_levels]
    if not levels:
        raise ValueError("At least one confidence level is required")
    return levels


def _check_horizon(horizon: int) -> int:
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError("Horizon must be a positive integer number of bars")
    return int(horizon)


def _minimum_observations(method: str, confidence: float) -> int:
    if method == "historical":
        return ceil(1.0 / (1.0 - confidence) - 1e-9)
    return MIN_PARAMETRIC_OBSERVATIONS


def _require_observations(observations: int, required: int, method: str) -> None:
    if observations < required:
        raise ValueError(
            f"{method} VaR requires at least {required} observations; found {observations}"
        )


def _clean_returns(returns: pd.Series) -> np.ndarray:
    values = pd.Series(returns, dtype=float).dropna().to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("Returns must be finite")
    return values


def _horizon_returns(values: np.ndarray, horizon: int) -> np.ndarray:
    if horizon == 1:
        return values
    if len(values) < horizon:
        return np.empty(0)
    return sliding_window_view(1.0 + values, horizon).prod(axis=1) - 1.0


def _tail(losses: np.ndarray, confidence: float) -> tuple[float, float]:
    # "higher" always returns an observed loss, so the ES tail is never empty.
    var = float(np.quantile(losses, confidence, method="higher"))
    return var, float(losses[losses >= var].mean())


def _moments(windows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Row-wise mean, sample std, and bias-corrected skewness and excess kurtosis."""
    n = windows.shape[1]
    mean = windows.mean(axis=1)
    deviations = windows - mean[:, np.newaxis]
    m2 = (deviations**2).mean(axis=1)
    m3 = (deviations**3).mean(axis=1)
    m4 = (deviations**4).mean(axis=1)
    std = np.sqrt(m2 * n / (n - 1))
    varying = m2 > 0
    safe_m2 = np.where(varying, m2, 1.0)
    skew = np.where(varying, m3 / safe_m2**1.5 * sqrt(n * (n - 1)) / (n - 2), 0.0)
    kurtosis = np.where(
        varying,
        (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * (m4 / safe_m2**2 - 3.0) + 6.0),
        0.0,
    )
    return mean, std, skew, kurtosis


def _parametric(
    mean: np.ndarray,
    std: np.ndarray,
    skew: np.ndarray,
    kurtosis: np.ndarray,
    confidence: float,
    horizon: int,
    cornish_fisher: bool,
) -> tuple[np.ndarray, np.ndarray]:
    z = _STANDARD_NORMAL.inv_cdf(confidence)
    tail_density = _STANDARD_NORMAL.pdf(z) / (1.0 - confidence)
    loss_mean = -mean * horizon
    scale = std * sqrt(horizon)
    if not cornish_fisher:
        return loss_mean + scale * z, loss_mean + scale * tail_density

    # Losses have the opposite skew of returns; i.i.d. cumulants aggregate with the horizon.
    s = -skew / sqrt(horizon)
    k = kurtosis / horizon
    z_cf = z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36
    # Closed-form tail mean of the Cornish-Fisher transform of a standard normal.
    es_factor = tail_density * (
        1 + s * z / 6 + k * (z**2 - 1) / 24 - s**2 * (2 * z**2 - 1) / 36
    )
    return loss_mean + scale * z_cf, loss_mean + scale * es_factor


def historical_var(
    returns: pd.Series,
    confidence: float = 0.95,
    horizon: int = 1,
) -> RiskEstimate:
    """Empirical VaR/ES using the "higher" quantile rule on observed losses.

    Multi-bar horizons use overlapping compounded returns rather than scaling.
    """
    confidence = _check_confidence(confidence)
    horizon = _check_horizon(horizon)
    losses = -_horizon_returns(_clean_returns(returns), horizon)
    _require_observations(len(losses), _minimum_observations("historical", confidence), "historical")
    var, expected_shortfall = _tail(losses, confidence)
    return RiskEstimate("historical", confidence, horizon, var, expected_shortfall, len(losses))


def parametric_var(
    returns: pd.Series,
    confidence: float = 0.95,
    horizon: int = 1,
    cornish_fisher: bool = False,
) -> RiskEstimate:
    """Gaussian VaR/ES, optionally with a Cornish-Fisher skew/kurtosis adjustment."""
    confidence = _check_confidence(confidence)
    horizon = _check_horizon(horizon)
    values = _clean_returns(returns)
    method = "cornish_fisher" if cornish_fisher else "parametric"
    _require_observations(len(values), MIN_PARAMETRIC_OBSERVATIONS, method)
    var, expected_shortfall = _parametric(
        *_moments(values[np.newaxis, :]), confidence, horizon, cornish_fisher
    )
    return RiskEstimate(
        method, confidence, horizon, float(var[0]), float(expected_shortfall[0]), len(values)
    )


def _check_positive_semidefinite(covariance: np.ndarray) -> None:
    if not np.allclose(covariance, covariance.T):
        raise ValueError("Covariance matrix must be symmetric")
    eigenvalues = np.linalg.eigvalsh(covariance)
    tolerance = 1e-10 * max(float(np.abs(eigenvalues).max()), np.finfo(float).tiny)
    if eigenvalues.min() < -tolerance:
        raise ValueError("Covariance matrix is not positive semidefinite")


def monte_carlo_var(
    asset_returns: pd.DataFrame,
    weights: pd.Series,
    confidence: float = 0.95,
    horizon: int = 1,
    n_simulations: int = 10_000,
    seed: int | None = None,
    bootstrap: bool = False,
) -> RiskEstimate:
    """Simulate holdings-level VaR/ES from aligned asset returns and equity weights.

    Weights are exposures as fractions of equity; uninvested cash earns zero. The
    default draws from a multivariate normal with the sample mean and covariance;
    ``bootstrap=True`` resamples blocks of ``horizon`` consecutive historical rows.
    """
    confidence = _check_confidence(confidence)
    horizon = _check_horizon(horizon)
    method = "bootstrap" if bootstrap else "monte_carlo"
    if not isinstance(asset_returns, pd.DataFrame) or asset_returns.shape[1] == 0:
        raise ValueError("Asset returns must be a DataFrame with at least one column")
    if set(weights.index) != set(asset_returns.columns):
        raise ValueError("Weights must be indexed by exactly the asset return columns")
    if n_simulations < _minimum_observations("historical", confidence):
        raise ValueError("n_simulations is too small for the requested confidence level")

    aligned = asset_returns.astype(float).dropna()
    values = aligned.to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("Asset returns must be finite")
    _require_observations(len(values), max(MIN_PARAMETRIC_OBSERVATIONS, horizon), method)
    weight_values = weights.reindex(aligned.columns).astype(float).to_numpy()
    if not np.isfinite(weight_values).all():
        raise ValueError("Weights must be finite")

    generator = np.random.default_rng(seed)
    if bootstrap:
        starts = generator.integers(0, len(values) - horizon + 1, size=n_simulations)
        paths = values[starts[:, np.newaxis] + np.arange(horizon)]
    else:
        covariance = np.atleast_2d(np.cov(values, rowvar=False, ddof=1))
        _check_positive_semidefinite(covariance)
        paths = generator.multivariate_normal(
            values.mean(axis=0), covariance, size=(n_simulations, horizon), method="eigh"
        )
    asset_growth = np.prod(1.0 + paths, axis=1) - 1.0
    var, expected_shortfall = _tail(-(asset_growth @ weight_values), confidence)
    return RiskEstimate(method, confidence, horizon, var, expected_shortfall, len(values))


def rolling_var(
    returns: pd.Series,
    window: int,
    confidence: float = 0.95,
    method: str = "historical",
) -> pd.DataFrame:
    """One-bar-ahead VaR forecasts compared with the loss realized on each bar.

    Row ``t`` holds VaR estimated from the ``window`` returns before ``t`` and the
    loss realized at ``t``; no return at or after ``t`` enters its forecast.
    """
    confidence = _check_confidence(confidence)
    if method not in VAR_METHODS:
        raise ValueError(f"Rolling VaR method must be one of {VAR_METHODS}")
    series = pd.Series(returns, dtype=float).dropna()
    values = _clean_returns(series)
    required = _minimum_observations(method, confidence)
    if window < required:
        raise ValueError(f"{method} rolling VaR requires a window of at least {required} bars")
    if len(values) <= window:
        raise ValueError("Rolling VaR requires more returns than the window length")

    windows = sliding_window_view(values[:-1], window)
    if method == "historical":
        var = np.quantile(-windows, confidence, axis=1, method="higher")
    else:
        var, _ = _parametric(
            *_moments(windows), confidence, 1, cornish_fisher=method == "cornish_fisher"
        )
    realized_loss = -values[window:]
    return pd.DataFrame(
        {"var": var, "realized_loss": realized_loss, "exceedance": realized_loss > var},
        index=series.index[window:],
    )


def _level_label(confidence: float) -> str:
    return f"{confidence * 100:g}"


def rolling_var_table(
    returns: pd.Series,
    window: int,
    confidence_levels: Sequence[float] = (0.95, 0.99),
    method: str = "historical",
) -> pd.DataFrame:
    """Combine rolling VaR forecasts for several confidence levels into one table."""
    columns: dict[str, pd.Series] = {}
    for confidence in _check_levels(confidence_levels):
        frame = rolling_var(returns, window, confidence, method)
        columns["realized_loss"] = frame["realized_loss"]
        columns[f"var_{_level_label(confidence)}"] = frame["var"]
        columns[f"exceedance_{_level_label(confidence)}"] = frame["exceedance"]
    return pd.DataFrame(columns)


def _exceedance_flags(exceedances: pd.Series) -> np.ndarray:
    flags = pd.Series(exceedances).dropna()
    if not flags.isin([0, 1]).all():
        raise ValueError("Exceedances must be boolean or 0/1 values")
    return flags.astype(int).to_numpy()


def _bernoulli_log_likelihood(successes: int, trials: int, probability: float) -> float:
    # Terms with zero counts are dropped so boundary estimates avoid log(0).
    failures = trials - successes
    total = 0.0
    if successes:
        total += successes * log(probability)
    if failures:
        total += failures * log(1.0 - probability)
    return total


def _chi_square_one_p_value(statistic: float) -> float:
    return erfc(sqrt(statistic / 2.0))


def kupiec_test(exceedances: pd.Series, confidence: float) -> dict[str, float]:
    """Kupiec proportion-of-failures likelihood-ratio test of VaR coverage."""
    confidence = _check_confidence(confidence)
    flags = _exceedance_flags(exceedances)
    observations = len(flags)
    if observations == 0:
        raise ValueError("At least one VaR forecast is required")
    count = int(flags.sum())
    expected_rate = 1.0 - confidence
    observed_rate = count / observations
    statistic = max(
        0.0,
        -2.0
        * (
            _bernoulli_log_likelihood(count, observations, expected_rate)
            - _bernoulli_log_likelihood(count, observations, observed_rate)
        ),
    )
    return {
        "observations": observations,
        "exceedances": count,
        "expected_exceedances": observations * expected_rate,
        "exceedance_rate": observed_rate,
        "expected_rate": expected_rate,
        "lr_statistic": statistic,
        "p_value": _chi_square_one_p_value(statistic),
    }


def christoffersen_test(exceedances: pd.Series) -> dict[str, float]:
    """Christoffersen likelihood-ratio test that exceedances do not cluster."""
    flags = _exceedance_flags(exceedances)
    if len(flags) < 2:
        raise ValueError("At least two VaR forecasts are required")
    previous, current = flags[:-1], flags[1:]
    n00 = int(((previous == 0) & (current == 0)).sum())
    n01 = int(((previous == 0) & (current == 1)).sum())
    n10 = int(((previous == 1) & (current == 0)).sum())
    n11 = int(((previous == 1) & (current == 1)).sum())
    after_calm = n00 + n01
    after_exceedance = n10 + n11
    pooled_rate = (n01 + n11) / (after_calm + after_exceedance)
    calm_rate = n01 / after_calm if after_calm else 0.0
    clustered_rate = n11 / after_exceedance if after_exceedance else 0.0
    statistic = max(
        0.0,
        -2.0
        * (
            _bernoulli_log_likelihood(n01 + n11, after_calm + after_exceedance, pooled_rate)
            - _bernoulli_log_likelihood(n01, after_calm, calm_rate)
            - _bernoulli_log_likelihood(n11, after_exceedance, clustered_rate)
        ),
    )
    return {
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
        "exceedance_after_calm_rate": calm_rate,
        "exceedance_after_exceedance_rate": clustered_rate,
        "lr_statistic": statistic,
        "p_value": _chi_square_one_p_value(statistic),
    }


def _worst_drawdown(path: pd.Series) -> float:
    return float((path / path.cummax() - 1.0).min())


def _scenario_bounds(name: str, window: Sequence[str]) -> tuple[pd.Timestamp, pd.Timestamp]:
    if len(window) != 2:
        raise ValueError(f"Scenario {name!r} must be a (start, end) pair")
    start, end = pd.Timestamp(window[0]), pd.Timestamp(window[1])
    if start >= end:
        raise ValueError(f"Scenario {name!r} must end after it starts")
    return start, end


def _scenario_growth(
    close: pd.Series | None,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series | None:
    """Growth of one unit held from the last close before ``start`` through ``end``."""
    if close is None or close.empty or close.index[0] >= start or close.index[-1] < end:
        return None
    base_date = close.index[close.index < start][-1]
    window = close.loc[base_date:end]
    return window / window.iloc[0]


def stress_test(
    exposures: pd.Series,
    prices: Mapping[str, pd.DataFrame],
    scenarios: Mapping[str, Sequence[str]] | None = None,
    proxies: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Replay historical windows against current currency exposures by symbol.

    A symbol needs data before the window starts and through its end; otherwise it
    is reported as missing (never assumed flat) unless an explicit proxy symbol is
    mapped to it. ``TOTAL`` rows exclude missing symbols and are marked ``partial``.
    """
    closes = {
        str(symbol): validate_ohlcv(frame)["Close"].astype(float)
        for symbol, frame in prices.items()
    }
    proxy_map = dict(proxies or {})
    holdings = exposures.astype(float)
    rows: list[dict[str, object]] = []
    for name, window in (DEFAULT_SCENARIOS if scenarios is None else scenarios).items():
        start, end = _scenario_bounds(name, window)
        scenario = {
            "scenario": name,
            "kind": "historical",
            "start": start.date().isoformat(),
            "end": end.date().isoformat(),
        }
        growth_paths: dict[str, pd.Series] = {}
        missing_exposure = False
        for symbol, exposure in holdings.items():
            growth, status = _scenario_growth(closes.get(symbol), start, end), "ok"
            if growth is None and symbol in proxy_map:
                proxy = proxy_map[symbol]
                growth, status = _scenario_growth(closes.get(proxy), start, end), f"proxy:{proxy}"
            if growth is None:
                missing_exposure = missing_exposure or exposure != 0
                rows.append(
                    {
                        **scenario,
                        "symbol": symbol,
                        "exposure": exposure,
                        "scenario_return": nan,
                        "pnl": nan,
                        "worst_drawdown": nan,
                        "status": "missing",
                    }
                )
                continue
            growth_paths[symbol] = growth
            scenario_return = float(growth.iloc[-1] - 1.0)
            rows.append(
                {
                    **scenario,
                    "symbol": symbol,
                    "exposure": exposure,
                    "scenario_return": scenario_return,
                    "pnl": exposure * scenario_return,
                    "worst_drawdown": _worst_drawdown(growth),
                    "status": status,
                }
            )

        covered_exposure = float(holdings.reindex(list(growth_paths)).sum())
        total = {
            **scenario,
            "symbol": "TOTAL",
            "exposure": float(holdings.sum()),
            "scenario_return": nan,
            "pnl": nan,
            "worst_drawdown": nan,
            "status": "ok",
        }
        if missing_exposure:
            total["status"] = "partial" if covered_exposure else "missing"
        if total["status"] != "missing":
            total["pnl"] = 0.0
            if growth_paths:
                growth = pd.concat(growth_paths, axis=1).sort_index().ffill().fillna(1.0)
                value = growth.mul(holdings.reindex(growth.columns), axis=1).sum(axis=1)
                total["pnl"] = float(value.iloc[-1] - value.iloc[0])
                if covered_exposure > 0 and (value > 0).all():
                    total["scenario_return"] = total["pnl"] / covered_exposure
                    total["worst_drawdown"] = _worst_drawdown(value)
        rows.append(total)
    return pd.DataFrame(rows, columns=list(STRESS_COLUMNS))


def shock_test(
    exposures: pd.Series,
    shocks: Mapping[str, float] | float,
    name: str = "Hypothetical shock",
) -> pd.DataFrame:
    """Apply instantaneous per-symbol shocks, or one uniform market shock, to exposures."""
    holdings = exposures.astype(float)
    if isinstance(shocks, Mapping):
        shock_by_symbol = {str(symbol): float(value) for symbol, value in shocks.items()}
        symbols = list(dict.fromkeys([*holdings.index, *shock_by_symbol]))
    else:
        shock_by_symbol = {str(symbol): float(shocks) for symbol in holdings.index}
        symbols = list(holdings.index)
    if any(not isfinite(value) or value < -1.0 for value in shock_by_symbol.values()):
        raise ValueError("Shocks must be finite returns no lower than -100%")

    rows: list[dict[str, object]] = []
    for symbol in symbols:
        exposure = float(holdings.get(symbol, 0.0))
        shock = shock_by_symbol.get(symbol, 0.0)
        rows.append(
            {
                "scenario": name,
                "kind": "hypothetical",
                "start": None,
                "end": None,
                "symbol": symbol,
                "exposure": exposure,
                "scenario_return": shock,
                "pnl": exposure * shock,
                "worst_drawdown": min(shock, 0.0),
                "status": "ok",
            }
        )
    total_exposure = sum(float(row["exposure"]) for row in rows)
    total_pnl = sum(float(row["pnl"]) for row in rows)
    total_return = total_pnl / total_exposure if total_exposure else nan
    rows.append(
        {
            "scenario": name,
            "kind": "hypothetical",
            "start": None,
            "end": None,
            "symbol": "TOTAL",
            "exposure": total_exposure,
            "scenario_return": total_return,
            "pnl": total_pnl,
            "worst_drawdown": min(total_return, 0.0) if total_exposure else nan,
            "status": "ok",
        }
    )
    return pd.DataFrame(rows, columns=list(STRESS_COLUMNS))


def equity_returns(equity_curve: pd.DataFrame) -> pd.Series:
    """Simple per-bar returns of a backtest equity curve."""
    if "equity" not in equity_curve:
        raise ValueError("Equity curve must contain an 'equity' column")
    equity = equity_curve["equity"].astype(float)
    if equity.empty or not np.isfinite(equity).all() or (equity <= 0).any():
        raise ValueError("Equity values must be finite, positive, and non-empty")
    return equity.pct_change().dropna().rename("return")


def current_exposures(equity_curve: pd.DataFrame) -> pd.Series:
    """Marked exposure by symbol on the final bar of a multi-asset equity curve."""
    last = exposure_report(equity_curve).iloc[-1]
    return pd.Series(
        {
            column.removeprefix("exposure_"): float(last[column])
            for column in last.index
            if column.startswith("exposure_")
        },
        name="exposure",
        dtype=float,
    )


def aligned_asset_returns(prices: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Close-to-close simple returns on the dates every symbol traded."""
    returns = {
        symbol: validate_ohlcv(frame)["Close"].pct_change(fill_method=None)
        for symbol, frame in prices.items()
    }
    return pd.concat(returns, axis=1).dropna()


def _estimate_record(estimate: RiskEstimate, equity: float) -> dict[str, object]:
    return {
        **asdict(estimate),
        "var_amount": estimate.var * equity,
        "expected_shortfall_amount": estimate.expected_shortfall * equity,
    }


def risk_report(
    equity_curve: pd.DataFrame,
    confidence_levels: Sequence[float] = (0.95, 0.99),
    horizon: int = 1,
    rolling_window: int | None = None,
) -> dict[str, object]:
    """Equity-curve VaR/ES by method, plus rolling VaR backtests when a window is given.

    Methods or backtests without enough observations are listed under ``skipped``.
    """
    levels = _check_levels(confidence_levels)
    horizon = _check_horizon(horizon)
    returns = equity_returns(equity_curve)
    latest_equity = float(equity_curve["equity"].iloc[-1])
    estimates: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    for confidence in levels:
        for method in VAR_METHODS:
            available = len(returns) - horizon + 1 if method == "historical" else len(returns)
            required = _minimum_observations(method, confidence)
            if available < required:
                skipped.append(
                    {
                        "method": method,
                        "confidence": confidence,
                        "reason": f"requires {required} observations; found {max(available, 0)}",
                    }
                )
                continue
            if method == "historical":
                estimate = historical_var(returns, confidence, horizon)
            else:
                estimate = parametric_var(
                    returns, confidence, horizon, cornish_fisher=method == "cornish_fisher"
                )
            estimates.append(_estimate_record(estimate, latest_equity))

    report: dict[str, object] = {
        "level": "equity_curve",
        "observations": int(len(returns)),
        "latest_equity": latest_equity,
        "horizon": horizon,
        "horizon_scaling": {method: HORIZON_SCALING[method] for method in VAR_METHODS},
        "estimates": estimates,
        "skipped": skipped,
    }
    if rolling_window is not None:
        backtests: list[dict[str, object]] = []
        for confidence in levels:
            required = _minimum_observations("historical", confidence)
            if rolling_window < required or len(returns) < rolling_window + 2:
                skipped.append(
                    {
                        "method": "rolling_historical",
                        "confidence": confidence,
                        "reason": (
                            f"requires a window of at least {required} bars and at least "
                            f"{rolling_window + 2} returns; found {len(returns)}"
                        ),
                    }
                )
                continue
            forecasts = rolling_var(returns, rolling_window, confidence)
            backtests.append(
                {
                    "confidence": confidence,
                    "window": rolling_window,
                    "method": "historical",
                    "kupiec": kupiec_test(forecasts["exceedance"], confidence),
                    "christoffersen": christoffersen_test(forecasts["exceedance"]),
                }
            )
        report["var_backtests"] = backtests
    return report


def holdings_risk_report(
    exposures: pd.Series,
    equity: float,
    asset_returns: pd.DataFrame,
    confidence_levels: Sequence[float] = (0.95, 0.99),
    horizon: int = 1,
    n_simulations: int = 10_000,
    seed: int | None = None,
) -> dict[str, object]:
    """Monte Carlo and bootstrap VaR/ES for the holdings held at the end of a run."""
    levels = _check_levels(confidence_levels)
    horizon = _check_horizon(horizon)
    if not isfinite(equity) or equity <= 0:
        raise ValueError("Equity must be finite and positive")
    weights = exposures.astype(float) / equity
    observations = len(asset_returns.dropna())
    report: dict[str, object] = {
        "level": "holdings",
        "equity": float(equity),
        "exposures": {str(symbol): float(value) for symbol, value in exposures.items()},
        "weights": {str(symbol): float(value) for symbol, value in weights.items()},
        "gross_exposure_fraction": float(weights.abs().sum()),
        "observations": observations,
        "horizon": horizon,
        "n_simulations": n_simulations,
        "seed": seed,
        "horizon_scaling": {method: HORIZON_SCALING[method] for method in SIMULATION_METHODS},
        "estimates": [],
        "skipped": [],
    }
    required = max(MIN_PARAMETRIC_OBSERVATIONS, horizon)
    if observations < required:
        report["skipped"] = [
            {
                "method": method,
                "reason": f"requires {required} aligned observations; found {observations}",
            }
            for method in SIMULATION_METHODS
        ]
        return report
    report["estimates"] = [
        _estimate_record(
            monte_carlo_var(
                asset_returns,
                weights,
                confidence,
                horizon,
                n_simulations=n_simulations,
                seed=seed,
                bootstrap=bootstrap,
            ),
            equity,
        )
        for confidence in levels
        for bootstrap in (False, True)
    ]
    return report
