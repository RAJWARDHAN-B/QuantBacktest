"""Time-ordered data splitting helpers for strategy research."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from math import isfinite
from pathlib import Path
from typing import Any

import pandas as pd

from quant_backtester.backtest import BacktestResult, run_backtest
from quant_backtester.data import validate_ohlcv
from quant_backtester.performance import performance_report
from quant_backtester.strategy import MeanReversionStrategy, Strategy


StrategyFactory = Callable[[Mapping[str, Any], pd.DataFrame], Strategy]


def mean_reversion_factory(parameters: Mapping[str, Any], training: pd.DataFrame) -> Strategy:
    """Default factory: rule-based strategies ignore the training window."""
    return MeanReversionStrategy(**dict(parameters))


@dataclass(frozen=True)
class FoldRun:
    """The selected parameter set's held-out run for one walk-forward fold."""

    fold: int
    parameters: dict[str, Any]
    strategy: Strategy
    result: BacktestResult


@dataclass
class WalkForwardResult:
    """Fold selections and out-of-sample parameter sensitivity results."""

    folds: pd.DataFrame
    sensitivity: pd.DataFrame
    parameter_summary: pd.DataFrame
    test_runs: list[FoldRun] = field(default_factory=list)


def append_experiment_record(path: str | Path, record: Mapping[str, Any]) -> None:
    """Append one reproducible experiment record as a JSON Lines entry."""
    ledger_path = Path(path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        **record,
    }
    with ledger_path.open("a", encoding="utf-8") as ledger:
        ledger.write(json.dumps(entry, sort_keys=True, default=_json_default) + "\n")


def _json_default(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def chronological_split(
    data: pd.DataFrame,
    train_fraction: float = 0.7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data by time order; the trailing observations are held out."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be strictly between 0 and 1")
    if len(data) < 2:
        raise ValueError("At least two observations are required")

    split_index = int(len(data) * train_fraction)
    split_index = min(max(split_index, 1), len(data) - 1)
    return data.iloc[:split_index].copy(), data.iloc[split_index:].copy()


def walk_forward_splits(
    data: pd.DataFrame,
    train_size: int,
    test_size: int,
    step_size: int | None = None,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Yield expanding training windows followed by non-overlapping test windows."""
    if train_size <= 0:
        raise ValueError("train_size must be positive")
    if test_size <= 0:
        raise ValueError("test_size must be positive")
    step = test_size if step_size is None else step_size
    if step <= 0:
        raise ValueError("step_size must be positive")
    if train_size >= len(data):
        raise ValueError("train_size must be smaller than the number of observations")

    test_start = train_size
    while test_start < len(data):
        test_end = min(test_start + test_size, len(data))
        yield data.iloc[:test_start].copy(), data.iloc[test_start:test_end].copy()
        test_start += step


def evaluate_walk_forward(
    prices: pd.DataFrame,
    symbol: str,
    parameter_sets: Sequence[Mapping[str, Any]],
    train_size: int,
    test_size: int,
    step_size: int | None = None,
    selection_metric: str = "sharpe_ratio",
    initial_cash: float = 10000.0,
    commission: float = 0.0,
    slippage_bps: float = 0.0,
    max_position_size: int | None = None,
    max_portfolio_exposure: float | None = None,
    market_impact_bps: float = 0.0,
    experiment_ledger_path: str | Path | None = None,
    strategy_factory: StrategyFactory | None = None,
    validation_size: int | None = None,
) -> WalkForwardResult:
    """Select parameters on each training window and report held-out results.

    Every parameter set is also run on each test window for sensitivity analysis.
    Training data is supplied only as indicator warm-up during those test runs.

    ``strategy_factory(parameters, training)`` builds each strategy; strategies that
    learn from ``training`` must also set ``validation_size`` so selection scores the
    last ``validation_size`` training bars with a strategy built from the bars before
    them. The selected parameters are then rebuilt on the full training window.
    """
    if not parameter_sets:
        raise ValueError("At least one parameter set is required")
    if train_size < 2 or test_size < 2:
        raise ValueError("Training and test windows must each contain at least two observations")
    if validation_size is not None and not 2 <= validation_size < train_size:
        raise ValueError("validation_size must be at least 2 and smaller than train_size")

    factory = mean_reversion_factory if strategy_factory is None else strategy_factory
    selection_column = (
        "selected_train_metric" if validation_size is None else "selected_validation_metric"
    )
    cleaned = validate_ohlcv(prices)
    backtest_options = {
        "initial_cash": initial_cash,
        "commission": commission,
        "slippage_bps": slippage_bps,
        "max_position_size": max_position_size,
        "max_portfolio_exposure": max_portfolio_exposure,
        "market_impact_bps": market_impact_bps,
    }
    fold_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    test_runs: list[FoldRun] = []

    for fold_number, (training, testing) in enumerate(
        walk_forward_splits(cleaned, train_size, test_size, step_size),
        start=1,
    ):
        if validation_size is None:
            fitting, selection_window = None, training
        else:
            fitting = training.iloc[:-validation_size]
            selection_window = training.iloc[-validation_size:]
        training_scores: list[float] = []
        training_metric_values: list[float | None] = []
        for parameters in parameter_sets:
            training_result = run_backtest(
                selection_window,
                symbol,
                factory(parameters, training if fitting is None else fitting),
                warmup_prices=fitting,
                **backtest_options,
            )
            training_report = performance_report(
                training_result.equity_curve,
                training_result.trades,
            )
            if selection_metric not in training_report:
                raise ValueError(f"Unknown selection metric: {selection_metric}")
            metric_value = training_report[selection_metric]
            is_valid_metric = (
                isinstance(metric_value, (int, float)) and isfinite(float(metric_value))
            )
            score = (
                float(metric_value)
                if is_valid_metric
                else float("-inf")
            )
            training_scores.append(score)
            training_metric_values.append(float(metric_value) if is_valid_metric else None)

        selected_index = max(range(len(training_scores)), key=training_scores.__getitem__)
        fold_row: dict[str, Any] = {
            "fold": fold_number,
            "train_start": training.index[0],
            "train_end": training.index[-1],
        }
        if validation_size is not None:
            fold_row["validation_start"] = selection_window.index[0]
            fold_row["validation_end"] = selection_window.index[-1]
        fold_rows.append(
            {
                **fold_row,
                "test_start": testing.index[0],
                "test_end": testing.index[-1],
                "selected_parameter_set": selected_index,
                "selected_parameters": dict(parameter_sets[selected_index]),
                selection_column: training_metric_values[selected_index],
            }
        )

        for parameter_index, parameters in enumerate(parameter_sets):
            strategy = factory(parameters, training)
            test_result = run_backtest(
                testing,
                symbol,
                strategy,
                warmup_prices=training,
                **backtest_options,
            )
            if parameter_index == selected_index:
                test_runs.append(FoldRun(fold_number, dict(parameters), strategy, test_result))
            test_report = performance_report(test_result.equity_curve, test_result.trades)
            sensitivity_rows.append(
                {
                    "fold": fold_number,
                    "parameter_set": parameter_index,
                    "parameters": dict(parameters),
                    "selected": parameter_index == selected_index,
                    **test_report,
                }
            )

    sensitivity = pd.DataFrame(sensitivity_rows)
    parameter_summary = (
        sensitivity.groupby("parameter_set", as_index=False)
        .agg(
            mean_total_return=("total_return", "mean"),
            std_total_return=("total_return", "std"),
            mean_sharpe_ratio=("sharpe_ratio", "mean"),
            mean_max_drawdown=("max_drawdown", "mean"),
            selected_folds=("selected", "sum"),
        )
    )
    parameter_summary.insert(
        1,
        "parameters",
        parameter_summary["parameter_set"].map(
            lambda parameter_index: dict(parameter_sets[int(parameter_index)])
        ),
    )
    result = WalkForwardResult(
        folds=pd.DataFrame(fold_rows),
        sensitivity=sensitivity,
        parameter_summary=parameter_summary,
        test_runs=test_runs,
    )
    if experiment_ledger_path is not None:
        configuration: dict[str, Any] = {
            "symbol": symbol,
            "parameter_sets": [dict(parameters) for parameters in parameter_sets],
            "train_size": train_size,
            "test_size": test_size,
            "step_size": step_size,
            "selection_metric": selection_metric,
            **backtest_options,
        }
        if strategy_factory is not None:
            configuration["strategy_factory"] = getattr(
                strategy_factory, "__qualname__", type(strategy_factory).__name__
            )
        if validation_size is not None:
            configuration["validation_size"] = validation_size
        append_experiment_record(
            experiment_ledger_path,
            {
                "kind": "walk_forward",
                "configuration": configuration,
                "folds": result.folds.to_dict(orient="records"),
                "parameter_summary": result.parameter_summary.to_dict(orient="records"),
            },
        )
    return result


def stitch_out_of_sample(runs: Sequence[FoldRun], initial_cash: float) -> pd.DataFrame:
    """Chain held-out fold returns into one equity curve starting at ``initial_cash``.

    Each fold starts flat with fresh cash, so the close-to-close move between the
    last bar of one fold and the first bar of the next is not earned.
    """
    if not runs:
        raise ValueError("At least one fold run is required")
    frames = []
    for run in runs:
        equity = run.result.equity_curve["equity"].astype(float)
        frames.append(
            pd.DataFrame({"return": equity.pct_change().fillna(0.0), "fold": run.fold})
        )
    stitched = pd.concat(frames)
    if not stitched.index.is_monotonic_increasing or stitched.index.has_duplicates:
        raise ValueError("Fold test windows must be chronological and non-overlapping")
    stitched.insert(0, "equity", initial_cash * (1.0 + stitched["return"]).cumprod())
    return stitched.drop(columns="return")