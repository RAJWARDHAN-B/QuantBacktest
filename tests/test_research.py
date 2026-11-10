import json

import numpy as np
import pandas as pd
import pytest

from quant_backtester.research import (
    chronological_split,
    evaluate_walk_forward,
    stitch_out_of_sample,
    walk_forward_splits,
)
from quant_backtester.strategy import MeanReversionStrategy


PARITY_PARAMETER_SETS = [
    {"lookback": 5, "z_threshold": 1.0},
    {"lookback": 10, "z_threshold": 1.5},
    {"lookback": 20, "z_threshold": 2.0},
]


def synthetic_prices(size: int = 260, seed: int = 42) -> pd.DataFrame:
    generator = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(generator.normal(0, 0.015, size)))
    open_ = close * (1 + generator.normal(0, 0.003, size))
    return pd.DataFrame(
        {
            "Open": open_,
            "High": np.maximum(open_, close) * 1.004,
            "Low": np.minimum(open_, close) * 0.996,
            "Close": close,
            "Volume": generator.integers(1000, 5000, size).astype(float),
        },
        index=pd.bdate_range("2020-01-01", periods=size),
    )


def test_chronological_split_keeps_earlier_observations_in_train() -> None:
    data = pd.DataFrame({"close": range(10)}, index=pd.date_range("2024-01-01", periods=10))

    train, test = chronological_split(data, train_fraction=0.6)

    assert len(train) == 6
    assert len(test) == 4
    assert train.index[-1] < test.index[0]
    assert train.iloc[-1]["close"] == 5
    assert test.iloc[0]["close"] == 6


def test_walk_forward_splits_produce_expanding_train_and_forward_test_windows() -> None:
    data = pd.DataFrame({"close": range(11)}, index=pd.date_range("2024-01-01", periods=11))

    folds = list(walk_forward_splits(data, train_size=5, test_size=2, step_size=2))

    assert [(len(train), len(test)) for train, test in folds] == [(5, 2), (7, 2), (9, 2)]
    assert all(train.index[-1] < test.index[0] for train, test in folds)


def test_walk_forward_splits_reject_invalid_sizes() -> None:
    with pytest.raises(ValueError, match="train_size"):
        list(walk_forward_splits(pd.DataFrame({"x": [1, 2]}), train_size=0, test_size=1))


def test_walk_forward_evaluation_reports_out_of_sample_parameter_sensitivity(tmp_path) -> None:
    closes = [100.0, 100.0, 80.0, 100.0, 100.0, 100.0, 80.0, 100.0, 100.0, 100.0]
    prices = pd.DataFrame(
        {
            "Open": closes,
            "High": [value + 1.0 for value in closes],
            "Low": [value - 1.0 for value in closes],
            "Close": closes,
            "Volume": [1000.0] * len(closes),
        },
        index=pd.date_range("2024-01-01", periods=len(closes)),
    )

    ledger_path = tmp_path / "research" / "experiments.jsonl"
    result = evaluate_walk_forward(
        prices,
        symbol="TEST",
        parameter_sets=[
            {"lookback": 3, "z_threshold": 1.0},
            {"lookback": 4, "z_threshold": 1.5},
        ],
        train_size=6,
        test_size=2,
        experiment_ledger_path=ledger_path,
    )

    assert len(result.folds) == 2
    assert len(result.sensitivity) == 4
    assert len(result.parameter_summary) == 2
    assert result.folds.iloc[0]["test_start"] == prices.index[6]
    assert result.sensitivity.groupby("fold")["observations"].first().tolist() == [1, 1]
    entry = json.loads(ledger_path.read_text().splitlines()[0])
    assert entry["kind"] == "walk_forward"
    assert len(entry["folds"]) == 2


def test_default_walk_forward_matches_pre_factory_results() -> None:
    # Values captured from the implementation before the strategy factory existed.
    result = evaluate_walk_forward(
        synthetic_prices(),
        symbol="SYN",
        parameter_sets=PARITY_PARAMETER_SETS,
        train_size=120,
        test_size=40,
        commission=1.0,
        slippage_bps=5,
    )

    assert result.folds["selected_parameter_set"].tolist() == [1, 1, 1, 1]
    assert result.folds["selected_train_metric"].tolist() == [
        0.42791663141519615,
        1.2807241696752192,
        1.3458305739605745,
        1.0452266153424492,
    ]
    assert result.sensitivity["final_equity"].tolist() == [
        10112.57308080826,
        10783.588983492564,
        10000.0,
        10054.51208141698,
        10206.646119321413,
        10493.82461372987,
        10124.170540419902,
        9890.048759457412,
        10246.293121651664,
        9983.997884714738,
        9988.790909505187,
        9988.790909505187,
    ]
    assert result.sensitivity["trade_count"].tolist() == [3, 2, 0, 3, 1, 1, 4, 1, 1, 2, 1, 1]


def test_explicit_mean_reversion_factory_matches_default() -> None:
    options = {
        "symbol": "SYN",
        "parameter_sets": PARITY_PARAMETER_SETS,
        "train_size": 120,
        "test_size": 40,
        "commission": 1.0,
    }

    default = evaluate_walk_forward(synthetic_prices(), **options)
    explicit = evaluate_walk_forward(
        synthetic_prices(),
        strategy_factory=lambda parameters, training: MeanReversionStrategy(**parameters),
        **options,
    )

    pd.testing.assert_frame_equal(default.folds, explicit.folds)
    pd.testing.assert_frame_equal(default.sensitivity, explicit.sensitivity)
    assert [run.fold for run in default.test_runs] == [1, 2, 3, 4]


def test_validation_size_selects_on_bars_unseen_by_the_fitted_strategy() -> None:
    prices = synthetic_prices()
    calls: list[tuple[pd.Timestamp, int]] = []

    def recording_factory(parameters, training):
        calls.append((training.index[-1], len(training)))
        return MeanReversionStrategy(**parameters)

    result = evaluate_walk_forward(
        prices,
        symbol="SYN",
        parameter_sets=PARITY_PARAMETER_SETS[:2],
        train_size=120,
        test_size=40,
        strategy_factory=recording_factory,
        validation_size=30,
    )

    first_fold = result.folds.iloc[0]
    assert first_fold["validation_start"] == prices.index[90]
    assert first_fold["validation_end"] == prices.index[119]
    assert "selected_validation_metric" in result.folds
    # Selection strategies are built on the 90 pre-validation bars, test strategies on all 120.
    assert calls[:4] == [(prices.index[89], 90)] * 2 + [(prices.index[119], 120)] * 2


def test_walk_forward_rejects_validation_window_as_large_as_training() -> None:
    with pytest.raises(ValueError, match="validation_size"):
        evaluate_walk_forward(
            synthetic_prices(),
            symbol="SYN",
            parameter_sets=PARITY_PARAMETER_SETS,
            train_size=120,
            test_size=40,
            validation_size=120,
        )


def test_stitch_out_of_sample_chains_selected_fold_returns() -> None:
    result = evaluate_walk_forward(
        synthetic_prices(),
        symbol="SYN",
        parameter_sets=PARITY_PARAMETER_SETS,
        train_size=120,
        test_size=40,
    )

    stitched = stitch_out_of_sample(result.test_runs, initial_cash=10000.0)

    fold_growth = [run.result.final_equity / 10000.0 for run in result.test_runs]
    assert len(stitched) == 140
    assert stitched["fold"].iloc[[0, -1]].tolist() == [1, 4]
    assert stitched["equity"].iloc[-1] == pytest.approx(10000.0 * np.prod(fold_growth))