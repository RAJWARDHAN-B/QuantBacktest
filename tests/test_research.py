import json

import pandas as pd
import pytest

from quant_backtester.research import (
    chronological_split,
    evaluate_walk_forward,
    walk_forward_splits,
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