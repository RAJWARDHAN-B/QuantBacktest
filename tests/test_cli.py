import json

import pandas as pd

from quant_backtester.cli import main


def test_cli_backtest_writes_equity_trades_and_metrics(tmp_path) -> None:
    prices = pd.DataFrame(
        {
            "Open": [100.0, 100.0, 100.0, 100.0],
            "High": [101.0, 101.0, 101.0, 101.0],
            "Low": [99.0, 99.0, 99.0, 99.0],
            "Close": [100.0, 100.0, 100.0, 100.0],
            "Volume": [1000.0] * 4,
        },
        index=pd.date_range("2024-01-01", periods=4),
    )
    input_path = tmp_path / "prices.csv"
    output_path = tmp_path / "report"
    prices.to_csv(input_path)

    main(
        [
            "backtest",
            str(input_path),
            "--symbol",
            "TEST",
            "--output-directory",
            str(output_path),
        ]
    )

    assert (output_path / "equity_curve.csv").exists()
    assert (output_path / "trades.csv").exists()
    assert (output_path / "rejections.csv").exists()
    report = json.loads((output_path / "metrics.json").read_text())
    assert report["final_equity"] == 10000.0


def test_cli_multi_backtest_writes_diagnostics_and_experiment_ledger(tmp_path) -> None:
    closes = [100.0, 100.0, 80.0, 100.0, 100.0, 100.0]
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
    prices.to_csv(tmp_path / "aaa.csv")
    (prices * 2).assign(Volume=1000.0).to_csv(tmp_path / "bbb.csv")
    config_path = tmp_path / "multi.json"
    config_path.write_text(
        json.dumps(
            {
                "initial_cash": 1000.0,
                "max_position_size": 5,
                "strategy": {"lookback": 3, "z_threshold": 1.0},
                "symbols": {
                    "AAA": {"csv": "aaa.csv"},
                    "BBB": {"csv": "bbb.csv"},
                },
            }
        )
    )
    output_path = tmp_path / "multi-report"
    ledger_path = tmp_path / "experiments.jsonl"

    main(
        [
            "multi-backtest",
            str(config_path),
            "--output-directory",
            str(output_path),
            "--experiment-ledger",
            str(ledger_path),
        ]
    )

    assert (output_path / "asset_correlation.csv").exists()
    assert (output_path / "exposure.csv").exists()
    metrics = json.loads((output_path / "metrics.json").read_text())
    assert "max_concentration_hhi" in metrics
    ledger_entry = json.loads(ledger_path.read_text().splitlines()[0])
    assert ledger_entry["kind"] == "multi_asset_backtest"
    assert ledger_entry["configuration"]["symbols"]["AAA"]["csv"] == "aaa.csv"