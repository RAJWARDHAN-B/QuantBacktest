"""Command-line workflows for data ingestion and reproducible backtests."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Sequence

import pandas as pd

from quant_backtester.backtest import run_backtest
from quant_backtester.data import download_history, load_csv
from quant_backtester.multi_asset import (
    asset_return_correlation,
    exposure_report,
    run_multi_asset_backtest,
)
from quant_backtester.performance import performance_report
from quant_backtester.research import append_experiment_record
from quant_backtester.strategy import MeanReversionStrategy


class _JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        reserved = logging.LogRecord("", 0, "", 0, "", (), None).__dict__
        context = {
            key: value
            for key, value in record.__dict__.items()
            if key not in reserved and key not in {"message", "asctime"}
        }
        return json.dumps(
            {
                "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "context": context,
            },
            sort_keys=True,
        )


def _configure_logging(level: str) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(_JsonLogFormatter())
    logging.basicConfig(level=getattr(logging, level), handlers=[handler], force=True)


def _add_log_level(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default="WARNING",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quant-backtester")
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download", help="download and validate daily OHLCV data")
    download.add_argument("ticker")
    download.add_argument("--start", required=True)
    download.add_argument("--end")
    download.add_argument("--output", required=True)
    _add_log_level(download)

    backtest = commands.add_parser("backtest", help="run a strategy against an OHLCV CSV")
    backtest.add_argument("csv_path")
    backtest.add_argument("--symbol", required=True)
    backtest.add_argument("--initial-cash", type=float, default=10000.0)
    backtest.add_argument("--commission", type=float, default=0.0)
    backtest.add_argument("--slippage-bps", type=float, default=0.0)
    backtest.add_argument("--max-position-size", type=int)
    backtest.add_argument("--max-portfolio-exposure", type=float)
    backtest.add_argument("--market-impact-bps", type=float, default=0.0)
    backtest.add_argument("--lookback", type=int, default=20)
    backtest.add_argument("--z-threshold", type=float, default=1.5)
    backtest.add_argument("--output-directory", default="reports/latest")
    backtest.add_argument("--experiment-ledger")
    _add_log_level(backtest)

    multi_backtest = commands.add_parser(
        "multi-backtest", help="run shared-cash multi-asset backtest from a JSON config"
    )
    multi_backtest.add_argument("config_path")
    multi_backtest.add_argument("--output-directory", default="reports/latest")
    multi_backtest.add_argument("--experiment-ledger")
    _add_log_level(multi_backtest)
    return parser


def _load_multi_asset_config(
    config_path: str | Path,
) -> tuple[
    dict[str, object],
    dict[str, pd.DataFrame],
    dict[str, MeanReversionStrategy],
    dict[str, object],
]:
    path = Path(config_path)
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("Multi-asset config must be a JSON object")
    symbols = config.get("symbols")
    if not isinstance(symbols, Mapping) or not symbols:
        raise ValueError("Config must include a non-empty 'symbols' object")
    default_strategy = config.get("strategy", {})
    if not isinstance(default_strategy, Mapping):
        raise ValueError("Config 'strategy' must be an object")

    prices = {}
    strategies = {}
    for configured_symbol, symbol_config in symbols.items():
        symbol = str(configured_symbol).strip().upper()
        if not symbol or symbol in prices:
            raise ValueError("Configured symbols must be unique non-empty names")
        if not isinstance(symbol_config, Mapping) or not symbol_config.get("csv"):
            raise ValueError(f"Symbol {symbol} must include a CSV path")
        csv_path = Path(str(symbol_config["csv"]))
        if not csv_path.is_absolute():
            csv_path = path.parent / csv_path
        prices[symbol] = load_csv(csv_path)
        strategy_parameters = dict(default_strategy)
        symbol_strategy = symbol_config.get("strategy", {})
        if not isinstance(symbol_strategy, Mapping):
            raise ValueError(f"Strategy config for {symbol} must be an object")
        strategy_parameters.update(symbol_strategy)
        unknown_parameters = set(strategy_parameters) - {"lookback", "z_threshold"}
        if unknown_parameters:
            raise ValueError(
                f"Unknown strategy parameters for {symbol}: "
                f"{', '.join(sorted(unknown_parameters))}"
            )
        strategies[symbol] = MeanReversionStrategy(**strategy_parameters)

    execution_keys = {
        "initial_cash",
        "commission",
        "slippage_bps",
        "max_position_size",
        "max_portfolio_exposure",
        "market_impact_bps",
    }
    execution = {key: config[key] for key in execution_keys if key in config}
    return dict(config), prices, strategies, execution


def _write_result_files(output_directory: Path, result, metrics: dict) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    result.equity_curve.to_csv(output_directory / "equity_curve.csv")
    result.trades.to_csv(output_directory / "trades.csv", index=False)
    result.rejections.to_csv(output_directory / "rejections.csv", index=False)
    (output_directory / "metrics.json").write_text(json.dumps(metrics, indent=2))


def main(argv: Sequence[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    _configure_logging(arguments.log_level)
    logger = logging.getLogger("quant_backtester.cli")
    if arguments.command == "download":
        logger.info("history_download_started", extra={"ticker": arguments.ticker})
        data = download_history(
            ticker=arguments.ticker,
            start=arguments.start,
            end=arguments.end,
            output_path=arguments.output,
        )
        print(f"Saved {len(data)} daily bars to {arguments.output}")
        return

    if arguments.command == "multi-backtest":
        config, prices, strategies, execution = _load_multi_asset_config(arguments.config_path)
        logger.info(
            "multi_asset_backtest_started",
            extra={"symbols": sorted(prices), "bar_counts": {key: len(value) for key, value in prices.items()}},
        )
        result = run_multi_asset_backtest(prices, strategies, **execution)
        exposure = exposure_report(result.equity_curve)
        metrics = performance_report(result.equity_curve, result.trades)
        metrics.update(
            {
                "mean_gross_exposure_fraction": float(exposure["gross_exposure_fraction"].mean()),
                "max_gross_exposure_fraction": float(exposure["gross_exposure_fraction"].max()),
                "max_concentration_hhi": float(exposure["concentration_hhi"].max()),
            }
        )
        output_directory = Path(arguments.output_directory)
        _write_result_files(output_directory, result, metrics)
        exposure.to_csv(output_directory / "exposure.csv")
        asset_return_correlation(prices).to_csv(output_directory / "asset_correlation.csv")
        if arguments.experiment_ledger:
            append_experiment_record(
                arguments.experiment_ledger,
                {"kind": "multi_asset_backtest", "configuration": config, "metrics": metrics},
            )
        logger.info(
            "multi_asset_backtest_completed",
            extra={"symbols": sorted(prices), "final_equity": metrics["final_equity"]},
        )
        print(json.dumps(metrics, indent=2))
        return

    prices = load_csv(arguments.csv_path)
    logger.info(
        "backtest_started",
        extra={"symbol": arguments.symbol, "bars": len(prices), "csv_path": arguments.csv_path},
    )
    result = run_backtest(
        prices,
        symbol=arguments.symbol,
        strategy=MeanReversionStrategy(
            lookback=arguments.lookback,
            z_threshold=arguments.z_threshold,
        ),
        initial_cash=arguments.initial_cash,
        commission=arguments.commission,
        slippage_bps=arguments.slippage_bps,
        max_position_size=arguments.max_position_size,
        max_portfolio_exposure=arguments.max_portfolio_exposure,
        market_impact_bps=arguments.market_impact_bps,
    )
    metrics = performance_report(result.equity_curve, result.trades)
    output_directory = Path(arguments.output_directory)
    _write_result_files(output_directory, result, metrics)
    if arguments.experiment_ledger:
        append_experiment_record(
            arguments.experiment_ledger,
            {
                "kind": "single_asset_backtest",
                "symbol": arguments.symbol,
                "configuration": {
                    "csv_path": arguments.csv_path,
                    "lookback": arguments.lookback,
                    "z_threshold": arguments.z_threshold,
                    "initial_cash": arguments.initial_cash,
                    "commission": arguments.commission,
                    "slippage_bps": arguments.slippage_bps,
                    "max_position_size": arguments.max_position_size,
                    "max_portfolio_exposure": arguments.max_portfolio_exposure,
                    "market_impact_bps": arguments.market_impact_bps,
                },
                "metrics": metrics,
            },
        )
    logger.info(
        "backtest_completed",
        extra={"symbol": arguments.symbol, "final_equity": metrics["final_equity"]},
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
