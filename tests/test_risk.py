from __future__ import annotations

import json
from math import log
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from quant_backtester.cli import main
from quant_backtester.risk import (
    _check_positive_semidefinite,
    _moments,
    christoffersen_test,
    current_exposures,
    historical_var,
    kupiec_test,
    monte_carlo_var,
    parametric_var,
    risk_report,
    rolling_var,
    shock_test,
    stress_test,
)


def gaussian_returns(size: int = 500, seed: int = 1, scale: float = 0.01) -> pd.Series:
    generator = np.random.default_rng(seed)
    return pd.Series(
        generator.normal(0.0005, scale, size),
        index=pd.bdate_range("2020-01-01", periods=size),
    )


def ohlcv(closes: list[float] | np.ndarray, index: pd.DatetimeIndex) -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {
            "Open": closes,
            "High": closes * 1.01,
            "Low": closes * 0.99,
            "Close": closes,
            "Volume": 1000.0,
        },
        index=index,
    )


def test_historical_var_and_es_match_hand_computed_tail() -> None:
    returns = pd.Series(-np.arange(1, 101) / 100.0)

    estimate = historical_var(returns.sample(frac=1.0, random_state=3), confidence=0.95)

    # "higher" quantile of losses 1%..100% at 95% is the 96th smallest loss.
    assert estimate.var == pytest.approx(0.96)
    assert estimate.expected_shortfall == pytest.approx((0.96 + 0.97 + 0.98 + 0.99 + 1.00) / 5)
    assert estimate.observations == 100


def test_historical_var_compounds_overlapping_multi_bar_returns() -> None:
    returns = pd.Series([0.10, -0.10] * 15)

    estimate = historical_var(returns, confidence=0.95, horizon=2)

    # Every overlapping two-bar window compounds to 1.1 * 0.9 - 1 = -1%.
    assert estimate.var == pytest.approx(0.01)
    assert estimate.expected_shortfall == pytest.approx(0.01)
    assert estimate.observations == 29


def test_parametric_var_matches_normal_quantile() -> None:
    raw = np.random.default_rng(0).normal(size=250)
    returns = pd.Series((raw - raw.mean()) / raw.std(ddof=1) * 0.01)

    estimate = parametric_var(returns, confidence=0.99)

    assert estimate.var == pytest.approx(0.01 * 2.3263, abs=1e-4)
    assert estimate.expected_shortfall == pytest.approx(
        0.01 * NormalDist().pdf(NormalDist().inv_cdf(0.99)) / 0.01
    )


def test_parametric_var_scales_with_square_root_of_horizon() -> None:
    raw = np.random.default_rng(0).normal(size=250)
    returns = pd.Series((raw - raw.mean()) / raw.std(ddof=1) * 0.01)

    one_day = parametric_var(returns, confidence=0.95)
    ten_day = parametric_var(returns, confidence=0.95, horizon=10)

    assert ten_day.var == pytest.approx(one_day.var * np.sqrt(10))


def test_sample_moments_match_pandas_estimators() -> None:
    returns = pd.Series(np.random.default_rng(5).standard_t(df=4, size=300) * 0.01)

    mean, std, skew, kurtosis = _moments(returns.to_numpy()[np.newaxis, :])

    assert mean[0] == pytest.approx(returns.mean())
    assert std[0] == pytest.approx(returns.std(ddof=1))
    assert skew[0] == pytest.approx(returns.skew())
    assert kurtosis[0] == pytest.approx(returns.kurt())


def test_cornish_fisher_raises_var_for_fat_left_tail() -> None:
    generator = np.random.default_rng(11)
    returns = pd.Series(generator.normal(0.0, 0.01, 1000))
    returns.iloc[::50] = -0.06

    gaussian = parametric_var(returns, confidence=0.99)
    adjusted = parametric_var(returns, confidence=0.99, cornish_fisher=True)

    assert adjusted.method == "cornish_fisher"
    assert adjusted.var > gaussian.var


def test_expected_shortfall_is_at_least_var_for_every_method() -> None:
    returns = gaussian_returns(size=600, seed=8)
    assets = pd.DataFrame(
        {"AAA": returns.to_numpy(), "BBB": gaussian_returns(size=600, seed=9).to_numpy()}
    )
    weights = pd.Series({"AAA": 0.6, "BBB": 0.3})

    for horizon in (1, 5):
        for confidence in (0.95, 0.99):
            estimates = [
                historical_var(returns, confidence, horizon),
                parametric_var(returns, confidence, horizon),
                parametric_var(returns, confidence, horizon, cornish_fisher=True),
                monte_carlo_var(assets, weights, confidence, horizon, 5000, seed=1),
                monte_carlo_var(assets, weights, confidence, horizon, 5000, seed=1, bootstrap=True),
            ]
            for estimate in estimates:
                assert estimate.expected_shortfall >= estimate.var, estimate


def test_monte_carlo_var_is_seeded_and_converges_to_parametric() -> None:
    generator = np.random.default_rng(21)
    assets = pd.DataFrame(
        generator.multivariate_normal(
            [0.0004, 0.0002], [[1e-4, 4e-5], [4e-5, 2.25e-4]], size=2000
        ),
        columns=["AAA", "BBB"],
    )
    weights = pd.Series({"BBB": 0.4, "AAA": 0.6})

    first = monte_carlo_var(assets, weights, confidence=0.99, n_simulations=5000, seed=7)
    second = monte_carlo_var(assets, weights, confidence=0.99, n_simulations=5000, seed=7)
    converged = monte_carlo_var(assets, weights, confidence=0.99, n_simulations=200_000, seed=3)
    parametric = parametric_var(assets @ weights.reindex(assets.columns), confidence=0.99)

    assert first == second
    assert converged.var == pytest.approx(parametric.var, rel=0.02)
    assert converged.expected_shortfall == pytest.approx(parametric.expected_shortfall, rel=0.02)


def test_bootstrap_var_only_samples_observed_portfolio_returns() -> None:
    # 10% of rows lose 5%, so the 95% tail of a 50% position is exactly a 2.5% loss.
    assets = pd.DataFrame({"AAA": [-0.05] * 4 + [0.01] * 36})

    estimate = monte_carlo_var(
        assets, pd.Series({"AAA": 0.5}), confidence=0.95, n_simulations=2000, seed=2, bootstrap=True
    )

    assert estimate.method == "bootstrap"
    assert estimate.var == pytest.approx(0.025)


def test_monte_carlo_var_rejects_misaligned_weights_and_invalid_covariance() -> None:
    assets = pd.DataFrame({"AAA": gaussian_returns().to_numpy()})

    with pytest.raises(ValueError, match="Weights"):
        monte_carlo_var(assets, pd.Series({"BBB": 1.0}))
    with pytest.raises(ValueError, match="positive semidefinite"):
        _check_positive_semidefinite(np.array([[1.0, 2.0], [2.0, 1.0]]))


def test_rolling_var_uses_only_returns_before_each_forecast() -> None:
    returns = gaussian_returns(size=400, seed=4)
    window = 100
    target = 251
    perturbed = returns.copy()
    perturbed.iloc[target:] *= 5.0

    baseline = rolling_var(returns, window, confidence=0.95)
    changed = rolling_var(perturbed, window, confidence=0.95)

    assert baseline.index[0] == returns.index[window]
    assert baseline["var"].iloc[0] == pytest.approx(historical_var(returns.iloc[:window]).var)
    assert baseline["realized_loss"].iloc[0] == pytest.approx(-returns.iloc[window])
    unchanged = baseline.index <= returns.index[target]
    pd.testing.assert_series_equal(baseline.loc[unchanged, "var"], changed.loc[unchanged, "var"])
    assert not baseline["var"].equals(changed["var"])


def test_rolling_parametric_var_matches_point_estimate() -> None:
    returns = gaussian_returns(size=200, seed=6)

    frame = rolling_var(returns, window=60, confidence=0.99, method="cornish_fisher")

    expected = parametric_var(returns.iloc[:60], confidence=0.99, cornish_fisher=True)
    assert frame["var"].iloc[0] == pytest.approx(expected.var)


def test_kupiec_test_matches_expected_coverage_and_handles_boundaries() -> None:
    on_target = kupiec_test(pd.Series([True] * 5 + [False] * 95), confidence=0.95)
    no_exceedances = kupiec_test(pd.Series([False] * 250), confidence=0.99)
    all_exceedances = kupiec_test(pd.Series([1] * 20), confidence=0.95)

    assert on_target["lr_statistic"] == pytest.approx(0.0)
    assert on_target["p_value"] == pytest.approx(1.0)
    assert no_exceedances["lr_statistic"] == pytest.approx(-2 * 250 * log(0.99))
    assert no_exceedances["p_value"] < 0.05
    assert all_exceedances["exceedances"] == 20
    assert all_exceedances["p_value"] < 1e-6


def test_christoffersen_test_detects_clustered_exceedances() -> None:
    clustered = christoffersen_test(pd.Series([0] * 90 + [1] * 10))
    spread = christoffersen_test(pd.Series(([0] * 9 + [1]) * 10))

    assert (clustered["n00"], clustered["n01"], clustered["n10"], clustered["n11"]) == (89, 1, 0, 9)
    assert clustered["p_value"] < 0.001
    assert spread["n11"] == 0
    assert spread["p_value"] > clustered["p_value"]


def test_stress_test_replays_scenario_against_exposures() -> None:
    index = pd.bdate_range("2020-02-17", "2020-03-24")
    aaa = pd.Series(100.0, index=index)
    aaa.loc["2020-02-19"] = 110.0
    aaa.loc["2020-02-20":"2020-03-23"] = 80.0
    aaa.loc["2020-03-24"] = 85.0
    bbb = pd.Series(50.0, index=index)
    bbb.loc["2020-03-23"] = 45.0
    exposures = pd.Series({"AAA": 1000.0, "BBB": 500.0, "CCC": 300.0})

    table = stress_test(
        exposures,
        {"AAA": ohlcv(aaa, index), "BBB": ohlcv(bbb, index)},
        scenarios={"Test crash": ("2020-02-19", "2020-03-23")},
    ).set_index("symbol")

    assert table.loc["AAA", "scenario_return"] == pytest.approx(-0.20)
    assert table.loc["AAA", "pnl"] == pytest.approx(-200.0)
    assert table.loc["AAA", "worst_drawdown"] == pytest.approx(80.0 / 110.0 - 1.0)
    assert table.loc["BBB", "pnl"] == pytest.approx(-50.0)
    assert table.loc["CCC", "status"] == "missing"
    assert pd.isna(table.loc["CCC", "pnl"])
    assert table.loc["TOTAL", "status"] == "partial"
    assert table.loc["TOTAL", "pnl"] == pytest.approx(-250.0)
    assert table.loc["TOTAL", "scenario_return"] == pytest.approx(-250.0 / 1500.0)
    assert table.loc["TOTAL", "worst_drawdown"] == pytest.approx(1250.0 / 1600.0 - 1.0)


def test_stress_test_uses_explicit_proxy_for_missing_history() -> None:
    index = pd.bdate_range("2020-02-17", "2020-03-24")
    spy = pd.Series(100.0, index=index)
    spy.loc["2020-03-23"] = 70.0
    late_listing = ohlcv([10.0] * 5, pd.bdate_range("2020-03-18", periods=5))

    table = stress_test(
        pd.Series({"NEW": 200.0}),
        {"SPY": ohlcv(spy, index), "NEW": late_listing},
        scenarios={"Test crash": ("2020-02-19", "2020-03-23")},
        proxies={"NEW": "SPY"},
    ).set_index("symbol")

    assert table.loc["NEW", "status"] == "proxy:SPY"
    assert table.loc["NEW", "pnl"] == pytest.approx(-60.0)
    assert table.loc["TOTAL", "status"] == "ok"


def test_shock_test_applies_per_symbol_and_uniform_shocks() -> None:
    exposures = pd.Series({"SPY": 1000.0, "QQQ": 500.0})

    per_symbol = shock_test(exposures, {"SPY": -0.2, "QQQ": -0.3}).set_index("symbol")
    uniform = shock_test(exposures, -0.1).set_index("symbol")

    assert per_symbol.loc["SPY", "pnl"] == pytest.approx(-200.0)
    assert per_symbol.loc["QQQ", "pnl"] == pytest.approx(-150.0)
    assert per_symbol.loc["TOTAL", "pnl"] == pytest.approx(-350.0)
    assert uniform.loc["TOTAL", "pnl"] == pytest.approx(-150.0)
    with pytest.raises(ValueError, match="Shocks"):
        shock_test(exposures, {"SPY": -1.5})


def test_current_exposures_reads_final_marked_holdings() -> None:
    curve = pd.DataFrame(
        {
            "equity": [100.0, 120.0],
            "position_AAA": [0, 2],
            "close_AAA": [10.0, 15.0],
            "position_BBB": [1, 0],
            "close_BBB": [20.0, 25.0],
        },
        index=pd.date_range("2024-01-01", periods=2),
    )

    exposures = current_exposures(curve)

    assert exposures.to_dict() == {"AAA": 30.0, "BBB": 0.0}


def test_risk_report_lists_estimates_backtests_and_skipped_methods() -> None:
    equity = 10000.0 * (1.0 + gaussian_returns(size=300, seed=12)).cumprod()
    curve = pd.DataFrame({"equity": equity})

    report = risk_report(curve, confidence_levels=(0.95, 0.99), rolling_window=250)

    methods = {(row["method"], row["confidence"]) for row in report["estimates"]}
    assert ("historical", 0.99) in methods
    assert ("cornish_fisher", 0.95) in methods
    assert report["estimates"][0]["var_amount"] == pytest.approx(
        report["estimates"][0]["var"] * equity.iloc[-1]
    )
    assert [backtest["confidence"] for backtest in report["var_backtests"]] == [0.95, 0.99]
    assert report["var_backtests"][0]["kupiec"]["observations"] == 299 - 250

    short = risk_report(curve.iloc[:40], confidence_levels=(0.99,))
    assert {row["method"] for row in short["estimates"]} == {"parametric", "cornish_fisher"}
    assert short["skipped"][0]["method"] == "historical"


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda returns: historical_var(returns, confidence=1.0), "Confidence"),
        (lambda returns: historical_var(returns, confidence=0.0), "Confidence"),
        (lambda returns: parametric_var(returns, horizon=0), "Horizon"),
        (lambda returns: historical_var(returns.iloc[:10], confidence=0.95), "at least 20"),
        (lambda returns: parametric_var(returns.iloc[:29]), "at least 30"),
        (lambda returns: rolling_var(returns, window=10, confidence=0.95), "window"),
        (lambda returns: rolling_var(returns, window=500), "more returns"),
        (lambda returns: rolling_var(returns, window=100, method="monte_carlo"), "method"),
    ],
)
def test_risk_functions_reject_invalid_inputs(call, message) -> None:
    with pytest.raises(ValueError, match=message):
        call(gaussian_returns(size=300))


def test_cli_backtest_risk_writes_risk_outputs_and_ledger(tmp_path) -> None:
    index = pd.bdate_range("2021-01-01", periods=320)
    closes = 100.0 * (1.0 + gaussian_returns(size=320, seed=14, scale=0.02)).cumprod()
    ohlcv(closes.to_numpy(), index).to_csv(tmp_path / "prices.csv")
    output_path = tmp_path / "report"
    ledger_path = tmp_path / "experiments.jsonl"

    main(
        [
            "backtest",
            str(tmp_path / "prices.csv"),
            "--symbol",
            "TEST",
            "--output-directory",
            str(output_path),
            "--experiment-ledger",
            str(ledger_path),
            "--risk",
            "--confidence",
            "0.95",
            "--risk-window",
            "100",
        ]
    )

    risk = json.loads((output_path / "risk.json").read_text())
    assert {row["method"] for row in risk["equity_curve"]["estimates"]} == {
        "historical",
        "parametric",
        "cornish_fisher",
    }
    assert risk["equity_curve"]["var_backtests"][0]["window"] == 100
    rolling = pd.read_csv(output_path / "rolling_var.csv", index_col=0)
    assert list(rolling.columns) == ["realized_loss", "var_95", "exceedance_95"]
    assert json.loads(ledger_path.read_text())["risk"] == risk


def test_cli_multi_backtest_risk_config_writes_holdings_and_stress(tmp_path) -> None:
    index = pd.bdate_range("2019-06-03", "2020-12-31")
    for symbol, seed in (("AAA", 15), ("BBB", 16)):
        closes = 100.0 * (1.0 + gaussian_returns(size=len(index), seed=seed, scale=0.02)).cumprod()
        ohlcv(closes.to_numpy(), index).to_csv(tmp_path / f"{symbol.lower()}.csv")
    config_path = tmp_path / "multi.json"
    config_path.write_text(
        json.dumps(
            {
                "initial_cash": 10000.0,
                "strategy": {"lookback": 10, "z_threshold": 1.0},
                "symbols": {"AAA": {"csv": "aaa.csv"}, "BBB": {"csv": "bbb.csv"}},
                "risk": {
                    "confidence_levels": [0.95],
                    "rolling_window": 100,
                    "monte_carlo": {"n_simulations": 2000, "seed": 7},
                    "shocks": {"AAA": -0.2, "BBB": -0.3},
                },
            }
        )
    )
    output_path = tmp_path / "multi-report"

    main(["multi-backtest", str(config_path), "--output-directory", str(output_path)])

    risk = json.loads((output_path / "risk.json").read_text())
    assert {row["method"] for row in risk["holdings"]["estimates"]} == {"monte_carlo", "bootstrap"}
    assert risk["holdings"]["seed"] == 7
    stress = pd.read_csv(output_path / "stress.csv")
    totals = stress.loc[stress["symbol"] == "TOTAL"].set_index("scenario")
    assert totals.loc["Global Financial Crisis", "status"] in {"ok", "missing"}
    assert totals.loc["Hypothetical shock", "kind"] == "hypothetical"
    assert {row["scenario"] for row in risk["stress"]} == set(totals.index)
    assert (output_path / "rolling_var.csv").exists()
