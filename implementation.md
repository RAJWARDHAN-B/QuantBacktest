# Implementation Plan: Portfolio Risk Metrics and ML Signals

This plan extends the existing backtester with two tracks that reuse its data validation, event loop, portfolio accounting, multi-asset engine, walk-forward research, experiment ledger, CLI, and dashboard:

- **Track A — Risk engine:** Value-at-Risk (VaR), Expected Shortfall (ES), VaR backtesting, and stress tests.
- **Track B — ML signals:** leakage-safe features, walk-forward model training, and an ML strategy that runs through the same event-driven engine and cost model.

Order: Track A first (smaller, no new engine interfaces), then Track B.

Out of scope here (separate repositories): C++ limit order book, RL execution sandbox, options pricing, live broker/tick data.

---

## 1. Guiding Principles

1. **No look-ahead.** Any value used at bar $t$ must be computable from bars $\le t$. Labels that use future prices exist only inside training windows and are purged at window edges.
2. **Same engine, same costs.** ML strategies run through `run_backtest` / `run_multi_asset_backtest` with identical commission, slippage, impact, and exposure limits. No separate "vectorized" shortcut for reported results.
3. **Out-of-sample or it didn't happen.** Reported ML and risk-model results come from walk-forward test windows only.
4. **Hand-checkable tests.** Every formula gets a test with a small input whose answer is computed by hand.
5. **Existing behavior is unchanged.** All current tests must keep passing after each milestone; defaults preserve current results bit-for-bit.
6. **Minimal dependencies.** Track A uses only NumPy, pandas, and the standard library (`statistics.NormalDist`, `math.erfc`). Track B adds scikit-learn as an optional extra.

---

## 2. Current Touchpoints

| Existing component | Relevant fact | Used by |
|---|---|---|
| `data.validate_ohlcv` | Data contract for all inputs | A, B |
| `returns.simple_returns`, `log_returns` | Close-to-close returns | A, B |
| `backtest.run_backtest` | Typed to `MeanReversionStrategy`; strategy receives `close_history` only; supports `warmup_prices` | B (generalize) |
| `multi_asset.run_multi_asset_backtest` | Equity curve has `position_<SYM>` and `close_<SYM>` columns | A (holdings for stress/VaR) |
| `multi_asset.exposure_report` | Per-symbol marked exposure and weights | A |
| `performance.performance_report` | Returns metrics dict | A (extend report, not replace) |
| `research.evaluate_walk_forward` | Hard-codes `MeanReversionStrategy(**parameters)` | B (generalize with a factory) |
| `research.append_experiment_record` | JSON Lines ledger | A, B |
| `cli.py` | `backtest`, `multi-backtest`, `download` | A, B |
| `dashboard.py` | Streamlit tabs: Equity, Risk, Trades, Metrics, Allocation | A, B |

Environment note: the project venv is Python 3.9. scikit-learn 1.7+ requires Python 3.10, so pin `scikit-learn>=1.3,<1.7` or upgrade the venv to Python 3.10+ before Track B.

---

## 3. Track A — Risk Engine

### 3.1 Definitions and Conventions

Work in **losses**: $L_t = -r_t$, where $r_t$ is the simple portfolio return over one bar. Confidence level $\alpha$ (e.g. 0.95, 0.99). Results are reported as **positive fractions of equity** and optionally in currency (fraction × latest equity).

- **VaR:** $\text{VaR}_\alpha = \inf\{\ell : P(L \le \ell) \ge \alpha\}$ — the $\alpha$-quantile of the loss distribution.
- **Expected Shortfall:** $\text{ES}_\alpha = E[L \mid L \ge \text{VaR}_\alpha]$ — the average loss in the tail beyond VaR.
- **Horizon scaling:** for an $h$-day horizon, the parametric method uses $\sqrt{h}$ scaling. Historical and Monte Carlo methods should use overlapping $h$-day compounded returns rather than $\sqrt{h}$, and the report must state which was used.

Invariant to test: $\text{ES}_\alpha \ge \text{VaR}_\alpha$ for every method.

### 3.2 Methods

**Historical simulation**

- Empirical quantile of the observed loss sample.
- Use a documented quantile rule (`numpy.quantile(..., method="higher")`) so small-sample results are deterministic and conservative.
- ES = mean of losses $\ge$ VaR.

**Parametric (Gaussian)**

$$
\text{VaR}_\alpha = -\mu + \sigma\, z_\alpha, \qquad
\text{ES}_\alpha = -\mu + \sigma\, \frac{\varphi(z_\alpha)}{1-\alpha}
$$

where $z_\alpha$ is the standard normal quantile and $\varphi$ is the standard normal density (both from `statistics.NormalDist`). Sample standard deviation with `ddof=1`.

Optional extension: **Cornish–Fisher** adjusted quantile using sample skewness $S$ and excess kurtosis $K$:

$$
z_{cf} = z + \frac{(z^2-1)S}{6} + \frac{(z^3-3z)K}{24} - \frac{(2z^3-5z)S^2}{36}
$$

**Monte Carlo**

- Portfolio level: simulate asset returns from a multivariate normal with the sample mean vector and covariance of aligned asset returns, then combine with current weights $w$: $r_p = w^\top r$.
- Alternative: **bootstrap** resampling of historical joint return rows (preserves fat tails and cross-correlation, loses autocorrelation).
- Accept `n_simulations` and `seed` (`numpy.random.default_rng(seed)`) so results are reproducible and recorded in the ledger.
- Reject non-positive-semidefinite covariance with a clear error; do not silently regularize.

### 3.3 Two Input Levels

1. **Equity-curve level** — `equity.pct_change()` from any backtest result. Works for single- and multi-asset runs. Measures realized strategy risk, including periods in cash.
2. **Holdings level** — current weights from the last row of `exposure_report(...)` combined with asset return history. Measures the risk of the portfolio held *now*. Required for Monte Carlo and stress tests.

The report must state which level was used.

### 3.4 Rolling VaR and VaR Backtesting

- `rolling_var(returns, window, confidence, method)` computes VaR at $t$ using only returns in $(t-\text{window}, t]$, then compares it with the realized loss at $t+1$ (shift by one bar to avoid look-ahead).
- **Exceedances:** count $x$ of days where $L_{t+1} > \text{VaR}_t$ over $n$ forecasts. Expected rate $p = 1-\alpha$.
- **Kupiec proportion-of-failures test:**

$$
LR_{POF} = -2 \ln\left[\frac{(1-p)^{n-x} p^{x}}{(1-\hat p)^{n-x} \hat p^{x}}\right], \quad \hat p = x/n
$$

  Under the null, $LR_{POF} \sim \chi^2_1$; p-value $= \operatorname{erfc}(\sqrt{LR/2})$ via `math.erfc`. Handle $x=0$ and $x=n$ with the limiting forms (avoid `log(0)`).
- Optional follow-up: Christoffersen independence test for exceedance clustering.

### 3.5 Stress Testing

**Historical scenarios** — replay the actual per-symbol cumulative returns over a named window against current holdings:

| Scenario | Window |
|---|---|
| Global Financial Crisis | 2008-09-15 to 2009-03-09 |
| COVID crash | 2020-02-19 to 2020-03-23 |
| 2022 rate shock | 2022-01-03 to 2022-10-12 |

- Scenario P&L $= \sum_i \text{exposure}_i \times R_i^{\text{scenario}}$.
- If a symbol has no data for the window, report it as `missing` rather than assuming zero; optionally allow a proxy symbol mapping (e.g. `QQQ -> SPY`) supplied explicitly by the user.
- Report worst intra-window drawdown as well as end-to-end return.

**Hypothetical shocks** — user-defined per-symbol shocks, e.g. `{"SPY": -0.20, "QQQ": -0.30}`, and a uniform market shock.

Scenario definitions live in a small dictionary in `risk.py` and can be overridden from the JSON config.

### 3.6 Module and API

New file `src/quant_backtester/risk.py`:

```python
@dataclass(frozen=True)
class RiskEstimate:
    method: str               # "historical" | "parametric" | "cornish_fisher" | "monte_carlo" | "bootstrap"
    confidence: float
    horizon: int
    var: float                # positive fraction of equity
    expected_shortfall: float
    observations: int

def historical_var(returns: pd.Series, confidence: float = 0.95, horizon: int = 1) -> RiskEstimate: ...
def parametric_var(returns: pd.Series, confidence: float = 0.95, horizon: int = 1,
                   cornish_fisher: bool = False) -> RiskEstimate: ...
def monte_carlo_var(asset_returns: pd.DataFrame, weights: pd.Series, confidence: float = 0.95,
                    horizon: int = 1, n_simulations: int = 10_000, seed: int | None = None,
                    bootstrap: bool = False) -> RiskEstimate: ...

def rolling_var(returns: pd.Series, window: int, confidence: float = 0.95,
                method: str = "historical") -> pd.DataFrame: ...   # columns: var, realized_loss, exceedance
def kupiec_test(exceedances: pd.Series, confidence: float) -> dict[str, float]: ...

def stress_test(exposures: pd.Series, prices: dict[str, pd.DataFrame],
                scenarios: Mapping[str, tuple[str, str]] | None = None) -> pd.DataFrame: ...
def shock_test(exposures: pd.Series, shocks: Mapping[str, float]) -> pd.DataFrame: ...

def risk_report(equity_curve: pd.DataFrame, confidence_levels: Sequence[float] = (0.95, 0.99),
                horizon: int = 1) -> dict[str, object]: ...
```

Validation at the boundary: confidence in $(0, 1)$, horizon $\ge 1$, at least a minimum number of finite observations (e.g. 30 for parametric, `ceil(1 / (1 - confidence))` for historical), weights aligned to asset columns.

### 3.7 Integration

- **CLI:** add `--risk` to `backtest` and `multi-backtest`, plus `--confidence` (repeatable) and `--risk-window`. Writes `risk.json` (point estimates, Kupiec result) and `rolling_var.csv`; `multi-backtest` also writes `stress.csv`. Config file may include:

  ```json
  "risk": {
    "confidence_levels": [0.95, 0.99],
    "horizon": 1,
    "rolling_window": 250,
    "monte_carlo": {"n_simulations": 10000, "seed": 7},
    "shocks": {"SPY": -0.2, "QQQ": -0.3}
  }
  ```

- **Ledger:** add a `risk` block to experiment records.
- **Dashboard:** extend the Risk tab with a VaR/ES table by method and confidence, a rolling VaR vs realized loss chart with exceedances highlighted, the Kupiec result, and (multi-asset) a stress-scenario table.

### 3.8 Tests (`tests/test_risk.py`)

- Historical VaR/ES on losses `[1..100]%`: VaR$_{0.95}$ and ES$_{0.95}$ equal hand-computed values.
- Parametric VaR with $\mu=0, \sigma=0.01, \alpha=0.99$ equals $0.01 \times 2.3263$ (tolerance $10^{-4}$).
- ES $\ge$ VaR for every method on random data.
- Monte Carlo with fixed seed is deterministic; with large `n_simulations` it converges to parametric within tolerance for Gaussian inputs.
- Rolling VaR at $t$ does not change when returns after $t$ are modified (look-ahead test).
- Kupiec: $x = np$ gives $LR \approx 0$ and p-value $\approx 1$; $x=0$ and $x=n$ do not raise.
- Stress test: two holdings with known scenario returns produce the hand-computed P&L; missing symbol reported as missing.
- Invalid confidence/horizon/insufficient data raise `ValueError`.
- CLI: `--risk` writes `risk.json`; existing CLI tests unchanged.

### 3.9 Track A Milestones

| # | Deliverable | Done when |
|---|---|---|
| A1 | Historical + parametric VaR/ES, `RiskEstimate` | Unit tests pass with hand-computed values |
| A2 | Rolling VaR + Kupiec backtest | Look-ahead test passes; exceedance rate reported |
| A3 | Monte Carlo + bootstrap at holdings level | Seeded determinism and convergence tests pass |
| A4 | Historical + hypothetical stress tests | Known-value P&L tests pass |
| A5 | CLI, ledger, dashboard integration | End-to-end CLI test writes all risk outputs; dashboard renders |
| A6 | README section + one real experiment (e.g. SPY/QQQ) recorded in ledger | Results and model limitations documented |

---

## 4. Track B — ML Signals

### 4.1 Prerequisite Refactor: Strategy Interface

Currently `run_backtest`, `run_multi_asset_backtest`, and `evaluate_walk_forward` are typed to `MeanReversionStrategy`. Introduce a protocol in `strategy.py`:

```python
class Strategy(Protocol):
    def generate_signal(self, market_event: MarketEvent, close_prices: pd.Series,
                        position: int = 0) -> SignalEvent | None: ...
```

- Change the engine type hints to `Strategy`. `MeanReversionStrategy` already satisfies it, so behavior is unchanged.
- ML features need full OHLCV, not just closes. Rather than changing the engine call signature, the ML strategy keeps its own bar buffer built from each `MarketEvent` (which already carries open/high/low/close/volume).
- Warm-up: add an optional hook `prime(history: pd.DataFrame) -> None`. `run_backtest` calls it with `warmup_prices` when the strategy defines it. Strategies without the hook are unaffected.

Walk-forward generalization in `research.py`:

```python
StrategyFactory = Callable[[Mapping[str, Any], pd.DataFrame], Strategy]

def evaluate_walk_forward(..., strategy_factory: StrategyFactory | None = None) -> WalkForwardResult: ...
```

- Default factory: `lambda params, training: MeanReversionStrategy(**params)` — preserves current results exactly.
- ML factory: trains a model on `training` using `params` (hyperparameters), returns a fitted `MLSignalStrategy`.
- **Important:** training-window parameter selection for ML must not reuse the same window for both fitting and scoring. Inside each fold, split the training window chronologically into fit and validation segments (with purge), select on validation, then refit on the full training window before the test window.

Tests: all existing tests pass unchanged; a new test confirms the default factory produces identical folds and sensitivity to the current implementation.

### 4.2 Features (`src/quant_backtester/features.py`)

All features at bar $t$ use only bars $\le t$:

| Feature | Definition |
|---|---|
| `ret_1`, `ret_5`, `ret_20` | Log return over 1, 5, 20 bars |
| `zscore_20` | $(C_t - \text{SMA}_{20}) / \text{SD}_{20}$ (same as the mean-reversion signal) |
| `vol_20` | Rolling std of daily log returns |
| `vol_ratio` | `vol_5 / vol_60` |
| `range_hl` | $(H_t - L_t) / C_t$ |
| `gap` | $O_t / C_{t-1} - 1$ |
| `volume_z_20` | z-score of log volume |
| `rsi_14` | Relative strength index |
| `dist_sma_50`, `dist_sma_200` | $C_t / \text{SMA} - 1$ |

```python
def build_features(prices: pd.DataFrame) -> pd.DataFrame: ...          # index aligned to prices; warm-up rows NaN
def build_labels(prices: pd.DataFrame, horizon: int = 5,
                 threshold: float = 0.0) -> pd.Series: ...              # 1 if C_{t+h}/C_t - 1 > threshold else 0; last h rows NaN
FEATURE_COLUMNS: tuple[str, ...]
```

Rules:

- Drop rows where any feature or the label is NaN before fitting; never forward-fill labels.
- **Purging:** because the label at $t$ uses prices through $t+h$, drop the last $h$ rows of every fit window before the next segment begins.
- **Embargo (optional):** skip an extra $e$ bars after each purge.
- Scaling (standardization) is fitted on the fit window only, inside a scikit-learn `Pipeline`.

Label alignment note: a signal generated at the close of $t$ executes at the open of $t+1$. An alternative, more execution-aligned label is $O_{t+1+h} / O_{t+1} - 1$. Implement both behind a `label_price` option and document which one results use.

### 4.3 Models (`src/quant_backtester/models.py`)

```python
@dataclass
class ModelSpec:
    kind: str = "logistic"          # "logistic" | "hist_gradient_boosting"
    params: dict[str, Any] = field(default_factory=dict)
    random_state: int = 0

def make_model(spec: ModelSpec) -> sklearn.pipeline.Pipeline: ...
def fit_model(spec: ModelSpec, prices: pd.DataFrame, horizon: int, purge: int) -> FittedModel: ...
```

- Baseline: `StandardScaler` + `LogisticRegression` (interpretable coefficients).
- Nonlinear: `HistGradientBoostingClassifier` (in scikit-learn; avoids an XGBoost dependency). XGBoost/LightGBM can be optional later.
- `FittedModel` stores the pipeline, feature columns, horizon, training period, and class balance for the ledger.
- Deterministic: fixed `random_state`, recorded in the ledger.
- No model pickles are loaded from untrusted sources; if persistence is added later, use `joblib` only for files the user created, and document that pickles execute code on load.

Dependency: add `ml = ["scikit-learn>=1.3,<1.7"]` to `[project.optional-dependencies]` (Python 3.9 compatibility). Import scikit-learn lazily inside `models.py` so the core package and existing tests do not require it.

### 4.4 ML Strategy (`src/quant_backtester/ml_strategy.py`)

```python
@dataclass
class MLSignalStrategy:
    model: FittedModel
    entry_probability: float = 0.55
    exit_probability: float = 0.50
    min_history: int = 200          # bars required before features are valid

    def prime(self, history: pd.DataFrame) -> None: ...
    def generate_signal(self, market_event, close_prices, position=0) -> SignalEvent | None: ...
```

- Appends the incoming bar to its internal buffer, computes the latest feature row with `build_features`, and calls `predict_proba`.
- `LONG` when flat and $p \ge$ `entry_probability`; `EXIT` when long and $p <$ `exit_probability`. `strength` = probability.
- Returns `None` during warm-up or when any feature is NaN.
- Performance: recomputing all features per bar is $O(n^2)$ over a backtest. Start simple, and optimize only if needed (bounded buffer of `min_history` + largest window, or precomputed features keyed by timestamp — the latter must still be generated with only past data and verified by the look-ahead test).

### 4.5 Evaluation

For each walk-forward test window, record:

- **Classification:** ROC AUC, accuracy, precision at the entry threshold, base rate, Brier score.
- **Trading (after costs):** the full `performance_report` plus `risk_report` from Track A.
- **Baselines on the same windows:** buy-and-hold, `MeanReversionStrategy` with walk-forward-selected parameters, and a **shuffled-label control** model (should show no edge; if it does, there is leakage).
- **Turnover and cost drag:** number of fills and total commission + slippage + impact.

Report both successful and unsuccessful experiments in the ledger and README, as required by Phase 8.

### 4.6 Integration

- **CLI:** new command:

  ```bash
  python -m quant_backtester.cli ml-walk-forward data/SPY.csv --symbol SPY \
    --model logistic --horizon 5 --train-size 756 --test-size 126 \
    --entry-probability 0.55 --output-directory reports/ml_spy \
    --experiment-ledger reports/experiments.jsonl
  ```

  Writes `folds.csv`, `predictions.csv` (timestamp, probability, label), `sensitivity.csv`, `summary.csv`, and `metrics.json`.
- **Dashboard:** strategy selector (`Mean reversion` / `ML signal`). For ML: model, horizon, and thresholds; shows out-of-sample equity vs baselines, probability-over-time chart, per-fold AUC, and logistic coefficients or permutation importance.
- **Multi-asset (later):** one model per symbol first; a pooled cross-sectional model is a follow-up.

### 4.7 Leakage Checklist (each is a test in `tests/test_ml.py`)

1. **Future-perturbation:** changing prices after $t$ leaves features at $\le t$ and the signal at $t$ unchanged.
2. **Label horizon:** the last $h$ labels are NaN; label at $t$ matches a hand calculation.
3. **Purge:** fit windows exclude the last $h$ rows before any validation/test segment.
4. **Scaler fit scope:** the pipeline's scaler mean equals the fit-window mean, not the full-sample mean.
5. **Shuffled-label control:** on synthetic random-walk data, out-of-sample AUC stays near 0.5.
6. **Planted signal:** on synthetic data with a known predictive feature, the model recovers AUC well above 0.5 (proves the pipeline can learn).
7. **Determinism:** same seed and data give identical predictions and ledger metrics.
8. **Engine parity:** `MLSignalStrategy` fills occur at the next bar's open with the same costs as `MeanReversionStrategy`.

### 4.8 Track B Milestones

| # | Deliverable | Done when |
|---|---|---|
| B0 | `Strategy` protocol, `prime` hook, `strategy_factory` in walk-forward | All existing tests pass; default-factory parity test passes |
| B1 | `features.py` + labels + purge helpers | Leakage tests 1–4 pass |
| B2 | `models.py` with logistic + gradient boosting, `ml` extra | Tests 5–7 pass; core tests pass without scikit-learn installed |
| B3 | `MLSignalStrategy` through `run_backtest` | Engine-parity test passes |
| B4 | ML walk-forward with nested fit/validation selection | Fold outputs and ledger records verified on fixture data |
| B5 | CLI `ml-walk-forward` + dashboard view | End-to-end CLI test; dashboard renders |
| B6 | Real experiment on SPY (and one more symbol) vs baselines | Out-of-sample results, including failures, documented in README |

---

## 5. File Changes Summary

| File | Change |
|---|---|
| `src/quant_backtester/risk.py` | New — Track A |
| `src/quant_backtester/features.py` | New — Track B |
| `src/quant_backtester/models.py` | New — Track B |
| `src/quant_backtester/ml_strategy.py` | New — Track B |
| `src/quant_backtester/strategy.py` | Add `Strategy` protocol |
| `src/quant_backtester/backtest.py` | Type to `Strategy`; call optional `prime` with warm-up data |
| `src/quant_backtester/multi_asset.py` | Type to `Strategy` |
| `src/quant_backtester/research.py` | `strategy_factory`; nested validation for ML; ML metrics in ledger |
| `src/quant_backtester/cli.py` | `--risk` options; `ml-walk-forward` command |
| `src/quant_backtester/dashboard.py` | Risk tab extensions; ML strategy view |
| `pyproject.toml` | `ml` optional extra |
| `tests/test_risk.py`, `tests/test_features.py`, `tests/test_ml.py` | New tests |
| `README.md` | Usage docs and recorded experiment results |

---

## 6. Risks and Open Questions

- **Small samples:** 99% VaR on a few years of daily data has few tail observations; report observation counts and prefer 95% for backtesting.
- **Normality:** parametric VaR understates fat-tailed losses; compare with historical and Cornish–Fisher and say so in the report.
- **Stationarity:** covariance and model relationships drift; walk-forward refitting mitigates but does not remove this.
- **Survivorship bias:** Yahoo Finance symbols are current survivors; results on today's index members are optimistic.
- **Adjusted prices:** confirm whether downloads are split/dividend adjusted and use the same convention for features, labels, and returns.
- **Overfitting via many experiments:** the ledger records every run; count how many configurations were tried before reporting the best one.
- **Runtime:** per-bar feature recomputation and per-fold refits can be slow on long histories; profile before optimizing.
- **Decision:** whether to upgrade the venv to Python 3.10+ (unlocks current scikit-learn/XGBoost) or keep 3.9 with pinned versions.
