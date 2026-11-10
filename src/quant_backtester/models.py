"""Model specifications, leakage-safe fitting, and classification metrics.

scikit-learn is imported lazily, so the core backtester runs without it.
Install it with ``pip install -e '.[ml]'``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from quant_backtester.features import FEATURE_COLUMNS, build_training_set


MODEL_KINDS = ("logistic", "hist_gradient_boosting")
DEFAULT_PARAMETER_SETS: dict[str, tuple[dict[str, Any], ...]] = {
    "logistic": ({"C": 0.01}, {"C": 0.1}, {"C": 1.0}),
    "hist_gradient_boosting": (
        {"max_depth": 2, "learning_rate": 0.05, "max_iter": 100},
        {"max_depth": 3, "learning_rate": 0.05, "max_iter": 200},
    ),
}
MIN_TRAINING_ROWS = 50


@dataclass
class ModelSpec:
    """Classifier family, hyperparameters, and the seed recorded with every fit."""

    kind: str = "logistic"
    params: dict[str, Any] = field(default_factory=dict)
    random_state: int = 0

    def __post_init__(self) -> None:
        if self.kind not in MODEL_KINDS:
            raise ValueError(f"Model kind must be one of {MODEL_KINDS}")
        if "random_state" in self.params:
            raise ValueError("Set random_state on ModelSpec rather than in params")


def make_model(spec: ModelSpec) -> Any:
    """Build an unfitted scikit-learn pipeline: standardization, then the classifier."""
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as error:
        raise ImportError(
            "ML models require scikit-learn; install it with: pip install -e '.[ml]'"
        ) from error

    if spec.kind == "logistic":
        estimator = LogisticRegression(
            **{"max_iter": 1000, **spec.params, "random_state": spec.random_state}
        )
    else:
        estimator = HistGradientBoostingClassifier(
            **{**spec.params, "random_state": spec.random_state}
        )
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])


@dataclass(frozen=True)
class FittedModel:
    """A fitted pipeline plus the labeling setup and training period it came from."""

    pipeline: Any
    spec: ModelSpec
    horizon: int
    threshold: float
    label_price: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    observations: int
    positive_rate: float
    shuffled_labels: bool = False
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Probability that the label is 1 for each feature row."""
        return self.pipeline.predict_proba(features.loc[:, list(self.feature_columns)])[:, 1]

    def coefficients(self) -> pd.Series | None:
        """Logistic coefficients on standardized features, or None for other models."""
        estimator = self.pipeline.named_steps["model"]
        if not hasattr(estimator, "coef_"):
            return None
        return pd.Series(estimator.coef_[0], index=list(self.feature_columns), name="coefficient")

    def summary(self) -> dict[str, Any]:
        """JSON-friendly description for experiment ledgers."""
        return {
            "kind": self.spec.kind,
            "params": dict(self.spec.params),
            "random_state": self.spec.random_state,
            "horizon": self.horizon,
            "threshold": self.threshold,
            "label_price": self.label_price,
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "observations": self.observations,
            "positive_rate": self.positive_rate,
            "shuffled_labels": self.shuffled_labels,
        }


def fit_model(
    spec: ModelSpec,
    prices: pd.DataFrame,
    horizon: int = 5,
    threshold: float = 0.0,
    label_price: str = "close",
    shuffle_labels: bool = False,
) -> FittedModel:
    """Fit on one window; labels needing prices after the window are purged first.

    ``shuffle_labels`` permutes the training labels (seeded by the spec) to build
    a control model that should show no out-of-sample edge.
    """
    features, labels = build_training_set(prices, horizon, threshold, label_price)
    if len(labels) < MIN_TRAINING_ROWS:
        raise ValueError(
            f"At least {MIN_TRAINING_ROWS} labeled rows are required to fit a model; "
            f"found {len(labels)}"
        )
    if labels.nunique() < 2:
        raise ValueError("Training labels must contain both classes")
    target = labels
    if shuffle_labels:
        generator = np.random.default_rng(spec.random_state)
        target = pd.Series(generator.permutation(labels.to_numpy()), index=labels.index)

    pipeline = make_model(spec)
    pipeline.fit(features, target)
    return FittedModel(
        pipeline=pipeline,
        spec=spec,
        horizon=horizon,
        threshold=threshold,
        label_price=label_price,
        train_start=features.index[0],
        train_end=features.index[-1],
        observations=len(labels),
        positive_rate=float(labels.mean()),
        shuffled_labels=shuffle_labels,
    )


def roc_auc(probabilities: pd.Series, labels: pd.Series) -> float | None:
    """Rank-based ROC AUC with ties counted as one half; None without both classes."""
    positives = labels == 1
    positive_count = int(positives.sum())
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        return None
    ranks = probabilities.rank()
    return float(
        (ranks[positives].sum() - positive_count * (positive_count + 1) / 2)
        / (positive_count * negative_count)
    )


def classification_metrics(
    probabilities: pd.Series,
    labels: pd.Series,
    entry_probability: float = 0.5,
) -> dict[str, float | int | None]:
    """Out-of-sample quality of probabilities on rows where the label is known."""
    frame = pd.concat({"probability": probabilities, "label": labels}, axis=1).dropna()
    probability = frame["probability"].astype(float)
    label = frame["label"].astype(float)
    entries = probability >= entry_probability
    return {
        "observations": int(len(frame)),
        "base_rate": float(label.mean()) if len(frame) else None,
        "roc_auc": roc_auc(probability, label),
        "accuracy": float(((probability >= 0.5) == (label == 1)).mean()) if len(frame) else None,
        "precision_at_entry": float(label[entries].mean()) if entries.any() else None,
        "entry_rate": float(entries.mean()) if len(frame) else None,
        "brier_score": float(((probability - label) ** 2).mean()) if len(frame) else None,
    }
