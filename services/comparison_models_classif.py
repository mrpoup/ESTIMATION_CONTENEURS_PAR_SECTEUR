from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Protocol, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score


class ClassifierLike(Protocol):
    """Minimal interface expected by the comparison service."""
    name: str

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ClassifierLike":
        ...

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ...


def mean_absolute_class_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Ordinal-friendly metric: mean absolute difference between class indices."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return float(np.mean(np.abs(y_pred - y_true)))


def severe_error_rate(y_true: np.ndarray, y_pred: np.ndarray, threshold: int = 2) -> float:
    """Share of samples where the class error is >= threshold."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return float(np.mean(np.abs(y_pred - y_true) >= threshold))


@dataclass
class ComparisonSpec:
    n_splits: int = 5
    shuffle: bool = True
    random_state: int = 42
    severe_threshold: int = 2
    return_oof: bool = True


@dataclass
class ComparisonResult:
    summary: pd.DataFrame
    fold_details: pd.DataFrame
    oof_predictions: Optional[pd.DataFrame] = None


class ClassModelComparisonService:
    """
    Compare multiple classification models with stratified CV using ordinal-aware metrics.
    """

    def __init__(self, spec: ComparisonSpec = ComparisonSpec()):
        self.spec = spec

    def compare(
        self,
        X: pd.DataFrame,
        y_class: pd.Series,
        models: List[ClassifierLike],
        sample_weight: Optional[np.ndarray] = None,
    ) -> ComparisonResult:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        if not isinstance(y_class, pd.Series):
            raise TypeError("y_class must be a pandas Series.")

        # Ensure aligned index
        if not X.index.equals(y_class.index):
            y_class = y_class.reindex(X.index)
            if y_class.isna().any():
                raise ValueError("y_class could not be aligned with X (missing indices).")

        y_np = y_class.to_numpy().astype(int)

        skf = StratifiedKFold(
            n_splits=self.spec.n_splits,
            shuffle=self.spec.shuffle,
            random_state=self.spec.random_state,
        )

        rows: List[Dict[str, Any]] = []
        oof_frames: List[pd.DataFrame] = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_np), start=1):
            X_train = X.iloc[train_idx]
            y_train = y_class.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_test = y_class.iloc[test_idx]
            y_test_np = y_test.to_numpy().astype(int)

            for model in models:
                m = model  # readability
                m.fit(X_train, y_train)
                y_pred = np.asarray(m.predict(X_test), dtype=int)

                metrics = {
                    "fold": fold,
                    "model": getattr(m, "name", m.__class__.__name__),
                    "accuracy": accuracy_score(y_test_np, y_pred),
                    "balanced_accuracy": balanced_accuracy_score(y_test_np, y_pred),
                    "mace": mean_absolute_class_error(y_test_np, y_pred),
                    "severe_error_rate": severe_error_rate(
                        y_test_np, y_pred, threshold=self.spec.severe_threshold
                    ),
                }
                rows.append(metrics)

                if self.spec.return_oof:
                    oof_frames.append(pd.DataFrame({
                        "index": X_test.index,
                        "fold": fold,
                        "model": getattr(m, "name", m.__class__.__name__),
                        "y_true": y_test_np,
                        "y_pred": y_pred,
                    }))

        fold_details = pd.DataFrame(rows)

        # Summary: mean/std by model
        metric_cols = ["accuracy", "balanced_accuracy", "mace", "severe_error_rate"]
        summary = (
            fold_details
            .groupby("model")[metric_cols]
            .agg(["mean", "std"])
            .sort_values(("balanced_accuracy", "mean"), ascending=False)
        )

        oof_predictions = None
        if self.spec.return_oof and len(oof_frames) > 0:
            oof_predictions = pd.concat(oof_frames, axis=0, ignore_index=True)

        return ComparisonResult(
            summary=summary,
            fold_details=fold_details,
            oof_predictions=oof_predictions,
        )
