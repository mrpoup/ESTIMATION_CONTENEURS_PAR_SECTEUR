from __future__ import annotations

import numpy as np
import pandas as pd

from typing import Protocol


class RegressorLike(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RegressorLike":
        ...

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ...


class FittedBinner:
    """
    Minimal fitted binner used to transform numeric predictions into classes
    using pre-computed integer thresholds.
    """

    def __init__(self, thresholds_finite: list[int], right_closed: bool = True):
        self.thresholds = np.asarray(thresholds_finite, dtype=int)
        self.right_closed = right_closed

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        values = np.clip(values, 0, None)  # no negative counts
        return np.digitize(values, bins=self.thresholds, right=self.right_closed)

class RegressionToClassBaseline:
    """
    Baseline that:
    1) fits a regression model on the continuous target
    2) learns class thresholds from y_train (continuous)
    3) maps regression predictions to classes
    """

    name = "baseline_regression_to_class"

    def __init__(
        self,
        regressor: RegressorLike,
        binning_service,
        binning_spec,
        y_continuous: pd.Series,
    ):
        self.regressor = regressor
        self.binning_service = binning_service
        self.binning_spec = binning_spec

        # Store the full continuous target (aligned by index later per fold)
        if not isinstance(y_continuous, pd.Series):
            raise TypeError("y_continuous must be a pandas Series.")
        self.y_continuous = y_continuous

        self._binner: FittedBinner | None = None

    def fit(self, X: pd.DataFrame, y_class: pd.Series) -> "RegressionToClassBaseline":
        # Align y_continuous to X (fold-safe)
        y_cont = self.y_continuous.reindex(X.index)
        if y_cont.isna().any():
            raise ValueError("y_continuous could not be aligned with X (missing indices).")

        # 1) Fit regression on continuous target
        self.regressor.fit(X, y_cont)

        # 2) Learn binning ONLY on the train fold continuous target
        artifacts = self.binning_service.build_classes(
            y=y_cont,
            spec=self.binning_spec,
        )

        # # Preferred: continuous edges with +inf (as you requested)
        # edges = artifacts.log.get("edges_with_inf") or artifacts.log.get("integer_edges_with_inf")
        # if edges is None:
        #     # Fallback if you stored integer ranges instead
        #     ranges = artifacts.log.get("integer_class_ranges")
        #     if ranges is None:
        #         raise RuntimeError("Binning log must contain edges_with_inf or integer_class_ranges.")
        #     upper_bounds = [hi for (_, hi) in ranges[:-1]]
        # else:
        #     # edges include +inf, keep finite ones only
        #     upper_bounds = [int(e) for e in edges if np.isfinite(e)]

        # self._binner = FittedBinner(
        #     thresholds_finite=upper_bounds,
        #     right_closed=self.binning_spec.right_closed,
        # )

        thresholds_finite = artifacts.log.get("thresholds_finite")
        if thresholds_finite is None:
            raise RuntimeError("Binning log must contain 'thresholds_finite' (normalized).")

        self._binner = FittedBinner(
            thresholds_finite=[int(t) for t in thresholds_finite],
            right_closed=self.binning_spec.right_closed,
        )

     
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._binner is None:
            raise RuntimeError("Model must be fitted before calling predict().")

        y_pred_cont = self.regressor.predict(X)
        return self._binner.transform(y_pred_cont)

