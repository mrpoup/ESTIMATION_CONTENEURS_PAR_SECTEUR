from __future__ import annotations

import numpy as np
import pandas as pd


class BaselineMajorityClass:
    """Always predicts the most frequent class seen during training."""

    name = "baseline_majority"

    def __init__(self):
        self.majority_class_: int | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaselineMajorityClass":
        values, counts = np.unique(y.to_numpy(), return_counts=True)
        self.majority_class_ = int(values[np.argmax(counts)])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.majority_class_ is None:
            raise RuntimeError("Model must be fitted before calling predict().")
        return np.full(len(X), self.majority_class_, dtype=int)
    
class BaselineStratifiedRandom:
    """
    Randomly predicts a class according to the empirical class distribution
    observed during training.
    """

    name = "baseline_stratified_random"

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.classes_: np.ndarray | None = None
        self.probas_: np.ndarray | None = None
        self._rng = np.random.default_rng(random_state)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaselineStratifiedRandom":
        values, counts = np.unique(y.to_numpy(), return_counts=True)
        self.classes_ = values.astype(int)
        self.probas_ = counts / counts.sum()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.classes_ is None or self.probas_ is None:
            raise RuntimeError("Model must be fitted before calling predict().")
        return self._rng.choice(
            self.classes_,
            size=len(X),
            replace=True,
            p=self.probas_,
        )

