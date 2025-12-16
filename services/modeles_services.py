import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.families import NegativeBinomial

class BaselineMeanPredictor:
    """Baseline A: predicts the training mean."""

    def fit(self, y: pd.Series):
        self.mean_ = y.mean()
        return self

    def predict(self, n_samples: int) -> np.ndarray:
        return np.full(shape=n_samples, fill_value=self.mean_, dtype=float)
    
class BaselineNegativeBinomial:
    """Baseline B: simple Negative Binomial regression."""

    def __init__(self, feature_cols):
        self.feature_cols = feature_cols
        self.model_ = None
        self.result_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X_ = sm.add_constant(X[self.feature_cols], has_constant="add")
        self.model_ = sm.GLM(y, X_, family=NegativeBinomial())
        self.result_ = self.model_.fit()
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_ = sm.add_constant(X[self.feature_cols], has_constant="add")
        return np.asarray(self.result_.predict(X_))

    
