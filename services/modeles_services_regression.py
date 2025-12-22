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
    


class ModelCPoissonLGBM:
    """Model C: LightGBM Poisson regressor (count intensity)."""

    def __init__(self, params: dict | None = None, random_state: int = 42):
        import lightgbm as lgb

        default_params = dict(
            objective="poisson",
            learning_rate=0.05,
            n_estimators=800,
            num_leaves=31,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=0.0,
            random_state=random_state,
            n_jobs=-1,
        )

        if params is not None:
            default_params.update(params)

        self.lgb = lgb
        self.params = default_params
        self.model_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model_ = self.lgb.LGBMRegressor(**self.params)
        self.model_.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model_.predict(X))


    
