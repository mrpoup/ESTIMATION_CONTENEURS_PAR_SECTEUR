import numpy as np
import pandas as pd
from typing import Dict
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance

class CountRegressionMetrics:
    """
    Metrics for count regression models.
    All methods assume y_true and y_pred are 1D arrays or Series.
    """

    @staticmethod
    def mae(y_true, y_pred) -> float:
        """Mean Absolute Error."""
        return mean_absolute_error(y_true, y_pred)

    @staticmethod
    def poisson_deviance(y_true, y_pred, eps: float = 1e-9) -> float:
        """
        Mean Poisson deviance.
        y_pred must be strictly positive; we clip for numerical safety.
        """
        y_pred_safe = np.clip(y_pred, eps, None)
        return mean_poisson_deviance(y_true, y_pred_safe)

    @staticmethod
    def relative_sum_error(y_true, y_pred, eps: float = 1e-9) -> float:
        """
        Relative error on the sum of predictions.
        """
        true_sum = np.sum(y_true)
        pred_sum = np.sum(y_pred)

        if true_sum < eps:
            return np.nan  # undefined, but shouldn't happen in your data

        return abs(pred_sum - true_sum) / true_sum

    @staticmethod
    def compute_all(y_true, y_pred) -> Dict[str, float]:
        """
        Compute all standard metrics at once.
        """
        return {
            "mae": CountRegressionMetrics.mae(y_true, y_pred),
            "poisson_deviance": CountRegressionMetrics.poisson_deviance(y_true, y_pred),
            "relative_sum_error": CountRegressionMetrics.relative_sum_error(y_true, y_pred),
        }