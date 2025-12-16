import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from services import modeles_services
from services import metrics_services


def cross_validate_baselines(
    X: pd.DataFrame,
    y: pd.Series,
    baseline_B_features: list,
    n_splits: int = 5,
    random_state: int = 42,
):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    records = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # ---------- Baseline A ----------
        A = modeles_services.BaselineMeanPredictor().fit(y_train)
        y_pred_A = A.predict(len(y_test))

        metrics_A =metrics_services.CountRegressionMetrics.compute_all(y_test, y_pred_A)
        metrics_A["baseline"] = "A_mean"
        metrics_A["fold"] = fold

        # ---------- Baseline B ----------
        B = modeles_services.BaselineNegativeBinomial(feature_cols=baseline_B_features)
        B.fit(X_train, y_train)
        y_pred_B = B.predict(X_test)

        metrics_B = metrics_services.CountRegressionMetrics.compute_all(y_test, y_pred_B)
        metrics_B["baseline"] = "B_neg_bin"
        metrics_B["fold"] = fold

        records.extend([metrics_A, metrics_B])

    return pd.DataFrame(records)