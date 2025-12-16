import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from services import modeles_services
from services import metrics_services
from services import cross_validation_services

def aggregated_error_distribution(y_true, y_pred, group_size, n_draws=1000, seed=42, eps=1e-9):
    """
    Draw random groups and compute relative sum error distribution.
    Works with numpy arrays only (pandas Series are converted).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    rng = np.random.default_rng(seed)
    n = len(y_true)
    idx = np.arange(n)

    rse = np.zeros(n_draws, dtype=float)

    for k in range(n_draws):
        sample_idx = rng.choice(idx, size=group_size, replace=False)
        S = y_true[sample_idx].sum()
        Shat = y_pred[sample_idx].sum()
        rse[k] = abs(Shat - S) / (S + eps)

    return rse


def cv_aggregated_protocol(
    X: pd.DataFrame,
    y: pd.Series,
    baseline_B_features: list,
    group_sizes=(30, 60, 120),
    n_draws=1000,
    n_splits=5,
    random_state=42
):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    all_records = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]  # keep aligned indices

        # ---- Baseline A (mean) ----
        A = modeles_services.BaselineMeanPredictor().fit(y_train)
        y_pred_A = A.predict(len(y_test))  # numpy array

        # ---- Baseline B (NB) ----
        B = modeles_services.BaselineNegativeBinomial(feature_cols=baseline_B_features)
        B.fit(X_train, y_train)  # aligned indices now
        y_pred_B = B.predict(X_test)  # numpy array

        # Convert y_test to numpy for random sampling
        y_test_np = y_test.to_numpy()

        for m in group_sizes:
            if m >= len(y_test_np):
                continue

            rse_A = aggregated_error_distribution(
                y_true=y_test_np,
                y_pred=y_pred_A,
                group_size=m,
                n_draws=n_draws,
                seed=fold * 1000 + m
            )
            rse_B = aggregated_error_distribution(
                y_true=y_test_np,
                y_pred=y_pred_B,
                group_size=m,
                n_draws=n_draws,
                seed=fold * 1000 + m + 1
            )

            all_records.append({
                "fold": fold, "model": "A_mean", "group_size": m,
                "mean_rse": float(np.mean(rse_A)),
                "median_rse": float(np.median(rse_A)),
                "p90_rse": float(np.quantile(rse_A, 0.90)),
                "p95_rse": float(np.quantile(rse_A, 0.95)),
            })
            all_records.append({
                "fold": fold, "model": "B_neg_bin", "group_size": m,
                "mean_rse": float(np.mean(rse_B)),
                "median_rse": float(np.median(rse_B)),
                "p90_rse": float(np.quantile(rse_B, 0.90)),
                "p95_rse": float(np.quantile(rse_B, 0.95)),
            })

    return pd.DataFrame(all_records)