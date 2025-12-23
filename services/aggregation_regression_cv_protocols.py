from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from services import aggregation_eval_services


def cv_aggregated_protocol(
    X_by_model: Dict[str, pd.DataFrame],
    models: Dict[str, Any],
    y: pd.Series,
    pks: List[int],
    p_n_draws: int,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Cross-validated evaluation of aggregated quantity prediction (sum over groups).

    Requirements:
      - models[name] must expose fit(X_train, y_train) and predict(X_test)
      - X_by_model[name] must be aligned with y (same row order / same index)
    """
    if set(models.keys()) != set(X_by_model.keys()):
        missing_in_X = set(models.keys()) - set(X_by_model.keys())
        missing_in_M = set(X_by_model.keys()) - set(models.keys())
        raise ValueError(
            f"Model/X mismatch. Missing X for: {missing_in_X}. Missing model for: {missing_in_M}."
        )

    # Use any X as reference for CV split
    ref_name = next(iter(X_by_model.keys()))
    X_ref = X_by_model[ref_name]

    # Defensive alignment check
    if len(X_ref) != len(y):
        raise ValueError("X and y must have the same number of rows.")
    for name, Xn in X_by_model.items():
        if len(Xn) != len(y):
            raise ValueError(f"X_by_model['{name}'] length != y length.")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    all_records: list[dict] = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_ref), start=1):
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        y_test_np = y_test.to_numpy(dtype=float)

        for model_name, model in models.items():
            X_train = X_by_model[model_name].iloc[train_idx]
            X_test = X_by_model[model_name].iloc[test_idx]

            # Fit / predict
            model.fit(X_train, y_train)
            y_pred = np.asarray(model.predict(X_test), dtype=float)

            # (optional but recommended for count regression)
            y_pred = np.clip(y_pred, 0.0, None)

            # Aggregate eval for this fold and this model
            fold_df = aggregation_eval_services.evaluate_aggregation_for_fold(
                y_true_fold=y_test_np,
                y_pred_fold=y_pred,
                pks=pks,
                p_n_draws=p_n_draws,
                seed_base=fold * 1000 + (abs(hash(model_name)) % 100),
            )
            fold_df["fold"] = fold
            fold_df["model"] = model_name

            all_records.append(fold_df)

    return pd.concat(all_records, ignore_index=True)


##USAGE
# models = {
#     "A_mean": modeles_services_regression.BaselineMeanPredictor(),
#     "B_neg_bin": modeles_services_regression.BaselineNegativeBinomial(feature_cols=baseline_B_features),
#     "C_lgbm_poisson": modeles_services_regression.ModelCPoissonLGBM(params=lgbm_params, random_state=42),
# }

# X_by_model = {
#     "A_mean": X_C,          # peu importe, A ignore X
#     "B_neg_bin": X_B,
#     "C_lgbm_poisson": X_C,
# }

# df = cv_aggregated_protocol(
#     X_by_model=X_by_model,
#     models=models,
#     y=y,
#     pks=pks,
#     p_n_draws=p_n_draws,
#     n_splits=5,
#     random_state=42,
# )

# summary = df.groupby(["model", "group_size"]).agg(["mean", "std"])
