import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr

from services import modeles_services_regression
from services import metrics_services_regression
from services import spatial_grouping_services
from services import aggregation_eval_services





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
        A = modeles_services_regression.BaselineMeanPredictor().fit(y_train)
        y_pred_A = A.predict(len(y_test))

        metrics_A =metrics_services_regression.CountRegressionMetrics.compute_all(y_test, y_pred_A)
        metrics_A["baseline"] = "A_mean"
        metrics_A["fold"] = fold

        # ---------- Baseline B ----------
        B = modeles_services_regression.BaselineNegativeBinomial(feature_cols=baseline_B_features)
        B.fit(X_train, y_train)
        y_pred_B = B.predict(X_test)

        metrics_B = metrics_services_regression.CountRegressionMetrics.compute_all(y_test, y_pred_B)
        metrics_B["baseline"] = "B_neg_bin"
        metrics_B["fold"] = fold

        records.extend([metrics_A, metrics_B])

    return pd.DataFrame(records)




def cv_aggregated_protocol_AB(
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
        A = modeles_services_regression.BaselineMeanPredictor().fit(y_train)
        y_pred_A = A.predict(len(y_test))  # numpy array

        # ---- Baseline B (NB) ----
        B = modeles_services_regression.BaselineNegativeBinomial(feature_cols=baseline_B_features)
        B.fit(X_train, y_train)  # aligned indices now
        y_pred_B = B.predict(X_test)  # numpy array

        # Convert y_test to numpy for random sampling
        y_test_np = y_test.to_numpy()

        for m in group_sizes:
            if m >= len(y_test_np):
                continue
            for m in group_sizes:
                if m >= len(y_test_np):
                    continue
                
                stats_A = aggregation_eval_services.summarize_aggregation_draws(aggregation_eval_services.draw_aggregated_errors(
                    y_true=y_test_np,
                    y_pred=y_pred_A,
                    group_size=m,
                    n_draws=n_draws,
                    seed=fold * 1000 + m
                    ))
                all_records.append({"fold": fold, "model": "A_mean", "group_size": m,**stats_A})


                stats_B = aggregation_eval_services.summarize_aggregation_draws(aggregation_eval_services.draw_aggregated_errors(
                    y_true=y_test_np,
                    y_pred=y_pred_B,
                    group_size=m,
                    n_draws=n_draws,
                    seed=fold * 1000 + m
                    ))
                all_records.append({"fold": fold, "model": "B_neg_bin", "group_size": m,**stats_B})

    return pd.DataFrame(all_records)



def cv_aggregated_protocol_ABC(
    X_B: pd.DataFrame,                 # features used by baseline B (can be X_C too)
    X_C: pd.DataFrame,                 # features used by model C
    y: pd.Series,
    baseline_B_features: list,
    lgbm_params: dict | None = None,
    group_sizes=(30, 60, 120),
    n_draws=1000,
    n_splits=5,
    random_state=42
):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    all_records = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_C), start=1):
        # Keep indices aligned
        Xb_train, Xb_test = X_B.iloc[train_idx], X_B.iloc[test_idx]
        Xc_train, Xc_test = X_C.iloc[train_idx], X_C.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # ---- A: mean baseline ----
        A =modeles_services_regression.BaselineMeanPredictor().fit(y_train)
        y_pred_A = A.predict(len(y_test))

        # ---- B: negative binomial baseline ----
        B = modeles_services_regression.BaselineNegativeBinomial(feature_cols=baseline_B_features)
        B.fit(Xb_train, y_train)
        y_pred_B = np.asarray(B.predict(Xb_test))

        # ---- C: LightGBM Poisson ----
        C = modeles_services_regression.ModelCPoissonLGBM(params=lgbm_params, random_state=random_state)
        C.fit(Xc_train, y_train)
        y_pred_C = np.asarray(C.predict(Xc_test))

        y_test_np = y_test.to_numpy()

        for m in group_sizes:
            if m >= len(y_test_np):
                continue

            stats_A = aggregation_eval_services.summarize_aggregation_draws(aggregation_eval_services.draw_aggregated_errors(
                y_true=y_test_np,
                y_pred=y_pred_A,
                group_size=m,
                n_draws=n_draws,
                seed=fold * 1000 + m
                ))
            stats_B = aggregation_eval_services.summarize_aggregation_draws(aggregation_eval_services.draw_aggregated_errors(
                y_true=y_test_np,
                y_pred=y_pred_B,
                group_size=m,
                n_draws=n_draws,
                seed=fold * 1000 + m
                ))
            stats_C = aggregation_eval_services.summarize_aggregation_draws(aggregation_eval_services.draw_aggregated_errors(
                y_true=y_test_np,
                y_pred=y_pred_C,
                group_size=m,
                n_draws=n_draws,
                seed=fold * 1000 + m
                ))
            
            for model_name, stat_o in [("A_mean", stats_A), ("B_neg_bin", stats_B), ("C_lgbm_poisson", stats_C)]:
                all_records.append({"fold": fold, "model": model_name, "group_size": m,**stat_o})


    return pd.DataFrame(all_records)

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr


def compute_group_metrics(y_true, y_pred, eps=1e-12):
    """
    Compute metrics between aggregated true/pred values.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    if np.std(y_true) < eps or np.std(y_pred) < eps:
        pearson = np.nan
        spearman = np.nan
    else:
        pearson = pearsonr(y_true, y_pred)[0]
        spearman = spearmanr(y_true, y_pred).correlation

    bias = np.mean(y_pred - y_true)
    rel_bias = np.mean((y_pred - y_true) / (y_true + eps))

    return {
        "pearson": pearson,
        "spearman": spearman,
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "rel_bias": rel_bias,
    }

def cv_spatial_knn_protocol_ABC(
    X_B,
    X_C,
    coords,
    y,
    baseline_B_features,
    k_groups=(30, 60, 120),
    n_splits=5,
    random_state=42,
    lgbm_params=None,
    max_groups=600,
    grouping: str = "knn",   # "knn" or "random"
):
    """
    Cross-validated aggregation evaluation for 3 models:
      A: mean baseline
      B: Negative Binomial baseline
      C: LightGBM Poisson

    grouping:
      - "knn": spatial kNN groups
      - "random": random groups of size k

    Metrics per fold & k:
      - RSE distribution (knn: per-point neighborhoods, random: per-random-group draws)
      - Metrics on aggregated SUMS
      - Metrics on aggregated MEANS (sum / k)
    """

    grouping = grouping.lower().strip()
    if grouping not in ("knn", "random"):
        raise ValueError("grouping must be 'knn' or 'random'")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    records = []

    coords = np.asarray(coords)

    # -------------------------
    # Helper functions
    # -------------------------
    def _safe_pearson(x, y):
        if len(x) < 2:
            return np.nan
        if np.allclose(np.std(x), 0) or np.allclose(np.std(y), 0):
            return np.nan
        return float(pearsonr(x, y)[0])

    def _safe_spearman(x, y):
        if len(x) < 2:
            return np.nan
        try:
            return float(spearmanr(x, y).correlation)
        except Exception:
            return np.nan

    def _agg_metrics(y_true_vec, y_pred_vec):
        mae = float(mean_absolute_error(y_true_vec, y_pred_vec))
        rmse = float(np.sqrt(mean_squared_error(y_true_vec, y_pred_vec)))
        pear = _safe_pearson(y_true_vec, y_pred_vec)
        spear = _safe_spearman(y_true_vec, y_pred_vec)
        bias = float(np.mean(y_pred_vec - y_true_vec))
        denom = np.where(np.abs(y_true_vec) < 1e-12, np.nan, y_true_vec)
        rel_bias = float(np.nanmean((y_pred_vec - y_true_vec) / denom))
        return mae, rmse, pear, spear, bias, rel_bias

    def _make_groups(coords_test, y_true_np, y_pred_np, k, seed):
        if grouping == "knn":
            return spatial_grouping_services.make_group_sums_knn(
                coords_test, y_true_np, y_pred_np,
                k=k, max_groups=max_groups, seed=seed
            )
        else:
            return spatial_grouping_services.make_group_sums_random(
                y_true_np, y_pred_np,
                k=k, max_groups=max_groups, seed=seed
            )

    # -------------------------
    # CV loop
    # -------------------------
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_C), start=1):
        Xb_train, Xb_test = X_B.iloc[train_idx], X_B.iloc[test_idx]
        Xc_train, Xc_test = X_C.iloc[train_idx], X_C.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        coords_test = coords[test_idx]

        # Fit / predict
        A = modeles_services_regression.BaselineMeanPredictor().fit(y_train)
        y_pred_A = np.asarray(A.predict(len(y_test)), dtype=float)

        B = modeles_services_regression.BaselineNegativeBinomial(
            feature_cols=baseline_B_features
        )
        B.fit(Xb_train, y_train)
        y_pred_B = np.asarray(B.predict(Xb_test), dtype=float)

        C = modeles_services_regression.ModelCPoissonLGBM(
            params=lgbm_params, random_state=random_state
        )
        C.fit(Xc_train, y_train)
        y_pred_C = np.asarray(C.predict(Xc_test), dtype=float)

        y_test_np = y_test.to_numpy(dtype=float)

        for k in k_groups:
            if k >= len(y_test_np):
                continue

            # -------------------------
            # 1) RSE distribution
            # -------------------------
            seed_rse = int(random_state + 1000 * fold + 10 * k)

            if grouping == "knn":
                rse_A = spatial_grouping_services.spatial_knn_rse(
                    coords_test, y_test_np, y_pred_A, k
                )
                rse_B = spatial_grouping_services.spatial_knn_rse(
                    coords_test, y_test_np, y_pred_B, k
                )
                rse_C = spatial_grouping_services.spatial_knn_rse(
                    coords_test, y_test_np, y_pred_C, k
                )
            else:
                rse_A = spatial_grouping_services.random_group_rse(
                    y_test_np, y_pred_A, k=k, max_groups=max_groups, seed=seed_rse
                )
                rse_B = spatial_grouping_services.random_group_rse(
                    y_test_np, y_pred_B, k=k, max_groups=max_groups, seed=seed_rse
                )
                rse_C = spatial_grouping_services.random_group_rse(
                    y_test_np, y_pred_C, k=k, max_groups=max_groups, seed=seed_rse
                )

            # Precompute RSE stats
            rse_stats = {
                "A_mean": (
                    float(np.mean(rse_A)), float(np.median(rse_A)),
                    float(np.quantile(rse_A, 0.90)), float(np.quantile(rse_A, 0.95))
                ),
                "B_neg_bin": (
                    float(np.mean(rse_B)), float(np.median(rse_B)),
                    float(np.quantile(rse_B, 0.90)), float(np.quantile(rse_B, 0.95))
                ),
                "C_lgbm_poisson": (
                    float(np.mean(rse_C)), float(np.median(rse_C)),
                    float(np.quantile(rse_C, 0.90)), float(np.quantile(rse_C, 0.95))
                ),
            }

            # -------------------------
            # 2) Aggregated groups (same seeds)
            # -------------------------
            seed_groups = seed_rse

            true_A, pred_A = _make_groups(coords_test, y_test_np, y_pred_A, k, seed_groups)
            true_B, pred_B = _make_groups(coords_test, y_test_np, y_pred_B, k, seed_groups)
            true_C, pred_C = _make_groups(coords_test, y_test_np, y_pred_C, k, seed_groups)

            true_sums = true_A
            true_means = true_sums / float(k)

            for model_name, pred_sums in [
                ("A_mean", pred_A),
                ("B_neg_bin", pred_B),
                ("C_lgbm_poisson", pred_C),
            ]:
                mean_rse, median_rse, p90_rse, p95_rse = rse_stats[model_name]

                # --- SUM metrics
                mae_sum, rmse_sum, pear_sum, spear_sum, bias_sum, rel_bias_sum = \
                    _agg_metrics(true_sums, pred_sums)

                # --- MEAN metrics
                pred_means = pred_sums / float(k)
                mae_mean, rmse_mean, pear_mean, spear_mean, bias_mean, rel_bias_mean = \
                    _agg_metrics(true_means, pred_means)

                records.append({
                    "fold": fold,
                    "model": model_name,
                    "group_size": k,
                    "grouping": grouping,

                    # RSE
                    "mean_rse": mean_rse,
                    "median_rse": median_rse,
                    "p90_rse": p90_rse,
                    "p95_rse": p95_rse,

                    # SUM metrics
                    "pearson_r_sum": pear_sum,
                    "spearman_r_sum": spear_sum,
                    "mae_sum": mae_sum,
                    "rmse_sum": rmse_sum,
                    "bias_sum": bias_sum,
                    "rel_bias_sum": rel_bias_sum,

                    # MEAN metrics
                    "pearson_r_mean_grp": pear_mean,
                    "spearman_r_mean_grp": spear_mean,
                    "mae_mean_grp": mae_mean,
                    "rmse_mean_grp": rmse_mean,
                    "bias_mean_grp": bias_mean,
                    "rel_bias_mean_grp": rel_bias_mean,

                    "n_groups": int(len(true_sums)),
                    "max_groups": None if max_groups is None else int(max_groups),
                })

    return pd.DataFrame(records)




def build_readable_cv_table(
    df_long: pd.DataFrame,
    target_name: str | None = None,
    model_order: list[str] | None = None,
    group_order: list[int] | None = None,
    round_digits: int = 3
) -> pd.DataFrame:
    df = df_long.copy()

    candidate_metrics = [
        "mean_rse", "median_rse", "p90_rse", "p95_rse",
        "pearson_r_sum", "spearman_r_sum", "mae_sum", "rmse_sum", "bias_sum", "rel_bias_sum",
        "pearson_r_mean_grp", "spearman_r_mean_grp", "mae_mean_grp", "rmse_mean_grp", "bias_mean_grp", "rel_bias_mean_grp",
        "n_groups",
    ]
    metrics = [m for m in candidate_metrics if m in df.columns]
    if not metrics:
        raise ValueError("No recognized metrics found in df_long.")

    # NEW: grouping dimension if present
    group_keys = ["model", "group_size"]
    if "grouping" in df.columns:
        group_keys = ["grouping"] + group_keys

    agg_map = {m: ["mean", "std"] for m in metrics}
    out = df.groupby(group_keys, as_index=True).agg(agg_map)

    # Optional ordering (model/group only; grouping stays first level)
    if model_order is not None:
        # model is level -2 if grouping exists, else level 0
        model_level = 1 if "grouping" in df.columns else 0
        out = out.reindex(model_order, level=model_level)
    if group_order is not None:
        group_level = 2 if "grouping" in df.columns else 1
        out = out.reindex(group_order, level=group_level)

    rename_metrics = {
        "mean_rse": "RSE_mean",
        "median_rse": "RSE_median",
        "p90_rse": "RSE_p90",
        "p95_rse": "RSE_p95",
        "pearson_r_sum": "Pearson_sum",
        "spearman_r_sum": "Spearman_sum",
        "mae_sum": "MAE_sum",
        "rmse_sum": "RMSE_sum",
        "bias_sum": "Bias_sum",
        "rel_bias_sum": "RelBias_sum",
        "pearson_r_mean_grp": "Pearson_meanGrp",
        "spearman_r_mean_grp": "Spearman_meanGrp",
        "mae_mean_grp": "MAE_meanGrp",
        "rmse_mean_grp": "RMSE_meanGrp",
        "bias_mean_grp": "Bias_meanGrp",
        "rel_bias_mean_grp": "RelBias_meanGrp",
        "n_groups": "N_groups",
    }
    out = out.rename(columns=rename_metrics, level=0)

    if target_name is not None:
        out.insert(0, ("Target", ""), target_name)

    out = out.copy()
    for col in out.columns:
        if col[0] == "N_groups":
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].round(round_digits)

    # NEW: reset index with grouping if present
    out = out.reset_index()
    out = out.rename(columns={"model": "Model", "group_size": "Group_size"})
    if "grouping" in out.columns:
        out = out.rename(columns={"grouping": "Grouping"})

    return out






def cv_collect_group_sums_modelC(
    X_C, coords, y, modelC_factory,
    k=120, n_splits=5, random_state=42, max_groups_per_fold=600
    ):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    all_true = []
    all_pred = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_C), start=1):
        X_train, X_test = X_C.iloc[train_idx], X_C.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        coords_test = coords[test_idx]

        model = modelC_factory()
        model.fit(X_train, y_train)
        y_pred_test = np.asarray(model.predict(X_test))

        t, p = spatial_grouping_services.make_group_sums_knn(
            coords=coords_test,
            y_true=y_test.to_numpy(),
            y_pred=y_pred_test,
            k=k,
            max_groups=max_groups_per_fold,
            seed=1000 * fold + k
        )
        all_true.append(t)
        all_pred.append(p)

    return np.concatenate(all_true), np.concatenate(all_pred)


