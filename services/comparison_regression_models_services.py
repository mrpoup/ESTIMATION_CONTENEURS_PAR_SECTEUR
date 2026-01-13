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
):
    """
    Cross-validated spatial kNN aggregation evaluation for 3 models:
      A: mean baseline
      B: Negative Binomial baseline
      C: LightGBM Poisson

    For each fold and each group size k, computes:
      - RSE distribution stats (mean/median/p90/p95) using spatial_knn_rse
      - Metrics on aggregated SUMS (sector totals)
      - Metrics on aggregated MEANS per group (sector average = total/k)

    Notes
    -----
    - RSE is computed per-point (each point as a seed) via spatial_knn_rse.
    - Aggregated metrics are computed on sector-like groups created by make_group_sums_knn,
      with optional random subsampling of seeds (max_groups).
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    records = []

    coords = np.asarray(coords)

    # Helper: safe correlations
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

    # Metrics on aggregated vectors
    def _agg_metrics(y_true_vec, y_pred_vec):
        mae = float(mean_absolute_error(y_true_vec, y_pred_vec))
        rmse = float(np.sqrt(mean_squared_error(y_true_vec, y_pred_vec)))
        pear = _safe_pearson(y_true_vec, y_pred_vec)
        spear = _safe_spearman(y_true_vec, y_pred_vec)
        bias = float(np.mean(y_pred_vec - y_true_vec))  # signed bias
        denom = np.where(np.abs(y_true_vec) < 1e-12, np.nan, y_true_vec)
        rel_bias = float(np.nanmean((y_pred_vec - y_true_vec) / denom))
        return mae, rmse, pear, spear, bias, rel_bias

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_C), start=1):
        Xb_train, Xb_test = X_B.iloc[train_idx], X_B.iloc[test_idx]
        Xc_train, Xc_test = X_C.iloc[train_idx], X_C.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        coords_test = coords[test_idx]

        # -------------------------
        # Fit / predict for each model
        # -------------------------
        A = modeles_services_regression.BaselineMeanPredictor().fit(y_train)
        y_pred_A = np.asarray(A.predict(len(y_test)), dtype=float)

        B = modeles_services_regression.BaselineNegativeBinomial(feature_cols=baseline_B_features)
        B.fit(Xb_train, y_train)
        y_pred_B = np.asarray(B.predict(Xb_test), dtype=float)

        C = modeles_services_regression.ModelCPoissonLGBM(params=lgbm_params, random_state=random_state)
        C.fit(Xc_train, y_train)
        y_pred_C = np.asarray(C.predict(Xc_test), dtype=float)

        y_test_np = y_test.to_numpy(dtype=float)

        for k in k_groups:
            if k >= len(y_test_np):
                continue

            # -------------------------
            # 1) RSE (per-point)
            # -------------------------
            rse_A = spatial_grouping_services.spatial_knn_rse(coords_test, y_test_np, y_pred_A, k)
            rse_B = spatial_grouping_services.spatial_knn_rse(coords_test, y_test_np, y_pred_B, k)
            rse_C = spatial_grouping_services.spatial_knn_rse(coords_test, y_test_np, y_pred_C, k)

            # -------------------------
            # 2) Sector-like aggregated sums (same seeds for all models)
            # -------------------------
            seed_groups = int(random_state + 1000 * fold + 10 * k)

            true_A, pred_A = spatial_grouping_services.make_group_sums_knn(
                coords_test, y_test_np, y_pred_A, k=k, max_groups=max_groups, seed=seed_groups
            )
            true_B, pred_B = spatial_grouping_services.make_group_sums_knn(
                coords_test, y_test_np, y_pred_B, k=k, max_groups=max_groups, seed=seed_groups
            )
            true_C, pred_C = spatial_grouping_services.make_group_sums_knn(
                coords_test, y_test_np, y_pred_C, k=k, max_groups=max_groups, seed=seed_groups
            )

            # Sanity: true vectors should match
            true_sums = true_A

            # Precompute the "mean per group" vectors
            true_means = true_sums / float(k)

            for model_name, rse, pred_sums in [
                ("A_mean", rse_A, pred_A),
                ("B_neg_bin", rse_B, pred_B),
                ("C_lgbm_poisson", rse_C, pred_C),
            ]:
                # --- SUM metrics
                mae_sum, rmse_sum, pear_sum, spear_sum, bias_sum, rel_bias_sum = _agg_metrics(true_sums, pred_sums)

                # --- MEAN-per-group metrics
                pred_means = pred_sums / float(k)
                mae_mean, rmse_mean, pear_mean, spear_mean, bias_mean, rel_bias_mean = _agg_metrics(true_means, pred_means)

                records.append({
                    "fold": fold,
                    "model": model_name,
                    "group_size": k,

                    # RSE distribution (per-point seeds)
                    "mean_rse": float(np.mean(rse)),
                    "median_rse": float(np.median(rse)),
                    "p90_rse": float(np.quantile(rse, 0.90)),
                    "p95_rse": float(np.quantile(rse, 0.95)),

                    # --- Aggregated SUMS metrics (sector totals)
                    "pearson_r_sum": pear_sum,
                    "spearman_r_sum": spear_sum,
                    "mae_sum": mae_sum,
                    "rmse_sum": rmse_sum,
                    "bias_sum": bias_sum,
                    "rel_bias_sum": rel_bias_sum,

                    # --- Aggregated MEANS metrics (sector mean = total/k)
                    "pearson_r_mean_grp": pear_mean,
                    "spearman_r_mean_grp": spear_mean,
                    "mae_mean_grp": mae_mean,
                    "rmse_mean_grp": rmse_mean,
                    "bias_mean_grp": bias_mean,
                    "rel_bias_mean_grp": rel_bias_mean,

                    # helpful to interpret robustness
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
    """
    Build an Excel-like summary table from a CV long dataframe.

    Expected input columns (at least):
      - model, group_size
      - mean_rse, median_rse, p90_rse, p95_rse

    Optional (if present):
      - *_sum metrics (sector totals): pearson_r_sum, spearman_r_sum, mae_sum, rmse_sum, bias_sum, rel_bias_sum
      - *_mean_grp metrics (sector mean = total/k): pearson_r_mean_grp, spearman_r_mean_grp, mae_mean_grp, rmse_mean_grp, bias_mean_grp, rel_bias_mean_grp
      - n_groups

    Output:
      A wide table with MultiIndex columns: (metric, stat) where stat in {"mean", "std"}.
    """
    df = df_long.copy()

    # Select numeric metrics that exist in df
    candidate_metrics = [
        # RSE
        "mean_rse", "median_rse", "p90_rse", "p95_rse",

        # SUM metrics (sector totals)
        "pearson_r_sum", "spearman_r_sum",
        "mae_sum", "rmse_sum",
        "bias_sum", "rel_bias_sum",

        # MEAN per group metrics (sector mean)
        "pearson_r_mean_grp", "spearman_r_mean_grp",
        "mae_mean_grp", "rmse_mean_grp",
        "bias_mean_grp", "rel_bias_mean_grp",

        # robustness
        "n_groups"
    ]
    metrics = [m for m in candidate_metrics if m in df.columns]

    if not metrics:
        raise ValueError("No recognized metrics found in df_long.")

    # Aggregate across folds (mean/std)
    agg_map = {m: ["mean", "std"] for m in metrics}
    out = (
        df.groupby(["model", "group_size"], as_index=True)
          .agg(agg_map)
    )

    # Optional ordering
    if model_order is not None:
        out = out.reindex(model_order, level=0)
    if group_order is not None:
        out = out.reindex(group_order, level=1)

    # Rename metrics to be more report-friendly
    rename_metrics = {
        # RSE
        "mean_rse": "RSE_mean",
        "median_rse": "RSE_median",
        "p90_rse": "RSE_p90",
        "p95_rse": "RSE_p95",

        # SUM metrics
        "pearson_r_sum": "Pearson_sum",
        "spearman_r_sum": "Spearman_sum",
        "mae_sum": "MAE_sum",
        "rmse_sum": "RMSE_sum",
        "bias_sum": "Bias_sum",
        "rel_bias_sum": "RelBias_sum",

        # MEAN-per-group metrics
        "pearson_r_mean_grp": "Pearson_meanGrp",
        "spearman_r_mean_grp": "Spearman_meanGrp",
        "mae_mean_grp": "MAE_meanGrp",
        "rmse_mean_grp": "RMSE_meanGrp",
        "bias_mean_grp": "Bias_meanGrp",
        "rel_bias_mean_grp": "RelBias_meanGrp",

        # other
        "n_groups": "N_groups"
    }
    out = out.rename(columns=rename_metrics, level=0)

    # Add target column (like your Excel "Type" block)
    if target_name is not None:
        out.insert(0, ("Target", ""), target_name)

    # Round numeric values (keep N_groups as int-ish if you want)
    out = out.copy()
    for col in out.columns:
        if col[0] == "N_groups":
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].round(round_digits)

    # Make table more Excel-like: reset index
    out = out.reset_index().rename(columns={"model": "Model", "group_size": "Group_size"})

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


