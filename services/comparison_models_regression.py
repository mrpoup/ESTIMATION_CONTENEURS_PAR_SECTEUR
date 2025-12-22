import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from services import modeles_services_regression
from services import metrics_services_regression

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

def set_test_A_B(target:str,X,Y):
    y_true = Y[target]

    baseline_A =modeles_services_regression.BaselineMeanPredictor().fit(y_true)
    y_pred_A = baseline_A.predict(len(y_true))

    metrics_A = metrics_services_regression.CountRegressionMetrics.compute_all(y_true, y_pred_A)

    print(f'\n{target} Metrics sur baseline mean:')
    print(metrics_A)

    #
    baseline_B_features = [
        "log1p_surf_batiment_source_m2",
        "hauteur_corrigee_m",
        "log1p_volume_batiment",
        'log1p_surf_buffer_m2_b10_m', 'log1p_surf_buffer_m2_b50_m'
    ]


    baseline_B = modeles_services_regression.BaselineNegativeBinomial(feature_cols=baseline_B_features)
    baseline_B.fit(X, y_true)

    y_pred_B = baseline_B.predict(X)

    metrics_B = metrics_services_regression.CountRegressionMetrics.compute_all(y_true, y_pred_B)

    print(f'\n{target} sur baseline regression binomiale negative:')
    print(metrics_B)


    cv_results = cross_validate_baselines(
        X=X,
        y=Y[target],
        baseline_B_features=baseline_B_features,
    )
    print(f'\n{target} CROSS VALIDATION RESULTS:  ')
    print(cv_results)

    summary = (
        cv_results
        .groupby("baseline")
        .agg(["mean", "std"])
    )
    print(f'\n{target} CROSS VALIDATION SUMMARY RESULTS: ')
    print(summary)

    #SIMULATION d'AGREGATION:
    groupes=(30, 60, 120)
    agg_results = cv_aggregated_protocol_AB(
        X=X,
        y=Y[target],
        baseline_B_features=baseline_B_features,
        group_sizes=groupes,
        n_draws=1000,
    )
    print(f'\n{target} CROSS VALIDATION ON AGGREG. RESULTS: ')
    print(agg_results)

    summary = (
        agg_results
        .groupby(["model", "group_size"])
        .agg(["mean", "std"])
    )

    print(f'\n{target} CROSS VALIDATION ON AGGREG. SUMMARY RESULTS: ')
    print(agg_results)
    print(summary)

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

            rse_A = aggregated_error_distribution(y_test_np, y_pred_A, group_size=m, n_draws=n_draws, seed=fold*1000 + m)
            rse_B = aggregated_error_distribution(y_test_np, y_pred_B, group_size=m, n_draws=n_draws, seed=fold*1000 + m + 1)
            rse_C = aggregated_error_distribution(y_test_np, y_pred_C, group_size=m, n_draws=n_draws, seed=fold*1000 + m + 2)

            for model_name, rse in [("A_mean", rse_A), ("B_neg_bin", rse_B), ("C_lgbm_poisson", rse_C)]:
                all_records.append({
                    "fold": fold,
                    "model": model_name,
                    "group_size": m,
                    "mean_rse": float(np.mean(rse)),
                    "median_rse": float(np.median(rse)),
                    "p90_rse": float(np.quantile(rse, 0.90)),
                    "p95_rse": float(np.quantile(rse, 0.95)),
                })

    return pd.DataFrame(all_records)

def spatial_knn_rse(
    coords: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int,
    eps: float = 1e-9
):
    """
    Compute relative sum error for spatial k-NN aggregation.
    
    coords : (n, 2) array of x, y coordinates
    y_true : (n,) true values
    y_pred : (n,) predicted values
    k      : number of neighbors
    """
    n = len(y_true)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(coords)
    _, indices = nbrs.kneighbors(coords)

    rse = np.zeros(n, dtype=float)

    for i in range(n):
        idx = indices[i]
        S = y_true[idx].sum()
        Shat = y_pred[idx].sum()
        rse[i] = abs(Shat - S) / (S + eps)

    return rse

def cv_spatial_knn_protocol_ABC(
    X_B,
    X_C,
    coords,
    y,
    baseline_B_features,
    k_values=(30, 60, 120),
    n_splits=5,
    random_state=42,
    lgbm_params=None,
):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    records = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_C), start=1):
        Xb_train, Xb_test = X_B.iloc[train_idx], X_B.iloc[test_idx]
        Xc_train, Xc_test = X_C.iloc[train_idx], X_C.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        coords_test = coords[test_idx]

        # A — mean
        A = modeles_services_regression.BaselineMeanPredictor().fit(y_train)
        y_pred_A = A.predict(len(y_test))

        # B — NB
        B = modeles_services_regression.BaselineNegativeBinomial(feature_cols=baseline_B_features)
        B.fit(Xb_train, y_train)
        y_pred_B = np.asarray(B.predict(Xb_test))

        # C — LGBM Poisson
        C = modeles_services_regression.ModelCPoissonLGBM(params=lgbm_params, random_state=random_state)
        C.fit(Xc_train, y_train)
        y_pred_C = np.asarray(C.predict(Xc_test))

        y_test_np = y_test.to_numpy()

        for k in k_values:
            if k >= len(y_test_np):
                continue

            rse_A = spatial_knn_rse(coords_test, y_test_np, y_pred_A, k)
            rse_B = spatial_knn_rse(coords_test, y_test_np, y_pred_B, k)
            rse_C = spatial_knn_rse(coords_test, y_test_np, y_pred_C, k)

            for model_name, rse in [
                ("A_mean", rse_A),
                ("B_neg_bin", rse_B),
                ("C_lgbm_poisson", rse_C),
            ]:
                records.append({
                    "fold": fold,
                    "model": model_name,
                    "group_size": k,
                    "mean_rse": float(np.mean(rse)),
                    "median_rse": float(np.median(rse)),
                    "p90_rse": float(np.quantile(rse, 0.90)),
                    "p95_rse": float(np.quantile(rse, 0.95)),
                })

    return pd.DataFrame(records)

def make_group_sums_knn(coords, y_true, y_pred, k=120, max_groups=600, seed=42):
    """
    Build sector-like groups with k-NN within a given TEST fold and return
    arrays of (true_sum, pred_sum). Uses random subsampling of seeds to limit overlap.
    """
    coords = np.asarray(coords)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n = len(y_true)
    k = min(k, n)

    nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(coords)
    _, indices = nbrs.kneighbors(coords)

    rng = np.random.default_rng(seed)
    seeds = np.arange(n)
    if max_groups is not None and max_groups < n:
        seeds = rng.choice(seeds, size=max_groups, replace=False)

    true_sums = np.array([y_true[indices[i]].sum() for i in seeds], dtype=float)
    pred_sums = np.array([y_pred[indices[i]].sum() for i in seeds], dtype=float)
    return true_sums, pred_sums


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

        t, p = make_group_sums_knn(
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


