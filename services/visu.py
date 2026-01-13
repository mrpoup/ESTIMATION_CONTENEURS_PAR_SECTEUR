from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split

from services import modeles_services_regression
from services import spatial_grouping_services


def _safe_corrs(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.allclose(np.std(x), 0) or np.allclose(np.std(y), 0):
        return np.nan, np.nan
    p = float(pearsonr(x, y)[0])
    s = float(spearmanr(x, y).correlation)
    return p, s


def plot_lgbm_true_vs_pred_by_group_size(
    dataset_obj,
    coords,
    target_col: str,
    features_cols: list[str],
    k_list: list[int] = (30, 60, 120),
    mode: str = "sum",               # "sum" or "mean"
    test_size: float = 0.2,
    random_state: int = 42,
    max_groups: int = 600,
    gridsize: int = 40,
    lgbm_params: dict | None = None,
):
    """
    Train a LightGBM Poisson on point-level data, then evaluate aggregated predictions
    over kNN groups in the TEST set. Produce one scatter/hexbin panel per k.

    Parameters
    ----------
    dataset_obj : object with .X and .Y
    coords : np.ndarray (n,2) coordinates aligned with dataset_obj rows
    target_col : str
    features_cols : list[str]
    k_list : list[int]
    mode : "sum" or "mean"
    test_size, random_state : split controls
    max_groups : max number of groups plotted (seed subsampling)
    gridsize : hexbin resolution
    lgbm_params : optional params passed to your ModelCPoissonLGBM

    Returns
    -------
    fig, axes
    """
    assert mode in ("sum", "mean"), "mode must be 'sum' or 'mean'"

    X = dataset_obj.X[features_cols].copy()
    y = dataset_obj.Y[target_col].copy()
    coords = np.asarray(coords)

    # Split indices to keep alignment with coords
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=random_state, shuffle=True)

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    coords_test = coords[test_idx]

    # Fit LightGBM Poisson (your wrapper)
    C = modeles_services_regression.ModelCPoissonLGBM(params=lgbm_params, random_state=random_state)
    C.fit(X_train, y_train)
    y_pred_test = np.asarray(C.predict(X_test), dtype=float)

    # Numerical safety
    y_test_np = y_test.to_numpy(dtype=float)
    y_pred_test = np.clip(y_pred_test, 1e-9, None)

    # Layout
    k_list = list(k_list)
    n = len(k_list)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.2 * ncols, 5.6 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, k in zip(axes, k_list):
        if k >= len(y_test_np):
            ax.set_visible(False)
            continue

        # Use deterministic seed so the same groups are used if you re-run
        seed_groups = int(random_state + 10 * k)

        true_sums, pred_sums = spatial_grouping_services.make_group_sums_knn(
            coords_test, y_test_np, y_pred_test, k=k, max_groups=max_groups, seed=seed_groups
        )

        if mode == "mean":
            true_vals = true_sums / float(k)
            pred_vals = pred_sums / float(k)
            xlab = f"Moyenne observée (k={k})"
            ylab = f"Moyenne estimée (k={k})"
        else:
            true_vals = true_sums
            pred_vals = pred_sums
            xlab = f"Somme observée (k={k})"
            ylab = f"Somme estimée (k={k})"

        # Corrs
        pr, sr = _safe_corrs(true_vals, pred_vals)

        # Plot as hexbin (more readable than scatter when dense)
        hb = ax.hexbin(true_vals, pred_vals, gridsize=gridsize, mincnt=1)
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label("Densité (hexbin)")

        # Reference y=x
        minv = float(min(np.nanmin(true_vals), np.nanmin(pred_vals)))
        maxv = float(max(np.nanmax(true_vals), np.nanmax(pred_vals)))
        ax.plot([minv, maxv], [minv, maxv], linewidth=2)

        # Title with correlations
        ax.set_title(
            f"{target_col} | mode={mode} | k={k}\n"
            f"Pearson r={pr:.3f} | Spearman ρ={sr:.3f}"
        )

        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.2)

    # Hide unused axes
    for j in range(len(k_list), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"LightGBM Poisson — Observé vs Estimé (agrégation kNN) — {target_col}", y=1.02, fontsize=14)
    plt.tight_layout()
    plt.show()

    return fig, axes



def _extract_curve(summary, model, metric, ks ):
    """
    summary: DataFrame with index (model, group_size) and columns MultiIndex (metric, agg)
    metric: "median_rse" or "p95_rse"
    returns: y_mean, y_std as numpy arrays aligned with ks
    """
    y_mean = []
    y_std = []
    
    for k in ks:
        y_mean.append(summary.loc[(model, k), (metric, "mean")])
        y_std.append(summary.loc[(model, k), (metric, "std")])
    return np.array(y_mean, dtype=float), np.array(y_std, dtype=float)
            
def plot_graph1_multitarget(summaries, ks , as_percent=True):
    """
    summaries: dict {target_name: summary_df}
    """
    targets = list(summaries.keys())

    models = [
        ("A_mean", "Baseline A (moyenne)"),
        ("B_neg_bin", "Baseline B (binomiale négative)"),
        ("C_lgbm_poisson", "Modèle C (surfaces + Poisson)"),
    ]

    metrics = [
        ("median_rse", "Erreur médiane sur le total"),
        ("p95_rse", "Erreur dans 95% des cas (p95)"),
    ]

    fig, axes = plt.subplots(
        nrows=len(targets),
        ncols=len(metrics),
        figsize=(12, 3.6 * len(targets)),
        sharex=True
    )

    if len(targets) == 1:
        axes = np.array([axes])  # normalise

    scale = 100.0 if as_percent else 1.0
    y_label = "Erreur relative (%)" if as_percent else "Erreur relative (ratio)"

    for i, target in enumerate(targets):
        summary = summaries[target]

        for j, (metric, metric_title) in enumerate(metrics):
            ax = axes[i, j]

            for model_code, model_label in models:
                y_mean, y_std = _extract_curve(summary, model_code, metric, ks=ks)

                ax.plot(ks, y_mean * scale, marker="o", label=model_label)
                # Option: bande +/- 1 std (ça aide à “voir” la stabilité)
                ax.fill_between(
                    ks,
                    (y_mean - y_std) * scale,
                    (y_mean + y_std) * scale,
                    alpha=0.12
                )

            if i == 0:
                ax.set_title(metric_title, fontsize=12)

            if j == 0:
                ax.set_ylabel(f"{target}\n{y_label}")

            ax.grid(True, alpha=0.2)
            ax.set_xticks(list(ks))
            ax.set_xlabel("Taille du pseudo-secteur (nombre de bâtiments)")

    # Légende unique en bas
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=1, frameon=False)
    fig.suptitle("Précision des estimations par agrégation spatiale k-NN", fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])

    plt.show()



def _extract_curve(summary, model, metric, ks ):
    y_mean = []
    y_std = []
    for k in ks:
        y_mean.append(summary.loc[(model, k), (metric, "mean")])
        y_std.append(summary.loc[(model, k), (metric, "std")])
    return np.array(y_mean, dtype=float), np.array(y_std, dtype=float)


def plot_graph1_ultra_decideur(
    summaries: dict,
    ks ,
    as_percent=True
):
    """
    Ultra-decider plot:
    - Only p95 (risk)
    - Only A_mean vs C_lgbm_poisson
    - One panel per target
    """
    targets = list(summaries.keys())

    models = [
        ("A_mean", "Sans modèle (moyenne)"),
        ("C_lgbm_poisson", "Avec modèle (surfaces + Poisson)"),
    ]

    metric = "p95_rse"
    title = "Risque d’erreur sur le total d’un secteur (p95)"
    subtitle = "Interprétation : dans 95% des cas, l’erreur sur le total est inférieure à ce pourcentage"

    fig, axes = plt.subplots(
        nrows=len(targets),
        ncols=1,
        figsize=(10, 3.2 * len(targets)),
        sharex=True
    )

    if len(targets) == 1:
        axes = [axes]

    scale = 100.0 if as_percent else 1.0
    y_label = "Erreur relative (%)" if as_percent else "Erreur relative (ratio)"

    for i, target in enumerate(targets):
        ax = axes[i]
        summary = summaries[target]

        for model_code, model_label in models:
            y_mean, y_std = _extract_curve(summary, model_code, metric, ks=ks)

            ax.plot(ks, y_mean * scale, marker="o", linewidth=2, label=model_label)
            ax.fill_between(
                ks,
                (y_mean - y_std) * scale,
                (y_mean + y_std) * scale,
                alpha=0.15
            )

            # Annotations (very readable): p95 values
            for x, yv in zip(ks, y_mean * scale):
                ax.annotate(f"{yv:.0f}%", (x, yv), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)

        ax.set_ylabel(f"{target}\n{y_label}")
        ax.grid(True, alpha=0.2)
        ax.set_xticks(list(ks))

    axes[-1].set_xlabel("Taille du secteur (nombre de bâtiments)")

    # Legend + titles
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)

    fig.suptitle(title, fontsize=14, y=0.995)
    fig.text(0.5, 0.965, subtitle, ha="center", fontsize=10)

    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    plt.show()





def plot_true_vs_pred_sector_sums(true_sums, pred_sums, k, min_max, title, gridsize=40):
    true_sums = np.asarray(true_sums, dtype=float)
    pred_sums = np.asarray(pred_sums, dtype=float)

    # Valeurs globales (comportement par défaut)
    minv = float(min(true_sums.min(), pred_sums.min()))
    maxv = float(max(true_sums.max(), pred_sums.max()))

    # Si min_max est fourni, on l'utilise pour l'affichage
    if min_max is not None:
        vmin, vmax = min_max
        minv_plot, maxv_plot = vmin, vmax
    else:
        minv_plot, maxv_plot = minv, maxv

    fig, ax = plt.subplots(figsize=(8.8, 7.2))

    hb = ax.hexbin(true_sums, pred_sums, gridsize=gridsize, mincnt=1)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("Densité de groupes (hexbin)")

    # Reference lines (tracées sur la zone visible)
    x = np.array([minv_plot, maxv_plot], dtype=float)

    # y = x
    ax.plot(x, x, linewidth=2)

    # ±10% (dashed)
    ax.plot(x, 1.10 * x, linestyle="--", linewidth=1.5)
    ax.plot(x, 0.90 * x, linestyle="--", linewidth=1.5)

    # ±20% (dotted)
    ax.plot(x, 1.20 * x, linestyle=":", linewidth=1.8)
    ax.plot(x, 0.80 * x, linestyle=":", linewidth=1.8)

    # Limitation des axes si demandé
    ax.set_xlim(minv_plot, maxv_plot)
    ax.set_ylim(minv_plot, maxv_plot)

    ax.set_xlabel(f"Total observé sur le groupe (k={k})")
    ax.set_ylabel(f"Total estimé sur le groupe (k={k})")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)

    # Legend (lines only)
    legend_elements = [
        Line2D([0], [0], color="black", lw=2, label="Parfait : estimé = observé (y=x)"),
        Line2D([0], [0], color="black", lw=1.5, linestyle="--", label="Bande ±10%"),
        Line2D([0], [0], color="black", lw=1.8, linestyle=":", label="Bande ±20%"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", frameon=True)

    plt.show()
