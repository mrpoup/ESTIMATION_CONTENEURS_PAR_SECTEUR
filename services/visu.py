from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold

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





def plot_true_vs_pred_sector_sums(true_sums, pred_sums, title, gridsize=40):
    true_sums = np.asarray(true_sums, dtype=float)
    pred_sums = np.asarray(pred_sums, dtype=float)

    minv = float(min(true_sums.min(), pred_sums.min()))
    maxv = float(max(true_sums.max(), pred_sums.max()))

    fig, ax = plt.subplots(figsize=(8.8, 7.2))

    hb = ax.hexbin(true_sums, pred_sums, gridsize=gridsize, mincnt=1)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("Densité de groupes (hexbin)")

    # Reference lines
    x = np.array([minv, maxv], dtype=float)

    # y = x
    ax.plot(x, x, linewidth=2)

    # ±10% (dashed)
    ax.plot(x, 1.10 * x, linestyle="--", linewidth=1.5)
    ax.plot(x, 0.90 * x, linestyle="--", linewidth=1.5)

    # ±20% (dotted)
    ax.plot(x, 1.20 * x, linestyle=":", linewidth=1.8)
    ax.plot(x, 0.80 * x, linestyle=":", linewidth=1.8)

    ax.set_xlabel("Total observé sur le groupe (k=120)")
    ax.set_ylabel("Total estimé sur le groupe (k=120)")
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