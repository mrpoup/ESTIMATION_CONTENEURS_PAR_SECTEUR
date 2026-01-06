import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

def log1p_series(s: pd.Series) -> pd.Series:
    """Safe log1p transformation (casts to float, replaces negatives if any)."""
    return np.log1p(s.astype(float).clip(lower=0))

 #'contenant enterré','grand contenant', 'petit contenant'
def plot_container_scatters(df: pd.DataFrame,
                            use_log1p_vf,
                            col_ent: str = "contenant enterré",
                            col_grand: str = "grand contenant",
                            col_petit: str = "petit contenant",
                            limites=None,
                            alpha: float = 0.2):
    """
    Plot pairwise scatter plots (log1p) for container counts.
    """

    suff=''
    if use_log1p_vf:

        data = {
            "enterré": log1p_series(df[col_ent]),
            "grand": log1p_series(df[col_grand]),
            "petit": log1p_series(df[col_petit]),
        }
    else:
        data = {
            "enterré": df[col_ent],
            "grand": df[col_grand],
            "petit": df[col_petit],
        }
       


    pairs = [
        ("grand", "petit"),
        ("enterré", "petit"),
        ("enterré", "grand"),
    ]

    for x, y in pairs:
        if use_log1p_vf:
            xlabel=(f"log1p({x})")
            ylabel=(f"log1p({y})")
            title=(f"{y} vs {x} (log1p)")
        else:
            xlabel=(f"{x}")
            ylabel=(f"{y}")
            title=(f"{y} vs {x}")


        plt.figure(figsize=(5, 5))
        plt.scatter(data[x], data[y], alpha=alpha, s=10)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        if not use_log1p_vf and limites is not None:
            plt.xlim(limites[x])
            plt.ylim(limites[y])
        
        plt.grid(True, linewidth=0.3)
        plt.tight_layout()
        plt.show()


def plot_conditional_histograms(df: pd.DataFrame,
                                target_col: str,
                                condition_col: str,
                                bins: int = 30):
    """
    Plot histograms of target_col conditioned on condition_col == 0 vs > 0.
    """

    target_log = log1p_series(df[target_col])

    mask_zero = df[condition_col] == 0
    mask_pos = df[condition_col] > 0

    plt.figure(figsize=(6, 4))
    plt.hist(target_log[mask_zero], bins=bins, alpha=0.6, label=f"{condition_col} == 0")
    plt.hist(target_log[mask_pos], bins=bins, alpha=0.6, label=f"{condition_col} > 0")
    plt.xlabel(f"log1p({target_col})")
    plt.ylabel("Frequency")
    plt.title(f"{target_col} conditioned on {condition_col}")
    plt.legend()
    plt.grid(True, linewidth=0.3)
    plt.tight_layout()
    plt.show()


def compute_spearman_correlations(df: pd.DataFrame,
                                   col_ent: str = "contenant enterré",
                            col_grand: str = "grand contenant",
                            col_petit: str = "petit contenant",) -> pd.DataFrame:
    """
    Compute Spearman correlations between container types (raw and log1p).
    """

    cols = {
        "enterré": df[col_ent],
        "grand": df[col_grand],
        "petit": df[col_petit],
    }

    results = []

    keys = list(cols.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            k1, k2 = keys[i], keys[j]

            rho_raw, _ = spearmanr(cols[k1], cols[k2])
            rho_log, _ = spearmanr(log1p_series(cols[k1]),
                                   log1p_series(cols[k2]))

            results.append({
                "var_1": k1,
                "var_2": k2,
                "spearman_raw": rho_raw,
                "spearman_log1p": rho_log,
            })

    return pd.DataFrame(results)

