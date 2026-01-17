import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.stats import pearsonr


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

def log1p_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    # log1p: OK pour 0 et positifs, mais si tu as des valeurs négatives -> NaN
    s = s.where(s >= 0)
    return np.log1p(s)

def compute_pearson_correlations(
    df: pd.DataFrame,
    col_ent: str = "contenant enterré",
    col_grand: str = "grand contenant",
    col_petit: str = "petit contenant",
) -> pd.DataFrame:
    """
    Compute Pearson correlations between container types (raw and log1p).
    Returns r and p-value for each pair.
    """

    cols = {
        "enterré": pd.to_numeric(df[col_ent], errors="coerce"),
        "grand": pd.to_numeric(df[col_grand], errors="coerce"),
        "petit": pd.to_numeric(df[col_petit], errors="coerce"),
    }

    results = []
    keys = list(cols.keys())

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            k1, k2 = keys[i], keys[j]

            # RAW (drop NaN pairwise)
            x = cols[k1]
            y = cols[k2]
            mask = x.notna() & y.notna()
            if mask.sum() >= 2:
                r_raw, p_raw = pearsonr(x[mask], y[mask])
            else:
                r_raw, p_raw = np.nan, np.nan

            # LOG1P (re-drop NaN after transform)
            xl = log1p_series(x)
            yl = log1p_series(y)
            maskl = xl.notna() & yl.notna()
            if maskl.sum() >= 2:
                r_log, p_log = pearsonr(xl[maskl], yl[maskl])
            else:
                r_log, p_log = np.nan, np.nan

            results.append({
                "var_1": k1,
                "var_2": k2,
                "n_raw": int(mask.sum()),
                "pearson_raw": r_raw,
                "pvalue_raw": p_raw,
                "n_log1p": int(maskl.sum()),
                "pearson_log1p": r_log,
                "pvalue_log1p": p_log,
            })

    return pd.DataFrame(results)