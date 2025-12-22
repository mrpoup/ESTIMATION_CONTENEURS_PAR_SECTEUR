import numpy as np
import pandas as pd
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

def evaluate_aggregated_prediction(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_sizes: list[int],
    n_draws: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    ...



def aggregated_error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_size: int,
    n_draws: int,
    seed: int = 42,
    eps: float = 1e-9,
) -> Dict[str, float]:
    """
    Draw random groups of size `group_size` from the provided arrays (without replacement),
    aggregate sums over each group, and compute relative sum error distribution.

    Returns summary stats for the Relative Sum Error (RSE):
        RSE = |Sum(pred) - Sum(true)| / (Sum(true) + eps)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    n = y_true.shape[0]
    if group_size <= 0 or group_size > n:
        raise ValueError(f"group_size must be in [1, {n}]. Got {group_size}.")

    rng = np.random.default_rng(seed)

    rse = np.empty(n_draws, dtype=float)
    bias = np.empty(n_draws, dtype=float)

    for i in range(n_draws):
        idx = rng.choice(n, size=group_size, replace=False)
        s_true = float(y_true[idx].sum())
        s_pred = float(y_pred[idx].sum())

        rse[i] = abs(s_pred - s_true) / (s_true + eps)
        bias[i] = (s_pred - s_true) / (s_true + eps)

    return {
        "mean_rse": float(np.mean(rse)),
        "median_rse": float(np.median(rse)),
        "p90_rse": float(np.quantile(rse, 0.90)),
        "p95_rse": float(np.quantile(rse, 0.95)),
        "mean_bias": float(np.mean(bias)),
        "median_bias": float(np.median(bias)),
        "p90_abs_bias": float(np.quantile(np.abs(bias), 0.90)),
        "p95_abs_bias": float(np.quantile(np.abs(bias), 0.95)),
    }


def evaluate_aggregation_for_fold(
    y_true_fold: np.ndarray,
    y_pred_fold: np.ndarray,
    pks: List[int],
    p_n_draws: int,
    seed_base: int,
) -> pd.DataFrame:
    """
    Evaluate aggregation error for one fold across multiple group sizes (pks).
    Returns a tidy DataFrame with one row per group_size.
    """
    rows = []
    for k in pks:
        stats = aggregated_error_distribution(
            y_true=y_true_fold,
            y_pred=y_pred_fold,
            group_size=k,
            n_draws=p_n_draws,
            seed=seed_base + 100 * int(k),
        )
        rows.append({"group_size": k, **stats})
    return pd.DataFrame(rows)

