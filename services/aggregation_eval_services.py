from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np


@dataclass(frozen=True)
class AggregationDraws:
    rse: np.ndarray
    bias: np.ndarray


def draw_aggregated_errors(
    y_true,
    y_pred,
    group_size: int,
    n_draws: int = 1000,
    seed: int = 42,
    eps: float = 1e-9,
) -> AggregationDraws:
    """
    Draw random groups (without replacement) and return arrays of RSE and bias.
    RSE = |S_pred - S_true| / (S_true + eps)
    bias = (S_pred - S_true) / (S_true + eps)
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
        denom = s_true + eps
        rse[i] = abs(s_pred - s_true) / denom
        bias[i] = (s_pred - s_true) / denom

    return AggregationDraws(rse=rse, bias=bias)

def summarize_aggregation_draws(draws: AggregationDraws) -> Dict[str, float]:
    rse = draws.rse
    bias = draws.bias

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
