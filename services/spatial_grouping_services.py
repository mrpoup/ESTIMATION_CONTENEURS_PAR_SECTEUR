import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold
from services import modeles_services_regression

#
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
        true_sum = y_true[idx].sum()
        pred_sum = y_pred[idx].sum()
        rse[i] = abs(pred_sum - true_sum) / (true_sum + eps)

    return rse

def make_group_sums_and_means_knn(
    coords,
    y_true,
    y_pred,
    k=120,
    max_groups=600,
    seed=42
):
    """
    Build kNN groups and return:
      - true_sums, pred_sums
      - true_means, pred_means
    """
    true_sums, pred_sums = make_group_sums_knn(
        coords=coords,
        y_true=y_true,
        y_pred=y_pred,
        k=k,
        max_groups=max_groups,
        seed=seed
    )

    true_means = true_sums / k
    pred_means = pred_sums / k

    return true_sums, pred_sums, true_means, pred_means


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


