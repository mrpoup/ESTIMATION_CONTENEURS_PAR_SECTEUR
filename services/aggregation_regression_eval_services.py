import numpy as np
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