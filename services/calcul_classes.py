from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Literal

import numpy as np
import pandas as pd


BinningMethod = Literal["quantile", "thresholds", "balanced_integers"]


@dataclass(frozen=True)
class BinningSpec:
    """Configuration for binning a numeric target into ordinal classes."""
    method: BinningMethod = "balanced_integers"
    n_classes: int = 5

    # If method == "thresholds", thresholds must be provided (ascending).
    # Example: thresholds=[0, 3, 10] => classes: (-inf,0], (0,3], (3,10], (10,inf)
    thresholds: Optional[List[float]] = None

    # Optional handling of zeros as a dedicated class 0 (useful for zero-inflated targets).
    # If True, all zeros become class 0, and remaining positives are binned into (n_classes-1) classes.
    zero_as_own_class: bool = False

    # If True, bin positive values on log1p scale before computing quantiles (often stabilizes heavy tails).
    log1p_before_quantiles: bool = False

    # If True, output labels 0..K-1 (recommended). Otherwise, output 1..K.
    zero_based_labels: bool = True

    # If True, include right edge in bins (like pandas.cut right=True).
    right_closed: bool = True

        # If method == "balanced_integers", classes are built as contiguous integer ranges
    # attempting to balance sample counts per class without splitting any integer value.
    min_class_count: int = 1  # safety: avoid empty classes
    max_classes_if_collapse: bool = True  # if ties prevent K classes, allow fewer


@dataclass
class BinningArtifacts:
    """Outputs of the binning service."""
    y_class: pd.Series
    column_name: str
    log: Dict[str, Any]


class TargetBinningService:
    """Service to convert a count target into ordinal classes (bins)."""

    def __init__(self, series_name_fallback: str = "target"):
        self.series_name_fallback = series_name_fallback

    def build_classes(
        self,
        y: pd.Series,
        spec: BinningSpec = BinningSpec(),
        column_prefix: str = "cls",
        column_name: Optional[str] = None,
    ) -> BinningArtifacts:
        if not isinstance(y, pd.Series):
            raise TypeError("y must be a pandas Series.")

        y_name = y.name if y.name is not None else self.series_name_fallback
        y_clean = pd.to_numeric(y, errors="coerce")

        if y_clean.isna().any():
            # We choose to fail fast: missing target labels usually means upstream issues.
            n_nan = int(y_clean.isna().sum())
            raise ValueError(f"Target contains NaN after coercion: {n_nan} missing values.")

        if spec.n_classes < 2:
            raise ValueError("n_classes must be >= 2.")

        # Build column name proposal
        if column_name is None:
            if spec.method == "quantile":
                            column_name = f"{column_prefix}_q{spec.n_classes}_{self._slugify(y_name)}"
            elif spec.method == "thresholds":
                column_name = f"{column_prefix}_thr{spec.n_classes}_{self._slugify(y_name)}"
            else:
                column_name = f"{column_prefix}_bi{spec.n_classes}_{self._slugify(y_name)}"

            if spec.zero_as_own_class:
                column_name = f"{column_name}_z0"

            if spec.log1p_before_quantiles and spec.method == "quantile":
                column_name = f"{column_name}_log1p"

        if spec.method == "quantile":
            y_class, edges = self._bin_quantiles(y_clean, spec)
            method_details = {
                spec.method+'_'+"edges": edges,
                "effective_n_classes": int(y_class.max() - y_class.min() + 1),
            }
        elif spec.method == "thresholds":
            y_class, edges = self._bin_thresholds(y_clean, spec)
            method_details = {
                spec.method+'_'+"edges": edges,
                "effective_n_classes": int(y_class.max() - y_class.min() + 1),
            }
        elif spec.method == "balanced_integers":
            y_class, edges = self._bin_balanced_integers(y_clean, spec)
            method_details = {
                spec.method+'_'+"edges": edges,  # list of (low_int, high_int) inclusive
                "effective_n_classes": int(y_class.max() - y_class.min() + 1),
            }
        else:
            raise ValueError(f"Unknown method: {spec.method}")

        # Build log
        log = self._build_log(y_clean, y_class, spec, column_name, method_details)

        return BinningArtifacts(
            y_class=y_class.rename(column_name),
            column_name=column_name,
            log=log,
        )

    # ------------------------
    # Binning implementations
    # ------------------------

    def _bin_quantiles(self, y: pd.Series, spec: BinningSpec) -> Tuple[pd.Series, List[float]]:
        if spec.zero_as_own_class:
            # Class 0 for zeros; bin strictly positive values into (n_classes-1) quantile classes.
            y_class = pd.Series(index=y.index, dtype="int64")
            is_zero = (y == 0)
            is_pos = (y > 0)

            y_class.loc[is_zero] = 0

            n_pos_classes = spec.n_classes - 1
            if n_pos_classes < 1:
                raise ValueError("With zero_as_own_class=True, n_classes must be >= 2.")

            y_pos = y.loc[is_pos]
            if len(y_pos) == 0:
                # All zeros => single class
                y_class.loc[is_pos] = 0
                edges = [0.0]
                return self._finalize_labels(y_class, spec), edges

            values_for_q = np.log1p(y_pos.to_numpy()) if spec.log1p_before_quantiles else y_pos.to_numpy()

            # Compute quantile edges (include 0% and 100% as well)
            q = np.linspace(0.0, 1.0, n_pos_classes + 1)
            raw_edges = np.quantile(values_for_q, q)

            # Ensure strict monotonicity of edges (handle duplicates due to discreteness)
            edges = self._make_strictly_increasing(raw_edges)

            # Bin positives into 1..n_pos_classes (then adjust to 0-based/1-based later)
            # Use numpy digitize for speed and control
            pos_bins = np.digitize(values_for_q, bins=edges[1:-1], right=spec.right_closed)  # 0..n_pos_classes-1
            y_class.loc[is_pos] = pos_bins + 1  # 1..n_pos_classes

            # For reporting, edges are reported on original scale (if log1p used)
            if spec.log1p_before_quantiles:
                edges_report = [float(np.expm1(e)) for e in edges]
            else:
                edges_report = [float(e) for e in edges]

            return self._finalize_labels(y_class, spec), edges_report

        # Standard quantile binning on full y
        values_for_q = np.log1p(y.to_numpy()) if spec.log1p_before_quantiles else y.to_numpy()
        q = np.linspace(0.0, 1.0, spec.n_classes + 1)
        raw_edges = np.quantile(values_for_q, q)
        edges = self._make_strictly_increasing(raw_edges)

        bins = np.digitize(values_for_q, bins=edges[1:-1], right=spec.right_closed)  # 0..n_classes-1
        y_class = pd.Series(bins, index=y.index, dtype="int64")

        if spec.log1p_before_quantiles:
            edges_report = [float(np.expm1(e)) for e in edges]
        else:
            edges_report = [float(e) for e in edges]

        return self._finalize_labels(y_class, spec), edges_report

    def _bin_thresholds(self, y: pd.Series, spec: BinningSpec) -> Tuple[pd.Series, List[float]]:
        if spec.thresholds is None or len(spec.thresholds) == 0:
            raise ValueError("thresholds must be provided when method='thresholds'.")

        thr = [float(t) for t in spec.thresholds]
        thr = sorted(thr)

        # edges = [-inf, t1, t2, ..., +inf] conceptually; we keep finite edges for logs
        bins = np.digitize(y.to_numpy(), bins=thr, right=spec.right_closed)  # 0..len(thr)
        y_class = pd.Series(bins, index=y.index, dtype="int64")

        edges_report = thr[:]  # user-provided
        return self._finalize_labels(y_class, spec), edges_report
    

    def _bin_balanced_integers(self, y: pd.Series, spec: BinningSpec) -> Tuple[pd.Series, List[float]]:
        """
        Balanced integer binning with continuous coverage:
        - Classes are contiguous integer intervals covering [min(y), +inf)
        - No holes even for unseen integer values
        - Returns thresholds as upper bounds + [inf] for logging.
        """
        if (y % 1 != 0).any():
            raise ValueError("balanced_integers requires integer-valued target (counts).")

        y_int = y.astype(int)

        # Optional: reserve class for zeros
        if spec.zero_as_own_class:
            y_class = pd.Series(index=y_int.index, dtype="int64")
            is_zero = (y_int == 0)
            is_pos = ~is_zero

            y_class.loc[is_zero] = 0

            y_pos = y_int.loc[is_pos]
            if len(y_pos) == 0:
                edges = [0.0, float(np.inf)]
                return self._finalize_labels(y_class.fillna(0), spec), edges

            pos_spec = BinningSpec(
                method="balanced_integers",
                n_classes=spec.n_classes - 1,
                thresholds=None,
                zero_as_own_class=False,
                log1p_before_quantiles=False,
                zero_based_labels=True,
                right_closed=spec.right_closed,
                min_class_count=spec.min_class_count,
                max_classes_if_collapse=spec.max_classes_if_collapse,
            )

            y_pos_class, pos_edges = self._bin_balanced_integers(y_pos, pos_spec)
            y_class.loc[is_pos] = y_pos_class + 1

            y_class_final = self._finalize_labels(y_class, spec)

            # Prepend 0 as an explicit first threshold edge for logging clarity
            # (Here edges are upper bounds + inf)
            edges = [0.0] + [e for e in pos_edges if np.isfinite(e)] + [float(np.inf)]
            # Remove duplicates while preserving order
            edges = list(dict.fromkeys(edges))
            return y_class_final, edges

        # --- Build a continuous histogram from 0..vmax (or min..vmax if you prefer) ---
        vmin = int(y_int.min())
        vmax = int(y_int.max())

        if vmin < 0:
            raise ValueError("balanced_integers expects non-negative counts.")

        # count occurrences for each integer in [0..vmax]
        counts = np.bincount(y_int.to_numpy(), minlength=vmax + 1)  # index=v
        total = int(counts.sum())

        # Effective number of classes cannot exceed number of non-empty integer values,
        # but we still can create thresholds on empty integers, so we keep spec.n_classes as target.
        k_eff = spec.n_classes

        target_per_class = total / float(k_eff)

        thresholds: List[int] = []
        acc = 0
        classes_built = 0

        # We will place K-1 thresholds (upper bounds), then last class is (last_threshold, +inf)
        for v in range(vmin, vmax + 1):
            acc += int(counts[v])

            remaining_classes = (k_eff - 1) - classes_built
            if remaining_classes <= 0:
                break

            # We only consider cutting if current class has at least min_class_count samples
            if acc < spec.min_class_count:
                continue

            if acc >= target_per_class:
                thresholds.append(v)
                classes_built += 1
                acc = 0

        # If we ended up with too few thresholds due to heavy ties, we accept fewer classes if allowed
        if len(thresholds) == 0 and k_eff > 1 and not spec.max_classes_if_collapse:
            raise ValueError("Could not create any threshold; consider max_classes_if_collapse=True or reduce n_classes.")

        # Ensure thresholds are strictly increasing
        thresholds = sorted(set(thresholds))

        # Build y_class via digitize on thresholds
        # bins are the finite thresholds; last class is values > last_threshold
        bins = np.digitize(y_int.to_numpy(), bins=np.array(thresholds, dtype=int), right=spec.right_closed)
        y_class = pd.Series(bins, index=y_int.index, dtype="int64")

        y_class = self._finalize_labels(y_class, spec)

        # For logging / downstream application on unseen data: provide edges with +inf
        edges_with_inf = [float(t) for t in thresholds] + [float(np.inf)]

        return y_class, edges_with_inf



    # ------------------------
    # Helpers
    # ------------------------

    def _finalize_labels(self, y_class: pd.Series, spec: BinningSpec) -> pd.Series:
        if spec.zero_based_labels:
            # Keep as is (already 0-based in both implementations)
            return y_class.astype("int64")
        # Convert to 1-based labels
        return (y_class + 1).astype("int64")

    def _build_log(
        self,
        y: pd.Series,
        y_class: pd.Series,
        spec: BinningSpec,
        column_name: str,
        method_details: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Class counts
        vc = y_class.value_counts(dropna=False).sort_index()
        class_counts = {int(k): int(v) for k, v in vc.items()}

        # Target descriptive stats
        y_desc = y.describe(percentiles=[0.25, 0.5, 0.75]).to_dict()
        y_desc = {k: float(v) for k, v in y_desc.items()}

        log: Dict[str, Any] = {
            "column_name": column_name,
            "series_name": y.name,
            "spec": asdict(spec),
            "target_describe": y_desc,
            "class_counts": class_counts,
            "n_samples": int(len(y)),
            "n_unique_target_values": int(y.nunique()),
        }
        log.update(method_details)
        return log

    def _make_strictly_increasing(self, edges: np.ndarray, min_step: float = 1e-12) -> np.ndarray:
        """Ensure edges are strictly increasing (needed for discrete / heavy-tied distributions)."""
        edges = np.asarray(edges, dtype=float).copy()
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + min_step
        return edges

    def _slugify(self, s: str) -> str:
        s = str(s).strip().lower()
        for ch in [" ", "/", "\\", "-", "’", "'", "(", ")", "[", "]", "{", "}", ",", ";", ":"]:
            s = s.replace(ch, "_")
        while "__" in s:
            s = s.replace("__", "_")
        return s.strip("_")
