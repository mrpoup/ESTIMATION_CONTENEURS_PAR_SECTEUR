from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetArtifacts:
    """Container for prepared ML artifacts."""
    X: pd.DataFrame
    Y: pd.DataFrame
    ids: pd.Series
    feature_names: List[str]
    target_names: List[str]

    def _reset_col_names_frm_data(self):
        all_data_cols=self.X.columns
        feature_names=all_data_cols


class DataSchemaError(ValueError):
    """Raised when expected columns are missing or invalid."""
    pass


class BuildingsCountDataPreparationService:
    """
    Prepare building-level features/targets for count modeling.

    Responsibilities:
    - Validate schema and basic data quality
    - Create derived features (e.g., volume)
    - Convert buffer surfaces to ratios (composition features)
    - Add optional log1p transforms for heavy-tailed variables
    - Return X/Y consistently aligned and ready for CV
    """

    def __init__(
        self,
        id_col: str = "id_maison",
        buffer_defs: Optional[Dict[str, str]] = None,
    ) -> None:
        # buffer_defs maps suffix -> buffer_area_column
        self.id_col = id_col
        self.buffer_defs = buffer_defs or {
            "b10_m": "surf_buffer_m2_b10_m",
            "b50_m": "surf_buffer_m2_b50_m",
        }

    # ---------- Public API ----------

    def prepare_dataset(
        self,
        df: pd.DataFrame,
        features_col: List[str],
        targets_col: List[str],
        *,
        add_volume: bool = True,
        volume_col: str = "volume_batiment",
        volume_default_height: float = 5.0,
        make_ratios: bool = True,
        ratio_prefix: str = "ratio_",
        add_log1p: bool = True,
        log1p_cols: Optional[List[str]] = None,
        drop_original_buffer_surfaces_after_ratio: bool = False,
        keep_buffer_area_cols: bool = True,
        target_cols_to_keep: Optional[List[str]] = None,
    ) -> DatasetArtifacts:
        """
        Returns:
            DatasetArtifacts with X (features), Y (targets), ids, and names.
        """
        self._validate_columns(df, features_col, targets_col)

        # Work on a copy to avoid side-effects
        data = df.copy()

        # Ensure id column exists and is not null
        if data[self.id_col].isna().any():
            raise DataSchemaError(f"Null values found in id column '{self.id_col}'.")

        # Keep only required columns (union of features + targets, unique)
        cols_needed = list(dict.fromkeys(features_col + targets_col))
        data = data[cols_needed].copy()

        # Optional: derived feature volume
        if add_volume:
            data = self._add_building_volume(
                data,
                surf_col="surf_batiment_source_m2",
                height_col="hauteur_corrigee_m",
                out_col=volume_col,
                default_height=volume_default_height,
            )
            if volume_col not in features_col:
                features_col = features_col + [volume_col]

        # Optional: ratio features for each buffer
        if make_ratios:
            data, ratio_cols = self._add_buffer_ratio_features(
                data=data,
                features_col=features_col,
                ratio_prefix=ratio_prefix,
                drop_original_surfaces=drop_original_buffer_surfaces_after_ratio,
                keep_buffer_area_cols=keep_buffer_area_cols,
            )
            # Add ratio columns to features list
            features_col = features_col + ratio_cols

            # If requested, optionally drop original buffer surface columns
            # (Already handled in the helper when drop_original_surfaces=True)

        # Optional: log1p transforms
        if add_log1p:
            data, log_cols = self._add_log1p_features(
                data=data,
                features_col=features_col,
                log1p_cols=log1p_cols,
            )
            features_col = features_col + log_cols

        # Build final X, Y
        X = data[[c for c in features_col if c != self.id_col]].copy()

        # Decide which targets to keep (by default keep the 3 main ones if present)
        if target_cols_to_keep is None:
            default_main_targets = ["contenant enterré", "grand contenant", "petit contenant"]
            target_cols_to_keep = [c for c in default_main_targets if c in targets_col]

        Y = data[target_cols_to_keep].copy()

        # Basic sanitization: ensure numeric types for X/Y except id
        X = self._coerce_numeric_frame(X, frame_name="X")
        Y = self._coerce_numeric_frame(Y, frame_name="Y")

        # Replace +/-inf in X, then check
        X = X.replace([np.inf, -np.inf], np.nan)
        if X.isna().any().any():
            # Minimal approach: raise; later we can add a policy (impute/drop)
            nan_cols = X.columns[X.isna().any()].tolist()
            raise DataSchemaError(f"NaNs detected in X after preparation. Columns: {nan_cols}")

        ids = data[self.id_col].copy()

        return DatasetArtifacts(
            X=X,
            Y=Y,
            ids=ids,
            feature_names=list(X.columns),
            target_names=list(Y.columns),
        )

    # ---------- Internal helpers ----------

    def _validate_columns(self, df: pd.DataFrame, features_col: List[str], targets_col: List[str]) -> None:
        missing = [c for c in set(features_col + targets_col) if c not in df.columns]
        if missing:
            raise DataSchemaError(f"Missing columns in dataframe: {missing}")

        if self.id_col not in df.columns:
            raise DataSchemaError(f"Missing id column '{self.id_col}' in dataframe.")

    def _add_building_volume(
        self,
        data: pd.DataFrame,
        *,
        surf_col: str,
        height_col: str,
        out_col: str,
        default_height: float,
    ) -> pd.DataFrame:
        """Compute volume = surface * height (use default_height if height is null)."""
        if surf_col not in data.columns or height_col not in data.columns:
            raise DataSchemaError(f"Cannot compute volume: '{surf_col}' or '{height_col}' missing.")

        height = data[height_col].astype(float)
        height_filled = height.fillna(default_height).where(height.notna(), default_height)

        # If you also want to treat 0 height as default, uncomment the next line:
        # height_filled = height_filled.where(height_filled > 0, default_height)

        surface = data[surf_col].astype(float)
        data[out_col] = surface * height_filled
        return data

    def _add_buffer_ratio_features(
        self,
        data: pd.DataFrame,
        features_col: List[str],
        *,
        ratio_prefix: str,
        drop_original_surfaces: bool,
        keep_buffer_area_cols: bool,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        For each buffer suffix, convert surf_*_{suffix} columns into ratios by dividing by buffer area.

        Example:
            surf_feuillu_b10_m / surf_buffer_m2_b10_m -> ratio_surf_feuillu_b10_m
        """
        ratio_cols: List[str] = []

        for suffix, buffer_area_col in self.buffer_defs.items():
            if buffer_area_col not in data.columns:
                raise DataSchemaError(f"Buffer area column '{buffer_area_col}' missing.")

            # Identify all surface columns for this suffix (excluding the buffer area itself)
            suffix_cols = [c for c in features_col if c.endswith(f"_{suffix}") and c != buffer_area_col]

            # Create safe denominator (avoid division by 0)
            denom = data[buffer_area_col].astype(float)
            denom_safe = denom.where(denom > 0, np.nan)

            for c in suffix_cols:
                ratio_col = f"{ratio_prefix}{c}"
                data[ratio_col] = data[c].astype(float) / denom_safe
                ratio_cols.append(ratio_col)

            # Optionally drop original surface columns (but typically keep them during exploration)
            if drop_original_surfaces:
                data = data.drop(columns=suffix_cols, errors="ignore")

            # Optionally drop buffer area columns (I recommend keeping them)
            if not keep_buffer_area_cols:
                data = data.drop(columns=[buffer_area_col], errors="ignore")

        # After ratios: check bounds roughly (allow small eps due to geometry artifacts)
        # We do not enforce hard clipping here; better to inspect first.
        return data, ratio_cols

    def _add_log1p_features(
        self,
        data: pd.DataFrame,
        features_col: List[str],
        *,
        log1p_cols: Optional[List[str]],
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Add log1p transforms for selected heavy-tailed variables.
        If log1p_cols is None, we choose a conservative default set.
        """
        if log1p_cols is None:
            candidates = [
                "surf_batiment_source_m2",
                "volume_batiment",
                "surf_buffer_m2_b10_m",
                "surf_buffer_m2_b50_m",
            ]
            log1p_cols = [c for c in candidates if c in data.columns]

        created: List[str] = []
        for c in log1p_cols:
            out = f"log1p_{c}"
            # log1p requires non-negative; if negatives exist, raise
            if (data[c].astype(float) < 0).any():
                raise DataSchemaError(f"Negative values found in '{c}', cannot apply log1p.")
            data[out] = np.log1p(data[c].astype(float))
            created.append(out)

        return data, created

    def _coerce_numeric_frame(self, df: pd.DataFrame, *, frame_name: str) -> pd.DataFrame:
        out = df.copy()
        for c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="raise")
        return out
    

        