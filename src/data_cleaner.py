"""
data_cleaner.py
─────────────────────────────────────────────────────────────────────
Clinical data cleaning with full audit trail.

Every transformation is logged to a CONSORT-style flow diagram and
an audit DataFrame — mandatory for methods transparency in peer review.

Usage:
    from src.data_cleaner import DataCleaner
    cleaner = DataCleaner(df, study_id="circumcision_outcomes_2024")
    df_clean = (
        cleaner
        .drop_duplicates(on="_notion_id")
        .enforce_types({"age": "int", "pain_score": "float", "group": "category"})
        .clip_outliers("age", lo=18, hi=90)
        .impute(strategy="median", columns=["bmi"])
        .flag_missing_threshold(threshold=0.3)
        .build()
    )
    cleaner.print_flow()          # CONSORT flow
    cleaner.audit_trail           # DataFrame of all transformations
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def _load_config() -> dict:
    cfg_path = Path(__file__).parents[1] / "config" / "settings.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)

CFG = _load_config()


class DataCleaner:
    """
    Fluent API for clinical data cleaning.
    All operations are non-destructive: each step works on a copy
    and appends to the audit trail.
    """

    def __init__(self, df: pd.DataFrame, study_id: str = "unnamed"):
        self._raw = df.copy()
        self._df = df.copy()
        self.study_id = study_id
        self._audit: list[dict] = []
        self._initial_n = len(df)
        self._record("initial", f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns")

    # ── DEDUPLICATION ────────────────────────────────────────────────────

    def drop_duplicates(self, on: str | list[str] | None = None) -> "DataCleaner":
        before = len(self._df)
        self._df = self._df.drop_duplicates(subset=on)
        dropped = before - len(self._df)
        self._record("deduplication", f"Dropped {dropped} duplicate rows (key: {on})")
        return self

    # ── TYPE ENFORCEMENT ─────────────────────────────────────────────────

    def enforce_types(self, schema: dict[str, str]) -> "DataCleaner":
        """
        schema: {'column_name': 'int' | 'float' | 'str' | 'category' | 'datetime' | 'bool'}
        """
        for col, dtype in schema.items():
            if col not in self._df.columns:
                logger.warning(f"Column '{col}' not found — skipping type enforcement")
                continue
            try:
                if dtype == "datetime":
                    self._df[col] = pd.to_datetime(self._df[col])
                elif dtype == "category":
                    self._df[col] = self._df[col].astype("category")
                elif dtype == "bool":
                    self._df[col] = self._df[col].astype(bool)
                else:
                    self._df[col] = pd.to_numeric(self._df[col], errors="coerce").astype(dtype)
                self._record("type_cast", f"Column '{col}' cast to {dtype}")
            except Exception as e:
                logger.error(f"Type cast failed for '{col}': {e}")
        return self

    # ── RANGE VALIDATION & CLIPPING ──────────────────────────────────────

    def clip_outliers(
        self,
        column: str,
        lo: float | None = None,
        hi: float | None = None,
        method: Literal["clip", "nullify"] = "nullify",
    ) -> "DataCleaner":
        """
        Values outside [lo, hi] are either clipped to boundary or set to NaN.
        'nullify' is preferred for peer review (transparent about exclusions).
        """
        mask = pd.Series([False] * len(self._df), index=self._df.index)
        if lo is not None:
            mask |= self._df[column] < lo
        if hi is not None:
            mask |= self._df[column] > hi
        n_affected = mask.sum()
        if method == "clip":
            self._df[column] = self._df[column].clip(lower=lo, upper=hi)
        else:
            self._df.loc[mask, column] = np.nan
        self._record(
            "outlier_handling",
            f"Column '{column}': {n_affected} values outside [{lo}, {hi}] → {method}",
        )
        return self

    # ── MISSING DATA ─────────────────────────────────────────────────────

    def flag_missing_threshold(self, threshold: float = 0.3) -> "DataCleaner":
        """
        Drop columns where missing > threshold.
        This is a design choice that must be disclosed in methods.
        """
        missing_frac = self._df.isna().mean()
        cols_to_drop = missing_frac[missing_frac > threshold].index.tolist()
        if cols_to_drop:
            self._df = self._df.drop(columns=cols_to_drop)
            self._record(
                "missing_column_exclusion",
                f"Dropped columns with >{threshold*100:.0f}% missing: {cols_to_drop}",
            )
        return self

    def drop_missing_rows(self, subset: list[str] | None = None) -> "DataCleaner":
        before = len(self._df)
        self._df = self._df.dropna(subset=subset)
        dropped = before - len(self._df)
        self._record(
            "row_exclusion_missing",
            f"Dropped {dropped} rows with missing values (subset: {subset})",
        )
        return self

    def impute(
        self,
        strategy: Literal["mean", "median", "mode", "constant"],
        columns: list[str] | None = None,
        fill_value: Any = None,
    ) -> "DataCleaner":
        """
        Simple imputation. For peer review, always disclose imputation strategy.
        Multiple imputation (MICE) is recommended for MAR data — use statsmodels.
        """
        cols = columns or self._df.select_dtypes(include="number").columns.tolist()
        for col in cols:
            n_missing = self._df[col].isna().sum()
            if n_missing == 0:
                continue
            if strategy == "mean":
                val = self._df[col].mean()
            elif strategy == "median":
                val = self._df[col].median()
            elif strategy == "mode":
                val = self._df[col].mode()[0]
            else:
                val = fill_value
            self._df[col] = self._df[col].fillna(val)
            self._record(
                "imputation",
                f"Column '{col}': {n_missing} missing → imputed with {strategy} ({val:.3f})",
            )
        return self

    # ── EXCLUSION CRITERIA ────────────────────────────────────────────────

    def exclude(self, query: str, reason: str) -> "DataCleaner":
        """
        Apply a pandas query-string exclusion criterion with explicit reason.
        Example: cleaner.exclude("age < 18", "minors excluded per protocol")
        """
        before = len(self._df)
        excluded = self._df.query(query)
        self._df = self._df.drop(index=excluded.index)
        dropped = before - len(self._df)
        self._record("exclusion", f"Excluded {dropped} rows: '{query}' ({reason})")
        return self

    # ── BUILD ────────────────────────────────────────────────────────────

    def build(self) -> pd.DataFrame:
        self._record(
            "final",
            f"Final dataset: {len(self._df)} rows ({self._initial_n - len(self._df)} excluded overall)",
        )
        # Save cleaned copy
        out_dir = Path(CFG["project"]["data_dir"]) / "processed"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"{self.study_id}_clean_{ts}.parquet"
        self._df.to_parquet(path, index=False)
        logger.info(f"Clean dataset saved → {path}")
        return self._df.copy()

    # ── AUDIT & REPORTING ────────────────────────────────────────────────

    @property
    def audit_trail(self) -> pd.DataFrame:
        return pd.DataFrame(self._audit)

    def print_flow(self) -> None:
        """Print a CONSORT-style participant flow."""
        print(f"\n{'─'*60}")
        print(f"  CONSORT-STYLE DATA FLOW — {self.study_id}")
        print(f"{'─'*60}")
        for step in self._audit:
            print(f"  [{step['step'].upper():30s}] {step['description']}")
        print(f"{'─'*60}\n")

    def _record(self, step: str, description: str) -> None:
        self._audit.append({
            "step": step,
            "description": description,
            "n_remaining": len(self._df),
            "timestamp": datetime.now().isoformat(),
        })
        logger.info(f"  [{step}] {description} (n={len(self._df)})")
