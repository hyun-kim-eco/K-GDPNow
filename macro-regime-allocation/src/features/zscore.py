"""Feature engineering utilities for macro regime signals."""

from __future__ import annotations

from typing import Dict

import pandas as pd


def pct_change_transform(df: pd.DataFrame, periods: int = 12, suffix: str = "_yoy") -> pd.DataFrame:
    """Apply percentage-change transform for each column."""
    transformed = df.pct_change(periods=periods) * 100.0
    transformed.columns = [f"{col}{suffix}" for col in transformed.columns]
    return transformed


def rolling_zscore(df: pd.DataFrame, window: int = 36, min_periods: int = 24) -> pd.DataFrame:
    """Compute rolling z-scores using backward-looking statistics only."""
    mean = df.rolling(window=window, min_periods=min_periods).mean()
    std = df.rolling(window=window, min_periods=min_periods).std(ddof=0)
    z = (df - mean) / std.replace(0, pd.NA)
    return z


def composite_scores(zscores: pd.DataFrame, grouped_columns: Dict[str, list[str]]) -> pd.DataFrame:
    """Build grouped composite indices by averaging z-scores within each group."""
    out = pd.DataFrame(index=zscores.index)
    for group_name, cols in grouped_columns.items():
        valid_cols = [col for col in cols if col in zscores.columns]
        if not valid_cols:
            raise ValueError(f"Group '{group_name}' has no valid columns in z-score table.")
        out[group_name] = zscores[valid_cols].mean(axis=1)
    return out
