from __future__ import annotations

import pandas as pd


class DataQualityError(ValueError):
    """Raised when a dataset fails minimum quality constraints."""


def validate_time_series_frame(df: pd.DataFrame, name: str) -> None:
    if df.empty:
        raise DataQualityError(f"[{name}] frame is empty")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataQualityError(f"[{name}] index must be DatetimeIndex")
    if not df.index.is_monotonic_increasing:
        raise DataQualityError(f"[{name}] index must be sorted ascending")
    if df.index.has_duplicates:
        raise DataQualityError(f"[{name}] duplicate timestamps found")

    all_nan_cols = df.columns[df.isna().all()].tolist()
    if all_nan_cols:
        raise DataQualityError(f"[{name}] all values are NaN for columns: {all_nan_cols}")


def missing_rate(df: pd.DataFrame) -> pd.Series:
    return df.isna().mean().sort_values(ascending=False)
