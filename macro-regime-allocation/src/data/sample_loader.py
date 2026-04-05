"""Sample data loader and publication-lag alignment utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class IndicatorSpec:
    """Metadata describing a macro indicator."""

    name: str
    release_lag_months: int = 1


def load_monthly_panel(path: str | Path, date_col: str = "date") -> pd.DataFrame:
    """Load a monthly panel from CSV and normalize the datetime index."""
    df = pd.read_csv(path)
    df[date_col] = pd.to_datetime(df[date_col])
    return df.sort_values(date_col).set_index(date_col)


def align_with_release_lags(df: pd.DataFrame, specs: list[IndicatorSpec]) -> pd.DataFrame:
    """Shift each series by its publication lag to emulate real-time observability."""
    aligned = pd.DataFrame(index=df.index)
    for spec in specs:
        if spec.name not in df.columns:
            raise KeyError(f"Missing indicator column: {spec.name}")
        aligned[spec.name] = df[spec.name].shift(spec.release_lag_months)
    return aligned
