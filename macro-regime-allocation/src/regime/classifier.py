"""Rule-based macro regime classifier."""

from __future__ import annotations

import pandas as pd

from src.config.settings import RegimeThresholds


def classify_regime(
    composites: pd.DataFrame,
    growth_col: str = "growth",
    inflation_col: str = "inflation",
    thresholds: RegimeThresholds = RegimeThresholds(),
) -> pd.Series:
    """Classify macro regimes using growth/inflation z-score quadrants."""
    required = {growth_col, inflation_col}
    missing = required - set(composites.columns)
    if missing:
        raise KeyError(f"Missing columns for regime classification: {sorted(missing)}")

    g = composites[growth_col]
    i = composites[inflation_col]

    conditions = [
        (g >= thresholds.growth_high) & (i < thresholds.inflation_high),
        (g < thresholds.growth_high) & (i < thresholds.inflation_high),
        (g >= thresholds.growth_high) & (i >= thresholds.inflation_high),
        (g < thresholds.growth_high) & (i >= thresholds.inflation_high),
    ]
    labels = ["expansion", "disinflation_slowdown", "inflation_overheat", "stagflation_recession"]

    regime = pd.Series(index=composites.index, dtype="object")
    for cond, label in zip(conditions, labels):
        regime.loc[cond] = label

    return regime.ffill()
