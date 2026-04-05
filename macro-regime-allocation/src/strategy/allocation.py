"""Regime-to-allocation mapping logic."""

from __future__ import annotations

import pandas as pd

from src.config.settings import DEFAULT_REGIME_WEIGHTS


def target_weights(
    regimes: pd.Series,
    weight_map: dict[str, dict[str, float]] | None = None,
    default_regime: str = "disinflation_slowdown",
) -> pd.DataFrame:
    """Map each timestamped regime label to target asset weights."""
    mapping = weight_map or DEFAULT_REGIME_WEIGHTS
    cleaned = regimes.ffill().bfill().fillna(default_regime)

    rows = []
    for dt, regime in cleaned.items():
        if regime not in mapping:
            raise KeyError(f"No weight rule for regime '{regime}'")
        row = pd.Series(mapping[regime], name=dt)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.index.name = regimes.index.name
    return out.fillna(0.0)
