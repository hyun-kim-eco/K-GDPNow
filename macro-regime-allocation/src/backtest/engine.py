"""Basic monthly backtesting engine for regime allocations."""

from __future__ import annotations

import pandas as pd


def run_monthly_backtest(
    asset_returns: pd.DataFrame,
    target_weights: pd.DataFrame,
    transaction_cost_bps: float = 10.0,
) -> pd.DataFrame:
    """Calculate net portfolio returns with turnover-based transaction costs."""
    aligned_w = target_weights.reindex(asset_returns.index).ffill().fillna(0.0)
    aligned_r = asset_returns.reindex(columns=aligned_w.columns).fillna(0.0)

    shifted_w = aligned_w.shift(1).fillna(0.0)
    gross = (shifted_w * aligned_r).sum(axis=1)

    turnover = aligned_w.diff().abs().sum(axis=1)
    costs = turnover * (transaction_cost_bps / 10_000)
    net = gross - costs

    wealth = (1 + net).cumprod()
    return pd.DataFrame({"gross_return": gross, "net_return": net, "turnover": turnover, "wealth": wealth})
