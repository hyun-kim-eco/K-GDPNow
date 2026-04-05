from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.backtest.engine import run_monthly_backtest
from src.config.settings import DEFAULT_REGIME_WEIGHTS, MacroGroups
from src.data.sample_loader import IndicatorSpec, align_with_release_lags, load_monthly_panel
from src.features.zscore import composite_scores, pct_change_transform, rolling_zscore
from src.regime.classifier import classify_regime
from src.strategy.allocation import target_weights


def test_end_to_end_shapes() -> None:
    macro = load_monthly_panel(ROOT / "data" / "raw" / "sample_macro.csv")
    specs = [IndicatorSpec("ip", 1), IndicatorSpec("retail", 1), IndicatorSpec("employment", 1), IndicatorSpec("cpi", 1)]
    aligned = align_with_release_lags(macro, specs)

    transformed = pct_change_transform(aligned[["ip", "retail", "employment", "cpi"]], periods=1, suffix="_mom")
    zscores = rolling_zscore(transformed, window=6, min_periods=3)
    groups = MacroGroups(growth=["ip_mom", "retail_mom", "employment_mom"], inflation=["cpi_mom"], liquidity=[], sentiment=[])

    composites = composite_scores(
        zscores,
        {
            "growth": groups.growth,
            "inflation": groups.inflation,
        },
    )
    regimes = classify_regime(composites)
    assert regimes.notna().sum() > 0


def test_backtest_outputs() -> None:
    dates = pd.date_range("2024-01-31", periods=4, freq="M")
    returns = pd.DataFrame(
        {
            "equity": [0.01, -0.02, 0.03, 0.01],
            "bond_long": [0.005, 0.004, -0.001, 0.006],
            "commodity": [0.0, 0.02, -0.01, 0.01],
            "gold": [0.01, 0.0, 0.005, -0.002],
            "usd": [0.0, 0.001, 0.002, -0.001],
        },
        index=dates,
    )
    regimes = pd.Series(["expansion", "inflation_overheat", "disinflation_slowdown", "stagflation_recession"], index=dates)

    weights = target_weights(regimes, DEFAULT_REGIME_WEIGHTS)
    out = run_monthly_backtest(returns, weights, transaction_cost_bps=10)

    assert set(out.columns) == {"gross_return", "net_return", "turnover", "wealth"}
    assert out["wealth"].iloc[-1] > 0
