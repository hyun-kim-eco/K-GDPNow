from pathlib import Path
import sys

import pandas as pd
import pytest

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
    zscores = rolling_zscore(transformed, window=12, min_periods=6)
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
    dates = pd.date_range("2024-01-31", periods=12, freq="ME")
    returns = pd.DataFrame(
        {
            "equity": [0.01, -0.02, 0.03, 0.01, 0.02, -0.01, 0.00, 0.02, -0.01, 0.01, 0.01, 0.00],
            "bond_long": [0.005, 0.004, -0.001, 0.006, 0.003, 0.002, 0.001, -0.002, 0.002, 0.003, 0.001, 0.002],
            "commodity": [0.0, 0.02, -0.01, 0.01, 0.01, 0.00, -0.01, 0.02, 0.01, -0.02, 0.01, 0.00],
            "gold": [0.01, 0.0, 0.005, -0.002, 0.003, 0.002, 0.001, -0.001, 0.002, 0.004, 0.0, 0.001],
            "usd": [0.0, 0.001, 0.002, -0.001, 0.0, 0.002, -0.001, 0.001, 0.002, -0.001, 0.0, 0.001],
        },
        index=dates,
    )
    regimes = pd.Series(["expansion", "inflation_overheat", "disinflation_slowdown", "stagflation_recession"] * 3, index=dates)

    weights = target_weights(regimes, DEFAULT_REGIME_WEIGHTS)
    out = run_monthly_backtest(returns, weights, transaction_cost_bps=10)

    assert set(out.columns) == {"gross_return", "net_return", "turnover", "wealth"}
    assert out["wealth"].iloc[-1] > 0


def test_report_outputs(tmp_path: Path) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    assert matplotlib is not None

    from src.visualization.report_builder import build_mvp_report

    dates = pd.date_range("2024-01-31", periods=12, freq="ME")
    composites = pd.DataFrame(
        {
            "growth": [0.2, -0.1, 0.5, -0.4] * 3,
            "inflation": [-0.2, 0.3, 0.1, -0.5] * 3,
            "liquidity": [0.0, 0.1, -0.1, 0.2] * 3,
            "sentiment": [0.1, -0.1, 0.2, -0.2] * 3,
        },
        index=dates,
    )
    regimes = pd.Series(["expansion", "inflation_overheat", "disinflation_slowdown", "stagflation_recession"] * 3, index=dates)

    returns = pd.DataFrame(0.001, index=dates, columns=["equity", "bond_long", "commodity", "gold", "usd"])
    weights = target_weights(regimes, DEFAULT_REGIME_WEIGHTS)
    backtest = run_monthly_backtest(returns, weights, transaction_cost_bps=10)

    report_path = build_mvp_report(composites, regimes, backtest, tmp_path)
    assert report_path.exists()
    assert (tmp_path / "composite_scores.png").exists()
    assert (tmp_path / "regime_distribution.png").exists()
    assert (tmp_path / "wealth_curve.png").exists()
