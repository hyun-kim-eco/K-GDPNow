"""Run an end-to-end MVP workflow on sample data."""

from __future__ import annotations

from pathlib import Path

from src.backtest.engine import run_monthly_backtest
from src.config.settings import DEFAULT_REGIME_WEIGHTS, MacroGroups
from src.data.sample_loader import IndicatorSpec, align_with_release_lags, load_monthly_panel
from src.features.zscore import composite_scores, pct_change_transform, rolling_zscore
from src.regime.classifier import classify_regime
from src.strategy.allocation import target_weights


def main() -> None:
    root = Path(__file__).resolve().parent
    macro = load_monthly_panel(root / "data" / "raw" / "sample_macro.csv")

    specs = [
        IndicatorSpec("ip", 1),
        IndicatorSpec("retail", 1),
        IndicatorSpec("employment", 1),
        IndicatorSpec("cpi", 1),
        IndicatorSpec("core_cpi", 1),
        IndicatorSpec("ppi", 1),
        IndicatorSpec("credit_spread", 0),
        IndicatorSpec("policy_rate", 0),
        IndicatorSpec("m2", 1),
        IndicatorSpec("pmi", 0),
        IndicatorSpec("consumer_sentiment", 0),
    ]

    aligned = align_with_release_lags(macro, specs)
    transformed = pct_change_transform(aligned[["ip", "retail", "employment", "cpi", "core_cpi", "ppi", "m2"]])
    transformed["credit_spread"] = aligned["credit_spread"]
    transformed["policy_rate"] = aligned["policy_rate"]
    transformed["pmi"] = aligned["pmi"]
    transformed["consumer_sentiment"] = aligned["consumer_sentiment"]

    zscores = rolling_zscore(transformed, window=12, min_periods=6)
    groups = MacroGroups()
    composites = composite_scores(
        zscores,
        {
            "growth": groups.growth,
            "inflation": groups.inflation,
            "liquidity": groups.liquidity,
            "sentiment": groups.sentiment,
        },
    )
    regimes = classify_regime(composites)

    asset_returns = load_monthly_panel(root / "data" / "raw" / "sample_assets.csv")
    weights = target_weights(regimes, DEFAULT_REGIME_WEIGHTS)
    backtest = run_monthly_backtest(asset_returns, weights, transaction_cost_bps=10)

    output = root / "data" / "processed"
    output.mkdir(parents=True, exist_ok=True)
    composites.to_csv(output / "composites.csv")
    regimes.rename("regime").to_csv(output / "regimes.csv")
    backtest.to_csv(output / "backtest_results.csv")

    print("Saved outputs:")
    print(f"- {output / 'composites.csv'}")
    print(f"- {output / 'regimes.csv'}")
    print(f"- {output / 'backtest_results.csv'}")


if __name__ == "__main__":
    main()
