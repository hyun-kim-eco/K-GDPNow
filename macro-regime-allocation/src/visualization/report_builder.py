"""Utilities for creating a readable MVP report with charts and tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _get_plt():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError:
        return None
    return plt


def _save_composite_chart(composites: pd.DataFrame, path: Path) -> bool:
    plt = _get_plt()
    if plt is None:
        return False
    fig, ax = plt.subplots(figsize=(11, 4.5))
    composites.plot(ax=ax, linewidth=1.5)
    ax.set_title("Macro Composite Scores (Growth/Inflation/Liquidity/Sentiment)")
    ax.set_ylabel("z-score")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def _save_wealth_chart(backtest: pd.DataFrame, path: Path) -> bool:
    plt = _get_plt()
    if plt is None:
        return False
    fig, ax = plt.subplots(figsize=(11, 4.5))
    backtest["wealth"].plot(ax=ax, color="tab:blue", linewidth=1.8)
    ax.set_title("Backtest Cumulative Wealth")
    ax.set_ylabel("wealth index")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def _save_regime_distribution(regimes: pd.Series, path: Path) -> bool:
    plt = _get_plt()
    if plt is None:
        return False
    fig, ax = plt.subplots(figsize=(9, 4.5))
    regimes.value_counts().sort_values(ascending=False).plot(kind="bar", ax=ax, color="tab:green")
    ax.set_title("Regime Observation Count")
    ax.set_ylabel("months")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def _regime_performance_table(backtest: pd.DataFrame, regimes: pd.Series) -> pd.DataFrame:
    joined = backtest.join(regimes.rename("regime"), how="inner")
    grouped = joined.groupby("regime")

    table = pd.DataFrame(
        {
            "months": grouped.size(),
            "avg_monthly_return": grouped["net_return"].mean(),
            "volatility": grouped["net_return"].std(ddof=0),
            "win_rate": grouped["net_return"].apply(lambda x: (x > 0).mean()),
            "max_drawdown": grouped["wealth"].apply(lambda x: (x / x.cummax() - 1).min()),
        }
    )
    table["sharpe_proxy"] = table["avg_monthly_return"] / table["volatility"].replace(0, pd.NA)
    return table.sort_values("avg_monthly_return", ascending=False)


def _annual_table(backtest: pd.DataFrame) -> pd.DataFrame:
    annual = (1 + backtest["net_return"]).groupby(backtest.index.year).prod() - 1
    return annual.rename("annual_return").to_frame()




def _to_markdown_table(df: pd.DataFrame) -> str:
    columns = [str(c) for c in df.reset_index().columns]
    rows = df.reset_index().values.tolist()
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join([header, sep] + body)


def build_mvp_report(
    composites: pd.DataFrame,
    regimes: pd.Series,
    backtest: pd.DataFrame,
    report_dir: Path,
) -> Path:
    """Create markdown report + charts to summarize MVP results."""
    report_dir.mkdir(parents=True, exist_ok=True)

    composite_chart = report_dir / "composite_scores.png"
    wealth_chart = report_dir / "wealth_curve.png"
    regime_chart = report_dir / "regime_distribution.png"

    chart_ok = {
        "composite": _save_composite_chart(composites, composite_chart),
        "wealth": _save_wealth_chart(backtest, wealth_chart),
        "regime": _save_regime_distribution(regimes, regime_chart),
    }

    perf_table = _regime_performance_table(backtest, regimes).round(4)
    annual_table = _annual_table(backtest).round(4)

    latest_regime = regimes.dropna().iloc[-1] if regimes.dropna().shape[0] else "unknown"
    latest_scores = composites.dropna().iloc[-1].round(3).to_dict() if composites.dropna().shape[0] else {}

    chart_note = ""
    if not all(chart_ok.values()):
        chart_note = "- Chart rendering skipped because `matplotlib` is not installed in this environment."

    report_path = report_dir / "mvp_report.md"
    report_path.write_text(
        "\n".join(
            [
                "# Macro Regime Allocation MVP Report",
                "",
                "## 1. Executive Summary",
                f"- Latest detected regime: **{latest_regime}**",
                f"- Latest composite scores: **{latest_scores}**",
                f"- Backtest period: **{backtest.index.min().date()} ~ {backtest.index.max().date()}**",
                chart_note,
                "",
                "## 2. Data & Method",
                "- Monthly macro panel aligned with publication lags.",
                "- Rolling z-score standardization (backward-looking only).",
                "- 2-axis regime classification using growth and inflation composites.",
                "- Monthly rebalancing with turnover-based transaction cost assumptions.",
                "",
                "## 3. Macro Composite Trend",
                "![composite scores](composite_scores.png)" if chart_ok["composite"] else "(Chart unavailable)",
                "",
                "## 4. Regime Distribution",
                "![regime distribution](regime_distribution.png)" if chart_ok["regime"] else "(Chart unavailable)",
                "",
                "## 5. Portfolio Backtest",
                "![wealth curve](wealth_curve.png)" if chart_ok["wealth"] else "(Chart unavailable)",
                "",
                "### 5-1. Regime Performance Table",
                _to_markdown_table(perf_table),
                "",
                "### 5-2. Annual Return Table",
                _to_markdown_table(annual_table),
                "",
                "## 6. Interpretation Notes",
                "- Regime counts should be checked for sample imbalance before optimizing weights.",
                "- This MVP uses sample data; production analysis should replace with real datasets.",
                "- Next step: compare rule-based regimes with HMM and add factor-level attribution.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    return report_path
