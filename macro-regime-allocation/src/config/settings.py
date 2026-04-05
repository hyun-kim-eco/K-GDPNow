"""Project-level configuration objects."""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class MacroGroups:
    """Indicator groups used for composite score calculation."""

    growth: List[str] = field(default_factory=lambda: ["ip_yoy", "retail_yoy", "employment_yoy"])
    inflation: List[str] = field(default_factory=lambda: ["cpi_yoy", "core_cpi_yoy", "ppi_yoy"])
    liquidity: List[str] = field(default_factory=lambda: ["credit_spread", "policy_rate", "m2_yoy"])
    sentiment: List[str] = field(default_factory=lambda: ["pmi", "consumer_sentiment"])


@dataclass(frozen=True)
class RegimeThresholds:
    """Thresholds for classifying growth/inflation regimes."""

    growth_high: float = 0.0
    inflation_high: float = 0.0


@dataclass(frozen=True)
class BacktestConfig:
    """Backtest configuration values."""

    transaction_cost_bps: float = 10.0
    rebalance_freq: str = "M"


DEFAULT_ASSET_UNIVERSE: List[str] = ["equity", "bond_long", "commodity", "gold", "usd"]

DEFAULT_REGIME_WEIGHTS: Dict[str, Dict[str, float]] = {
    "expansion": {"equity": 0.50, "bond_long": 0.20, "commodity": 0.15, "gold": 0.05, "usd": 0.10},
    "disinflation_slowdown": {"equity": 0.25, "bond_long": 0.45, "commodity": 0.05, "gold": 0.10, "usd": 0.15},
    "inflation_overheat": {"equity": 0.30, "bond_long": 0.05, "commodity": 0.35, "gold": 0.20, "usd": 0.10},
    "stagflation_recession": {"equity": 0.10, "bond_long": 0.40, "commodity": 0.10, "gold": 0.25, "usd": 0.15},
}
