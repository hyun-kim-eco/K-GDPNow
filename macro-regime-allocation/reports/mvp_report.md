# Macro Regime Allocation MVP Report

## 1. Executive Summary
- Latest detected regime: **inflation_overheat**
- Latest composite scores: **{'growth': 1.832, 'inflation': 0.553, 'liquidity': 0.825, 'sentiment': 2.111}**
- Backtest period: **2000-01-31 ~ 2025-12-31**
- Chart rendering skipped because `matplotlib` is not installed in this environment.

## 2. Data & Method
- Monthly macro panel aligned with publication lags.
- Rolling z-score standardization (backward-looking only).
- 2-axis regime classification using growth and inflation composites.
- Monthly rebalancing with turnover-based transaction cost assumptions.

## 3. Macro Composite Trend
(Chart unavailable)

## 4. Regime Distribution
(Chart unavailable)

## 5. Portfolio Backtest
(Chart unavailable)

### 5-1. Regime Performance Table
| regime | months | avg_monthly_return | volatility | win_rate | max_drawdown | sharpe_proxy |
| --- | --- | --- | --- | --- | --- | --- |
| stagflation_recession | 67 | 0.0006 | 0.0046 | 0.5672 | -0.1853 | 0.1366 |
| inflation_overheat | 55 | 0.0002 | 0.0052 | 0.5636 | -0.1721 | 0.0392 |
| expansion | 92 | -0.001 | 0.0059 | 0.4783 | -0.1723 | -0.166 |
| disinflation_slowdown | 80 | -0.0011 | 0.0054 | 0.4 | -0.1789 | -0.202 |

### 5-2. Annual Return Table
| date | annual_return |
| --- | --- |
| 2000.0 | -0.0308 |
| 2001.0 | -0.0095 |
| 2002.0 | 0.0317 |
| 2003.0 | 0.0087 |
| 2004.0 | 0.0197 |
| 2005.0 | -0.0253 |
| 2006.0 | -0.0266 |
| 2007.0 | -0.033 |
| 2008.0 | -0.0076 |
| 2009.0 | -0.0049 |
| 2010.0 | -0.0091 |
| 2011.0 | 0.0214 |
| 2012.0 | -0.0547 |
| 2013.0 | 0.0098 |
| 2014.0 | -0.0154 |
| 2015.0 | 0.0044 |
| 2016.0 | -0.0214 |
| 2017.0 | 0.0157 |
| 2018.0 | 0.002 |
| 2019.0 | -0.0128 |
| 2020.0 | -0.0063 |
| 2021.0 | 0.02 |
| 2022.0 | -0.0305 |
| 2023.0 | 0.0059 |
| 2024.0 | -0.0088 |
| 2025.0 | 0.0004 |

## 6. Interpretation Notes
- Regime counts should be checked for sample imbalance before optimizing weights.
- This MVP uses sample data; production analysis should replace with real datasets.
- Next step: compare rule-based regimes with HMM and add factor-level attribution.
