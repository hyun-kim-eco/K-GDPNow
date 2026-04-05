# AGENTS.md

## Project objective
Build a macro regime detection and asset allocation research pipeline.

## Core workflow
1. Ingest macroeconomic and market datasets.
2. Align them to a monthly timeline with publication lags.
3. Forecast unavailable values using VBAR/VAR-style models.
4. Standardize signals with rolling z-scores.
5. Classify macro regimes.
6. Evaluate historical asset performance by regime.
7. Produce regime-based allocation recommendations.

## Coding rules
- Use Python.
- Prefer pandas, numpy, statsmodels, scikit-learn, matplotlib.
- Keep all business logic under src/.
- Avoid putting core logic only in notebooks.
- Write reusable functions with docstrings and type hints.
- Save intermediate processed datasets in data/processed.
- All plots should be reproducible from scripts.

## Research rules
- Distinguish between data release date and observation date.
- Avoid look-ahead bias.
- Make transformation choices explicit.
- Keep config values centralized.
- Document every assumption for regime classification.

## Deliverables
- Clean data pipeline
- Forecast module
- Regime classification module
- Backtest module
- Summary report tables and charts
