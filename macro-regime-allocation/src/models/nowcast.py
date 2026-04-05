"""Simple VAR-based nowcasting utilities."""

from __future__ import annotations

import pandas as pd
from statsmodels.tsa.api import VAR


class VARNowcaster:
    """Minimal VAR nowcaster for filling short publication gaps."""

    def __init__(self, maxlags: int = 3) -> None:
        self.maxlags = maxlags

    def forecast_next(self, df: pd.DataFrame, steps: int = 1) -> pd.DataFrame:
        """Forecast the next step(s) from a multivariate monthly panel."""
        clean = df.dropna()
        if len(clean) <= self.maxlags + 1:
            raise ValueError("Not enough observations to fit VAR model.")

        model = VAR(clean)
        fitted = model.fit(maxlags=self.maxlags)
        fcst = fitted.forecast(clean.values[-fitted.k_ar :], steps=steps)

        idx = pd.date_range(clean.index[-1] + pd.offsets.MonthEnd(1), periods=steps, freq="M")
        return pd.DataFrame(fcst, index=idx, columns=clean.columns)
