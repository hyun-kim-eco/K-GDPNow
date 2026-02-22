from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import requests

from ..models import DataSeriesSpec, Frequency


class BOKECOSIngestor:
    base_url = "https://ecos.bok.or.kr/api/StatisticSearch"

    def __init__(self, api_key: str | None = None, timeout_sec: int = 30) -> None:
        self.api_key = api_key or os.getenv("BOK_API_KEY")
        self.timeout_sec = timeout_sec
        if not self.api_key:
            raise ValueError("BOK_API_KEY is required")

    def _end_period(self, frequency: Frequency) -> str:
        now = datetime.today()
        if frequency == Frequency.MONTHLY:
            return now.strftime("%Y%m")
        quarter = (now.month - 1) // 3 + 1
        return f"{now.year}Q{quarter}"

    def fetch_series(self, spec: DataSeriesSpec) -> pd.DataFrame:
        end_period = self._end_period(spec.frequency)
        freq_token = spec.frequency.value

        args = [
            self.base_url,
            self.api_key,
            "json",
            "kr",
            "1",
            "100000",
            spec.api_code,
            freq_token,
            spec.start_period,
            end_period,
            spec.item_code,
        ]
        if spec.item_code_2:
            args.append(spec.item_code_2)

        url = "/".join(args)
        response = requests.get(url, timeout=self.timeout_sec)
        response.raise_for_status()
        payload = response.json()

        rows = payload.get("StatisticSearch", {}).get("row", [])
        if not rows:
            return pd.DataFrame(columns=[spec.name])

        df = pd.DataFrame(rows)
        s = pd.to_numeric(df["DATA_VALUE"].astype(str).str.replace(",", "", regex=False), errors="coerce")

        if spec.frequency == Frequency.MONTHLY:
            index = pd.to_datetime(df["TIME"], format="%Y%m") + pd.offsets.MonthEnd(0)
        else:
            years = df["TIME"].str.split("Q").str[0].astype(int)
            quarters = df["TIME"].str.split("Q").str[1].astype(int)
            months = quarters * 3
            index = pd.to_datetime(dict(year=years, month=months, day=1)) + pd.offsets.QuarterEnd(0)

        out = pd.DataFrame({spec.name: s.values}, index=index)
        out = out[~out.index.duplicated(keep="last")].sort_index()
        out.index.name = "date"
        return out
