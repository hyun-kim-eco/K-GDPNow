from __future__ import annotations

from pathlib import Path

import pandas as pd

from .models import DataSeriesSpec, Frequency


def load_specs_from_csv(path: str | Path, frequency: Frequency) -> list[DataSeriesSpec]:
    df = pd.read_csv(path).fillna("")
    specs = []
    for _, row in df.iterrows():
        specs.append(
            DataSeriesSpec(
                name=row["Name"].strip(),
                api_code=row["API_Code"].strip(),
                item_code=row["Item_code"].strip(),
                item_code_2=str(row.get("Item_code_2", "")).strip() or None,
                frequency=frequency,
                start_period="199501" if frequency == Frequency.MONTHLY else "1995Q1",
            )
        )
    return specs
