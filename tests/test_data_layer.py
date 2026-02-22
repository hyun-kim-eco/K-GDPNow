from pathlib import Path

import pandas as pd

from src.kgdpnow.data.quality import validate_time_series_frame
from src.kgdpnow.data.store import DataStore


def test_validate_time_series_frame_passes():
    idx = pd.date_range("2020-01-31", periods=3, freq="M")
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]}, index=idx)
    validate_time_series_frame(df, "x")


def test_data_store_writes_csv_or_parquet(tmp_path: Path):
    idx = pd.date_range("2020-01-31", periods=2, freq="M")
    df = pd.DataFrame({"x": [1.0, 2.0]}, index=idx)

    store = DataStore(root=tmp_path)
    out = store.write_table(df, "raw", "x", "2026-01-01")

    assert out.exists()
    assert "as_of_date=2026-01-01" in str(out)
