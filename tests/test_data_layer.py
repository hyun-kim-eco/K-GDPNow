from pathlib import Path

import pandas as pd

from src.kgdpnow.data.models import DataSeriesSpec, Frequency
from src.kgdpnow.data.pipeline import DataIngestionPipeline
from src.kgdpnow.data.quality import validate_time_series_frame
from src.kgdpnow.data.store import DataStore


class DummyIngestor:
    def __init__(self, frames: dict[str, pd.DataFrame]):
        self.frames = frames

    def fetch_series(self, spec: DataSeriesSpec) -> pd.DataFrame:
        return self.frames[spec.name]


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


def test_pipeline_skips_invalid_series(tmp_path: Path):
    valid_idx = pd.date_range("2020-01-31", periods=2, freq="M")
    valid_df = pd.DataFrame({"valid": [1.0, 2.0]}, index=valid_idx)
    invalid_df = pd.DataFrame({"invalid": []})

    specs = [
        DataSeriesSpec("valid", "A", "1", Frequency.MONTHLY),
        DataSeriesSpec("invalid", "B", "2", Frequency.MONTHLY),
    ]

    ingestor = DummyIngestor({"valid": valid_df, "invalid": invalid_df})
    pipeline = DataIngestionPipeline(data_store=DataStore(tmp_path), ingestor=ingestor)

    manifest = pipeline.run(specs, as_of_date="2026-01-01")

    assert manifest["series_count"] == 1
    assert manifest["skipped_count"] == 1
    assert manifest["skipped_series"][0]["name"] == "invalid"
