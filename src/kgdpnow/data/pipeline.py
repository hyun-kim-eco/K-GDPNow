from __future__ import annotations

from datetime import datetime

import pandas as pd

from .ingestors.bok_ecos import BOKECOSIngestor
from .models import DataSeriesSpec
from .quality import DataQualityError, missing_rate, validate_time_series_frame
from .store import DataStore


class DataIngestionPipeline:
    """Collect ECOS series, validate, and persist by as-of snapshot."""

    def __init__(self, data_store: DataStore, ingestor: BOKECOSIngestor) -> None:
        self.data_store = data_store
        self.ingestor = ingestor

    def run(self, specs: list[DataSeriesSpec], as_of_date: str | None = None) -> dict:
        if not specs:
            raise ValueError("At least one DataSeriesSpec is required")

        as_of_date = as_of_date or datetime.today().strftime("%Y-%m-%d")
        series_frames = []
        series_meta = []
        skipped_series = []

        for spec in specs:
            df = self.ingestor.fetch_series(spec)

            try:
                validate_time_series_frame(df, spec.name)
            except DataQualityError as exc:
                skipped_series.append({"name": spec.name, "reason": str(exc)})
                continue

            path = self.data_store.write_table(df, "raw", spec.name, as_of_date)
            series_frames.append(df)
            series_meta.append(
                {
                    "name": spec.name,
                    "frequency": spec.frequency.value,
                    "rows": int(df.shape[0]),
                    "start": str(df.index.min().date()),
                    "end": str(df.index.max().date()),
                    "path": str(path),
                    "missing_rate": float(missing_rate(df).iloc[0]),
                }
            )

        if not series_frames:
            raise ValueError("No valid series were ingested. Check API responses and metadata specs.")

        combined = pd.concat(series_frames, axis=1).sort_index()
        validate_time_series_frame(combined, "combined_raw")
        combined_path = self.data_store.write_table(combined, "staging", "combined_raw", as_of_date)

        manifest = {
            "as_of_date": as_of_date,
            "series_count": len(series_meta),
            "skipped_count": len(skipped_series),
            "generated_at": datetime.utcnow().isoformat(),
            "staging_path": str(combined_path),
            "series": series_meta,
            "skipped_series": skipped_series,
        }
        manifest_path = self.data_store.write_manifest(manifest, as_of_date)
        manifest["manifest_path"] = str(manifest_path)

        return manifest
