from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class DataLakePaths:
    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def staging(self) -> Path:
        return self.root / "staging"

    @property
    def feature_store(self) -> Path:
        return self.root / "feature_store"

    @property
    def vintage_store(self) -> Path:
        return self.root / "vintage_store"


class DataStore:
    def __init__(self, root: str | Path = "data") -> None:
        self.paths = DataLakePaths(Path(root))
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        for p in [
            self.paths.root,
            self.paths.raw,
            self.paths.staging,
            self.paths.feature_store,
            self.paths.vintage_store,
        ]:
            p.mkdir(parents=True, exist_ok=True)

    def write_table(self, df: pd.DataFrame, layer: str, name: str, as_of_date: str) -> Path:
        layer_path = getattr(self.paths, layer)
        out_dir = layer_path / f"as_of_date={as_of_date}"
        out_dir.mkdir(parents=True, exist_ok=True)

        parquet_path = out_dir / f"{name}.parquet"
        try:
            df.to_parquet(parquet_path)
            return parquet_path
        except Exception:
            csv_path = out_dir / f"{name}.csv"
            df.to_csv(csv_path, encoding="utf-8-sig")
            return csv_path

    def write_manifest(self, payload: dict[str, Any], as_of_date: str) -> Path:
        out_dir = self.paths.vintage_store / f"as_of_date={as_of_date}"
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest_path
