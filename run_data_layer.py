from __future__ import annotations

import os

from dotenv import load_dotenv

from src.kgdpnow.data import DataIngestionPipeline, DataStore, Frequency
from src.kgdpnow.data.ingestors.bok_ecos import BOKECOSIngestor
from src.kgdpnow.data.spec_loader import load_specs_from_csv


if __name__ == "__main__":
    load_dotenv(".env")

    api_key = os.getenv("BOK_API_KEY")
    if not api_key:
        raise SystemExit("BOK_API_KEY 환경변수가 필요합니다. (.env 또는 시스템 환경변수)")
    api_key = os.getenv("BOK_API_KEY")
    if not api_key:
        raise SystemExit("BOK_API_KEY 환경변수가 필요합니다.")

    store = DataStore(root="data")
    ingestor = BOKECOSIngestor(api_key=api_key)
    pipeline = DataIngestionPipeline(data_store=store, ingestor=ingestor)

    specs = load_specs_from_csv("data_list.csv", frequency=Frequency.MONTHLY)
    manifest = pipeline.run(specs)
    print("수집 완료")
    print(manifest)
