from __future__ import annotations

import argparse
import os

import pandas as pd
import requests
from dotenv import load_dotenv


def fetch_items(api_key: str, stat_code: str) -> pd.DataFrame:
    url = f"https://ecos.bok.or.kr/api/StatisticItemList/{api_key}/json/kr/1/10000/{stat_code}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    payload = r.json()

    if payload.get("RESULT"):
        raise RuntimeError(payload["RESULT"].get("MESSAGE", "ECOS API error"))

    rows = payload.get("StatisticItemList", {}).get("row", [])
    if not rows:
        raise RuntimeError("No item rows returned")

    df = pd.DataFrame(rows)
    keep_cols = [c for c in ["STAT_CODE", "STAT_NAME", "ITEM_CODE", "ITEM_NAME", "CYCLE", "START_TIME", "END_TIME"] if c in df.columns]
    return df[keep_cols]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Discover valid ECOS item codes for a statistic code")
    parser.add_argument("stat_code", help="ECOS statistic code, e.g., 901Y009")
    parser.add_argument("--out", default=None, help="Optional output CSV path")
    args = parser.parse_args()

    load_dotenv(".env")
    key = os.getenv("BOK_API_KEY")
    if not key:
        raise SystemExit("BOK_API_KEY is required in .env or environment")

    out_df = fetch_items(key, args.stat_code)
    print(out_df.head(30).to_string(index=False))

    if args.out:
        out_df.to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"\nSaved: {args.out}")
