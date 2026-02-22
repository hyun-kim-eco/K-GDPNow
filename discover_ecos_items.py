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
    keep_cols = [
        c
        for c in ["STAT_CODE", "STAT_NAME", "ITEM_CODE", "ITEM_NAME", "CYCLE", "START_TIME", "END_TIME"]
        if c in df.columns
    ]
    return df[keep_cols]


def filter_items(df: pd.DataFrame, cycle: str | None, contains: str | None) -> pd.DataFrame:
    out = df.copy()

    # 기본: 월간(M) 우선 필터링 (GDP nowcast 설명변수는 대부분 월간)
    if cycle:
        out = out[out["CYCLE"].astype(str).str.upper() == cycle.upper()]

    # 종료시점이 있는 항목(대개 현재 사용 가능한 코드) 우선
    if "END_TIME" in out.columns:
        out = out[out["END_TIME"].notna()]

    if contains:
        out = out[out["ITEM_NAME"].astype(str).str.contains(contains, case=False, na=False)]

    sort_cols = [c for c in ["ITEM_NAME", "ITEM_CODE"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Discover valid ECOS item codes for a statistic code")
    parser.add_argument("stat_code", help="ECOS statistic code, e.g., 901Y009")
    parser.add_argument("--cycle", default="M", help="Cycle filter (default: M). Use '' to disable")
    parser.add_argument("--contains", default=None, help="Filter ITEM_NAME by keyword (e.g., 계절조정)")
    parser.add_argument("--limit", type=int, default=20, help="Rows to preview in console")
    parser.add_argument("--out", default=None, help="Optional output CSV path")
    args = parser.parse_args()

    load_dotenv(".env")
    key = os.getenv("BOK_API_KEY")
    if not key:
        raise SystemExit("BOK_API_KEY is required in .env or environment")

    raw_df = fetch_items(key, args.stat_code)
    cycle = args.cycle if args.cycle else None
    out_df = filter_items(raw_df, cycle=cycle, contains=args.contains)

    if out_df.empty:
        print("No rows matched filters. Try --cycle '' or remove --contains.")
    else:
        print(out_df.head(args.limit).to_string(index=False))
        print(f"\nMatched rows: {len(out_df)}")

    if args.out:
        out_df.to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"Saved: {args.out}")
