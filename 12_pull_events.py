#!/usr/bin/env python3
"""
12_pull_events.py  —  Pull EDGAR 8-K filings as events for Stage 2.

For each universe firm with a CIK, fetches all 8-K filings from 2020-01-01
to 2026-04-30 via the EDGAR submissions API. Extracts the first 800 chars of
each filing document as event_text. Writes one JSON object per filing.

8-K items filtered to material events:
  1.01 Entry into Material Agreement
  1.02 Termination of Material Agreement
  2.02 Results of Operations / Earnings
  5.02 Departure/Appointment of Officers / Directors
  7.01 Regulation FD Disclosure
  8.01 Other Events

OUTPUT:
    data/raw/events/events_8k.jsonl

USAGE:
    export SEC_USER_AGENT="Your Name your.name@example.com"
    python 12_pull_events.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
EDGAR_BASE      = "https://www.sec.gov/Archives/edgar/data"

DATE_START = "2020-01-01"
DATE_END   = "2026-04-30"

MATERIAL_ITEMS = {"1.01", "1.02", "2.02", "5.02", "7.01", "8.01"}

SLEEP_SEC = 0.12   # ≤10 req/s SEC limit


def get_headers() -> dict:
    agent = os.environ.get("SEC_USER_AGENT", "")
    if not agent or "placeholder" in agent.lower():
        print("[FATAL] Set SEC_USER_AGENT to 'Name email@domain'")
        sys.exit(2)
    return {"User-Agent": agent, "Accept": "application/json"}


def fetch_submissions(cik: int, headers: dict) -> dict | None:
    url = SUBMISSIONS_URL.format(cik=cik)
    try:
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  [warn] CIK {cik}: submissions fetch failed — {e}")
        return None


def collect_8k_filings(sub: dict, date_start: str, date_end: str) -> list[dict]:
    """Extract 8-K filings from submissions JSON within the date window."""
    filings = []
    recent = sub.get("filings", {}).get("recent", {})
    forms   = recent.get("form", [])
    dates   = recent.get("filingDate", [])
    accnos  = recent.get("accessionNumber", [])
    pdocs   = recent.get("primaryDocument", [])
    items   = recent.get("items", [""] * len(forms))   # not always present

    for form, date, accno, pdoc, item in zip(forms, dates, accnos, pdocs, items):
        if form not in ("8-K", "8-K/A"):
            continue
        if not (date_start <= date <= date_end):
            continue
        filings.append({
            "filing_date":       date,
            "accession_number":  accno,
            "primary_document":  pdoc,
            "items":             str(item),
        })
    return filings


def extract_text(cik: int, accno: str, pdoc: str, headers: dict, max_chars: int = 800) -> str:
    """Fetch the primary 8-K document and return up to max_chars of plain text."""
    acc_path = accno.replace("-", "")
    url = f"{EDGAR_BASE}/{cik}/{acc_path}/{pdoc}"
    try:
        r = requests.get(url, headers={**headers, "Accept": "text/html,*/*"}, timeout=20)
        r.raise_for_status()
        raw = r.text
        # Strip HTML tags
        clean = re.sub(r"<[^>]+>", " ", raw)
        clean = re.sub(r"\s+", " ", clean).strip()
        # Skip boilerplate SEC headers (first 200 chars often say "SECURITIES AND EXCHANGE")
        return clean[:max_chars]
    except Exception as e:
        return f"[text unavailable: {e}]"


def has_material_item(items_str: str) -> bool:
    if not items_str:
        return True  # unknown items — keep
    for item in MATERIAL_ITEMS:
        if item in items_str:
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", type=Path, default=Path("config/universe.csv"))
    ap.add_argument("--out-dir",  type=Path, default=Path("data/raw/events"))
    ap.add_argument("--date-start", default=DATE_START)
    ap.add_argument("--date-end",   default=DATE_END)
    ap.add_argument("--skip-text",  action="store_true",
                    help="Skip document fetch (metadata-only mode — faster)")
    args = ap.parse_args()

    headers = get_headers()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "events_8k.jsonl"

    uni = pd.read_csv(args.universe)
    cik_rows = uni[uni["cik"].notna() & (uni["cik"].astype(str).str.strip() != "")]
    print(f"[info] {len(cik_rows)} firms with CIK out of {len(uni)} total")

    written = 0
    with open(out_path, "w") as fout:
        for _, row in cik_rows.iterrows():
            ticker = str(row["ticker"])
            try:
                cik = int(float(str(row["cik"])))
            except (ValueError, TypeError):
                continue

            print(f"  {ticker} (CIK {cik}) ...", end=" ", flush=True)
            sub = fetch_submissions(cik, headers)
            time.sleep(SLEEP_SEC)
            if sub is None:
                print("skip")
                continue

            filings = collect_8k_filings(sub, args.date_start, args.date_end)
            # Filter to material items where items field is present
            filings = [f for f in filings if has_material_item(f["items"])]

            for f in filings:
                event_text = ""
                if not args.skip_text and f["primary_document"]:
                    event_text = extract_text(
                        cik, f["accession_number"], f["primary_document"], headers
                    )
                    time.sleep(SLEEP_SEC)

                acc_dashed = f["accession_number"]
                acc_path   = acc_dashed.replace("-", "")
                form_url   = f"{EDGAR_BASE}/{cik}/{acc_path}/{f['primary_document']}"

                record = {
                    "event_id":         f"{ticker}_{f['filing_date']}_{acc_dashed[:18]}",
                    "primary_ticker":   ticker,
                    "cik":              cik,
                    "filing_date":      f["filing_date"],
                    "items":            f["items"],
                    "accession_number": acc_dashed,
                    "form_url":         form_url,
                    "event_text":       event_text,
                }
                fout.write(json.dumps(record) + "\n")
                written += 1

            print(f"{len(filings)} events")

    print(f"\n[done] {written} 8-K events -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
