#!/usr/bin/env python3
"""
01_resolve_entities.py  —  Resolve every firm to its canonical IDs.

Maps each ticker to its SEC CIK (for EDGAR) and Wikidata Q-ID (for hierarchy edges),
records ADR/home-exchange info, and writes the enriched universe. Ambiguous matches are
written to a REVIEW file for human confirmation -- the agent never silently guesses identity.

OUTPUTS:
    config/universe.csv                      (enriched: + cik, qid, is_adr, ...)
    review/entity_resolution_review.csv      (ambiguous rows for you to confirm)

NETWORK: needs SEC (data.sec.gov / www.sec.gov) + Wikidata (www.wikidata.org). Run locally.
SEC requires a descriptive User-Agent with a contact email -- set SEC_USER_AGENT below.

USAGE:
    export SEC_USER_AGENT="TCD MSc Research your.name@tcd.ie"
    python 01_resolve_entities.py --universe config/universe.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT", "TCD MSc Research your.name@tcd.ie")
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
WIKIDATA_SEARCH = "https://www.wikidata.org/w/api.php"

# Reuse the price-script seed if no CSV exists, so 01 can bootstrap an empty repo.
SEED_UNIVERSE = [
    {"ticker": "NVDA", "name": "NVIDIA", "country": "US", "has_price": True},
    {"ticker": "INTC", "name": "Intel", "country": "US", "has_price": True},
    {"ticker": "TSM", "name": "Taiwan Semiconductor Manufacturing", "country": "US", "has_price": True},
    {"ticker": "ASML", "name": "ASML Holding", "country": "US", "has_price": True},
    {"ticker": "AAPL", "name": "Apple", "country": "US", "has_price": True},
    {"ticker": "IFX.DE", "name": "Infineon Technologies", "country": "DE", "has_price": True},
    {"ticker": "BOSCH", "name": "Robert Bosch", "country": "DE", "has_price": False},
]


def load_universe(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    print(f"[info] no {path}; bootstrapping from built-in seed.")
    return pd.DataFrame(SEED_UNIVERSE)


def fetch_sec_ticker_map() -> dict[str, str]:
    """Download SEC's ticker->CIK map. CIK is zero-padded to 10 digits for the APIs."""
    r = requests.get(SEC_TICKERS_URL, headers={"User-Agent": SEC_USER_AGENT}, timeout=30)
    r.raise_for_status()
    out: dict[str, str] = {}
    for row in r.json().values():
        out[str(row["ticker"]).upper()] = f"{int(row['cik_str']):010d}"
    return out


def resolve_qid(name: str) -> tuple[str | None, bool]:
    """
    Resolve a company name to a Wikidata Q-ID via wbsearchentities.
    Returns (qid, ambiguous). ambiguous=True when the top two hits are both plausible
    companies -> needs human confirmation.
    """
    params = {"action": "wbsearchentities", "search": name, "language": "en",
              "type": "item", "format": "json", "limit": 5}
    try:
        r = requests.get(WIKIDATA_SEARCH, params=params,
                         headers={"User-Agent": SEC_USER_AGENT}, timeout=20)
        r.raise_for_status()
        hits = r.json().get("search", [])
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] wikidata search failed for {name!r}: {exc}")
        return None, True
    if not hits:
        return None, True
    top = hits[0]["id"]
    ambiguous = len(hits) >= 2 and hits[1].get("label", "").lower()[:4] == name.lower()[:4]
    return top, ambiguous


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", type=Path, default=Path("config/universe.csv"))
    ap.add_argument("--review", type=Path, default=Path("review/entity_resolution_review.csv"))
    args = ap.parse_args()

    if "your.name@tcd.ie" in SEC_USER_AGENT:
        print("[FATAL] set SEC_USER_AGENT to your real email before running (SEC requires it).")
        return 2

    uni = load_universe(args.universe)
    sec_map = fetch_sec_ticker_map()
    print(f"[info] SEC ticker map: {len(sec_map)} US filers.")

    rows, review = [], []
    for _, row in uni.iterrows():
        ticker = str(row["ticker"]).strip()
        name = str(row["name"]).strip()
        country = str(row.get("country", "US"))
        base = ticker.split(".")[0].upper()           # IFX.DE -> IFX for SEC lookup

        cik = sec_map.get(base) or sec_map.get(ticker.upper())
        qid, ambiguous = resolve_qid(name)
        is_adr = bool(row.get("is_adr", country != "US" and "." not in ticker))

        rec = {
            "ticker": ticker, "name": name, "country": country,
            "has_price": bool(row.get("has_price", True)),
            "cik": cik or "", "qid": qid or "", "is_adr": is_adr,
            "home_exchange": row.get("home_exchange", ""),
            "adr_ticker": row.get("adr_ticker", ""),
            "name_aliases": row.get("name_aliases", ""),
        }
        rows.append(rec)

        # Flag for human review: no CIK on a US filer, no QID, or ambiguous QID match.
        if (not cik and country == "US") or not qid or ambiguous:
            review.append({**rec, "reason":
                           ("ambiguous_qid" if ambiguous else
                            "missing_cik" if not cik and country == "US" else "missing_qid")})
        print(f"[ok] {ticker}: cik={cik or '-'} qid={qid or '-'}"
              f"{' [REVIEW]' if rec['ticker'] in {r['ticker'] for r in review} else ''}")
        time.sleep(0.15)  # polite to both APIs

    args.universe.parent.mkdir(parents=True, exist_ok=True)
    args.review.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.universe, index=False)
    pd.DataFrame(review).to_csv(args.review, index=False)

    print(f"\n[done] universe -> {args.universe} ({len(rows)} firms)")
    print(f"[done] review   -> {args.review} ({len(review)} need confirmation)")
    print("       Confirm the REVIEW rows by hand before running 03/04 (TCD AI-use policy).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
