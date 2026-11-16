#!/usr/bin/env python3
"""
03_pull_edgar.py  —  The supplier/customer edge workhorse.

For each firm with a CIK: fetch its filing history, download the latest 10-K (US) / 20-F
(foreign) per year, extract the text, and detect customer-concentration disclosures
("significant customer", ">= 10% of revenue", etc.). Each detection is saved with the
verbatim sentence as `raw_quote` so 06 can mint a provenance-backed `supplier_of` edge.

CRITICAL SEC RULES (https://www.sec.gov/os/webmaster-faq#developers):
    * A descriptive User-Agent with a contact email is MANDATORY (set SEC_USER_AGENT).
    * Hard limit 10 requests/second -> we sleep 0.12s and back off on HTTP 403.

KEY CAVEAT: ASC 280 / IFRS 8 require disclosing the FACT and AMOUNT of a >=10% customer,
but NOT always the customer's NAME. When unnamed, we store the % as a NODE attribute
(see 06), not an edge. This bounds supplier/customer edge recall to named cases -- expected.

OUTPUTS:
    data/raw/edgar/<TICKER>/submissions.json
    data/raw/edgar/<TICKER>/<accession>.txt            (extracted filing text)
    data/raw/edgar/concentration_hits.jsonl            (one line per detected disclosure)

USAGE:
    export SEC_USER_AGENT="TCD MSc Research your.name@tcd.ie"
    python 03_pull_edgar.py --universe config/universe.csv --years 2020 2026
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

SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT", "TCD MSc Research your.name@tcd.ie")
HEADERS = {"User-Agent": SEC_USER_AGENT}
SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"
RATE_PAUSE = 0.12  # ~ <10 req/s

# Phrases that flag customer-concentration disclosures (confirmed real-world patterns).
CONCENTRATION_PATTERNS = [
    r"significant customer", r"major customer", r"largest customer",
    r"accounted for (?:approximately )?\d{1,2}(?:\.\d+)?%",
    r"\b\d{1,2}(?:\.\d+)?% of (?:our )?(?:total )?(?:net )?(?:revenue|sales)",
    r"concentration of credit risk",
]
PATTERN_RE = re.compile("|".join(CONCENTRATION_PATTERNS), re.IGNORECASE)
SENTENCE_RE = re.compile(r"[^.!?]*(?:significant customer|major customer|largest customer|"
                         r"accounted for|% of (?:our )?(?:total )?(?:net )?(?:revenue|sales))"
                         r"[^.!?]*[.!?]", re.IGNORECASE)


def sec_get(url: str, max_retries: int = 5) -> requests.Response | None:
    """GET with SEC rate-limit etiquette and exponential backoff on 403."""
    backoff = 1.0
    for attempt in range(max_retries):
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code == 200:
            time.sleep(RATE_PAUSE)
            return r
        if r.status_code == 403:
            print(f"[warn] 403 from SEC (rate limit / UA). backoff {backoff:.0f}s ...")
            time.sleep(backoff)
            backoff *= 2
            continue
        print(f"[warn] {r.status_code} for {url}")
        return None
    return None


def latest_filings_per_year(submissions: dict, forms: set[str],
                            y0: int, y1: int) -> list[dict]:
    """From a submissions blob, pick the latest filing of each wanted form per year."""
    recent = submissions.get("filings", {}).get("recent", {})
    out: dict[tuple[int, str], dict] = {}
    for form, date, acc, doc in zip(recent.get("form", []), recent.get("filingDate", []),
                                    recent.get("accessionNumber", []),
                                    recent.get("primaryDocument", [])):
        if form not in forms:
            continue
        year = int(date[:4])
        if not (y0 <= year <= y1):
            continue
        key = (year, form)
        if key not in out or date > out[key]["date"]:
            out[key] = {"form": form, "date": date, "accession": acc, "doc": doc}
    return list(out.values())


def extract_text(html_or_txt: str) -> str:
    """Crude tag strip -- good enough to scan for concentration phrases."""
    text = re.sub(r"<[^>]+>", " ", html_or_txt)
    text = re.sub(r"&[a-z]+;", " ", text)
    return re.sub(r"\s+", " ", text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", type=Path, default=Path("config/universe.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/raw/edgar"))
    ap.add_argument("--years", nargs=2, type=int, default=[2020, 2026])
    args = ap.parse_args()

    if "your.name@tcd.ie" in SEC_USER_AGENT:
        print("[FATAL] set SEC_USER_AGENT to your real email (SEC requires it).")
        return 2

    uni = pd.read_csv(args.universe)
    args.out.mkdir(parents=True, exist_ok=True)
    hits_path = args.out / "concentration_hits.jsonl"
    n_hits = 0

    with open(hits_path, "w") as hits_file:
        for _, row in uni.iterrows():
            ticker, cik = str(row["ticker"]), str(row.get("cik", "")).strip()
            if not cik or cik in ("nan", ""):
                print(f"[skip] {ticker}: no CIK (non-US filer or unresolved).")
                continue
            cik = f"{int(float(cik)):010d}"

            sub = sec_get(SUBMISSIONS.format(cik=cik))
            if sub is None:
                continue
            submissions = sub.json()
            fdir = args.out / ticker
            fdir.mkdir(parents=True, exist_ok=True)
            (fdir / "submissions.json").write_text(json.dumps(submissions))

            filings = latest_filings_per_year(submissions, {"10-K", "20-F"},
                                               args.years[0], args.years[1])
            print(f"[ok] {ticker} (CIK {cik}): {len(filings)} annual filings to scan.")

            for f in filings:
                acc = f["accession"].replace("-", "")
                doc_resp = sec_get(ARCHIVE.format(cik=int(cik), acc=acc, doc=f["doc"]))
                if doc_resp is None:
                    continue
                text = extract_text(doc_resp.text)
                (fdir / f"{acc}.txt").write_text(text[:5_000_000])  # cap huge filings

                if PATTERN_RE.search(text):
                    for sent in SENTENCE_RE.findall(text):
                        sent = sent.strip()
                        if 20 < len(sent) < 400:  # plausible single disclosure sentence
                            hits_file.write(json.dumps({
                                "discloser_ticker": ticker, "cik": cik,
                                "form": f["form"], "filing_date": f["date"],
                                "accession": f["accession"],
                                "source_url": ARCHIVE.format(cik=int(cik), acc=acc, doc=f["doc"]),
                                "raw_quote": sent,
                            }) + "\n")
                            n_hits += 1

    print(f"\n[done] concentration hits -> {hits_path} ({n_hits} sentences)")
    print("       06_build_edges.py turns NAMED-counterparty hits into supplier_of edges;")
    print("       unnamed % disclosures become node attributes (recall bound -- expected).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
