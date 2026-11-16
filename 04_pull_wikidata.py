#!/usr/bin/env python3
"""
04_pull_wikidata.py  —  Hierarchy edges only.

Wikidata has NO property for supplier / customer / supply-chain / competitor / joint-venture
relationships (verified against Wikidata:WikiProject Companies/Properties). It reliably
supplies ONLY hierarchy:
    P355  child organization  -> subsidiary_of
    P749  parent organization -> parent_of
    P127  owned by            -> ownership supplement (with P1107 proportion qualifier)
    P1830 owner of            -> ownership supplement
    P1327 partner in business -> CANDIDATE partner_of ONLY (noisy; review-only in 06)

So everything value-chain comes from EDGAR + Apple + manual curation, NOT from here.

SPARQL ETIQUETTE (https://query.wikidata.org/):
    * Descriptive User-Agent is required (non-compliant clients are blocked).
    * ~60s query-time/min/IP; <=5 parallel; honour Retry-After on HTTP 429.

OUTPUTS:
    data/raw/wikidata/<TICKER>.json   (raw bindings)
    data/raw/wikidata/edges.jsonl     (typed candidate edges, all from hierarchy)

USAGE:
    python 04_pull_wikidata.py --universe config/universe.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

SPARQL = "https://query.wikidata.org/sparql"
UA = os.environ.get("SEC_USER_AGENT", "TCD MSc Research your.name@tcd.ie")  # reuse contact UA

QUERY = """
SELECT ?p ?other ?otherLabel WHERE {{
  VALUES ?p {{ wdt:P355 wdt:P749 wdt:P127 wdt:P1830 wdt:P1327 }}
  wd:{qid} ?p ?other .
  ?other wdt:P31/wdt:P279* wd:Q4830453 .          # only keep business-entity targets
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
"""

PROP_TO_EDGE = {
    # semantics: subsidiary_of(A→B) = "A is a subsidiary of B"
    "P355": ("subsidiary_of", "other_to_focal"),   # focal HAS subsidiary other  → other sub_of focal
    "P749": ("subsidiary_of", "focal_to_other"),   # focal's PARENT is other     → focal sub_of other
    "P127": ("subsidiary_of", "focal_to_other"),   # focal OWNED BY other        → focal sub_of other (conf≤0.6)
    "P1830": ("subsidiary_of", "other_to_focal"),  # focal OWNS other            → other sub_of focal (conf≤0.6)
    "P1327": ("partner_of", "symmetric"),          # NOISY partner candidate     → review_only only
}


def run_sparql(qid: str, max_retries: int = 4) -> list[dict] | None:
    headers = {"User-Agent": UA, "Accept": "application/sparql-results+json"}
    for _ in range(max_retries):
        r = requests.get(SPARQL, params={"query": QUERY.format(qid=qid), "format": "json"},
                         headers=headers, timeout=60)
        if r.status_code == 200:
            time.sleep(0.2)
            return r.json().get("results", {}).get("bindings", [])
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", "5"))
            print(f"[warn] 429 from WDQS; honouring Retry-After={wait}s")
            time.sleep(wait)
            continue
        print(f"[warn] WDQS {r.status_code} for {qid}")
        return None
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", type=Path, default=Path("config/universe.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/raw/wikidata"))
    args = ap.parse_args()

    uni = pd.read_csv(args.universe)
    # qid -> ticker, so we can map related entities back to our universe where possible.
    qid_to_ticker = {str(r["qid"]): str(r["ticker"]) for _, r in uni.iterrows()
                     if str(r.get("qid", "")).startswith("Q")}
    args.out.mkdir(parents=True, exist_ok=True)
    edges_path = args.out / "edges.jsonl"
    n_edges = 0

    with open(edges_path, "w") as ef:
        for _, row in uni.iterrows():
            ticker, qid = str(row["ticker"]), str(row.get("qid", ""))
            if not qid.startswith("Q"):
                continue
            bindings = run_sparql(qid)
            if bindings is None:
                continue
            (args.out / f"{ticker}.json").write_text(json.dumps(bindings))

            for b in bindings:
                prop = b["p"]["value"].rsplit("/", 1)[-1]
                other_qid = b["other"]["value"].rsplit("/", 1)[-1]
                other_label = b.get("otherLabel", {}).get("value", other_qid)
                edge_type, direction = PROP_TO_EDGE.get(prop, (None, None))
                if edge_type is None:
                    continue
                other_node = qid_to_ticker.get(other_qid, other_qid)
                if direction == "focal_to_other":
                    from_t, to_t = ticker, other_node
                elif direction == "other_to_focal":
                    from_t, to_t = other_node, ticker
                else:  # symmetric (partner_of)
                    from_t, to_t = sorted([ticker, other_node])
                if from_t == to_t:
                    continue  # never emit self-referential edges
                conf = 0.6 if prop in ("P127", "P1830", "P1327") else 0.85
                ef.write(json.dumps({
                    "from_ticker": from_t, "to_ticker": to_t,
                    "focal_ticker": ticker, "type": edge_type,
                    "wikidata_property": prop, "other_qid": other_qid,
                    "other_label": other_label,
                    "source_url": f"https://www.wikidata.org/wiki/{qid}",
                    "method": "wikidata", "confidence": conf,
                    "review_only": prop == "P1327",
                }) + "\n")
                n_edges += 1
            print(f"[ok] {ticker} ({qid}): {len(bindings)} hierarchy bindings")

    print(f"\n[done] wikidata edges -> {edges_path} ({n_edges})")
    print("       Only hierarchy + ownership. P1327 rows are review_only (noisy).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
