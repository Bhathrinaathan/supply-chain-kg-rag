#!/usr/bin/env python3
"""
06_build_edges.py  —  Merge all sources into typed, provenance-backed edge tables.

Reads the raw extracts from 03/04/05, builds one Parquet per relation type, and writes
review/candidate_edges.csv containing EVERY non-Wikidata edge for human approve/reject
(the "review-not-research" gate: the agent pre-fills source_url + raw_quote; you only judge).

ENFORCED IN CODE (the DB cannot do it -- Community Edition lacks relationship constraints):
    * every edge MUST have source_url + method  -> rejected otherwise.
    * symmetric edges (competitor_of, partner_of, joint_venture_with) use a CANONICAL
      single-edge convention: lexicographically smaller ticker is 'from'.

NOTE on EDGAR: 03 captures concentration SENTENCES. Naming the counterparty reliably from
free text is hard, so by default each hit becomes a CANDIDATE edge with the focal firm as
supplier and an EMPTY `to_ticker` for you to fill during review. Unnamed/percentage-only
disclosures should instead be recorded as a node attribute (out of scope for this script).

OUTPUTS:
    data/edges/<type>.parquet
    review/candidate_edges.csv      (approved column defaults to '' -> set Y/N)

USAGE:
    python 06_build_edges.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SYMMETRIC = {"competitor_of", "partner_of", "joint_venture_with"}
REQUIRED = ("from_ticker", "to_ticker", "type", "source_url", "method")


def canonicalise(e: dict) -> dict:
    """For symmetric edges, order endpoints lexicographically so A-B == B-A (no double count)."""
    if e["type"] in SYMMETRIC and e["from_ticker"] and e["to_ticker"]:
        a, b = sorted([e["from_ticker"], e["to_ticker"]])
        e["from_ticker"], e["to_ticker"] = a, b
    return e


def valid(e: dict) -> bool:
    """Provenance gate enforced in code."""
    return bool(e.get("source_url")) and bool(e.get("method")) and bool(e.get("type"))


def load_wikidata(p: Path) -> list[dict]:
    out = []
    f = p / "wikidata" / "edges.jsonl"
    if not f.exists():
        return out
    for line in f.read_text().splitlines():
        d = json.loads(line)
        out.append({
            "from_ticker": d["from_ticker"], "to_ticker": d["to_ticker"],
            "type": d["type"], "source_url": d["source_url"], "method": "wikidata",
            "valid_from": d.get("valid_from") or "2020-01-01", "confidence": d.get("confidence", 0.85),
            "raw_quote": d.get("other_label", ""), "review_only": d.get("review_only", False),
        })
    return out


def load_edgar(p: Path) -> list[dict]:
    out = []
    f = p / "edgar" / "concentration_hits.jsonl"
    if not f.exists():
        return out
    for line in f.read_text().splitlines():
        d = json.loads(line)
        out.append({
            "from_ticker": d["discloser_ticker"],   # discloser is the supplier
            "to_ticker": "",                          # named customer -> fill during review
            "type": "supplier_of", "source_url": d["source_url"], "method": "edgar",
            "valid_from": d["filing_date"], "confidence": 0.5,
            "raw_quote": d["raw_quote"], "review_only": True,
        })
    return out


def load_competitors(p: Path) -> list[dict]:
    out = []
    f = p / "derived" / "competitor_hits.jsonl"
    if not f.exists():
        return out
    for line in f.read_text().splitlines():
        d = json.loads(line)
        out.append({
            "from_ticker": d["from_ticker"], "to_ticker": d["to_ticker"],
            "type": "competitor_of", "source_url": d["source_url"],
            "method": d.get("method", "edgar_competition"),
            "valid_from": d.get("valid_from", ""), "confidence": d.get("confidence", 0.7),
            "raw_quote": d.get("raw_quote", ""), "review_only": True,
        })
    return out


def load_partners(p: Path) -> list[dict]:
    out = []
    f = p / "derived" / "partner_hits.jsonl"
    if not f.exists():
        return out
    for line in f.read_text().splitlines():
        d = json.loads(line)
        etype = d.get("type", "partner_of")
        if etype not in {"partner_of", "joint_venture_with"}:
            etype = "partner_of"
        out.append({
            "from_ticker": d["from_ticker"], "to_ticker": d["to_ticker"],
            "type": etype, "source_url": d["source_url"],
            "method": d.get("method", "edgar_partnership"),
            "valid_from": d.get("valid_from", ""), "confidence": d.get("confidence", 0.7),
            "raw_quote": d.get("raw_quote", ""), "review_only": True,
        })
    return out


def load_apple(p: Path) -> list[dict]:
    out = []
    f = p / "apple" / "apple_edges.csv"
    if not f.exists():
        return out
    for _, d in pd.read_csv(f).iterrows():
        out.append({
            "from_ticker": (str(d["supplier_ticker"]) if pd.notna(d.get("supplier_ticker")) else str(d["supplier_name"])),
            "to_ticker": d["customer_ticker"], "type": "supplier_of",
            "source_url": d["source_url"], "method": "apple",
            "valid_from": d.get("valid_from", ""), "confidence": d.get("confidence", 0.9),
            "raw_quote": str(d["supplier_name"]), "review_only": True,
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--edges-dir", type=Path, default=Path("data/edges"))
    ap.add_argument("--review", type=Path, default=Path("review/candidate_edges.csv"))
    args = ap.parse_args()

    edges = (load_wikidata(args.raw) + load_edgar(args.raw)
             + load_apple(args.raw) + load_competitors(args.raw) + load_partners(args.raw))
    kept = [canonicalise(e) for e in edges if valid(e)]
    dropped = len(edges) - len(kept)
    if dropped:
        print(f"[warn] dropped {dropped} edges failing the provenance gate (no source_url/method).")

    df = pd.DataFrame(kept)
    if df.empty:
        print("[FATAL] no valid edges built. Run 03/04/05 first.")
        return 2

    # Per-type Parquet (Wikidata hierarchy can load directly; others go via review).
    args.edges_dir.mkdir(parents=True, exist_ok=True)
    for etype, g in df.groupby("type"):
        g.to_parquet(args.edges_dir / f"{etype}.parquet")
        print(f"[ok] {etype}: {len(g)} edges")

    # Review file: every non-Wikidata edge (and Wikidata P1327 partner candidates).
    review_mask = (df["method"] != "wikidata") | (df.get("review_only", False) == True)  # noqa: E712
    review = df[review_mask].copy()
    review["approved"] = ""      # human sets Y / N
    review["counterparty_to_fill"] = review["to_ticker"].eq("")
    args.review.parent.mkdir(parents=True, exist_ok=True)
    review.to_csv(args.review, index=False)

    print(f"\n[done] edge tables -> {args.edges_dir}/  (total {len(df)})")
    print(f"[done] REVIEW      -> {args.review}  ({len(review)} rows need approve/reject)")
    print("       07_load_neo4j loads Wikidata hierarchy + only the rows you mark approved=Y.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
