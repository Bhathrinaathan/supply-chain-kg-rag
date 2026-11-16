#!/usr/bin/env python3
"""
merge_research_edges.py  —  Fold manually-researched edges into the review pipeline.

Takes a JSON array of researched relationships (the format produced by a curation pass),
validates each against the universe + the 7-type schema, re-applies the canonical
single-edge convention for symmetric types, de-duplicates against any existing
candidate file, and writes rows in the SAME schema as 06_build_edges.py's
`candidate_edges.csv` so they pass through the identical human approve/reject gate.

Nothing here auto-approves anything: every row lands with approved="" and edges below
0.9 confidence (i.e. press-release / news sourced) are additionally flagged
`needs_extra_review=Y`.

USAGE:
    python merge_research_edges.py --in research_edges.json \
        --universe config/universe.csv --out review/candidate_research_edges.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

VALID_TYPES = {"supplier_of", "customer_of", "competitor_of", "partner_of",
               "joint_venture_with", "subsidiary_of", "parent_of"}
SYMMETRIC = {"competitor_of", "partner_of", "joint_venture_with"}
REQUIRED = ("from_ticker", "to_ticker", "type", "source_url", "valid_from",
            "raw_quote", "confidence")


def canonicalise(e: dict) -> dict:
    """Symmetric edges: lexicographically smaller ticker becomes `from` (A-B == B-A)."""
    if e["type"] in SYMMETRIC:
        a, b = sorted([e["from_ticker"], e["to_ticker"]])
        e["from_ticker"], e["to_ticker"] = a, b
    return e


def validate(e: dict, universe: set[str]) -> tuple[bool, str]:
    for k in REQUIRED:
        if not str(e.get(k, "")).strip():
            return False, f"missing field: {k}"
    if e["type"] not in VALID_TYPES:
        return False, f"type not in 7-type schema: {e['type']}"
    if e["from_ticker"] not in universe:
        return False, f"from_ticker not in universe: {e['from_ticker']}"
    if e["to_ticker"] not in universe:
        return False, f"to_ticker not in universe: {e['to_ticker']}"
    if e["from_ticker"] == e["to_ticker"]:
        return False, "self-referential edge (from == to)"
    if len(str(e["raw_quote"])) > 250:
        return False, "raw_quote exceeds 250 chars"
    return True, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, default=Path("research_edges.json"))
    ap.add_argument("--universe", type=Path, default=Path("config/universe.csv"))
    ap.add_argument("--out", type=Path, default=Path("review/candidate_research_edges.csv"))
    ap.add_argument("--method", default="manual_research")
    args = ap.parse_args()

    edges = json.loads(args.inp.read_text())

    # Universe membership. If the CSV isn't present (e.g. running standalone), accept all
    # tickers seen in the edges so the tool still demonstrates — but warn loudly.
    if args.universe.exists():
        universe = set(pd.read_csv(args.universe)["ticker"].astype(str))
    else:
        universe = {e[k] for e in edges for k in ("from_ticker", "to_ticker")}
        print(f"[warn] {args.universe} not found — skipping universe membership check.")

    # Existing candidate keys for de-dup, AND preserve prior rows + their approvals.
    seen: set[tuple] = set()
    prev_records: list[dict] = []
    if args.out.exists():
        prev = pd.read_csv(args.out)
        prev_records = prev.to_dict("records")
        seen = {(r.from_ticker, r.to_ticker, r.type, r.source_url)
                for r in prev.itertuples()}

    kept, rejected = [], []
    for e in edges:
        e = canonicalise(dict(e))
        ok, why = validate(e, universe)
        if not ok:
            rejected.append({**e, "reason": why})
            continue
        key = (e["from_ticker"], e["to_ticker"], e["type"], e["source_url"])
        if key in seen:
            continue
        seen.add(key)
        conf = float(e["confidence"])
        kept.append({
            "from_ticker": e["from_ticker"], "to_ticker": e["to_ticker"],
            "type": e["type"], "source_url": e["source_url"],
            "method": args.method, "valid_from": e["valid_from"],
            "confidence": conf, "raw_quote": e["raw_quote"],
            "review_only": True, "approved": "",
            "needs_extra_review": "Y" if conf < 0.9 else "",
            "counterparty_to_fill": False,
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(prev_records + kept)   # keep prior rows + approvals, append new only
    df.to_csv(args.out, index=False)

    by_type = df["type"].value_counts().to_dict() if not df.empty else {}
    print(f"[done] added {len(kept)} new edges ({len(df)} total in file) -> {args.out}")
    print(f"       by type: {by_type}")
    print(f"       flagged needs_extra_review (conf<0.9): {(df['needs_extra_review']=='Y').sum() if not df.empty else 0}")
    if rejected:
        print(f"[warn] {len(rejected)} edges rejected:")
        for r in rejected:
            print(f"        - {r.get('from_ticker')}->{r.get('to_ticker')} "
                  f"[{r.get('type')}]: {r['reason']}")
    print("\n   Next: review approved/needs_extra_review, set approved=Y/N, then append")
    print("   these rows to review/candidate_edges.csv so 07_load_neo4j picks them up.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
