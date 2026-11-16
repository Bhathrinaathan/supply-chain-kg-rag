#!/usr/bin/env python3
"""
07_load_neo4j.py  —  Load the KG into Neo4j (+ a NetworkX mirror fallback).

Loads Company nodes from config/universe.csv and edges from data/edges/ plus the APPROVED
rows of review/candidate_edges.csv. Provenance is enforced HERE (the DB cannot): any edge
without source_url is rejected. MERGE is idempotent and binds both end nodes first.

NEO4J COMMUNITY EDITION LIMITS (confirmed in the Cypher manual):
    relationship-property-existence, relationship-uniqueness, and node-key constraints are
    ENTERPRISE-ONLY. Only the Company.ticker uniqueness constraint is set in-DB; the rest is
    enforced in this loader and re-checked by 08_verify_kg.

CONNECTION: bolt://localhost:7687, password from env NEO4J_PASSWORD.

OUTPUTS:
    Neo4j graph (local) + kg/networkx_export.gpickle

USAGE:
    pip install neo4j networkx pandas pyarrow
    export NEO4J_PASSWORD=...   # from Neo4j Desktop
    python 07_load_neo4j.py
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd

EDGE_TYPES = ["supplier_of", "customer_of", "competitor_of", "partner_of",
              "joint_venture_with", "subsidiary_of", "parent_of"]

SCHEMA = [
    "CREATE CONSTRAINT company_ticker IF NOT EXISTS "
    "FOR (c:Company) REQUIRE c.ticker IS UNIQUE",
    "CREATE INDEX company_sector IF NOT EXISTS FOR (c:Company) ON (c.sector)",
]

NODE_MERGE = """
MERGE (c:Company {ticker:$ticker})
ON CREATE SET c += $props, c.first_seen=$now
ON MATCH  SET c += $props
"""

# Edge type must be interpolated (Cypher can't parameterise relationship types); it is
# validated against EDGE_TYPES first, so this is safe from injection.
EDGE_MERGE = """
MATCH (a:Company {{ticker:$from}}), (b:Company {{ticker:$to}})
MERGE (a)-[r:{etype} {{source_url:$src, valid_from:$vf}}]->(b)
ON CREATE SET r.verified_at=$now, r.method=$method,
              r.confidence=$conf, r.raw_quote=$quote
ON MATCH  SET r.verified_at=$now
"""


def flatten_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance MultiIndex columns like ('Close','AAPL') -> 'Close'."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df


def load_nodes(session, uni: pd.DataFrame, now: str) -> int:
    n = 0
    for _, r in uni.iterrows():
        props = {k: ("" if pd.isna(r.get(k)) else r.get(k))
                 for k in ("name", "cik", "qid", "sector", "country",
                           "is_adr", "home_exchange", "has_price")}
        session.run(NODE_MERGE, ticker=str(r["ticker"]), props=props, now=now)
        n += 1
    return n


def collect_edges(edges_dir: Path, review_path: Path) -> list[dict]:
    """Wikidata hierarchy edges (auto) + approved rows from ALL review/candidate_*.csv files
    + derived customer_of inverses.

    Reading every candidate_*.csv means manually-researched edges (candidate_research_edges.csv,
    candidate_research_edges_batch2.csv, …) are picked up automatically without touching the
    script-06-generated candidate_edges.csv.  Re-running 06 therefore never clobbers approvals.
    """
    out: list[dict] = []

    for etype in EDGE_TYPES:
        f = edges_dir / f"{etype}.parquet"
        if not f.exists():
            continue
        for _, e in pd.read_parquet(f).iterrows():
            if str(e.get("method")) == "wikidata" and not bool(e.get("review_only", False)):
                out.append(e.to_dict())

    # Collect all candidate review files in the review/ directory.
    review_dir = review_path.parent
    review_files = sorted(review_dir.glob("candidate_*.csv"))
    if not review_files:
        review_files = [review_path] if review_path.exists() else []

    all_approved: list[dict] = []
    for rv_file in review_files:
        if not rv_file.exists():
            continue
        rev = pd.read_csv(rv_file, dtype=str, keep_default_na=False)
        approved = rev[rev["approved"].str.upper() == "Y"].to_dict("records")
        all_approved.extend(approved)
        print(f"[info] {rv_file.name}: {len(approved)} approved rows")

    out.extend(all_approved)

    # Task D: derive customer_of(B→A) for every approved supplier_of(A→B).
    # Inherits parent's provenance; does NOT re-enter any candidate file.
    n_derived = 0
    for e in all_approved:
        if e.get("type") == "supplier_of" and e.get("from_ticker") and e.get("to_ticker"):
            out.append({
                "from_ticker": e["to_ticker"],
                "to_ticker": e["from_ticker"],
                "type": "customer_of",
                "source_url": e["source_url"],
                "method": "derived_inverse",
                "valid_from": e.get("valid_from", ""),
                "confidence": e.get("confidence", 0.5),
                "raw_quote": e.get("raw_quote", ""),
                "review_only": False,
            })
            n_derived += 1
    print(f"[info] Derived {n_derived} customer_of inverse edges from approved supplier_of edges.")

    # provenance gate + valid endpoints + known type
    clean = []
    for e in out:
        if (e.get("source_url") and e.get("from_ticker") and e.get("to_ticker")
                and e.get("type") in EDGE_TYPES):
            clean.append(e)
    return clean


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", type=Path, default=Path("config/universe.csv"))
    ap.add_argument("--edges-dir", type=Path, default=Path("data/edges"))
    ap.add_argument("--review", type=Path, default=Path("review/candidate_edges.csv"))
    ap.add_argument("--nx-out", type=Path, default=Path("kg/networkx_export.gpickle"))
    ap.add_argument("--uri", default="bolt://localhost:7687")
    ap.add_argument("--user", default="neo4j")
    ap.add_argument("--database", default="neo4j")
    args = ap.parse_args()

    pw = os.environ.get("NEO4J_PASSWORD")
    if not pw:
        print("[FATAL] set NEO4J_PASSWORD (from Neo4j Desktop).")
        return 2

    now = date.today().isoformat()
    uni = pd.read_csv(args.universe)
    edges = collect_edges(args.edges_dir, args.review)
    print(f"[info] {len(uni)} nodes, {len(edges)} provenance-clean edges to load.")

    # ---- Neo4j load ----
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(args.uri, auth=(args.user, pw))
        with driver.session(database=args.database) as s:
            for stmt in SCHEMA:
                s.run(stmt)
            n_nodes = load_nodes(s, uni, now)
            n_edges = 0
            for e in edges:
                etype = e["type"]
                if etype not in EDGE_TYPES:           # defensive: never interpolate unknown
                    continue
                s.run(EDGE_MERGE.format(etype=etype),
                      **{"from": e["from_ticker"], "to": e["to_ticker"],
                         "src": e["source_url"], "vf": str(e.get("valid_from", "")),
                         "now": now, "method": e.get("method", ""),
                         "conf": float(e.get("confidence", 0.5) or 0.5),
                         "quote": str(e.get("raw_quote", ""))})
                n_edges += 1
        driver.close()
        print(f"[ok] Neo4j: {n_nodes} nodes, {n_edges} edges merged.")
    except Exception as exc:  # noqa: BLE001 -- Neo4j must never be a hard blocker
        print(f"[warn] Neo4j load failed ({type(exc).__name__}: {exc}).")
        print("       Falling back to NetworkX only -- see runbook 'Neo4j blocked' rule.")

    # ---- NetworkX mirror (always produced; portable fallback) ----
    try:
        import pickle
        import networkx as nx
        g = nx.MultiDiGraph()
        for _, r in uni.iterrows():
            g.add_node(str(r["ticker"]), **{k: r.get(k) for k in ("name", "sector", "country")})
        for e in edges:
            g.add_edge(e["from_ticker"], e["to_ticker"], key=e["type"],
                       method=e.get("method"), source_url=e["source_url"],
                       valid_from=str(e.get("valid_from", "")))
        args.nx_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.nx_out, "wb") as fh:
            pickle.dump(g, fh)
        print(f"[ok] NetworkX mirror -> {args.nx_out} "
              f"({g.number_of_nodes()} nodes, {g.number_of_edges()} edges)")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] NetworkX export failed: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
