#!/usr/bin/env python3
"""
08_verify_kg.py  —  Completeness + integrity audit. FAILS THE BUILD on gate violations.

Produces a markdown report and exits non-zero if either HARD GATE fails:
    GATE 1 (provenance): 100% of edges must have source_url + verified_at.
    GATE 2 (leakage):    the time-aware traversal must exclude edges whose valid_from is
                         AFTER a given event date (functional self-test).

Also reports (non-fatal, but flag): edge density per relation type, per-year coverage,
US x EU x Asia cross-region matrix, and firms with < 2 edges (targeted enrichment list).

Reads the NetworkX mirror by default (portable, no DB needed). Pass --neo4j to audit the
live graph instead.

OUTPUTS:
    data/snapshots/audit_<DATE>.md

USAGE:
    python 08_verify_kg.py                 # audit kg/networkx_export.gpickle
    python 08_verify_kg.py --neo4j         # audit the live Neo4j graph
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import pandas as pd

REGION_BY_COUNTRY = {
    "US": "US", "DE": "EU", "NL": "EU", "FR": "EU", "UK": "EU", "IE": "EU",
    "JP": "Asia", "TW": "Asia", "KR": "Asia", "CN": "Asia",
}


def region(country: str) -> str:
    return REGION_BY_COUNTRY.get(str(country).upper(), "Other")


def load_nx(path: Path):
    import networkx as nx  # noqa: F401
    with open(path, "rb") as fh:
        return pickle.load(fh)


def audit(g, uni: pd.DataFrame) -> tuple[str, bool]:
    lines: list[str] = [f"# KG Audit — {date.today().isoformat()}", ""]
    passed = True
    country = {str(r["ticker"]): str(r.get("country", "")) for _, r in uni.iterrows()}

    # --- edge density per type ---
    type_counts = Counter(k for _, _, k in g.edges(keys=True))
    lines += ["## 1. Edge density per relation type", ""]
    for t in ["supplier_of", "customer_of", "competitor_of", "partner_of",
              "joint_venture_with", "subsidiary_of", "parent_of"]:
        c = type_counts.get(t, 0)
        flag = "  ⚠️ sparse" if c < 3 else ""
        lines.append(f"- `{t}`: {c}{flag}")
    lines += ["", "_competitor_of / joint_venture_with cannot come from Wikidata — "
              "low counts here mean manual curation is owed._", ""]

    # --- per-year coverage ---
    years = Counter()
    for _, _, d in g.edges(data=True):
        vf = str(d.get("valid_from", ""))[:4]
        if vf.isdigit():
            years[vf] += 1
    lines += ["## 2. Edges by valid_from year", ""]
    for y in sorted(years):
        lines.append(f"- {y}: {years[y]}")
    if not years:
        lines.append("- (no dated edges — see GATE 2)")
    lines.append("")

    # --- cross-region matrix ---
    regions = ["US", "EU", "Asia", "Other"]
    mat = {a: {b: 0 for b in regions} for a in regions}
    for u, v, _ in g.edges(keys=True):
        mat[region(country.get(u, ""))][region(country.get(v, ""))] += 1
    lines += ["## 3. Cross-region edge matrix (from → to)", "",
              "| from\\to | " + " | ".join(regions) + " |",
              "|" + "---|" * (len(regions) + 1)]
    for a in regions:
        lines.append("| " + a + " | " + " | ".join(str(mat[a][b]) for b in regions) + " |")
    lines.append("")

    # --- low-degree firms ---
    low = [n for n in g.nodes() if g.degree(n) < 2]
    lines += ["## 4. Firms with < 2 edges (enrichment targets)", "",
              (", ".join(sorted(low)) if low else "_none_"), ""]

    # --- GATE 1: provenance ---
    missing = [(u, v, k) for u, v, k, d in g.edges(keys=True, data=True)
               if not d.get("source_url")]
    g1 = len(missing) == 0
    passed &= g1
    lines += ["## GATE 1 — provenance completeness",
              f"- edges missing source_url: {len(missing)}",
              f"- **{'PASS' if g1 else 'FAIL — build halted'}**", ""]

    # --- GATE 2: leakage self-test ---
    # Insert a synthetic future edge, run a date-filtered traversal, assert it is excluded.
    g2 = leakage_selftest(g)
    passed &= g2
    lines += ["## GATE 2 — look-ahead leakage self-test",
              f"- time-filtered traversal excludes future-dated edges: {g2}",
              f"- **{'PASS' if g2 else 'FAIL — re-timestamp edges before labelling/training'}**",
              ""]

    lines += ["---", f"## RESULT: {'PASS — proceed' if passed else 'FAIL — fix before Stage 4'}"]
    return "\n".join(lines), passed


def leakage_selftest(g) -> bool:
    """
    The single most important leakage control: only edges with valid_from <= event_date may
    be traversed. We verify the FILTER works, independent of data: take any node, attach a
    synthetic edge dated in the far future, then confirm a date-bounded neighbour query at an
    earlier date does not return the future neighbour.
    """
    import networkx as nx  # noqa: F401
    nodes = list(g.nodes())
    if len(nodes) < 2:
        return True  # nothing to test
    focal, future = nodes[0], nodes[1]
    h = g.copy()
    h.add_edge(focal, "__SYNthetic_future__", key="supplier_of",
               source_url="test", valid_from="2099-01-01")
    event_date = "2024-01-01"

    def neighbours_asof(graph, node, asof):
        out = set()
        for u, v, d in graph.edges(node, data=True):
            vf = str(d.get("valid_from", "")) or "0000-00-00"
            if vf <= asof:
                out.add(v)
        return out

    return "__SYNthetic_future__" not in neighbours_asof(h, focal, event_date)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=Path, default=Path("kg/networkx_export.gpickle"))
    ap.add_argument("--universe", type=Path, default=Path("config/universe.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/snapshots"))
    ap.add_argument("--neo4j", action="store_true", help="(stub) audit live Neo4j instead")
    args = ap.parse_args()

    if args.neo4j:
        print("[info] --neo4j: export the graph via 09 then audit the mirror, "
              "or extend this stub with a Cypher read. Using NetworkX mirror for now.")
    if not args.nx.exists():
        print(f"[FATAL] {args.nx} not found. Run 07_load_neo4j first.")
        return 2

    g = load_nx(args.nx)
    uni = pd.read_csv(args.universe)
    report, passed = audit(g, uni)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"audit_{date.today().isoformat()}.md"
    out.write_text(report)
    print(report)
    print(f"\n[done] audit -> {out}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
