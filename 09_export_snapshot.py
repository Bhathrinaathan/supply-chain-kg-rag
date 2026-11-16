#!/usr/bin/env python3
"""
09_export_snapshot.py  —  Versioned, git-committable KG snapshot.

Exports the graph to a dated JSON file (nodes + edges with all properties) so each
experiment can pin an exact KG version. Reads Neo4j if reachable, else the NetworkX mirror.

OUTPUTS:
    data/snapshots/kg_<DATE>.json

USAGE:
    python 09_export_snapshot.py             # prefers Neo4j, falls back to NetworkX
    python 09_export_snapshot.py --from-nx   # force NetworkX mirror
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from datetime import date
from pathlib import Path


def export_from_neo4j(uri: str, user: str, pw: str) -> dict | None:
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(uri, auth=(user, pw))
        nodes, edges = [], []
        with driver.session() as s:
            for rec in s.run("MATCH (c:Company) RETURN c"):
                nodes.append(dict(rec["c"]))
            for rec in s.run(
                "MATCH (a:Company)-[r]->(b:Company) "
                "RETURN a.ticker AS f, b.ticker AS t, type(r) AS ty, properties(r) AS p"
            ):
                edges.append({"from": rec["f"], "to": rec["t"],
                              "type": rec["ty"], **rec["p"]})
        driver.close()
        return {"nodes": nodes, "edges": edges, "source": "neo4j"}
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] Neo4j export failed ({type(exc).__name__}); trying NetworkX mirror.")
        return None


def export_from_nx(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        g = pickle.load(fh)
    nodes = [{"ticker": n, **{k: _json_safe(v) for k, v in d.items()}}
             for n, d in g.nodes(data=True)]
    edges = [{"from": u, "to": v, "type": k, **{kk: _json_safe(vv) for kk, vv in d.items()}}
             for u, v, k, d in g.edges(keys=True, data=True)]
    return {"nodes": nodes, "edges": edges, "source": "networkx"}


def _json_safe(v):
    try:
        json.dumps(v)
        return v
    except TypeError:
        return str(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=Path, default=Path("kg/networkx_export.gpickle"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/snapshots"))
    ap.add_argument("--from-nx", action="store_true")
    ap.add_argument("--uri", default="bolt://localhost:7687")
    ap.add_argument("--user", default="neo4j")
    args = ap.parse_args()

    snap = None
    if not args.from_nx:
        pw = os.environ.get("NEO4J_PASSWORD")
        if pw:
            snap = export_from_neo4j(args.uri, args.user, pw)
    if snap is None:
        snap = export_from_nx(args.nx)
    if snap is None:
        print("[FATAL] no graph found (neither Neo4j nor NetworkX mirror). Run 07 first.")
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"kg_{date.today().isoformat()}.json"
    out.write_text(json.dumps(snap, indent=2, default=str))
    print(f"[done] snapshot ({snap['source']}): "
          f"{len(snap['nodes'])} nodes, {len(snap['edges'])} edges -> {out}")
    print("       git add this file so every experiment can pin an exact KG version.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
