#!/usr/bin/env python3
"""
14_build_retrieval_docs.py  —  Build retrieval documents for RAG/RAFT.

Produces two document corpora:
  1. firm_profiles.jsonl   — one document per universe firm describing its
       KG edges, sector, and recent price volatility. These are the "passages"
       a retrieval system can return.
  2. event_contexts.jsonl  — one document per event (from 12_pull_events.py)
       combining the 8-K text with the primary firm's 1-hop KG neighbourhood.
       Used as the query-side context in RAFT.

OUTPUTS:
    data/retrieval/firm_profiles.jsonl
    data/retrieval/event_contexts.jsonl

USAGE:
    python 14_build_retrieval_docs.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
import warnings

import networkx as nx
import numpy as np
import pandas as pd

PRICES_DIR  = Path("data/raw/prices")
EVENTS_PATH = Path("data/raw/events/events_8k.jsonl")
KG_PATH     = Path("kg/networkx_export.gpickle")
OUT_DIR     = Path("data/retrieval")

ROLLING_WINDOW = 60

EDGE_TYPE_LABELS = {
    "supplier_of":       "supplies to",
    "customer_of":       "buys from",
    "competitor_of":     "competes with",
    "partner_of":        "partners with",
    "joint_venture_with":"has a joint venture with",
    "subsidiary_of":     "is a subsidiary of",
    "parent_of":         "is the parent of",
}

ITEM_LABELS = {
    "1.01": "Entry into Material Agreement",
    "1.02": "Termination of Material Agreement",
    "2.02": "Results of Operations / Earnings",
    "5.02": "Departure or Appointment of Officers/Directors",
    "7.01": "Regulation FD Disclosure",
    "8.01": "Other Material Events",
    "9.01": "Financial Statements and Exhibits",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_kg(path: Path) -> nx.MultiDiGraph:
    with open(path, "rb") as f:
        return pickle.load(f)


def get_close_price(ticker: str, as_of: pd.Timestamp | None = None) -> float | None:
    safe = ticker.replace(".", "_")
    path = PRICES_DIR / f"{safe}__adjusted.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    col = next((c for c in df.columns if isinstance(c, tuple) and c[0] == "Close"), None)
    if col is None:
        col = next((c for c in df.columns if c == "Close"), None)
    if col is None:
        return None
    s = df[col].dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    if as_of is not None:
        s = s[s.index <= as_of]
    return float(s.iloc[-1]) if not s.empty else None


def get_rolling_vol(ticker: str, as_of: pd.Timestamp | None = None) -> float | None:
    safe = ticker.replace(".", "_")
    path = PRICES_DIR / f"{safe}__adjusted.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    col = next((c for c in df.columns if isinstance(c, tuple) and c[0] == "Close"), None)
    if col is None:
        col = next((c for c in df.columns if c == "Close"), None)
    if col is None:
        return None
    s = df[col].dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    if as_of is not None:
        s = s[s.index < as_of]
    rets = np.log(s / s.shift(1)).dropna()
    window = rets.tail(ROLLING_WINDOW)
    if len(window) < 20:
        return None
    return float(window.std())


def describe_items(items_str: str) -> str:
    parts = [p.strip() for p in str(items_str).split(",") if p.strip()]
    labels = [ITEM_LABELS.get(p, p) for p in parts if p in ITEM_LABELS]
    return "; ".join(labels) if labels else items_str


def build_firm_profile_text(
    ticker: str,
    name: str,
    exchange: str,
    country: str,
    sector: str,
    g: nx.MultiDiGraph,
    uni_set: set[str],
    price: float | None,
    vol: float | None,
) -> str:
    """Render a firm's edges + price context as a plain-text passage."""
    lines = [
        f"{ticker} — {name}",
        f"Exchange: {exchange}  Country: {country}  Sector: {sector}",
    ]

    if price is not None:
        price_str = f"${price:.2f}"
    else:
        price_str = "N/A"

    if vol is not None:
        vol_str = f"{vol*100:.2f}% daily"
    else:
        vol_str = "N/A"

    lines.append(f"Recent price: {price_str}  |  60-day rolling volatility: {vol_str}")
    lines.append("Supply chain relationships (universe firms only):")

    # Collect edges by type — only those where BOTH endpoints are in universe
    by_type: dict[str, list[str]] = {k: [] for k in EDGE_TYPE_LABELS}
    if ticker in g:
        for _, nbr, data in g.out_edges(ticker, data=True):
            etype = data.get("type", data.get("relationship", ""))
            if etype in by_type and nbr in uni_set:
                nbr_name = g.nodes[nbr].get("name", nbr) if nbr in g.nodes else nbr
                by_type[etype].append(f"{nbr} ({nbr_name})")
        for src, _, data in g.in_edges(ticker, data=True):
            etype = data.get("type", data.get("relationship", ""))
            # Reverse label
            reverse = {
                "supplier_of": "customer_of",
                "customer_of": "supplier_of",
                "subsidiary_of": "parent_of",
                "parent_of": "subsidiary_of",
            }
            rev_type = reverse.get(etype, etype)
            if rev_type in by_type and src in uni_set:
                src_name = g.nodes[src].get("name", src) if src in g.nodes else src
                by_type[rev_type].append(f"{src} ({src_name})")

    has_any = False
    for etype, label in EDGE_TYPE_LABELS.items():
        peers = sorted(set(by_type[etype]))
        if peers:
            has_any = True
            peers_str = ", ".join(peers[:10])
            if len(peers) > 10:
                peers_str += f" (and {len(peers)-10} more)"
            lines.append(f"  {label}: {peers_str}")

    if not has_any:
        lines.append("  (no edges to other universe firms on record)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    uni = pd.read_csv("config/universe.csv")
    uni_set = set(uni["ticker"].astype(str))
    uni_index = {str(r["ticker"]): r for _, r in uni.iterrows()}

    g = load_kg(KG_PATH)
    print(f"[info] KG: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")

    # ------------------------------------------------------------------ #
    # 1. Firm profiles (one per universe firm, using latest prices)
    # ------------------------------------------------------------------ #
    profiles_path = OUT_DIR / "firm_profiles.jsonl"
    written_profiles = 0
    with open(profiles_path, "w") as fout:
        for ticker in sorted(uni_set):
            row = uni_index.get(ticker, {})
            name     = str(row.get("name", ticker)) if hasattr(row, "get") else ticker
            exchange = str(row.get("exchange", "")) if hasattr(row, "get") else ""
            country  = str(row.get("country", "")) if hasattr(row, "get") else ""
            sector   = str(row.get("sector", "")) if hasattr(row, "get") else ""

            price = get_close_price(ticker)
            vol   = get_rolling_vol(ticker)

            text = build_firm_profile_text(
                ticker, name, exchange, country, sector, g, uni_set, price, vol
            )
            record = {
                "doc_id":   f"firm_{ticker}",
                "ticker":   ticker,
                "doc_type": "firm_profile",
                "text":     text,
            }
            fout.write(json.dumps(record) + "\n")
            written_profiles += 1

    print(f"[ok] {written_profiles} firm profiles -> {profiles_path}")

    # ------------------------------------------------------------------ #
    # 2. Event contexts (event text + primary firm's 1-hop neighbourhood)
    # ------------------------------------------------------------------ #
    if not EVENTS_PATH.exists():
        print(f"[warn] {EVENTS_PATH} not found — skipping event contexts")
        return 0

    events_path = OUT_DIR / "event_contexts.jsonl"
    written_events = 0
    with open(EVENTS_PATH) as fin, open(events_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            ev = json.loads(line)
            primary   = ev["primary_ticker"]
            ev_date   = pd.Timestamp(ev["filing_date"])
            event_text = ev.get("event_text", "").strip()
            items_desc = describe_items(ev.get("items", ""))

            # Collect direct neighbours in universe
            neighbours: list[dict] = []
            if primary in g:
                seen = set()
                for src, tgt, data in list(g.out_edges(primary, data=True)) + list(g.in_edges(primary, data=True)):
                    nbr = tgt if src == primary else src
                    etype = data.get("type", data.get("relationship", ""))
                    if nbr in uni_set and nbr != primary and nbr not in seen:
                        seen.add(nbr)
                        nbr_name = g.nodes[nbr].get("name", nbr) if nbr in g.nodes else nbr
                        neighbours.append({"ticker": nbr, "name": nbr_name, "rel": etype})

            # Build neighbourhood text
            if neighbours:
                nbr_lines = [f"  {n['rel']:20s}: {n['ticker']} ({n['name']})"
                             for n in neighbours[:20]]
                nbr_text = "Direct KG neighbours of " + primary + ":\n" + "\n".join(nbr_lines)
                if len(neighbours) > 20:
                    nbr_text += f"\n  ... and {len(neighbours)-20} more"
            else:
                nbr_text = f"{primary} has no recorded edges to other universe firms."

            # Primary firm's vol at event date
            pv = get_rolling_vol(primary, as_of=ev_date)
            vol_str = f"{pv*100:.2f}%" if pv is not None else "N/A"

            context_text = (
                f"EVENT ({ev['filing_date']}): {primary} filed an 8-K\n"
                f"8-K items: {items_desc}\n"
                f"{primary} 60-day rolling volatility at filing: {vol_str}\n\n"
                f"8-K filing text (excerpt):\n{event_text[:600]}\n\n"
                f"{nbr_text}"
            )

            record = {
                "doc_id":         f"event_{ev['event_id']}",
                "event_id":       ev["event_id"],
                "primary_ticker": primary,
                "filing_date":    ev["filing_date"],
                "doc_type":       "event_context",
                "text":           context_text,
                "neighbour_tickers": [n["ticker"] for n in neighbours],
            }
            fout.write(json.dumps(record) + "\n")
            written_events += 1

    print(f"[ok] {written_events} event contexts -> {events_path}")
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sys.exit(main())
