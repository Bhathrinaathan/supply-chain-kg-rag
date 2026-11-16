#!/usr/bin/env python3
"""
13_label_affected.py  —  Label which universe firms were affected by each event.

For every 8-K event from 12_pull_events.py:
  1. Compute 60-day rolling vol (σ) for every universe ticker as of the day
     before the event.
  2. Look at daily log-returns for +1 to +5 trading days after the event.
  3. A ticker is AFFECTED if max(|return|) in that window > 1.5 × σ.
  4. Annotate each affected ticker with its KG hop-distance from the primary firm.
  5. Assign train / test / holdout split by event date.

With --abnormal: subtract the daily cross-sectional mean return of the full universe
from each ticker's return before computing vol and threshold. This removes market-wide
co-movement and isolates supply-chain-specific contagion (standard event-study AR method).

Locked parameters (decisions_log.md — do not change here):
    Rolling-vol window : 60 trading days
    Threshold          : 1.5 × rolling σ  (sensitivity re-run at 2×)
    T1 (train cutoff)  : 2024-07-01
    T2 (test cutoff)   : 2025-07-01

OUTPUTS:
    data/labeled/event_ticker_pairs.parquet        — raw returns
    data/labeled/events.parquet                    — raw returns
    data/labeled/event_ticker_pairs_ar.parquet     — abnormal returns (--abnormal)
    data/labeled/events_ar.parquet                 — abnormal returns (--abnormal)

USAGE:
    python 13_label_affected.py
    python 13_label_affected.py --threshold 2.0        # sensitivity run
    python 13_label_affected.py --abnormal             # abnormal return labels
    python 13_label_affected.py --abnormal --threshold 1.5
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROLLING_WINDOW = 60       # trading days for vol baseline
EVENT_WINDOW   = 5        # +1 to +N trading days post-event
T1 = pd.Timestamp("2024-07-01")
T2 = pd.Timestamp("2025-07-01")

PRICES_DIR = Path("data/raw/prices")
EVENTS_PATH = Path("data/raw/events/events_8k.jsonl")
KG_PATH     = Path("kg/networkx_export.gpickle")
OUT_DIR     = Path("data/labeled")


# ---------------------------------------------------------------------------
# Price helpers
# ---------------------------------------------------------------------------

def load_prices(universe_tickers: list[str], prices_dir: Path = PRICES_DIR) -> dict[str, pd.Series]:
    """Return {ticker: Series of adjusted close prices, indexed by date}."""
    prices: dict[str, pd.Series] = {}
    for ticker in universe_tickers:
        safe = ticker.replace(".", "_")
        path = prices_dir / f"{safe}__adjusted.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        # MultiIndex columns: ('Close', TICKER) — flatten
        col = None
        for c in df.columns:
            if isinstance(c, tuple) and c[0] == "Close":
                col = c
                break
            if c == "Close":
                col = c
                break
        if col is None:
            continue
        s = df[col].dropna()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        prices[ticker] = s.sort_index()
    return prices


def log_returns(price: pd.Series) -> pd.Series:
    return np.log(price / price.shift(1)).dropna()


def build_market_returns(prices: dict[str, pd.Series]) -> pd.Series:
    """Daily cross-sectional mean log-return across all universe tickers."""
    all_rets = pd.DataFrame({t: log_returns(ps) for t, ps in prices.items()})
    return all_rets.mean(axis=1)


def abnormal_returns(ticker_rets: pd.Series, market_rets: pd.Series) -> pd.Series:
    """Idiosyncratic return: ticker return minus universe-average return on same day."""
    aligned = ticker_rets.reindex(market_rets.index)
    ar = aligned - market_rets
    return ar.dropna()


# ---------------------------------------------------------------------------
# KG helpers
# ---------------------------------------------------------------------------

def load_kg_undirected(path: Path) -> nx.Graph:
    with open(path, "rb") as f:
        g = pickle.load(f)
    # Convert to undirected so shock propagation flows both ways
    return g.to_undirected() if g.is_directed() else g


def build_distance_map(g: nx.Graph, source: str, universe: set[str]) -> dict[str, int]:
    """Shortest path distance from source to all universe tickers (BFS, unweighted)."""
    if source not in g:
        return {}
    try:
        lengths = nx.single_source_shortest_path_length(g, source)
    except Exception:
        return {}
    return {t: lengths[t] for t in universe if t in lengths and t != source}


# ---------------------------------------------------------------------------
# Core labelling
# ---------------------------------------------------------------------------

def next_trading_day(date: pd.Timestamp, index: pd.DatetimeIndex) -> pd.Timestamp | None:
    """Return the first date in index that is >= date."""
    future = index[index >= date]
    return future[0] if len(future) else None


def label_event(
    event_date: pd.Timestamp,
    primary: str,
    prices: dict[str, pd.Series],
    universe_tickers: list[str],
    threshold: float,
    market_rets: pd.Series | None = None,
) -> list[dict]:
    """
    Compute per-ticker labels for one event.
    Returns list of dicts — one per universe ticker.
    """
    rows = []
    for ticker in universe_tickers:
        if ticker not in prices:
            rows.append({
                "secondary_ticker": ticker,
                "is_affected":      None,
                "max_abs_return":   None,
                "rolling_vol":      None,
                "threshold_vol":    None,
                "has_price":        False,
            })
            continue

        ps = prices[ticker]
        raw_rets = log_returns(ps)
        rets = abnormal_returns(raw_rets, market_rets) if market_rets is not None else raw_rets

        # Rolling vol: last ROLLING_WINDOW returns strictly before event_date
        pre = rets[rets.index < event_date].tail(ROLLING_WINDOW)
        if len(pre) < ROLLING_WINDOW // 2:
            rows.append({
                "secondary_ticker": ticker,
                "is_affected":      None,
                "max_abs_return":   None,
                "rolling_vol":      None,
                "threshold_vol":    None,
                "has_price":        True,
            })
            continue

        rv = pre.std()
        if rv == 0 or np.isnan(rv):
            rows.append({
                "secondary_ticker": ticker,
                "is_affected":      None,
                "max_abs_return":   None,
                "rolling_vol":      float(rv) if not np.isnan(rv) else None,
                "threshold_vol":    None,
                "has_price":        True,
            })
            continue

        thresh = threshold * rv

        # Post-event returns: next N trading days after event_date
        post_idx = ps.index[ps.index > event_date]
        post_rets = rets[rets.index.isin(post_idx[:EVENT_WINDOW])]
        if post_rets.empty:
            rows.append({
                "secondary_ticker": ticker,
                "is_affected":      None,
                "max_abs_return":   None,
                "rolling_vol":      float(rv),
                "threshold_vol":    float(thresh),
                "has_price":        True,
            })
            continue

        max_abs = float(post_rets.abs().max())
        is_affected = bool(max_abs > thresh)

        rows.append({
            "secondary_ticker": ticker,
            "is_affected":      is_affected,
            "max_abs_return":   max_abs,
            "rolling_vol":      float(rv),
            "threshold_vol":    float(thresh),
            "has_price":        True,
        })
    return rows


def assign_split(date: pd.Timestamp) -> str:
    if date < T1:
        return "train"
    elif date < T2:
        return "test"
    else:
        return "holdout"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events",    type=Path, default=EVENTS_PATH)
    ap.add_argument("--kg",        type=Path, default=KG_PATH)
    ap.add_argument("--universe",  type=Path, default=Path("config/universe.csv"))
    ap.add_argument("--prices-dir", type=Path, default=PRICES_DIR)
    ap.add_argument("--out-dir",   type=Path, default=OUT_DIR)
    ap.add_argument("--threshold", type=float, default=1.5,
                    help="Vol multiplier for affected label (default 1.5)")
    ap.add_argument("--abnormal", action="store_true",
                    help="Use abnormal returns (raw minus universe daily mean) to remove market noise")
    args = ap.parse_args()

    if not args.events.exists():
        print(f"[FATAL] {args.events} not found — run 12_pull_events.py first")
        return 2

    # Load universe
    uni = pd.read_csv(args.universe)
    universe_tickers = uni["ticker"].astype(str).tolist()
    universe_set = set(universe_tickers)
    print(f"[info] universe: {len(universe_tickers)} tickers")

    # Load prices
    prices = load_prices(universe_tickers, args.prices_dir)
    print(f"[info] price series loaded: {len(prices)}/{len(universe_tickers)}")

    market_rets: pd.Series | None = None
    if args.abnormal:
        market_rets = build_market_returns(prices)
        print(f"[info] abnormal-return mode: market index from {len(prices)} tickers")

    # Load KG
    g = load_kg_undirected(args.kg)
    print(f"[info] KG: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")

    # Load events
    events: list[dict] = []
    with open(args.events) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    print(f"[info] events loaded: {len(events)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    pair_rows: list[dict] = []
    event_rows: list[dict] = []

    for ev in events:
        primary   = ev["primary_ticker"]
        ev_date   = pd.Timestamp(ev["filing_date"])
        split     = assign_split(ev_date)
        event_id  = ev["event_id"]

        # KG distances from primary ticker
        dist_map = build_distance_map(g, primary, universe_set)

        # Per-ticker labels
        ticker_labels = label_event(ev_date, primary, prices, universe_tickers,
                                    args.threshold, market_rets)

        affected_tickers = []
        kg_reachable_affected = []

        for lbl in ticker_labels:
            sec = lbl["secondary_ticker"]
            if sec == primary:
                continue
            kg_dist = dist_map.get(sec, None)

            pair_rows.append({
                "event_id":         event_id,
                "primary_ticker":   primary,
                "event_date":       ev_date,
                "secondary_ticker": sec,
                "is_affected":      lbl["is_affected"],
                "max_abs_return":   lbl["max_abs_return"],
                "rolling_vol":      lbl["rolling_vol"],
                "threshold_vol":    lbl["threshold_vol"],
                "has_price":        lbl["has_price"],
                "kg_distance":      kg_dist,
                "split":            split,
            })

            if lbl["is_affected"] is True:
                affected_tickers.append(sec)
                if kg_dist is not None:
                    kg_reachable_affected.append(sec)

        event_rows.append({
            "event_id":                event_id,
            "primary_ticker":          primary,
            "event_date":              ev_date,
            "filing_date":             ev["filing_date"],
            "items":                   ev.get("items", ""),
            "form_url":                ev.get("form_url", ""),
            "event_text":              ev.get("event_text", ""),
            "affected_count":          len(affected_tickers),
            "affected_tickers":        ",".join(affected_tickers),
            "kg_reachable_affected":   ",".join(kg_reachable_affected),
            "kg_reachable_count":      len(kg_reachable_affected),
            "split":                   split,
        })

    pair_df  = pd.DataFrame(pair_rows)
    event_df = pd.DataFrame(event_rows)

    suffix = "_ar" if args.abnormal else ""
    pair_path  = args.out_dir / f"event_ticker_pairs{suffix}.parquet"
    event_path = args.out_dir / f"events{suffix}.parquet"
    pair_df.to_parquet(pair_path,  index=False)
    event_df.to_parquet(event_path, index=False)

    # Summary
    total_events   = len(event_df)
    with_affected  = (event_df["affected_count"] > 0).sum()
    with_kg        = (event_df["kg_reachable_count"] > 0).sum()
    label_mode = "abnormal returns (AR)" if args.abnormal else "raw returns"
    print(f"\n[done] threshold={args.threshold}×σ  mode={label_mode}")
    print(f"  events:               {total_events}")
    print(f"  with ≥1 affected:     {with_affected} ({with_affected/total_events*100:.1f}%)")
    print(f"  with KG-reachable:    {with_kg} ({with_kg/total_events*100:.1f}%)")
    if total_events > 0:
        split_counts = event_df["split"].value_counts()
        for sp, n in split_counts.items():
            print(f"  {sp:10s}:          {n}")
    print(f"\n  -> {pair_path}")
    print(f"  -> {event_path}")
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sys.exit(main())
