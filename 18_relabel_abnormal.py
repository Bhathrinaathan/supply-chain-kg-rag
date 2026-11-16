#!/usr/bin/env python3
"""
18_relabel_abnormal.py  —  Market-adjusted affected labels + KG hop gradient (Stage 3.5, L2/L3).

For each benchmark's abnormal-return series (from 17), and each (event, candidate firm) pair:
  - CAR   = cumulative abnormal return over the event window [d0, d0+H).
  - sigma = pre-event 60-day rolling std of daily AR (info up to d0-1).
  - SCAR  = CAR / (sigma * sqrt(H))          # standardized cumulative abnormal return
  - is_affected = |SCAR| > k                 # k = 1.5 primary, 2.0 sensitivity
  - direction   = sign(CAR)
Then attaches the TIME-AWARE hop distance from the event's focal firm to each candidate
(only KG edges with valid_from <= event_date are traversed — the leakage control), and reports
the affected-rate-by-hop gradient. That gradient table is the input to Decision Gate L4.

Why this matters: the old label used RAW returns (45% base rate, dominated by market/sector
co-movement). Market-adjusted labels should be SPARSE; if the hop-1 lift survives here, the
core hypothesis stands on sound labels.

INPUTS
  data/labeled/benchmarks/{loo,ff,pca}_abnormal_returns.parquet   (from 17)
  data/events/events.parquet          (event_id, focal_ticker, date)   [adapt path/cols as needed]
  kg/networkx_export.gpickle          (from 07; edges carry valid_from)
OUTPUTS
  data/labeled/event_ticker_pairs_<method>.parquet
  data/labeled/hop_gradient_<method>.csv   (+ printed table)

USAGE
  python 18_relabel_abnormal.py --method loo --window 5 --k 1.5
  python 18_relabel_abnormal.py --method loo --k 2.0     # sensitivity
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_kg(path: Path):
    import networkx as nx  # noqa: F401
    with open(path, "rb") as fh:
        return pickle.load(fh)


def hop1_rel_types(g, focal: str, j: str, event_date: pd.Timestamp) -> list[str]:
    """
    Return relation-type keys of direct edges between focal and j (either direction)
    whose valid_from <= event_date. Time-aware: excludes future edges.
    Uses g[u][v] adjacency lookup (O(degree)) instead of scanning all edges.
    """
    types: set[str] = set()
    for u, v in [(focal, j), (j, focal)]:
        if u in g and v in g[u]:
            for key, d in g[u][v].items():
                vf = str(d.get("valid_from", "") or "0001-01-01")[:10]
                try:
                    if pd.to_datetime(vf) <= event_date:
                        types.add(key)
                except Exception:  # noqa: BLE001
                    pass
    return sorted(types)


def hop_distances_asof(g, focal: str, event_date: pd.Timestamp) -> dict[str, int]:
    """
    Undirected BFS hop distance from `focal` over the subgraph of edges whose valid_from
    <= event_date (time-aware leakage control). Returns {ticker: hops}. Unreached omitted.
    """
    import networkx as nx
    keep = nx.Graph()
    keep.add_nodes_from(g.nodes())
    for u, v, d in g.edges(data=True):
        vf = str(d.get("valid_from", "") or "0001-01-01")[:10]
        try:
            ok = pd.to_datetime(vf) <= event_date
        except Exception:  # noqa: BLE001 - undated edge -> treat as unknown, exclude
            ok = False
        if ok:
            keep.add_edge(u, v)
    if focal not in keep:
        return {}
    return nx.single_source_shortest_path_length(keep, focal)


def rolling_sigma(ar_wide: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Pre-event 60-day rolling std of daily AR, shifted so day t uses info up to t-1."""
    return ar_wide.rolling(window, min_periods=window // 2).std().shift(1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["loo", "ff", "pca"], default="loo")
    ap.add_argument("--benchmarks", type=Path, default=Path("data/labeled/benchmarks"))
    ap.add_argument("--events", type=Path, default=Path("data/labeled/events.parquet"))
    ap.add_argument("--kg", type=Path, default=Path("kg/networkx_export.gpickle"))
    ap.add_argument("--out", type=Path, default=Path("data/labeled"))
    ap.add_argument("--window", type=int, default=5, help="event window length H (trading days)")
    ap.add_argument("--k", type=float, default=1.5, help="SCAR threshold (1.5 primary, 2.0 sens.)")
    ap.add_argument("--vol-window", type=int, default=60)
    args = ap.parse_args()

    # --- abnormal returns -> wide panel + rolling sigma ---
    ar_long = pd.read_parquet(args.benchmarks / f"{args.method}_abnormal_returns.parquet")
    ar_long["date"] = pd.to_datetime(ar_long["date"])
    ar_wide = ar_long.pivot(index="date", columns="ticker", values="ar").sort_index()
    sigma = rolling_sigma(ar_wide, args.vol_window)
    trading_days = ar_wide.index
    universe = list(ar_wide.columns)
    H = args.window

    # --- events ---
    events = pd.read_parquet(args.events)
    # events.parquet uses primary_ticker / event_date (from 12_pull_events / 13_label_affected)
    if "focal_ticker" not in events.columns and "primary_ticker" in events.columns:
        events = events.rename(columns={"primary_ticker": "focal_ticker"})
    if "date" not in events.columns and "event_date" in events.columns:
        events = events.rename(columns={"event_date": "date"})
    events["date"] = pd.to_datetime(events["date"])
    events = events[events["focal_ticker"].isin(universe)].reset_index(drop=True)
    print(f"[info] {len(events):,} events with in-universe focal; {len(universe)} candidate firms")

    kg = load_kg(args.kg)
    rows = []
    for ev in events.itertuples():
        # snap event date to the next available trading day
        pos = trading_days.searchsorted(ev.date)
        if pos + H > len(trading_days):
            continue
        d0 = trading_days[pos]
        window_idx = trading_days[pos: pos + H]

        car = ar_wide.loc[window_idx].sum(axis=0)               # CAR per firm over [d0, d0+H)
        sig = sigma.loc[d0]                                     # pre-event sigma per firm
        scar = car / (sig * np.sqrt(H))
        affected = scar.abs() > args.k
        direction = np.sign(car)

        hops = hop_distances_asof(kg, ev.focal_ticker, d0)      # time-aware

        # focal firm's own shock metrics (same benchmark, same window)
        f = ev.focal_ticker
        focal_car_val   = float(car[f])   if f in car.index   and not pd.isna(car[f])  else np.nan
        focal_scar_val  = float(scar[f])  if f in scar.index  and not pd.isna(scar[f]) else np.nan
        focal_shock_val = bool(abs(focal_scar_val) > args.k) if not np.isnan(focal_scar_val) else False
        focal_dir_val   = int(np.sign(focal_car_val)) if not np.isnan(focal_car_val) else 0

        for j in universe:
            if j == ev.focal_ticker or pd.isna(scar.get(j)):
                continue
            hop_j = int(hops.get(j, -1))
            rel = hop1_rel_types(kg, f, j, d0) if hop_j == 1 else []
            rows.append({
                "event_id": getattr(ev, "event_id", ev.Index),
                "focal_ticker": ev.focal_ticker, "ticker": j,
                "event_date": d0, "car": float(car[j]), "scar": float(scar[j]),
                "is_affected": bool(affected[j]), "direction": int(direction[j]),
                "hop": hop_j,
                "focal_car": focal_car_val, "focal_scar": focal_scar_val,
                "focal_shock": focal_shock_val, "focal_direction": focal_dir_val,
                "rel_types": rel,             # list; non-empty only for hop-1
            })

    pairs = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(args.out / f"event_ticker_pairs_{args.method}.parquet")

    # --- base rate + hop gradient (the L4 gate input) ---
    base = pairs["is_affected"].mean()
    def bucket(h: int) -> str:
        return "unconnected" if h < 0 else ("hop-1" if h == 1 else
               "hop-2" if h == 2 else "hop-3+" if h >= 3 else "focal")
    pairs["bucket"] = pairs["hop"].map(bucket)
    grad = (pairs.groupby("bucket")["is_affected"]
            .agg(["mean", "count"]).rename(columns={"mean": "affected_rate"}))
    order = ["hop-1", "hop-2", "hop-3+", "unconnected"]
    grad = grad.reindex([b for b in order if b in grad.index])
    grad["lift_vs_unconnected_pp"] = (
        (grad["affected_rate"] - grad.loc["unconnected", "affected_rate"]) * 100
        if "unconnected" in grad.index else np.nan)
    grad.to_csv(args.out / f"hop_gradient_{args.method}.csv")

    print(f"\n=== {args.method.upper()}  (window={H}, k={args.k}) ===")
    print(f"pairs: {len(pairs):,}   base affected rate: {base:.1%}   "
          f"(target: well below the old 45%)")
    print(grad.to_string(formatters={"affected_rate": "{:.1%}".format,
                                     "lift_vs_unconnected_pp": "{:+.1f}".format}))
    print("\n[gate L4] Does hop-1 show a positive, monotonic lift here AND across ff/pca?")
    print("          If yes -> log primary benchmark in decisions_log.md, proceed to Stage 4-A.")
    print("          If no  -> STOP; do not fine-tune. Reframe with supervisor.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
