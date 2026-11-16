#!/usr/bin/env python3
"""
18b_conditional_gradient.py  —  Conditional focal-shock gradient + per-relation-type breakdown.

PRE-REGISTERED analysis (see decisions_log.md entry 2026-07-03):
  Restrict to events where the focal firm itself was shocked (|focal_SCAR| > k, same k as label).
  Within those events compare hop-1 vs unconnected affected rates — the signal now measures
  genuine supply-chain propagation, not diluted-by-routine-filings co-movement.

THREE OUTPUTS per benchmark:
  1. Conditional gradient table (unconditional vs conditional side-by-side)
  2. Per-relation-type gradient (hop-1, conditional)
  3. Directional concordance (among affected hop-1 pairs, share with same direction as focal)

Written to:
  data/labeled/conditional_gradient_<method>.csv
  data/labeled/relation_type_gradient_<method>.csv

USAGE
  python 18b_conditional_gradient.py --method loo
  python 18b_conditional_gradient.py --method ff
  python 18b_conditional_gradient.py --method pca
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


LABELS_DIR = Path("data/labeled")
MIN_N = 100   # flag cells below this as under-powered


def bucket(h: int) -> str:
    if h < 0:
        return "unconnected"
    if h == 1:
        return "hop-1"
    if h == 2:
        return "hop-2"
    return "hop-3+"


def gradient_table(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Affected-rate by hop bucket; returns tidy DataFrame with label column."""
    order = ["hop-1", "hop-2", "hop-3+", "unconnected"]
    df = df.copy()
    df["bucket"] = df["hop"].map(bucket)
    g = (df.groupby("bucket")["is_affected"]
           .agg(affected_rate="mean", count="count")
           .reindex(order)
           .dropna(how="all"))
    base = g.loc["unconnected", "affected_rate"] if "unconnected" in g.index else np.nan
    g["lift_pp"] = (g["affected_rate"] - base) * 100
    g.insert(0, "condition", label)
    return g


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["loo", "ff", "pca"], default="loo")
    ap.add_argument("--out", type=Path, default=LABELS_DIR)
    ap.add_argument("--k", type=float, default=1.5,
                    help="focal shock threshold (same as label k)")
    ap.add_argument("--min-n", type=int, default=MIN_N,
                    help="flag relation-type cells below this N as under-powered")
    args = ap.parse_args()

    pairs_path = args.out / f"event_ticker_pairs_{args.method}.parquet"
    if not pairs_path.exists():
        print(f"[FATAL] {pairs_path} not found — run 18_relabel_abnormal.py --method {args.method} first")
        return 1

    df = pd.read_parquet(pairs_path)
    print(f"[info] loaded {len(df):,} pairs for method={args.method}")

    # Verify enriched fields exist
    required = {"focal_scar", "focal_shock", "focal_direction", "focal_car", "rel_types"}
    missing = required - set(df.columns)
    if missing:
        print(f"[FATAL] pairs file missing enriched fields: {missing}")
        print("        Re-run: python 18_relabel_abnormal.py --method", args.method)
        return 1

    # Ensure rel_types is a list (may be stored as string repr or pyarrow list)
    if df["rel_types"].dtype == object and len(df) > 0 and isinstance(df["rel_types"].iloc[0], str):
        import ast
        df["rel_types"] = df["rel_types"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # ------------------------------------------------------------------- 1. Gradient tables
    unconditional = gradient_table(df, "unconditional")
    conditional   = gradient_table(df[df["focal_shock"] == True], "conditional (|focal SCAR|>k)")  # noqa: E712

    n_shock_events = df[df["focal_shock"] == True]["event_id"].nunique()
    n_total_events = df["event_id"].nunique()
    print(f"\n[info] focal_shock events: {n_shock_events:,} / {n_total_events:,} "
          f"({n_shock_events/n_total_events:.1%})")

    combined = pd.concat([unconditional, conditional])
    combined.to_csv(args.out / f"conditional_gradient_{args.method}.csv")

    print(f"\n=== {args.method.upper()} — unconditional vs conditional (|focal SCAR| > {args.k}) ===")
    print(combined.to_string(
        formatters={"affected_rate": "{:.3%}".format, "lift_pp": "{:+.2f}".format}))

    # ------------------------------------------------------------------- 2. Per-relation-type (hop-1, conditional)
    cond = df[df["focal_shock"] == True]
    h1   = cond[cond["hop"] == 1].copy()
    h1_exp = h1.explode("rel_types")
    h1_exp = h1_exp[h1_exp["rel_types"].notna() & (h1_exp["rel_types"] != "")]

    base_rate = cond[cond["hop"] < 0]["is_affected"].mean()  # unconnected, same focal-shock events
    base_n    = (cond["hop"] < 0).sum()

    by_type = (h1_exp.groupby("rel_types")["is_affected"]
                     .agg(affected_rate="mean", count="count")
                     .sort_values("count", ascending=False))
    by_type["lift_pp"]       = (by_type["affected_rate"] - base_rate) * 100
    by_type["underpowered"]  = by_type["count"] < args.min_n
    by_type.to_csv(args.out / f"relation_type_gradient_{args.method}.csv")

    print(f"\n--- Per-relation-type gradient (hop-1, conditional, k={args.k}) ---")
    print(f"    baseline (unconnected, conditional): {base_rate:.3%}  n={base_n:,}")
    print(f"    [⚠] = N < {args.min_n} — under-powered, do not over-interpret\n")
    for rtype, row in by_type.iterrows():
        flag = "  [⚠ under-powered]" if row["underpowered"] else ""
        print(f"  {rtype:<25s}  {row['affected_rate']:.3%}  "
              f"n={int(row['count']):>6,}  lift={row['lift_pp']:+.2f}pp{flag}")

    # ------------------------------------------------------------------- 3. Directional concordance
    print(f"\n--- Directional concordance (hop-1, conditional, is_affected=True) ---")
    h1_affected = h1[h1["is_affected"] == True]
    if len(h1_affected) == 0:
        print("  [warn] no affected hop-1 pairs in conditional subset")
    else:
        concordant = (h1_affected["direction"] == h1_affected["focal_direction"]).mean()
        print(f"  Overall hop-1 concordance: {concordant:.3%}  "
              f"(n={len(h1_affected):,}, > 50% = same-direction as focal)")
        for rtype, grp in h1_affected.groupby(h1_affected["rel_types"].apply(
                lambda x: x[0] if x is not None and len(x) > 0 else "unknown")):
            c = (grp["direction"] == grp["focal_direction"]).mean()
            flag = "  [⚠ under-powered]" if len(grp) < args.min_n else ""
            print(f"  {rtype:<25s}  {c:.3%}  n={len(grp):>5,}{flag}")

    print(f"\n[done] {args.method}: outputs in {args.out}")
    print("       next: python kp_significance.py --method", args.method, "--conditional")
    return 0


if __name__ == "__main__":
    sys.exit(main())
