#!/usr/bin/env python3
"""
kp_significance.py  —  Event-clustered bootstrap + BH correction for the hop-1 contagion signal.

Why event-clustered bootstrap (not a standard z-test):
  Events in the same calendar window are cross-sectionally correlated (the K-P 2010 problem).
  A naive proportion test over firm-day pairs treats them as i.i.d., overstating significance.
  Clustering by event and resampling at the EVENT level propagates the within-event correlation
  into the bootstrap distribution, giving an honest test of whether hop-1 > unconnected.

Kolari & Pynnönen 2010 (RFS) parametric form applies to continuous AR; for our binary
is_affected outcome we use the equivalent non-parametric event-clustered bootstrap — the same
principle (cluster at the event level to absorb cross-sectional correlation).

OUTPUTS (per --method / --conditional flag):
  data/labeled/kp_results_<method>[_cond].json   — full bootstrap results
  data/labeled/kp_summary_<method>[_cond].md     — human-readable summary

USAGE
  python kp_significance.py --method loo --conditional
  python kp_significance.py --method loo            # unconditional (no --conditional)
  python kp_significance.py --method ff  --conditional
  python kp_significance.py --method pca --conditional
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


LABELS_DIR = Path("data/labeled")
B          = 1000    # bootstrap iterations
RNG_SEED   = 42
MIN_N      = 100     # flag under-powered cells


def event_clustered_bootstrap(
    events: list[pd.DataFrame],
    B: int,
    rng: np.random.Generator,
) -> tuple[float, np.ndarray]:
    """
    Resample EVENTS with replacement; in each resample compute:
      delta = hop-1 affected rate - unconnected affected rate.
    Returns observed delta and bootstrap distribution of deltas.
    """
    n = len(events)

    def delta_from_events(ev_list: list[pd.DataFrame]) -> float:
        df = pd.concat(ev_list, ignore_index=True)
        h1 = df[df["hop"] == 1]["is_affected"].mean()
        u  = df[df["hop"] < 0]["is_affected"].mean()
        return float(h1 - u) if not (np.isnan(h1) or np.isnan(u)) else np.nan

    observed = delta_from_events(events)
    boot = np.full(B, np.nan)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        boot[b] = delta_from_events([events[i] for i in idx])
    return observed, boot


def bootstrap_p(observed: float, boot: np.ndarray) -> float:
    """One-sided p: fraction of bootstrap samples where delta <= 0 (under H0: no positive lift)."""
    valid = boot[~np.isnan(boot)]
    if len(valid) == 0 or np.isnan(observed):
        return np.nan
    return float((valid <= 0).mean())


def ci95(boot: np.ndarray) -> tuple[float, float]:
    valid = boot[~np.isnan(boot)]
    if len(valid) == 0:
        return np.nan, np.nan
    return float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5))


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Return True for each hypothesis that is rejected at FDR alpha."""
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    reject = [False] * n
    for rank, (i, p) in enumerate(indexed, 1):
        if p <= alpha * rank / n:
            for j in range(rank):
                reject[indexed[j][0]] = True
    return reject


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["loo", "ff", "pca"], default="loo")
    ap.add_argument("--conditional", action="store_true",
                    help="Restrict to events where focal_shock == True")
    ap.add_argument("--out", type=Path, default=LABELS_DIR)
    ap.add_argument("--b", type=int, default=B, help="Bootstrap iterations")
    ap.add_argument("--k", type=float, default=1.5)
    ap.add_argument("--ar", type=Path, default=None,
                    help="Unused (kept for CLI compatibility with stage4 run order)")
    args = ap.parse_args()

    suffix = "_cond" if args.conditional else ""
    pairs_path = args.out / f"event_ticker_pairs_{args.method}.parquet"
    if not pairs_path.exists():
        print(f"[FATAL] {pairs_path} not found")
        return 1

    df = pd.read_parquet(pairs_path)
    print(f"[info] {len(df):,} pairs, method={args.method}, conditional={args.conditional}")

    if args.conditional:
        if "focal_shock" not in df.columns:
            print("[FATAL] focal_shock column missing — re-run 18_relabel_abnormal.py")
            return 1
        df = df[df["focal_shock"] == True]  # noqa: E712
        print(f"[info] filtered to {len(df):,} pairs in focal_shock events "
              f"({df['event_id'].nunique():,} events)")

    # Group into per-event DataFrames for the bootstrap
    event_groups = [grp for _, grp in df.groupby("event_id")]
    print(f"[info] {len(event_groups):,} events entering bootstrap (B={args.b})")

    rng = np.random.default_rng(RNG_SEED)

    # ------------------------------------------------------------------- Overall hop-1 test
    obs, boot = event_clustered_bootstrap(event_groups, args.b, rng)
    p         = bootstrap_p(obs, boot)
    lo, hi    = ci95(boot)

    print(f"\n=== {args.method.upper()} {'CONDITIONAL' if args.conditional else 'UNCONDITIONAL'} ===")
    print(f"  hop-1 vs unconnected lift: {obs*100:+.3f}pp")
    print(f"  95% CI (bootstrap):  [{lo*100:+.3f}pp, {hi*100:+.3f}pp]")
    print(f"  p-value (one-sided): {p:.4f}  {'*' if p<0.05 else '(n.s.)'}")

    results: dict = {
        "method": args.method,
        "conditional": args.conditional,
        "n_events": len(event_groups),
        "n_pairs": int(len(df)),
        "bootstrap_B": args.b,
        "hop1_vs_unconnected": {
            "observed_delta_pp": round(obs * 100, 4),
            "ci95_lo_pp": round(lo * 100, 4),
            "ci95_hi_pp": round(hi * 100, 4),
            "p_one_sided": round(p, 4),
            "significant_05": bool(p < 0.05),
        },
    }

    # ------------------------------------------------------------------- Per-relation-type test (hop-1 only)
    if "rel_types" in df.columns and args.conditional:
        df2 = df.copy()
        # Normalise rel_types: parquet may load as numpy.ndarray, str repr, or None
        def _norm_rtypes(x):
            if x is None:
                return []
            if isinstance(x, str):
                import ast
                try:
                    return ast.literal_eval(x)
                except Exception:
                    return []
            try:
                return list(x)   # numpy.ndarray, list, or any iterable
            except Exception:
                return []
        df2["rel_types"] = df2["rel_types"].apply(_norm_rtypes)

        h1_exp  = df2[df2["hop"] == 1].explode("rel_types")
        h1_exp  = h1_exp[h1_exp["rel_types"].notna() & (h1_exp["rel_types"] != "")]
        rel_types = [r for r in h1_exp["rel_types"].unique() if isinstance(r, str)]

        print(f"\n--- Per-relation-type bootstrap (B={args.b} per type) ---")
        rel_results = {}
        p_values: list[float] = []
        type_names: list[str] = []

        for rtype in sorted(rel_types):
            h1_r = df2[(df2["hop"] == 1) & df2["rel_types"].apply(
                lambda x: x is not None and len(x) > 0 and rtype in x)]
            unc  = df2[df2["hop"] < 0]
            n_h1 = len(h1_r)
            n_u  = len(unc)

            if n_h1 < 10:  # too thin to bootstrap meaningfully
                print(f"  {rtype:<25s}  skip (n={n_h1} < 10)")
                continue

            # Build per-event groups: for each event include its hop-1-rtype pairs + unconnected pairs
            event_ids = df2["event_id"].unique()
            ev_grps_r = []
            for eid in event_ids:
                ev_df = pd.concat([
                    df2[(df2["event_id"] == eid) & (df2["hop"] == 1) &
                        df2["rel_types"].apply(lambda x: x is not None and len(x) > 0 and rtype in x)],
                    df2[(df2["event_id"] == eid) & (df2["hop"] < 0)],
                ], ignore_index=True)
                if len(ev_df) > 0 and (ev_df["hop"] == 1).any() and (ev_df["hop"] < 0).any():
                    ev_grps_r.append(ev_df)

            if len(ev_grps_r) < 10:
                print(f"  {rtype:<25s}  skip (too few events with both hop-1 and unconnected)")
                continue

            obs_r, boot_r = event_clustered_bootstrap(ev_grps_r, args.b, rng)
            p_r = bootstrap_p(obs_r, boot_r)
            lo_r, hi_r = ci95(boot_r)

            underpowered = n_h1 < MIN_N
            flag = "  [⚠ under-powered]" if underpowered else ""
            p_values.append(p_r)
            type_names.append(rtype)
            rel_results[rtype] = {
                "n_hop1": n_h1, "n_unconnected": n_u,
                "observed_delta_pp": round(obs_r * 100, 4),
                "ci95_lo_pp": round(lo_r * 100, 4),
                "ci95_hi_pp": round(hi_r * 100, 4),
                "p_one_sided": round(p_r, 4),
                "underpowered": underpowered,
            }
            print(f"  {rtype:<25s}  delta={obs_r*100:+.2f}pp  "
                  f"CI=[{lo_r*100:+.2f},{hi_r*100:+.2f}]  "
                  f"p={p_r:.4f}  n={n_h1}{flag}")

        # Benjamini-Hochberg correction across relation types
        if p_values:
            bh_reject = benjamini_hochberg(p_values)
            print(f"\n  BH-corrected rejections at FDR=0.05: "
                  f"{sum(bh_reject)}/{len(bh_reject)} relation types")
            for name, rej in zip(type_names, bh_reject):
                if rej:
                    print(f"    [REJECT H0] {name}")
                else:
                    print(f"    [retain]    {name}")
            for name, rej in zip(type_names, bh_reject):
                if name in rel_results:
                    rel_results[name]["bh_reject_fdr05"] = rej

        results["per_relation_type"] = rel_results

    # Write results
    args.out.mkdir(parents=True, exist_ok=True)
    out_json = args.out / f"kp_results_{args.method}{suffix}.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    # Markdown summary
    sig = results["hop1_vs_unconnected"]
    md_lines = [
        f"# KP Bootstrap Results — {args.method.upper()} {'(conditional)' if args.conditional else '(unconditional)'}",
        "",
        f"**N events:** {results['n_events']:,}  |  **N pairs:** {results['n_pairs']:,}  |  **B:** {args.b}",
        "",
        "## Hop-1 vs Unconnected",
        f"| δ (pp) | 95% CI | p (one-sided) | Significant |",
        f"|--------|--------|---------------|-------------|",
        f"| {sig['observed_delta_pp']:+.3f} | [{sig['ci95_lo_pp']:+.3f}, {sig['ci95_hi_pp']:+.3f}] | "
        f"{sig['p_one_sided']:.4f} | {'YES' if sig['significant_05'] else 'no'} |",
        "",
        "*(Event-clustered bootstrap, B=1000; one-sided H₀: δ ≤ 0)*",
    ]
    (args.out / f"kp_summary_{args.method}{suffix}.md").write_text("\n".join(md_lines))
    print(f"\n[done] -> {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
