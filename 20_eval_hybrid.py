#!/usr/bin/env python3
"""
20_eval_hybrid.py  —  Stage 4-A retrieval evaluation (Task R2).

Evaluates all three retrieval modes against all three market-adjusted label sets,
using the test split (split == "test") filtered to focal_shock == True.

Retrieval modes:
  focal_only   — lower-bound baseline
  hybrid_text  — BM25 + BGE-base-en-v1.5 via RRF (no graph)
  hybrid_graph — hybrid_text + time-aware k-hop KG expansion

Label sets (ground truth):
  loo  — Leave-one-out sector benchmark labels
  ff   — Fama-French 5+momentum residual labels (primary)
  pca  — PCA approximate-factor labels

For each (mode × label) combination: P@K, R@K, F1@K, MRR at K∈{5,10,20}.

OUTPUTS
  data/eval/hybrid_results_<label>.json      — full per-mode metrics
  data/eval/hybrid_summary.md                — comparison table across all combinations

USAGE
  python 20_eval_hybrid.py                   # evaluates ff labels (primary)
  python 20_eval_hybrid.py --labels loo ff pca   # all three (full comparative study)
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

LABELS_DIR    = Path("data/labeled")
RETRIEVAL_DIR = Path("data/retrieval")
EVENTS_DIR    = LABELS_DIR
OUT_DIR       = Path("data/eval")

K_VALUES = [5, 10, 20]
MODES    = ["focal_only", "hybrid_text", "hybrid_graph"]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def precision_at_k(predicted: list[str], relevant: set[str], k: int) -> float:
    return sum(1 for t in predicted[:k] if t in relevant) / k if k else 0.0


def recall_at_k(predicted: list[str], relevant: set[str], k: int) -> float:
    return sum(1 for t in predicted[:k] if t in relevant) / len(relevant) if relevant else 0.0


def f1_at_k(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def mrr(predicted: list[str], relevant: set[str]) -> float:
    for i, t in enumerate(predicted):
        if t in relevant:
            return 1.0 / (i + 1)
    return 0.0


def avg(lst: list[float]) -> float:
    return float(np.mean(lst)) if lst else 0.0


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_label_set(label: str, retriever, events_raw: pd.DataFrame) -> dict:
    """
    Evaluate all three retrieval modes against the given label set.
    Returns nested dict: {mode: {metric: value}}.
    """
    pairs_path = LABELS_DIR / f"event_ticker_pairs_{label}.parquet"
    if not pairs_path.exists():
        print(f"[warn] {pairs_path} not found — skipping {label}")
        return {}

    pairs = pd.read_parquet(pairs_path)
    affected_by_event: dict[str, set[str]] = (
        pairs[pairs["is_affected"] == True]  # noqa: E712
        .groupby("event_id")["ticker"]
        .apply(set)
        .to_dict()
    )

    # Use test-split, focal_shock events
    if "focal_shock" in pairs.columns:
        shock_eids = set(pairs[pairs["focal_shock"] == True]["event_id"].unique())  # noqa: E712
    else:
        shock_eids = set(events_raw["event_id"].unique())
        print(f"[warn] focal_shock not in {label} pairs — using all events")

    # Filter events to test split + shock
    eval_eids = set(events_raw[events_raw["split"] == "test"]["event_id"].unique())
    eval_eids = eval_eids & shock_eids

    # Load event contexts for text
    import json as _json
    contexts = {d["event_id"]: d
                for d in [_json.loads(l) for l in open(RETRIEVAL_DIR / "event_contexts.jsonl")]}

    results: dict[str, dict] = {}
    for mode in MODES:
        scores_p  = {k: [] for k in K_VALUES}
        scores_r  = {k: [] for k in K_VALUES}
        scores_f1 = {k: [] for k in K_VALUES}
        scores_mrr: list[float] = []
        skipped = 0

        for _, ev in events_raw[events_raw["event_id"].isin(eval_eids)].iterrows():
            eid     = ev["event_id"]
            primary = ev["primary_ticker"]
            relevant = affected_by_event.get(eid, set()) & retriever.ticker_set
            if not relevant:
                skipped += 1
                continue

            ctx = contexts.get(eid)
            query = ctx["text"] if ctx else str(ev.get("event_text", ""))
            edate = ev.get("event_date", ev.get("filing_date", "2024-01-01"))

            retrieved = retriever.retrieve(
                query, primary, edate, mode=mode, k=max(K_VALUES),
            )
            ranked = [r["ticker"] for r in retrieved]

            scores_mrr.append(mrr(ranked, relevant))
            for k in K_VALUES:
                p = precision_at_k(ranked, relevant, k)
                r = recall_at_k(ranked, relevant, k)
                scores_p[k].append(p)
                scores_r[k].append(r)
                scores_f1[k].append(f1_at_k(p, r))

        n = len(eval_eids) - skipped
        print(f"  [{label}][{mode}] n={n}  skipped={skipped}")
        results[mode] = {
            "n_events": n,
            "MRR": avg(scores_mrr),
            **{f"P@{k}": avg(scores_p[k]) for k in K_VALUES},
            **{f"R@{k}": avg(scores_r[k]) for k in K_VALUES},
            **{f"F1@{k}": avg(scores_f1[k]) for k in K_VALUES},
        }

    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", choices=["loo", "ff", "pca"],
                    default=["ff"], help="Label sets to evaluate against")
    ap.add_argument("--k", nargs="+", type=int, default=K_VALUES)
    ap.add_argument("--max-hops", type=int, default=2)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build retriever — import via spec because module name starts with digit
    print("[info] initialising Retriever ...")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "hybrid_retrieval", Path(__file__).parent / "19_hybrid_retrieval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        retriever = mod.Retriever()

    # Load base events file (for split + event_date + primary_ticker)
    events_raw = pd.read_parquet(LABELS_DIR / "events.parquet")
    events_raw["event_date"] = pd.to_datetime(events_raw["event_date"])
    if "primary_ticker" not in events_raw.columns and "focal_ticker" in events_raw.columns:
        events_raw = events_raw.rename(columns={"focal_ticker": "primary_ticker"})

    all_results: dict[str, dict] = {}
    for label in args.labels:
        print(f"\n[info] evaluating against label={label} ...")
        res = evaluate_label_set(label, retriever, events_raw)
        if res:
            all_results[label] = res
            out = OUT_DIR / f"hybrid_results_{label}.json"
            with open(out, "w") as f:
                json.dump(res, f, indent=2)
            print(f"[ok] {out}")

    if not all_results:
        print("[warn] no results produced")
        return 1

    # Markdown comparison table
    md_lines = [
        "# Stage 4-A Hybrid Retrieval Evaluation",
        "",
        f"**Evaluation:** test split, focal_shock=True  |  **K:** {K_VALUES}",
        "",
    ]

    for label, res in all_results.items():
        md_lines += [f"## Label: {label.upper()}", ""]
        md_lines += [
            "| Mode | MRR | P@5 | P@10 | P@20 | R@5 | R@10 | R@20 | F1@10 |",
            "|------|-----|-----|------|------|-----|------|------|-------|",
        ]
        for mode in MODES:
            if mode not in res:
                continue
            m = res[mode]
            md_lines.append(
                f"| {mode:<14} | {m['MRR']:.4f} | "
                f"{m['P@5']:.4f} | {m['P@10']:.4f} | {m['P@20']:.4f} | "
                f"{m['R@5']:.4f} | {m['R@10']:.4f} | {m['R@20']:.4f} | "
                f"{m['F1@10']:.4f} |"
            )

        # Delta: hybrid_graph vs hybrid_text
        if "hybrid_graph" in res and "hybrid_text" in res:
            g, t = res["hybrid_graph"], res["hybrid_text"]
            md_lines += [
                "",
                "**Δ hybrid_graph vs hybrid_text:**",
                f"  MRR {g['MRR']-t['MRR']:+.4f}  |  "
                f"P@10 {g['P@10']-t['P@10']:+.4f}  |  "
                f"F1@10 {g['F1@10']-t['F1@10']:+.4f}",
            ]
        md_lines.append("")

    md = "\n".join(md_lines)
    summary_path = OUT_DIR / "hybrid_summary.md"
    summary_path.write_text(md)
    print(f"\n[done] summary -> {summary_path}")
    print("\n" + md)
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sys.exit(main())
