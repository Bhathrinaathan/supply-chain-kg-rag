#!/usr/bin/env python3
"""
16_eval_rag_baseline.py  —  RAG baseline vs KG-augmented retrieval evaluation.

Measures how well two retrieval strategies predict affected tickers:

  BASELINE (BM25, no KG):
    Query = event text (8-K excerpt).
    Retrieve top-K firm profiles by BM25 score.
    Predict: retrieved firm tickers.

  KG-AUGMENTED:
    Start from primary firm's 1-hop KG neighbours.
    Rank by (hop-distance, rolling-vol) to simulate graph-guided retrieval.
    Predict: top-K KG neighbours.

Evaluation (per event, primary_moved=True, test split):
  Precision@K, Recall@K, F1@K, MRR  vs ground truth (is_affected=True tickers).

OUTPUTS:
    data/eval/baseline_results.json
    data/eval/kg_results.json
    data/eval/eval_summary.md

USAGE:
    pip install rank-bm25
    python 16_eval_rag_baseline.py
    python 16_eval_rag_baseline.py --abnormal    # use abnormal-return labels
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import sys
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    print("[FATAL] pip install rank-bm25")
    sys.exit(2)

RETRIEVAL_DIR = Path("data/retrieval")
LABELS_DIR    = Path("data/labeled")
OUT_DIR       = Path("data/eval")
KG_PATH       = Path("kg/networkx_export.gpickle")

K_VALUES = [5, 10, 20]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-zA-Z0-9\.]+\b", text.lower())


def precision_at_k(predicted: list[str], relevant: set[str], k: int) -> float:
    top = predicted[:k]
    return sum(1 for t in top if t in relevant) / k if k else 0.0


def recall_at_k(predicted: list[str], relevant: set[str], k: int) -> float:
    top = predicted[:k]
    return sum(1 for t in top if t in relevant) / len(relevant) if relevant else 0.0


def f1_at_k(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def reciprocal_rank(predicted: list[str], relevant: set[str]) -> float:
    for i, t in enumerate(predicted):
        if t in relevant:
            return 1.0 / (i + 1)
    return 0.0


def load_kg(path: Path) -> nx.Graph:
    with open(path, "rb") as f:
        g = pickle.load(f)
    return g.to_undirected() if g.is_directed() else g


# ---------------------------------------------------------------------------
# Retrieval strategies
# ---------------------------------------------------------------------------

def build_bm25(firm_profiles: list[dict]) -> tuple[BM25Okapi, list[str]]:
    """Build BM25 index over firm profile texts."""
    corpus = [tokenize(d["text"]) for d in firm_profiles]
    tickers = [d["ticker"] for d in firm_profiles]
    return BM25Okapi(corpus), tickers


def bm25_retrieve(
    bm25: BM25Okapi,
    tickers: list[str],
    query_text: str,
    k: int,
    exclude: set[str],
) -> list[str]:
    scores = bm25.get_scores(tokenize(query_text))
    ranked = sorted(zip(tickers, scores), key=lambda x: -x[1])
    return [t for t, _ in ranked if t not in exclude][:k]


def kg_retrieve(
    g: nx.Graph,
    primary: str,
    k: int,
    exclude: set[str],
) -> list[str]:
    """Return universe-firm neighbours sorted by hop distance."""
    if primary not in g:
        return []
    try:
        lengths = nx.single_source_shortest_path_length(g, primary, cutoff=4)
    except Exception:
        return []
    ranked = sorted(
        [(t, d) for t, d in lengths.items() if t not in exclude and t != primary],
        key=lambda x: x[1],
    )
    return [t for t, _ in ranked[:k]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--abnormal", action="store_true",
                    help="Evaluate against abnormal-return labels (events_ar.parquet)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    firm_profiles  = load_jsonl(RETRIEVAL_DIR / "firm_profiles.jsonl")
    event_contexts = {d["event_id"]: d for d in load_jsonl(RETRIEVAL_DIR / "event_contexts.jsonl")}

    suffix = "_ar" if args.abnormal else ""
    events = pd.read_parquet(LABELS_DIR / f"events{suffix}.parquet")
    events["event_date"] = pd.to_datetime(events["event_date"])
    # primary_moved is only in the raw events file — join if missing
    if "primary_moved" not in events.columns:
        raw = pd.read_parquet(LABELS_DIR / "events.parquet")[["event_id", "primary_moved"]]
        events = events.merge(raw, on="event_id", how="left")
    # Evaluate on primary-moved, test-split events only (unseen during any tuning)
    eval_events = events[
        (events["primary_moved"] == True) & (events["split"] == "test")
    ].copy()
    print(f"[info] evaluation events (test split, primary moved): {len(eval_events)}")

    pairs = pd.read_parquet(LABELS_DIR / f"event_ticker_pairs{suffix}.parquet")
    affected_by_event: dict[str, set[str]] = (
        pairs[pairs["is_affected"] == True]
        .groupby("event_id")["secondary_ticker"]
        .apply(set)
        .to_dict()
    )

    # Build BM25
    bm25, bm25_tickers = build_bm25(firm_profiles)
    bm25_ticker_set = set(bm25_tickers)

    # Load KG
    g = load_kg(KG_PATH)
    uni_set = set(bm25_tickers)
    print(f"[info] KG: {g.number_of_nodes()} nodes  |  BM25 corpus: {len(firm_profiles)} docs")

    baseline_scores: dict[int, list[float]] = {k: [] for k in K_VALUES}
    baseline_recall: dict[int, list[float]] = {k: [] for k in K_VALUES}
    baseline_f1:     dict[int, list[float]] = {k: [] for k in K_VALUES}
    baseline_mrr: list[float] = []

    kg_scores:   dict[int, list[float]] = {k: [] for k in K_VALUES}
    kg_recall:   dict[int, list[float]] = {k: [] for k in K_VALUES}
    kg_f1:       dict[int, list[float]] = {k: [] for k in K_VALUES}
    kg_mrr: list[float] = []

    skipped = 0
    for _, ev in eval_events.iterrows():
        eid     = ev["event_id"]
        primary = ev["primary_ticker"]
        relevant = affected_by_event.get(eid, set()) & uni_set
        if not relevant:
            skipped += 1
            continue

        ev_ctx = event_contexts.get(eid)
        query  = ev_ctx["text"] if ev_ctx else str(ev.get("event_text", ""))
        exclude = {primary}

        max_k = max(K_VALUES)

        # BM25 baseline
        bm25_ranked = bm25_retrieve(bm25, bm25_tickers, query, max_k, exclude)
        # KG-augmented
        kg_ranked = kg_retrieve(g, primary, max_k, exclude)
        # For tickers with no KG path, append BM25 remainder (hybrid fallback)
        kg_set = set(kg_ranked)
        kg_ranked_hybrid = kg_ranked + [t for t in bm25_ranked if t not in kg_set]

        baseline_mrr.append(reciprocal_rank(bm25_ranked, relevant))
        kg_mrr.append(reciprocal_rank(kg_ranked_hybrid, relevant))

        for k in K_VALUES:
            bp = precision_at_k(bm25_ranked, relevant, k)
            br = recall_at_k(bm25_ranked, relevant, k)
            baseline_scores[k].append(bp)
            baseline_recall[k].append(br)
            baseline_f1[k].append(f1_at_k(bp, br))

            kp = precision_at_k(kg_ranked_hybrid, relevant, k)
            kr = recall_at_k(kg_ranked_hybrid, relevant, k)
            kg_scores[k].append(kp)
            kg_recall[k].append(kr)
            kg_f1[k].append(f1_at_k(kp, kr))

    n = len(eval_events) - skipped
    print(f"[info] evaluated {n} events ({skipped} skipped — no affected tickers)")

    def avg(lst): return float(np.mean(lst)) if lst else 0.0

    results = {
        "n_events": n,
        "baseline_bm25": {
            "MRR": avg(baseline_mrr),
            **{f"P@{k}": avg(baseline_scores[k]) for k in K_VALUES},
            **{f"R@{k}": avg(baseline_recall[k]) for k in K_VALUES},
            **{f"F1@{k}": avg(baseline_f1[k]) for k in K_VALUES},
        },
        "kg_augmented": {
            "MRR": avg(kg_mrr),
            **{f"P@{k}": avg(kg_scores[k]) for k in K_VALUES},
            **{f"R@{k}": avg(kg_recall[k]) for k in K_VALUES},
            **{f"F1@{k}": avg(kg_f1[k]) for k in K_VALUES},
        },
    }

    # Write JSON results
    with open(OUT_DIR / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Write markdown summary
    bm25_r = results["baseline_bm25"]
    kg_r   = results["kg_augmented"]

    md_lines = [
        "# RAG Retrieval Evaluation — Stage 3 Baseline",
        f"\n**Evaluation set:** test split, primary_moved=True, n={n} events\n",
        "## Results\n",
        "| Metric | BM25 (no KG) | KG-Augmented | Δ |",
        "|--------|-------------|--------------|---|",
        f"| MRR    | {bm25_r['MRR']:.4f} | {kg_r['MRR']:.4f} | {kg_r['MRR']-bm25_r['MRR']:+.4f} |",
    ]
    for k in K_VALUES:
        for metric in ("P", "R", "F1"):
            key = f"{metric}@{k}"
            b, kg = bm25_r[key], kg_r[key]
            md_lines.append(f"| {key:6s} | {b:.4f} | {kg:.4f} | {kg-b:+.4f} |")

    md_lines += [
        "\n## Notes",
        "- **BM25 baseline**: ranks all universe firm profiles by BM25 score against the event text.",
        "- **KG-augmented**: ranks by KG hop distance from primary ticker; BM25 fills remaining slots.",
        "- Ground truth: tickers with is_affected=True (1.5×σ threshold, 5-day window) for events",
        "  where the primary ticker also moved (primary_moved=True).",
        "- Δ > 0 means KG-augmented outperforms BM25 baseline.",
        "- This evaluation measures **retrieval quality only** — RAFT fine-tuning results follow.",
    ]

    md = "\n".join(md_lines)
    (OUT_DIR / "eval_summary.md").write_text(md)
    print(f"\n[done] results -> {OUT_DIR}/eval_results.json + eval_summary.md")

    # Print to stdout
    print("\n" + md)
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sys.exit(main())
