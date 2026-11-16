#!/usr/bin/env python3
"""
15_build_raft_dataset.py  —  Build RAFT fine-tuning triples.

RAFT recipe: Zhang et al. 2024, "RAFT: Adapting Language Model to Domain Specific RAG,"
arXiv:2403.10131 (UC Berkeley Gorilla team). [VERIFY before citing in dissertation]

For each event where focal_shock=True (|focal SCAR| > 1.5 on the primary benchmark),
constructs one training example:
  - question    : natural-language prompt describing the 8-K event
  - context     : oracle docs (primary profile + affected hop-1 neighbours) +
                  distractor docs (non-adjacent, non-affected firms), shuffled
  - oracle_ids  : doc_ids of oracle documents (for loss masking if needed)
  - cot_answer  : chain-of-thought that names relationship paths, then lists affected tickers
  - split       : train / test / holdout (from the temporal split)

Market-adjusted labels are read from event_ticker_pairs_<label>.parquet (from script 18).
Default: ff (Fama-French 5-factor + momentum residual, the primary benchmark).
Filter: focal_shock=True (only events where the source firm was genuinely shocked).

Oracle probability P (Zhang et al.): fraction of training examples that include the correct
oracle documents in context; rest are distractor-only (model learns to handle absent evidence).
Recommended: 0.6–0.8 for this task. Default: 0.6.

OUTPUTS
  data/raft/train.jsonl
  data/raft/test.jsonl
  data/raft/holdout.jsonl
  data/raft/all.jsonl     (combined, for inspection)

USAGE
  python 15_build_raft_dataset.py
  python 15_build_raft_dataset.py --label ff --oracle-p 0.6 --n-distractors 4
  python 15_build_raft_dataset.py --label loo --oracle-p 0.8   # robustness variant
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RETRIEVAL_DIR = Path("data/retrieval")
LABELS_DIR    = Path("data/labeled")
OUT_DIR       = Path("data/raft")

ITEM_LABELS = {
    "1.01": "entry into a material agreement",
    "1.02": "termination of a material agreement",
    "2.02": "results of operations and financial condition (earnings)",
    "5.02": "departure or appointment of directors/officers",
    "7.01": "Regulation FD disclosure",
    "8.01": "other material events",
}

SYSTEM_PROMPT = (
    "You are a financial analyst specialising in supply chain risk and equity impact assessment "
    "for the technology hardware, semiconductor, and automotive sectors. "
    "Use the provided company profiles and supply chain relationship context to identify which "
    "companies are likely to experience significant abnormal stock price movement following the "
    "described event. Cite the specific supply chain relationships that drive your reasoning."
)

QUESTION_TEMPLATE = (
    "On {date}, {ticker} ({name}) filed an 8-K with the SEC reporting {items_desc}. "
    "The filing states: \"{excerpt}\"\n\n"
    "Based on {ticker}'s position in the value chain and the company profiles provided, "
    "which companies in the universe are likely to experience significant abnormal stock price "
    "impact (standardised abnormal return |SCAR| > 1.5 within 5 trading days)? "
    "Explain your reasoning using specific supply chain relationships, then list the affected tickers."
)


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def describe_items(items_str: str) -> str:
    parts = [p.strip() for p in str(items_str).split(",") if p.strip()]
    labels = [ITEM_LABELS[p] for p in parts if p in ITEM_LABELS]
    if not labels:
        return "material events"
    return labels[0] if len(labels) == 1 else f"{labels[0]} and {labels[1]}"


def rel_type_label(rt: str) -> str:
    return rt.replace("_", " ") if rt else "related to"


def build_cot(
    primary: str,
    primary_name: str,
    hop1_affected: list[dict],       # [{"ticker": ..., "rel_types": [...], "scar": ...}]
    hop1_unaffected_sample: list[str],
    all_affected: list[str],
    items_desc: str,
    benchmark: str,
) -> str:
    """Chain-of-thought that names relationship paths and cites the benchmark."""
    lines = [
        f"{primary} ({primary_name}) disclosed {items_desc}.",
        "",
        "Step 1 — direct supply chain relationships (hop-1, as of event date):",
    ]
    if hop1_affected:
        for n in hop1_affected[:8]:
            rels = ", ".join(rel_type_label(r) for r in (n.get("rel_types") or ["related to"]))
            lines.append(f"  • [{rels}] {n['ticker']}")
    else:
        lines.append("  • No direct supply chain edges to other universe firms in KG.")

    lines += [
        "",
        "Step 2 — assess impact pathway:",
        f"  Under the {benchmark.upper()} market-adjusted return benchmark, events at {primary} "
        f"propagate to firms with direct supply chain linkages (hop-1). "
        f"Firms two or more hops away face attenuated indirect exposure.",
        "",
        "Step 3 — market-adjusted abnormal returns:",
    ]
    if all_affected:
        lines.append(
            f"  The following firms showed |SCAR| > 1.5 (standardised cumulative abnormal return, "
            f"[0,+5] window, {benchmark.upper()} benchmark):"
        )
        lines.append(f"  {', '.join(all_affected)}")
    else:
        lines.append("  No universe firms exceeded |SCAR| > 1.5 in the 5-day window.")

    lines += [
        "",
        f"Answer: {', '.join(all_affected) if all_affected else 'None'}",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label",          default="ff", choices=["ff", "loo", "pca"],
                    help="Market-adjusted label set (primary: ff)")
    ap.add_argument("--oracle-p",       type=float, default=0.6,
                    help="Oracle probability P (Zhang et al. 2024 recommends 0.6–0.8)")
    ap.add_argument("--n-distractors",  type=int,   default=4,
                    help="Number of distractor firm profiles per example")
    ap.add_argument("--max-affected",   type=int,   default=20)
    ap.add_argument("--seed",           type=int,   default=42)
    ap.add_argument("--no-focal-shock", action="store_true",
                    help="Include all events (not just focal_shock=True). Not recommended.")
    args = ap.parse_args()

    random.seed(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ inputs
    firm_profiles  = {d["ticker"]: d for d in load_jsonl(RETRIEVAL_DIR / "firm_profiles.jsonl")}
    event_contexts = {d["event_id"]: d for d in load_jsonl(RETRIEVAL_DIR / "event_contexts.jsonl")}
    all_tickers    = list(firm_profiles.keys())

    uni = pd.read_csv("config/universe.csv")
    uni_name = {str(r["ticker"]): str(r.get("name", r["ticker"])) for _, r in uni.iterrows()}

    # Market-adjusted pairs (from script 18)
    pairs_path = LABELS_DIR / f"event_ticker_pairs_{args.label}.parquet"
    if not pairs_path.exists():
        print(f"[FATAL] {pairs_path} not found — run 18_relabel_abnormal.py --method {args.label}")
        return 1

    pairs = pd.read_parquet(pairs_path)
    # Column name: script 18 uses "ticker" (not "secondary_ticker" from script 13)
    ticker_col = "ticker" if "ticker" in pairs.columns else "secondary_ticker"

    # Focal shock filter (pre-registered condition)
    if not args.no_focal_shock:
        if "focal_shock" not in pairs.columns:
            print("[warn] focal_shock column missing — using all events")
        else:
            pairs = pairs[pairs["focal_shock"] == True]  # noqa: E712
            print(f"[info] focal_shock filter: {pairs['event_id'].nunique():,} events retained")

    # Load base events for split + metadata
    events_raw = pd.read_parquet(LABELS_DIR / "events.parquet")
    events_raw["event_date"] = pd.to_datetime(events_raw["event_date"])
    event_meta = (events_raw
                  .drop_duplicates("event_id")
                  .set_index("event_id")[["primary_ticker", "filing_date", "items", "split", "event_text"]]
                  .to_dict("index"))

    # Affected lookup: event_id -> list[ticker]
    affected_by_event: dict[str, list[str]] = (
        pairs[pairs["is_affected"] == True]  # noqa: E712
        .groupby("event_id")[ticker_col]
        .apply(list)
        .to_dict()
    )

    # hop-1 affected with rel_types: event_id -> list[{ticker, rel_types, scar}]
    hop1_pairs = pairs[(pairs["hop"] == 1) & (pairs["is_affected"] == True)]  # noqa: E712

    def safe_list(x):
        if x is None:
            return []
        try:
            return list(x)
        except Exception:
            return []

    hop1_by_event: dict[str, list[dict]] = {}
    for eid, grp in hop1_pairs.groupby("event_id"):
        hop1_by_event[eid] = [
            {
                "ticker": row[ticker_col],
                "rel_types": safe_list(row.get("rel_types")),
                "scar": float(row.get("scar", 0)),
            }
            for _, row in grp.iterrows()
        ]

    print(f"[info] {len(affected_by_event):,} events with ≥1 affected ticker  |  "
          f"label={args.label}  oracle_p={args.oracle_p}")

    # ------------------------------------------------------------------ build
    by_split: dict[str, list[dict]] = {"train": [], "test": [], "holdout": []}

    for event_id, affected_raw in affected_by_event.items():
        meta = event_meta.get(event_id)
        if meta is None:
            continue

        primary  = meta["primary_ticker"]
        split    = meta["split"]
        ev_ctx   = event_contexts.get(event_id)
        if ev_ctx is None:
            continue
        primary_profile = firm_profiles.get(primary)
        if primary_profile is None:
            continue

        affected = affected_raw[:args.max_affected]
        hop1_aff = hop1_by_event.get(event_id, [])

        # Oracle docs: primary + up to 3 affected hop-1 neighbours (most informative)
        oracle_docs: list[dict] = [primary_profile]
        for n in sorted(hop1_aff, key=lambda x: -abs(x["scar"]))[:3]:
            if n["ticker"] in firm_profiles:
                oracle_docs.append(firm_profiles[n["ticker"]])

        # Distractor pool: not primary, not in hop-1 affected set
        aff_set = set(affected)
        h1_set  = {n["ticker"] for n in hop1_aff}
        exclude = {primary} | h1_set
        distractor_pool = [t for t in all_tickers if t not in exclude]
        # Prefer non-affected firms as distractors (harder negatives)
        non_aff_pool = [t for t in distractor_pool if t not in aff_set]
        if len(non_aff_pool) >= args.n_distractors:
            distractors = random.sample(non_aff_pool, args.n_distractors)
        else:
            distractors = random.sample(distractor_pool, min(args.n_distractors, len(distractor_pool)))
        distractor_docs = [firm_profiles[t] for t in distractors if t in firm_profiles]

        # RAFT: oracle_p fraction get oracle docs in context
        include_oracle = random.random() < args.oracle_p
        if include_oracle:
            context_docs = oracle_docs + distractor_docs
            oracle_ids   = [d["doc_id"] for d in oracle_docs]
        else:
            # Distractor-only: always keep primary profile (event must still be understood)
            context_docs = distractor_docs + [primary_profile]
            oracle_ids   = [primary_profile["doc_id"]]

        random.shuffle(context_docs)

        # Build question
        ev_text = ev_ctx["text"]
        if "8-K filing text (excerpt):" in ev_text:
            excerpt = ev_text.split("8-K filing text (excerpt):")[1].strip()[:200]
        else:
            excerpt = ev_text[:200]

        primary_name = uni_name.get(primary, primary)
        items_desc   = describe_items(str(meta.get("items", "")))
        filing_date  = str(meta["filing_date"])[:10]

        question = QUESTION_TEMPLATE.format(
            date=filing_date, ticker=primary, name=primary_name,
            items_desc=items_desc, excerpt=excerpt.replace('"', "'"),
        )

        cot = build_cot(
            primary, primary_name,
            hop1_aff, [n["ticker"] for n in hop1_aff if n["ticker"] not in aff_set],
            affected, items_desc, args.label,
        )

        messages = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": (question + "\n\nCOMPANY PROFILES:\n\n" +
                                              "\n\n---\n\n".join(d["text"] for d in context_docs))},
            {"role": "assistant", "content": cot},
        ]

        record = {
            "event_id":        event_id,
            "primary_ticker":  primary,
            "filing_date":     filing_date,
            "split":           split,
            "label":           args.label,
            "include_oracle":  include_oracle,
            "oracle_doc_ids":  oracle_ids,
            "context_doc_ids": [d["doc_id"] for d in context_docs],
            "affected_tickers": affected,
            "affected_count":   len(affected_raw),
            "hop1_affected":    [n["ticker"] for n in hop1_aff],
            "messages":        messages,
        }
        by_split.setdefault(split, []).append(record)

    # ------------------------------------------------------------------ write
    all_records: list[dict] = []
    for split, records in sorted(by_split.items()):
        if not records:
            continue
        path = OUT_DIR / f"{split}.jsonl"
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        all_records.extend(records)
        n_oracle = sum(1 for r in records if r["include_oracle"])
        print(f"[ok] {split:8s}: {len(records):,} examples  "
              f"({n_oracle} oracle / {len(records)-n_oracle} distractor-only)")

    with open(OUT_DIR / "all.jsonl", "w") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")

    print(f"\n[done] {len(all_records):,} total RAFT examples -> {OUT_DIR}/")
    print(f"       label={args.label}  oracle_p={args.oracle_p}  n_distractors={args.n_distractors}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
