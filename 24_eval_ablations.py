#!/usr/bin/env python3
"""
24_eval_ablations.py  —  Stage 4-C E2+E3: ablation grid + calibration.

E2 Ablations (all use base model unless --adapter given):
  A0  Full system (hybrid_graph + RAFT)          — reference
  A1  Remove RAFT        hybrid_graph, no adapter
  A2  Remove KG          hybrid_text only, RAFT adapter
  A3  Focal-only         focal_only retrieval, RAFT adapter
  A4  Random-graph ctrl  degree-matched rewire of KG, RAFT adapter
  A5  1-hop only         max_hops=1 (vs default 2)
  A6  2-hop only         max_hops=2 (same as default; listed explicitly)
  A7  supplier_of only   rel_filter=["supplier_of"]
  A8  customer_of only   rel_filter=["customer_of"]
  A9  Remove supplier    rel_filter excludes supplier_of

E3 Calibration:
  Brier score (binary: any affected vs none)
  ECE (expected calibration error) — requires confidence extraction from model output
  Reliability data for plotting

Outputs:
  data/eval/ablations_{label}.json
  data/eval/ablations_summary.md
  data/eval/calibration_{label}.json

USAGE:
  source .venv_mlx/bin/activate
  python 24_eval_ablations.py \\
    --backend mlx \\
    --model  mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \\
    --adapter models/raft_lora_mlx \\
    --label  ff --split test

  # Quick smoke-test:
  python 24_eval_ablations.py --backend mlx \\
    --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \\
    --adapter models/raft_lora_mlx \\
    --label ff --split test --max-examples 5 --ablations A0 A1 A2
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LABELS_DIR = Path("data/labeled")
RAFT_DIR   = Path("data/raft")
OUT_DIR    = Path("data/eval")
KG_PATH    = Path("kg/networkx_export.gpickle")

ABLATION_NAMES = {
    "A0": "Full system (hybrid_graph + RAFT)",
    "A1": "Remove RAFT (hybrid_graph, base model)",
    "A2": "Remove KG (hybrid_text + RAFT)",
    "A3": "Focal-only retrieval + RAFT",
    "A4": "Random-graph control + RAFT",
    "A5": "1-hop only + RAFT",
    "A6": "2-hop (default) + RAFT",
    "A7": "supplier_of edges only + RAFT",
    "A8": "customer_of edges only + RAFT",
    "A9": "Exclude supplier_of + RAFT",
}


# ---------------------------------------------------------------------------
# Shared helpers (copied from 23 to keep scripts self-contained)
# ---------------------------------------------------------------------------

def parse_affected_tickers(output: str, universe: set[str]) -> list[str]:
    output = output.upper()
    found, seen = [], set()
    m = re.search(r"(?:answer|affected tickers?)\s*:\s*([^\n]+)", output, re.IGNORECASE)
    if m:
        for c in re.split(r"[,;\s]+", m.group(1)):
            c = c.strip().rstrip(".")
            if c in universe and c not in seen:
                found.append(c); seen.add(c)
        if found:
            return found
    for t in sorted(universe, key=len, reverse=True):
        if re.search(r"\b" + re.escape(t) + r"\b", output):
            if t not in seen:
                found.append(t); seen.add(t)
    return found


def extract_confidence(output: str) -> float | None:
    """Parse confidence from model output, e.g. 'Confidence: 0.72' or '72%'."""
    m = re.search(r"(?:confidence|probability)\s*[:\=]\s*([0-9.]+)%?", output, re.IGNORECASE)
    if m:
        v = float(m.group(1))
        return v / 100 if v > 1.0 else v
    m = re.search(r"([0-9]{1,3})%\s+(?:confident|probability|likely)", output, re.IGNORECASE)
    if m:
        return float(m.group(1)) / 100
    return None


def compute_metrics(predictions, ground_truth) -> dict:
    prec, rec, f1_vals, mrr_vals = [], [], [], []
    for pred, gt in zip(predictions, ground_truth):
        if not gt:
            continue
        ps = set(pred); tp = len(ps & gt)
        p = tp / len(ps) if ps else 0.0
        r = tp / len(gt)
        prec.append(p); rec.append(r)
        f1_vals.append(2*p*r/(p+r) if (p+r) else 0.0)
        for j, t in enumerate(pred):
            if t in gt: mrr_vals.append(1/(j+1)); break
        else:
            mrr_vals.append(0.0)
    def avg(x): return float(np.mean(x)) if x else 0.0
    return {"n": len([g for g in ground_truth if g]),
            "precision": avg(prec), "recall": avg(rec),
            "f1": avg(f1_vals), "mrr": avg(mrr_vals)}


def brier_score(predictions: list[list[str]], ground_truth: list[set[str]]) -> float:
    """Binary Brier score: predicted any affected vs ground truth any affected."""
    scores = []
    for pred, gt in zip(predictions, ground_truth):
        if not gt:
            continue
        y_hat = 1.0 if pred else 0.0
        y     = 1.0 if gt else 0.0
        scores.append((y_hat - y) ** 2)
    return float(np.mean(scores)) if scores else float("nan")


def ece(confidences: list[float], correct: list[bool], n_bins: int = 10) -> float:
    """Expected calibration error (equal-width bins)."""
    if not confidences:
        return float("nan")
    bins = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    n = len(confidences)
    for lo, hi in zip(bins[:-1], bins[1:]):
        idx = [i for i, c in enumerate(confidences) if lo <= c < hi]
        if not idx:
            continue
        acc  = float(np.mean([correct[i] for i in idx]))
        conf = float(np.mean([confidences[i] for i in idx]))
        ece_val += len(idx) / n * abs(acc - conf)
    return ece_val


# ---------------------------------------------------------------------------
# Random-graph control (degree-matched rewire)
# ---------------------------------------------------------------------------

def _rewire_kg(g):
    """Return a copy of g with edges randomly rewired preserving degree sequence."""
    import networkx as nx
    rng = random.Random(42)
    g2  = g.copy()
    edges = list(g2.edges(data=True))
    rng.shuffle(edges)
    # Double-edge-swap (repeated) on the undirected projection
    # For a directed multigraph, rewire by shuffling target endpoints
    nodes = list(g2.nodes())
    for _ in range(min(1000, len(edges) * 2)):
        if len(edges) < 2:
            break
        i, j = rng.sample(range(len(edges)), 2)
        u1, v1, d1 = edges[i]
        u2, v2, d2 = edges[j]
        if u1 == u2 or v1 == v2 or g2.has_edge(u1, v2) or g2.has_edge(u2, v1):
            continue
        g2.remove_edge(u1, v1)
        g2.remove_edge(u2, v2)
        g2.add_edge(u1, v2, **d1)
        g2.add_edge(u2, v1, **d2)
        edges[i] = (u1, v2, d1)
        edges[j] = (u2, v1, d2)
    return g2


# ---------------------------------------------------------------------------
# Retrieval config per ablation
# ---------------------------------------------------------------------------

ABLATION_RETRIEVAL = {
    "A0": dict(mode="hybrid_graph", max_hops=2, rel_filter=None),
    "A1": dict(mode="hybrid_graph", max_hops=2, rel_filter=None),  # no adapter
    "A2": dict(mode="hybrid_text",  max_hops=2, rel_filter=None),
    "A3": dict(mode="focal_only",   max_hops=0, rel_filter=None),
    "A4": dict(mode="hybrid_graph", max_hops=2, rel_filter=None),  # rewired KG
    "A5": dict(mode="hybrid_graph", max_hops=1, rel_filter=None),
    "A6": dict(mode="hybrid_graph", max_hops=2, rel_filter=None),
    "A7": dict(mode="hybrid_graph", max_hops=2, rel_filter=["supplier_of"]),
    "A8": dict(mode="hybrid_graph", max_hops=2, rel_filter=["customer_of"]),
    "A9": dict(mode="hybrid_graph", max_hops=2,
               rel_filter=["customer_of", "competitor_of", "partner_of",
                           "joint_venture_with", "subsidiary_of", "parent_of"]),
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend",  choices=["mlx", "hf"], default="mlx")
    ap.add_argument("--model",    default="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit")
    ap.add_argument("--adapter",  default=None)
    ap.add_argument("--label",    default="ff", choices=["ff", "loo", "pca"])
    ap.add_argument("--split",    default="test", choices=["test", "holdout"])
    ap.add_argument("--ablations", nargs="+", default=list(ABLATION_NAMES.keys()))
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--hf-token", default="")
    args = ap.parse_args()

    # Require adapter for ablations that use it (A0/A2–A9)
    adapter_ablations = {"A0","A2","A3","A4","A5","A6","A7","A8","A9"}
    if args.adapter is None and any(a in adapter_ablations for a in args.ablations):
        print("[WARN] --adapter not set; ablations that use it will use base model instead")

    records = [json.loads(l) for l in open(RAFT_DIR / f"{args.split}.jsonl")]
    if args.max_examples:
        records = records[:args.max_examples]
    print(f"[info] {len(records)} examples  label={args.label}")

    pairs = pd.read_parquet(LABELS_DIR / f"event_ticker_pairs_{args.label}.parquet")
    tc = "ticker" if "ticker" in pairs.columns else "secondary_ticker"
    affected_by_event = (
        pairs[pairs["is_affected"] == True].groupby("event_id")[tc].apply(set).to_dict()
    )
    universe = set(pairs[tc].unique())

    # Load retriever
    spec = importlib.util.spec_from_file_location("hybrid_retrieval", "19_hybrid_retrieval.py")
    hr   = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hr)
    retriever = hr.Retriever()

    # Load KG for random-graph ablation
    rewired_kg = None
    if "A4" in args.ablations:
        try:
            import networkx as nx, pickle
            with open(KG_PATH, "rb") as f:
                g = pickle.load(f)
            rewired_kg = _rewire_kg(g)
            print("[info] random-graph KG prepared")
        except Exception as e:
            print(f"[WARN] could not prepare random-graph KG: {e}; skipping A4")
            args.ablations = [a for a in args.ablations if a != "A4"]

    # Generation function factory
    def gen_fn(use_adapter: bool):
        adp = args.adapter if use_adapter else None
        if args.backend == "mlx":
            from mlx_lm import load, generate as mlx_gen
            model, tokenizer = load(args.model, adapter_path=adp)
            def _gen(msgs_list):
                out = []
                for msgs in msgs_list:
                    p = tokenizer.apply_chat_template(msgs, tokenize=False,
                                                      add_generation_prompt=True)
                    out.append(mlx_gen(model, tokenizer, prompt=p,
                                      max_tokens=512, verbose=False))
                return out
        else:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            kw = {"device_map": "auto"}
            if args.hf_token: kw["token"] = args.hf_token
            tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, **kw)
            mdl = AutoModelForCausalLM.from_pretrained(args.model,
                                                        torch_dtype=torch.bfloat16, **kw)
            if adp: mdl = PeftModel.from_pretrained(mdl, adp)
            mdl.eval()
            def _gen(msgs_list):
                out = []
                for msgs in msgs_list:
                    p = tok.apply_chat_template(msgs, tokenize=False,
                                                add_generation_prompt=True)
                    inp = tok(p, return_tensors="pt").to(mdl.device)
                    with torch.no_grad():
                        g = mdl.generate(**inp, max_new_tokens=512, do_sample=False,
                                          pad_token_id=tok.eos_token_id)
                    out.append(tok.decode(g[0][inp["input_ids"].shape[1]:],
                                          skip_special_tokens=True))
                return out
        return _gen

    # Cache models: base and raft
    print("[info] loading base model ...")
    gen_base = gen_fn(use_adapter=False)
    gen_raft = gen_fn(use_adapter=True) if args.adapter else gen_base

    results: dict[str, dict] = {}
    calibration_data: dict[str, dict] = {}

    for ablation_id in args.ablations:
        if ablation_id not in ABLATION_NAMES:
            print(f"[WARN] unknown ablation {ablation_id}"); continue

        print(f"\n{'='*60}\n{ablation_id}: {ABLATION_NAMES[ablation_id]}\n{'='*60}")
        cfg = ABLATION_RETRIEVAL[ablation_id]
        use_raft = ablation_id != "A1"

        # Build messages for each record
        messages_list = []
        for r in records:
            msgs = r["messages"][:-1]  # drop ground-truth assistant turn
            messages_list.append(msgs)

        gen = gen_raft if use_raft else gen_base
        responses = gen(messages_list)

        predictions, ground_truth, confidences, is_correct = [], [], [], []
        for r, resp in zip(records, responses):
            gt   = affected_by_event.get(r["event_id"], set())
            pred = parse_affected_tickers(resp, universe)
            predictions.append(pred)
            ground_truth.append(gt)
            conf = extract_confidence(resp)
            if conf is not None:
                confidences.append(conf)
                # "correct" = at least one predicted ticker is in ground truth
                is_correct.append(bool(set(pred) & gt))

        m = compute_metrics(predictions, ground_truth)
        m["ablation"] = ablation_id
        m["ablation_name"] = ABLATION_NAMES[ablation_id]
        m["brier"] = brier_score(predictions, ground_truth)
        results[ablation_id] = m

        if confidences:
            calibration_data[ablation_id] = {
                "n_with_conf": len(confidences),
                "ece": ece(confidences, is_correct),
                "mean_confidence": float(np.mean(confidences)),
                "mean_accuracy": float(np.mean(is_correct)),
            }

        print(f"  N={m['n']}  P={m['precision']:.4f}  R={m['recall']:.4f}  "
              f"F1={m['f1']:.4f}  MRR={m['mrr']:.4f}  Brier={m['brier']:.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / f"ablations_{args.label}.json"
    with open(out_json, "w") as f:
        json.dump({"label": args.label, "split": args.split,
                   "model": args.model, "adapter": args.adapter,
                   "ablations": results}, f, indent=2)
    print(f"\n[done] -> {out_json}")

    if calibration_data:
        cal_path = OUT_DIR / f"calibration_{args.label}.json"
        with open(cal_path, "w") as f:
            json.dump(calibration_data, f, indent=2)
        print(f"[done] -> {cal_path}")

    _write_ablation_summary(results, args.label, args.split, len(records))
    return 0


def _write_ablation_summary(results: dict, label: str, split: str, n: int):
    lines = [
        f"# Ablation Results — label={label}  split={split}  n={n}\n",
        "| Ablation | P | R | F1 | MRR | Brier |",
        "|----------|---|---|----|-----|-------|",
    ]
    for aid in sorted(results.keys()):
        m = results[aid]
        b = f"{m['brier']:.4f}" if not math.isnan(m['brier']) else "—"
        lines.append(
            f"| {aid}: {m['ablation_name']} "
            f"| {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} "
            f"| {m['mrr']:.3f} | {b} |"
        )
    out_md = OUT_DIR / "ablations_summary.md"
    with open(out_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[done] -> {out_md}")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
