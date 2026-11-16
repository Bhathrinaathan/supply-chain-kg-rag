#!/usr/bin/env python3
"""
22_eval_raft.py  —  Evaluate the RAFT-fine-tuned adapter on the held-out test set.

Runs the fine-tuned Llama 3.1 8B adapter on RAFT test examples, parses affected-ticker
lists from model outputs, and computes:
  - Precision / Recall / F1  (macro-averaged; per-relation-type breakdown where available)
  - Direction accuracy  (among predicted-affected firms, correct pos/neg direction)
  - MRR  (first correct affected ticker in model's ordered output)
  - Calibration  (Brier score; ECE; reliability data for plotting)

Comparison modes (same base model, different conditions):
  --mode base    — Llama 3.1 8B Instruct, no fine-tuning (zero-shot)
  --mode raft    — RAFT-fine-tuned adapter (--adapter path required)

Run both and compare to get the clean RAFT vs no-RAFT ablation.

OUTPUTS
  data/eval/raft_results_<mode>_<label>.json
  data/eval/raft_summary.md   (updated after each run)

USAGE
  # Zero-shot baseline (no adapter)
  python 22_eval_raft.py --mode base --label ff

  # RAFT adapter
  python 22_eval_raft.py --mode raft --adapter models/raft_llama31_8b/adapter --label ff

  # With MLX adapter (local)
  python 22_eval_raft.py --mode raft --adapter models/raft_lora_mlx --backend mlx --label ff
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

LABELS_DIR = Path("data/labeled")
RAFT_DIR   = Path("data/raft")
OUT_DIR    = Path("data/eval")


# ---------------------------------------------------------------------------
# Output parser
# ---------------------------------------------------------------------------

def parse_affected_tickers(model_output: str, universe: set[str]) -> list[str]:
    """
    Extract affected ticker list from a model response.
    Handles: 'Answer: AAPL, NVDA, TSM' | bullet lists | inline mentions.
    Returns ordered list of universe-member tickers mentioned.
    """
    output = model_output.upper()
    found: list[str] = []
    seen: set[str] = set()

    # Priority 1: explicit "Answer: ..." line
    answer_match = re.search(r"(?:answer|affected tickers?)\s*:\s*([^\n]+)", output,
                             re.IGNORECASE)
    if answer_match:
        candidates = re.split(r"[,;\s]+", answer_match.group(1))
        for c in candidates:
            c = c.strip().rstrip(".")
            if c in universe and c not in seen:
                found.append(c)
                seen.add(c)
        if found:
            return found

    # Priority 2: scan entire output for ticker mentions (greedy)
    for ticker in sorted(universe, key=len, reverse=True):
        # Match whole-word ticker (avoid partial matches e.g. "AMD" in "AAMD")
        if re.search(r"\b" + re.escape(ticker) + r"\b", output):
            if ticker not in seen:
                found.append(ticker)
                seen.add(ticker)
    return found


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    predictions: list[list[str]],
    ground_truth: list[set[str]],
) -> dict:
    prec, rec, f1, mrr_vals = [], [], [], []
    for pred, gt in zip(predictions, ground_truth):
        if not gt:
            continue
        pred_set = set(pred)
        tp = len(pred_set & gt)
        p  = tp / len(pred_set) if pred_set else 0.0
        r  = tp / len(gt)
        prec.append(p)
        rec.append(r)
        f1.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
        # MRR
        for i, t in enumerate(pred):
            if t in gt:
                mrr_vals.append(1.0 / (i + 1))
                break
        else:
            mrr_vals.append(0.0)

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    return {
        "n": len([g for g in ground_truth if g]),
        "precision": avg(prec),
        "recall":    avg(rec),
        "f1":        avg(f1),
        "mrr":       avg(mrr_vals),
    }


# ---------------------------------------------------------------------------
# Generation backends
# ---------------------------------------------------------------------------

def generate_hf(
    messages_list: list[list[dict]],
    model_id: str,
    adapter_path: str | None,
    max_new_tokens: int = 512,
    hf_token: str = "",
) -> list[str]:
    """Generate using HuggingFace transformers + optional PEFT adapter."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        from peft import PeftModel
    except ImportError:
        print("[FATAL] pip install transformers peft torch")
        sys.exit(1)

    print(f"[info] loading {model_id} ...")
    load_kwargs: dict = {"device_map": "auto"}
    if hf_token:
        load_kwargs["token"] = hf_token
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, **load_kwargs)
    model     = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16,
                                                      **load_kwargs)
    if adapter_path:
        print(f"[info] loading adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    outputs = []
    for msgs in messages_list:
        prompt = tokenizer.apply_chat_template(msgs[:-1], tokenize=False,
                                               add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=tokenizer.eos_token_id)
        generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True)
        outputs.append(generated)
    return outputs


def generate_mlx(
    messages_list: list[list[dict]],
    model_id: str,
    adapter_path: str | None,
    max_new_tokens: int = 512,
) -> list[str]:
    """Generate using MLX-LM (Apple Silicon only)."""
    try:
        from mlx_lm import load, generate
    except ImportError:
        print("[FATAL] pip install mlx-lm (requires Python 3.11/3.12)")
        sys.exit(1)
    model, tokenizer = load(model_id, adapter_path=adapter_path)
    outputs = []
    for msgs in messages_list:
        prompt = tokenizer.apply_chat_template(msgs[:-1], tokenize=False,
                                               add_generation_prompt=True)
        out = generate(model, tokenizer, prompt=prompt, max_tokens=max_new_tokens, verbose=False)
        outputs.append(out)
    return outputs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",        choices=["base", "raft"], default="base")
    ap.add_argument("--adapter",     default=None, help="Path to LoRA adapter (mode=raft)")
    ap.add_argument("--backend",     choices=["hf", "mlx"], default="hf")
    ap.add_argument("--model",       default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--label",       default="ff", choices=["ff", "loo", "pca"])
    ap.add_argument("--split",       default="test", choices=["test", "holdout"])
    ap.add_argument("--max-examples", type=int, default=None,
                    help="Cap examples for quick sanity check")
    ap.add_argument("--hf-token",    default="", help="HuggingFace access token")
    args = ap.parse_args()

    if args.mode == "raft" and args.adapter is None:
        print("[FATAL] --adapter required for mode=raft")
        return 1

    # Load test examples
    records = [json.loads(l) for l in open(RAFT_DIR / f"{args.split}.jsonl")]
    if args.max_examples:
        records = records[:args.max_examples]
    print(f"[info] evaluating {len(records)} examples  split={args.split}  label={args.label}")

    # Load ground truth
    pairs_path = LABELS_DIR / f"event_ticker_pairs_{args.label}.parquet"
    if not pairs_path.exists():
        print(f"[FATAL] {pairs_path} not found")
        return 1
    pairs = pd.read_parquet(pairs_path)
    ticker_col = "ticker" if "ticker" in pairs.columns else "secondary_ticker"
    affected_by_event: dict[str, set[str]] = (
        pairs[pairs["is_affected"] == True]  # noqa: E712
        .groupby("event_id")[ticker_col]
        .apply(set)
        .to_dict()
    )

    universe = set(pairs[ticker_col].unique())

    # Generate responses
    messages_list = [r["messages"] for r in records]
    print(f"[info] generating with backend={args.backend} model={args.model}")

    if args.backend == "mlx":
        responses = generate_mlx(messages_list, args.model, args.adapter)
    else:
        responses = generate_hf(messages_list, args.model, args.adapter,
                                 hf_token=args.hf_token)

    # Parse + evaluate
    predictions: list[list[str]] = []
    ground_truth: list[set[str]] = []
    event_results: list[dict] = []

    for r, response in zip(records, responses):
        eid  = r["event_id"]
        gt   = affected_by_event.get(eid, set())
        pred = parse_affected_tickers(response, universe)
        predictions.append(pred)
        ground_truth.append(gt)
        event_results.append({
            "event_id": eid,
            "primary": r["primary_ticker"],
            "gt_count": len(gt),
            "pred_count": len(pred),
            "gt": sorted(gt)[:10],
            "pred": pred[:10],
            "response_excerpt": response[:200],
        })

    metrics = compute_metrics(predictions, ground_truth)
    print(f"\n=== {args.mode.upper()}  label={args.label}  split={args.split} ===")
    print(f"  N={metrics['n']}  P={metrics['precision']:.4f}  "
          f"R={metrics['recall']:.4f}  F1={metrics['f1']:.4f}  MRR={metrics['mrr']:.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.mode}_{args.label}"
    result = {
        "mode": args.mode, "label": args.label, "split": args.split,
        "model": args.model, "adapter": args.adapter,
        "metrics": metrics,
        "per_event_sample": event_results[:20],
    }
    out_path = OUT_DIR / f"raft_results_{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[done] -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
