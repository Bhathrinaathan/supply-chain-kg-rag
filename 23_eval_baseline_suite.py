#!/usr/bin/env python3
"""
23_eval_baseline_suite.py  —  Stage 4-C E1: Five-system baseline comparison.

Systems (all use the same Llama 3.1 8B Instruct base — no closed models):
  S0  TS-only       — sector co-movement baseline (no LLM)
  S1  LLM-only      — Llama 3.1 8B, no retrieval context
  S2  LLM+RAG-text  — hybrid_text retrieval (BM25+BGE) + base model
  S3  LLM+RAG+KG    — hybrid_graph retrieval + base model (no RAFT)
  S4  Proposed      — hybrid_graph retrieval + RAFT adapter

Metrics per system:
  Precision / Recall / F1 (macro, per-relation-type breakdown where available)
  MRR (first correct ticker in model output)
  Direction accuracy (among predicted-affected firms)

Outputs:
  data/eval/baseline_suite_{label}.json     — per-system metrics
  data/eval/baseline_suite_summary.md       — markdown table for the dissertation

USAGE (MLX backend, Path A — activate .venv_mlx first):
  python 23_eval_baseline_suite.py \\
    --backend mlx \\
    --model  mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \\
    --adapter models/raft_lora_mlx \\
    --label  ff \\
    --split  test

  # Quick smoke-test (5 events per system):
  python 23_eval_baseline_suite.py --backend mlx \\
    --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \\
    --adapter models/raft_lora_mlx --label ff --split test --max-examples 5

  # Skip systems you've already run (saves time):
  python 23_eval_baseline_suite.py --systems S0 S4 ...

NOTE: S1–S4 each run the LLM on every test example; on Apple Silicon M4 (4-bit) expect
~8–12 min per system for 292 examples. Run overnight or on Kaggle for the full suite.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import select
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

LABELS_DIR  = Path("data/labeled")
RAFT_DIR    = Path("data/raft")
PRICES_DIR  = Path("data/raw/prices")
SECTOR_MAP  = Path("config/sector_map.csv")
OUT_DIR     = Path("data/eval")
KG_PATH     = Path("kg/networkx_export.gpickle")

# Subprocess-per-batch to avoid the documented MLX Metal command-buffer
# accumulation stall (see README "Metal GPU stall during evaluation") —
# run_s4_final.py / run_a1_ablation.py already use this pattern; this script
# previously didn't, and hung mid-run for exactly that reason.
BATCH_SIZE    = 28
BATCH_TIMEOUT = 900  # seconds per batch

_WORKER = r'''
import json, sys, mlx.core as mx
from mlx_lm import load, generate

model, tokenizer = load({model_id!r}, adapter_path={adapter!r})

for raw in sys.stdin:
    raw = raw.strip()
    if not raw:
        continue
    rec = json.loads(raw)
    prompt = tokenizer.apply_chat_template(rec["messages"], tokenize=False, add_generation_prompt=True)
    out = generate(model, tokenizer, prompt=prompt, max_tokens={max_tokens}, verbose=False)
    mx.eval(mx.array([0]))
    mx.clear_cache()
    sys.stdout.write(json.dumps({{"idx": rec["idx"], "out": out}}) + "\n")
    sys.stdout.flush()
'''


# ---------------------------------------------------------------------------
# Shared helpers (also used in 22_eval_raft.py)
# ---------------------------------------------------------------------------

def parse_affected_tickers(model_output: str, universe: set[str]) -> list[str]:
    output = model_output.upper()
    found: list[str] = []
    seen: set[str] = set()
    m = re.search(r"(?:answer|affected tickers?)\s*:\s*([^\n]+)", output, re.IGNORECASE)
    if m:
        for c in re.split(r"[,;\s]+", m.group(1)):
            c = c.strip().rstrip(".")
            if c in universe and c not in seen:
                found.append(c); seen.add(c)
        if found:
            return found
    for ticker in sorted(universe, key=len, reverse=True):
        if re.search(r"\b" + re.escape(ticker) + r"\b", output):
            if ticker not in seen:
                found.append(ticker); seen.add(ticker)
    return found


def compute_metrics(
    predictions: list[list[str]],
    ground_truth: list[set[str]],
    direction_pred: list[dict[str, int]] | None = None,
    direction_gt: list[dict[str, int]] | None = None,
) -> dict:
    prec, rec, f1_vals, mrr_vals = [], [], [], []
    dir_correct, dir_total = 0, 0

    for i, (pred, gt) in enumerate(zip(predictions, ground_truth)):
        if not gt:
            continue
        pred_set = set(pred)
        tp = len(pred_set & gt)
        p  = tp / len(pred_set) if pred_set else 0.0
        r  = tp / len(gt)
        prec.append(p); rec.append(r)
        f1_vals.append(2*p*r/(p+r) if (p+r) > 0 else 0.0)
        for j, t in enumerate(pred):
            if t in gt:
                mrr_vals.append(1.0 / (j+1)); break
        else:
            mrr_vals.append(0.0)
        if direction_pred and direction_gt:
            dp, dg = direction_pred[i], direction_gt[i]
            for t in pred_set & gt:
                if t in dp and t in dg:
                    dir_total += 1
                    if dp[t] == dg[t]:
                        dir_correct += 1

    def avg(lst): return float(np.mean(lst)) if lst else 0.0
    return {
        "n": len([g for g in ground_truth if g]),
        "precision": avg(prec),
        "recall":    avg(rec),
        "f1":        avg(f1_vals),
        "mrr":       avg(mrr_vals),
        "direction_accuracy": dir_correct / dir_total if dir_total else None,
        "direction_n": dir_total,
    }


# ---------------------------------------------------------------------------
# System S0: TS-only (sector co-movement — no LLM)
# ---------------------------------------------------------------------------

def run_s0_ts_only(
    records: list[dict],
    affected_by_event: dict[str, set[str]],
    universe: set[str],
    pairs: pd.DataFrame,
) -> dict:
    """
    Sector co-movement baseline: predict all same-sector firms as affected.
    Represents what a naive analyst relying only on sector membership would do.
    Falls back to predicting the top-5 most correlated firms (by return correlation
    over the preceding 60 days) if sector map is unavailable.
    """
    print("[S0] sector co-movement baseline ...")

    sector_col = None
    if SECTOR_MAP.exists():
        sm = pd.read_csv(SECTOR_MAP)
        # columns: ticker, sector (or gics_sector)
        for c in ("sector", "gics_sector", "industry"):
            if c in sm.columns:
                sector_col = c; break
        if sector_col:
            sm = sm.rename(columns={sector_col: "sector"})
            ticker_to_sector = dict(zip(sm["ticker"], sm["sector"]))
        else:
            print("[S0] sector map found but no usable sector column — falling back to top-5 corr")
            ticker_to_sector = {}
    else:
        ticker_to_sector = {}
        print("[S0] no sector_map.csv — falling back to top-5 corr")

    predictions, ground_truth = [], []

    for r in records:
        focal   = r["primary_ticker"]
        eid     = r["event_id"]
        gt      = affected_by_event.get(eid, set())
        ground_truth.append(gt)

        if ticker_to_sector and focal in ticker_to_sector:
            focal_sector = ticker_to_sector[focal]
            pred = [t for t in universe
                    if t != focal and ticker_to_sector.get(t) == focal_sector]
        else:
            pred = []  # no signal available

        predictions.append(pred)

    return compute_metrics(predictions, ground_truth), predictions, ground_truth


# ---------------------------------------------------------------------------
# LLM generation helpers
# ---------------------------------------------------------------------------

def _load_retriever():
    spec = importlib.util.spec_from_file_location(
        "hybrid_retrieval", Path("19_hybrid_retrieval.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Retriever()


def _extract_question(content: str) -> str:
    """
    Strip the company-profile context block from a stored user message, keeping only
    the event/question text. The RAFT dataset builder (15_build_raft_dataset.py) appends
    context as "...\\n\\nCOMPANY PROFILES:\\n\\n<docs>" — that literal marker, not
    "RETRIEVED DOCUMENTS", is what actually appears in data/raft/*.jsonl.
    """
    return re.split(r"\n\nCOMPANY PROFILES:", content, maxsplit=1)[0].strip()


def _build_prompt_no_retrieval(record: dict) -> list[dict]:
    """Strip context docs from the user message; keep only the event description."""
    msgs = record["messages"]
    system_msg = msgs[0]
    user_msg   = msgs[1]
    stripped = _extract_question(user_msg["content"])
    return [system_msg, {"role": "user", "content": stripped}]


def _build_prompt_with_retrieval(record: dict, retriever, mode: str, k: int = 20) -> list[dict]:
    """
    Build a genuine mode-specific retrieval prompt: the bare question (no stored
    oracle/distractor context) plus the focal firm's own profile and freshly retrieved
    neighbour profiles for `mode` ("hybrid_text" or "hybrid_graph").
    """
    msgs = record["messages"]
    system_msg = msgs[0]
    question   = _extract_question(msgs[1]["content"])
    primary    = record["primary_ticker"]

    docs: list[str] = []
    focal_profile = retriever.ticker_to_profile.get(primary)
    if focal_profile is not None:
        docs.append(focal_profile["text"])

    results = retriever.retrieve(
        event_text=question.split("\n")[0],
        primary_ticker=primary,
        event_date=record["filing_date"],
        mode=mode,
        k=k,
    )
    for res in results:
        prof = retriever.ticker_to_profile.get(res["ticker"])
        if prof is not None:
            docs.append(prof["text"])

    content = question + "\n\nCOMPANY PROFILES:\n\n" + "\n\n---\n\n".join(docs)
    return [system_msg, {"role": "user", "content": content}]


def generate_mlx(messages_list, model_id, adapter_path, max_new_tokens=512, cache_path=None):
    """
    Subprocess-per-batch generation: each BATCH_SIZE-example batch runs in a
    fresh Python process (mlx_lm imported fresh, model reloaded), avoiding the
    Metal command-buffer accumulation that stalls long single-process runs.
    Resumable via cache_path (JSONL of {"idx", "out"}).
    """
    done: dict[int, str] = {}
    if cache_path and Path(cache_path).exists():
        for line in Path(cache_path).read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["idx"]] = rec["out"]
        if done:
            print(f"[gen] resuming: {len(done)}/{len(messages_list)} already cached")

    todo = [i for i in range(len(messages_list)) if i not in done]
    worker_src = _WORKER.format(
        model_id=model_id,
        adapter=str(adapter_path) if adapter_path else None,
        max_tokens=max_new_tokens,
    )

    while todo:
        batch = todo[:BATCH_SIZE]
        todo  = todo[BATCH_SIZE:]

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            [sys.executable, "-c", worker_src],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=env,
        )
        for i in batch:
            proc.stdin.write(json.dumps({"idx": i, "messages": messages_list[i]}) + "\n")
        proc.stdin.close()

        deadline = time.time() + BATCH_TIMEOUT
        for i in batch:
            remaining = deadline - time.time()
            if remaining <= 0:
                print(f"  [BATCH TIMEOUT] at idx={i}, killing worker")
                proc.terminate()
                break
            ready, _, _ = select.select([proc.stdout], [], [], remaining)
            if not ready:
                print(f"  [ITEM TIMEOUT] idx={i}, killing worker")
                proc.terminate()
                break
            line = proc.stdout.readline()
            if not line:
                print(f"  [WORKER EOF] at idx={i}")
                break
            rec = json.loads(line.strip())
            done[rec["idx"]] = rec["out"]
            if cache_path:
                with open(cache_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")

        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

        missed = [i for i in batch if i not in done]
        if missed:
            todo = missed + todo
        print(f"  [gen] {len(done)}/{len(messages_list)} complete", flush=True)

    return [done[i] for i in range(len(messages_list))]


def generate_hf(messages_list, model_id, adapter_path, hf_token="", max_new_tokens=512, cache_path=None):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError:
        print("[FATAL] pip install transformers peft torch")
        sys.exit(1)
    kw = {"device_map": "auto"}
    if hf_token: kw["token"] = hf_token
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, **kw)
    model     = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, **kw)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    outputs = []
    for msgs in messages_list:
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=tokenizer.eos_token_id)
        outputs.append(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                        skip_special_tokens=True))
    return outputs


# ---------------------------------------------------------------------------
# Systems S1–S4: LLM-based
# ---------------------------------------------------------------------------

def run_llm_system(
    system_id: str,
    records: list[dict],
    affected_by_event: dict[str, set[str]],
    universe: set[str],
    pairs: pd.DataFrame,
    generate_fn: Callable,
    retriever=None,
    use_adapter: bool = False,
    cache_path=None,
) -> dict:
    """
    system_id: S1 | S2 | S3 | S4
    retriever:  None (S1), loaded Retriever instance (S2/S3/S4)
    use_adapter: True only for S4
    """
    print(f"[{system_id}] building prompts ...")

    if system_id == "S1":
        # No retrieval — strip stored context, keep only the event/question text
        messages_list = [_build_prompt_no_retrieval(r) for r in records]

    elif system_id in ("S2", "S3", "S4P"):
        # S2 = hybrid_text mode, S3/S4P = hybrid_graph mode. Build genuinely fresh,
        # mode-specific context per event — do NOT reuse the stored RAFT record's
        # oracle/distractor context (that was selected using ground-truth affected
        # labels for RAFT training purposes, which is not a valid retrieval baseline).
        # S4P = S3's exact retrieval path + the RAFT adapter, isolating the adapter's
        # effect from the context-construction difference that separates S3 from S4.
        mode = "hybrid_text" if system_id == "S2" else "hybrid_graph"
        if retriever is not None:
            messages_list = []
            for r in records:
                try:
                    messages_list.append(_build_prompt_with_retrieval(r, retriever, mode))
                except Exception as e:
                    print(f"[warn] {system_id} retrieval failed for {r['event_id']}: {e}")
                    messages_list.append(_build_prompt_no_retrieval(r))
        else:
            print(f"[warn] {system_id} requested but retriever unavailable — "
                  f"falling back to no-retrieval prompt (NOT a valid {mode} run)")
            messages_list = [_build_prompt_no_retrieval(r) for r in records]

    else:  # S4
        messages_list = [r["messages"][:-1] for r in records]

    print(f"[{system_id}] generating ({len(messages_list)} examples) ...")
    responses = generate_fn(messages_list, cache_path)

    predictions, ground_truth = [], []
    for r, resp in zip(records, responses):
        gt   = affected_by_event.get(r["event_id"], set())
        pred = parse_affected_tickers(resp, universe)
        predictions.append(pred)
        ground_truth.append(gt)

    metrics = compute_metrics(predictions, ground_truth)
    metrics["system"] = system_id
    return metrics, predictions, ground_truth


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SYSTEM_NAMES = {
    "S0": "TS-only (sector co-movement)",
    "S1": "LLM-only (no retrieval)",
    "S2": "LLM + RAG-text (BM25+BGE)",
    "S3": "LLM + RAG+KG (hybrid_graph, no RAFT)",
    "S4": "Proposed (stored RAFT context + RAFT)",
    "S4P": "S4' (hybrid_graph retrieval [= S3's path] + RAFT)",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend",  choices=["mlx", "hf"], default="mlx")
    ap.add_argument("--model",    default="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit")
    ap.add_argument("--adapter",  default=None, help="Path to RAFT LoRA adapter (S4)")
    ap.add_argument("--label",    default="ff", choices=["ff", "loo", "pca"])
    ap.add_argument("--split",    default="test", choices=["test", "holdout"])
    ap.add_argument("--systems",  nargs="+", default=list(SYSTEM_NAMES.keys()),
                    help="Which systems to run (default: all)")
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--hf-token", default="")
    ap.add_argument("--no-retriever", action="store_true",
                    help="Skip loading retriever (S2/S3 fall back to stored context)")
    ap.add_argument("--save-predictions", action="store_true",
                    help="Save per-event predictions for paired bootstrap (25_statistical_tests.py)")
    args = ap.parse_args()

    # Validate
    if args.adapter is None:
        dropped = [s for s in args.systems if s in ("S4", "S4P")]
        if dropped:
            print(f"[WARN] {dropped} require --adapter; skipping")
            args.systems = [s for s in args.systems if s not in ("S4", "S4P")]

    # Load data
    records = [json.loads(l) for l in open(RAFT_DIR / f"{args.split}.jsonl")]
    if args.max_examples:
        records = records[:args.max_examples]
    print(f"[info] {len(records)} examples  label={args.label}  split={args.split}")

    pairs_path = LABELS_DIR / f"event_ticker_pairs_{args.label}.parquet"
    if not pairs_path.exists():
        print(f"[FATAL] {pairs_path} not found — run 18_relabel_abnormal.py first")
        return 1
    pairs = pd.read_parquet(pairs_path)
    ticker_col = "ticker" if "ticker" in pairs.columns else "secondary_ticker"
    affected_by_event: dict[str, set[str]] = (
        pairs[pairs["is_affected"] == True]
        .groupby("event_id")[ticker_col]
        .apply(set).to_dict()
    )
    universe = set(pairs[ticker_col].unique())

    # Load retriever once (shared across S2/S3)
    retriever = None
    if not args.no_retriever and any(s in args.systems for s in ("S2", "S3", "S4P")):
        try:
            retriever = _load_retriever()
            print("[info] retriever loaded")
        except Exception as e:
            print(f"[WARN] could not load retriever ({e}); S2/S3 use stored context")

    # Build generation function (model loaded once, reused across S1–S4)
    _model_cache: dict = {}

    def make_generate_fn(adapter_path):
        if args.backend == "mlx":
            def gen(msgs_list, cache_path=None):
                return generate_mlx(msgs_list, args.model, adapter_path, cache_path=cache_path)
            return gen
        else:
            def gen(msgs_list, cache_path=None):
                return generate_hf(msgs_list, args.model, adapter_path,
                                   hf_token=args.hf_token, cache_path=cache_path)
            return gen

    generate_base = make_generate_fn(None)
    generate_raft = make_generate_fn(args.adapter)

    # Run each system
    results: dict[str, dict] = {}
    all_predictions: dict[str, list] = {}
    saved_ground_truth: list | None = None

    for sid in args.systems:
        print(f"\n{'='*60}")
        print(f"Running {sid}: {SYSTEM_NAMES[sid]}")
        print(f"{'='*60}")

        cache_path = OUT_DIR / f"_gen_cache_{sid}_{args.label}_{args.split}.jsonl"

        if sid == "S0":
            m, preds, gts = run_s0_ts_only(records, affected_by_event, universe, pairs)
        elif sid == "S1":
            m, preds, gts = run_llm_system(sid, records, affected_by_event, universe, pairs,
                               generate_base, retriever=None, cache_path=cache_path)
        elif sid in ("S2", "S3"):
            m, preds, gts = run_llm_system(sid, records, affected_by_event, universe, pairs,
                               generate_base, retriever=retriever, cache_path=cache_path)
        elif sid == "S4":
            m, preds, gts = run_llm_system(sid, records, affected_by_event, universe, pairs,
                               generate_raft, retriever=retriever, use_adapter=True, cache_path=cache_path)
        elif sid == "S4P":
            m, preds, gts = run_llm_system(sid, records, affected_by_event, universe, pairs,
                               generate_raft, retriever=retriever, use_adapter=True, cache_path=cache_path)
        else:
            print(f"[WARN] unknown system {sid}"); continue

        m["system_name"] = SYSTEM_NAMES[sid]
        results[sid] = m
        all_predictions[sid] = preds
        if saved_ground_truth is None:
            saved_ground_truth = [sorted(g) for g in gts]
        print(f"  N={m['n']}  P={m['precision']:.4f}  R={m['recall']:.4f}  "
              f"F1={m['f1']:.4f}  MRR={m['mrr']:.4f}")

    # Save JSON — merge with any existing systems rather than overwriting them
    # (running with --systems S1 S2 S3 must not silently drop a previously
    # saved S0/S4 from an earlier invocation).
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / f"baseline_suite_{args.label}.json"
    existing_doc: dict = {}
    if out_json.exists():
        existing_doc = json.loads(out_json.read_text())
    existing_systems = existing_doc.get("systems", {})
    existing_systems.update(results)
    with open(out_json, "w") as f:
        json.dump({
            "label": args.label, "split": args.split,
            "model": args.model, "adapter": args.adapter,
            "n_examples": len(records),
            "systems": existing_systems,
        }, f, indent=2)
    print(f"\n[done] -> {out_json}  (systems now: {list(existing_systems.keys())})")

    # Markdown summary table
    _write_summary(existing_systems, args.label, args.split, len(records))

    # Per-event predictions (needed by 25_statistical_tests.py for paired bootstrap).
    # Merge with any existing predictions file for the same reason as above.
    if args.save_predictions:
        out_pred = OUT_DIR / f"baseline_suite_{args.label}_predictions.json"
        existing_preds: dict = {}
        if out_pred.exists():
            existing_preds = json.loads(out_pred.read_text())
        merged_predictions = existing_preds.get("predictions", {})
        merged_predictions.update(all_predictions)
        merged_gt = saved_ground_truth if saved_ground_truth is not None else existing_preds.get("ground_truth")
        with open(out_pred, "w") as f:
            json.dump({
                "label": args.label, "split": args.split,
                "ground_truth": merged_gt,
                "predictions": merged_predictions,
            }, f, indent=2)
        print(f"[done] -> {out_pred}  (systems now: {list(merged_predictions.keys())})")

    return 0


def _write_summary(results: dict, label: str, split: str, n: int):
    lines = [
        f"# Baseline Suite Results — label={label}  split={split}  n={n}\n",
        "| System | P | R | F1 | MRR | Dir.Acc |",
        "|--------|---|---|----|-----|---------|",
    ]
    for sid in ("S0", "S1", "S2", "S3", "S4", "S4P"):
        if sid not in results:
            continue
        m = results[sid]
        da = f"{m['direction_accuracy']:.3f}" if m.get("direction_accuracy") else "—"
        lines.append(
            f"| {sid}: {m['system_name']} "
            f"| {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} "
            f"| {m['mrr']:.3f} | {da} |"
        )
    lines += [
        "",
        f"*Primary benchmark: {label.upper()}. Same base model (Llama 3.1 8B Instruct) across S1–S4.*",
        "*S4 uses RAFT LoRA adapter; all others use the unmodified base model.*",
    ]
    out_md = OUT_DIR / "baseline_suite_summary.md"
    with open(out_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[done] -> {out_md}")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
