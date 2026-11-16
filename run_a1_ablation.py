#!/usr/bin/env python3
"""
A1 ablation: base model (no RAFT adapter), same prompts as S4.
Subprocess-per-batch to prevent Metal state accumulation.
Results fed into ablations_ff.json and predictions file for bootstrap tests.

Usage:
  caffeinate -i python run_a1_ablation.py
"""
from __future__ import annotations
import json, os, re, subprocess, sys, time
from pathlib import Path

import numpy as np

MODEL_ID   = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
DATA_FILE  = Path("data/raft/test.jsonl")
LABEL_DIR  = Path("data/labeled")
OUT_DIR    = Path("data/eval")
LABEL      = "ff"
MAX_TOKENS = 256
BATCH_SIZE = 28
BATCH_TIMEOUT = 900  # 15 min; 28 × 17s ≈ 8 min expected

_WORKER = r'''
import json, sys, time, mlx.core as mx
from mlx_lm import load, generate

MODEL_ID   = "{model_id}"
MAX_TOKENS = {max_tokens}

sys.stderr.write("worker: loading base model...\n"); sys.stderr.flush()
model, tokenizer = load(MODEL_ID)
sys.stderr.write("worker: ready\n"); sys.stderr.flush()

for raw in sys.stdin:
    raw = raw.strip()
    if not raw: continue
    rec = json.loads(raw)
    msgs = rec["messages"]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    t0 = time.time()
    out = generate(model, tokenizer, prompt=prompt, max_tokens=MAX_TOKENS, verbose=False)
    elapsed = time.time() - t0
    mx.eval(mx.array([0]))
    mx.clear_cache()
    sys.stdout.write(json.dumps({{"idx": rec["idx"], "out": out, "elapsed": round(elapsed,2)}}) + "\n")
    sys.stdout.flush()
'''.format(model_id=MODEL_ID, max_tokens=MAX_TOKENS)


def load_universe_and_gt():
    import pandas as pd
    pairs_file = LABEL_DIR / f"event_ticker_pairs_{LABEL}.parquet"
    pairs = pd.read_parquet(pairs_file)
    universe = set(pairs["ticker"].unique())
    affected = {}
    for eid, grp in pairs[pairs["is_affected"] == True].groupby("event_id"):  # noqa: E712
        affected[str(eid)] = set(grp["ticker"].tolist())
    return universe, affected


def parse_tickers(text, universe):
    out = text.upper(); found, seen = [], set()
    m = re.search(r"(?:answer|affected tickers?)\s*:\s*([^\n]+)", out, re.IGNORECASE)
    if m:
        for c in re.split(r"[,;\s]+", m.group(1)):
            c = c.strip().rstrip(".")
            if c in universe and c not in seen: found.append(c); seen.add(c)
        if found: return found
    for t in sorted(universe, key=len, reverse=True):
        if re.search(r"\b" + re.escape(t) + r"\b", out):
            if t not in seen: found.append(t); seen.add(t)
    return found


def compute_metrics(predictions, ground_truth):
    prec, rec, f1v, mrrv = [], [], [], []
    for pred, gt in zip(predictions, ground_truth):
        if not gt: continue
        ps = set(pred); tp = len(ps & gt)
        p = tp / len(ps) if ps else 0.0; r = tp / len(gt)
        prec.append(p); rec.append(r)
        f1v.append(2*p*r/(p+r) if (p+r) else 0.0)
        for j, t in enumerate(pred):
            if t in gt: mrrv.append(1/(j+1)); break
        else: mrrv.append(0.0)
    avg = lambda x: float(np.mean(x)) if x else 0.0
    return {"n": len([g for g in ground_truth if g]),
            "precision": avg(prec), "recall": avg(rec),
            "f1": avg(f1v), "mrr": avg(mrrv)}


def run_batch(batch):
    env = os.environ.copy(); env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        [sys.executable, "-c", _WORKER],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, env=env, cwd=str(Path.cwd()),
    )
    for rec in batch:
        proc.stdin.write(json.dumps({"idx": rec["idx"], "messages": rec["messages"][:-1]}) + "\n")
    proc.stdin.close()

    results = []
    deadline = time.time() + BATCH_TIMEOUT
    import select
    for rec in batch:
        remaining = deadline - time.time()
        if remaining <= 0:
            print(f"  [BATCH TIMEOUT] at idx={rec['idx']}", flush=True)
            proc.terminate(); break
        ready, _, _ = select.select([proc.stdout], [], [], remaining)
        if not ready:
            print(f"  [ITEM TIMEOUT] idx={rec['idx']}", flush=True)
            proc.terminate(); break
        line = proc.stdout.readline()
        if not line:
            print(f"  [WORKER EOF] idx={rec['idx']}", flush=True); break
        result = json.loads(line.strip())
        results.append(result)
        print(f"  ex {result['idx']:3d}: {result['elapsed']:.1f}s", flush=True)
    try: proc.wait(timeout=5)
    except subprocess.TimeoutExpired: proc.kill()
    return results


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = [json.loads(l) for l in DATA_FILE.read_text().splitlines() if l.strip()]
    for i, r in enumerate(records): r["idx"] = i

    per_ex_file = OUT_DIR / "a1_per_example.jsonl"
    done = {}
    if per_ex_file.exists():
        for line in per_ex_file.read_text().splitlines():
            if line.strip():
                r = json.loads(line); done[r["idx"]] = r

    todo = [r for r in records if r["idx"] not in done]
    print(f"A1 ablation (base model, no adapter)")
    print(f"Total: {len(records)}  done: {len(done)}  remaining: {len(todo)}", flush=True)

    universe, affected_by_event = load_universe_and_gt()
    t_start = time.time(); batch_num = 0

    while todo:
        batch = todo[:BATCH_SIZE]; todo = todo[BATCH_SIZE:]
        batch_num += 1
        print(f"\nBatch {batch_num}: idx {batch[0]['idx']}–{batch[-1]['idx']}", flush=True)
        batch_results = run_batch(batch)
        with per_ex_file.open("a") as f:
            for res in batch_results:
                f.write(json.dumps(res) + "\n"); done[res["idx"]] = res
        completed = len(done)
        elapsed = time.time() - t_start
        avg_s = elapsed / completed if completed else 0
        remain = (len(records) - completed) * avg_s
        print(f"  batch done: {len(batch_results)}/{len(batch)}  "
              f"total={completed}/{len(records)}  remaining≈{remain/60:.0f}min", flush=True)
        if len(batch_results) < len(batch):
            failed = [b for b in batch if b["idx"] not in done]
            todo = failed + todo

    # Compute metrics
    predictions, ground_truth = [], []
    a1_preds_list = []
    for i, rec in enumerate(records):
        res = done.get(i)
        out = res["out"] if res else ""
        pred = parse_tickers(out, universe)
        eid = str(rec.get("event_id", i))
        gt = affected_by_event.get(eid, set())
        predictions.append(pred)
        ground_truth.append(gt)
        a1_preds_list.append(pred)

    metrics = compute_metrics(predictions, ground_truth)
    total = time.time() - t_start
    print(f"\nA1 done in {total/60:.1f}min")
    print(f"  N={metrics['n']}  P={metrics['precision']:.4f}  R={metrics['recall']:.4f}"
          f"  F1={metrics['f1']:.4f}  MRR={metrics['mrr']:.4f}", flush=True)

    # Build ablations_ff.json with A0 (=S4) and A1
    suite = json.loads((OUT_DIR / "baseline_suite_ff.json").read_text())
    s4_m = suite["systems"]["S4"]
    ablations = {
        "A0": {
            "ablation": "A0", "ablation_name": "Full system (hybrid_graph + RAFT)",
            "n": s4_m["n"], "precision": s4_m["precision"],
            "recall": s4_m["recall"], "f1": s4_m["f1"], "mrr": s4_m["mrr"],
            "brier": float("nan"),
        },
        "A1": {
            "ablation": "A1", "ablation_name": "Remove RAFT (hybrid_graph, base model)",
            **metrics, "brier": float("nan"),
        },
    }
    abl_json = {
        "label": LABEL, "split": "test",
        "model": MODEL_ID, "adapter": None,
        "note": "A2-A9 require retrieval context rebuild per ablation — not yet implemented",
        "ablations": ablations,
    }
    (OUT_DIR / f"ablations_{LABEL}.json").write_text(json.dumps(abl_json, indent=2, default=str))
    print(f"[saved] data/eval/ablations_{LABEL}.json", flush=True)

    # Add A1 predictions to the predictions file for bootstrap
    pred_path = OUT_DIR / f"baseline_suite_{LABEL}_predictions.json"
    preds_data = json.loads(pred_path.read_text())
    preds_data["predictions"]["A1"] = a1_preds_list
    pred_path.write_text(json.dumps(preds_data))
    print(f"[saved] updated predictions file with A1", flush=True)

    print("\nNow run: python 25_statistical_tests.py --label ff", flush=True)


if __name__ == "__main__":
    main()
