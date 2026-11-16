#!/usr/bin/env python3
"""
S4 eval — subprocess-per-batch to prevent Metal state accumulation.

Each batch of BATCH_SIZE examples runs in a fresh Python subprocess.
Results are saved per-example to a JSONL file; run is fully resumable.

Usage:
  source .venv_mlx/bin/activate
  PYTHONUNBUFFERED=1 python run_s4_final.py
"""
from __future__ import annotations
import json, os, re, subprocess, sys, time
from pathlib import Path

import numpy as np

ADAPTER    = Path("models/raft_qlora_kaggle")
MODEL_ID   = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
DATA_FILE  = Path("data/raft/test.jsonl")
LABEL_DIR  = Path("data/labeled")
OUT_DIR    = Path("data/eval")
LABEL      = "ff"
MAX_TOKENS = 256
BATCH_SIZE = 28          # safely below the observed ~75-example stall; fresh process each batch
BATCH_TIMEOUT = 900      # 15 min max per batch (28 × 21s ≈ 9.8 min); 53% headroom

SYSTEM_NAMES = {
    "S0": "TS-only (sector co-movement)",
    "S1": "LLM-only (no retrieval)",
    "S2": "LLM+RAG-text (BM25+BGE)",
    "S3": "LLM+RAG+KG (hybrid_graph)",
    "S4": "Proposed (hybrid_graph + RAFT)",
}

# ── worker code (runs in each subprocess) ─────────────────────────────────────

_WORKER = r'''
import json, sys, time, mlx.core as mx
from pathlib import Path
from mlx_lm import load, generate

ADAPTER   = Path("{adapter}")
MODEL_ID  = "{model_id}"
MAX_TOKENS = {max_tokens}

sys.stderr.write("worker: loading model...\n"); sys.stderr.flush()
model, tokenizer = load(MODEL_ID, adapter_path=str(ADAPTER))
sys.stderr.write("worker: ready\n"); sys.stderr.flush()

for raw in sys.stdin:
    raw = raw.strip()
    if not raw:
        continue
    rec = json.loads(raw)
    msgs = rec["messages"]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    t0 = time.time()
    out = generate(model, tokenizer, prompt=prompt, max_tokens=MAX_TOKENS, verbose=False)
    elapsed = time.time() - t0
    mx.eval(mx.array([0]))
    mx.clear_cache()
    result = {{"idx": rec["idx"], "out": out, "elapsed": round(elapsed, 2)}}
    sys.stdout.write(json.dumps(result) + "\n")
    sys.stdout.flush()
'''.format(adapter=str(ADAPTER), model_id=MODEL_ID, max_tokens=MAX_TOKENS)

# ── helpers ────────────────────────────────────────────────────────────────────

def load_universe_and_gt(label_dir: Path):
    import pandas as pd
    pairs_file = label_dir / f"event_ticker_pairs_{LABEL}.parquet"
    if not pairs_file.exists():
        pairs_file = label_dir / "event_ticker_pairs.parquet"
    pairs = pd.read_parquet(pairs_file)
    universe = set(pairs["ticker"].unique())
    affected: dict[str, set[str]] = {}
    for eid, grp in pairs[pairs["is_affected"] == True].groupby("event_id"):  # noqa: E712
        affected[str(eid)] = set(grp["ticker"].tolist())
    return universe, affected

def parse_tickers(text: str, universe: set[str]) -> list[str]:
    out = text.upper()
    found, seen = [], set()
    m = re.search(r"(?:answer|affected tickers?)\s*:\s*([^\n]+)", out, re.IGNORECASE)
    if m:
        for c in re.split(r"[,;\s]+", m.group(1)):
            c = c.strip().rstrip(".")
            if c in universe and c not in seen:
                found.append(c); seen.add(c)
        if found:
            return found
    for ticker in sorted(universe, key=len, reverse=True):
        if re.search(r"\b" + re.escape(ticker) + r"\b", out):
            if ticker not in seen:
                found.append(ticker); seen.add(ticker)
    return found

def compute_metrics(predictions, ground_truth):
    prec, rec, f1v, mrrv = [], [], [], []
    for pred, gt in zip(predictions, ground_truth):
        if not gt:
            continue
        ps = set(pred)
        tp = len(ps & gt)
        p = tp / len(ps) if ps else 0.0
        r = tp / len(gt)
        prec.append(p); rec.append(r)
        f1v.append(2*p*r/(p+r) if (p+r) > 0 else 0.0)
        for j, t in enumerate(pred):
            if t in gt:
                mrrv.append(1.0/(j+1)); break
        else:
            mrrv.append(0.0)
    avg = lambda lst: float(np.mean(lst)) if lst else 0.0
    return {
        "n": len([g for g in ground_truth if g]),
        "precision": avg(prec), "recall": avg(rec),
        "f1": avg(f1v), "mrr": avg(mrrv),
        "direction_accuracy": None, "direction_n": 0,
    }

def write_summary(systems: dict, n: int):
    lines = [
        f"# Baseline Suite Results — label={LABEL}  split=test  n={n}\n",
        "| System | P | R | F1 | MRR | Dir.Acc |",
        "|--------|---|---|----|-----|---------|",
    ]
    for sid in ("S0", "S1", "S2", "S3", "S4"):
        if sid not in systems:
            continue
        m = systems[sid]
        da = f"{m['direction_accuracy']:.3f}" if m.get("direction_accuracy") else "—"
        lines.append(
            f"| {sid}: {m['system_name']} "
            f"| {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} "
            f"| {m['mrr']:.3f} | {da} |"
        )
    lines += [
        "",
        f"*Primary benchmark: {LABEL.upper()}. Same base model (Llama 3.1 8B Instruct) across S1–S4.*",
        "*S4 uses RAFT LoRA adapter; all others use the unmodified base model.*",
    ]
    out_md = OUT_DIR / "baseline_suite_summary.md"
    out_md.write_text("\n".join(lines) + "\n")
    print(f"[saved] {out_md}", flush=True)
    print("\n" + "\n".join(lines), flush=True)

# ── batch runner ───────────────────────────────────────────────────────────────

def run_batch(batch: list[dict]) -> list[dict]:
    """Run one batch in a fresh subprocess. Returns list of result dicts."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        [sys.executable, "-c", _WORKER],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        cwd=str(Path.cwd()),
    )

    # Feed all examples to worker
    for rec in batch:
        proc.stdin.write(json.dumps({"idx": rec["idx"], "messages": rec["messages"][:-1]}) + "\n")
    proc.stdin.close()

    results = []
    deadline = time.time() + BATCH_TIMEOUT

    for i, rec in enumerate(batch):
        remaining = deadline - time.time()
        if remaining <= 0:
            print(f"  [BATCH TIMEOUT] at idx={rec['idx']}, killing worker", flush=True)
            proc.terminate()
            break

        import select
        ready, _, _ = select.select([proc.stdout], [], [], remaining)
        if not ready:
            print(f"  [ITEM TIMEOUT]  idx={rec['idx']}", flush=True)
            proc.terminate()
            break

        line = proc.stdout.readline()
        if not line:
            print(f"  [WORKER EOF]    idx={rec['idx']}", flush=True)
            break

        result = json.loads(line.strip())
        results.append(result)
        print(f"  ex {result['idx']:3d}: {result['elapsed']:.1f}s", flush=True)

    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()

    return results

# ── main ───────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    records = [json.loads(l) for l in DATA_FILE.read_text().splitlines() if l.strip()]
    for i, r in enumerate(records):
        r["idx"] = i

    # Resume: load already-completed example indices
    per_ex_file = OUT_DIR / "s4_per_example.jsonl"
    done: dict[int, dict] = {}
    if per_ex_file.exists():
        for line in per_ex_file.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["idx"]] = r

    todo = [r for r in records if r["idx"] not in done]
    print(f"Total: {len(records)}  already done: {len(done)}  remaining: {len(todo)}", flush=True)

    universe, affected_by_event = load_universe_and_gt(LABEL_DIR)

    t_start = time.time()
    batch_num = 0

    while todo:
        batch = todo[:BATCH_SIZE]
        todo  = todo[BATCH_SIZE:]
        batch_num += 1
        start_idx = batch[0]["idx"]
        end_idx   = batch[-1]["idx"]
        print(f"\nBatch {batch_num}: idx {start_idx}–{end_idx}  "
              f"({len(batch)} examples, timeout={BATCH_TIMEOUT}s)", flush=True)

        batch_results = run_batch(batch)

        # Persist each result
        with per_ex_file.open("a") as f:
            for res in batch_results:
                f.write(json.dumps(res) + "\n")
                done[res["idx"]] = res

        completed = len(done)
        elapsed   = time.time() - t_start
        avg_s     = elapsed / completed if completed else 0
        remain    = (len(records) - completed) * avg_s
        print(f"  batch done: {len(batch_results)}/{len(batch)}  "
              f"total={completed}/{len(records)}  "
              f"avg={avg_s:.1f}s  remaining≈{remain/60:.0f}min", flush=True)

        # If batch was incomplete (timeout), re-queue the failed examples individually
        if len(batch_results) < len(batch):
            failed = [b for b in batch if b["idx"] not in done]
            print(f"  re-queuing {len(failed)} failed examples one by one", flush=True)
            todo = failed + todo

    # Build final metrics
    predictions, ground_truth = [], []
    for i, rec in enumerate(records):
        res = done.get(i)
        out = res["out"] if res else ""
        pred = parse_tickers(out, universe)
        eid  = str(rec.get("event_id", i))
        gt   = affected_by_event.get(eid, set())
        predictions.append(pred)
        ground_truth.append(gt)

    metrics = compute_metrics(predictions, ground_truth)
    metrics["system_name"] = SYSTEM_NAMES["S4"]
    total = time.time() - t_start
    skipped = len(records) - len(done)
    print(f"\nS4 done in {total/60:.1f}min  (skipped={skipped})", flush=True)
    print(f"  N={metrics['n']}  P={metrics['precision']:.4f}  R={metrics['recall']:.4f}"
          f"  F1={metrics['f1']:.4f}  MRR={metrics['mrr']:.4f}", flush=True)

    # Merge with existing S0-S3 results
    out_json = OUT_DIR / f"baseline_suite_{LABEL}.json"
    existing: dict = {}
    if out_json.exists():
        existing = json.loads(out_json.read_text())
        print(f"Merging with: {list(existing.get('systems', {}).keys())}", flush=True)

    existing.setdefault("systems", {})["S4"] = metrics
    existing.update({
        "label": LABEL, "split": "test",
        "model": MODEL_ID, "adapter": str(ADAPTER),
        "n_examples": len(records),
    })
    out_json.write_text(json.dumps(existing, indent=2))
    print(f"[saved] {out_json}", flush=True)

    write_summary(existing["systems"], len(records))


if __name__ == "__main__":
    main()
