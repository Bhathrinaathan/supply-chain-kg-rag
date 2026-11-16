#!/usr/bin/env python3
"""
21_prep_finetune_data.py  —  Convert RAFT jsonl to fine-tuning formats.

Takes data/raft/{train,test,holdout}.jsonl (built by script 15) and writes:

  data/finetune/mlx/train.jsonl   — MLX-LM chat format ({"messages": [...]})
  data/finetune/mlx/valid.jsonl   — MLX-LM uses 'valid', not 'test'
  data/finetune/hf/train.jsonl    — HuggingFace / TRL SFTTrainer format (same schema)
  data/finetune/hf/test.jsonl
  data/finetune/hf/holdout.jsonl
  data/finetune/stats.json        — token-count estimates and split sizes

Llama 3.1 chat template: system / user / assistant messages using the Instruct format.
Both output formats use the same messages array; the difference is the filename convention.

USAGE
  python 21_prep_finetune_data.py
  python 21_prep_finetune_data.py --max-tokens 2048   # truncate long contexts

After running, see data/finetune/TRAINING_COMMANDS.md for the exact commands
to run on MLX (local M4) and Kaggle/Colab (QLoRA T4).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RAFT_DIR    = Path("data/raft")
OUT_DIR     = Path("data/finetune")
APPROX_CHARS_PER_TOKEN = 4   # rough estimate for logging

# Llama 3.1 special tokens (for token-count estimate only; actual tokenization by model)
BOS = "<|begin_of_text|>"
EOS = "<|eot_id|>"


def messages_to_llama3_text(messages: list[dict], add_generation_prompt: bool = False) -> str:
    """
    Apply Llama 3.1 Instruct chat template manually (no tokenizer needed here).
    Format:
      <|begin_of_text|><|start_header_id|>system<|end_header_id|>
      {system}<|eot_id|><|start_header_id|>user<|end_header_id|>
      {user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
      {assistant}<|eot_id|>
    """
    text = "<|begin_of_text|>"
    for msg in messages:
        role    = msg["role"]
        content = msg["content"].strip()
        text += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
    if add_generation_prompt:
        text += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return text


def approx_tokens(text: str) -> int:
    return len(text) // APPROX_CHARS_PER_TOKEN


def truncate_context(messages: list[dict], max_tokens: int) -> list[dict]:
    """
    If messages are too long, truncate the user's COMPANY PROFILES section
    to fit within max_tokens (approximate). System and assistant messages are kept whole.
    """
    estimated = approx_tokens("".join(m["content"] for m in messages))
    if estimated <= max_tokens:
        return messages

    user_msg = messages[1]["content"]
    if "COMPANY PROFILES:" in user_msg:
        question, profiles = user_msg.split("COMPANY PROFILES:", 1)
        # Trim profiles proportionally
        budget_chars = max_tokens * APPROX_CHARS_PER_TOKEN
        used_chars   = len(messages[0]["content"]) + len(messages[2]["content"]) + len(question)
        profile_budget = max(200, budget_chars - used_chars - 50)
        truncated_profiles = profiles[:int(profile_budget)] + "\n[profiles truncated]"
        messages = [
            messages[0],
            {"role": "user", "content": question + "COMPANY PROFILES:" + truncated_profiles},
            messages[2],
        ]
    return messages


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=3072,
                    help="Approximate max tokens per example (truncate if exceeded)")
    ap.add_argument("--raft-dir",   type=Path, default=RAFT_DIR)
    ap.add_argument("--out-dir",    type=Path, default=OUT_DIR)
    args = ap.parse_args()

    stats: dict = {"max_tokens": args.max_tokens, "splits": {}}
    token_counts: list[int] = []
    truncated_count = 0

    for split_name in ("train", "test", "holdout"):
        records = load_jsonl(args.raft_dir / f"{split_name}.jsonl")
        if not records:
            print(f"[warn] {split_name}.jsonl empty or missing — skipping")
            continue

        mlx_records: list[dict] = []
        hf_records:  list[dict] = []

        for r in records:
            msgs = r["messages"]
            # Truncate if needed
            orig_est = approx_tokens("".join(m["content"] for m in msgs))
            msgs = truncate_context(msgs, args.max_tokens)
            new_est  = approx_tokens("".join(m["content"] for m in msgs))
            if new_est < orig_est:
                truncated_count += 1
            token_counts.append(new_est)

            # MLX format — messages array (mlx-lm ≥0.19 supports this directly)
            mlx_records.append({
                "messages": msgs,
            })

            # HuggingFace/TRL format — same messages schema, plus metadata
            hf_records.append({
                "event_id":         r["event_id"],
                "primary_ticker":   r["primary_ticker"],
                "split":            r["split"],
                "label":            r.get("label", "ff"),
                "include_oracle":   r["include_oracle"],
                "affected_tickers": r["affected_tickers"],
                "messages":         msgs,
            })

        # MLX uses 'valid' not 'test'
        mlx_split = "valid" if split_name == "test" else split_name
        write_jsonl(mlx_records, args.out_dir / "mlx" / f"{mlx_split}.jsonl")
        write_jsonl(hf_records,  args.out_dir / "hf"  / f"{split_name}.jsonl")

        oracle_frac = sum(1 for r in records if r["include_oracle"]) / len(records)
        stats["splits"][split_name] = {
            "n": len(records),
            "oracle_fraction": round(oracle_frac, 3),
        }
        print(f"[ok] {split_name:8s}: {len(records):,} examples  oracle={oracle_frac:.1%}")

    if token_counts:
        import statistics
        stats["token_estimate"] = {
            "mean":   round(statistics.mean(token_counts)),
            "median": round(statistics.median(token_counts)),
            "p95":    sorted(token_counts)[int(0.95 * len(token_counts))],
            "max":    max(token_counts),
            "truncated": truncated_count,
        }
        print(f"\n[info] approx token counts: mean={stats['token_estimate']['mean']}  "
              f"p95={stats['token_estimate']['p95']}  truncated={truncated_count}")

    (args.out_dir / "stats.json").parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Write training commands reference
    _write_training_commands(args.out_dir)
    print(f"\n[done] -> {args.out_dir}/")
    print(f"         see {args.out_dir}/TRAINING_COMMANDS.md for exact run commands")
    return 0


def _write_training_commands(out_dir: Path) -> None:
    md = """\
# RAFT Fine-tuning Commands

Base model: **Llama 3.1 8B Instruct** (LOCKED — do not substitute a closed model).
Same base model across all LLM systems in the ablation (evaluation_protocol.md §2).

---

## Path A — Local MLX prototype (M4 Mac, 4-bit)

Requires Python 3.11 or 3.12 (mlx-lm has a transformers compatibility issue with 3.13).
Create a dedicated venv:

```bash
# One-time: install compatible stack
brew install python@3.11       # if not already installed
python3.11 -m venv .venv_mlx
source .venv_mlx/bin/activate
pip install "mlx-lm>=0.19" "transformers>=4.43,<4.46"

# Download base model (Hugging Face, requires login: huggingface-cli login)
# Model: meta-llama/Meta-Llama-3.1-8B-Instruct
# OR use a pre-quantised MLX version: mlx-community/Meta-Llama-3.1-8B-Instruct-4bit

# Fine-tune (LoRA rank-8, 4-bit quant, ~1 epoch)
mlx_lm.lora \\
  --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \\
  --train \\
  --data data/finetune/mlx \\
  --batch-size 1 \\
  --num-layers 16 \\
  --lora-rank 8 \\
  --iters 1500 \\
  --learning-rate 2e-5 \\
  --adapter-path models/raft_lora_mlx \\
  --grad-checkpoint

# Generate with adapter (smoke-test)
mlx_lm.generate \\
  --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \\
  --adapter-path models/raft_lora_mlx \\
  --prompt "NVDA filed an 8-K reporting earnings. Which semiconductor firms are affected?"
```

MLX prototype is for verifying the training loop and loss curve only.
Run ≥3 validation loss checkpoints to confirm convergence before committing to Kaggle time.

---

## Path B — Full QLoRA run (Kaggle 2×T4, free tier)

Upload `data/finetune/hf/` to a Kaggle dataset, then run `train_kaggle.py` in a notebook.
Kaggle free tier: 30 GPU h/week, 2×T4 (16 GB each).

Key hyperparameters (rag_architecture.md §4):
  - rank: 8–16  (start 8, increase if validation loss plateaus)
  - alpha: 2× rank
  - learning rate: 1–2e-5
  - epochs: 1 (avoid overfitting on 1,104 train examples)
  - batch size: 1 + gradient_accumulation_steps=4 → effective batch 4
  - gradient checkpointing: True
  - precision: 4-bit NF4 (bitsandbytes)

Reproducibility: log seed + dataset version + config hash to experiment_log.csv.
Report mean ± std over ≥3 seeds on headline metrics (rag_architecture.md §6).

---

## Evaluation after training

After fine-tuning, run against the test split:
  python 22_eval_raft.py --adapter models/raft_lora_mlx --split test --label ff

Expected: RAFT adapter > no-RAFT on held-out F1 / calibration. If not, report the
negative result — do not tune until it "works" (that would be leakage).
"""
    (out_dir / "TRAINING_COMMANDS.md").write_text(md)


if __name__ == "__main__":
    sys.exit(main())
