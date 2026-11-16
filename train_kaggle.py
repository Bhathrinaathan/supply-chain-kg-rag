#!/usr/bin/env python3
"""
train_kaggle.py  —  QLoRA fine-tune Llama 3.1 8B Instruct on the RAFT dataset.

Designed for Kaggle 2×T4 (16 GB each) or Google Colab T4.
Upload data/finetune/hf/ as a Kaggle dataset before running.

SETUP (in Kaggle notebook cell, before running this script):
    !pip install -q trl peft bitsandbytes transformers accelerate datasets

USAGE:
    python train_kaggle.py --data-dir /kaggle/input/raft-dataset/hf
    python train_kaggle.py --data-dir /kaggle/input/raft-dataset/hf --rank 16 --lr 1e-5

Model: meta-llama/Meta-Llama-3.1-8B-Instruct (HuggingFace, requires access token)
  OR:  unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit (pre-quantised, no token needed)

Config locked per rag_architecture.md §4:
  rank 8–16, alpha = 2×rank, lr 1–2e-5, ~1 epoch, batch 1 + grad_accum 4
  precision: 4-bit NF4 (bitsandbytes), gradient checkpointing
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",       default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--data-dir",    default="data/finetune/hf")
    ap.add_argument("--output-dir",  default="models/raft_llama31_8b")
    ap.add_argument("--rank",        type=int,   default=8)
    ap.add_argument("--alpha",       type=int,   default=None,  # defaults to 2×rank
                    help="LoRA alpha (default: 2×rank)")
    ap.add_argument("--lr",          type=float, default=2e-5)
    ap.add_argument("--epochs",      type=int,   default=1)
    ap.add_argument("--batch-size",  type=int,   default=1)
    ap.add_argument("--grad-accum",  type=int,   default=4)
    ap.add_argument("--max-tokens",  type=int,   default=3072)
    ap.add_argument("--seed",        type=int,   default=42)
    ap.add_argument("--hf-token",    default=os.environ.get("HF_TOKEN", ""),
                    help="HuggingFace token (or set HF_TOKEN env var)")
    args = ap.parse_args()

    if args.alpha is None:
        args.alpha = args.rank * 2

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                   BitsAndBytesConfig, TrainingArguments)
        from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
    except ImportError as e:
        print(f"[FATAL] Missing package: {e}")
        print("  pip install trl peft bitsandbytes transformers accelerate datasets")
        return 1

    # Llama 3.1 chat template: assistant turn starts after this token sequence
    RESPONSE_TEMPLATE = "<|start_header_id|>assistant<|end_header_id|>\n\n"

    print(f"[info] model={args.model}  rank={args.rank}  alpha={args.alpha}  lr={args.lr}")
    print(f"       epochs={args.epochs}  seed={args.seed}")

    # ------------------------------------------------------------------ load data
    def load_jsonl(path: str) -> list[dict]:
        with open(path) as f:
            return [json.loads(l) for l in f if l.strip()]

    data_dir = Path(args.data_dir)
    train_records = load_jsonl(data_dir / "train.jsonl")
    eval_records  = load_jsonl(data_dir / "test.jsonl")
    print(f"[info] train={len(train_records)}  eval={len(eval_records)}")

    # ------------------------------------------------------------------ tokenizer
    tok_kwargs = {"use_fast": True}
    if args.hf_token:
        tok_kwargs["token"] = args.hf_token
    tokenizer = AutoTokenizer.from_pretrained(args.model, **tok_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def apply_template(record: dict) -> dict:
        text = tokenizer.apply_chat_template(
            record["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    train_ds = Dataset.from_list(train_records).map(apply_template)
    eval_ds  = Dataset.from_list(eval_records).map(apply_template)

    # ------------------------------------------------------------------ model (4-bit)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model_kwargs = {
        "quantization_config": bnb_config,
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
    }
    if args.hf_token:
        model_kwargs["token"] = args.hf_token

    print("[info] loading model ...")
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.config.use_cache = False
    model.enable_input_require_grads()

    # ------------------------------------------------------------------ LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.rank,
        lora_alpha=args.alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ------------------------------------------------------------------ training
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=False,
        bf16=True,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        seed=args.seed,
        dataloader_pin_memory=False,
    )

    # Mask prompt tokens — train on assistant completions only (mirrors mask_prompt=true in MLX)
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=RESPONSE_TEMPLATE,
        tokenizer=tokenizer,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        dataset_text_field="text",
        max_seq_length=args.max_tokens,
        packing=False,
    )

    print("[info] training ...")
    trainer.train()

    # ------------------------------------------------------------------ save
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir + "/adapter")
    tokenizer.save_pretrained(args.output_dir + "/adapter")

    # Log config for reproducibility
    config_log = {
        "model": args.model, "rank": args.rank, "alpha": args.alpha,
        "lr": args.lr, "epochs": args.epochs, "seed": args.seed,
        "grad_accum": args.grad_accum, "max_tokens": args.max_tokens,
        "n_train": len(train_records), "n_eval": len(eval_records),
    }
    with open(args.output_dir + "/train_config.json", "w") as f:
        json.dump(config_log, f, indent=2)
    print(f"[done] adapter -> {args.output_dir}/adapter")
    print(f"       config  -> {args.output_dir}/train_config.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
