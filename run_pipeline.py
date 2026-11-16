#!/usr/bin/env python3
"""
run_pipeline.py — driver for the supply-chain KG + RAFT pipeline.

Runs the numbered stage scripts in the order the dissertation used, grouped by which
Python environment each group needs (see README.md for the two-environment setup).

By default this only PRINTS the plan for the requested group — it does not execute
anything. Pass --execute to actually run the stages. Each stage script has its own
CLI flags (label, method, split, etc.); this driver calls each with reasonable
defaults and prints the exact command it runs, so a step can be re-run by hand with
different flags if needed.

USAGE
  python run_pipeline.py --list                     # show every stage and its group
  python run_pipeline.py --group data                # print stage 1 plan
  python run_pipeline.py --group data --execute       # actually run stage 1
  python run_pipeline.py --group labels --execute
  python run_pipeline.py --group retrieval --execute
  python run_pipeline.py --group raft --execute
  python run_pipeline.py --group eval --execute       # needs .venv_mlx active

Two human-review gates interrupt the "data" and "labels" groups (entity resolution
review, edge approval) — the driver stops and prints what to check before continuing,
it does not attempt to auto-approve anything.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

# Each stage: (script, args, human_review_note_after)
# human_review_note_after is None unless a manual review gate follows this step.
STAGES: dict[str, list[tuple[str, list[str], str | None]]] = {
    "data": [
        ("01_resolve_entities.py", [], "Review review/entity_resolution_review.csv for "
         "ambiguous ticker->CIK/QID matches before continuing."),
        ("02_pull_prices.py", [], None),
        ("03_pull_edgar.py", [], None),
        ("04_pull_wikidata.py", [], None),
        ("05_parse_apple.py", [], "Requires Apple supplier-responsibility PDFs to already "
         "be present under data/raw/apple/ (not distributed with this archive — download "
         "from apple.com and name them apple_suppliers_FY<year>.pdf)."),
        ("06_build_edges.py", [], "Set approved=Y/N on every row of "
         "review/candidate_edges.csv before loading — nothing here auto-approves."),
        ("07_load_neo4j.py", [], "Needs NEO4J_PASSWORD set and a local Neo4j instance "
         "running at bolt://localhost:7687; falls back to a NetworkX-only export if "
         "Neo4j is unreachable."),
        ("08_verify_kg.py", [], "Hard gate — must print GATE 1 PASS and GATE 2 PASS "
         "before proceeding to labelling."),
        ("09_export_snapshot.py", [], None),
    ],
    "labels": [
        ("12_pull_events.py", [], None),
        ("13_label_affected.py", [], None),
        ("17_build_benchmarks.py", [], "Builds all three abnormal-return benchmarks "
         "(FF/LOO/PCA) in one run — no --method flag, unlike the scripts below."),
        ("18_relabel_abnormal.py", ["--method", "ff"], None),
        ("18_relabel_abnormal.py", ["--method", "loo"], None),
        ("18_relabel_abnormal.py", ["--method", "pca"], None),
        ("18b_conditional_gradient.py", ["--method", "ff"], None),
        ("kp_significance.py", ["--method", "ff", "--conditional"],
         "Hard gate (Gate L4) — check data/labeled/kp_results_ff_cond.json for "
         "significant_05=true before proceeding to retrieval/RAFT."),
    ],
    "retrieval": [
        ("14_build_retrieval_docs.py", [], None),
        ("19_hybrid_retrieval.py", ["--build"], None),
        ("20_eval_hybrid.py", ["--labels", "ff", "loo", "pca"], None),
    ],
    "raft": [
        ("15_build_raft_dataset.py", ["--label", "ff", "--oracle-p", "0.6"], None),
        ("21_prep_finetune_data.py", [], None),
    ],
    "train": [
        # Not driven from here: training runs on Kaggle (2x T4 GPU, free tier).
        # Upload data/finetune/hf/ as a Kaggle dataset, run kaggle_notebook.ipynb there,
        # download the resulting adapter into models/raft_qlora_kaggle/.
        # train_kaggle.py is the script-form equivalent for a non-notebook Kaggle run.
    ],
    "eval": [
        ("run_s4_final.py", [], "Needs an MLX environment (.venv_mlx) and the trained "
         "adapter at models/raft_qlora_kaggle/."),
        ("run_a1_ablation.py", [], None),
        ("23_eval_baseline_suite.py", ["--label", "ff", "--split", "test",
         "--adapter", "models/raft_qlora_kaggle", "--save-predictions"], None),
        ("25_statistical_tests.py", ["--label", "ff", "--B", "2000"], None),
    ],
}

GROUP_ORDER = ["data", "labels", "retrieval", "raft", "train", "eval"]


def print_plan(group: str) -> None:
    stages = STAGES[group]
    if not stages:
        print(f"[{group}] no local steps — see the note in run_pipeline.py's STAGES dict.")
        return
    for script, args, note in stages:
        cmd = [sys.executable, script] + args
        print("  $", " ".join(cmd))
        if note:
            print(f"    -> HUMAN REVIEW GATE: {note}")


def run_group(group: str, execute: bool) -> int:
    stages = STAGES[group]
    if not stages:
        print(f"[{group}] no local steps to run — see run_pipeline.py's STAGES dict / README.md.")
        return 0

    for script, args, note in stages:
        cmd = [sys.executable, str(ROOT / script)] + args
        print(f"\n{'='*70}\n$ {' '.join(cmd)}\n{'='*70}")
        if not execute:
            continue
        result = subprocess.run(cmd, cwd=str(ROOT))
        if result.returncode != 0:
            print(f"[FATAL] {script} exited {result.returncode} — stopping.", file=sys.stderr)
            return result.returncode
        if note:
            print(f"\n>>> HUMAN REVIEW GATE — do not continue until this is done:\n>>> {note}\n")
            if execute:
                input(">>> Press Enter once the review above is complete to continue... ")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", choices=GROUP_ORDER, help="Which stage group to run/print")
    ap.add_argument("--execute", action="store_true",
                     help="Actually run the stage scripts (default: print the plan only)")
    ap.add_argument("--list", action="store_true", help="List every group and its stages, then exit")
    args = ap.parse_args()

    if args.list or not args.group:
        print("Pipeline groups, in dependency order (see README.md for the full narrative):\n")
        for g in GROUP_ORDER:
            print(f"[{g}]")
            print_plan(g)
            print()
        if not args.group:
            print("Pass --group <name> to run one group. Add --execute to actually run it "
                  "(default is a dry-run print of the commands).")
        return 0

    return run_group(args.group, args.execute)


if __name__ == "__main__":
    sys.exit(main())
