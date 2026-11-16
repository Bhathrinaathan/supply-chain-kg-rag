# Knowledge-Graph-Augmented Retrieval for Supply-Chain Event Impact Identification

Code for my MSc dissertation (Computer Science — Intelligent Systems, Trinity College Dublin, August 2026). Supervisor: Marco Ruffini.

**Question.** When a listed firm files a material-event report (SEC Form 8-K), connected firms often show abnormal price moves within days. Does an explicit knowledge graph of inter-company relationships, combined with retrieval-augmented generation (RAG) and retrieval-augmented fine-tuning (RAFT), help identify which firms are affected?

**What was built**

- A provenance-bearing knowledge graph over **149 firms** in the semiconductor, technology-hardware and automotive value chains: **1,496 nodes, 2,610 typed edges of 7 relationship types**, 1,444 of them approved by hand from 16,880 machine-generated candidates. Every edge carries a source, a verification timestamp and a validity date, so traversal is time-correct and leakage-free.
- An event corpus of **6,608 8-K filings**, 5,873 of them labelled under three abnormal-return benchmarks (Fama–French, leave-one-out, PCA).
- A **BM25 + dense + knowledge-graph-hop hybrid retriever**, a RAFT dataset builder (oracle/distractor chain-of-thought examples), and a **QLoRA adapter over Llama 3.1 8B Instruct** trained on Kaggle's free tier.
- A pre-registered evaluation with event-clustered bootstrap significance tests.

**What was found** (asymmetric, and reported as such)

| Hypothesis | Result |
|---|---|
| H1: graph proximity predicts post-event abnormal movement | **Supported.** One-hop firms show an affected rate +1.44 pp above unconnected firms, 95% CI [+0.10, +2.88], one-sided p = 0.016. |
| H2: graph-aware retrieval improves affected-firm identification | **Near-null.** At fixed candidate-set size, end-to-end F1 changed by +0.001 while the candidate set changed in 88% of events. |
| H3: fine-tuning improves it further | **Null.** With inference-time context held fixed, the adapter is indistinguishable from the base model (p = 0.386); an earlier apparent degradation was a context-construction confound. |

No language-model configuration approached a simple sector heuristic (F1 0.184 vs 0.074 for the best LLM system); recall was the binding constraint throughout. Contributions: the dataset and graph, the validated proximity signal, a methodological finding that benchmark self-reference can invert event-study conclusions in a relationship-dense universe, and a diagnosed negative result locating where the graph signal fails to transfer.

---

## Running the code

This is the source code for the data-collection, knowledge-graph-construction, retrieval,
fine-tuning, and evaluation pipeline described in the accompanying dissertation. It is submitted
as the electronic source-code component of the M.Sc. dissertation.

## What this is, and what it is not

This archive contains **code only**. It does not include:

- Any pulled or derived data (prices, SEC filings, Wikidata dumps, labelled event/ticker pairs,
  evaluation results).
- The trained RAFT adapter or any model weights.
- The knowledge graph itself, in any exported form.

Running the pipeline end-to-end from this archive means re-collecting all of the above from their
original public sources. It is not a one-command reproduction of the dissertation's reported
numbers from a bundled dataset — it is the method by which those numbers were originally produced,
runnable again against fresh pulls from SEC EDGAR, Yahoo Finance / Stooq, Wikidata, the Kenneth
French Data Library, and Apple's published supplier-responsibility PDFs. Re-running data collection
will not reproduce the exact historical price/filing snapshot used in the dissertation, since these
sources update over time; the pipeline logic is otherwise identical.

Two scripts present in the original project (`10_mine_competitors.py`, `11_mine_partners.py`) are
not included here — they performed an optional, LLM-assisted edge-mining pass over EDGAR filings
to propose `competitor_of`/`partner_of` candidate edges for human review. The rest of the pipeline
does not depend on their output; edges from these two relation types can also be added by hand
through `merge_research_edges.py`, which accepts a JSON array of researched relationships in any
form and folds them into the same human-review gate as every other edge source.

## Environment

Two separate Python environments are required, matching the two distinct dependency sets the
pipeline uses.

**`.venv`** — the data-collection, KG-construction, and labelling pipeline. Python 3.11.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**`.venv_mlx`** — the retrieval, RAFT-dataset, and MLX-inference/evaluation stages. Python 3.11 or
3.12 (`mlx-lm` has known issues on 3.13).

```bash
python3.11 -m venv .venv_mlx && source .venv_mlx/bin/activate
pip install "mlx-lm>=0.19" "transformers>=4.43,<4.46" chromadb sentence-transformers \
            rank_bm25 scipy numpy pandas pyarrow
```

Both environments need SEC EDGAR / Wikidata access to be configured before any pull:

```bash
export SEC_USER_AGENT="Your Name your.email@example.com"   # mandatory — SEC blocks requests without a real contact string
export NEO4J_PASSWORD="..."                                 # from a local Neo4j Desktop install
```

Neo4j Community Edition must be running locally at `bolt://localhost:7687` before stage 1's KG-load
step; the loader falls back to a NetworkX-only export if Neo4j is unreachable, so a Neo4j install is
convenient but not strictly required to exercise the rest of the pipeline.

Fine-tuning (`kaggle_notebook.ipynb`) is designed to run on Kaggle's free-tier 2×T4 GPU allocation,
not locally — see the notebook and `train_kaggle.py` (a script-form equivalent for a non-notebook
Kaggle run). `kaggle_notebook.ipynb` expects a `HF_TOKEN` Hugging Face access token to download the
base Llama 3.1 8B Instruct weights (gated on Hugging Face — request access first) — the placeholder
`YOUR_HF_TOKEN_HERE` in the notebook must be replaced with a real token before it will run; it is
never written back to disk by anything else in this pipeline.

## Running it

`run_pipeline.py` is a driver over the stage scripts below. It prints the commands it would run by
default; pass `--execute` to actually run a group.

```bash
python run_pipeline.py --list                       # show every stage, grouped
python run_pipeline.py --group data --execute        # stage 1: KG construction
python run_pipeline.py --group labels --execute      # stage 2: events + labelling + Gate L4
python run_pipeline.py --group retrieval --execute   # stage 3: hybrid retrieval
python run_pipeline.py --group raft --execute        # stage 4a: RAFT dataset construction
#  -- fine-tune on Kaggle here: kaggle_notebook.ipynb / train_kaggle.py --
python run_pipeline.py --group eval --execute        # stage 4b: evaluation (needs .venv_mlx)
```

Two points in the pipeline are **human-review gates**, not automatic steps — the driver stops and
prints what needs checking before continuing:

1. After entity resolution (`01_resolve_entities.py`): confirm ambiguous ticker→CIK/QID matches in
   `review/entity_resolution_review.csv`.
2. After edge candidate generation (`06_build_edges.py`): set `approved=Y` or `N` on every row of
   `review/candidate_edges.csv` before the KG load step. Nothing in this pipeline auto-approves an
   edge.

## Pipeline stages

| Stage | Script(s) | Produces |
|---|---|---|
| 1 — Knowledge graph | `01_resolve_entities.py` → `02_pull_prices.py` → `03_pull_edgar.py` → `04_pull_wikidata.py` → `05_parse_apple.py` → `06_build_edges.py` → `07_load_neo4j.py` → `08_verify_kg.py` → `09_export_snapshot.py` | A provenance-bearing, 7-edge-type supply-chain knowledge graph, gated on two hard checks (edge provenance completeness; temporal leakage prevention) |
| 2 — Events and labelling | `12_pull_events.py` → `13_label_affected.py` → `17_build_benchmarks.py` → `18_relabel_abnormal.py` → `18b_conditional_gradient.py` → `kp_significance.py` | 8-K event corpus, three market-adjusted "affected company" label sets (Fama-French/LOO/PCA), and an event-clustered bootstrap test of whether the knowledge-graph signal is statistically real |
| 3 — Retrieval | `14_build_retrieval_docs.py` → `19_hybrid_retrieval.py` → `20_eval_hybrid.py` | Firm-profile documents, a BM25 + dense + KG-hop-expansion hybrid retriever, and its own evaluation |
| 4a — RAFT dataset | `15_build_raft_dataset.py` → `21_prep_finetune_data.py` | Oracle/distractor-augmented chain-of-thought training examples, formatted for both the Kaggle (HF) and local (MLX) training paths |
| 4a (training) | `kaggle_notebook.ipynb` / `train_kaggle.py` | A QLoRA adapter over Llama 3.1 8B Instruct, trained on Kaggle's free-tier GPUs |
| 4b — Evaluation | `run_s4_final.py`, `run_a1_ablation.py`, `23_eval_baseline_suite.py`, `24_eval_ablations.py`, `25_statistical_tests.py` | The full system comparison and its statistical significance tests |

`16_eval_rag_baseline.py` and `22_eval_raft.py` are earlier evaluation scripts, superseded by
`20_eval_hybrid.py` and `23_eval_baseline_suite.py`/`run_s4_final.py` respectively; included for
completeness of the pipeline's history, not because either is on the critical path to the final
reported comparison.

## What this archive does not attempt to settle

The dissertation itself is the authoritative source for results, their statistical significance,
and every correction made to the evaluation methodology over the course of the project — none of
that is repeated here. This archive is the method, not the findings.
