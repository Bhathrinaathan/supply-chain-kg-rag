#!/usr/bin/env python3
"""
25_statistical_tests.py  —  Stage 4-C E5: statistical significance tests.

Reads per-event predictions from baseline_suite_{label}.json and ablations_{label}.json,
then runs:
  T1  Paired bootstrap (B=1000) for F1 delta: S4 vs each other system
  T2  McNemar test for direction accuracy differences
  T3  Holm–Bonferroni FWER correction across all ablation comparisons
  T4  Benjamini–Hochberg FDR correction (alternative to T3)

Outputs:
  data/eval/stat_tests_{label}.json
  data/eval/stat_tests_summary.md

USAGE:
  python 25_statistical_tests.py --label ff

NOTE: requires that 22_eval_raft.py, 23_eval_baseline_suite.py, and 24_eval_ablations.py
have already been run and their JSON results are in data/eval/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

OUT_DIR = Path("data/eval")


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_f1_delta(
    pred_a: list[list[str]],
    pred_b: list[list[str]],
    ground_truth: list[set[str]],
    B: int = 1000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """
    Paired bootstrap for F1 difference (system A minus system B).
    Returns (observed_delta, p_value_one_sided, p_value_two_sided).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    def f1_per_event(pred, gt):
        if not gt: return None
        ps = set(pred); tp = len(ps & gt)
        p = tp / len(ps) if ps else 0.0
        r = tp / len(gt)
        return 2*p*r/(p+r) if (p+r) else 0.0

    scores_a = [f1_per_event(p, g) for p, g in zip(pred_a, ground_truth)]
    scores_b = [f1_per_event(p, g) for p, g in zip(pred_b, ground_truth)]
    pairs    = [(a, b) for a, b in zip(scores_a, scores_b) if a is not None]
    n        = len(pairs)

    obs_a = np.mean([p[0] for p in pairs])
    obs_b = np.mean([p[1] for p in pairs])
    obs_delta = obs_a - obs_b

    boot_deltas = []
    indices = np.arange(n)
    for _ in range(B):
        idx = rng.choice(indices, size=n, replace=True)
        ba  = np.mean([pairs[i][0] for i in idx])
        bb  = np.mean([pairs[i][1] for i in idx])
        boot_deltas.append(ba - bb)

    boot_deltas = np.array(boot_deltas)
    # Shift distribution to be centred on 0 (the null)
    shifted = boot_deltas - boot_deltas.mean()
    p_one   = float(np.mean(shifted >= obs_delta))
    p_two   = float(np.mean(np.abs(shifted) >= abs(obs_delta)))

    return obs_delta, p_one, p_two


# ---------------------------------------------------------------------------
# McNemar
# ---------------------------------------------------------------------------

def mcnemar_test(
    pred_a: list[list[str]],
    pred_b: list[list[str]],
    ground_truth: list[set[str]],
) -> tuple[float, float]:
    """
    McNemar test for direction accuracy.
    Returns (test_statistic, p_value) using continuity-corrected form.
    """
    from scipy.stats import chi2

    b, c = 0, 0  # b = A correct, B wrong; c = A wrong, B correct
    for pa, pb, gt in zip(pred_a, pred_b, ground_truth):
        if not gt: continue
        # "correct direction": at least one predicted ticker in GT
        ca = bool(set(pa) & gt)
        cb = bool(set(pb) & gt)
        if ca and not cb: b += 1
        if not ca and cb: c += 1

    if b + c == 0:
        return 0.0, 1.0
    stat = (abs(b - c) - 1)**2 / (b + c)
    pval = float(1 - chi2.cdf(stat, df=1))
    return stat, pval


# ---------------------------------------------------------------------------
# Multiple-comparison corrections
# ---------------------------------------------------------------------------

def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    n = len(p_values)
    order = sorted(range(n), key=lambda i: p_values[i])
    reject = [False] * n
    for rank, idx in enumerate(order):
        threshold = alpha / (n - rank)
        if p_values[idx] <= threshold:
            reject[idx] = True
        else:
            break  # once we fail, all subsequent also fail
    return reject


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    n = len(p_values)
    order = sorted(range(n), key=lambda i: p_values[i])
    reject = [False] * n
    for rank, idx in enumerate(order):
        if p_values[idx] <= alpha * (rank + 1) / n:
            reject[idx] = True
    # Enforce monotonicity: if rank k rejects, all lower ranks do too
    for rank in range(n - 2, -1, -1):
        if reject[order[rank + 1]]:
            reject[order[rank]] = True
    return reject


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label",  default="ff", choices=["ff", "loo", "pca"])
    ap.add_argument("--B",      type=int, default=1000, help="Bootstrap resamples")
    ap.add_argument("--alpha",  type=float, default=0.05)
    args = ap.parse_args()

    # Load pre-event predictions from eval results
    suite_path = OUT_DIR / f"baseline_suite_{args.label}.json"
    ablat_path = OUT_DIR / f"ablations_{args.label}.json"

    if not suite_path.exists():
        print(f"[FATAL] {suite_path} not found — run 23_eval_baseline_suite.py first")
        return 1

    suite = json.loads(suite_path.read_text())
    print(f"[info] baseline suite loaded: {list(suite['systems'].keys())}")

    # NOTE: per-event predictions are not stored in the suite JSON (only aggregate metrics).
    # For a proper paired bootstrap we need per-event F1 values.
    # This script runs the bootstrap from per-event predictions if a _predictions file exists,
    # otherwise it reports the aggregate metrics and notes that paired tests require per-event data.

    pred_path = OUT_DIR / f"baseline_suite_{args.label}_predictions.json"
    has_preds = pred_path.exists()

    results: dict = {"label": args.label, "B": args.B, "alpha": args.alpha, "tests": {}}

    if has_preds:
        preds_data = json.loads(pred_path.read_text())
        gt_sets    = [set(g) for g in preds_data["ground_truth"]]
        system_preds = preds_data["predictions"]  # {S0: [[...], ...], S1: ..., ...}

        rng = np.random.default_rng(42)

        # T1: paired bootstrap F1 delta — S4 vs all others
        if "S4" in system_preds:
            t1: dict = {}
            for sid, preds in system_preds.items():
                if sid == "S4": continue
                delta, p1, p2 = bootstrap_f1_delta(
                    system_preds["S4"], preds, gt_sets, B=args.B, rng=rng)
                t1[f"S4_vs_{sid}"] = {
                    "delta_f1": delta, "p_one_sided": p1, "p_two_sided": p2}
                print(f"  S4 vs {sid}: ΔF1={delta:+.4f}  p(one)={p1:.4f}  p(two)={p2:.4f}")
            results["tests"]["T1_bootstrap"] = t1

            # T2: McNemar direction
            t2: dict = {}
            for sid, preds in system_preds.items():
                if sid == "S4": continue
                stat, pval = mcnemar_test(system_preds["S4"], preds, gt_sets)
                t2[f"S4_vs_{sid}"] = {"mcnemar_stat": stat, "p_value": pval}
            results["tests"]["T2_mcnemar"] = t2

        # T3/T4: corrections across ablation comparisons if ablation preds available
        abl_pred_path = OUT_DIR / f"ablations_{args.label}_predictions.json"
        if abl_pred_path.exists() and "A0" in json.loads(abl_pred_path.read_text()).get("predictions", {}):
            abl_preds = json.loads(abl_pred_path.read_text())["predictions"]
            abl_gt    = [set(g) for g in json.loads(abl_pred_path.read_text())["ground_truth"]]
            comparisons, p_vals = [], []
            for aid, preds in abl_preds.items():
                if aid == "A0": continue
                delta, p1, _ = bootstrap_f1_delta(abl_preds["A0"], preds, abl_gt,
                                                   B=args.B, rng=rng)
                comparisons.append(f"A0_vs_{aid}")
                p_vals.append(p1)
            reject_hb = holm_bonferroni(p_vals, args.alpha)
            reject_bh = benjamini_hochberg(p_vals, args.alpha)
            t34 = {c: {"p": p, "holm_bonferroni_reject": rh, "bh_reject": rb}
                   for c, p, rh, rb in zip(comparisons, p_vals, reject_hb, reject_bh)}
            results["tests"]["T3_holm_bonferroni"] = t34
            results["tests"]["T4_bh_fdr"] = t34  # same data, different correction
    else:
        print("[WARN] per-event prediction files not found.")
        print("       Re-run 23_eval_baseline_suite.py with --save-predictions flag")
        print("       (add that flag to the script) to enable paired bootstrap tests.")
        print("       For now, reporting aggregate metrics only.")

        if suite.get("systems"):
            print("\nAggregate metrics from baseline suite:")
            for sid, m in suite["systems"].items():
                print(f"  {sid}: F1={m['f1']:.4f}  MRR={m['mrr']:.4f}")
        results["note"] = "Paired tests require per-event predictions. Re-run with --save-predictions."

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"stat_tests_{args.label}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[done] -> {out_path}")

    _write_summary(results, args.label)
    return 0


def _write_summary(results: dict, label: str):
    lines = [f"# Statistical Tests — label={label}  B={results['B']}  α={results['alpha']}\n"]
    tests = results.get("tests", {})
    if "T1_bootstrap" in tests:
        lines += ["## T1 — Paired bootstrap F1 delta (S4 = proposed system)",
                  "| Comparison | ΔF1 | p (one-sided) | p (two-sided) |",
                  "|------------|-----|---------------|---------------|"]
        for k, v in tests["T1_bootstrap"].items():
            lines.append(f"| {k} | {v['delta_f1']:+.4f} | {v['p_one_sided']:.4f} | {v['p_two_sided']:.4f} |")
        lines.append("")
    if "T2_mcnemar" in tests:
        lines += ["## T2 — McNemar direction accuracy",
                  "| Comparison | Stat | p |",
                  "|------------|------|---|"]
        for k, v in tests["T2_mcnemar"].items():
            lines.append(f"| {k} | {v['mcnemar_stat']:.4f} | {v['p_value']:.4f} |")
        lines.append("")
    if "note" in results:
        lines.append(f"> **Note:** {results['note']}")

    out_md = OUT_DIR / "stat_tests_summary.md"
    out_md.write_text("\n".join(lines) + "\n")
    print(f"[done] -> {out_md}")


if __name__ == "__main__":
    sys.exit(main())
