#!/usr/bin/env python3
"""
compare_baselines.py — Analyse SEACR results vs baseline

Loads evaluation_summary.json files from a baseline run and a SEACR run
and prints a structured comparison: overall accuracy, per-category, per-
calculator, and FMAS (if available).

Usage:
    python pipeline/compare_baselines.py \\
        --baseline-dir ../outputs/contrastive_evaluation_20251205_031352_test1047 \\
        --seacr-dir    ../outputs/contrastive_evaluation_XXXXXXXX \\
        [--output-csv  ../outputs/comparison_results.csv]

Can also compare multiple SEACR dirs (comma-separated --seacr-dir).
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional


def load_summary(eval_dir: str) -> Optional[Dict]:
    path = Path(eval_dir) / "evaluations" / "evaluation_summary.json"
    if not path.exists():
        print(f"⚠️  No evaluation_summary.json at {eval_dir}")
        return None
    with open(path) as f:
        return json.load(f)


def load_fmas(eval_dir: str) -> Optional[float]:
    path = Path(eval_dir) / "evaluations" / "fmas_report.json"
    if path.exists():
        with open(path) as f:
            return json.load(f).get("fmas")
    # Also check top-level fmas key in eval_summary
    s = load_summary(eval_dir)
    return s.get("fmas") if s else None


def format_delta(delta: float) -> str:
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta*100:.2f}pp"


def compare(baseline_dir: str, seacr_dirs: List[str], output_csv: Optional[str] = None):
    baseline = load_summary(baseline_dir)
    if baseline is None:
        print("❌ Cannot load baseline. Aborting.")
        sys.exit(1)

    base_acc = baseline.get("contrastive_few_shot", {}).get("overall_accuracy", 0.0)
    base_model = baseline.get("model", "unknown")
    base_n = baseline.get("test_set_size", "?")

    print("=" * 70)
    print("BASELINE vs SEACR COMPARISON")
    print("=" * 70)
    print(f"\nBaseline: {Path(baseline_dir).name}")
    print(f"  Model:    {base_model}")
    print(f"  N:        {base_n}")
    print(f"  Accuracy: {base_acc:.4f} ({base_acc*100:.2f}%)")

    rows = []

    for seacr_dir in seacr_dirs:
        seacr = load_summary(seacr_dir)
        if seacr is None:
            continue
        fmas = load_fmas(seacr_dir)

        seacr_acc = seacr.get("contrastive_few_shot", {}).get("overall_accuracy", 0.0)
        seacr_model = seacr.get("model", "unknown")
        seacr_n = seacr.get("test_set_size", "?")
        delta = seacr_acc - base_acc

        print(f"\nSEACR run: {Path(seacr_dir).name}")
        print(f"  Model:    {seacr_model}")
        print(f"  N:        {seacr_n}")
        print(f"  Accuracy: {seacr_acc:.4f} ({seacr_acc*100:.2f}%)")
        print(f"  Δ vs baseline: {format_delta(delta)}")
        if fmas is not None:
            print(f"  FMAS:     {fmas:.4f}  (bank alignment score)")

        # Per-category comparison
        base_cats = baseline.get("contrastive_few_shot", {}).get("by_category", {})
        seacr_cats = seacr.get("contrastive_few_shot", {}).get("by_category", {})
        if base_cats and seacr_cats:
            print("\n  Per-category accuracy:")
            print(f"  {'Category':<15} {'Baseline':>10} {'SEACR':>10} {'Δ':>10}")
            print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10}")
            for cat in sorted(set(list(base_cats.keys()) + list(seacr_cats.keys()))):
                b_acc = base_cats.get(cat, {}).get("accuracy", None)
                s_acc = seacr_cats.get(cat, {}).get("accuracy", None)
                if b_acc is not None and s_acc is not None:
                    d = s_acc - b_acc
                    marker = " ↑" if d > 0.01 else (" ↓" if d < -0.01 else "  ")
                    print(f"  {cat:<15} {b_acc*100:>9.1f}% {s_acc*100:>9.1f}% {format_delta(d):>9}{marker}")

        # Per-calculator: top gains and losses
        base_calcs = baseline.get("contrastive_few_shot", {}).get("by_calculator", {})
        seacr_calcs = seacr.get("contrastive_few_shot", {}).get("by_calculator", {})
        if base_calcs and seacr_calcs:
            deltas = []
            for calc in set(list(base_calcs.keys()) + list(seacr_calcs.keys())):
                b = base_calcs.get(calc, {}).get("accuracy", None)
                s = seacr_calcs.get(calc, {}).get("accuracy", None)
                if b is not None and s is not None:
                    deltas.append((calc, b, s, s - b))

            deltas.sort(key=lambda x: x[3], reverse=True)
            print(f"\n  Top 5 calculator improvements:")
            for calc, b, s, d in deltas[:5]:
                print(f"    {calc[:45]:<45} {b*100:.0f}% → {s*100:.0f}% ({format_delta(d)})")

            print(f"\n  Top 5 calculator regressions:")
            for calc, b, s, d in deltas[-5:]:
                print(f"    {calc[:45]:<45} {b*100:.0f}% → {s*100:.0f}% ({format_delta(d)})")

        rows.append({
            "run": Path(seacr_dir).name,
            "model": seacr_model,
            "n": seacr_n,
            "baseline_acc": round(base_acc, 4),
            "seacr_acc": round(seacr_acc, 4),
            "delta_pp": round((seacr_acc - base_acc) * 100, 2),
            "fmas": round(fmas, 4) if fmas is not None else None,
        })

    # Save CSV if requested
    if output_csv and rows:
        import csv
        fieldnames = ["run", "model", "n", "baseline_acc", "seacr_acc", "delta_pp", "fmas"]
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n💾 Comparison saved to: {output_csv}")

    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"\n{'Run':<45} {'Acc':>8} {'Δ':>10} {'FMAS':>8}")
    print(f"{'Baseline: ' + Path(baseline_dir).name:<45} {base_acc*100:>7.2f}%   {'(ref)':>10}")
    for row in rows:
        fmas_str = f"{row['fmas']:.4f}" if row["fmas"] is not None else "  N/A "
        print(f"{row['run']:<45} {row['seacr_acc']*100:>7.2f}% {format_delta(row['seacr_acc']-base_acc):>10} {fmas_str:>8}")

    if rows:
        best = max(rows, key=lambda r: r["seacr_acc"])
        print(f"\n🏆 Best SEACR run: {best['run']}")
        print(f"   Accuracy: {best['seacr_acc']*100:.2f}%  (Δ {format_delta(best['seacr_acc']-base_acc)} vs baseline)")


def main():
    parser = argparse.ArgumentParser(
        description="Compare SEACR evaluation results against baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--baseline-dir", required=True,
                        help="Path to baseline contrastive_evaluation_* directory")
    parser.add_argument("--seacr-dir", required=True,
                        help="Path(s) to SEACR contrastive_evaluation_* directory "
                             "(comma-separated for multiple)")
    parser.add_argument("--output-csv", default=None,
                        help="Save comparison table to CSV file")
    args = parser.parse_args()

    seacr_dirs = [d.strip() for d in args.seacr_dir.split(",") if d.strip()]
    compare(args.baseline_dir, seacr_dirs, args.output_csv)


if __name__ == "__main__":
    main()
