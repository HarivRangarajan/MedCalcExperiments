#!/usr/bin/env python3
"""
Stage 3: Bank Lifecycle Manager
================================

The demonstration bank is a PERSISTENT artifact built once from a previous
model's failures. It is intentionally reused across model generations — the
whole research claim is that a bank built from gpt-4o failures can remain
useful (possibly degraded) when applied to gpt-5, gpt-6, etc.

This manager does NOT destroy or rebuild the bank. Instead, after each
model generation is evaluated (Stage 2), it:

  1. Loads the evaluation_summary.json (per-calculator accuracy + FMAS)
  2. Computes U(d, g) = difficulty(d) × (1 - accuracy(g, calc(d))) for every entry
  3. Tags each bank entry in-place with its utility for this generation:
         entry["utility_by_generation"][model_name] = U
  4. Records FMAS and utility statistics in bank_metadata.json under the
     generation key — building a cross-generation utility decay curve
  5. Optionally archives low-utility entries (toggle: --archive / --no-archive)

The "bank coverage warning" (formerly "rebuild recommendation") is now
purely diagnostic: it tells you how well the bank, as originally built,
still covers the current model's errors. It never triggers an action.

Archival modes:
  --no-archive (default): all entries stay in bank.jsonl; utilities are
      tagged in the entries and in metadata only. Use this for observing
      cross-generation utility decay without destroying bank state.
  --archive: physically remove entries where U(d,g) < epsilon from bank.jsonl
      and write them to archived/archived_gen{N}_{model}.jsonl. Use this
      for periodic bank pruning when you want to reduce bank size.

Utility formula:
    U(d, g) = difficulty(d) × (1 - accuracy(g, calc(d)))
    difficulty(d) = contrastive_sharpness  (stored in bank entry)

Usage:
    # Observe cross-generation utility (bank unchanged):
    python pipeline/bank_lifecycle_manager.py \\
        --bank-dir ../outputs/seacr_bank_XYZ \\
        --eval-summary ../outputs/contrastive_evaluation_XYZ/evaluations/evaluation_summary.json \\
        --model-name gpt-5 --generation 2

    # Periodic archival (physically prune low-utility entries):
    python pipeline/bank_lifecycle_manager.py \\
        --bank-dir ../outputs/seacr_bank_XYZ \\
        --eval-summary ../outputs/contrastive_evaluation_XYZ/evaluations/evaluation_summary.json \\
        --model-name gpt-5 --generation 2 --archive
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


class BankLifecycleManager:
    """
    Manages demonstration bank lifecycle across model generations.

    The bank is a persistent artifact — it is never automatically destroyed.
    U(d, g) scores are tagged per generation to produce a cross-generation
    utility decay curve. Archival is an explicit, optional action.
    """

    def __init__(
        self,
        bank_dir: str,
        epsilon: float = 0.05,
        fmas_threshold: float = 0.15,
    ):
        """
        Args:
            bank_dir:        Path to seacr_bank_* directory.
            epsilon:         Utility threshold below which an entry is considered
                             "low utility" for the purpose of reporting and (optional) archival.
            fmas_threshold:  FMAS below this emits a bank coverage warning.
        """
        self.bank_dir = Path(bank_dir)
        self.epsilon = epsilon
        self.fmas_threshold = fmas_threshold

        self.bank: List[Dict] = []
        bank_file = self.bank_dir / "bank.jsonl"
        if bank_file.exists():
            with open(bank_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.bank.append(json.loads(line))
        else:
            print(f"⚠️  bank.jsonl not found at {bank_file} — bank is empty")

        meta_file = self.bank_dir / "bank_metadata.json"
        if meta_file.exists():
            with open(meta_file) as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
            print(f"⚠️  bank_metadata.json not found — starting with empty metadata")

        self.archived_dir = self.bank_dir / "archived"
        self.archived_dir.mkdir(exist_ok=True)

        # Record original size once (on first lifecycle run)
        if "original_bank_size" not in self.metadata:
            self.metadata["original_bank_size"] = len(self.bank)

        print(f"✅ BankLifecycleManager loaded: {len(self.bank)} entries "
              f"(original_size={self.metadata['original_bank_size']})")

    # ------------------------------------------------------------------
    # Core utility computation
    # ------------------------------------------------------------------

    def _get_calculator_accuracy(
        self, calculator_name: str, eval_summary: Dict,
        prompt_type: str = "contrastive_few_shot"
    ) -> float:
        """
        Look up per-calculator accuracy from evaluation_summary.json.
        Falls back to overall accuracy if the calculator is not found.
        """
        by_calc = eval_summary.get(prompt_type, {}).get("by_calculator", {})
        if calculator_name in by_calc:
            return float(by_calc[calculator_name].get("accuracy", 0.0))
        return float(eval_summary.get(prompt_type, {}).get("overall_accuracy", 0.0))

    def compute_utility(self, entry: Dict, eval_summary: Dict) -> float:
        """
        Polarity-aware utility:

        Negative entries:
            U(d, g) = contrastive_sharpness(d) × (1 - accuracy(g, calc(d)))
            High when model still fails on this calculator.

        Positive entries:
            U(d, g) = 1.0 - accuracy(g, calc(d))
            High when model needs guidance on this calculator.

        Returns float in [0, 1].
        """
        polarity = entry.get("polarity")
        result = entry.get("Result", "")
        is_positive = (polarity == "positive") or (polarity is None and result == "Correct")

        calc_name = entry.get("Calculator Name", "")
        accuracy  = self._get_calculator_accuracy(calc_name, eval_summary)

        if is_positive:
            return 1.0 - accuracy
        else:
            difficulty = float(entry.get("contrastive_sharpness", 0.5))
            return difficulty * (1.0 - accuracy)

    def _compute_all_utilities(
        self, eval_summary: Dict
    ) -> Tuple[List[float], int]:
        """Compute U(d, g) for every entry. Returns (utility_list, below_epsilon_count)."""
        utilities = [self.compute_utility(e, eval_summary) for e in self.bank]
        below_epsilon = sum(1 for u in utilities if u < self.epsilon)
        return utilities, below_epsilon

    # ------------------------------------------------------------------
    # Main lifecycle update
    # ------------------------------------------------------------------

    def update_generation(
        self,
        eval_summary_path: str,
        model_name: str,
        generation: int,
        archive: bool = False,
    ) -> Dict:
        """
        Tag every bank entry with its utility for this model generation,
        update bank_metadata.json, and optionally archive low-utility entries.

        The bank.jsonl is ALWAYS rewritten (to persist the utility tags),
        but entries are only removed when archive=True.

        Args:
            eval_summary_path: Path to evaluation_summary.json from Stage 2
            model_name:        e.g. "gpt-5"
            generation:        Generation number (1, 2, 3, ...)
            archive:           If True, remove entries where U < epsilon from bank.jsonl.
                               If False (default), tag utilities but keep all entries.

        Returns:
            Dict with generation stats: mean_utility, below_epsilon_count,
            fmas, coverage_warning, archived_count (0 if archive=False)
        """
        with open(eval_summary_path) as f:
            eval_summary = json.load(f)

        fmas = eval_summary.get("fmas")   # written by Stage 2 Change D

        # 1. Compute utilities for all entries
        utilities, below_epsilon_count = self._compute_all_utilities(eval_summary)

        # 2. Tag each entry with its utility for this generation
        gen_key = f"{model_name}_gen{generation}"
        for entry, u in zip(self.bank, utilities):
            entry.setdefault("utility_by_generation", {})[gen_key] = round(u, 4)

        # 3. Split into active/archived (used only if archive=True)
        active_entries, archived_entries = [], []
        for entry, u in zip(self.bank, utilities):
            if archive and u < self.epsilon:
                entry["archived_at_generation"] = generation
                entry["archived_for_model"]     = model_name
                archived_entries.append(entry)
            else:
                active_entries.append(entry)

        # 4. Write archived file if archival was requested
        if archive and archived_entries:
            archived_file = self.archived_dir / f"archived_gen{generation}_{model_name}.jsonl"
            with open(archived_file, 'w') as f:
                for entry in archived_entries:
                    f.write(json.dumps(entry) + "\n")
            self.bank = active_entries

        # 5. Rewrite bank.jsonl (always, to persist utility tags)
        with open(self.bank_dir / "bank.jsonl", 'w') as f:
            for entry in self.bank:
                f.write(json.dumps(entry) + "\n")

        # 6. Update bank_metadata.json with per-generation stats
        # Count by polarity
        positive_count = sum(
            1 for e in self.bank
            if e.get("polarity") == "positive" or
               (e.get("polarity") is None and e.get("Result") == "Correct")
        )
        negative_count = len(self.bank) - positive_count
        positive_archived = sum(
            1 for e in archived_entries
            if e.get("polarity") == "positive" or
               (e.get("polarity") is None and e.get("Result") == "Correct")
        ) if archive else 0
        negative_archived = (len(archived_entries) - positive_archived) if archive else 0

        gen_stats = {
            "mean_utility":         round(float(np.mean(utilities)), 4) if utilities else 0.0,
            "min_utility":          round(float(np.min(utilities)),  4) if utilities else 0.0,
            "max_utility":          round(float(np.max(utilities)),  4) if utilities else 0.0,
            "below_epsilon_count":  below_epsilon_count,
            "below_epsilon_frac":   round(below_epsilon_count / len(utilities), 3) if utilities else 0.0,
            "archived":             len(archived_entries) if archive else 0,
            "positive_archived":    positive_archived,
            "negative_archived":    negative_archived,
            "bank_size_after":      len(self.bank),
            "positive_remaining":   positive_count,
            "negative_remaining":   negative_count,
            "fmas":                 round(fmas, 4) if fmas is not None else None,
        }
        self.metadata.setdefault("generation_history", {})[gen_key] = gen_stats
        if fmas is not None:
            self.metadata.setdefault("fmas_history", {})[gen_key] = round(fmas, 4)
        self.metadata["last_generation"]  = generation
        self.metadata["last_model"]       = model_name
        self.metadata["active_count"]     = len(self.bank)
        with open(self.bank_dir / "bank_metadata.json", 'w') as f:
            json.dump(self.metadata, f, indent=2)

        # 7. Compute coverage warning (purely diagnostic)
        warning = self.bank_coverage_warning(fmas, below_epsilon_count, len(utilities))

        report = {
            "generation":           generation,
            "model_name":           model_name,
            "gen_key":              gen_key,
            "total_entries":        len(utilities),
            "archived_count":       len(archived_entries) if archive else 0,
            "remaining_count":      len(self.bank),
            "below_epsilon_count":  below_epsilon_count,
            "mean_utility":         gen_stats["mean_utility"],
            "fmas":                 round(fmas, 4) if fmas is not None else None,
            "coverage_warning":     warning,
        }

        mode_label = "ARCHIVE MODE" if archive else "TAG-ONLY (bank unchanged)"
        print(f"\n🗂  Lifecycle update [{mode_label}] — {gen_key}:")
        print(f"   • Total entries:      {len(utilities)}")
        print(f"   • U < ε ({self.epsilon}):    {below_epsilon_count} entries "
              f"({gen_stats['below_epsilon_frac']*100:.1f}% low-utility for {model_name})")
        if archive:
            print(f"   • Archived:           {len(archived_entries)} entries removed")
            print(f"   • Remaining:          {len(self.bank)} entries")
        print(f"   • Mean U:             {gen_stats['mean_utility']:.4f}  "
              f"(min={gen_stats['min_utility']:.4f}  max={gen_stats['max_utility']:.4f})")
        if fmas is not None:
            print(f"   • FMAS:               {fmas:.4f}  "
                  f"({'⚠️  LOW' if fmas < self.fmas_threshold else '✅ OK'})")
        if warning["any"]:
            print(f"   • ⚠️  Coverage warning: {warning['reason']}")

        return report

    # ------------------------------------------------------------------
    # Cross-generation utility summary
    # ------------------------------------------------------------------

    def print_decay_curve(self):
        """
        Print the cross-generation utility decay curve from bank_metadata.json.
        Shows how bank utility evolves as newer model generations are evaluated.
        """
        history = self.metadata.get("generation_history", {})
        if not history:
            print("No generation history recorded yet.")
            return

        print("\n📈 Cross-Generation Utility Decay Curve")
        print("=" * 65)
        print(f"  {'Generation':<25} {'Mean U':>8} {'Low-U%':>8} {'FMAS':>8} {'Archived':>10}")
        print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
        for gen_key, stats in sorted(history.items()):
            fmas_str = f"{stats['fmas']:.4f}" if stats.get("fmas") is not None else "   N/A"
            arch_str = str(stats.get("archived", 0))
            low_pct  = f"{stats.get('below_epsilon_frac', 0)*100:.1f}%"
            print(f"  {gen_key:<25} {stats['mean_utility']:>8.4f} {low_pct:>8} {fmas_str:>8} {arch_str:>10}")

        # Show trend
        mean_utils = [s["mean_utility"] for s in history.values()]
        if len(mean_utils) > 1:
            decay = mean_utils[0] - mean_utils[-1]
            print(f"\n  Utility decay (first→last generation): {decay:+.4f}")

    # ------------------------------------------------------------------
    # Bank coverage warning (diagnostic only — no action)
    # ------------------------------------------------------------------

    def bank_coverage_warning(
        self,
        fmas: Optional[float] = None,
        below_epsilon_count: int = 0,
        total: int = 1,
    ) -> Dict:
        """
        Returns a diagnostic warning dict if the bank may be poorly aligned
        with the current model. This is NEVER an action trigger — it is
        purely an observation to inform your research analysis.

        Warning conditions:
          1. FMAS < fmas_threshold: wrong-answer embeddings in the bank no
             longer resemble what this model says when unconstrained.
          2. More than 70% of entries are below epsilon: the bank was built
             for a much weaker model — very few entries are still informative.
        """
        reasons = []
        if fmas is not None and fmas < self.fmas_threshold:
            reasons.append(
                f"FMAS={fmas:.4f} < threshold={self.fmas_threshold}: "
                "bank wrong-answers may not resemble this model's actual errors"
            )
        if total > 0 and below_epsilon_count / total > 0.70:
            frac = below_epsilon_count / total
            reasons.append(
                f"{frac*100:.0f}% of entries have U < ε: "
                "bank may have been built for a significantly weaker model"
            )
        return {
            "any": len(reasons) > 0,
            "reason": "; ".join(reasons) if reasons else "none",
            "fmas_low": fmas is not None and fmas < self.fmas_threshold,
            "high_low_utility_fraction": total > 0 and below_epsilon_count / total > 0.70,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3: Bank Lifecycle Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: tag utilities for gpt-5, keep all entries
  python pipeline/bank_lifecycle_manager.py \\
      --bank-dir ../outputs/seacr_bank_XYZ \\
      --eval-summary .../evaluations/evaluation_summary.json \\
      --model-name gpt-5 --generation 2

  # Periodic archival: remove low-utility entries
  python pipeline/bank_lifecycle_manager.py \\
      --bank-dir ../outputs/seacr_bank_XYZ \\
      --eval-summary .../evaluations/evaluation_summary.json \\
      --model-name gpt-5 --generation 2 --archive

  # Show cross-generation decay curve
  python pipeline/bank_lifecycle_manager.py \\
      --bank-dir ../outputs/seacr_bank_XYZ --decay-curve-only
""",
    )
    parser.add_argument('--bank-dir',       required=True,
                        help='Path to seacr_bank_* directory')
    parser.add_argument('--eval-summary',   default=None,
                        help='Path to evaluation_summary.json from Stage 2')
    parser.add_argument('--model-name',     default=None,
                        help='Model name (e.g. gpt-5)')
    parser.add_argument('--generation',     default=None, type=int,
                        help='Generation number (1, 2, 3, ...)')
    parser.add_argument('--epsilon',        default=0.05, type=float,
                        help='Low-utility threshold (default: 0.05)')
    parser.add_argument('--fmas-threshold', default=0.15, type=float,
                        help='FMAS below this emits a coverage warning (default: 0.15)')

    # Archival toggle — mutually exclusive, default is no-archive
    archive_group = parser.add_mutually_exclusive_group()
    archive_group.add_argument(
        '--archive', dest='archive', action='store_true',
        help='Physically remove entries where U < epsilon from bank.jsonl '
             '(periodic pruning). Writes removed entries to archived/ subdir.'
    )
    archive_group.add_argument(
        '--no-archive', dest='archive', action='store_false',
        help='(Default) Tag utilities in-place, keep all entries in bank.jsonl.'
    )
    parser.set_defaults(archive=False)

    parser.add_argument(
        '--decay-curve-only', action='store_true',
        help='Print the cross-generation utility decay curve from metadata and exit.'
    )
    parser.add_argument(
        '--print-coverage-warning', action='store_true',
        help='Print WARN or OK to stdout after update (for shell script branching).'
    )
    args = parser.parse_args()

    manager = BankLifecycleManager(args.bank_dir, args.epsilon, args.fmas_threshold)

    if args.decay_curve_only:
        manager.print_decay_curve()
        sys.exit(0)

    if not args.eval_summary or args.model_name is None or args.generation is None:
        print("❌ --eval-summary, --model-name, and --generation are required "
              "unless using --decay-curve-only", file=sys.stderr)
        sys.exit(1)

    report = manager.update_generation(
        eval_summary_path=args.eval_summary,
        model_name=args.model_name,
        generation=args.generation,
        archive=args.archive,
    )

    if args.print_coverage_warning:
        print("WARN" if report["coverage_warning"]["any"] else "OK")


if __name__ == "__main__":
    main()
