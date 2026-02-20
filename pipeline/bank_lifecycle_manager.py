#!/usr/bin/env python3
"""
Stage 3: Bank Lifecycle Manager
================================

Manages the demonstration bank lifecycle across model generations.

After each model generation is evaluated (Stage 2), this manager:
  1. Loads the evaluation_summary.json (which includes per-calculator accuracy)
  2. Computes U(d, g) for every active bank entry
  3. Archives entries where U(d, g) < epsilon
  4. Rewrites bank.jsonl with only the active (high-utility) entries
  5. Recommends a Stage 1 rebuild if FMAS has decayed below threshold
     or if fewer than 30% of the original entries remain active

Utility formula:
    U(d, g) = difficulty(d) × (1 - accuracy(g, calc(d)))

    difficulty(d) = contrastive_sharpness  (high = easy contrast; low = near-miss)
    accuracy(g, calc(d)) = per-calculator accuracy from evaluation_summary.json

    Interpretation:
      - If the model now answers a calculator correctly (accuracy → 1), the
        incorrect examples for that calculator lose utility (U → 0) because
        they no longer teach the model anything new.
      - Near-miss examples (low sharpness) are penalised more slowly — they
        remain useful even as average accuracy improves.

Rebuild triggers:
  1. FMAS < fmas_threshold: bank wrong-answers no longer resemble model errors
  2. Active bank < 30% of original size: too few entries for effective retrieval

Usage:
    python pipeline/bank_lifecycle_manager.py \\
        --bank-dir ../outputs/seacr_bank_XYZ \\
        --eval-summary ../outputs/contrastive_evaluation_XYZ/evaluations/evaluation_summary.json \\
        --model-name gpt-5 \\
        --generation 1 \\
        --print-rebuild-flag
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


class BankLifecycleManager:
    """
    Manages demonstration bank lifecycle across model generations.

    Computes U(d, g) = difficulty(d) × (1 - accuracy(g, calc(d), failure_mode(d)))
    Archives entries where U < epsilon. Recommends Stage 1 rebuild when FMAS decays.
    """

    def __init__(self, bank_dir: str, epsilon: float = 0.05,
                 fmas_threshold: float = 0.15):
        """
        Args:
            bank_dir:        Path to seacr_bank_* directory.
            epsilon:         Archive threshold. U(d,g) < epsilon → archive. Default 0.05.
            fmas_threshold:  FMAS below this triggers rebuild recommendation. Default 0.15.
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

        # Store original size on first run
        if "original_bank_size" not in self.metadata:
            self.metadata["original_bank_size"] = len(self.bank)

        print(f"✅ BankLifecycleManager loaded: {len(self.bank)} entries "
              f"(original_size={self.metadata['original_bank_size']})")

    def _get_calculator_accuracy(
        self, calculator_name: str, eval_summary: Dict,
        prompt_type: str = "contrastive_few_shot"
    ) -> float:
        """
        Look up per-calculator accuracy from evaluation_summary.json.
        eval_summary["contrastive_few_shot"]["by_calculator"][calc_name]["accuracy"]
        Falls back to overall accuracy if calculator not found.
        """
        by_calc = eval_summary.get(prompt_type, {}).get("by_calculator", {})
        if calculator_name in by_calc:
            return float(by_calc[calculator_name].get("accuracy", 0.0))
        return float(eval_summary.get(prompt_type, {}).get("overall_accuracy", 0.0))

    def compute_utility(self, entry: Dict, eval_summary: Dict) -> float:
        """
        U(d, g) = difficulty(d) × (1 - accuracy(g, calc(d), failure_mode(d)))

        difficulty(d) = contrastive_sharpness stored in the bank entry.
                        If missing (correct examples or not yet computed), use 0.5.
        accuracy(...)  = per-calculator accuracy from evaluation_summary for the
                         current model generation.

        Returns float in [0, 1].
        """
        difficulty = float(entry.get("contrastive_sharpness", 0.5))
        calc_name  = entry.get("Calculator Name", "")
        accuracy   = self._get_calculator_accuracy(calc_name, eval_summary)
        return difficulty * (1.0 - accuracy)

    def update_generation(
        self,
        eval_summary_path: str,
        model_name: str,
        generation: int
    ) -> Dict:
        """
        Main entry point for Stage 3.

        Steps:
        1. Load evaluation_summary.json from Stage 2 eval run
        2. Compute U(d, g) for each active bank entry
        3. Entries with U < epsilon → archived/archived_gen{N}_{model}.jsonl
        4. Rewrite bank.jsonl with remaining active entries
        5. Update bank_metadata.json with generation history
        6. Return lifecycle report dict

        Args:
            eval_summary_path: Full path to evaluation_summary.json
            model_name:        e.g. "gpt-5" — the model just evaluated
            generation:        Integer generation number (1, 2, 3, ...)

        Returns:
            Dict: archived_count, remaining_count, mean_utility, fmas, rebuild_recommended
        """
        with open(eval_summary_path) as f:
            eval_summary = json.load(f)

        fmas = eval_summary.get("fmas")   # written by Stage 2 Change D

        active, archived = [], []
        utilities = []

        for entry in self.bank:
            u = self.compute_utility(entry, eval_summary)
            utilities.append(u)
            if u < self.epsilon:
                entry["archived_at_generation"] = generation
                entry["archived_for_model"]     = model_name
                entry["final_utility"]          = round(u, 4)
                archived.append(entry)
            else:
                active.append(entry)

        # Write archived entries
        archived_file = self.archived_dir / f"archived_gen{generation}_{model_name}.jsonl"
        with open(archived_file, 'w') as f:
            for entry in archived:
                f.write(json.dumps(entry) + "\n")

        # Rewrite active bank
        with open(self.bank_dir / "bank.jsonl", 'w') as f:
            for entry in active:
                f.write(json.dumps(entry) + "\n")
        self.bank = active

        # Update metadata
        self.metadata["last_generation"]  = generation
        self.metadata["last_model"]       = model_name
        self.metadata["active_count"]     = len(active)
        self.metadata.setdefault("archived_counts", {})[f"gen{generation}_{model_name}"] = len(archived)
        if fmas is not None:
            self.metadata.setdefault("fmas_history", {})[f"gen{generation}_{model_name}"] = round(fmas, 4)
        with open(self.bank_dir / "bank_metadata.json", 'w') as f:
            json.dump(self.metadata, f, indent=2)

        rebuild = self.should_rebuild(fmas)

        report = {
            "generation":           generation,
            "model_name":           model_name,
            "total_before":         len(active) + len(archived),
            "archived_count":       len(archived),
            "remaining_count":      len(active),
            "mean_utility":         round(float(np.mean(utilities)), 4) if utilities else 0.0,
            "fmas":                 round(fmas, 4) if fmas is not None else None,
            "rebuild_recommended":  rebuild
        }

        print(f"\n🗂  Lifecycle update (gen {generation}, {model_name}):")
        print(f"   • Archived:  {len(archived)} entries (U < {self.epsilon})")
        print(f"   • Active:    {len(active)} entries remaining")
        print(f"   • Mean U:    {report['mean_utility']:.4f}")
        if fmas is not None:
            print(f"   • FMAS:      {fmas:.4f}")
        print(f"   • Rebuild:   {'YES — run Stage 1 for next generation' if rebuild else 'No'}")

        return report

    def should_rebuild(self, fmas: Optional[float] = None) -> bool:
        """
        Returns True if the bank should be rebuilt from scratch (Stage 1 re-run).

        Two triggers:
          1. FMAS < fmas_threshold (bank wrong-answer space no longer covers model errors)
          2. Active bank < 30% of original size (too few entries remain after archival)
        """
        original_size = self.metadata.get("original_bank_size", len(self.bank))
        size_depleted = original_size > 0 and len(self.bank) < 0.30 * original_size
        fmas_decayed  = fmas is not None and fmas < self.fmas_threshold
        return size_depleted or fmas_decayed


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3: Bank Lifecycle Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--bank-dir',       required=True,
                        help='Path to seacr_bank_* directory')
    parser.add_argument('--eval-summary',   required=True,
                        help='Path to evaluation_summary.json from Stage 2')
    parser.add_argument('--model-name',     required=True,
                        help='Model name (e.g. gpt-5)')
    parser.add_argument('--generation',     required=True, type=int,
                        help='Generation number (1, 2, 3, ...)')
    parser.add_argument('--epsilon',        default=0.05, type=float,
                        help='Archive threshold (default: 0.05)')
    parser.add_argument('--fmas-threshold', default=0.15, type=float,
                        help='FMAS below this triggers rebuild (default: 0.15)')
    parser.add_argument('--print-rebuild-flag', action='store_true',
                        help='Print REBUILD or OK to stdout (for shell script branching)')
    args = parser.parse_args()

    manager = BankLifecycleManager(args.bank_dir, args.epsilon, args.fmas_threshold)
    report  = manager.update_generation(args.eval_summary, args.model_name, args.generation)

    if args.print_rebuild_flag:
        print("REBUILD" if report["rebuild_recommended"] else "OK")


if __name__ == "__main__":
    main()
