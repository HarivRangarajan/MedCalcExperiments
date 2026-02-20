#!/usr/bin/env python3
"""
Stage 1: Submodular Bank Construction
======================================

Builds a principled few-shot demonstration bank by greedily selecting
from the candidate pool (existing bank entries + probe-generated failures)
to maximise coverage across failure modes, calculators, and error sharpness.

Submodular objective:
    Utility(S) = alpha * FailureModeCoverage(S)
               + beta  * CalculatorCoverage(S)
               + gamma * ErrorProximitySharpness(S)
               - delta * Redundancy(S)

Outputs (to outputs/seacr_bank_{timestamp}/):
    bank.jsonl          enriched entries (failure_mode, contrastive_sharpness, generation)
    embeddings.npz      forward index: question embeddings (N, 1536)
    inverted_index.npz  inverted index: wrong-answer embeddings (N, 1536)
    bank_metadata.json  coverage stats, fmas_baseline, failure_mode_distribution

Bootstrapping adaptation:
    When the existing bank has 0 incorrect entries (e.g., a new model that
    is very accurate on training data), generate_probe_failures() probes the
    model on --probe-size training examples without demonstrations and
    collects wrong predictions as the candidate pool. This is spec-faithful:
    §2.1.2 explicitly includes "new examples generated via probe inference
    on the wider 10k pool" in the candidate set.

Usage:
    python pipeline/submodular_bank_construction.py \\
        --existing-bank-dir ../outputs/medcalc_contrastive_edits_evaluation_* \\
        --train-csv MedCalc-Bench/dataset/train_data.csv \\
        --target-size 170 \\
        --probe-size 60 \\
        --labeling-model gpt-4o
"""

import argparse
import asyncio
import json
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from openai import AsyncOpenAI, OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from shared_utils import embed_texts_batch, evaluate_answer, load_medcalc_one_shot_examples

FAILURE_MODES = [
    "input_extraction",
    "formula_selection",
    "arithmetic",
    "unit_conversion",
    "threshold_boundary",
]

LABELING_PROMPT_SYSTEM = (
    "You are an expert in medical calculation errors. "
    "Classify LLM mistakes into one of five categories."
)

LABELING_PROMPT_USER = """\
Question: {Question}
Model's answer: {LLM Answer}
Correct answer: {Ground Truth Answer}
Model's reasoning: {LLM Explanation}

Classify the PRIMARY reason the model was wrong into exactly one of:
- input_extraction: wrong value extracted from patient note
- formula_selection: wrong formula chosen for this calculator
- arithmetic: correct formula and values, but arithmetic error
- unit_conversion: wrong unit or unit conversion applied
- threshold_boundary: wrong cutoff or boundary condition applied

Output only the category label, nothing else."""


class SubmodularBankBuilder:
    """
    Builds a submodular demonstration bank for SEACR retrieval.

    Orchestrates:
      1. Loading existing bank entries
      2. Probe inference to generate failures when bank is empty
      3. LLM failure-mode labeling (async, batched)
      4. Contrastive sharpness computation
      5. Greedy submodular selection
      6. Index construction (forward + inverted)
      7. FMAS baseline measurement
    """

    def __init__(
        self,
        api_key: str,
        existing_bank_dir: str,
        train_csv_path: str,
        output_dir: str,
        target_size: int = 170,
        probe_size: int = 60,
        alpha: float = 0.35,
        beta: float = 0.35,
        gamma: float = 0.20,
        delta: float = 0.10,
        labeling_model: str = "gpt-4o",
        inference_model: str = "gpt-4o",
        fmas_probe_size: int = 20,
    ):
        self.api_key = api_key
        self.existing_bank_dir = Path(existing_bank_dir)
        self.train_csv_path = Path(train_csv_path)
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        self.probe_size = probe_size
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.labeling_model = labeling_model
        self.inference_model = inference_model
        self.fmas_probe_size = fmas_probe_size

        self.client = OpenAI(api_key=api_key)
        self.async_client = AsyncOpenAI(api_key=api_key)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "archived").mkdir(exist_ok=True)

        self._emb_cache: Dict[str, np.ndarray] = {}  # entry id → question embedding
        self._metadata: Dict = {}

        print(f"✅ SubmodularBankBuilder initialised")
        print(f"   • Bank dir:   {self.existing_bank_dir}")
        print(f"   • Train CSV:  {self.train_csv_path}")
        print(f"   • Output:     {self.output_dir}")
        print(f"   • Target K:   {self.target_size}")
        print(f"   • α={alpha} β={beta} γ={gamma} δ={delta}")

    # ------------------------------------------------------------------
    # Step 1: Load existing bank
    # ------------------------------------------------------------------

    def load_existing_bank(self) -> Tuple[List[Dict], List[Dict]]:
        """Load all correct and incorrect entries from existing_bank_dir.

        Reads from correct/*.jsonl and incorrect/*.jsonl subdirectories.
        Returns (correct_examples, incorrect_examples).
        """
        correct, incorrect = [], []

        correct_dir = self.existing_bank_dir / "correct"
        if correct_dir.exists():
            for f in sorted(correct_dir.glob("*.jsonl")):
                with open(f) as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            correct.append(json.loads(line))

        incorrect_dir = self.existing_bank_dir / "incorrect"
        if incorrect_dir.exists():
            for f in sorted(incorrect_dir.glob("*.jsonl")):
                with open(f) as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            incorrect.append(json.loads(line))

        print(f"   ✓ Loaded {len(correct)} correct + {len(incorrect)} incorrect from existing bank")
        return correct, incorrect

    # ------------------------------------------------------------------
    # Step 2 (optional): Probe inference to generate incorrect examples
    # ------------------------------------------------------------------

    async def _probe_one(self, row: pd.Series, one_shot_examples: Dict) -> Optional[Dict]:
        """Run zero-shot probe on one training row. Returns incorrect entry or None."""
        calc_id = str(row["Calculator ID"])
        question = row["Question"]
        patient_note = row["Patient Note"]
        ground_truth = str(row["Ground Truth Answer"])

        example = one_shot_examples.get(calc_id)
        if example is None:
            return None

        system_msg = (
            "You are a helpful assistant for calculating a score for a given patient note. "
            "Please think step-by-step to solve the question and then generate the required score. "
            'Your output should only contain a JSON dict formatted as '
            '{"step_by_step_thinking": str(your_step_by_step_thinking), '
            '"answer": str(short_and_direct_answer_of_the_question)}.'
        )
        system_msg += f'\nHere is an example patient note:\n\n{example["Patient Note"]}'
        system_msg += f'\n\nHere is an example task:\n\n{question}'
        system_msg += (
            f'\n\nPlease directly output the JSON dict:\n\n'
            f'{json.dumps({"step_by_step_thinking": example["Response"]["step_by_step_thinking"], "answer": example["Response"]["answer"]})}'
        )
        user_msg = (
            f"Here is the patient note:\n\n{patient_note}\n\n"
            f"Here is the task:\n\n{question}\n\n"
            "Please directly output the JSON dict with your step-by-step thinking and final answer."
        )

        try:
            resp = await self.async_client.chat.completions.create(
                model=self.inference_model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
            )
            raw = re.sub(r"\s+", " ", resp.choices[0].message.content)
            extracted = re.findall(r'[Aa]nswer":\s*(.*?)\}', raw)
            llm_answer = extracted[-1].strip().strip('"') if extracted else "N/A"
            explanation_m = re.findall(r'"step_by_step_thinking":\s*"([^"]+)"', raw)
            llm_explanation = explanation_m[-1] if explanation_m else ""
        except Exception as e:
            return None

        # Evaluate correctness
        try:
            upper = str(row.get("Upper Limit", ""))
            lower = str(row.get("Lower Limit", ""))
            is_correct = evaluate_answer(
                llm_answer, ground_truth, int(calc_id),
                upper, lower
            )
        except Exception:
            is_correct = (llm_answer.strip() == ground_truth.strip())

        if is_correct:
            return None  # Only keep incorrect predictions

        return {
            "Row Number": int(row.get("Row Number", row.name)),
            "Calculator Name": str(row["Calculator Name"]),
            "Calculator ID": calc_id,
            "Category": str(row.get("Category", "")),
            "Note ID": str(row.get("Note ID", "")),
            "Patient Note": patient_note,
            "Question": question,
            "LLM Answer": llm_answer,
            "LLM Explanation": llm_explanation,
            "Ground Truth Answer": ground_truth,
            "Ground Truth Explanation": str(row.get("Ground Truth Explanation", "")),
            "Result": "Incorrect",
            "Prompt Type": "probe_inference",
        }

    async def _generate_probe_failures_async(self, n: int) -> List[Dict]:
        """Probe model on n training examples, return incorrect predictions."""
        df = pd.read_csv(self.train_csv_path)
        sample = df.sample(min(n, len(df)), random_state=42).reset_index(drop=True)
        one_shot_examples = load_medcalc_one_shot_examples()

        print(f"   • Probing model on {len(sample)} training examples for failure bootstrap...")
        tasks = [self._probe_one(row, one_shot_examples) for _, row in sample.iterrows()]
        results = await asyncio.gather(*tasks)
        failures = [r for r in results if r is not None]
        print(f"   • Found {len(failures)} failures from {len(sample)} probes "
              f"({len(failures)/len(sample)*100:.1f}% error rate)")
        return failures

    def generate_probe_failures(self, n: int) -> List[Dict]:
        """Synchronous wrapper for probe failure generation."""
        return asyncio.run(self._generate_probe_failures_async(n))

    # ------------------------------------------------------------------
    # Step 3: Failure mode labeling
    # ------------------------------------------------------------------

    async def _label_one(self, entry: Dict) -> str:
        """Call LLM to classify failure mode for one incorrect entry."""
        prompt = LABELING_PROMPT_USER.format(**entry)
        try:
            resp = await self.async_client.chat.completions.create(
                model=self.labeling_model,
                messages=[
                    {"role": "system", "content": LABELING_PROMPT_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
            label = resp.choices[0].message.content.strip().lower()
            return label if label in FAILURE_MODES else "arithmetic"
        except Exception:
            return "arithmetic"

    async def _label_all_async(self, incorrect: List[Dict]) -> List[Dict]:
        """Label all incorrect entries concurrently."""
        print(f"   • Labeling failure modes for {len(incorrect)} incorrect entries...")
        labels = await asyncio.gather(*[self._label_one(e) for e in incorrect])
        for entry, label in zip(incorrect, labels):
            entry["failure_mode"] = label
        return incorrect

    def label_failure_modes(self, incorrect_examples: List[Dict]) -> List[Dict]:
        """Run async LLM labeling on all incorrect examples. Modifies in-place."""
        labeled = asyncio.run(self._label_all_async(incorrect_examples))
        dist = {}
        for e in labeled:
            fm = e.get("failure_mode", "unknown")
            dist[fm] = dist.get(fm, 0) + 1
        self._metadata["failure_mode_distribution"] = dist
        print(f"   • Failure mode distribution: {dist}")
        return labeled

    # ------------------------------------------------------------------
    # Step 4: Contrastive sharpness
    # ------------------------------------------------------------------

    def compute_contrastive_sharpness(
        self, wrong_emb: np.ndarray, correct_emb: np.ndarray
    ) -> float:
        """1.0 - cosine_sim(wrong_emb, correct_emb). Higher = farther apart = easy contrast."""
        w = wrong_emb / (np.linalg.norm(wrong_emb) + 1e-10)
        c = correct_emb / (np.linalg.norm(correct_emb) + 1e-10)
        return float(1.0 - float(w @ c))

    def embed_texts_batch(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """Batch embed texts using text-embedding-3-small. Returns (N, 1536) float32."""
        return embed_texts_batch(texts, self.client, batch_size)

    # ------------------------------------------------------------------
    # Step 5: Submodular objective and greedy selection
    # ------------------------------------------------------------------

    def _failure_mode_coverage(self, S: List[Dict]) -> float:
        """Weighted fraction of the 5 failure mode types represented in S."""
        freq = self._metadata.get("failure_mode_distribution", {})
        total = sum(freq.values()) or 1
        weights = {fm: (1.0 / (freq.get(fm, 1) / total)) for fm in FAILURE_MODES}
        w_sum = sum(weights.values())
        weights = {fm: w / w_sum for fm, w in weights.items()}

        min_quota = max(1, self.target_size // len(FAILURE_MODES))
        coverage = 0.0
        for fm in FAILURE_MODES:
            count = sum(1 for e in S if e.get("failure_mode") == fm)
            coverage += weights[fm] * min(1.0, count / min_quota)
        return coverage

    def _calculator_coverage(self, S: List[Dict]) -> float:
        """Fraction of calculator IDs with at least 1 incorrect entry in S."""
        all_ids = set(str(e["Calculator ID"]) for e in S)
        # normalise by 55 (total calculators in MedCalc-Bench)
        return min(1.0, len(all_ids) / 55)

    def _error_proximity_sharpness(self, S: List[Dict]) -> float:
        """1 - mean(contrastive_sharpness). Lower sharpness = near-miss = higher score."""
        inc = [e for e in S if e.get("Result") == "Incorrect"]
        if not inc:
            return 0.0
        return 1.0 - float(np.mean([e.get("contrastive_sharpness", 0.5) for e in inc]))

    def _redundancy(self, S: List[Dict]) -> float:
        """Mean pairwise cosine sim above 0.9 threshold (penalises near-duplicates)."""
        if len(S) < 2:
            return 0.0
        ids = [self._entry_id(e) for e in S]
        embs = np.array([self._emb_cache[i] for i in ids if i in self._emb_cache])
        if len(embs) < 2:
            return 0.0
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10
        normed = embs / norms
        sim_matrix = normed @ normed.T
        n = len(sim_matrix)
        penalties = []
        for i in range(n):
            for j in range(i + 1, n):
                penalties.append(max(0.0, sim_matrix[i, j] - 0.9))
        return float(np.mean(penalties)) if penalties else 0.0

    def _compute_utility(self, S: List[Dict]) -> float:
        """Full utility of set S."""
        return (
            self.alpha * self._failure_mode_coverage(S)
            + self.beta * self._calculator_coverage(S)
            + self.gamma * self._error_proximity_sharpness(S)
            - self.delta * self._redundancy(S)
        )

    @staticmethod
    def _entry_id(entry: Dict) -> str:
        return f"{entry.get('Row Number', id(entry))}_{entry.get('Prompt Type', '')}"

    def compute_utility(self, S: List[Dict], candidate: Dict) -> float:
        """Marginal gain of adding candidate to S."""
        return self._compute_utility(S + [candidate]) - self._compute_utility(S)

    def greedy_select(self, labeled_incorrect: List[Dict]) -> List[Dict]:
        """
        Greedy submodular selection of K=target_size entries from candidates.
        Pre-embeds all candidates to avoid repeated API calls.
        """
        candidates = labeled_incorrect
        if not candidates:
            print("   ⚠️  No candidates for greedy selection — bank will be empty.")
            return []

        print(f"   • Greedy selection: K={self.target_size} from {len(candidates)} candidates")

        S: List[Dict] = []
        candidate_set = list(candidates)

        for step in range(min(self.target_size, len(candidate_set))):
            best_gain, best_d = -float("inf"), None
            for d in candidate_set:
                if d in S:
                    continue
                gain = self.compute_utility(S, d)
                if gain > best_gain:
                    best_gain, best_d = gain, d
            if best_d is None:
                break
            S.append(best_d)
            if (step + 1) % 20 == 0:
                print(f"      step {step+1}/{self.target_size}  utility={self._compute_utility(S):.4f}")

        print(f"   ✓ Selected {len(S)} entries  final utility={self._compute_utility(S):.4f}")
        return S

    # ------------------------------------------------------------------
    # Step 6: Build indexes
    # ------------------------------------------------------------------

    def build_indexes(
        self, selected_incorrect: List[Dict], all_correct: List[Dict]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build and save forward (question) + inverted (wrong-answer) indexes.

        bank = selected_incorrect + all_correct
        forward_embs[i]  = embed(bank[i].Question)       for all entries
        inverted_embs[i] = embed(bank[i].LLM_Answer)     for incorrect entries; zeros for correct
        """
        bank = selected_incorrect + all_correct

        # Forward index: embed questions
        questions = [e["Question"] for e in bank]
        print(f"   • Embedding {len(questions)} questions for forward index...")
        forward_embs = self.embed_texts_batch(questions)

        # Inverted index: embed wrong answers for incorrect, zeros for correct
        inverted_embs = np.zeros_like(forward_embs)
        inc_idx = list(range(len(selected_incorrect)))  # first len(selected) entries are incorrect
        if inc_idx:
            wrong_answers = [bank[i]["LLM Answer"] for i in inc_idx]
            print(f"   • Embedding {len(wrong_answers)} wrong answers for inverted index...")
            wrong_embs = self.embed_texts_batch(wrong_answers)
            for local_i, global_i in enumerate(inc_idx):
                inverted_embs[global_i] = wrong_embs[local_i]

        # Compute and store sharpness for each incorrect entry
        for local_i, global_i in enumerate(inc_idx):
            entry = bank[global_i]
            correct_ans = entry.get("Ground Truth Answer", "")
            if correct_ans:
                correct_emb = self.embed_texts_batch([correct_ans])[0]
                sharpness = self.compute_contrastive_sharpness(
                    inverted_embs[global_i], correct_emb
                )
            else:
                sharpness = 0.5
            bank[global_i]["contrastive_sharpness"] = round(sharpness, 4)
            bank[global_i].setdefault("generation", 1)

        # Save bank.jsonl
        bank_file = self.output_dir / "bank.jsonl"
        with open(bank_file, "w") as f:
            for entry in bank:
                f.write(json.dumps(entry) + "\n")
        print(f"   ✓ Saved bank.jsonl ({len(bank)} entries)")

        # Save embeddings
        np.savez(self.output_dir / "embeddings.npz", embeddings=forward_embs)
        np.savez(self.output_dir / "inverted_index.npz", embeddings=inverted_embs)
        print(f"   ✓ Saved embeddings.npz and inverted_index.npz")

        # Cache question embeddings for redundancy computation
        for i, entry in enumerate(bank):
            self._emb_cache[self._entry_id(entry)] = forward_embs[i]

        self._metadata["bank_size"] = len(bank)
        self._metadata["num_incorrect"] = len(selected_incorrect)
        self._metadata["num_correct"] = len(all_correct)
        self._metadata["original_bank_size"] = len(bank)

        return forward_embs, inverted_embs

    # ------------------------------------------------------------------
    # Step 7: FMAS baseline
    # ------------------------------------------------------------------

    def compute_fmas_baseline(
        self, inverted_embs: np.ndarray, bank: List[Dict], sample_size: int = 20
    ) -> float:
        """
        Probe model on sample_size random training examples (no demonstrations).
        Compute mean max cosine sim to nearest wrong-answer in inverted_embs.
        Saves result to bank_metadata.json under 'fmas_baseline'.
        """
        incorrect_indices = [i for i, e in enumerate(bank) if e["Result"] == "Incorrect"]
        if not incorrect_indices:
            print("   ⚠️  No incorrect entries in bank — FMAS baseline = 0.0")
            self._metadata["fmas_baseline"] = 0.0
            return 0.0

        print(f"   • Computing FMAS baseline (probing {sample_size} training examples)...")
        probe_failures = self.generate_probe_failures(sample_size)
        if not probe_failures:
            print("   ⚠️  No probe failures for FMAS baseline — returning 0.0")
            self._metadata["fmas_baseline"] = 0.0
            return 0.0

        inc_embs = inverted_embs[incorrect_indices]
        probe_answers = [e["LLM Answer"] for e in probe_failures]
        probe_embs = self.embed_texts_batch(probe_answers)

        per_max = []
        for emb in probe_embs:
            q = emb / (np.linalg.norm(emb) + 1e-10)
            norms = np.linalg.norm(inc_embs, axis=1, keepdims=True) + 1e-10
            sims = (inc_embs / norms) @ q
            per_max.append(float(np.max(sims)))

        fmas = float(np.mean(per_max))
        self._metadata["fmas_baseline"] = round(fmas, 4)
        print(f"   ✓ FMAS baseline: {fmas:.4f}")
        return fmas

    # ------------------------------------------------------------------
    # Main orchestrator
    # ------------------------------------------------------------------

    def run(self) -> str:
        """Orchestrate all steps. Returns path to output_dir."""
        print("\n🏗  Stage 1: Submodular Bank Construction")
        print("=" * 60)

        # 1. Load existing bank
        print("\n[1/7] Loading existing bank...")
        all_correct, existing_incorrect = self.load_existing_bank()

        # 2. Bootstrap with probe failures if needed
        incorrect = list(existing_incorrect)
        if len(incorrect) < self.target_size // 2:
            print(f"\n[2/7] Bootstrapping: existing incorrect={len(incorrect)} "
                  f"< {self.target_size//2} — running probe inference...")
            probe_failures = self.generate_probe_failures(self.probe_size)
            incorrect.extend(probe_failures)
            print(f"   → Total incorrect candidates: {len(incorrect)}")
        else:
            print(f"\n[2/7] Sufficient incorrect entries ({len(incorrect)}) — skipping probe bootstrap")

        if not incorrect:
            print("   ⚠️  No incorrect candidates available. Bank will contain only correct entries.")

        # 3. Label failure modes
        print(f"\n[3/7] Labeling failure modes...")
        labeled_incorrect = self.label_failure_modes(incorrect)

        # 4. Build embedding cache for sharpness + redundancy (embed questions in bulk)
        print(f"\n[4/7] Pre-embedding candidate questions for redundancy cache...")
        if labeled_incorrect:
            q_texts = [e["Question"] for e in labeled_incorrect]
            q_embs = self.embed_texts_batch(q_texts)
            for e, emb in zip(labeled_incorrect, q_embs):
                self._emb_cache[self._entry_id(e)] = emb

        # 5. Greedy submodular selection
        print(f"\n[5/7] Greedy submodular selection...")
        selected = self.greedy_select(labeled_incorrect)

        # 6. Build indexes (also writes bank.jsonl)
        print(f"\n[6/7] Building forward + inverted indexes...")
        forward_embs, inverted_embs = self.build_indexes(selected, all_correct)

        # Load bank for FMAS (includes both selected + correct)
        bank = selected + all_correct

        # 7. FMAS baseline
        print(f"\n[7/7] Computing FMAS baseline...")
        self.compute_fmas_baseline(inverted_embs, bank, self.fmas_probe_size)

        # Write metadata
        self._metadata.update({
            "timestamp": datetime.now().isoformat(),
            "existing_bank_dir": str(self.existing_bank_dir),
            "target_size": self.target_size,
            "probe_size": self.probe_size,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "delta": self.delta,
            "labeling_model": self.labeling_model,
            "inference_model": self.inference_model,
            "generation": 1,
            "coverage_stats": {
                "failure_modes_covered": len(set(
                    e.get("failure_mode") for e in selected if e.get("failure_mode")
                )),
                "calculators_covered": len(set(str(e["Calculator ID"]) for e in selected)),
            },
        })
        meta_file = self.output_dir / "bank_metadata.json"
        with open(meta_file, "w") as f:
            json.dump(self._metadata, f, indent=2)
        print(f"\n✅ Bank construction complete!")
        print(f"   📁 Output: {self.output_dir}")
        print(f"   • Total entries: {len(bank)}")
        print(f"   • Incorrect: {len(selected)}  |  Correct: {len(all_correct)}")
        print(f"   • FMAS baseline: {self._metadata.get('fmas_baseline', 'N/A')}")
        print(f"   • Failure modes covered: {self._metadata['coverage_stats']['failure_modes_covered']}/5")
        print(f"   • Calculators covered: {self._metadata['coverage_stats']['calculators_covered']}")

        return str(self.output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: Submodular Bank Construction for SEACR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--existing-bank-dir", required=True,
        help="Path to medcalc_contrastive_edits_evaluation_* directory"
    )
    parser.add_argument(
        "--train-csv", required=True,
        help="Path to MedCalc-Bench/dataset/train_data.csv"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: ../outputs/seacr_bank_{timestamp})"
    )
    parser.add_argument("--target-size", type=int, default=170)
    parser.add_argument(
        "--probe-size", type=int, default=60,
        help="Number of training examples to probe when incorrect pool is empty (default: 60)"
    )
    parser.add_argument("--alpha", type=float, default=0.35, help="FailureModeCoverage weight")
    parser.add_argument("--beta",  type=float, default=0.35, help="CalculatorCoverage weight")
    parser.add_argument("--gamma", type=float, default=0.20, help="ErrorProximitySharpness weight")
    parser.add_argument("--delta", type=float, default=0.10, help="Redundancy penalty weight")
    parser.add_argument("--labeling-model", default="gpt-4o")
    parser.add_argument("--inference-model", default="gpt-4o",
                        help="Model to use for probe inference (default: gpt-4o)")
    parser.add_argument("--fmas-probe-size", type=int, default=20,
                        help="Training examples to probe for FMAS baseline (default: 20)")

    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        sys.exit(1)

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = str(
            Path(__file__).parent.parent / "outputs" / f"seacr_bank_{timestamp}"
        )

    builder = SubmodularBankBuilder(
        api_key=api_key,
        existing_bank_dir=args.existing_bank_dir,
        train_csv_path=args.train_csv,
        output_dir=args.output_dir,
        target_size=args.target_size,
        probe_size=args.probe_size,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        delta=args.delta,
        labeling_model=args.labeling_model,
        inference_model=args.inference_model,
        fmas_probe_size=args.fmas_probe_size,
    )

    output_path = builder.run()
    print(f"\n📁 Bank saved to: {output_path}")


if __name__ == "__main__":
    main()
