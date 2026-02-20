#!/usr/bin/env python3
"""
SEACR: Self-Error-Anchored Contrastive Retrieval
=================================================

Stage 2 of the Lifecycle-Aware Contrastive Few-Shot pipeline.

Replaces the Calculator ID → random.sample() retrieval in
ContrastiveFewShotEvaluator.get_contrastive_examples() with
error-anchored retrieval using a precomputed inverted index
of wrong answers.

Key idea:
    The model is first probed without demonstrations (probe inference).
    The probe prediction is embedded and matched against an inverted
    index of known wrong answers from the bank. The bank entry whose
    wrong answer most closely resembles the probe prediction is selected
    as the negative demonstration — because that is the mistake this
    specific model is most likely to make on this type of question.

Retrieval formula:
    score(d_i) = alpha * sim(embed(probe_prediction), inverted_index[i])
               + (1-alpha) * sim(embed(question), forward_index[i])

FMAS (Failure Mode Alignment Score):
    FMAS = E[max_{d in bank} sim(embed(probe), embed(d.LLM_Answer))]
    High FMAS → bank wrong-answers match model errors → SEACR is precise
    Low FMAS → bank is stale → trigger Stage 1 rebuild
"""

import numpy as np
import json
from pathlib import Path
from openai import OpenAI
from typing import Dict, List, Tuple, Optional


class SEACRRetriever:
    """
    Self-Error-Anchored Contrastive Retrieval.

    Replaces the Calculator ID → random.sample() logic in
    ContrastiveFewShotEvaluator.get_contrastive_examples() at
    pipeline/evaluate_contrastive_fewshot_method.py:169-205.
    """

    def __init__(self, bank_dir: str, alpha: float = 0.8):
        """
        Args:
            bank_dir: Path to seacr_bank_* directory from Stage 1.
            alpha: Weight on the error-anchoring term. (1-alpha) goes to question-matching.
                   0.8 recommended. 1.0 = pure error-anchoring, 0.0 = pure question-matching.
        """
        self.bank_dir = Path(bank_dir)
        self.alpha = alpha

        # Load all bank entries
        self.bank: List[Dict] = []
        with open(self.bank_dir / "bank.jsonl") as f:
            for line in f:
                self.bank.append(json.loads(line))

        # Load precomputed embeddings
        fwd = np.load(self.bank_dir / "embeddings.npz")
        inv = np.load(self.bank_dir / "inverted_index.npz")
        self.forward_embs: np.ndarray = fwd["embeddings"]    # shape (N, 1536)
        self.inverted_embs: np.ndarray = inv["embeddings"]   # shape (N, 1536)

        # Build lookup: calculator_id -> list of bank indices (correct entries only)
        self.correct_index: Dict[str, List[int]] = {}
        for i, entry in enumerate(self.bank):
            if entry["Result"] == "Correct":
                cid = str(entry["Calculator ID"])
                self.correct_index.setdefault(cid, []).append(i)

        # Indices into self.bank for incorrect entries (for inverted retrieval)
        self.incorrect_indices: List[int] = [
            i for i, e in enumerate(self.bank) if e["Result"] == "Incorrect"
        ]

        print(f"✅ SEACRRetriever loaded: {len(self.bank)} total entries, "
              f"{len(self.incorrect_indices)} incorrect, "
              f"{len(self.bank) - len(self.incorrect_indices)} correct, "
              f"alpha={alpha}")

    def _embed_text(self, text: str, client: OpenAI) -> np.ndarray:
        """Embed one string. Returns shape (1536,). Truncates at 8000 chars."""
        resp = client.embeddings.create(model="text-embedding-3-small", input=text[:8000])
        return np.array(resp.data[0].embedding, dtype=np.float32)

    def _cosine_sim_many(self, query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Cosine similarity between query (1536,) and every row of matrix (M, 1536).
        Returns shape (M,)."""
        q = query / (np.linalg.norm(query) + 1e-10)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
        return (matrix / norms) @ q

    def retrieve(
        self,
        question: str,
        probe_prediction: str,
        calculator_id: str,
        client: OpenAI,
        num_positive: int = 1,
        num_negative: int = 1,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Retrieve contrastive pair using SEACR.

        Negative (incorrect) retrieval — the novel part:
            score(d_i) = alpha * sim(embed(probe_prediction), inverted_index[i])
                       + (1-alpha) * sim(embed(question), forward_index[i])
            Search is restricted to self.incorrect_indices only.

        Positive (correct) retrieval — same as baseline:
            Filter by calculator_id, then random.choice.
            The novel contribution is entirely in the negative (error-anchored) retrieval.

        Args:
            question:          current test question text (from test_data.csv)
            probe_prediction:  model's output from the probe inference step (no demonstrations)
            calculator_id:     string calculator ID for this test example (e.g. "38")
            client:            synchronous OpenAI client for embedding calls
            num_positive:      number of correct demonstrations to return
            num_negative:      number of incorrect demonstrations to return

        Returns:
            Tuple (positive_examples, negative_examples), each a list of bank entry dicts
        """
        # --- Negative retrieval via inverted index ---
        if self.incorrect_indices:
            probe_emb    = self._embed_text(probe_prediction, client)
            question_emb = self._embed_text(question, client)

            inc_inv = self.inverted_embs[self.incorrect_indices]   # (M, 1536)
            inc_fwd = self.forward_embs[self.incorrect_indices]    # (M, 1536)

            error_scores    = self._cosine_sim_many(probe_emb, inc_inv)    # (M,)
            question_scores = self._cosine_sim_many(question_emb, inc_fwd) # (M,)
            composite = self.alpha * error_scores + (1 - self.alpha) * question_scores

            top_local_indices = np.argsort(-composite)[:num_negative]
            negative_examples = [
                self.bank[self.incorrect_indices[i]] for i in top_local_indices
            ]
        else:
            negative_examples = []

        # --- Positive retrieval via Calculator ID (unchanged from baseline) ---
        available_pos = self.correct_index.get(str(calculator_id), [])
        if available_pos:
            chosen = np.random.choice(
                available_pos,
                size=min(num_positive, len(available_pos)),
                replace=False
            ).tolist()
            positive_examples = [self.bank[i] for i in chosen]
        else:
            positive_examples = []

        return positive_examples, negative_examples

    def compute_fmas(self, probe_predictions: List[str], client: OpenAI) -> float:
        """
        Compute Failure Mode Alignment Score over a set of probe predictions.

        FMAS = E_{probe in probe_predictions} [
                   max_{d in incorrect bank} sim(embed(probe), embed(d.LLM_Answer))
               ]

        Interpretation:
            FMAS ≈ 1 → model errors perfectly match bank entries → SEACR retrieval is precise
            FMAS ≈ 0 → model errors have no match in bank → bank is stale for this model

        Args:
            probe_predictions: list of model probe outputs (no demonstrations), one per test example
            client: OpenAI client for embedding calls

        Returns:
            FMAS scalar in [0, 1]
        """
        if not probe_predictions or not self.incorrect_indices:
            return 0.0

        inc_embs = self.inverted_embs[self.incorrect_indices]   # (M, 1536)
        per_example_max_sims = []

        for pred in probe_predictions:
            emb = self._embed_text(pred, client)
            sims = self._cosine_sim_many(emb, inc_embs)
            per_example_max_sims.append(float(np.max(sims)))

        return float(np.mean(per_example_max_sims))
