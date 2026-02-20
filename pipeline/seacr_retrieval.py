#!/usr/bin/env python3
"""
SEACR: Self-Error-Anchored Contrastive Retrieval
=================================================

Stage 2 of the Lifecycle-Aware Contrastive Few-Shot pipeline.

Replaces the Calculator ID → random.sample() retrieval in
ContrastiveFewShotEvaluator.get_contrastive_examples() with
error-anchored retrieval using a precomputed inverted index
of wrong answers.

Negative retrieval (error-anchored):
    score(d_i) = alpha * sim(embed(probe_prediction), inverted_index[i])
               + (1-alpha) * sim(embed(question), forward_index[i])

Positive retrieval (smart):
    score(d_i) = beta_pos * sim(embed(question), forward_embs[i])
               + (1-beta_pos) * calculator_match(calculator_id, d_i)

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

    def __init__(self, bank_dir: str, alpha: float = 0.8, beta_pos: float = 0.6):
        """
        Args:
            bank_dir:  Path to seacr_bank_* directory from Stage 1.
            alpha:     Weight on the error-anchoring term for negative retrieval.
                       (1-alpha) goes to question-matching. 0.8 recommended.
            beta_pos:  Weight on question-similarity for positive retrieval.
                       (1-beta_pos) goes to calculator-match bonus. 0.6 recommended.
        """
        self.bank_dir = Path(bank_dir)
        self.alpha = alpha
        self.beta_pos = beta_pos

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

        # Build indices using polarity field (with backward compat for Result field)
        self.positive_indices: List[int] = []
        self.incorrect_indices: List[int] = []
        self.correct_index: Dict[str, List[int]] = {}  # calculator_id -> positive indices

        for i, entry in enumerate(self.bank):
            polarity = entry.get("polarity")
            result = entry.get("Result", "")

            is_positive = (polarity == "positive") or (polarity is None and result == "Correct")
            is_negative = (polarity == "negative") or (polarity is None and result == "Incorrect")

            if is_positive:
                self.positive_indices.append(i)
                cid = str(entry["Calculator ID"])
                self.correct_index.setdefault(cid, []).append(i)
            elif is_negative:
                self.incorrect_indices.append(i)

        print(f"✅ SEACRRetriever loaded: {len(self.bank)} total entries, "
              f"{len(self.incorrect_indices)} negative, "
              f"{len(self.positive_indices)} positive, "
              f"alpha={alpha}, beta_pos={beta_pos}")

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

        Negative (incorrect) retrieval — error-anchored:
            score(d_i) = alpha * sim(embed(probe_prediction), inverted_index[i])
                       + (1-alpha) * sim(embed(question), forward_index[i])
            Search is restricted to self.incorrect_indices only.

        Positive (correct) retrieval — smart:
            score(d_i) = beta_pos * sim(embed(question), forward_embs[i])
                       + (1-beta_pos) * calculator_match(calculator_id, d_i)
            Search is restricted to self.positive_indices only.

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
        # Embed question once (used by both negative and positive retrieval)
        question_emb = self._embed_text(question, client)

        # --- Negative retrieval via inverted index ---
        if self.incorrect_indices:
            probe_emb = self._embed_text(probe_prediction, client)

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

        # --- Smart positive retrieval ---
        if self.positive_indices:
            pos_fwd = self.forward_embs[self.positive_indices]  # (P, 1536)
            q_scores = self._cosine_sim_many(question_emb, pos_fwd)  # (P,)

            # Calculator-match bonus: 1.0 if same calculator, 0.0 otherwise
            calc_bonus = np.array([
                1.0 if str(self.bank[i].get("Calculator ID")) == str(calculator_id) else 0.0
                for i in self.positive_indices
            ], dtype=np.float32)

            pos_composite = self.beta_pos * q_scores + (1 - self.beta_pos) * calc_bonus
            top_pos_local = np.argsort(-pos_composite)[:num_positive]
            positive_examples = [
                self.bank[self.positive_indices[i]] for i in top_pos_local
            ]
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
