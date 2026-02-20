# Lifecycle-Aware Contrastive Few-Shot Prompting: Implementation Specification

**Paper Title**: *Lifecycle-Aware Contrastive Few-Shot Prompting*
**Status**: Implementation-ready specification

---

## 0. Current Pipeline Map (Baseline)

Before touching anything, here is the exact execution graph of the existing system:

```
[10k train CSV]
      |
      v
pipeline/contrastive_demonstration_generation.py
      |  --sample-size 170
      |  Runs gpt-4o on 170 examples × 3 prompt variants (original, CoT, CoD)
      |
      v
outputs/medcalc_contrastive_edits_evaluation_*/
  correct/   {original,chain_of_thought,chain_of_draft}_correct.jsonl
  incorrect/ {original,chain_of_thought,chain_of_draft}_incorrect.jsonl
  prompts/   enhanced_prompts.json
  data/      training_sample_indices.json
      |
      v
pipeline/prompt_refinement_pipeline.py
      |  Refiner: gpt-5 rewrites unified_prompt.txt over 34 iterations
      |  Evaluator (hardcoded line 197): gpt-4o re-evaluates on 170 training examples
      |
      v
outputs/refined_prompts_*/final/unified_prompt.txt
      |
      v
pipeline/evaluate_contrastive_fewshot_method.py
      |  Retrieval: Calculator ID → random.sample() from correct/incorrect pools
      |  Evaluation: test_data.csv (1047 examples)
      |
      v
outputs/contrastive_evaluation_*/
  responses/contrastive_few_shot_responses.jsonl
  evaluations/evaluation_summary.json
```

**Current retrieval bottleneck** — the exact lines being replaced:
`pipeline/evaluate_contrastive_fewshot_method.py:193-204` inside `get_contrastive_examples()`:
```python
# CURRENT (Calculator ID lookup → random.sample)
if len(available_positive) >= num_positive:
    positive = random.sample(available_positive, num_positive)
if len(available_negative) >= num_negative:
    negative = random.sample(available_negative, num_negative)
```

---

## 1. Three-Stage Architecture

```
Stage 1: Bank Construction (runs once, offline)
  pipeline/submodular_bank_construction.py
  → outputs/seacr_bank_{timestamp}/
      bank.jsonl           (enriched examples with failure_mode + sharpness fields)
      embeddings.npz       (forward index: question embeddings, shape N×1536)
      inverted_index.npz   (inverted index: wrong-answer embeddings, shape N×1536)
      bank_metadata.json   (coverage stats, FMAS baseline, generation history)

Stage 2: SEACR Retrieval (replaces get_contrastive_examples at inference time)
  pipeline/seacr_retrieval.py                     (new standalone module)
  pipeline/evaluate_contrastive_fewshot_method.py (modified: probe call + SEACR + FMAS)
  → same output dir as before + evaluations/fmas_report.json

Stage 3: Lifecycle Management (runs after each model generation evaluation)
  pipeline/bank_lifecycle_manager.py
  → outputs/seacr_bank_{timestamp}/archived/   (retired examples per generation)
  → triggers re-run of Stage 1 when FMAS < threshold
```

---

## 2. Stage 1 — Submodular Bank Construction

### 2.1 New file: `pipeline/submodular_bank_construction.py`

**Purpose**: Replace the current random-170 sample with a principled greedy selection from
the full 10k train set that maximizes coverage across failure modes, calculators, and
error-proximity sharpness.

**Inputs**:
- `MedCalc-Bench/dataset/train_data.csv` (10,053 rows)
- `outputs/medcalc_contrastive_edits_evaluation_*/` (existing 510 bank entries, used as warm-start candidates)
- OpenAI API (for embeddings + probe inference)

**Outputs** written to `outputs/seacr_bank_{timestamp}/`:
- `bank.jsonl` — enriched bank entries (see schema §2.1.1)
- `embeddings.npz` — shape `(N, 1536)`, forward index keyed by question text
- `inverted_index.npz` — shape `(N, 1536)`, inverted index keyed by wrong-answer text
- `bank_metadata.json` — coverage statistics, FMAS baseline, failure mode distribution

#### 2.1.1 Enriched bank entry schema

The existing incorrect JSONL entries have these fields (confirmed from actual data):
```
Row Number, Calculator Name, Calculator ID, Category, Note ID,
Patient Note, Question, LLM Answer, LLM Explanation,
Ground Truth Answer, Ground Truth Explanation, Result, Prompt Type
```

The enriched schema adds three new fields:
```json
{
  "...all existing fields unchanged...",
  "failure_mode": "arithmetic | unit_conversion | input_extraction | formula_selection | threshold_boundary",
  "contrastive_sharpness": 0.73,
  "generation": 1
}
```

- `failure_mode`: assigned by a one-time LLM labeling pass (see §2.1.3). Null for correct examples.
- `contrastive_sharpness`: `1.0 - cosine_sim(embed(LLM_Answer), embed(Ground_Truth_Answer))`.
  High value = wrong and correct answer are far apart in embedding space (easy contrast).
  Low value = near-miss (harder to distinguish, more informative). Applies to incorrect entries only.
- `generation`: integer starting at 1, incremented by Stage 3 when bank is rebuilt after model generation advances.

#### 2.1.2 Submodular objective

Select set S ⊆ CandidatePool (all 510 existing incorrect examples from the bank, plus any
new examples generated via probe inference on the wider 10k pool) such that |S| = K (default K=170):

```
Utility(S) = α · FailureModeCoverage(S)
           + β · CalculatorCoverage(S)
           + γ · ErrorProximitySharpness(S)
           - δ · Redundancy(S)
```

**FailureModeCoverage(S)**: fraction of 5 failure mode types that are represented in S,
weighted by inverse frequency (rarer modes get higher weight so they aren't squeezed out):
```
FailureModeCoverage(S) = Σ_{fm ∈ {5 types}} w_fm · min(1, count(fm in S) / min_quota_fm)
```
where `min_quota_fm` = max(1, target_size / 5) and `w_fm = 1 / frequency_fm` (normalized).

**CalculatorCoverage(S)**: fraction of 55 calculator IDs with at least 1 incorrect example in S:
```
CalculatorCoverage(S) = |{calc_id : ∃ entry in S with that calc_id}| / 55
```

**ErrorProximitySharpness(S)**: this term is the key coupling with SEACR. Rather than just
measuring within-pair distance, it measures how likely the wrong answers in S are to match
future probe predictions from the target model. Approximated as mean contrastive_sharpness
across entries in S (lower sharpness = near-miss = more useful for error-anchored retrieval):
```
ErrorProximitySharpness(S) = 1 - mean_{d ∈ S, d.Result=="Incorrect"} [contrastive_sharpness(d)]
```
This is a maximization objective, so lower sharpness (harder contrasts) scores higher.

**Redundancy(S)**: penalizes near-duplicate questions that add no new coverage:
```
Redundancy(S) = mean_{i,j ∈ S, i≠j} max(0, cosine_sim(embed(q_i), embed(q_j)) - 0.9)
```
The 0.9 floor means only very high similarity pairs are penalized.

**Defaults**: `α=0.35, β=0.35, γ=0.20, δ=0.10`. All exposed as CLI args.

**Greedy algorithm**:
```python
S = []
candidates = all_incorrect_examples   # pre-embedded
while len(S) < target_size:
    best_d, best_gain = None, -inf
    for d in candidates:
        if d not in S:
            gain = Utility(S + [d]) - Utility(S)
            if gain > best_gain:
                best_gain, best_d = gain, d
    S.append(best_d)
```
For N=510 candidates and K=170, this runs O(N·K) = ~87k iterations, each doing constant-time
lookups with precomputed embeddings. Runs in seconds on CPU.

#### 2.1.3 Failure mode labeling pass

Run this once over the 145 existing incorrect examples before greedy selection. The prompt:

```
SYSTEM: "You are an expert in medical calculation errors. Classify LLM mistakes into one of five categories."

USER:
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

Output only the category label, nothing else.
```

Use `gpt-4o` with temperature=0. Make one API call per incorrect example (can batch using
asyncio). Save label distribution to `bank_metadata.json` under key `"failure_mode_distribution"`.

This distribution drives `w_fm` weights in FailureModeCoverage and is used by Stage 3
to look up per-failure-mode accuracy from evaluation summaries.

#### 2.1.4 Class structure

```python
class SubmodularBankBuilder:
    def __init__(
        self,
        api_key: str,
        existing_bank_dir: str,   # outputs/medcalc_contrastive_edits_evaluation_*/
        train_csv_path: str,      # MedCalc-Bench/dataset/train_data.csv
        output_dir: str,          # outputs/seacr_bank_{timestamp}/
        target_size: int = 170,
        alpha: float = 0.35,
        beta: float = 0.35,
        gamma: float = 0.20,
        delta: float = 0.10,
        labeling_model: str = "gpt-4o",
    ): ...

    def load_existing_bank(self) -> Tuple[List[Dict], List[Dict]]:
        """Load all correct and incorrect entries from existing_bank_dir.
        Returns (correct_examples, incorrect_examples)."""
        ...

    def label_failure_modes(self, incorrect_examples: List[Dict]) -> List[Dict]:
        """Run async LLM labeling on all incorrect examples.
        Adds 'failure_mode' key in-place. Saves distribution to bank_metadata.json.
        Returns labeled examples."""
        ...

    def compute_contrastive_sharpness(
        self, wrong_answer: str, correct_answer: str,
        wrong_emb: np.ndarray, correct_emb: np.ndarray
    ) -> float:
        """1.0 - cosine_sim(wrong_emb, correct_emb). Pre-embed all answers before calling."""
        ...

    def embed_texts_batch(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """Embed texts using text-embedding-3-small. Returns shape (len(texts), 1536).
        Delegates to shared_utils.embed_texts_batch()."""
        ...

    def compute_utility(self, S: List[Dict], candidate: Dict) -> float:
        """Compute marginal gain of adding candidate to S.
        Uses precomputed embeddings stored in self._emb_cache (dict: entry_id -> np.ndarray)."""
        ...

    def greedy_select(self, labeled_incorrect: List[Dict]) -> List[Dict]:
        """Main greedy submodular loop. Returns selected list of size target_size.
        Pre-embeds all candidates before the loop to avoid repeated API calls."""
        ...

    def build_indexes(
        self, selected: List[Dict], all_correct: List[Dict]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build and save forward + inverted indexes.
        Bank = selected (incorrect) + all_correct examples (for positive retrieval).
        - Forward index: embed(Question) for all entries
        - Inverted index: embed(LLM_Answer) for incorrect entries; zeros for correct entries
        Saves bank.jsonl, embeddings.npz, inverted_index.npz.
        Returns (forward_embs, inverted_embs)."""
        ...

    def compute_fmas_baseline(
        self, inverted_embs: np.ndarray, sample_size: int = 20
    ) -> float:
        """Probe model on sample_size random train examples (no demonstrations).
        Compute mean max cosine sim to nearest wrong-answer in inverted_embs.
        Saves to bank_metadata.json under 'fmas_baseline'."""
        ...

    def run(self) -> str:
        """Orchestrates all steps. Returns path to output_dir."""
        # 1. Load existing bank
        # 2. Label failure modes (async, batched)
        # 3. Compute contrastive_sharpness for each incorrect entry
        # 4. Embed all questions and answers in bulk
        # 5. Greedy submodular selection
        # 6. Build forward + inverted indexes
        # 7. Compute FMAS baseline
        # 8. Write bank_metadata.json
        ...
```

#### 2.1.5 CLI interface

```bash
python pipeline/submodular_bank_construction.py \
  --existing-bank-dir ../outputs/medcalc_contrastive_edits_evaluation_20251010_054434 \
  --train-csv MedCalc-Bench/dataset/train_data.csv \
  --target-size 170 \
  --alpha 0.35 --beta 0.35 --gamma 0.20 --delta 0.10 \
  --labeling-model gpt-4o \
  --output-dir ../outputs/seacr_bank_$(date +%Y%m%d_%H%M%S)
```

Stage 1 runs **once per model generation**, not per evaluation. The bank it produces is
reused across all test runs for that generation.

---

## 3. Stage 2 — SEACR Retrieval

### 3.1 New file: `pipeline/seacr_retrieval.py`

**Purpose**: Self-contained retrieval module. Loads the bank from Stage 1 and exposes a
`retrieve()` method that replaces `get_contrastive_examples()` in the evaluator.

```python
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
                cid = entry["Calculator ID"]
                self.correct_index.setdefault(cid, []).append(i)

        # Indices into self.bank for incorrect entries (for inverted retrieval)
        self.incorrect_indices: List[int] = [
            i for i, e in enumerate(self.bank) if e["Result"] == "Incorrect"
        ]

        print(f"✅ SEACRRetriever loaded: {len(self.bank)} total entries, "
              f"{len(self.incorrect_indices)} incorrect, alpha={alpha}")

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
            Rationale: correct examples should match the same calculator type.
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
        available_pos = self.correct_index.get(calculator_id, [])
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

        FMAS = E_{probe in probe_predictions} [ max_{d in incorrect bank} sim(embed(probe), embed(d.LLM_Answer)) ]

        Interpretation: how well does the bank's wrong-answer space cover the model's actual errors?
        FMAS ≈ 1 → model errors perfectly match bank entries → SEACR retrieval is precise.
        FMAS ≈ 0 → model errors have no match in bank → bank is stale for this model.

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
```

### 3.2 Changes to `pipeline/evaluate_contrastive_fewshot_method.py`

Four targeted changes. All other code is untouched.

#### Change A — `__init__()`: add SEACR attributes

**Location**: the end of `__init__()` body, after `self.contrastive_examples = self._load_contrastive_examples()` (line ~96).

Add:
```python
# SEACR retrieval (Stage 2 — overrides get_contrastive_examples if provided)
self._probe_predictions: List[str] = []   # accumulates during evaluation for FMAS

if seacr_bank_dir:
    from seacr_retrieval import SEACRRetriever
    self.seacr_retriever = SEACRRetriever(seacr_bank_dir, alpha=seacr_alpha)
else:
    self.seacr_retriever = None
    print("   ℹ️  SEACR disabled — using Calculator ID retrieval")
```

And add to `__init__()` signature:
```python
def __init__(self, ..., seacr_bank_dir: str = None, seacr_alpha: float = 0.8):
```

#### Change B — Add `_probe_inference_async()` method

Insert this new method directly before `_process_single_example_async()` (before line 326):

```python
async def _probe_inference_async(self, row: pd.Series) -> str:
    """
    Run the model with zero contrastive demonstrations to get its unconstrained prediction.
    Uses the same one-shot system prompt as the baseline (just no contrastive pairs).
    Returns extracted answer string (e.g. "7.75") or "N/A" on failure.
    """
    patient_note = row["Patient Note"]
    question     = row["Question"]
    calculator_id = str(row["Calculator ID"])

    try:
        system_msg, user_msg = self.create_original_one_shot_prompt(
            patient_note, question, calculator_id
        )
        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ]
        )
        raw = re.sub(r"\s+", " ", response.choices[0].message.content)
        probe_answer, _ = self.extract_answer(raw, int(calculator_id))
        return probe_answer if probe_answer != "Not Found" else "N/A"
    except Exception:
        return "N/A"
```

#### Change C — Modify `_process_single_example_async()` to call probe + SEACR

**Location**: inside the `elif prompt_type == "contrastive_few_shot":` branch (around line 342).

Replace:
```python
elif prompt_type == "contrastive_few_shot":
    system_msg, user_msg = self.create_contrastive_few_shot_prompt(
        patient_note, question, calculator_id, unified_prompt
    )
```

With:
```python
elif prompt_type == "contrastive_few_shot":
    if self.seacr_retriever is not None:
        # SEACR path: probe first, then error-anchored retrieval
        probe_prediction = await self._probe_inference_async(row)
        self._probe_predictions.append(probe_prediction)

        positive_examples, negative_examples = self.seacr_retriever.retrieve(
            question=question,
            probe_prediction=probe_prediction,
            calculator_id=calculator_id,
            client=self.client,        # synchronous client for embedding calls
            num_positive=self.num_positive,
            num_negative=self.num_negative
        )
        # Reuse existing prompt-builder but feed SEACR-retrieved examples directly
        system_msg = unified_prompt
        one_shot_example = self.one_shot_examples.get(calculator_id)
        if one_shot_example:
            system_msg += f'\n\n**Example (Correct Approach):**\n'
            system_msg += f'Patient Note: {one_shot_example["Patient Note"][:500]}...\n'
            system_msg += f'Task: {question}\n'
            system_msg += f'Response: {json.dumps({"step_by_step_thinking": one_shot_example["Response"]["step_by_step_thinking"], "answer": one_shot_example["Response"]["answer"]})}'
        if positive_examples:
            system_msg += f'\n\n**Additional Correct Examples:**\n'
            for i, ex in enumerate(positive_examples, 1):
                system_msg += f'\nExample {i} (CORRECT):\n'
                system_msg += f'Patient Note: {ex["Patient Note"][:300]}...\n'
                system_msg += f'Task: {ex["Question"][:200]}...\n'
                system_msg += f'LLM Answer: {ex["LLM Answer"]}\n'
                system_msg += f'Ground Truth: {ex["Ground Truth Answer"]}\n'
        if negative_examples:
            system_msg += f'\n\n**Examples to AVOID (Common Mistakes):**\n'
            for i, ex in enumerate(negative_examples, 1):
                system_msg += f'\nExample {i} (INCORRECT - Learn from this mistake):\n'
                system_msg += f'Patient Note: {ex["Patient Note"][:300]}...\n'
                system_msg += f'Task: {ex["Question"][:200]}...\n'
                system_msg += f'Incorrect LLM Answer: {ex["LLM Answer"]}\n'
                system_msg += f'Correct Answer Should Be: {ex["Ground Truth Answer"]}\n'
        user_msg = f'Here is the patient note:\n\n{patient_note}\n\nHere is the task:\n\n{question}\n\nPlease directly output the JSON dict with your step-by-step thinking and final answer.'
    else:
        # Fallback: existing Calculator ID → random.sample() path (unchanged)
        system_msg, user_msg = self.create_contrastive_few_shot_prompt(
            patient_note, question, calculator_id, unified_prompt
        )
```

#### Change D — Add FMAS computation in `run_complete_evaluation()`

**Location**: after `eval_file` is written (around line 645), before the final print statements.

Add:
```python
# Compute and save FMAS if SEACR was used
if self.seacr_retriever is not None and self._probe_predictions:
    print("\n📐 Computing FMAS (Failure Mode Alignment Score)...")
    fmas = self.seacr_retriever.compute_fmas(self._probe_predictions, self.client)
    print(f"   • FMAS: {fmas:.4f}  (1.0 = perfect alignment, 0.0 = stale bank)")

    fmas_report = {
        "fmas": fmas,
        "model": self.model,
        "num_probe_predictions": len(self._probe_predictions),
        "seacr_alpha": self.seacr_retriever.alpha,
        "bank_dir": str(self.seacr_retriever.bank_dir),
        "timestamp": datetime.now().isoformat()
    }
    fmas_file = self.output_dir / "evaluations" / "fmas_report.json"
    with open(fmas_file, 'w') as f:
        json.dump(fmas_report, f, indent=2)
    print(f"   💾 FMAS report: {fmas_file}")

    # Add FMAS into eval_summary for the lifecycle manager to pick up
    eval_summary["fmas"] = fmas
    with open(eval_file, 'w') as f:
        json.dump(eval_summary, f, indent=2)
```

#### Change E — Add CLI args in `main()`

Add to argparse in `main()` (after existing `--model` arg):
```python
parser.add_argument(
    '--seacr-bank-dir', type=str, default=None,
    help='Path to seacr_bank_* directory (Stage 1 output). '
         'If omitted, uses Calculator ID retrieval (baseline).'
)
parser.add_argument(
    '--seacr-alpha', type=float, default=0.8,
    help='Error-anchoring weight for SEACR (default: 0.8). '
         '1.0 = pure error-anchoring, 0.0 = pure question-matching.'
)
```

And pass to constructor:
```python
evaluator = ContrastiveFewShotEvaluator(
    ...,
    seacr_bank_dir=args.seacr_bank_dir,
    seacr_alpha=args.seacr_alpha
)
```

### 3.3 Add embedding utility to `pipeline/shared_utils.py`

Append to the bottom of `shared_utils.py` (after `evaluate_answer()` at line 156):

```python
def embed_texts_batch(texts: List[str], client, batch_size: int = 100):
    """
    Batch-embed texts using text-embedding-3-small.
    Used by SubmodularBankBuilder (Stage 1) and SEACRRetriever (Stage 2).

    Args:
        texts:      list of strings to embed
        client:     synchronous OpenAI client instance
        batch_size: texts per API call (OpenAI max is 2048)

    Returns:
        np.ndarray of shape (len(texts), 1536), dtype float32
    """
    import numpy as np
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = [t[:8000] for t in texts[i:i + batch_size]]
        response = client.embeddings.create(model="text-embedding-3-small", input=batch)
        batch_embs = [r.embedding for r in sorted(response.data, key=lambda x: x.index)]
        all_embeddings.extend(batch_embs)
    return np.array(all_embeddings, dtype=np.float32)
```

---

## 4. Stage 3 — Generational Utility Decay

### 4.1 New file: `pipeline/bank_lifecycle_manager.py`

**Purpose**: After evaluating a model generation, compute U(d, g) for every bank entry,
archive low-utility entries, rewrite bank.jsonl, and print a rebuild recommendation.

```python
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import argparse


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
        with open(self.bank_dir / "bank.jsonl") as f:
            for line in f:
                self.bank.append(json.loads(line))

        with open(self.bank_dir / "bank_metadata.json") as f:
            self.metadata = json.load(f)

        self.archived_dir = self.bank_dir / "archived"
        self.archived_dir.mkdir(exist_ok=True)

        # Store original size on first run
        if "original_bank_size" not in self.metadata:
            self.metadata["original_bank_size"] = len(self.bank)

    def _get_calculator_accuracy(
        self, calculator_name: str, eval_summary: Dict, prompt_type: str = "contrastive_few_shot"
    ) -> float:
        """
        Look up per-calculator accuracy from evaluation_summary.json.
        eval_summary["contrastive_few_shot"]["by_calculator"][calc_name]["accuracy"]
        Falls back to overall accuracy if calculator not found.
        """
        by_calc = eval_summary.get(prompt_type, {}).get("by_calculator", {})
        if calculator_name in by_calc:
            return float(by_calc[calculator_name].get("accuracy", 0.0))
        # Fallback to overall accuracy
        return float(eval_summary.get(prompt_type, {}).get("overall_accuracy", 0.0))

    def compute_utility(self, entry: Dict, eval_summary: Dict) -> float:
        """
        U(d, g) = difficulty(d) × (1 - accuracy(g, calc(d), failure_mode(d)))

        difficulty(d) = contrastive_sharpness stored in the bank entry.
                        If missing (correct examples), use 0.5 as neutral difficulty.
        accuracy(...)  = per-calculator accuracy from evaluation_summary for the
                         current model generation. Failure-mode breakdown not yet
                         in eval_summary, so per-calculator is the proxy.

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
            model_name:        e.g. "gpt-4o" — the model just evaluated
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
            "mean_utility":         round(float(np.mean(utilities)), 4),
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
        size_depleted = len(self.bank) < 0.30 * original_size
        fmas_decayed  = fmas is not None and fmas < self.fmas_threshold
        return size_depleted or fmas_decayed


def main():
    parser = argparse.ArgumentParser(description="Bank Lifecycle Manager (Stage 3)")
    parser.add_argument('--bank-dir',        required=True)
    parser.add_argument('--eval-summary',    required=True)
    parser.add_argument('--model-name',      required=True)
    parser.add_argument('--generation',      required=True, type=int)
    parser.add_argument('--epsilon',         default=0.05, type=float)
    parser.add_argument('--fmas-threshold',  default=0.15, type=float)
    parser.add_argument('--print-rebuild-flag', action='store_true',
                        help='Print REBUILD or OK to stdout (for shell script branching)')
    args = parser.parse_args()

    manager = BankLifecycleManager(args.bank_dir, args.epsilon, args.fmas_threshold)
    report  = manager.update_generation(args.eval_summary, args.model_name, args.generation)

    if args.print_rebuild_flag:
        print("REBUILD" if report["rebuild_recommended"] else "OK")


if __name__ == "__main__":
    main()
```

---

## 5. New Shell Scripts

### 5.1 `run_lifecycle_pipeline.sh` — full 3-stage orchestration

```bash
#!/usr/bin/env bash
set -euo pipefail
# Full lifecycle pipeline: Stage 1 (optional) → Prompt Refinement → Stage 2 → Stage 3
#
# Usage:
#   First run (build new bank):
#     ./run_lifecycle_pipeline.sh --model gpt-4o --generation 1
#   Reuse existing bank:
#     ./run_lifecycle_pipeline.sh --model gpt-5 --generation 2 --bank-dir outputs/seacr_bank_XYZ

MODEL="gpt-4o"
GENERATION=1
BANK_DIR=""
EXISTING_RESULTS_DIR="../outputs/medcalc_contrastive_edits_evaluation_20251010_054434"

while [[ $# -gt 0 ]]; do
  case $1 in
    --model)        MODEL="$2";       shift 2 ;;
    --generation)   GENERATION="$2";  shift 2 ;;
    --bank-dir)     BANK_DIR="$2";    shift 2 ;;
    --results-dir)  EXISTING_RESULTS_DIR="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

cd /Users/harivallabharangarajan/Desktop/CMU/PromptResearch/medcalc-evaluation
source ../mohs-llm-as-a-judge/llm-judge-env/bin/activate
[[ -z "${OPENAI_API_KEY:-}" ]] && { echo "OPENAI_API_KEY not set" >&2; exit 1; }

echo "🔬 Lifecycle-Aware Contrastive Few-Shot Pipeline"
echo "================================================"
echo "Model: $MODEL  |  Generation: $GENERATION"

# Stage 1: Build submodular bank (skip if bank already provided)
if [[ -z "$BANK_DIR" ]]; then
  echo ""
  echo "🏗  Stage 1: Submodular Bank Construction"
  python pipeline/submodular_bank_construction.py \
    --existing-bank-dir "$EXISTING_RESULTS_DIR" \
    --train-csv MedCalc-Bench/dataset/train_data.csv \
    --target-size 170 \
    --labeling-model gpt-4o
  BANK_DIR=$(ls -td ../outputs/seacr_bank_* | head -n 1)
  echo "   ✓ Bank: $BANK_DIR"
else
  echo ""
  echo "🏗  Stage 1: Reusing bank at $BANK_DIR"
fi

# Prompt refinement (unchanged from existing pipeline)
echo ""
echo "📝 Part 2: Prompt Refinement (gpt-5 refiner)"
python pipeline/prompt_refinement_pipeline.py \
  --results-dir "$EXISTING_RESULTS_DIR" \
  --batch-size 5 \
  --max-iterations 34 \
  --model gpt-5
REFINED_DIR=$(ls -td ../outputs/refined_prompts_* | head -n 1)

# Stage 2: SEACR evaluation
echo ""
echo "🔍 Stage 2: SEACR Evaluation on 1047 test examples"
python pipeline/evaluate_contrastive_fewshot_method.py \
  --refined-prompts-dir "$REFINED_DIR" \
  --training-results-dir "$EXISTING_RESULTS_DIR" \
  --num-test-examples 1047 \
  --num-positive 1 \
  --num-negative 1 \
  --batch-size 15 \
  --save-frequency 50 \
  --model "$MODEL" \
  --seacr-bank-dir "$BANK_DIR" \
  --seacr-alpha 0.8
EVAL_DIR=$(ls -td ../outputs/contrastive_evaluation_* | head -n 1)

python pipeline/visualize_results.py --evaluation-dir "$EVAL_DIR"

# Stage 3: Lifecycle update
echo ""
echo "🗂  Stage 3: Lifecycle Update"
REBUILD=$(python pipeline/bank_lifecycle_manager.py \
  --bank-dir "$BANK_DIR" \
  --eval-summary "$EVAL_DIR/evaluations/evaluation_summary.json" \
  --model-name "$MODEL" \
  --generation "$GENERATION" \
  --epsilon 0.05 \
  --fmas-threshold 0.15 \
  --print-rebuild-flag)

echo ""
if [[ "$REBUILD" == "REBUILD" ]]; then
  echo "⚠️  Bank rebuild recommended for next generation."
  echo "   Omit --bank-dir on next run to trigger Stage 1."
else
  echo "✅ Bank healthy. Pass --bank-dir $BANK_DIR to next run."
fi

echo ""
echo "📁 Bank:    $BANK_DIR"
echo "📁 Results: $EVAL_DIR"
```

### 5.2 `run_ablation_study.sh` — 5-condition ablation table

```bash
#!/usr/bin/env bash
set -euo pipefail
# Runs the 5-condition ablation table from the paper.
# Requires: existing bank dir (--bank-dir) and refined prompts dir (--refined-dir)
#
# Usage: ./run_ablation_study.sh --model gpt-4o --bank-dir outputs/seacr_bank_XYZ \
#                                --refined-dir outputs/refined_prompts_XYZ

MODEL="gpt-4o"
BANK_DIR=""
REFINED_DIR=""
EXISTING_RESULTS_DIR="../outputs/medcalc_contrastive_edits_evaluation_20251010_054434"

while [[ $# -gt 0 ]]; do
  case $1 in
    --model)       MODEL="$2";       shift 2 ;;
    --bank-dir)    BANK_DIR="$2";    shift 2 ;;
    --refined-dir) REFINED_DIR="$2"; shift 2 ;;
    *) echo "Unknown: $1" >&2; exit 1 ;;
  esac
done

cd /Users/harivallabharangarajan/Desktop/CMU/PromptResearch/medcalc-evaluation
source ../mohs-llm-as-a-judge/llm-judge-env/bin/activate

COMMON="--training-results-dir $EXISTING_RESULTS_DIR --num-test-examples 1047 \
        --num-positive 1 --num-negative 1 --batch-size 15 --save-frequency 50 \
        --model $MODEL --refined-prompts-dir $REFINED_DIR"

# Condition 1: Baseline (random 170, Calculator ID retrieval, no SEACR bank)
echo "🔬 Condition 1: Baseline"
python pipeline/evaluate_contrastive_fewshot_method.py $COMMON

# Condition 2: +Stage 1 only (submodular bank, but question-matching retrieval, no error-anchoring)
echo "🔬 Condition 2: +Submodular Bank only (alpha=0.0 = question-matching, no error-anchoring)"
python pipeline/evaluate_contrastive_fewshot_method.py $COMMON \
  --seacr-bank-dir "$BANK_DIR" --seacr-alpha 0.0

# Condition 3: +Stage 2 only (SEACR on original unoptimized bank)
echo "🔬 Condition 3: +SEACR only (original random bank, alpha=0.8)"
python pipeline/evaluate_contrastive_fewshot_method.py $COMMON \
  --seacr-bank-dir "$EXISTING_RESULTS_DIR" --seacr-alpha 0.8

# Condition 4: Full Stage 1 + Stage 2 (submodular bank + SEACR, no lifecycle yet)
echo "🔬 Condition 4: +Stage 1 + Stage 2 (submodular bank + SEACR)"
python pipeline/evaluate_contrastive_fewshot_method.py $COMMON \
  --seacr-bank-dir "$BANK_DIR" --seacr-alpha 0.8

# Condition 5: Full system (all three stages — lifecycle applied offline before this run)
echo "🔬 Condition 5: Full SEACR-Lifecycle (lifecycle-pruned bank + SEACR)"
# (same as Condition 4 but bank.jsonl has already been pruned by bank_lifecycle_manager.py)
python pipeline/evaluate_contrastive_fewshot_method.py $COMMON \
  --seacr-bank-dir "$BANK_DIR" --seacr-alpha 0.8

echo "✅ Ablation complete. Compare evaluation_summary.json files in each output dir."
```

---

## 6. Complete Output Directory Structure

```
outputs/
  medcalc_contrastive_edits_evaluation_20251010_054434/   ← existing, UNCHANGED
    correct/     {original,chain_of_thought,chain_of_draft}_correct.jsonl
    incorrect/   {original,chain_of_thought,chain_of_draft}_incorrect.jsonl
    prompts/     enhanced_prompts.json
    data/        training_sample_indices.json, sampled_medcalc_data.csv

  seacr_bank_{timestamp}/                                 ← Stage 1 output (NEW)
    bank.jsonl              all selected entries (incorrect + correct), enriched schema
    embeddings.npz          forward index: shape (N, 1536), all entries
    inverted_index.npz      inverted index: shape (N, 1536), wrong-answer embeddings
    bank_metadata.json      {failure_mode_distribution, fmas_baseline, original_bank_size,
                             coverage_stats, alpha, beta, gamma, delta, generation_history}
    archived/
      archived_gen1_gpt-4o.jsonl     entries retired after generation 1 evaluation
      archived_gen2_gpt-5.jsonl      entries retired after generation 2 evaluation

  refined_prompts_{timestamp}/                            ← Part 2 output (UNCHANGED)
    final/unified_prompt.txt
    iterations/unified_iteration_*.json

  contrastive_evaluation_{timestamp}/                     ← Stage 2 output
    responses/
      contrastive_few_shot_responses.jsonl    same schema as existing
    evaluations/
      evaluation_summary.json                same schema + "fmas" top-level field (NEW)
      fmas_report.json                       {fmas, model, num_probe_predictions,
                                              seacr_alpha, bank_dir, timestamp} (NEW)
    visualizations/
      ... (unchanged)
```

---

## 7. Integration Summary (exact file:line references)

| What changes | File | Location | Nature |
|---|---|---|---|
| Add `seacr_bank_dir`, `seacr_alpha` params + `seacr_retriever` init | [evaluate_contrastive_fewshot_method.py:43-96](medcalc-evaluation/pipeline/evaluate_contrastive_fewshot_method.py#L43) | Change A: extend `__init__()` |
| New probe inference method | [evaluate_contrastive_fewshot_method.py:326](medcalc-evaluation/pipeline/evaluate_contrastive_fewshot_method.py#L326) | Change B: insert `_probe_inference_async()` |
| Replace `random.sample` retrieval with SEACR | [evaluate_contrastive_fewshot_method.py:342](medcalc-evaluation/pipeline/evaluate_contrastive_fewshot_method.py#L342) | Change C: modify contrastive_few_shot branch |
| Add FMAS computation + save | [evaluate_contrastive_fewshot_method.py:645](medcalc-evaluation/pipeline/evaluate_contrastive_fewshot_method.py#L645) | Change D: append to `run_complete_evaluation()` |
| New CLI args `--seacr-bank-dir`, `--seacr-alpha` | [evaluate_contrastive_fewshot_method.py:665](medcalc-evaluation/pipeline/evaluate_contrastive_fewshot_method.py#L665) | Change E: extend argparse in `main()` |
| New `embed_texts_batch()` utility | [shared_utils.py:157](medcalc-evaluation/pipeline/shared_utils.py#L157) | Append to bottom of file |
| Submodular bank builder | `pipeline/submodular_bank_construction.py` | **New file** |
| SEACR retrieval module | `pipeline/seacr_retrieval.py` | **New file** |
| Lifecycle manager | `pipeline/bank_lifecycle_manager.py` | **New file** |
| Full pipeline orchestration | `run_lifecycle_pipeline.sh` | **New file** |
| Ablation runner | `run_ablation_study.sh` | **New file** |

**Files NOT touched**: `prompt_refinement_pipeline.py`, `contrastive_demonstration_generation.py`,
`visualize_results.py`, `custom_llm_judge.py`, both existing `.sh` scripts.

---

## 8. End-to-End Data Flow

```
train_data.csv (10k rows)
       │
       ▼  [Stage 1 — once per model generation, ~30 min for embedding + greedy]
submodular_bank_construction.py
  1. Load existing 145 incorrect + 365 correct examples from medcalc_contrastive_edits_*/
  2. LLM labeling pass: gpt-4o assigns failure_mode to each incorrect example (async, batched)
  3. Compute contrastive_sharpness for each incorrect entry (bulk embedding)
  4. Embed all questions in bulk → candidate_forward_embs
  5. Embed all wrong answers in bulk → candidate_inverted_embs
  6. Greedy submodular loop: select K=170 incorrect examples maximizing Utility(S)
  7. Build bank.jsonl (selected incorrect + all correct examples)
  8. Save embeddings.npz (question embeddings) and inverted_index.npz (wrong-answer embeddings)
  9. Probe model on 20 train examples → compute FMAS baseline → save to bank_metadata.json
       │
       ▼
seacr_bank_{timestamp}/
  bank.jsonl  ·  embeddings.npz  ·  inverted_index.npz  ·  bank_metadata.json
       │
       │  [Stage 2 — per evaluation run]
       ▼
evaluate_contrastive_fewshot_method.py (modified)
  For each of 1047 test examples (async, batch_size=15):
    1. probe_prediction = model(patient_note, question, one_shot_only)   ← NEW probe call
    2. score(d_i) = 0.8 · sim(embed(probe), inverted_index[i])           ← NEW inverted search
                 + 0.2 · sim(embed(question), forward_index[i])
    3. Retrieve top-1 incorrect by score + top-1 correct by calculator_id
    4. final_answer = model(patient_note, question, contrastive_pair)
  After all examples:
    5. FMAS = mean over probes of max sim to nearest wrong-answer in bank  ← NEW metric
    6. Save fmas_report.json + add "fmas" to evaluation_summary.json
       │
       ▼
evaluation_summary.json  ·  fmas_report.json
       │
       │  [Stage 3 — after each generation's evaluation]
       ▼
bank_lifecycle_manager.py
  For each bank entry d:
    U(d, g) = contrastive_sharpness(d) × (1 - per_calculator_accuracy(g, d.Calculator_Name))
    if U < 0.05 → archive to archived/archived_gen{N}_{model}.jsonl
  Rewrite bank.jsonl with active entries
  if FMAS < 0.15 OR active_count < 30% original → print REBUILD recommendation
  → Next generation: re-run Stage 1 with fresh failures from updated model
```

---

## 9. First Experiment: Validating SEACR (Minimal Run, Existing Data)

This experiment requires NO new data collection. Uses existing infrastructure to validate the
core SEACR claim using only what is already on disk.

```bash
# 0. Environment
cd /Users/harivallabharangarajan/Desktop/CMU/PromptResearch/medcalc-evaluation
source ../mohs-llm-as-a-judge/llm-judge-env/bin/activate
export OPENAI_API_KEY="sk-proj-..."

# 1. Stage 1: Build submodular bank from the existing 510 examples
#    (the existing 170 train examples × 3 variants already have LLM Answer + Result)
#    This only needs the existing bank as input — no new API calls for generation
python pipeline/submodular_bank_construction.py \
  --existing-bank-dir ../outputs/medcalc_contrastive_edits_evaluation_20251010_054434 \
  --train-csv MedCalc-Bench/dataset/train_data.csv \
  --target-size 170 \
  --labeling-model gpt-4o
BANK_DIR=$(ls -td ../outputs/seacr_bank_* | head -n 1)
echo "Bank: $BANK_DIR"

# 2. Stage 2 (SEACR) vs Baseline — side-by-side comparison on existing refined prompt
EXISTING_REFINED="../outputs/refined_prompts_20251010_132955"

# Baseline (Calculator ID → random sample, same as existing pipeline)
python pipeline/evaluate_contrastive_fewshot_method.py \
  --refined-prompts-dir "$EXISTING_REFINED" \
  --training-results-dir ../outputs/medcalc_contrastive_edits_evaluation_20251010_054434 \
  --num-test-examples 1047 --model gpt-4o --batch-size 15
BASELINE_DIR=$(ls -td ../outputs/contrastive_evaluation_* | head -n 1)

# SEACR (error-anchored retrieval)
python pipeline/evaluate_contrastive_fewshot_method.py \
  --refined-prompts-dir "$EXISTING_REFINED" \
  --training-results-dir ../outputs/medcalc_contrastive_edits_evaluation_20251010_054434 \
  --num-test-examples 1047 --model gpt-4o --batch-size 15 \
  --seacr-bank-dir "$BANK_DIR" --seacr-alpha 0.8
SEACR_DIR=$(ls -td ../outputs/contrastive_evaluation_* | head -n 1)

# 3. Compare results
echo "--- BASELINE ---"
python3 -c "import json; d=json.load(open('$BASELINE_DIR/evaluations/evaluation_summary.json')); \
  print(f'Accuracy: {d[\"contrastive_few_shot\"][\"overall_accuracy\"]:.2%}')"

echo "--- SEACR ---"
python3 -c "import json; \
  e=json.load(open('$SEACR_DIR/evaluations/evaluation_summary.json')); \
  f=json.load(open('$SEACR_DIR/evaluations/fmas_report.json')); \
  print(f'Accuracy: {e[\"contrastive_few_shot\"][\"overall_accuracy\"]:.2%}'); \
  print(f'FMAS: {f[\"fmas\"]:.4f}')"

# 4. Run lifecycle update
python pipeline/bank_lifecycle_manager.py \
  --bank-dir "$BANK_DIR" \
  --eval-summary "$SEACR_DIR/evaluations/evaluation_summary.json" \
  --model-name gpt-4o \
  --generation 1
```

**Expected outputs**:
- `$BANK_DIR/bank_metadata.json` contains `failure_mode_distribution` and `fmas_baseline`
- `$SEACR_DIR/evaluations/fmas_report.json` contains `fmas` score for gpt-4o
- Accuracy comparison between Baseline and SEACR gives the primary claim: SEACR > random retrieval for gpt-4o
- Lifecycle report shows which bank entries have decayed utility after gpt-4o evaluation

---

## 10. Dependencies

Add to `requirements.txt`:
```
numpy>=1.24.0       # for npz index files and matrix operations (likely already present)
openai>=1.0.0       # already present
```

No FAISS or scikit-learn required. All ANN search is exact linear scan using `numpy @ matmul`
over N=170 entries, which takes <1ms per query on CPU and needs no additional dependencies.
