# Lifecycle-Aware Contrastive Few-Shot Prompting — Implementation Progress

**Branch**: `lifecycle-aware-contrastive-bank`
**Spec**: `medcalc-evaluation/Lifecycle-Aware Contrastive Few Shot Prompting.md`

---

## Core Philosophy: Demonstrations as Living Language

The contrastive demonstration bank is built **once** using a previous model (gpt-4o) and
**reused across model generations** without rebuilding. The bank contains **both positive
(correct) and negative (incorrect) examples**, curated via separate submodular objectives.
FMAS scores are tagged per-generation in Stage 3, enabling cross-generation utility decay
analysis.

**Key design decisions**:
- Bank is NOT rebuilt each generation — it persists as a living record
- Archival is **optional** and off by default (`--no-archive`)
- Both positive and negative examples are curated (contrastive, not just negative)
- Bank size = 550 (45% positive / 55% negative)
- Refinement = 5 iterations × batch 20 (100 examples) — focus shifted to bank quality
- Evaluate across gpt-4o, gpt-5, gpt-3.5-turbo, gpt-4o-mini

---

## Why This Matters

Current few-shot contrastive prompting retrieves examples by `Calculator ID → random.sample()`.
This ignores *which mistakes the target model actually makes* and provides random positive examples
without regard to question similarity. SEACR fixes both retrieval paths:

**SEACR (Self-Error-Anchored Contrastive Retrieval)** fixes this by:
1. **Probing** the model on the test question with one-shot calculator-ID-based example (no contrastive demonstrations) → unconstrained prediction
2. **Negative retrieval**: Matching probe prediction to inverted index of wrong answers (error-anchored)
3. **Positive retrieval**: Smart embedding-based selection weighted by question similarity + calculator match
4. **Showing** the most relevant contrastive pair — a positive example the model can learn from and
   a negative example that mirrors the specific mistake the model would make

**Motivating numbers (baselines + SEACR results on MedCalc-Bench, 1047 test examples)**:
| Run | Model | Accuracy | Method |
|---|---|---|---|
| MedCalc-Bench paper | gpt-4o | 50.91% | One-shot |
| MedCalc-Bench paper | gpt-5 | 65.71% | One-shot |
| MedCalc-Bench paper | gpt-4o-mini | 44.36% | One-shot |
| MedCalc-Bench paper | gpt-3.5-turbo | 36.48% | One-shot |
| **SEACR (ours)** | **gpt-4o** | **58.55%** | Lifecycle-Aware Contrastive |
| **SEACR (ours)** | **gpt-5** | **68.67%** | Lifecycle-Aware Contrastive |
| **SEACR (ours)** | **gpt-4o-mini** | **49.95%** | Lifecycle-Aware Contrastive |
| SEACR (ours) | gpt-3.5-turbo | 36.01% | Lifecycle-Aware Contrastive |

---

## Architecture

```
Stage 1: Bank Construction (runs ONCE with gpt-4o, reused across generations)
  pipeline/submodular_bank_construction.py
  → outputs/seacr_bank_{timestamp}/
      bank.jsonl           (550 entries: 248 positive + 302 negative, with polarity field)
      embeddings.npz       (forward index: question embeddings, N×1536)
      inverted_index.npz   (inverted index: wrong-answer embeddings, N×1536)
      bank_metadata.json   (coverage stats, FMAS baseline, failure_mode_distribution)

Stage 2: SEACR Retrieval (replaces random.sample at inference, per model)
  pipeline/seacr_retrieval.py           (smart positive + error-anchored negative)
  pipeline/evaluate_contrastive_fewshot_method.py  (probe + SEACR + FMAS)
  → outputs/contrastive_evaluation_{timestamp}/
      evaluations/fmas_report.json
      evaluations/evaluation_summary.json  (+ "fmas" field)

Stage 3: Lifecycle Management (tags utility per generation, archival optional)
  pipeline/bank_lifecycle_manager.py
  → tags U(d,g) per generation; archives only when --archive is set
  → polarity-aware utility: positive and negative entries have different formulas
```

**Key new metric**: FMAS (Failure Mode Alignment Score)
```
FMAS = E[max_{d in bank} sim(embed(probe_prediction), embed(d.LLM_Answer))]
```
FMAS ≈ 1 → bank wrong-answers match model's actual errors → SEACR retrieval is precise
FMAS ≈ 0 → bank is stale for this model → trigger rebuild

---

## Implementation Phases

| Phase | File(s) | Status | Commit |
|---|---|---|---|
| 1 | Git setup + this doc | ✅ Done | `87ab599` |
| 2 | `shared_utils.py` — `embed_texts_batch()` | ✅ Done | `68e1ae5` |
| 3 | `pipeline/seacr_retrieval.py` | ✅ Done | `3acc028` |
| 4 | `pipeline/submodular_bank_construction.py` | ✅ Done | `87da986` |
| 5 | `evaluate_contrastive_fewshot_method.py` (Changes A-E) | ✅ Done | `f694709` |
| 6 | `pipeline/bank_lifecycle_manager.py` | ✅ Done | `d048cc7` |
| 7 | Shell scripts + `compare_baselines.py` | ✅ Done | `fa04d34` |
| 8 | Positive selection + smart retrieval + polarity + params | ✅ Done | `d1e627f`→`1b70459` |
| 9 | Spec + progress doc updates | ✅ Done | (this commit) |

### Phase 8 Details (Living Language + Contrastive Bank)

Changes across all pipeline files to implement:

**submodular_bank_construction.py**:
- target_size 170→550, added positive_ratio (0.45)
- New `greedy_select_positive()` with objective: 0.45·CalculatorCoverage + 0.35·CategoryCoverage - 0.20·Redundancy
- `build_indexes()` stamps `polarity` field ("positive" / "negative") on each entry

**seacr_retrieval.py**:
- Added `beta_pos` param (0.6) for smart positive retrieval
- Positive: `beta_pos · question_sim + (1-beta_pos) · calculator_match`
- Backward compat: supports both `polarity` field and `Result` field

**evaluate_contrastive_fewshot_method.py**:
- Added `--seacr-beta-pos` and `--eval-models` CLI args
- Multi-model evaluation loop

**prompt_refinement_pipeline.py**:
- batch_size 5→20, max_iterations 34→5
- Added `--bank-dir` support: loads positive/negative examples directly from SEACR bank.jsonl
- `--results-dir` is now optional (used only for seeding initial prompt from enhanced_prompts.json)

**bank_lifecycle_manager.py**:
- Polarity-aware utility: positive U = 1-accuracy, negative U = (1-sharpness)×(1-accuracy)
- Polarity counts in gen_stats

**Shell scripts**:
- Multi-model outer loop (`--eval-models`)
- Bank always built with gpt-4o (`--inference-model gpt-4o`)
- target_size=550, positive_ratio=0.45, seacr_beta_pos=0.6

---

## Key Implementation Notes

### Stage 1 Bootstrapping (important adaptation)
The spec assumes an existing bank with ~510 entries (145 incorrect). Earlier training data
had 0 incorrect entries (gpt-5 is too accurate on training examples). To handle this,
Stage 1 includes `generate_probe_failures()`:
- Runs probe inference on N training examples with one-shot calculator-ID-based examples (no contrastive demos)
- Collects wrong answers → these seed the incorrect pool
- Union with any existing incorrect entries → candidate pool for greedy selection

### Negative Submodular Objective
```
Utility(S) = 0.35 · FailureModeCoverage(S)   # diverse failure types
           + 0.35 · CalculatorCoverage(S)     # broad calculator coverage
           + 0.20 · ErrorProximitySharpness(S) # near-miss examples (hard contrasts)
           - 0.10 · Redundancy(S)              # penalize near-duplicate questions
```

### Positive Submodular Objective
```
PositiveUtility(S) = 0.45 · CalculatorCoverage(S)
                   + 0.35 · CategoryCoverage(S)
                   - 0.20 · Redundancy(S)
```

### SEACR Retrieval Formulas
```
Negative: score(d_i) = 0.8 · sim(embed(probe), inverted_index[i])   # error-anchoring
                     + 0.2 · sim(embed(question), forward_index[i])  # question-matching

Positive: score(d_i) = 0.6 · sim(embed(question), forward_index[i]) # question similarity
                     + 0.4 · calculator_match(calc_id, d_i)          # calculator bonus
```

### Evaluation Strategy
- **Correctness**: Deterministic `evaluate_answer()` → MedCalc's `check_correctness()` (numerical tolerance, range checks). Same function used by the original MedCalc-Bench repo. NOT LLM-based.
- **Failure mode classification**: LLM-based (`gpt-4o`) — classifies incorrect answers into 5 categories (arithmetic, formula_selection, input_extraction, unit_conversion, threshold_boundary).
- **Probing**: One-shot using calculator-ID-based examples from `one_shot_finalized_explanation.json` (no contrastive demonstrations). This provides the formula/context the model needs since medical calculators are domain-specific.

### Polarity-Aware Utility Decay (Stage 3)
```
Positive: U(d, g) = 1.0 - accuracy(g, calc(d))
Negative: U(d, g) = (1 - contrastive_sharpness(d)) × (1 - accuracy(g, calc(d)))
```
When a model masters a calculator → accuracy → 1 → U → 0. Archival is optional (`--no-archive` default).

---

## Ablation Conditions (run_ablation_study.sh)

Per-model ablation (runs for each model in `--eval-models`):

| Condition | Description | Expected |
|---|---|---|
| 1 | Baseline (random bank, Calculator ID retrieval) | 65.71% (gpt-5) |
| 2 | +Submodular bank only (alpha=0.0, no error-anchoring) | +? |
| 3 | +SEACR only (original bank, alpha=0.8) | +? |
| 4 | +Stage 1 + Stage 2 (submodular bank + SEACR) | +?? |
| 5 | Full system (lifecycle-pruned bank + SEACR) | +?? |

---

## How to Run

See §9 of the spec for the full experiment. Summary:

### Quick validation (Stage 1 only):
```bash
cd /Users/harivallabharangarajan/Desktop/CMU/PromptResearch/medcalc-evaluation
source ../mohs-llm-as-a-judge/llm-judge-env/bin/activate
export OPENAI_API_KEY="sk-proj-..."

python pipeline/submodular_bank_construction.py \
  --existing-bank-dir ../outputs/medcalc_contrastive_edits_evaluation_20260218_234824 \
  --train-csv MedCalc-Bench/dataset/train_data.csv \
  --target-size 550 \
  --positive-ratio 0.45 \
  --probe-size 60 \
  --labeling-model gpt-4o \
  --inference-model gpt-4o
```

### Full lifecycle pipeline (multi-model):
```bash
./run_lifecycle_pipeline.sh --model gpt-5 --generation 1 \
  --eval-models "gpt-4o,gpt-5,gpt-3.5-turbo,gpt-4o-mini"
```

### Ablation study (multi-model, all 5 conditions):
```bash
./run_ablation_study.sh --model gpt-5 \
  --eval-models "gpt-4o,gpt-5,gpt-3.5-turbo,gpt-4o-mini" \
  --bank-dir ../outputs/seacr_bank_XXXXXXXX \
  --refined-dir ../outputs/refined_prompts_20251205_023422
```

### Compare results:
```bash
python pipeline/compare_baselines.py \
  --baseline-dir ../outputs/contrastive_evaluation_20251205_031352_test1047 \
  --seacr-dir ../outputs/contrastive_evaluation_XXXXXXXX
```

---

## Results

### Bank Construction (Stage 1)
- **Bank**: `outputs/seacr_bank_20260219_231419/` — 498 entries (251 negative + 247 positive)
- **Probe size**: 300 (one-shot with calculator-ID-based examples)
- **FMAS baseline**: 0.847
- **Refined prompts**: `outputs/refined_prompts_20260220_005738/` — 5 iterations × 20 examples, best iteration 2 (60.85%)

### SEACR Evaluation (Stage 2) — Full 1047 Test Set

| Condition | Model | SEACR Acc | Paper Baseline | Delta | FMAS |
|---|---|---|---|---|---|
| SEACR (Stage 1+2) | **gpt-4o** | **58.55%** (613/1047) | 50.91% | **+7.64%** | 0.808 |
| SEACR (Stage 1+2) | **gpt-5** | **68.67%** (719/1047) | 65.71% | **+2.96%** | 0.806 |
| SEACR (Stage 1+2) | **gpt-4o-mini** | **49.95%** (523/1047) | 44.36% | **+5.59%** | 0.815 |
| SEACR (Stage 1+2) | gpt-3.5-turbo | 36.01% (377/1047) | 36.48% | -0.47% | 0.812 |

**3/4 models show improvement over paper baselines.** Largest improvement is gpt-4o (+7.64%).

### Category Breakdown (gpt-5, 68.67% overall)

| Category | Total | Correct | Accuracy |
|---|---|---|---|
| physical | 240 | 233 | 97.1% |
| lab | 327 | 260 | 79.5% |
| dosage | 40 | 25 | 62.5% |
| diagnosis | 60 | 34 | 56.7% |
| date | 60 | 33 | 55.0% |
| risk | 240 | 105 | 43.8% |
| severity | 80 | 29 | 36.3% |

### Lifecycle Tagging (Stage 3) — Cross-Generation Utility Decay

| Generation | Model | Mean U | Low-U% (< 0.05) | FMAS |
|---|---|---|---|---|
| gen3 | gpt-3.5-turbo | 0.450 | 0.6% | 0.812 |
| gen4 | gpt-4o-mini | 0.358 | 12.0% | 0.815 |
| gen1 | gpt-4o | 0.251 | 21.7% | 0.808 |
| gen2 | gpt-5 | 0.133 | 43.0% | 0.806 |

**Key insight**: Utility inversely correlates with model capability — the bank is most useful
for weaker models (gpt-3.5-turbo: 0.450 mean U) and least useful for stronger models (gpt-5:
0.133 mean U, 43.0% of entries are low-utility). This validates the "Demonstrations as Living
Language" philosophy: as models improve, demonstrations naturally decay in utility.

**Note on utility formula**: Negative utility uses `(1 - contrastive_sharpness) × (1 - accuracy)`,
consistent with bank construction's ErrorProximitySharpness objective — near-misses (low sharpness)
are more informative contrastive examples and retain higher utility.

### Output Directories

| Model | Evaluation Dir |
|---|---|
| gpt-4o | `outputs/contrastive_evaluation_20260220_011226_test1047/` |
| gpt-5 | `outputs/contrastive_evaluation_20260220_013141_test1047/` |
| gpt-3.5-turbo | `outputs/contrastive_evaluation_20260220_034850_test1047/` |
| gpt-4o-mini | `outputs/contrastive_evaluation_20260220_040502_test1047/` |
