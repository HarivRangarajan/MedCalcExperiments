# Lifecycle-Aware Contrastive Few-Shot Prompting — Implementation Progress

**Branch**: `lifecycle-aware-contrastive-bank`
**Spec**: `medcalc-evaluation/Lifecycle-Aware Contrastive Few Shot Prompting.md`

---

## Why This Matters

Current few-shot contrastive prompting retrieves negative examples by `Calculator ID → random.sample()`.
This ignores *which mistakes the target model actually makes*. A negative example showing a unit-conversion
error is useless when the model's real failure mode is arithmetic. The demonstration becomes noise.

**SEACR (Self-Error-Anchored Contrastive Retrieval)** fixes this by:
1. **Probing** the model on the test question without demonstrations → unconstrained prediction
2. **Matching** that probe prediction to the inverted index of wrong answers in the bank
3. **Showing** the bank entry whose wrong answer most resembles what the model would say

This personalizes the negative demonstration to the specific model's error profile. The lifecycle
component ensures the bank stays fresh: as the model improves (accuracy → 1), utility of entries
teaching that mistake → 0, and they are archived. When a new model generation arrives, the bank
rebuilds from fresh failures.

**Motivating numbers (current baselines on MedCalc-Bench, 1047 test examples)**:
| Run | Model | Accuracy |
|---|---|---|
| MedCalc-Bench paper | gpt-4o | 50.91% |
| `contrastive_evaluation_20251107_031128` | (unknown) | 68.67% |
| `contrastive_evaluation_20251205_031352` | gpt-5 | **65.71%** ← primary baseline to beat |

SEACR should improve over the 65.71% gpt-5 baseline by ensuring negative demonstrations
target the model's actual failure modes rather than random examples.

---

## Architecture

```
Stage 1: Bank Construction (once per model generation)
  pipeline/submodular_bank_construction.py
  → outputs/seacr_bank_{timestamp}/
      bank.jsonl           (enriched entries: failure_mode, contrastive_sharpness, generation)
      embeddings.npz       (forward index: question embeddings, N×1536)
      inverted_index.npz   (inverted index: wrong-answer embeddings, N×1536)
      bank_metadata.json   (coverage stats, FMAS baseline, failure_mode_distribution)

Stage 2: SEACR Retrieval (replaces random.sample at inference)
  pipeline/seacr_retrieval.py           (new module)
  pipeline/evaluate_contrastive_fewshot_method.py  (5 targeted changes)
  → outputs/contrastive_evaluation_{timestamp}/
      evaluations/fmas_report.json      (NEW: FMAS metric)
      evaluations/evaluation_summary.json  (+ "fmas" field added)

Stage 3: Lifecycle Management (after each model generation eval)
  pipeline/bank_lifecycle_manager.py
  → archives low-utility entries, recommends Stage 1 rebuild when FMAS decays
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

---

## Key Implementation Notes

### Stage 1 Bootstrapping (important adaptation)
The spec assumes an existing bank with ~510 entries (145 incorrect). Our current
`medcalc_contrastive_edits_evaluation_20260218_234824` has 0 incorrect entries (gpt-5 is
too accurate on training examples). To handle this, Stage 1 includes `generate_probe_failures()`:
- Runs probe inference on N training examples (no demonstrations)
- Collects wrong answers → these seed the incorrect pool
- Union with any existing incorrect entries → candidate pool for greedy selection

This is spec-faithful: §2.1.2 explicitly says candidates include "new examples generated via
probe inference on the wider 10k pool."

### Stage 1 Submodular Objective
```
Utility(S) = 0.35 · FailureModeCoverage(S)   # diverse failure types
           + 0.35 · CalculatorCoverage(S)     # broad calculator coverage
           + 0.20 · ErrorProximitySharpness(S) # near-miss examples (hard contrasts)
           - 0.10 · Redundancy(S)              # penalize near-duplicate questions
```

### SEACR Retrieval Formula
```
score(d_i) = 0.8 · sim(embed(probe_prediction), inverted_index[i])  # error-anchoring
           + 0.2 · sim(embed(question), forward_index[i])            # question-matching
```

### Utility Decay (Stage 3)
```
U(d, g) = contrastive_sharpness(d) × (1 - accuracy(g, calculator_of_d))
```
When a model masters a calculator → accuracy → 1 → U → 0 → entry archived.

---

## Ablation Conditions (run_ablation_study.sh)

| Condition | Description | Expected |
|---|---|---|
| 1 | Baseline (random bank, Calculator ID retrieval) | 65.71% |
| 2 | +Submodular bank only (alpha=0.0, no error-anchoring) | +? |
| 3 | +SEACR only (original bank, alpha=0.8) | +? |
| 4 | +Stage 1 + Stage 2 (submodular bank + SEACR) | +?? |
| 5 | Full system (lifecycle-pruned bank + SEACR) | +?? |

---

## How to Run

See §9 of the spec for the minimal first experiment. Full command at the end of this doc.

### Quick validation (Stage 1 only):
```bash
cd /Users/harivallabharangarajan/Desktop/CMU/PromptResearch/medcalc-evaluation
source ../mohs-llm-as-a-judge/llm-judge-env/bin/activate
export OPENAI_API_KEY="sk-proj-..."

python pipeline/submodular_bank_construction.py \
  --existing-bank-dir ../outputs/medcalc_contrastive_edits_evaluation_20260218_234824 \
  --train-csv MedCalc-Bench/dataset/train_data.csv \
  --target-size 100 \
  --probe-size 60 \
  --labeling-model gpt-4o
```

### Full lifecycle pipeline:
```bash
./run_lifecycle_pipeline.sh --model gpt-5 --generation 1
```

### Ablation study (compare all 5 conditions):
```bash
./run_ablation_study.sh --model gpt-5 \
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

## Results (updated as experiments run)

| Condition | Model | Accuracy | FMAS | Notes |
|---|---|---|---|---|
| Baseline (contrastive, random) | gpt-5 | 65.71% | N/A | Dec 2025 run |
| SEACR (Stage 1+2) | gpt-5 | TBD | TBD | |
| Full lifecycle | gpt-5 | TBD | TBD | |
