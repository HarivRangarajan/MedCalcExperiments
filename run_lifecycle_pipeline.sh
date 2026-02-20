#!/usr/bin/env bash
set -euo pipefail
# =============================================================================
# Full Lifecycle Pipeline: Stage 1 (optional) → Prompt Refinement → Stage 2 → Stage 3
#
# Usage:
#   First run (build new bank):
#     ./run_lifecycle_pipeline.sh --model gpt-5 --generation 1
#
#   Reuse existing bank (skip Stage 1):
#     ./run_lifecycle_pipeline.sh --model gpt-5 --generation 2 \
#       --bank-dir ../outputs/seacr_bank_XYZ
#
#   Skip prompt refinement (use existing refined prompt):
#     ./run_lifecycle_pipeline.sh --model gpt-5 --generation 1 \
#       --refined-dir ../outputs/refined_prompts_20251205_023422
# =============================================================================

MODEL="gpt-5"
GENERATION=1
BANK_DIR=""
REFINED_DIR=""
EXISTING_RESULTS_DIR="../outputs/medcalc_contrastive_edits_evaluation_20260218_234824"
PROBE_SIZE=60
TARGET_SIZE=100
NUM_TEST_EXAMPLES=1047
SEACR_ALPHA=0.8
ARCHIVE_FLAG="--no-archive"   # default: tag utilities, never remove entries

while [[ $# -gt 0 ]]; do
  case $1 in
    --model)          MODEL="$2";                shift 2 ;;
    --generation)     GENERATION="$2";           shift 2 ;;
    --bank-dir)       BANK_DIR="$2";             shift 2 ;;
    --refined-dir)    REFINED_DIR="$2";          shift 2 ;;
    --results-dir)    EXISTING_RESULTS_DIR="$2"; shift 2 ;;
    --probe-size)     PROBE_SIZE="$2";           shift 2 ;;
    --target-size)    TARGET_SIZE="$2";          shift 2 ;;
    --num-examples)   NUM_TEST_EXAMPLES="$2";    shift 2 ;;
    --seacr-alpha)    SEACR_ALPHA="$2";          shift 2 ;;
    --archive)        ARCHIVE_FLAG="--archive";  shift 1 ;;
    --no-archive)     ARCHIVE_FLAG="--no-archive"; shift 1 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

cd "$(dirname "$0")"
VENV="../mohs-llm-as-a-judge/llm-judge-env/bin/activate"
if [[ -f "$VENV" ]]; then
  source "$VENV"
fi
[[ -z "${OPENAI_API_KEY:-}" ]] && { echo "❌ OPENAI_API_KEY not set" >&2; exit 1; }

echo "🔬 Lifecycle-Aware Contrastive Few-Shot Pipeline"
echo "================================================"
echo "Model: $MODEL  |  Generation: $GENERATION"
echo "Bank dir: ${BANK_DIR:-'(will build new)'}"
echo "Refined dir: ${REFINED_DIR:-'(will run refinement)'}"

# ---------------------------------------------------------------------------
# Stage 1: Build submodular bank (skip if bank already provided)
# ---------------------------------------------------------------------------
if [[ -z "$BANK_DIR" ]]; then
  echo ""
  echo "🏗  Stage 1: Submodular Bank Construction"
  echo "   probe-size=$PROBE_SIZE  target-size=$TARGET_SIZE"
  python pipeline/submodular_bank_construction.py \
    --existing-bank-dir "$EXISTING_RESULTS_DIR" \
    --train-csv MedCalc-Bench/dataset/train_data.csv \
    --target-size "$TARGET_SIZE" \
    --probe-size "$PROBE_SIZE" \
    --labeling-model gpt-4o \
    --inference-model "$MODEL"
  BANK_DIR=$(ls -td ../outputs/seacr_bank_* | head -n 1)
  echo "   ✓ Bank: $BANK_DIR"
else
  echo ""
  echo "🏗  Stage 1: Reusing bank at $BANK_DIR"
fi

# ---------------------------------------------------------------------------
# Prompt refinement (skip if refined-dir provided)
# ---------------------------------------------------------------------------
if [[ -z "$REFINED_DIR" ]]; then
  echo ""
  echo "📝 Prompt Refinement"
  python pipeline/prompt_refinement_pipeline.py \
    --results-dir "$EXISTING_RESULTS_DIR" \
    --batch-size 5 \
    --max-iterations 34 \
    --model gpt-5
  REFINED_DIR=$(ls -td ../outputs/refined_prompts_* | head -n 1)
  echo "   ✓ Refined prompt: $REFINED_DIR"
else
  echo ""
  echo "📝 Prompt Refinement: Reusing $REFINED_DIR"
fi

# ---------------------------------------------------------------------------
# Stage 2: SEACR Evaluation
# ---------------------------------------------------------------------------
echo ""
echo "🔍 Stage 2: SEACR Evaluation on $NUM_TEST_EXAMPLES test examples"
python pipeline/evaluate_contrastive_fewshot_method.py \
  --refined-prompts-dir "$REFINED_DIR" \
  --training-results-dir "$EXISTING_RESULTS_DIR" \
  --num-test-examples "$NUM_TEST_EXAMPLES" \
  --num-positive 1 \
  --num-negative 1 \
  --batch-size 15 \
  --save-frequency 50 \
  --model "$MODEL" \
  --seacr-bank-dir "$BANK_DIR" \
  --seacr-alpha "$SEACR_ALPHA"
EVAL_DIR=$(ls -td ../outputs/contrastive_evaluation_* | head -n 1)

# Visualise results if script exists
if [[ -f "pipeline/visualize_results.py" ]]; then
  python pipeline/visualize_results.py --evaluation-dir "$EVAL_DIR" 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# Stage 3: Lifecycle Update
# ---------------------------------------------------------------------------
echo ""
echo "🗂  Stage 3: Lifecycle Update (${ARCHIVE_FLAG})"
COVERAGE=$(python pipeline/bank_lifecycle_manager.py \
  --bank-dir "$BANK_DIR" \
  --eval-summary "$EVAL_DIR/evaluations/evaluation_summary.json" \
  --model-name "$MODEL" \
  --generation "$GENERATION" \
  --epsilon 0.05 \
  --fmas-threshold 0.15 \
  $ARCHIVE_FLAG \
  --print-coverage-warning)

echo ""
if [[ "$COVERAGE" == "WARN" ]]; then
  echo "⚠️  Bank coverage warning — FMAS may be low or many entries have decayed utility."
  echo "   The bank is unchanged (run with --archive to prune, or rebuild Stage 1)."
else
  echo "✅ Bank coverage looks healthy for $MODEL."
fi
echo "   Pass --bank-dir $BANK_DIR to reuse this bank for the next generation."

echo ""
echo "📈 Cross-generation utility decay curve:"
python pipeline/bank_lifecycle_manager.py \
  --bank-dir "$BANK_DIR" \
  --decay-curve-only 2>/dev/null || true

# ---------------------------------------------------------------------------
# Compare against baseline
# ---------------------------------------------------------------------------
BEST_BASELINE="../outputs/contrastive_evaluation_20251205_031352_test1047"
if [[ -f "$BEST_BASELINE/evaluations/evaluation_summary.json" ]]; then
  echo ""
  echo "📊 Comparison vs baseline:"
  python pipeline/compare_baselines.py \
    --baseline-dir "$BEST_BASELINE" \
    --seacr-dir "$EVAL_DIR" 2>/dev/null || true
fi

echo ""
echo "📁 Bank:    $BANK_DIR"
echo "📁 Results: $EVAL_DIR"
