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
EXISTING_RESULTS_DIR="../outputs/medcalc_contrastive_edits_evaluation_20251010_054434"
PROBE_SIZE=300
TARGET_SIZE=550
POSITIVE_RATIO=0.45
NUM_TEST_EXAMPLES=1047
SEACR_ALPHA=0.8
SEACR_BETA_POS=0.6
ARCHIVE_FLAG="--no-archive"   # default: tag utilities, never remove entries
EVAL_MODELS=""                 # empty = use MODEL; set to "gpt-4o,gpt-5,gpt-3.5-turbo,gpt-4o-mini" for multi-model

while [[ $# -gt 0 ]]; do
  case $1 in
    --model)          MODEL="$2";                shift 2 ;;
    --generation)     GENERATION="$2";           shift 2 ;;
    --bank-dir)       BANK_DIR="$2";             shift 2 ;;
    --refined-dir)    REFINED_DIR="$2";          shift 2 ;;
    --results-dir)    EXISTING_RESULTS_DIR="$2"; shift 2 ;;
    --probe-size)     PROBE_SIZE="$2";           shift 2 ;;
    --target-size)    TARGET_SIZE="$2";          shift 2 ;;
    --positive-ratio) POSITIVE_RATIO="$2";       shift 2 ;;
    --num-examples)   NUM_TEST_EXAMPLES="$2";    shift 2 ;;
    --seacr-alpha)    SEACR_ALPHA="$2";          shift 2 ;;
    --seacr-beta-pos) SEACR_BETA_POS="$2";      shift 2 ;;
    --archive)        ARCHIVE_FLAG="--archive";  shift 1 ;;
    --no-archive)     ARCHIVE_FLAG="--no-archive"; shift 1 ;;
    --eval-models)    EVAL_MODELS="$2";          shift 2 ;;
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
  echo "🏗  Stage 1: Submodular Bank Construction (bank built with gpt-4o)"
  echo "   probe-size=$PROBE_SIZE  target-size=$TARGET_SIZE  positive-ratio=$POSITIVE_RATIO"
  python pipeline/submodular_bank_construction.py \
    --existing-bank-dir "$EXISTING_RESULTS_DIR" \
    --train-csv MedCalc-Bench/dataset/train_data.csv \
    --target-size "$TARGET_SIZE" \
    --positive-ratio "$POSITIVE_RATIO" \
    --probe-size "$PROBE_SIZE" \
    --labeling-model gpt-4o \
    --inference-model gpt-4o
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
  echo "📝 Prompt Refinement (bank-based, 5 iterations × 20 examples)"
  python pipeline/prompt_refinement_pipeline.py \
    --bank-dir "$BANK_DIR" \
    --results-dir "$EXISTING_RESULTS_DIR" \
    --batch-size 20 \
    --max-iterations 5 \
    --model gpt-5
  REFINED_DIR=$(ls -td ../outputs/refined_prompts_* | head -n 1)
  echo "   ✓ Refined prompt: $REFINED_DIR"
else
  echo ""
  echo "📝 Prompt Refinement: Reusing $REFINED_DIR"
fi

# ---------------------------------------------------------------------------
# Stage 2: SEACR Evaluation (single-model or multi-model)
# ---------------------------------------------------------------------------

# Determine model list
if [[ -n "$EVAL_MODELS" ]]; then
  IFS=',' read -ra MODEL_LIST <<< "$EVAL_MODELS"
else
  MODEL_LIST=("$MODEL")
fi

EVAL_DIRS=()
for EVAL_MODEL in "${MODEL_LIST[@]}"; do
  echo ""
  echo "🔍 Stage 2: SEACR Evaluation — model=$EVAL_MODEL on $NUM_TEST_EXAMPLES test examples"
  python pipeline/evaluate_contrastive_fewshot_method.py \
    --refined-prompts-dir "$REFINED_DIR" \
    --training-results-dir "$EXISTING_RESULTS_DIR" \
    --num-test-examples "$NUM_TEST_EXAMPLES" \
    --num-positive 1 \
    --num-negative 1 \
    --batch-size 15 \
    --save-frequency 50 \
    --model "$EVAL_MODEL" \
    --seacr-bank-dir "$BANK_DIR" \
    --seacr-alpha "$SEACR_ALPHA" \
    --seacr-beta-pos "$SEACR_BETA_POS"
  EVAL_DIR=$(ls -td ../outputs/contrastive_evaluation_* | head -n 1)
  EVAL_DIRS+=("$EVAL_DIR")

  # Visualise results if script exists
  if [[ -f "pipeline/visualize_results.py" ]]; then
    python pipeline/visualize_results.py --evaluation-dir "$EVAL_DIR" 2>/dev/null || true
  fi

  # ---------------------------------------------------------------------------
  # Stage 3: Lifecycle Update (per model)
  # ---------------------------------------------------------------------------
  echo ""
  echo "🗂  Stage 3: Lifecycle Update for $EVAL_MODEL (${ARCHIVE_FLAG})"
  COVERAGE=$(python pipeline/bank_lifecycle_manager.py \
    --bank-dir "$BANK_DIR" \
    --eval-summary "$EVAL_DIR/evaluations/evaluation_summary.json" \
    --model-name "$EVAL_MODEL" \
    --generation "$GENERATION" \
    --epsilon 0.05 \
    --fmas-threshold 0.15 \
    $ARCHIVE_FLAG \
    --print-coverage-warning)

  echo ""
  if [[ "$COVERAGE" == "WARN" ]]; then
    echo "⚠️  Bank coverage warning for $EVAL_MODEL — FMAS may be low or many entries have decayed utility."
    echo "   The bank is unchanged (run with --archive to prune, or rebuild Stage 1)."
  else
    echo "✅ Bank coverage looks healthy for $EVAL_MODEL."
  fi
done

echo ""
echo "📈 Cross-generation utility decay curve:"
python pipeline/bank_lifecycle_manager.py \
  --bank-dir "$BANK_DIR" \
  --decay-curve-only 2>/dev/null || true

# ---------------------------------------------------------------------------
# Compare against baseline (use last eval dir)
# ---------------------------------------------------------------------------
BEST_BASELINE="../outputs/contrastive_evaluation_20251205_031352_test1047"
LAST_EVAL_DIR="${EVAL_DIRS[-1]}"
if [[ -f "$BEST_BASELINE/evaluations/evaluation_summary.json" ]]; then
  echo ""
  echo "📊 Comparison vs baseline:"
  python pipeline/compare_baselines.py \
    --baseline-dir "$BEST_BASELINE" \
    --seacr-dir "$LAST_EVAL_DIR" 2>/dev/null || true
fi

echo ""
echo "📁 Bank:    $BANK_DIR"
echo "   Pass --bank-dir $BANK_DIR to reuse this bank for the next generation."
for ED in "${EVAL_DIRS[@]}"; do
  echo "📁 Results: $ED"
done
