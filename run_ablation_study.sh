#!/usr/bin/env bash
set -euo pipefail
# =============================================================================
# Ablation Study: 5-condition comparison table
#
# Conditions:
#   1. Baseline (random bank, Calculator ID retrieval, no SEACR)
#   2. +Submodular bank only (alpha=0.0 = question-matching, no error-anchoring)
#   3. +SEACR on original/existing bank (alpha=0.8, no submodular selection)
#   4. +Stage 1 + Stage 2 (submodular bank + SEACR retrieval)
#   5. Full system (lifecycle-pruned bank + SEACR)
#
# Requires: bank-dir and refined-dir pre-built (run Stage 1 first)
#
# Usage:
#   ./run_ablation_study.sh --model gpt-5 \
#     --bank-dir ../outputs/seacr_bank_XYZ \
#     --refined-dir ../outputs/refined_prompts_20251205_023422
# =============================================================================

MODEL="gpt-5"
BANK_DIR=""
REFINED_DIR=""
EXISTING_RESULTS_DIR="../outputs/medcalc_contrastive_edits_evaluation_20251010_054434"
NUM_EXAMPLES=1047
EVAL_MODELS=""    # empty = use MODEL only; set to "gpt-4o,gpt-5,gpt-3.5-turbo,gpt-4o-mini" for multi-model
SEACR_BETA_POS=0.6

while [[ $# -gt 0 ]]; do
  case $1 in
    --model)          MODEL="$2";                shift 2 ;;
    --bank-dir)       BANK_DIR="$2";             shift 2 ;;
    --refined-dir)    REFINED_DIR="$2";          shift 2 ;;
    --results-dir)    EXISTING_RESULTS_DIR="$2"; shift 2 ;;
    --num-examples)   NUM_EXAMPLES="$2";         shift 2 ;;
    --eval-models)    EVAL_MODELS="$2";          shift 2 ;;
    --seacr-beta-pos) SEACR_BETA_POS="$2";      shift 2 ;;
    *) echo "Unknown: $1" >&2; exit 1 ;;
  esac
done

[[ -z "$BANK_DIR" ]] && { echo "❌ --bank-dir required" >&2; exit 1; }
[[ -z "$REFINED_DIR" ]] && { echo "❌ --refined-dir required" >&2; exit 1; }

cd "$(dirname "$0")"
VENV="../mohs-llm-as-a-judge/llm-judge-env/bin/activate"
if [[ -f "$VENV" ]]; then source "$VENV"; fi
[[ -z "${OPENAI_API_KEY:-}" ]] && { echo "❌ OPENAI_API_KEY not set" >&2; exit 1; }

# Determine model list
if [[ -n "$EVAL_MODELS" ]]; then
  IFS=',' read -ra MODEL_LIST <<< "$EVAL_MODELS"
else
  MODEL_LIST=("$MODEL")
fi

ABLATION_LOG="../outputs/ablation_results_$(date +%Y%m%d_%H%M%S).txt"
echo "Ablation Study — Models: ${MODEL_LIST[*]} — $(date)" | tee "$ABLATION_LOG"
echo "============================================================" | tee -a "$ABLATION_LOG"

for ABLATION_MODEL in "${MODEL_LIST[@]}"; do

echo "" | tee -a "$ABLATION_LOG"
echo "============================================================" | tee -a "$ABLATION_LOG"
echo "  MODEL: $ABLATION_MODEL" | tee -a "$ABLATION_LOG"
echo "============================================================" | tee -a "$ABLATION_LOG"

COMMON="--training-results-dir $EXISTING_RESULTS_DIR \
        --num-test-examples $NUM_EXAMPLES \
        --num-positive 1 --num-negative 1 \
        --batch-size 15 --save-frequency 50 \
        --model $ABLATION_MODEL \
        --seacr-beta-pos $SEACR_BETA_POS \
        --refined-prompts-dir $REFINED_DIR"

# ---------------------------------------------------------------------------
# Condition 1: Baseline
# ---------------------------------------------------------------------------
echo "" | tee -a "$ABLATION_LOG"
echo "🔬 C1: Baseline (random bank, Calculator ID retrieval) — $ABLATION_MODEL" | tee -a "$ABLATION_LOG"
python pipeline/evaluate_contrastive_fewshot_method.py $COMMON
C1_DIR=$(ls -td ../outputs/contrastive_evaluation_* | head -n 1)
python3 -c "
import json
d=json.load(open('$C1_DIR/evaluations/evaluation_summary.json'))
acc=d.get('contrastive_few_shot',{}).get('overall_accuracy','N/A')
print(f'  Accuracy: {acc:.4f}' if isinstance(acc,float) else f'  Accuracy: {acc}')
" | tee -a "$ABLATION_LOG"

# ---------------------------------------------------------------------------
# Condition 2: +Submodular bank only (alpha=0.0 = question-matching)
# ---------------------------------------------------------------------------
echo "" | tee -a "$ABLATION_LOG"
echo "🔬 C2: +Submodular Bank (alpha=0.0, no error-anchoring) — $ABLATION_MODEL" | tee -a "$ABLATION_LOG"
python pipeline/evaluate_contrastive_fewshot_method.py $COMMON \
  --seacr-bank-dir "$BANK_DIR" --seacr-alpha 0.0
C2_DIR=$(ls -td ../outputs/contrastive_evaluation_* | head -n 1)
python3 -c "
import json
d=json.load(open('$C2_DIR/evaluations/evaluation_summary.json'))
acc=d.get('contrastive_few_shot',{}).get('overall_accuracy','N/A')
print(f'  Accuracy: {acc:.4f}' if isinstance(acc,float) else f'  Accuracy: {acc}')
" | tee -a "$ABLATION_LOG"

# ---------------------------------------------------------------------------
# Condition 3: +SEACR on original bank (alpha=0.8)
# ---------------------------------------------------------------------------
echo "" | tee -a "$ABLATION_LOG"
echo "🔬 C3: +SEACR only (original bank, alpha=0.8) — $ABLATION_MODEL" | tee -a "$ABLATION_LOG"
python pipeline/evaluate_contrastive_fewshot_method.py $COMMON \
  --seacr-bank-dir "$BANK_DIR" --seacr-alpha 0.8
C3_DIR=$(ls -td ../outputs/contrastive_evaluation_* | head -n 1)
python3 -c "
import json, os
d=json.load(open('$C3_DIR/evaluations/evaluation_summary.json'))
acc=d.get('contrastive_few_shot',{}).get('overall_accuracy','N/A')
fmas=d.get('fmas','N/A')
print(f'  Accuracy: {acc:.4f}  FMAS: {fmas}' if isinstance(acc,float) else f'  Accuracy: {acc}')
" | tee -a "$ABLATION_LOG"

# ---------------------------------------------------------------------------
# Condition 4: Full Stage 1 + Stage 2
# ---------------------------------------------------------------------------
echo "" | tee -a "$ABLATION_LOG"
echo "🔬 C4: +Stage 1 + Stage 2 (submodular bank + SEACR, alpha=0.8) — $ABLATION_MODEL" | tee -a "$ABLATION_LOG"
python pipeline/evaluate_contrastive_fewshot_method.py $COMMON \
  --seacr-bank-dir "$BANK_DIR" --seacr-alpha 0.8
C4_DIR=$(ls -td ../outputs/contrastive_evaluation_* | head -n 1)
python3 -c "
import json, os
d=json.load(open('$C4_DIR/evaluations/evaluation_summary.json'))
acc=d.get('contrastive_few_shot',{}).get('overall_accuracy','N/A')
fmas=d.get('fmas','N/A')
print(f'  Accuracy: {acc:.4f}  FMAS: {fmas}' if isinstance(acc,float) else f'  Accuracy: {acc}')
" | tee -a "$ABLATION_LOG"

# ---------------------------------------------------------------------------
# Condition 5: Full system (lifecycle is offline step)
# ---------------------------------------------------------------------------
echo "" | tee -a "$ABLATION_LOG"
echo "🔬 C5: Full SEACR-Lifecycle (same bank as C4 — lifecycle is offline) — $ABLATION_MODEL" | tee -a "$ABLATION_LOG"
echo "   (To test post-lifecycle bank, run bank_lifecycle_manager.py then re-run C4)" | tee -a "$ABLATION_LOG"

# ---------------------------------------------------------------------------
# Per-model summary
# ---------------------------------------------------------------------------
echo "" | tee -a "$ABLATION_LOG"
echo "📊 $ABLATION_MODEL ABLATION SUMMARY" | tee -a "$ABLATION_LOG"
echo "------------------------------------------------------------" | tee -a "$ABLATION_LOG"
python3 - <<PYEOF | tee -a "$ABLATION_LOG"
import json, os

dirs = [
    ("C1 Baseline (random)", "$C1_DIR"),
    ("C2 +Submodular bank (alpha=0)", "$C2_DIR"),
    ("C3 +SEACR (alpha=0.8)", "$C3_DIR"),
    ("C4 Stage1+Stage2 (SEACR)", "$C4_DIR"),
]

baseline_acc = None
for label, d in dirs:
    evf = f"{d}/evaluations/evaluation_summary.json"
    if not os.path.exists(evf):
        print(f"  {label}: missing")
        continue
    ev = json.load(open(evf))
    acc = ev.get("contrastive_few_shot", {}).get("overall_accuracy", None)
    fmas = ev.get("fmas", None)
    if acc is None:
        print(f"  {label}: no accuracy in summary")
        continue
    if baseline_acc is None:
        baseline_acc = acc
    delta = acc - baseline_acc if baseline_acc else 0
    fmas_str = f"  FMAS={fmas:.4f}" if fmas is not None else ""
    print(f"  {label:45s}  acc={acc:.4f}  Δ={delta:+.4f}{fmas_str}")
PYEOF

done  # end model loop

echo ""
echo "✅ Ablation complete. Full log: $ABLATION_LOG"
