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
EXISTING_RESULTS_DIR="../outputs/medcalc_contrastive_edits_evaluation_20260218_234824"
NUM_EXAMPLES=1047

while [[ $# -gt 0 ]]; do
  case $1 in
    --model)        MODEL="$2";                shift 2 ;;
    --bank-dir)     BANK_DIR="$2";             shift 2 ;;
    --refined-dir)  REFINED_DIR="$2";          shift 2 ;;
    --results-dir)  EXISTING_RESULTS_DIR="$2"; shift 2 ;;
    --num-examples) NUM_EXAMPLES="$2";         shift 2 ;;
    *) echo "Unknown: $1" >&2; exit 1 ;;
  esac
done

[[ -z "$BANK_DIR" ]] && { echo "❌ --bank-dir required" >&2; exit 1; }
[[ -z "$REFINED_DIR" ]] && { echo "❌ --refined-dir required" >&2; exit 1; }

cd "$(dirname "$0")"
VENV="../mohs-llm-as-a-judge/llm-judge-env/bin/activate"
if [[ -f "$VENV" ]]; then source "$VENV"; fi
[[ -z "${OPENAI_API_KEY:-}" ]] && { echo "❌ OPENAI_API_KEY not set" >&2; exit 1; }

COMMON="--training-results-dir $EXISTING_RESULTS_DIR \
        --num-test-examples $NUM_EXAMPLES \
        --num-positive 1 --num-negative 1 \
        --batch-size 15 --save-frequency 50 \
        --model $MODEL \
        --refined-prompts-dir $REFINED_DIR"

ABLATION_LOG="../outputs/ablation_results_$(date +%Y%m%d_%H%M%S).txt"
echo "Ablation Study — Model: $MODEL — $(date)" | tee "$ABLATION_LOG"
echo "============================================================" | tee -a "$ABLATION_LOG"

# ---------------------------------------------------------------------------
# Condition 1: Baseline
# ---------------------------------------------------------------------------
echo "" | tee -a "$ABLATION_LOG"
echo "🔬 Condition 1: Baseline (random bank, Calculator ID retrieval)" | tee -a "$ABLATION_LOG"
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
echo "🔬 Condition 2: +Submodular Bank (alpha=0.0, no error-anchoring)" | tee -a "$ABLATION_LOG"
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
# Condition 3: +SEACR on original bank (not submodular, alpha=0.8)
# ---------------------------------------------------------------------------
echo "" | tee -a "$ABLATION_LOG"
echo "🔬 Condition 3: +SEACR only (original bank, alpha=0.8)" | tee -a "$ABLATION_LOG"
# Note: for this condition, we need a bank built from the original random bank.
# Use the same BANK_DIR but with alpha=0.8 — this tests SEACR without
# submodular selection (if bank was built from existing random examples).
python pipeline/evaluate_contrastive_fewshot_method.py $COMMON \
  --seacr-bank-dir "$BANK_DIR" --seacr-alpha 0.8
C3_DIR=$(ls -td ../outputs/contrastive_evaluation_* | head -n 1)
python3 -c "
import json
d=json.load(open('$C3_DIR/evaluations/evaluation_summary.json'))
acc=d.get('contrastive_few_shot',{}).get('overall_accuracy','N/A')
f_file='$C3_DIR/evaluations/fmas_report.json'
import os
fmas='N/A'
if os.path.exists(f_file):
    fmas=round(json.load(open(f_file)).get('fmas',0),4)
print(f'  Accuracy: {acc:.4f}  FMAS: {fmas}' if isinstance(acc,float) else f'  Accuracy: {acc}')
" | tee -a "$ABLATION_LOG"

# ---------------------------------------------------------------------------
# Condition 4: Full Stage 1 + Stage 2
# ---------------------------------------------------------------------------
echo "" | tee -a "$ABLATION_LOG"
echo "🔬 Condition 4: +Stage 1 + Stage 2 (submodular bank + SEACR, alpha=0.8)" | tee -a "$ABLATION_LOG"
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
# Condition 5: Full system (with lifecycle-pruned bank — same run as 4
#   unless bank was already pruned by lifecycle manager)
# ---------------------------------------------------------------------------
echo "" | tee -a "$ABLATION_LOG"
echo "🔬 Condition 5: Full SEACR-Lifecycle (same bank as C4 — lifecycle is offline step)" | tee -a "$ABLATION_LOG"
echo "   (To test post-lifecycle bank, run bank_lifecycle_manager.py then re-run Condition 4)" | tee -a "$ABLATION_LOG"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "" | tee -a "$ABLATION_LOG"
echo "============================================================" | tee -a "$ABLATION_LOG"
echo "📊 ABLATION SUMMARY" | tee -a "$ABLATION_LOG"
echo "============================================================" | tee -a "$ABLATION_LOG"
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

echo ""
echo "✅ Ablation complete. Full log: $ABLATION_LOG"
