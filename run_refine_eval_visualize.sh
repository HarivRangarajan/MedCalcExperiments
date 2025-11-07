#!/usr/bin/env bash
set -euo pipefail

# Recreate the full refinement → evaluation → visualization pipeline command
# Notes:
# - pipeline/prompt_refinement_pipeline.py:229-253 caps refinement to 50 training rows with --batch-size 10 and --max-iterations 5
# - pipeline/evaluate_contrastive_fewshot_method.py:169-180 limits MedCalc test coverage via --num-test-examples 100 while keeping 1 positive/negative contrastive example per query
# - pipeline/visualize_results.py:1-34 only needs the evaluation directory to drop PNG/PDF figures into visualizations/

cd /Users/harivallabharangarajan/Desktop/CMU/PromptResearch/medcalc-evaluation
source ../mohs-llm-as-a-judge/llm-judge-env/bin/activate

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY environment variable must be set before running this script." >&2
  exit 1
fi

python pipeline/prompt_refinement_pipeline.py \
  --results-dir /Users/harivallabharangarajan/Desktop/CMU/PromptResearch/outputs/medcalc_contrastive_edits_evaluation_20251010_054434 \
  --batch-size 10 \
  --max-iterations 5

REFINED_DIR=$(ls -td outputs/refined_prompts_* | head -n 1)

python pipeline/evaluate_contrastive_fewshot_method.py \
  --refined-prompts-dir "$REFINED_DIR" \
  --training-results-dir /Users/harivallabharangarajan/Desktop/CMU/PromptResearch/outputs/medcalc_contrastive_edits_evaluation_20251010_054434 \
  --num-test-examples 100 \
  --num-positive 1 \
  --num-negative 1 \
  --batch-size 10 \
  --save-frequency 25

EVAL_DIR=$(ls -td outputs/contrastive_evaluation_* | head -n 1)

python pipeline/visualize_results.py --evaluation-dir "$EVAL_DIR"
