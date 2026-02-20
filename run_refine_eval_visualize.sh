#!/usr/bin/env bash
set -euo pipefail

# Full refinement → evaluation → visualization pipeline
# 
# Refinement: Uses GPT-5 to refine a unified prompt (not separate CoT/CoD prompts)
# Evaluation: Uses specified model (gpt-4o or gpt-5) for test set evaluation
#
# Usage: ./run_refine_eval_visualize.sh --model gpt-4o
#        ./run_refine_eval_visualize.sh --model gpt-5

# Default model
MODEL="gpt-4o"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Usage: $0 --model <model_name>" >&2
      exit 1
      ;;
  esac
done

cd /Users/harivallabharangarajan/Desktop/CMU/PromptResearch/medcalc-evaluation
source ../mohs-llm-as-a-judge/llm-judge-env/bin/activate

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY environment variable must be set before running this script." >&2
  exit 1
fi

echo "🚀 Running pipeline"
echo "================================================"
echo "Refinement model: gpt-5"
echo "Evaluation model: $MODEL"
echo ""

# Prompt refinement (uses gpt-5 by default, unified prompt approach)
# Evaluates on 170 training samples iteratively
python pipeline/prompt_refinement_pipeline.py \
  --results-dir /Users/harivallabharangarajan/Desktop/CMU/PromptResearch/outputs/medcalc_contrastive_edits_evaluation_20251010_054434 \
  --batch-size 5 \
  --max-iterations 34 \
  --model gpt-5

REFINED_DIR=$(ls -td outputs/refined_prompts_* | head -n 1)

# Full test set evaluation (1047 examples) with specified model
python pipeline/evaluate_contrastive_fewshot_method.py \
  --refined-prompts-dir "$REFINED_DIR" \
  --training-results-dir /Users/harivallabharangarajan/Desktop/CMU/PromptResearch/outputs/medcalc_contrastive_edits_evaluation_20251010_054434 \
  --num-test-examples 1047 \
  --num-positive 1 \
  --num-negative 1 \
  --batch-size 15 \
  --save-frequency 50 \
  --model "$MODEL"

EVAL_DIR=$(ls -td outputs/contrastive_evaluation_* | head -n 1)

# Generate visualizations
python pipeline/visualize_results.py --evaluation-dir "$EVAL_DIR"

echo "✅ Pipeline complete! Model: $MODEL"
echo "📁 Results: $EVAL_DIR"
