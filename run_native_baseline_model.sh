#!/usr/bin/env bash
set -euo pipefail

# Run Native Baseline Evaluation with any OpenAI model
# This evaluates MedCalc's original one-shot prompt on the full test set
# Now uses the unified evaluate_contrastive_fewshot_method.py script
#
# Usage:
#   ./run_native_baseline_model.sh --model gpt-4o                    # Run with GPT-4o on full test set
#   ./run_native_baseline_model.sh --model gpt-5 --num-examples 100  # Run with GPT-5 on 100 examples

# Default values
NUM_EXAMPLES=1047
MODEL=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --num-examples)
      NUM_EXAMPLES="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Usage: $0 --model <model_name> [--num-examples N]" >&2
      exit 1
      ;;
  esac
done

# Check if model is provided
if [[ -z "$MODEL" ]]; then
  echo "Error: --model parameter is required" >&2
  echo "Usage: $0 --model <model_name> [--num-examples N]" >&2
  echo "Example: $0 --model gpt-4o" >&2
  exit 1
fi

cd /Users/harivallabharangarajan/Desktop/CMU/PromptResearch/medcalc-evaluation
source ../mohs-llm-as-a-judge/llm-judge-env/bin/activate

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY environment variable must be set before running this script." >&2
  exit 1
fi

echo "🏥 Running Native Baseline Evaluation with $MODEL"
echo "================================================"
echo "Model: $MODEL"
echo "Test examples: $NUM_EXAMPLES"
echo ""

# Use unified script with --baseline-only flag
python pipeline/evaluate_contrastive_fewshot_method.py \
  --refined-prompts-dir /Users/harivallabharangarajan/Desktop/CMU/PromptResearch/outputs/medcalc_contrastive_edits_evaluation_20251010_054434 \
  --training-results-dir /Users/harivallabharangarajan/Desktop/CMU/PromptResearch/outputs/medcalc_contrastive_edits_evaluation_20251010_054434 \
  --num-test-examples "$NUM_EXAMPLES" \
  --batch-size 15 \
  --save-frequency 50 \
  --model "$MODEL" \
  --baseline-only

echo ""
echo "✅ Native baseline evaluation complete!"

