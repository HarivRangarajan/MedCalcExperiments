#!/usr/bin/env bash
set -euo pipefail

# Run Native Baseline Evaluation with GPT-5
# This evaluates MedCalc's original one-shot prompt on the full test set
# Now uses the unified evaluate_contrastive_fewshot_method.py script
#
# Usage:
#   ./run_native_baseline_gpt5.sh                    # Run on full test set (1047 examples)
#   ./run_native_baseline_gpt5.sh --num-examples 100 # Run on 100 examples

# Default number of test examples
NUM_EXAMPLES=1047

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --num-examples)
      NUM_EXAMPLES="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Usage: $0 [--num-examples N]" >&2
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

echo "🏥 Running Native Baseline Evaluation with GPT-5"
echo "================================================"
echo "Test examples: $NUM_EXAMPLES"
echo ""

# Use unified script with --baseline-only flag
python pipeline/evaluate_contrastive_fewshot_method.py \
  --refined-prompts-dir /Users/harivallabharangarajan/Desktop/CMU/PromptResearch/outputs/medcalc_contrastive_edits_evaluation_20251010_054434 \
  --training-results-dir /Users/harivallabharangarajan/Desktop/CMU/PromptResearch/outputs/medcalc_contrastive_edits_evaluation_20251010_054434 \
  --num-test-examples "$NUM_EXAMPLES" \
  --batch-size 15 \
  --save-frequency 50 \
  --model gpt-5 \
  --baseline-only

echo ""
echo "✅ Native baseline evaluation complete!"

