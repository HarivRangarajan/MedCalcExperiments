#!/bin/bash
# Quick Start Script for Complete Optimization Pipeline
# 
# This script sets up the environment and runs the complete pipeline
# with sensible defaults.

set -e  # Exit on error

echo "================================="
echo "PROMPT OPTIMIZATION QUICK START"
echo "================================="
echo ""

# Check if we're in the right directory
if [ ! -f "run_complete_optimization_pipeline.py" ]; then
    echo "❌ Error: Please run this script from the medcalc-evaluation directory"
    exit 1
fi

# Check for API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ Error: OPENAI_API_KEY not set"
    echo ""
    echo "Please set your API key:"
    echo "  export OPENAI_API_KEY=\"your-api-key-here\""
    echo ""
    exit 1
fi

# Activate virtual environment if it exists
if [ -d "../mohs-llm-as-a-judge/llm-judge-env" ]; then
    echo "✓ Activating virtual environment..."
    source ../mohs-llm-as-a-judge/llm-judge-env/bin/activate
else
    echo "⚠️  Virtual environment not found, using system Python"
fi

# Set default training results directory
TRAINING_DIR="${1:-/Users/harivallabharangarajan/Desktop/CMU/PromptResearch/outputs/medcalc_contrastive_edits_evaluation_20251010_054434}"

if [ ! -d "$TRAINING_DIR" ]; then
    echo "❌ Error: Training results directory not found: $TRAINING_DIR"
    echo ""
    echo "Usage: $0 [training-results-dir]"
    echo ""
    echo "Example:"
    echo "  $0 /path/to/training/results"
    echo ""
    exit 1
fi

echo "✓ Using training results: $TRAINING_DIR"
echo ""

# Configuration
NUM_ITERATIONS=10
NUM_CANDIDATES=5
VALIDATION_SIZE=100

echo "Configuration:"
echo "  • Refinement iterations: $NUM_ITERATIONS"
echo "  • Candidate prompts: $NUM_CANDIDATES"
echo "  • Validation size: $VALIDATION_SIZE"
echo "  • Test all models: YES"
echo ""

# Confirm with user
read -p "Continue with this configuration? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "================================="
echo "STARTING PIPELINE"
echo "================================="
echo ""
echo "This will take approximately 2-3 hours."
echo "Progress will be displayed below."
echo ""

# Run the pipeline
python run_complete_optimization_pipeline.py \
    --training-results-dir "$TRAINING_DIR" \
    --num-refinement-iterations $NUM_ITERATIONS \
    --num-candidates $NUM_CANDIDATES \
    --validation-size $VALIDATION_SIZE \
    --test-all-models

echo ""
echo "================================="
echo "PIPELINE COMPLETE!"
echo "================================="
echo ""
echo "Check the optimization_results/ directory for outputs."
echo ""

