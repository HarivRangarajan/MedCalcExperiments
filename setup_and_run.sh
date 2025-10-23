#!/bin/bash
# Setup and Run Script for MedCalc Contrastive Boosted Edits
# This script prepares the environment and runs the pipeline

set -e  # Exit on error

echo "🏥 MedCalc Contrastive Boosted Edits - Setup & Run"
echo "=================================================="
echo ""

# Check if we're in the right directory
if [ ! -d "MedCalc-Bench" ]; then
    echo "❌ Error: MedCalc-Bench directory not found!"
    echo "   Please run this script from the medcalc-evaluation directory"
    exit 1
fi

echo "✅ In correct directory"

# Check for virtual environment
VENV_PATH="../mohs-llm-as-a-judge/llm-judge-env/bin/activate"
if [ ! -f "$VENV_PATH" ]; then
    echo "❌ Error: Virtual environment not found at $VENV_PATH"
    exit 1
fi

echo "✅ Virtual environment found"

# Activate virtual environment
echo ""
echo "🔧 Activating virtual environment..."
source "$VENV_PATH"
echo "✅ Virtual environment activated"

# Check for API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo ""
    echo "⚠️  OPENAI_API_KEY not set!"
    echo "   Please set your API key:"
    echo "   export OPENAI_API_KEY='sk-...'"
    exit 1
fi

echo "✅ API key found"

# Check for train data
TRAIN_DATA="MedCalc-Bench/dataset/train_data.csv"
if [ ! -f "$TRAIN_DATA" ]; then
    echo ""
    echo "📦 Train data not found, extracting..."
    if [ -f "MedCalc-Bench/dataset/train_data.csv.zip" ]; then
        cd MedCalc-Bench/dataset
        unzip -q train_data.csv.zip
        cd ../..
        echo "✅ Train data extracted"
    else
        echo "❌ Error: train_data.csv.zip not found"
        exit 1
    fi
else
    echo "✅ Train data found"
fi

# Show menu
echo ""
echo "📋 What would you like to do?"
echo ""
echo "1) Run test (2 samples, ~$0.05)"
echo "2) Run quick test (10 samples, ~$0.50)"
echo "3) Run full evaluation (500 samples, ~$15-25)"
echo "4) Run custom sample size"
echo ""
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        echo ""
        echo "🧪 Running test with 2 samples..."
        python test_contrastive_edits.py
        ;;
    2)
        echo ""
        echo "🚀 Running quick test with 10 samples..."
        python run_medcalc_with_contrastive_boosted_edits.py
        ;;
    3)
        echo ""
        echo "🚀 Running full evaluation with 500 samples..."
        echo "⚠️  This will take 30-45 minutes and cost approximately $15-25"
        read -p "Continue? (y/n): " confirm
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            python medcalc_with_contrastive_boosted_edits.py --sample-size 500
        else
            echo "Cancelled."
            exit 0
        fi
        ;;
    4)
        echo ""
        read -p "Enter sample size: " sample_size
        echo ""
        echo "🚀 Running with $sample_size samples..."
        python medcalc_with_contrastive_boosted_edits.py --sample-size "$sample_size"
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "✅ Done!"

