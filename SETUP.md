# MedCalc-Bench Evaluation Setup

This guide will help you set up a complete, self-contained environment for running MedCalc-Bench evaluations with PromptEngineer techniques.

## Prerequisites

1. **Python 3.8+**: Required for running the evaluation pipeline
2. **Git**: For cloning repositories
3. **OpenAI API Key**: For running LLM evaluations

## Step 1: Clone Required Repositories

### Clone the MedCalc-Bench Repository

The evaluation pipeline requires the MedCalc-Bench dataset and evaluation code.
Clone it into this directory:

```bash
git clone https://github.com/ncbi-nlp/MedCalc-Bench.git
```

### Clone the PromptEngineer Library

Clone the PromptEngineer library one level above this directory:

```bash
# Go up one level from medcalc-evaluation
cd ..

# Clone the PromptEngineer library
git clone https://github.com/HarivRangarajan/promptengineer.git

# Return to medcalc-evaluation directory
cd medcalc-evaluation
```

Expected directory structure:
```
PromptResearch/
├── promptengineer/           # PromptEngineer library
├── medcalc-evaluation/       # This directory
│   ├── MedCalc-Bench/        # MedCalc-Bench repository
│   ├── requirements.txt
│   ├── run_medcalc_evaluation.py
│   └── ...
└── outputs/                  # Generated results (gitignored)
```

## Step 2: Set Up Virtual Environment

Create a dedicated virtual environment for this project:

```bash
# Create virtual environment
python3 -m venv medcalc-env

# Activate the virtual environment
# On macOS/Linux:
source medcalc-env/bin/activate
# On Windows:
# medcalc-env\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

## Step 3: Install Dependencies

Install all required packages:

```bash
# Install the main requirements
pip install -r requirements.txt

# Install the PromptEngineer library in development mode
pip install -e ../promptengineer

# Verify installations
python -c "import pandas, numpy, matplotlib, seaborn, openai, tiktoken; print('✅ Core dependencies installed')"
python -c "from promptengineer import PromptPipeline; print('✅ PromptEngineer library installed')"
```

## Step 4: Configure API Key

Set up your OpenAI API key using one of these methods:

### Option 1: Environment Variable (Recommended)
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Option 2: Create a local config file
```bash
# Create a config.py file in this directory
cat > config.py << EOF
OPENAI_API_KEY = "your-api-key-here"
EOF

# Add config.py to .gitignore (it's already there)
```

## Step 5: Verify Setup

Run the setup verification script:

```bash
# Test the complete pipeline setup
python test_setup.py

# Verify MedCalc-Bench data is available
ls MedCalc-Bench/dataset/test_data.csv

# Test imports
python -c "
import sys
sys.path.append('MedCalc-Bench/evaluation')
from evaluate import evaluate_answer
from promptengineer import PromptPipeline
from promptengineer.techniques.base import PromptContext
print('✅ All imports successful')
"
```

## Quick Start

Once everything is set up, you can run evaluations:

```bash
# Make sure your virtual environment is activated
source medcalc-env/bin/activate

# Run with default settings (20 samples, budget limit $10)
python run_medcalc_evaluation.py

# Or run the pipeline directly with custom parameters
python medcalc_prompt_evaluation_pipeline.py --sample-size 2 --skip-judge --budget-limit 5.0
```

## Output Structure

All results will be saved to the centralized `../outputs/` directory:

```
outputs/
└── medcalc_evaluation_YYYYMMDD_HHMMSS/
    ├── data/
    │   └── sampled_medcalc_data.csv
    ├── prompts/
    │   ├── original_medcalc_prompts.json
    │   └── all_enhanced_prompts.json
    ├── responses/
    │   └── all_responses.json
    ├── evaluations/
    │   ├── accuracy_results.json
    │   └── llm_judge_results.json
    ├── visualizations/
    │   ├── overall_accuracy.png
    │   └── category_accuracy_heatmap.png
    └── reports/
        ├── evaluation_report.txt
        └── evaluation_summary.json
```

## Troubleshooting

### Virtual Environment Issues
```bash
# If virtual environment activation fails
which python3
python3 --version

# Recreate virtual environment if needed
rm -rf medcalc-env
python3 -m venv medcalc-env
```

### Import Errors
```bash
# Verify PromptEngineer installation
pip list | grep promptengineer

# Reinstall if needed
pip uninstall promptengineer
pip install -e ../promptengineer
```

### MedCalc-Bench Import Errors
```bash
# Check MedCalc-Bench structure
ls MedCalc-Bench/evaluation/
python -c "import sys; sys.path.append('MedCalc-Bench/evaluation'); import evaluate"
```

### API Key Issues
```bash
# Check if API key is set
echo $OPENAI_API_KEY

# Test API key
python -c "
import openai
import os
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
print('✅ API key configured correctly')
"
```