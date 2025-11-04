# Setup Instructions

## Overview: Three-Part Pipeline

This repository implements a comprehensive prompt engineering evaluation system with three distinct parts:

### Part 1: Generating Contrastive Demonstrations
Creates the foundation by generating positive and negative examples from training data.

### Part 2: Prompt Refinement
Iteratively refines prompts using feedback from contrastive demonstrations.

### Part 3: Evaluations
Evaluates both baseline methods and our contrastive few-shot approach.

---

## What Do We Need to Set Up?

1. **MedCalc-Bench dataset** - The benchmark dataset for medical calculations
2. **`promptengineer` library** - Our custom prompt engineering library
3. **OpenAI API key** - For GPT-4 API access
4. **Python dependencies** - Installed via virtual environment

**Note**: This guide assumes your virtual environment is at `../mohs-llm-as-a-judge/llm-judge-env/bin/activate`. Check `setup_and_run.sh` and modify paths as necessary.

---

## Quick Setup (Using Automated Script)

The easiest way to get started:

```bash
cd to the place where you have cloned this repo
export OPENAI_API_KEY="sk-..."
./setup_and_run.sh
```

The script will:
1. ✅ Check all prerequisites
2. ✅ Activate virtual environment
3. ✅ Extract train data if needed

## Manual Setup

If you prefer to set things up manually:

### Step 1: Navigate to Directory

```bash
cd /to the place where you have cloned this repo
```

### Step 2: Activate Virtual Environment

```bash
source ../mohs-llm-as-a-judge/llm-judge-env/bin/activate
```

**Important**: The virtual environment contains all required dependencies (pandas, numpy, openai, tqdm, etc.)

### Step 3: Verify Environment

```bash
# Check Python version
python --version  # Should be 3.7+

# Check if packages are available
python -c "import pandas, numpy, openai; print('✅ All packages available')"
```

### Step 4: Set API Key

```bash
export OPENAI_API_KEY="sk-..."

# Verify it's set
echo $OPENAI_API_KEY
```

### Step 5: Extract Train Data (if needed)

```bash
# Check if already extracted
if [ ! -f MedCalc-Bench/dataset/train_data.csv ]; then
    cd MedCalc-Bench/dataset
    unzip train_data.csv.zip
    cd ../..
fi
```

---

## Directory Structure

Before running, ensure this structure exists:

```
PromptResearch/
├── medcalc-evaluation/                           ← You are here
│   ├── MedCalc-Bench/
│   │   ├── dataset/
│   │   │   ├── train_data.csv                   ← Must exist (unzip if needed)
│   │   │   └── test_data.csv
│   │   └── evaluation/
│   │       ├── llm_inference.py
│   │       ├── evaluate.py
│   │       └── run.py
│   ├── runner_contrastive_demonstration_generation.py  ← Top-level runner
│   ├── runner_refinement_plus_evaluation_plus_visualization.py  ← Top-level runner
│   ├── pipeline/                                 ← Pipeline scripts
│   │   ├── contrastive_demonstration_generation.py
│   │   ├── prompt_refinement_pipeline.py
│   │   ├── evaluate_contrastive_fewshot_method.py
│   │   └── visualize_results.py
│   ├── baseline_evaluation/                      ← Baseline comparisons
│   │   └── evaluate_baselines.py
│   ├── tests/                                    ← Test scripts
│   │   ├── test_contrastive_demonstration_generation.py
│   │   └── test_prompt_refinement_pipeline.py
│   ├── modules/                                  ← Helper modules
│   └── setup_and_run.sh
├── promptengineer/                               ← Must exist
└── mohs-llm-as-a-judge/
    └── llm-judge-env/                            ← Virtual environment
        └── bin/
            └── activate
```

---

## Part 1: Generating Contrastive Demonstrations

**Purpose**: Create positive and negative examples from training data to guide prompt refinement.

**What it does**:
1. Takes the base prompt from the MedCalc-Bench paper
2. Creates `promptengineer` enhanced versions (CoT, CoD, etc.)
3. Generates responses for a subset of train examples
4. Evaluates responses to identify correct (positive) and incorrect (negative) demonstrations

### Running Part 1

**Quick test (10 samples)**:
```bash
python runner_contrastive_demonstration_generation.py
```

**Full run (500 samples)**:
```bash
python pipeline/contrastive_demonstration_generation.py --sample-size 500
```

**Output**: Creates `outputs/medcalc_contrastive_edits_evaluation_TIMESTAMP/` with:
- `correct/` - Positive demonstrations
- `incorrect/` - Negative demonstrations
- `prompts/` - Enhanced prompt versions

---

## Part 2: Prompt Refinement

**Purpose**: Iteratively refine prompts using feedback from contrastive demonstrations.

**What it does**:
1. Loads contrastive demonstrations from Part 1
2. Uses LLM-as-a-judge to analyze incorrect responses
3. Generates feedback and refined prompt versions
4. Selects best examples for few-shot demonstrations

### Running Part 2

**Test refinement pipeline**:
```bash
python tests/test_prompt_refinement_pipeline.py
```

**Full refinement** (requires Part 1 outputs):
```bash
python pipeline/prompt_refinement_pipeline.py \
  --training-results-dir outputs/medcalc_contrastive_edits_evaluation_TIMESTAMP \
  --batch-size 17
```

**Parameters**:
- `--training-results-dir`: Directory from Part 1
- `--batch-size`: Number of examples per refinement batch (default: 17)
- `--max-iterations`: Maximum refinement iterations (default: 3)

**Output**: Creates `outputs/refined_prompts_TIMESTAMP/` with:
- `refined_prompts/` - Iteratively improved prompts
- `feedback/` - LLM judge feedback
- `selected_examples/` - Best contrastive demonstrations

---

## Part 3: Evaluations

**Purpose**: Evaluate both baseline methods and our contrastive few-shot approach on the full test set.

### Part 3A: Evaluating Baselines

Evaluates original MedCalc-Bench prompts and PromptEngineer techniques.

```bash
python baseline_evaluation/evaluate_baselines.py --sample-size 300 --output-dir results_baselines
```

**What it evaluates**:
- Original MedCalc prompts (Direct, Zero-shot CoT, One-shot CoT)
- PromptEngineer generated prompts (Chain of Thought, Chain of Draft)

### Part 3B: Evaluating Our Method

Evaluates the refined contrastive few-shot prompt on the full test set.

```bash
python pipeline/evaluate_contrastive_fewshot_method.py \
  --refined-prompts-dir outputs/test_refined_prompts \
  --training-results-dir outputs/medcalc_contrastive_edits_evaluation_20251010_054434 \
  --num-test-examples 600 \
  --num-positive 1 \
  --num-negative 1 \
  --batch-size 10 \
  --save-frequency 50
```

**Parameters**:
- `--refined-prompts-dir`: Directory containing refined prompts from Part 2
- `--training-results-dir`: Directory containing training results from Part 1
- `--num-test-examples`: Number of test examples to evaluate (default: 600)
- `--num-positive`: Number of positive demonstrations to include (default: 1)
- `--num-negative`: Number of negative demonstrations to include (default: 1)
- `--batch-size`: Batch size for API calls (default: 10)
- `--save-frequency`: Save progress every N examples (default: 50)

**What it evaluates**:
- Original one-shot prompt (baseline)
- Unified refined prompt with contrastive few-shot examples (our method)

**Output**: Creates `outputs/contrastive_evaluation_TIMESTAMP/` with:
- `evaluations/evaluation_summary.json` - Accuracy metrics
- `responses/` - All test responses
- `analysis/` - Detailed error analysis

### Part 3C: Generate Visualizations

```bash
python pipeline/visualize_results.py \
  --evaluation-dir outputs/contrastive_evaluation_TIMESTAMP
```

**Generates**:
- Overall accuracy comparison
- Category-wise performance
- Statistical significance tests
- Error distribution analysis

---

## Running the Complete Pipeline

For Parts 2 + 3 together (assumes Part 1 is complete):

```bash
python runner_refinement_plus_evaluation_plus_visualization.py \
  --training-results-dir outputs/medcalc_contrastive_edits_evaluation_TIMESTAMP \
  --batch-size 17
```

**This script**:
1. Runs prompt refinement (Part 2)
2. Evaluates refined prompts on test set (Part 3B)
3. Generates publication-ready visualizations (Part 3C)

**Options**:
- `--skip-refinement`: Skip refinement if already done
- `--skip-evaluation`: Skip evaluation if already done
- `--skip-visualization`: Skip visualization if already done

---

## Complete Workflow Summary

### The Three Parts for Our Method:

1. **Generate Contrastive Demonstrations** (Part 1)
   ```bash
   python pipeline/contrastive_demonstration_generation.py --sample-size 500
   ```

2. **Refine Prompts + Evaluate + Visualize** (Parts 2 & 3)
   ```bash
   python runner_refinement_plus_evaluation_plus_visualization.py \
     --training-results-dir outputs/medcalc_contrastive_edits_evaluation_TIMESTAMP \
     --batch-size 17
   ```

These two runner scripts accomplish all three phases of our method.

### Minimal Commands

```bash
# 1. Navigate
cd /path/to/PromptResearch/medcalc-evaluation

# 2. Activate venv
source ../mohs-llm-as-a-judge/llm-judge-env/bin/activate

# 3. Set API key
export OPENAI_API_KEY="sk-..."

# 4. Extract data (if needed)
[ ! -f MedCalc-Bench/dataset/train_data.csv ] && \
  cd MedCalc-Bench/dataset && unzip train_data.csv.zip && cd ../..

# 5. Test Part 1
python tests/test_contrastive_demonstration_generation.py

# 6. Run Part 1 (full)
python pipeline/contrastive_demonstration_generation.py --sample-size 500

# 7. Run Parts 2 & 3 (full pipeline)
python runner_refinement_plus_evaluation_plus_visualization.py \
  --training-results-dir outputs/medcalc_contrastive_edits_evaluation_TIMESTAMP \
  --batch-size 17

# Or run Part 3B separately with custom parameters
python pipeline/evaluate_contrastive_fewshot_method.py \
  --refined-prompts-dir outputs/test_refined_prompts \
  --training-results-dir outputs/medcalc_contrastive_edits_evaluation_TIMESTAMP \
  --num-test-examples 600 \
  --num-positive 1 \
  --num-negative 1 \
  --batch-size 10 \
  --save-frequency 50
```

---

## Expected Outputs

After running the complete pipeline:

```
outputs/
├── medcalc_contrastive_edits_evaluation_TIMESTAMP/  # Part 1
│   ├── correct/
│   ├── incorrect/
│   └── prompts/
├── refined_prompts_TIMESTAMP/                        # Part 2
│   ├── refined_prompts/
│   ├── feedback/
│   └── selected_examples/
└── contrastive_evaluation_TIMESTAMP/                 # Part 3
    ├── evaluations/
    │   └── evaluation_summary.json
    ├── responses/
    ├── analysis/
    └── visualizations/                               # Part 3C
        ├── overall_comparison.png
        ├── category_comparison.png
        └── statistical_significance.png
```

---
