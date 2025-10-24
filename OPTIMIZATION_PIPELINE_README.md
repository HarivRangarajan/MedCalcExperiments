# Complete Prompt Optimization and Evaluation Pipeline

A comprehensive pipeline for optimizing medical calculation prompts through iterative refinement, validation-based model selection, and rigorous evaluation.

## Overview

This pipeline implements a systematic approach to prompt optimization:

1. **Iterative Refinement**: Creates refined prompts through feedback-based iteration
2. **Candidate Generation**: Generates multiple candidate prompts with different strategic angles
3. **Validation Split**: Creates a held-out validation set for hyperparameter tuning
4. **Grid Search**: Tests all (prompt, model) combinations to find optimal configuration
5. **Full Evaluation**: Evaluates optimal configuration on complete test set
6. **Visualization**: Creates publication-ready visualizations and comparisons

## Pipeline Architecture

```
Training Results (contrastive edits)
         ↓
[1] Iterative Prompt Refinement (10 iterations × 3 types)
         ↓
    5 Candidate Prompts (different angles)
         ↓
[2] Validation Split (100 examples)
         ↓
[3] Grid Search (5 candidates × 3 models = 15 configurations)
         ↓
    Optimal (Prompt, Model) Pair
         ↓
[4] Full Test Evaluation (1047 examples)
         ↓
[5] Comprehensive Visualization & Analysis
```

## Models Evaluated

The pipeline tests three OpenAI models, balancing capability and cost:

1. **gpt-4o** - Highest capability, highest cost
2. **gpt-4o-mini** - Good balance of capability and cost
3. **gpt-3.5-turbo** - Most cost-effective

## Installation

### Prerequisites

```bash
# Python 3.8+
python --version

# Virtual environment (recommended)
cd /Users/harivallabharangarajan/Desktop/CMU/PromptResearch/medcalc-evaluation
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `openai` - OpenAI API client
- `pandas` - Data manipulation
- `matplotlib` - Visualization
- `seaborn` - Statistical visualization
- `tqdm` - Progress bars
- `numpy` - Numerical computing

## Quick Start

### Complete Pipeline (One Command)

```bash
# Set API key
export OPENAI_API_KEY="your-api-key-here"

# Activate environment
cd /Users/harivallabharangarajan/Desktop/CMU/PromptResearch/medcalc-evaluation
source ../mohs-llm-as-a-judge/llm-judge-env/bin/activate

# Run complete pipeline
python run_complete_optimization_pipeline.py \
    --training-results-dir /Users/harivallabharangarajan/Desktop/CMU/PromptResearch/outputs/medcalc_contrastive_edits_evaluation_20251010_054434 \
    --num-refinement-iterations 10 \
    --num-candidates 5 \
    --validation-size 100 \
    --test-all-models
```

This single command runs all 5 stages and takes approximately 2-3 hours depending on API rate limits.

### Output Structure

```
optimization_results/run_TIMESTAMP/
├── 1_refined_prompts/
│   ├── iterations/              # Individual refinement iterations
│   ├── final/
│   │   ├── candidates/          # 5 candidate prompts
│   │   ├── candidate_prompts.json
│   │   └── unified_prompt.txt
│   └── refinement_summary.json
├── 2_validation_split/
│   ├── validation_examples.jsonl
│   └── validation_metadata.json
├── 3_grid_search/
│   ├── results/                 # Results for each (prompt, model) pair
│   ├── optimal/
│   │   ├── optimal_config.json  # ⭐ Optimal configuration
│   │   └── optimal_prompt.txt   # ⭐ Optimal prompt
│   └── grid_search_summary.json
├── 4_test_evaluation/
│   ├── responses/               # Detailed responses
│   ├── evaluations/             # Per-model evaluations
│   └── test_evaluation_summary.json  # ⭐ Final results
├── 5_visualizations/
│   ├── grid_search_heatmap.png
│   ├── test_comparison.png
│   ├── category_comparison.png
│   ├── improvement_breakdown.png
│   ├── model_efficiency.png
│   └── summary_report.txt       # ⭐ Comprehensive report
└── pipeline_summary.json
```

## Stage-by-Stage Usage

If you prefer to run stages individually or resume from a specific stage:

### Stage 1: Iterative Prompt Refinement

Generate refined prompts through iterative feedback:

```bash
python prompt_refinement_pipeline.py \
    --results-dir /path/to/training/results \
    --batch-size 17 \
    --max-iterations 10 \
    --num-candidates 5 \
    --output-dir ./1_refined_prompts
```

**Output**: 5 candidate prompts in `1_refined_prompts/final/candidates/`

### Stage 2: Create Validation Split

Create a validation set from training examples:

```bash
python create_validation_split.py \
    --training-results-dir /path/to/training/results \
    --validation-size 100 \
    --output-dir ./2_validation_split \
    --seed 42
```

**Output**: `2_validation_split/validation_examples.jsonl`

### Stage 3: Grid Search on Validation

Test all (prompt, model) combinations:

```bash
python grid_search_validation.py \
    --refined-prompts-dir ./1_refined_prompts \
    --validation-split-dir ./2_validation_split \
    --training-results-dir /path/to/training/results \
    --output-dir ./3_grid_search \
    --batch-size 5
```

**Output**: Optimal configuration in `3_grid_search/optimal/`

### Stage 4: Full Test Evaluation

Evaluate optimal configuration on full test set:

```bash
python full_test_evaluation.py \
    --grid-search-dir ./3_grid_search \
    --training-results-dir /path/to/training/results \
    --output-dir ./4_test_evaluation \
    --test-all-models \
    --batch-size 10 \
    --save-frequency 50
```

**Flags**:
- `--test-all-models`: Test all 3 models (not just optimal) for comparison
- Without flag: Only test optimal model (faster, cheaper)

**Output**: Test results in `4_test_evaluation/test_evaluation_summary.json`

### Stage 5: Comprehensive Visualization

Create visualizations and reports:

```bash
python visualize_complete_results.py \
    --grid-search-dir ./3_grid_search \
    --test-results-dir ./4_test_evaluation \
    --output-dir ./5_visualizations
```

**Output**: Publication-ready figures in `5_visualizations/`

## Configuration Options

### Pipeline Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num-refinement-iterations` | 10 | Iterations per prompt type |
| `--num-candidates` | 5 | Number of candidate prompts |
| `--validation-size` | 100 | Validation examples |
| `--test-all-models` | False | Test all models on test set |

### Candidate Prompt Angles

The 5 candidate prompts use different strategic angles:

1. **Balanced Synthesis** - Equal balance of all refinements
2. **Precision-Focused** - Mathematical precision and accuracy
3. **Context-Aware** - Patient context and clinical reasoning
4. **Step-by-Step Methodical** - Systematic problem-solving
5. **Error-Prevention** - Avoiding common mistakes

## Resuming from Intermediate Stages

If a stage completes but later stages fail, you can resume:

```bash
# Resume from Stage 3 (skip refinement and validation)
python run_complete_optimization_pipeline.py \
    --training-results-dir /path/to/training/results \
    --skip-refinement \
    --refined-prompts-dir ./1_refined_prompts \
    --skip-validation-split \
    --validation-split-dir ./2_validation_split

# Resume from Stage 4 (skip refinement, validation, grid search)
python run_complete_optimization_pipeline.py \
    --training-results-dir /path/to/training/results \
    --skip-refinement \
    --refined-prompts-dir ./1_refined_prompts \
    --skip-validation-split \
    --validation-split-dir ./2_validation_split \
    --skip-grid-search \
    --grid-search-dir ./3_grid_search
```

## Cost Estimation

Approximate API costs for full pipeline (1047 test examples):

| Stage | Calls | Tokens | Estimated Cost |
|-------|-------|--------|----------------|
| Refinement | ~35 | ~140K | $1-2 |
| Grid Search | 1,500 | ~3M | $5-10 |
| Test Eval (optimal) | 1,047 | ~2M | $3-5 |
| Test Eval (all models) | 3,141 | ~6M | $10-15 |
| **Total (optimal only)** | | | **$9-17** |
| **Total (all models)** | | | **$16-27** |

_Note: Costs are approximate and depend on prompt length and model used_

## Key Results Files

After completion, check these files for results:

1. **Optimal Configuration**
   ```bash
   cat optimization_results/run_TIMESTAMP/3_grid_search/optimal/optimal_config.json
   ```

2. **Optimal Prompt**
   ```bash
   cat optimization_results/run_TIMESTAMP/3_grid_search/optimal/optimal_prompt.txt
   ```

3. **Test Results Summary**
   ```bash
   cat optimization_results/run_TIMESTAMP/4_test_evaluation/test_evaluation_summary.json
   ```

4. **Comprehensive Report**
   ```bash
   cat optimization_results/run_TIMESTAMP/5_visualizations/summary_report.txt
   ```

## Troubleshooting

### API Rate Limits

If you hit rate limits:

1. Reduce `--batch-size` to process fewer examples concurrently
2. Add delays between batches (modify scripts)
3. Use a higher-tier API account

### Out of Memory

If you run out of memory:

1. Reduce batch sizes
2. Process stages separately instead of full pipeline
3. Use a machine with more RAM

### API Key Issues

```bash
# Verify API key is set
echo $OPENAI_API_KEY

# Set if needed
export OPENAI_API_KEY="your-api-key-here"

# Test API connection
python -c "from openai import OpenAI; print(OpenAI().models.list())"
```

### Missing Dependencies

```bash
# Reinstall requirements
pip install --upgrade -r requirements.txt

# Verify MedCalc-Bench is present
ls MedCalc-Bench/evaluation/
```

## Example: Quick Test Run

For testing the pipeline with minimal cost:

```bash
python run_complete_optimization_pipeline.py \
    --training-results-dir /path/to/training/results \
    --num-refinement-iterations 3 \
    --num-candidates 3 \
    --validation-size 50
```

This runs faster and costs ~$3-5, useful for testing setup.

## Interpreting Results

### Grid Search Heatmap

- **Darker green**: Higher validation accuracy
- **Blue box**: Optimal configuration
- Look for consistent patterns across models

### Test Comparison

- **Gray bar**: Baseline (MedCalc paper GPT-4)
- **Colored bars**: Our results
- Values above bars show improvement over baseline

### Category Comparison

- Shows per-category performance across models
- Identifies which categories benefit most from optimization
- Useful for understanding model strengths

### Model Efficiency

- Plots accuracy vs. relative cost
- Upper-left = best value (high accuracy, low cost)
- Helps choose model based on budget constraints

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{medcalc_optimization_pipeline,
  title={Complete Prompt Optimization Pipeline for Medical Calculations},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/repo}
}
```

## License

[Specify your license here]

## Contact

For questions or issues:
- Email: your.email@example.com
- GitHub: https://github.com/yourusername/repo

## Acknowledgments

- MedCalc-Bench dataset and evaluation framework
- OpenAI API for language models
- Research team contributions

