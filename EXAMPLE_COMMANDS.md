# Example Commands for Optimization Pipeline

This file contains copy-paste ready commands for running the optimization pipeline with your specific setup.

## Setup Environment

```bash
# Navigate to directory
cd /Users/harivallabharangarajan/Desktop/CMU/PromptResearch/medcalc-evaluation

# Activate virtual environment
source ../mohs-llm-as-a-judge/llm-judge-env/bin/activate

# Set API key
export OPENAI_API_KEY="<your api-key- here>"
```

## Option 1: Complete Pipeline (Recommended)

Run everything in one command:

```bash
python run_complete_optimization_pipeline.py \
    --training-results-dir /Users/harivallabharangarajan/Desktop/CMU/PromptResearch/outputs/medcalc_contrastive_edits_evaluation_20251010_054434 \
    --num-refinement-iterations 10 \
    --num-candidates 5 \
    --validation-size 100 \
    --test-all-models
```

**Time**: ~2-3 hours  
**Cost**: ~$15-25

## Option 2: Quick Test Run

Faster test with reduced parameters:

```bash
python run_complete_optimization_pipeline.py \
    --training-results-dir /Users/harivallabharangarajan/Desktop/CMU/PromptResearch/outputs/medcalc_contrastive_edits_evaluation_20251010_054434 \
    --num-refinement-iterations 3 \
    --num-candidates 3 \
    --validation-size 50
```

**Time**: ~30-45 minutes  
**Cost**: ~$3-5

## Option 3: Stage-by-Stage Execution

### Stage 1: Prompt Refinement (10 iterations)

```bash
python prompt_refinement_pipeline.py \
    --results-dir /Users/harivallabharangarajan/Desktop/CMU/PromptResearch/outputs/medcalc_contrastive_edits_evaluation_20251010_054434 \
    --batch-size 17 \
    --max-iterations 10 \
    --num-candidates 5 \
    --output-dir ./optimization_results/1_refined_prompts
```

### Stage 2: Create Validation Split

```bash
python create_validation_split.py \
    --training-results-dir /Users/harivallabharangarajan/Desktop/CMU/PromptResearch/outputs/medcalc_contrastive_edits_evaluation_20251010_054434 \
    --validation-size 100 \
    --output-dir ./optimization_results/2_validation_split \
    --seed 42
```

### Stage 3: Grid Search on Validation

```bash
python grid_search_validation.py \
    --refined-prompts-dir ./optimization_results/1_refined_prompts \
    --validation-split-dir ./optimization_results/2_validation_split \
    --training-results-dir /Users/harivallabharangarajan/Desktop/CMU/PromptResearch/outputs/medcalc_contrastive_edits_evaluation_20251010_054434 \
    --output-dir ./optimization_results/3_grid_search \
    --batch-size 5 \
    --num-positive 1 \
    --num-negative 1
```

**Note**: Grid search now uses contrastive few-shot (1 positive + 1 negative example) just like the final evaluation.

### Stage 4: Full Test Evaluation (All Models)

```bash
python full_test_evaluation.py \
    --grid-search-dir ./optimization_results/3_grid_search \
    --training-results-dir /Users/harivallabharangarajan/Desktop/CMU/PromptResearch/outputs/medcalc_contrastive_edits_evaluation_20251010_054434 \
    --output-dir ./optimization_results/4_test_evaluation \
    --test-all-models \
    --batch-size 10 \
    --save-frequency 50
```

**Or just test optimal model** (faster, cheaper):

```bash
python full_test_evaluation.py \
    --grid-search-dir ./optimization_results/3_grid_search \
    --training-results-dir /Users/harivallabharangarajan/Desktop/CMU/PromptResearch/outputs/medcalc_contrastive_edits_evaluation_20251010_054434 \
    --output-dir ./optimization_results/4_test_evaluation \
    --batch-size 10 \
    --save-frequency 50
```

### Stage 5: Visualization

```bash
python visualize_complete_results.py \
    --grid-search-dir ./optimization_results/3_grid_search \
    --test-results-dir ./optimization_results/4_test_evaluation \
    --output-dir ./optimization_results/5_visualizations
```

## Option 4: Resume from Checkpoint

If a run was interrupted, you can resume:

### Resume from Stage 3 (skip refinement and validation)

```bash
python run_complete_optimization_pipeline.py \
    --training-results-dir /Users/harivallabharangarajan/Desktop/CMU/PromptResearch/outputs/medcalc_contrastive_edits_evaluation_20251010_054434 \
    --skip-refinement \
    --refined-prompts-dir ./optimization_results/1_refined_prompts \
    --skip-validation-split \
    --validation-split-dir ./optimization_results/2_validation_split \
    --num-refinement-iterations 10 \
    --num-candidates 5 \
    --validation-size 100 \
    --test-all-models
```

### Resume from Stage 4 (skip everything except test eval)

```bash
python run_complete_optimization_pipeline.py \
    --training-results-dir /Users/harivallabharangarajan/Desktop/CMU/PromptResearch/outputs/medcalc_contrastive_edits_evaluation_20251010_054434 \
    --skip-refinement \
    --refined-prompts-dir ./optimization_results/1_refined_prompts \
    --skip-validation-split \
    --validation-split-dir ./optimization_results/2_validation_split \
    --skip-grid-search \
    --grid-search-dir ./optimization_results/3_grid_search \
    --test-all-models
```

## Using Quick Start Script

The easiest way to run the complete pipeline:

```bash
./quick_start.sh
```

Or specify a different training directory:

```bash
./quick_start.sh /path/to/your/training/results
```

## Checking Results

After completion, view key results:

```bash
# View optimal configuration
cat optimization_results/run_*/3_grid_search/optimal/optimal_config.json | python -m json.tool

# View optimal prompt
cat optimization_results/run_*/3_grid_search/optimal/optimal_prompt.txt

# View test results summary
cat optimization_results/run_*/4_test_evaluation/test_evaluation_summary.json | python -m json.tool

# View comprehensive report
cat optimization_results/run_*/5_visualizations/summary_report.txt

# Open visualizations
open optimization_results/run_*/5_visualizations/*.png
```

## Troubleshooting

### Check API Connection

```bash
python -c "from openai import OpenAI; client = OpenAI(); print('✓ API key valid')"
```

### Check MedCalc-Bench Setup

```bash
ls -la MedCalc-Bench/evaluation/evaluate.py
ls -la MedCalc-Bench/dataset/test_data.csv
```

### Check Training Results

```bash
ls -la /Users/harivallabharangarajan/Desktop/CMU/PromptResearch/outputs/medcalc_contrastive_edits_evaluation_20251010_054434/correct/
ls -la /Users/harivallabharangarajan/Desktop/CMU/PromptResearch/outputs/medcalc_contrastive_edits_evaluation_20251010_054434/incorrect/
```

### Test Individual Scripts

```bash
# Test refinement pipeline (minimal)
python prompt_refinement_pipeline.py \
    --results-dir /Users/harivallabharangarajan/Desktop/CMU/PromptResearch/outputs/medcalc_contrastive_edits_evaluation_20251010_054434 \
    --batch-size 17 \
    --max-iterations 1 \
    --num-candidates 2

# Test validation split
python create_validation_split.py \
    --training-results-dir /Users/harivallabharangarajan/Desktop/CMU/PromptResearch/outputs/medcalc_contrastive_edits_evaluation_20251010_054434 \
    --validation-size 10
```

## Cost Optimization

To reduce costs while testing:

```bash
# Minimal test configuration
python run_complete_optimization_pipeline.py \
    --training-results-dir /Users/harivallabharangarajan/Desktop/CMU/PromptResearch/outputs/medcalc_contrastive_edits_evaluation_20251010_054434 \
    --num-refinement-iterations 2 \
    --num-candidates 2 \
    --validation-size 20
    # Don't use --test-all-models (only test optimal model)
```

**Cost**: ~$1-2

## Next Steps After Completion

1. **Review results**:
   ```bash
   cat optimization_results/run_*/5_visualizations/summary_report.txt
   ```

2. **Use optimal prompt**:
   ```bash
   cp optimization_results/run_*/3_grid_search/optimal/optimal_prompt.txt ./production_prompt.txt
   ```

3. **Deploy with optimal model**:
   - Check `optimal_config.json` for the best model
   - Use the optimal prompt from `optimal_prompt.txt`
   - Expected improvement: +5-10% over baseline

4. **Analyze failures**:
   ```bash
   # Find incorrect responses for analysis
   jq 'select(.Result == "Incorrect")' optimization_results/run_*/4_test_evaluation/responses/*_responses.jsonl | less
   ```

## Advanced: Parallel Execution

For faster execution on a powerful machine:

```bash
# Run stages in parallel (after Stage 1 completes)
# Terminal 1: Grid search
python grid_search_validation.py --refined-prompts-dir ./1_refined_prompts ...

# Terminal 2: Can start Stage 4 immediately after Stage 3 completes
# (monitor Stage 3 completion)
```

## Notes

- All commands assume you're in the `medcalc-evaluation` directory
- API key must be set before running any commands
- Default batch sizes are optimized for standard API rate limits
- Increase `--save-frequency` if you want more frequent checkpoints
- Decrease `--batch-size` if you hit rate limits

