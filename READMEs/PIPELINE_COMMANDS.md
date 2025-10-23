# Pipeline Commands - Quick Reference

## Setup

```bash
cd /Users/harivallabharangarajan/Desktop/CMU/PromptResearch/medcalc-evaluation
source ../mohs-llm-as-a-judge/llm-judge-env/bin/activate
export OPENAI_API_KEY="sk-..."
```

## Complete Pipeline (One Command)

```bash
# Run everything: refinement → evaluation → visualization
python run_complete_pipeline.py \
  --training-results-dir outputs/medcalc_contrastive_edits_evaluation_20251010_054434 \
  --batch-size 17
```

## Individual Steps

### Step 1: Iterative Prompt Refinement

```bash
python prompt_refinement_pipeline.py \
  --results-dir outputs/medcalc_contrastive_edits_evaluation_20251010_054434 \
  --batch-size 17
```

**Output**: `outputs/refined_prompts_TIMESTAMP/`

### Step 2: Contrastive Few-Shot Evaluation

```bash
python contrastive_few_shot_evaluation.py \
  --refined-prompts-dir outputs/refined_prompts_TIMESTAMP \
  --training-results-dir outputs/medcalc_contrastive_edits_evaluation_20251010_054434
```

**Output**: `outputs/contrastive_evaluation_TIMESTAMP/`

### Step 3: Visualization

```bash
python visualize_results.py \
  --evaluation-dir outputs/contrastive_evaluation_TIMESTAMP
```

**Output**: `outputs/contrastive_evaluation_TIMESTAMP/visualizations/`

## With Your Current Results

Based on your successful run at `outputs/medcalc_contrastive_edits_evaluation_20251010_054434`:

```bash
# Full pipeline
python run_complete_pipeline.py \
  --training-results-dir /Users/harivallabharangarajan/Desktop/CMU/PromptResearch/outputs/medcalc_contrastive_edits_evaluation_20251010_054434 \
  --batch-size 17

# Or step-by-step:

# 1. Refinement
python prompt_refinement_pipeline.py \
  --results-dir /Users/harivallabharangarajan/Desktop/CMU/PromptResearch/outputs/medcalc_contrastive_edits_evaluation_20251010_054434 \
  --batch-size 17

# 2. Evaluation (use the output dir from step 1)
python contrastive_few_shot_evaluation.py \
  --refined-prompts-dir outputs/refined_prompts_20251010_HHMMSS \
  --training-results-dir /Users/harivallabharangarajan/Desktop/CMU/PromptResearch/outputs/medcalc_contrastive_edits_evaluation_20251010_054434

# 3. Visualization (use the output dir from step 2)
python visualize_results.py \
  --evaluation-dir outputs/contrastive_evaluation_20251010_HHMMSS
```

## Quick Test (Limited Iterations)

```bash
# Test with only 3 iterations (faster, cheaper)
python prompt_refinement_pipeline.py \
  --results-dir outputs/medcalc_contrastive_edits_evaluation_20251010_054434 \
  --batch-size 17 \
  --max-iterations 3
```

## Skip Completed Steps

If you've already run some steps:

```bash
# Skip refinement, go straight to evaluation
python run_complete_pipeline.py \
  --training-results-dir outputs/medcalc_contrastive_edits_evaluation_20251010_054434 \
  --skip-refinement \
  --refined-prompts-dir outputs/refined_prompts_TIMESTAMP

# Skip both refinement and evaluation, just visualize
python run_complete_pipeline.py \
  --training-results-dir outputs/medcalc_contrastive_edits_evaluation_20251010_054434 \
  --skip-refinement \
  --skip-evaluation \
  --refined-prompts-dir outputs/refined_prompts_TIMESTAMP \
  --evaluation-dir outputs/contrastive_evaluation_TIMESTAMP
```

## Check Results

```bash
# View evaluation summary
cat outputs/contrastive_evaluation_TIMESTAMP/evaluations/evaluation_summary.json | jq .

# View unified prompt
cat outputs/refined_prompts_TIMESTAMP/final/unified_prompt.txt

# List visualizations
ls -lh outputs/contrastive_evaluation_TIMESTAMP/visualizations/

# View summary table
cat outputs/contrastive_evaluation_TIMESTAMP/visualizations/summary_table.txt
```

## Time & Cost Estimates

| Step | Time | Cost (GPT-4o) |
|------|------|---------------|
| Refinement (all iterations) | 30-60 min | $5-10 |
| Evaluation (1047 test examples) | 3-5 hours | $40-60 |
| Visualization | 1-2 min | $0 |
| **Total** | **4-6 hours** | **$50-80** |

## Expected Improvements

Based on training results (170 examples):
- Original one-shot: **71.2%** accuracy
- Chain of Thought: **74.1%** accuracy (+2.9%)
- Contrastive few-shot (expected): **~74-77%** accuracy

## What to Include in Your Paper

1. **Figures** (from visualizations/):
   - overall_accuracy_comparison.pdf
   - category_comparison.pdf
   - improvement_distribution.pdf
   - statistical_significance.pdf

2. **Tables**:
   - summary_table.csv (convert to LaTeX)

3. **Statistics**:
   - From evaluation_summary.json
   - McNemar's test p-value
   - Confidence intervals

## Troubleshooting

```bash
# If imports fail
which python  # Should show venv path
pip list | grep openai  # Should show openai package

# If API calls fail
echo $OPENAI_API_KEY  # Should show your key
python -c "from openai import OpenAI; print('OK')"  # Should print OK

# If out of space
du -sh outputs/*  # Check disk usage
rm -rf outputs/old_experiment_*  # Clean up old runs
```

## Pro Tips

1. **Run refinement first** (30-60 min, cheap)
   - Check the unified prompt quality before committing to full evaluation
   
2. **Test on subset first**
   - Modify contrastive_few_shot_evaluation.py to test on 50 examples
   - Verify it's working before running all 1047

3. **Run evaluation overnight**
   - Full evaluation takes 3-5 hours
   - Perfect for overnight or weekend runs

4. **Save intermediate results**
   - Don't delete refined prompts directory
   - You might want to re-run evaluation with different parameters

5. **Track costs**
   - Monitor your OpenAI usage dashboard
   - Set billing alerts if concerned about costs

