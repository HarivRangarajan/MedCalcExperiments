## Iterative Prompt Refinement and Contrastive Few-Shot Evaluation Pipeline

This pipeline implements a complete system for:
1. **Iterative Prompt Refinement** using feedback from correct/incorrect responses
2. **Contrastive Few-Shot Evaluation** with positive and negative demonstrations
3. **Publication-Ready Visualizations** comparing performance

### 📋 Overview

The pipeline consists of three main phases:

```
Phase 1: Iterative Refinement
├─ Load initial evaluation results
├─ For each prompt type (original, CoT, CoD):
│  ├─ Batch correct/incorrect examples
│  ├─ Use GPT-4o to refine prompt iteratively
│  └─ Save refinement history
└─ Combine all refined prompts into unified prompt

Phase 2: Contrastive Evaluation
├─ Load unified refined prompt
├─ For each test example:
│  ├─ Get 1 one-shot + 2 positive + 2 negative examples
│  ├─ Generate response with both prompts
│  └─ Evaluate using MedCalc's check_correctness
└─ Compare original vs refined performance

Phase 3: Visualization
├─ Overall accuracy comparison
├─ Per-category performance
├─ Improvement distribution
├─ Error analysis
├─ Statistical significance
└─ Summary tables
```

---

## 🚀 Quick Start

### Prerequisites

```bash
cd /path/to/PromptResearch/medcalc-evaluation
source ../mohs-llm-as-a-judge/llm-judge-env/bin/activate
export OPENAI_API_KEY="sk-..."
```

### Run Complete Pipeline

```bash
# Run everything in one command
python run_complete_pipeline.py \
  --training-results-dir outputs/medcalc_contrastive_edits_evaluation_20251010_054434 \
  --batch-size 17
```

**Time Estimate**: 4-6 hours for full pipeline (refinement + 1047 test evaluations)

**Cost Estimate**: ~$50-80 USD (depends on prompt lengths and iterations)

---

## 📝 Individual Scripts

### 1. Prompt Refinement Pipeline

**Purpose**: Iteratively refine prompts using feedback from performance

**Usage**:
```bash
python prompt_refinement_pipeline.py \
  --results-dir outputs/medcalc_contrastive_edits_evaluation_20251010_054434 \
  --batch-size 17 \
  --max-iterations 10  # Optional limit
```

**Parameters**:
- `--results-dir`: Directory with initial evaluation results (required)
- `--batch-size`: Number of examples per refinement batch (default: 17)
- `--max-iterations`: Max refinement iterations (default: None = use all)
- `--output-dir`: Custom output directory (default: auto-generated)

**Output Structure**:
```
refined_prompts_TIMESTAMP/
├── iterations/
│   ├── chain_of_thought_iteration_1.json
│   ├── chain_of_thought_iteration_2.json
│   └── ...
├── final/
│   ├── final_refined_prompts.json
│   └── unified_prompt.txt  ← Main output
├── logs/
└── refinement_summary.json
```

**What It Does**:
1. Loads correct and incorrect responses for each prompt type
2. Creates batches of examples with mixed correct/incorrect
3. For each batch:
   - Sends current prompt + batch to GPT-4o
   - GPT-4o analyzes failures and refines the prompt
   - Saves iteration result
4. After all iterations, combines insights from all three prompts into one unified prompt

**Time**: ~30-60 minutes (depends on number of iterations)
**Cost**: ~$5-10 USD

---

### 2. Contrastive Few-Shot Evaluation

**Purpose**: Evaluate refined prompt vs original on full test set using contrastive examples

**Usage**:
```bash
python contrastive_few_shot_evaluation.py \
  --refined-prompts-dir outputs/refined_prompts_TIMESTAMP \
  --training-results-dir outputs/medcalc_contrastive_edits_evaluation_20251010_054434
```

**Parameters**:
- `--refined-prompts-dir`: Directory with refined prompts (required)
- `--training-results-dir`: Directory with training results for contrastive examples (required)
- `--output-dir`: Custom output directory (default: auto-generated)

**Output Structure**:
```
contrastive_evaluation_TIMESTAMP/
├── responses/
│   ├── original_one_shot_responses.jsonl
│   └── contrastive_few_shot_responses.jsonl
├── evaluations/
│   └── evaluation_summary.json
├── logs/
└── visualizations/  ← Created by visualization script
```

**What It Does**:
1. For each of 1047 test examples:
   - **Original**: Uses MedCalc's one-shot prompt
   - **Contrastive**: Uses refined prompt + 1 one-shot + 2 positive + 2 negative examples
2. Generates responses using GPT-4o
3. Evaluates using MedCalc's `check_correctness()`
4. Compares performance metrics

**Contrastive Example Selection**:
- **One-shot**: From MedCalc's curated examples
- **Positive (2)**: Correct responses from training with same calculator_id
- **Negative (2)**: Incorrect responses from training with same calculator_id
- Falls back gracefully if exact calculator_id not available

**Time**: ~3-5 hours (1047 examples × 2 prompts × ~5 sec/call)
**Cost**: ~$40-60 USD

---

### 3. Results Visualization

**Purpose**: Create publication-ready figures and analysis

**Usage**:
```bash
python visualize_results.py \
  --evaluation-dir outputs/contrastive_evaluation_TIMESTAMP
```

**Parameters**:
- `--evaluation-dir`: Directory with evaluation results (required)

**Generated Visualizations**:

1. **overall_accuracy_comparison.{png,pdf}**
   - Bar chart comparing original vs contrastive
   - Shows improvement percentage

2. **category_comparison.{png,pdf}**
   - Per-category accuracy comparison
   - Side-by-side bars for all medical categories

3. **improvement_distribution.{png,pdf}**
   - Histogram of improvements across calculators
   - Top 10 largest improvements

4. **error_analysis.{png,pdf}**
   - Pie chart of correctness changes
   - Net improvement visualization

5. **statistical_significance.{png,pdf}**
   - McNemar's test contingency table
   - P-value and significance level

6. **summary_table.{csv,txt}**
   - Comprehensive metrics table
   - Formatted for LaTeX/Word

**Time**: ~1-2 minutes
**Cost**: Free (no API calls)

---

## 📊 Expected Results

Based on the initial training results:

| Metric | Original One-Shot | Contrastive Few-Shot | Improvement |
|--------|------------------|---------------------|-------------|
| Accuracy | 71.2% | ~74-77%* | +3-6%* |
| Test Set Size | 1047 | 1047 | - |

*Estimated based on training performance (74.1% for CoT)

---

## 🔬 Technical Details

### Iterative Refinement Process

**Refinement Instruction Template**:
```
You are an expert prompt engineer...

Current Prompt: [current prompt]

Performance Feedback:
- CORRECT responses (what worked): [examples]
- INCORRECT responses (what failed): [examples]

Your Task:
1. Analyze failure patterns
2. Identify successful strategies
3. Refine the prompt to fix issues while maintaining strengths
4. Keep few-shot compatibility
5. Maintain JSON output format
```

**Combination Instruction**:
```
Given three refined prompts (original, CoT, CoD),
create ONE unified prompt that combines their strengths.
```

### Contrastive Few-Shot Format

**System Message Structure**:
```
[Unified Refined Prompt]

**Example (Correct Approach):**
[One-shot example from MedCalc]

**Additional Correct Examples:**
Example 1 (CORRECT):
- Patient Note: ...
- LLM Answer: ...
- Ground Truth: ...

**Examples to AVOID (Common Mistakes):**
Example 1 (INCORRECT - Learn from this):
- Patient Note: ...
- Incorrect Answer: ...
- Correct Answer Should Be: ...
```

### Evaluation Methodology

1. **Extract Answer**: Same regex-based extraction as MedCalc
2. **Check Correctness**: Uses MedCalc's `check_correctness()` function
   - Date type: Exact match
   - Integer type: Rounded match
   - Decimal type: Range-based (upper/lower limits)
3. **Statistical Test**: McNemar's test for paired nominal data

---

## 💾 File Formats

### Refined Prompt JSON
```json
{
  "iteration": 3,
  "prompt": "You are an expert medical calculator...",
  "num_correct": 10,
  "num_incorrect": 7,
  "timestamp": "2025-10-10T..."
}
```

### Response JSONL
```json
{
  "Row Number": 1,
  "Calculator Name": "Creatinine Clearance",
  "Calculator ID": "2",
  "LLM Answer": "141.042",
  "Ground Truth Answer": "141.042",
  "Result": "Correct",
  "Prompt Type": "contrastive_few_shot"
}
```

### Evaluation Summary JSON
```json
{
  "original_one_shot": {
    "overall_accuracy": 0.712,
    "total": 1047,
    "correct": 745,
    "by_category": {...},
    "by_calculator": {...}
  },
  "contrastive_few_shot": {...},
  "improvement": 0.034
}
```

---

## 🎯 Use Cases

### Research Paper
```bash
# Run complete pipeline
python run_complete_pipeline.py \
  --training-results-dir outputs/medcalc_contrastive_edits_evaluation_TIMESTAMP \
  --batch-size 17

# Use generated figures in paper:
# - Figure 1: overall_accuracy_comparison.pdf
# - Figure 2: category_comparison.pdf
# - Figure 3: improvement_distribution.pdf
# - Table 1: summary_table.csv
```

### Experimenting with Batch Sizes
```bash
# Try different batch sizes
for batch_size in 10 15 20; do
  python prompt_refinement_pipeline.py \
    --results-dir outputs/medcalc_contrastive_edits_evaluation_TIMESTAMP \
    --batch-size $batch_size \
    --output-dir outputs/refined_batch${batch_size}
done
```

### Quick Iteration Limit
```bash
# Limit to 5 iterations for quick testing
python prompt_refinement_pipeline.py \
  --results-dir outputs/medcalc_contrastive_edits_evaluation_TIMESTAMP \
  --batch-size 17 \
  --max-iterations 5
```

---

## ⚠️ Important Notes

### Contrastive Example Matching
- Examples MUST have exact `calculator_id` match
- If no contrastive examples available, falls back to one-shot only
- Randomized selection from available examples

### Few-Shot Compatibility
- All prompts maintain ability to inject demonstrations at runtime
- JSON output format preserved throughout refinement
- System message structure designed for runtime modification

### Cost Management
- **Refinement**: ~$0.50-1.00 per iteration per prompt
- **Evaluation**: ~$0.04 per test example (2 prompts)
- **Total for 1047 examples**: ~$50-80

### Time Management
- Run refinement overnight or during low-priority time
- Evaluation can take 3-5 hours (cannot be parallelized easily)
- Visualization is fast (<2 min)

---

## 🐛 Troubleshooting

### "No contrastive examples found"
- Check that `--training-results-dir` points to correct location
- Verify that correct/incorrect JSONL files exist
- Some calculators may not have examples in training set

### "Refinement taking too long"
- Use `--max-iterations` to limit iterations
- Increase `--batch-size` to process more examples per iteration

### "API rate limit exceeded"
- Add delays between requests (modify scripts if needed)
- Use OpenAI's tier-appropriate API key

### "Out of memory"
- Process large files in chunks
- Clear intermediate results if disk space low

---

## 📚 Related Files

- `medcalc_with_contrastive_boosted_edits.py`: Initial training evaluation
- `CONTRASTIVE_EDITS_README.md`: Training evaluation documentation
- `QUICK_START.md`: Quick start guide for training
- `IMPLEMENTATION_SUMMARY.md`: Technical implementation details

---

## 🔗 Citation

If you use this pipeline in your research, please cite:

```bibtex
@misc{medcalc_refinement_2025,
  title={Iterative Prompt Refinement with Contrastive Few-Shot Learning for Medical Calculations},
  author={Your Name},
  year={2025},
  note={MedCalc-Bench Evaluation Pipeline}
}
```

---

## ✨ Features

- ✅ Configurable batch sizes
- ✅ Configurable iteration limits
- ✅ Exact calculator_id matching for contrastive examples
- ✅ Maintains few-shot compatibility
- ✅ Publication-ready visualizations
- ✅ Statistical significance testing
- ✅ Comprehensive error analysis
- ✅ Full pipeline orchestration
- ✅ Modular design (run phases independently)

---

## 📧 Support

For questions or issues:
1. Check troubleshooting section
2. Review generated logs in output directories
3. Verify API key and environment setup

**Happy refining! 🎉**

